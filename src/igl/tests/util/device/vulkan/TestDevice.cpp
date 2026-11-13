/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/tests/util/device/vulkan/TestDevice.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <igl/Common.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_IOS || \
    IGL_PLATFORM_LINUX
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanFeatures.h>
#endif

#if IGL_PLATFORM_MACOSX
#include <igl/vulkan/moltenvk/MoltenVkHelpers.h>
#endif

namespace igl::tests::util::device::vulkan {

//
// createTestDevice
//
// Used by clients to get an IGL device.
//

igl::vulkan::VulkanContextConfig getContextConfig(bool enableValidation) {
  igl::vulkan::VulkanContextConfig config;
  config.enableValidation = enableValidation;
  config.enableGPUAssistedValidation = enableValidation;

#if IGL_PLATFORM_MACOSX
  config.terminateOnValidationError = false;
#elif IGL_DEBUG
  config.terminateOnValidationError = enableValidation;
#else
  config.enableValidation = false;
  config.terminateOnValidationError = false;
#endif
#ifdef IGL_DISABLE_VALIDATION
  config.enableValidation = false;
  config.terminateOnValidationError = false;
#endif
  config.swapChainColorSpace = igl::ColorSpace::SRGBNonlinear;
  config.enableExtraLogs = enableValidation;

  return config;
}

std::unique_ptr<igl::vulkan::Device> createTestDevice(
    const igl::vulkan::VulkanContextConfig& config) {
  return createTestDevice(config, {});
}

std::unique_ptr<igl::vulkan::Device> createTestDevice(
    const igl::vulkan::VulkanContextConfig& config,
    const std::function<void(igl::vulkan::VulkanFeatures&)>& configureFeatures) {
#if IGL_PLATFORM_MACOSX && !defined(IGL_USE_STATIC_KOSMICKRISP)
  // KosmicKrisp is statically linked; there is no Vulkan loader to point at a MoltenVK ICD.
  ::igl::vulkan::setupMoltenVKEnvironment();
#endif

  std::unique_ptr<igl::vulkan::Device> iglDev = nullptr;
  Result ret;

  auto ctx = igl::vulkan::HWDevice::createContext(config, nullptr);
  if (!ctx) {
    return nullptr;
  }

  std::vector<HWDeviceDesc> devices =
      igl::vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::Unknown), &ret);

  if (ret.isOk()) {
    std::vector<const char*> extraDeviceExtensions;
    extraDeviceExtensions.emplace_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);

    igl::vulkan::VulkanFeatures features(config);
    features.populateWithAvailablePhysicalDeviceFeatures(
        *ctx,
        reinterpret_cast<VkPhysicalDevice>(devices[0].guid)); // NOLINT(performance-no-int-to-ptr)
    if (configureFeatures) {
      configureFeatures(features);
    }

    iglDev = igl::vulkan::HWDevice::create(std::move(ctx),
                                           devices[0],
                                           0, // width
                                           0, // height,
                                           extraDeviceExtensions.size(),
                                           extraDeviceExtensions.data(),
                                           &features,
                                           "Test Device",
                                           &ret);

    if (!ret.isOk()) {
      iglDev = nullptr;
    }
  }

  return iglDev;
}

std::unique_ptr<igl::vulkan::Device> createTestDevice(bool enableValidation) {
  return createTestDevice(getContextConfig(enableValidation));
}

//
// FaultingDevice
//

namespace {

constexpr size_t kCallCount = static_cast<size_t>(FaultingDevice::Call::Count);
constexpr size_t kCountedCount = static_cast<size_t>(FaultingDevice::Counted::Count);

// One slot per hooked entry point. The KHR aliases of the `...2` binds have their own slot (their
// real pointers differ) and share the fault entry of their core name.
enum Entry : size_t {
  kCreateImage,
  kCreateBuffer,
  kCreateImageView,
  kAllocateMemory,
  kMapMemory,
  kBindBufferMemory,
  kBindBufferMemory2,
  kBindBufferMemory2KHR,
  kBindImageMemory,
  kBindImageMemory2,
  kBindImageMemory2KHR,
  kDestroyImage,
  kDestroyBuffer,
  kFreeMemory,
  kSetDebugUtilsObjectName,
  kQueueWaitIdle,
  kEntryCount,
};

struct FaultState {
  PFN_vkGetDeviceProcAddr realGetDeviceProcAddr = nullptr;
  std::array<PFN_vkVoidFunction, kEntryCount> real{};
  std::array<std::atomic<uint32_t>, kCallCount> reached{};
  std::array<std::atomic<uint32_t>, kCallCount> failed{};
  std::array<std::atomic<uint32_t>, kCallCount> skip{};
  std::array<std::atomic<uint32_t>, kCallCount> remaining{};
  std::array<std::atomic<int32_t>, kCallCount> result{};
  // An armed sequence: the n-th failed call returns sequence[n] (the last entry repeats).
  std::array<std::array<std::atomic<int32_t>, 4>, kCallCount> sequence{};
  std::array<std::atomic<uint32_t>, kCallCount> sequenceLength{};
  std::array<std::atomic<uint32_t>, kCallCount> ordinal{};
  std::array<std::atomic<uint32_t>, kCountedCount> counted{};
  // The counters as they stood at the first `vkQueueWaitIdle()`, which the context calls when it
  // is torn down first thing: what the failing creation destroyed before any teardown.
  std::array<std::atomic<uint32_t>, kCountedCount> countedAtTeardown{};
  std::atomic<bool> teardownSeen{false};
  bool live = false;
};

FaultState gFault;

// True when this call is to be failed; `outResult` then holds the VkResult it returns.
bool shouldFail(FaultingDevice::Call call, VkResult* outResult) {
  const auto index = static_cast<size_t>(call);
  gFault.reached[index].fetch_add(1);
  uint32_t skip = gFault.skip[index].load();
  while (skip > 0) {
    if (gFault.skip[index].compare_exchange_weak(skip, skip - 1)) {
      return false;
    }
  }
  uint32_t remaining = gFault.remaining[index].load();
  while (remaining > 0) {
    if (gFault.remaining[index].compare_exchange_weak(remaining, remaining - 1)) {
      gFault.failed[index].fetch_add(1);
      const uint32_t length = gFault.sequenceLength[index].load();
      const uint32_t ordinal = gFault.ordinal[index].fetch_add(1);
      *outResult = static_cast<VkResult>(
          length > 0 ? gFault.sequence[index][std::min(ordinal, length - 1)].load()
                     : gFault.result[index].load());
      return true;
    }
  }
  return false;
}

#define IGL_FAULT_STUB(NAME, ENTRY, CALL, PARAMS, ARGS)             \
  VKAPI_ATTR VkResult VKAPI_CALL stub##NAME PARAMS {                \
    VkResult injected = VK_SUCCESS;                                 \
    if (shouldFail(FaultingDevice::Call::CALL, &injected)) {        \
      return injected;                                              \
    }                                                               \
    return reinterpret_cast<PFN_vk##NAME>(gFault.real[ENTRY]) ARGS; \
  }

IGL_FAULT_STUB(CreateImage,
               kCreateImage,
               CreateImage,
               (VkDevice d, const VkImageCreateInfo* i, const VkAllocationCallbacks* a, VkImage* o),
               (d, i, a, o))
IGL_FAULT_STUB(
    CreateBuffer,
    kCreateBuffer,
    CreateBuffer,
    (VkDevice d, const VkBufferCreateInfo* i, const VkAllocationCallbacks* a, VkBuffer* o),
    (d, i, a, o))
IGL_FAULT_STUB(
    CreateImageView,
    kCreateImageView,
    CreateImageView,
    (VkDevice d, const VkImageViewCreateInfo* i, const VkAllocationCallbacks* a, VkImageView* o),
    (d, i, a, o))
IGL_FAULT_STUB(
    AllocateMemory,
    kAllocateMemory,
    AllocateMemory,
    (VkDevice d, const VkMemoryAllocateInfo* i, const VkAllocationCallbacks* a, VkDeviceMemory* o),
    (d, i, a, o))
IGL_FAULT_STUB(
    MapMemory,
    kMapMemory,
    MapMemory,
    (VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize s, VkMemoryMapFlags f, void** p),
    (d, m, o, s, f, p))
IGL_FAULT_STUB(BindBufferMemory,
               kBindBufferMemory,
               BindBufferMemory,
               (VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o),
               (d, b, m, o))
IGL_FAULT_STUB(BindBufferMemory2,
               kBindBufferMemory2,
               BindBufferMemory2,
               (VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* i),
               (d, n, i))
IGL_FAULT_STUB(BindBufferMemory2KHR,
               kBindBufferMemory2KHR,
               BindBufferMemory2,
               (VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* i),
               (d, n, i))
IGL_FAULT_STUB(BindImageMemory,
               kBindImageMemory,
               BindImageMemory,
               (VkDevice d, VkImage b, VkDeviceMemory m, VkDeviceSize o),
               (d, b, m, o))
IGL_FAULT_STUB(BindImageMemory2,
               kBindImageMemory2,
               BindImageMemory2,
               (VkDevice d, uint32_t n, const VkBindImageMemoryInfo* i),
               (d, n, i))
IGL_FAULT_STUB(BindImageMemory2KHR,
               kBindImageMemory2KHR,
               BindImageMemory2,
               (VkDevice d, uint32_t n, const VkBindImageMemoryInfo* i),
               (d, n, i))
#undef IGL_FAULT_STUB

VKAPI_ATTR void VKAPI_CALL stubDestroyImage(VkDevice d, VkImage i, const VkAllocationCallbacks* a) {
  gFault.counted[static_cast<size_t>(FaultingDevice::Counted::DestroyImage)].fetch_add(1);
  reinterpret_cast<PFN_vkDestroyImage>(gFault.real[kDestroyImage])(d, i, a);
}

VKAPI_ATTR void VKAPI_CALL stubDestroyBuffer(VkDevice d,
                                             VkBuffer b,
                                             const VkAllocationCallbacks* a) {
  gFault.counted[static_cast<size_t>(FaultingDevice::Counted::DestroyBuffer)].fetch_add(1);
  reinterpret_cast<PFN_vkDestroyBuffer>(gFault.real[kDestroyBuffer])(d, b, a);
}

VKAPI_ATTR void VKAPI_CALL stubFreeMemory(VkDevice d,
                                          VkDeviceMemory m,
                                          const VkAllocationCallbacks* a) {
  gFault.counted[static_cast<size_t>(FaultingDevice::Counted::FreeMemory)].fetch_add(1);
  reinterpret_cast<PFN_vkFreeMemory>(gFault.real[kFreeMemory])(d, m, a);
}

VKAPI_ATTR VkResult VKAPI_CALL
stubSetDebugUtilsObjectNameEXT(VkDevice d, const VkDebugUtilsObjectNameInfoEXT* i) {
  gFault.counted[static_cast<size_t>(FaultingDevice::Counted::SetDebugUtilsObjectName)].fetch_add(
      1);
  return reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(gFault.real[kSetDebugUtilsObjectName])(
      d, i);
}

VKAPI_ATTR VkResult VKAPI_CALL stubQueueWaitIdle(VkQueue q) {
  bool expected = false;
  if (gFault.teardownSeen.compare_exchange_strong(expected, true)) {
    for (size_t i = 0; i < kCountedCount; ++i) {
      gFault.countedAtTeardown[i].store(gFault.counted[i].load());
    }
  }
  return reinterpret_cast<PFN_vkQueueWaitIdle>(gFault.real[kQueueWaitIdle])(q);
}

struct Hook {
  const char* name;
  Entry entry;
  PFN_vkVoidFunction stub;
};

#define IGL_HOOK(NAME, ENTRY, STUB) {NAME, ENTRY, reinterpret_cast<PFN_vkVoidFunction>(&STUB)}

const std::array<Hook, kEntryCount> kHooks = {{
    IGL_HOOK("vkCreateImage", kCreateImage, stubCreateImage),
    IGL_HOOK("vkCreateBuffer", kCreateBuffer, stubCreateBuffer),
    IGL_HOOK("vkCreateImageView", kCreateImageView, stubCreateImageView),
    IGL_HOOK("vkAllocateMemory", kAllocateMemory, stubAllocateMemory),
    IGL_HOOK("vkMapMemory", kMapMemory, stubMapMemory),
    IGL_HOOK("vkBindBufferMemory", kBindBufferMemory, stubBindBufferMemory),
    IGL_HOOK("vkBindBufferMemory2", kBindBufferMemory2, stubBindBufferMemory2),
    IGL_HOOK("vkBindBufferMemory2KHR", kBindBufferMemory2KHR, stubBindBufferMemory2KHR),
    IGL_HOOK("vkBindImageMemory", kBindImageMemory, stubBindImageMemory),
    IGL_HOOK("vkBindImageMemory2", kBindImageMemory2, stubBindImageMemory2),
    IGL_HOOK("vkBindImageMemory2KHR", kBindImageMemory2KHR, stubBindImageMemory2KHR),
    IGL_HOOK("vkDestroyImage", kDestroyImage, stubDestroyImage),
    IGL_HOOK("vkDestroyBuffer", kDestroyBuffer, stubDestroyBuffer),
    IGL_HOOK("vkFreeMemory", kFreeMemory, stubFreeMemory),
    IGL_HOOK("vkSetDebugUtilsObjectNameEXT",
             kSetDebugUtilsObjectName,
             stubSetDebugUtilsObjectNameEXT),
    IGL_HOOK("vkQueueWaitIdle", kQueueWaitIdle, stubQueueWaitIdle),
}};
#undef IGL_HOOK

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL stubGetDeviceProcAddr(VkDevice device, const char* name) {
  const PFN_vkVoidFunction real = gFault.realGetDeviceProcAddr(device, name);
  if (real == nullptr || name == nullptr) {
    return real;
  }
  for (const Hook& hook : kHooks) {
    if (std::strcmp(name, hook.name) == 0) {
      gFault.real[hook.entry] = real;
      return hook.stub;
    }
  }
  return real;
}

} // namespace

FaultingDevice::FaultingDevice(const igl::vulkan::VulkanContextConfig& config,
                               const std::function<void(FaultingDevice&)>& armBeforeCreate) {
  IGL_DEBUG_ASSERT(!gFault.live, "only one FaultingDevice may be alive at a time");
  gFault.live = true;
  disarmAll();
  resetCounters();
  gFault.real.fill(nullptr);

#if IGL_PLATFORM_MACOSX && !defined(IGL_USE_STATIC_KOSMICKRISP)
  ::igl::vulkan::setupMoltenVKEnvironment();
#endif

  auto ctx = igl::vulkan::HWDevice::createContext(config, nullptr);
  if (!ctx) {
    return;
  }

  Result ret;
  std::vector<HWDeviceDesc> devices =
      igl::vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::Unknown), &ret);
  if (!ret.isOk() || devices.empty()) {
    return;
  }

  // The heap-allocated context owns a non-const table: the wrapper goes in before the device
  // functions are loaded.
  auto& table = const_cast<VulkanFunctionTable&>(ctx->vf_);
  gFault.realGetDeviceProcAddr = table.vkGetDeviceProcAddr;
  table.vkGetDeviceProcAddr = stubGetDeviceProcAddr;
  // vkSetDebugUtilsObjectNameEXT is an INSTANCE function: the context loaded it already (when the
  // debug-utils extension is enabled), not through the device entry replaced above, so its counting
  // stub goes into the table directly.
  if (table.vkSetDebugUtilsObjectNameEXT != nullptr) {
    gFault.real[kSetDebugUtilsObjectName] =
        reinterpret_cast<PFN_vkVoidFunction>(table.vkSetDebugUtilsObjectNameEXT);
    table.vkSetDebugUtilsObjectNameEXT = stubSetDebugUtilsObjectNameEXT;
    namingObservable_ = true;
  }

  if (armBeforeCreate) {
    armBeforeCreate(*this);
  }

  std::vector<const char*> extraDeviceExtensions;
  extraDeviceExtensions.emplace_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);

  igl::vulkan::VulkanFeatures features(config);
  features.populateWithAvailablePhysicalDeviceFeatures(
      *ctx,
      reinterpret_cast<VkPhysicalDevice>(devices[0].guid)); // NOLINT(performance-no-int-to-ptr)

  device_ = igl::vulkan::HWDevice::create(std::move(ctx),
                                          devices[0],
                                          0, // width
                                          0, // height,
                                          extraDeviceExtensions.size(),
                                          extraDeviceExtensions.data(),
                                          &features,
                                          "Faulting Test Device",
                                          &ret);
  creationResult_ = ret;
  if (!ret.isOk()) {
    device_ = nullptr;
  }
}

FaultingDevice::~FaultingDevice() {
  device_.reset(); // the stubs stay valid until the device is gone
  disarmAll();
  gFault.live = false;
}

igl::vulkan::VulkanContext& FaultingDevice::context() const {
  return device_->getVulkanContext();
}

void FaultingDevice::arm(Call call, VkResult result, uint32_t skip, uint32_t count) {
  const auto index = static_cast<size_t>(call);
  gFault.result[index].store(static_cast<int32_t>(result));
  gFault.sequenceLength[index].store(0);
  gFault.ordinal[index].store(0);
  gFault.skip[index].store(skip);
  gFault.remaining[index].store(count);
}

void FaultingDevice::armSequence(Call call,
                                 std::initializer_list<VkResult> results,
                                 uint32_t skip) {
  IGL_DEBUG_ASSERT(results.size() > 0 && results.size() <= 4);
  const auto index = static_cast<size_t>(call);
  uint32_t n = 0;
  for (const VkResult result : results) {
    if (n < 4) {
      gFault.sequence[index][n++].store(static_cast<int32_t>(result));
    }
  }
  gFault.sequenceLength[index].store(n);
  gFault.ordinal[index].store(0);
  gFault.skip[index].store(skip);
  gFault.remaining[index].store(n);
}

void FaultingDevice::disarm(Call call) {
  const auto index = static_cast<size_t>(call);
  gFault.skip[index].store(0);
  gFault.remaining[index].store(0);
}

void FaultingDevice::disarmAll() {
  for (size_t i = 0; i < kCallCount; ++i) {
    disarm(static_cast<Call>(i));
  }
}

uint32_t FaultingDevice::reached(Call call) const {
  return gFault.reached[static_cast<size_t>(call)].load();
}

uint32_t FaultingDevice::failed(Call call) const {
  return gFault.failed[static_cast<size_t>(call)].load();
}

uint32_t FaultingDevice::counted(Counted counted) const {
  return gFault.counted[static_cast<size_t>(counted)].load();
}

bool FaultingDevice::teardownObserved() const {
  return gFault.teardownSeen.load();
}

uint32_t FaultingDevice::countedBeforeTeardown(Counted counted) const {
  return gFault.countedAtTeardown[static_cast<size_t>(counted)].load();
}

void FaultingDevice::resetCounters() {
  for (size_t i = 0; i < kCallCount; ++i) {
    gFault.reached[i].store(0);
    gFault.failed[i].store(0);
  }
  for (std::atomic<uint32_t>& counter : gFault.counted) {
    counter.store(0);
  }
  for (std::atomic<uint32_t>& counter : gFault.countedAtTeardown) {
    counter.store(0);
  }
  gFault.teardownSeen.store(false);
}

} // namespace igl::tests::util::device::vulkan
