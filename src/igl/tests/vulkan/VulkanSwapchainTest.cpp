/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/Common.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
#include <igl/tests/util/device/vulkan/TestDevice.h> // IWYU pragma: export
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanSwapchain.h> // IWYU pragma: keep

#if IGL_PLATFORM_MACOSX
#include <igl/vulkan/moltenvk/MoltenVkHelpers.h>
#endif

namespace igl::tests {

namespace {
[[maybe_unused]] constexpr uint32_t kWidth = 1024;
[[maybe_unused]] constexpr uint32_t kHeight = 1024;
} // namespace

//
// VulkanSwapchainTest
//
// Unit tests for igl::vulkan::VulkanSwapchain
//
#if !IGL_PLATFORM_MACOSX
class VulkanSwapchainTest : public ::testing::Test {
 public:
  // Set up common resources.
  void SetUp() override {
#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID
    // @fb-only
    GTEST_SKIP() << "Fix these tests on Windows and Android, no headless surface support there.";
#else
    igl::setDebugBreakEnabled(false);

    // Add headless config for creating swapchains without a window
    igl::vulkan::VulkanContextConfig config = util::device::vulkan::getContextConfig(true);
    config.headless = true;

    device_ = igl::tests::util::device::vulkan::createTestDevice(config);
    ASSERT_TRUE(device_ != nullptr);
    auto& device = static_cast<igl::vulkan::Device&>(*device_);
    context_ = &device.getVulkanContext();
    ASSERT_TRUE(context_ != nullptr);
#endif
  }

 protected:
  std::shared_ptr<IDevice> device_;
  vulkan::VulkanContext* context_ = nullptr;
};

TEST_F(VulkanSwapchainTest, CreateVulkanSwapchain) {
#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID
  // @fb-only
  GTEST_SKIP() << "Fix these tests on Windows and Android, no headless surface support there.";
#else
  auto swapchain = std::make_unique<igl::vulkan::VulkanSwapchain>(*context_, kWidth, kHeight);
  ASSERT_NE(swapchain, nullptr);

  ASSERT_EQ(swapchain->getWidth(), kWidth);
  ASSERT_EQ(swapchain->getHeight(), kHeight);

  const VkExtent2D extent = swapchain->getExtent();
  ASSERT_EQ(extent.width, kWidth);
  ASSERT_EQ(extent.height, kHeight);

  ASSERT_NE(swapchain->getFormatColor(), VK_FORMAT_UNDEFINED);

  ASSERT_GT(swapchain->getNumSwapchainImages(), 0);

  ASSERT_EQ(swapchain->getCurrentImageIndex(), 0);
#endif
}

TEST_F(VulkanSwapchainTest, DepthTextureIsLazilyAllocated) {
#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID
  GTEST_SKIP() << "Fix these tests on Windows and Android, no headless surface support there.";
#else
  auto swapchain = std::make_unique<igl::vulkan::VulkanSwapchain>(*context_, kWidth, kHeight);
  ASSERT_NE(swapchain, nullptr);

  const auto depthTexture = swapchain->getCurrentDepthTexture();
  ASSERT_NE(depthTexture, nullptr);
  EXPECT_NE(depthTexture->image.getVkImage(), VK_NULL_HANDLE);
#endif
}

TEST_F(VulkanSwapchainTest, AcquireNextImage) {
#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID
  GTEST_SKIP() << "Fix these tests on Windows and Android, no headless surface support there.";
#else
  auto swapchain = std::make_unique<igl::vulkan::VulkanSwapchain>(*context_, kWidth, kHeight);
  ASSERT_NE(swapchain, nullptr);

  const Result result = swapchain->acquireNextImage();
  EXPECT_TRUE(result.isOk()) << result.message.c_str();

  EXPECT_NE(swapchain->getCurrentVkImage(), VK_NULL_HANDLE);
  EXPECT_NE(swapchain->getCurrentVkImageView(), VK_NULL_HANDLE);
#endif
}
#endif // !IGL_PLATFORM_MACOSX

class VulkanSwapchainRecoveryTest : public ::testing::TestWithParam<bool> {
 public:
  void SetUp() override {
    igl::setDebugBreakEnabled(false);
#if IGL_PLATFORM_MACOSX && !defined(IGL_USE_STATIC_KOSMICKRISP)
    vulkan::setupMoltenVKEnvironment();
#endif
    active = this;
    auto config = util::device::vulkan::getContextConfig();
    config.enableValidationOnAndroid = true;
    config.enableGPUAssistedValidation = false;
    auto context = vulkan::HWDevice::createContext(config, nullptr);
    ASSERT_NE(context, nullptr);
    auto& table = const_cast<VulkanFunctionTable&>(context->vf_);
    table.vkCreateHeadlessSurfaceEXT = mockCreateHeadlessSurface;
    table.vkDestroySurfaceKHR = mockDestroySurface;
    table.vkGetPhysicalDeviceSurfaceSupportKHR = mockGetSurfaceSupport;
    table.vkGetPhysicalDeviceSurfaceCapabilitiesKHR = mockGetSurfaceCapabilities;
    table.vkGetPhysicalDeviceSurfaceFormatsKHR = mockGetSurfaceFormats;
    table.vkGetPhysicalDeviceSurfacePresentModesKHR = mockGetSurfacePresentModes;
    context->createHeadlessSurface();
    Result result;
    const auto devices =
        vulkan::HWDevice::queryDevices(*context, HWDeviceQueryDesc(HWDeviceType::Unknown), &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_FALSE(devices.empty());
    device_ = vulkan::HWDevice::create(std::move(context),
                                       devices.front(),
                                       0,
                                       0,
                                       0,
                                       nullptr,
                                       nullptr,
                                       "Swapchain recovery",
                                       &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(device_, nullptr);
    auto& ctx = device_->getVulkanContext();
    auto& deviceTable = const_cast<VulkanFunctionTable&>(ctx.vf_);
    original_ = deviceTable;
    deviceTable.vkCreateSwapchainKHR = mockCreateSwapchain;
    deviceTable.vkDestroySwapchainKHR = mockDestroySwapchain;
    deviceTable.vkGetSwapchainImagesKHR = mockGetSwapchainImages;
    deviceTable.vkSetDebugUtilsObjectNameEXT = mockSetDebugUtilsObjectName;
    deviceTable.vkAcquireNextImageKHR = mockAcquireNextImage;
    for (uint32_t index = 0; index < kImageCount; ++index) {
      images_.push_back(ctx.createImage(VK_IMAGE_TYPE_2D,
                                        {.width = kSize, .height = kSize, .depth = 1},
                                        VK_FORMAT_R8G8B8A8_UNORM,
                                        1,
                                        1,
                                        VK_IMAGE_TILING_OPTIMAL,
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        0,
                                        VK_SAMPLE_COUNT_1_BIT,
                                        &result,
                                        "Swapchain recovery image"));
      ASSERT_TRUE(result.isOk()) << result.message;
      ASSERT_NE(images_.back().getVkImage(), VK_NULL_HANDLE);
    }
    ASSERT_TRUE(ctx.initSwapchain(kSize, kSize).isOk());
    if (!GetParam()) {
      ctx.timelineSemaphore_.reset();
    } else if (ctx.timelineSemaphore_ == nullptr) {
      GTEST_SKIP() << "Timeline semaphores are unavailable on this device";
    }
    RecordProperty("validation_layers", ctx.areValidationLayersEnabled());
  }

  void TearDown() override {
    if (device_ != nullptr) {
      auto& ctx = device_->getVulkanContext();
      if (ctx.swapchain_ != nullptr) {
        for (auto& fence : ctx.swapchain_->acquireFences) {
          if (!fence.wait(0)) {
            EXPECT_TRUE(fence.signal(ctx.deviceQueues_.graphicsQueue));
          }
        }
        ctx.swapchain_.reset();
      }
      images_.clear();
      device_.reset();
    }
    active = nullptr;
  }

 protected:
  static VKAPI_ATTR VkResult VKAPI_CALL
  mockCreateHeadlessSurface(VkInstance /*unused*/,
                            const VkHeadlessSurfaceCreateInfoEXT* /*unused*/,
                            const VkAllocationCallbacks* /*unused*/,
                            VkSurfaceKHR* surface) {
    *surface = mockHandle<VkSurfaceKHR>();
    return VK_SUCCESS;
  }

  static VKAPI_ATTR void VKAPI_CALL mockDestroySurface(VkInstance /*unused*/,
                                                       VkSurfaceKHR /*unused*/,
                                                       const VkAllocationCallbacks* /*unused*/) {}

  static VKAPI_ATTR VkResult VKAPI_CALL mockGetSurfaceSupport(VkPhysicalDevice /*unused*/,
                                                              uint32_t /*unused*/,
                                                              VkSurfaceKHR /*unused*/,
                                                              VkBool32* supported) {
    *supported = VK_TRUE;
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL mockGetSurfaceCapabilities(VkPhysicalDevice /*unused*/,
                                                                   VkSurfaceKHR /*unused*/,
                                                                   VkSurfaceCapabilitiesKHR* caps) {
    *caps = {.minImageCount = kImageCount,
             .maxImageCount = kImageCount,
             .currentExtent = {kSize, kSize},
             .maxImageArrayLayers = 1,
             .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
             .supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
             .supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL mockGetSurfaceFormats(VkPhysicalDevice /*unused*/,
                                                              VkSurfaceKHR /*unused*/,
                                                              uint32_t* count,
                                                              VkSurfaceFormatKHR* formats) {
    *count = 1;
    if (formats != nullptr) {
      *formats = {.format = VK_FORMAT_R8G8B8A8_UNORM,
                  .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL mockGetSurfacePresentModes(VkPhysicalDevice /*unused*/,
                                                                   VkSurfaceKHR /*unused*/,
                                                                   uint32_t* count,
                                                                   VkPresentModeKHR* modes) {
    *count = 1;
    if (modes != nullptr) {
      *modes = VK_PRESENT_MODE_FIFO_KHR;
    }
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL
  mockCreateSwapchain(VkDevice /*unused*/,
                      const VkSwapchainCreateInfoKHR* /*unused*/,
                      const VkAllocationCallbacks* /*unused*/,
                      VkSwapchainKHR* swapchain) {
    *swapchain = mockHandle<VkSwapchainKHR>();
    return VK_SUCCESS;
  }

  static VKAPI_ATTR void VKAPI_CALL mockDestroySwapchain(VkDevice /*unused*/,
                                                         VkSwapchainKHR /*unused*/,
                                                         const VkAllocationCallbacks* /*unused*/) {}

  static VKAPI_ATTR VkResult VKAPI_CALL mockGetSwapchainImages(VkDevice /*unused*/,
                                                               VkSwapchainKHR /*unused*/,
                                                               uint32_t* count,
                                                               VkImage* images) {
    *count = kImageCount;
    if (images != nullptr) {
      for (uint32_t index = 0; index < kImageCount; ++index) {
        images[index] = active->images_[index].getVkImage();
      }
    }
    return VK_SUCCESS;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL
  mockSetDebugUtilsObjectName(VkDevice device, const VkDebugUtilsObjectNameInfoEXT* info) {
    if (info->objectType == VK_OBJECT_TYPE_SWAPCHAIN_KHR ||
        active->original_.vkSetDebugUtilsObjectNameEXT == nullptr) {
      return VK_SUCCESS;
    }
    return active->original_.vkSetDebugUtilsObjectNameEXT(device, info);
  }

  static VKAPI_ATTR VkResult VKAPI_CALL mockAcquireNextImage(VkDevice device,
                                                             VkSwapchainKHR /*unused*/,
                                                             uint64_t /*unused*/,
                                                             VkSemaphore semaphore,
                                                             VkFence fence,
                                                             uint32_t* imageIndex) {
    if (++active->acquireCalls_ == 1) {
      *imageIndex = UINT32_MAX;
      if (fence != VK_NULL_HANDLE) {
        EXPECT_EQ(active->original_.vkGetFenceStatus(device, fence), VK_NOT_READY);
      }
      return VK_ERROR_OUT_OF_DATE_KHR;
    }
    *imageIndex = 1;
    const VkSubmitInfo signal = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                 .signalSemaphoreCount = 1,
                                 .pSignalSemaphores = &semaphore};
    return active->original_.vkQueueSubmit(
        active->device_->getVulkanContext().deviceQueues_.graphicsQueue, 1, &signal, fence);
  }

  static void recordDebugAbort(const char* /*unused*/,
                               const char* /*unused*/,
                               const char* /*unused*/,
                               const char* /*unused*/,
                               int /*unused*/,
                               const char* /*unused*/,
                               va_list /*unused*/) {
    ++active->debugAborts_;
  }

  template<typename Handle>
  static Handle mockHandle() {
    if constexpr (std::is_pointer_v<Handle>) {
      return reinterpret_cast<Handle>(active);
    } else {
      return static_cast<Handle>(reinterpret_cast<uintptr_t>(active));
    }
  }

  static constexpr uint32_t kImageCount = 2;
  static constexpr uint32_t kSize = 32;
  inline static thread_local VulkanSwapchainRecoveryTest* active = nullptr;
  std::unique_ptr<vulkan::Device> device_;
  std::vector<vulkan::VulkanImage> images_;
  VulkanFunctionTable original_{};
  uint32_t acquireCalls_ = 0;
  uint32_t debugAborts_ = 0;
};

TEST_P(VulkanSwapchainRecoveryTest, OutOfDateAcquirePreservesImageAndRecovers) {
  auto& ctx = device_->getVulkanContext();
  auto& swapchain = *ctx.swapchain_;
  auto* const platformDevice =
      static_cast<IDevice&>(*device_).getPlatformDevice<vulkan::PlatformDevice>();
  ASSERT_NE(platformDevice, nullptr);
  const uint32_t initialIndex = swapchain.getCurrentImageIndex();
  const auto listener = iglGetDebugAbortListener();
  iglSetDebugAbortListener(recordDebugAbort);
  Result failed;
  const auto failedDrawable = platformDevice->createTextureFromNativeDrawable(&failed);
  iglSetDebugAbortListener(listener);
  EXPECT_EQ(failedDrawable, nullptr);
  EXPECT_EQ(failed.code, Result::Code::InvalidOperation);
  EXPECT_EQ(failed.message, "Swapchain has no valid texture");
  EXPECT_EQ(debugAborts_, 0);
  EXPECT_EQ(acquireCalls_, 1);
  ASSERT_EQ(swapchain.getCurrentImageIndex(), initialIndex);
  ASSERT_FALSE(ctx.immediate_->hasPendingInjectedSemaphores());
  if (!GetParam()) {
    ASSERT_TRUE(swapchain.acquireFences[initialIndex].wait(0));
  }

  const Result recovered = swapchain.acquireNextImage();
  ASSERT_TRUE(recovered.isOk()) << recovered.message;
  EXPECT_EQ(swapchain.getCurrentImageIndex(), 1);
  EXPECT_EQ(acquireCalls_, 2);
  const auto& wrapper = ctx.immediate_->acquire();
  if (!GetParam()) {
    ASSERT_TRUE(ctx.immediate_->waitSemaphore(swapchain.getSemaphore()));
    EXPECT_TRUE(swapchain.acquireFences[initialIndex].wait(1'000'000'000));
  }
  const auto handle = ctx.immediate_->submit(wrapper);
  EXPECT_EQ(ctx.immediate_->wait(handle, 1'000'000'000), VK_SUCCESS);
  EXPECT_FALSE(ctx.immediate_->hasPendingInjectedSemaphores());
}

INSTANTIATE_TEST_SUITE_P(LegacyAndTimeline,
                         VulkanSwapchainRecoveryTest,
                         ::testing::Values(false, true));

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
