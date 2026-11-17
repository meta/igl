/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <igl/vulkan/Device.h>

namespace igl::tests::util::device::vulkan {

/**
 Configure and return a context configuration.
 */
igl::vulkan::VulkanContextConfig getContextConfig(bool enableValidation = true);

/**
 Create and return an igl::Device that is suitable for running tests against.
 */
std::unique_ptr<igl::vulkan::Device> createTestDevice(
    const igl::vulkan::VulkanContextConfig& config);
std::unique_ptr<igl::vulkan::Device> createTestDevice(
    const igl::vulkan::VulkanContextConfig& config,
    const std::function<void(igl::vulkan::VulkanFeatures&)>& configureFeatures);

/**
 Helper to create a Vulkan device with default configuration and optional validation.
 */
std::unique_ptr<igl::vulkan::Device> createTestDevice(bool enableValidation = true);

/**
 A device whose own function table lets a test fail chosen Vulkan calls.

 The device is built in the two public steps `HWDevice::createContext()` and `HWDevice::create()`.
 Between them the context's `vkGetDeviceProcAddr` is replaced with a wrapper that returns forwarding
 stubs for the calls in `Call` (each captures the real pointer when it is looked up) and, for the
 calls in `Counted`, stubs that only count. `initContext()` loads the device table through that
 entry and VMA copies the table's pointers when it creates its allocator, so the production
 `iglVulkan` objects (VulkanBuffer, VulkanImage, the staging device, VulkanContext and VMA) run as
 compiled for production and a test only arms a VkResult per call. No production code and no
 preprocessor flag are involved.

 Faults are armed after `create()` returned, except through the `armBeforeCreate` callback of the
 constructor, which runs between the table replacement and `create()`. One instance
 may be alive at a time (the stubs are plain functions over one shared state); the destructor
 destroys the device before the state is cleared.
 */
class FaultingDevice {
 public:
  /// The calls a test can fail. The `...2` bind calls and their KHR aliases share one entry.
  enum class Call : uint8_t {
    CreateImage,
    CreateBuffer,
    CreateImageView,
    AllocateMemory,
    MapMemory,
    BindBufferMemory,
    BindBufferMemory2,
    BindImageMemory,
    BindImageMemory2,
    CreateFence,
    CreateSemaphore,
    AllocateCommandBuffers,
    CreateCommandPool,
    /// vkGetAndroidHardwareBufferPropertiesANDROID (only looked up on devices that expose it).
    GetAndroidHardwareBufferProperties,
    Count,
  };
  /// The calls the fixture only counts.
  enum class Counted : uint8_t {
    DestroyImage,
    DestroyBuffer,
    FreeMemory,
    SetDebugUtilsObjectName,
    /// The destroy calls of fences, semaphores and command pools count only a non-null handle.
    DestroyFence,
    DestroySemaphore,
    DestroyCommandPool,
    /// A destroy call of a fence, semaphore or command pool with a handle this fixture did not see
    /// created (not forwarded to the driver).
    BadDestroy,
    Count,
  };

  /// Builds the device. `armBeforeCreate` runs between the table replacement and `create()`.
  explicit FaultingDevice(const igl::vulkan::VulkanContextConfig& config,
                          const std::function<void(FaultingDevice&)>& armBeforeCreate = {});
  ~FaultingDevice();
  FaultingDevice(const FaultingDevice&) = delete;
  FaultingDevice& operator=(const FaultingDevice&) = delete;

  /// Null when the device could not be created (for example because an armed fault hit `create()`).
  [[nodiscard]] igl::vulkan::Device* device() const {
    return device_.get();
  }
  [[nodiscard]] igl::vulkan::VulkanContext& context() const;
  /// The result `HWDevice::create()` returned (OK when the device exists).
  /// Whether `Counted::SetDebugUtilsObjectName` can move: the debug-utils entry point exists on
  /// this device. A test that asserts nothing was named checks this first and asserts that a named
  /// object moves the counter.
  [[nodiscard]] bool namingObservable() const {
    return namingObservable_;
  }

  [[nodiscard]] const Result& creationResult() const {
    return creationResult_;
  }

  /// Fails calls of `call` with `result`: the first `skip` calls pass, then `count` calls fail.
  void arm(Call call, VkResult result, uint32_t skip = 0, uint32_t count = 1);
  /// Fails the next `results.size()` (at most 4) calls after `skip`, the n-th with `results[n]`.
  void armSequence(Call call, std::initializer_list<VkResult> results, uint32_t skip = 0);
  void disarm(Call call);
  void disarmAll();

  /// Calls that reached the stub (forwarded or failed), and calls that were failed.
  [[nodiscard]] uint32_t reached(Call call) const;
  [[nodiscard]] uint32_t failed(Call call) const;
  [[nodiscard]] uint32_t counted(Counted counted) const;
  /// Whether the context was torn down (its first `vkQueueWaitIdle()` was seen) since the last
  /// `resetCounters()`, and the counters as they stood at that moment: the calls a failing
  /// creation made before any teardown could drain deferred destruction.
  [[nodiscard]] bool teardownObserved() const;
  [[nodiscard]] uint32_t countedBeforeTeardown(Counted counted) const;
  void resetCounters();

  /// Waits for the staging pool, returns every finished block and releases every staging buffer
  /// that is recovered whole (the context's own dummy upload used one).
  void drainStaging() const;

 private:
  std::unique_ptr<igl::vulkan::Device> device_;
  Result creationResult_;
  bool namingObservable_ = false;
};

} // namespace igl::tests::util::device::vulkan
