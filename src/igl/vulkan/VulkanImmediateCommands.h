/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <vector>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanFence.h>
#include <igl/vulkan/VulkanHelpers.h>
#include <igl/vulkan/VulkanSemaphore.h>

namespace igl::tests {
class VulkanImmediateCommandsTest;
}

namespace igl::vulkan {

/// @brief This class provides a simplified interface for obtaining and submitting Command Buffers,
/// while providing features to help manage their synchronization.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class VulkanImmediateCommands final {
 public:
  // The maximum number of command buffers which can simultaneously exist in the system; when we run
  // out of buffers, we stall and wait until an existing buffer becomes available
  static constexpr uint32_t kMaxCommandBuffers = 32;
  static constexpr uint32_t kMaxInjectedSemaphores = 4;

  /** @brief Creates an instance of the class for a specific queue family and whether the fences
   * created for each command buffer are exportable (see VulkanFence for more details about the
   * exportable flag). The optional `debugName` parameter can be used to name the resource to make
   * it easier for debugging
   * The constructor initializes the vector of `CommandBufferWrapper` structures with
   * a total of `kMaxCommandBuffers`
   * When `chainSubmitsWithSemaphores` is true, every submission waits on a binary semaphore
   * signaled by the previous one. When false, each command buffer starts with a full pipeline
   * barrier instead, which orders it after all earlier work on the queue; only a submission that
   * carries injected wait semaphores still chains to the next one, because later submissions are
   * otherwise not ordered after a semaphore wait.
   */
  VulkanImmediateCommands(const VulkanFunctionTable& vf,
                          VkDevice device,
                          uint32_t queueFamilyIndex,
                          bool exportableFences,
                          bool useTimelineSemaphore,
                          bool useSynchronization2,
                          const char* debugName,
                          bool chainSubmitsWithSemaphores = true);
  ~VulkanImmediateCommands();
  VulkanImmediateCommands(const VulkanImmediateCommands&) = delete;
  VulkanImmediateCommands& operator=(const VulkanImmediateCommands&) = delete;
  VulkanImmediateCommands(VulkanImmediateCommands&&) = delete;
  VulkanImmediateCommands& operator=(VulkanImmediateCommands&&) = delete;

  /** @brief A structure that encapsulates synchronization information about command buffers and
   * that is used by the `VulkanImmediateCommands` class to manage command buffer acquisition and
   * reuse.
   * A `SubmitHandle` is composed of two 32-bit integers, a buffer index (`bufferIndex_`) and a
   * submit id (`submitId_`). The buffer index is associated with the location of the command buffer
   * in the vector in which they are stored in `VulkanImmediateCommands` class. The submit id is a
   * monotonically increasing index that is incremented every time we `submit()` a command buffer
   * for execution (any command buffer). A handle is a combination of those two values into a 64-bit
   * integer: the submit id is shifted and occupies the 32 most significant bits of the handle,
   * while the buffer index occupies the least significant 32 bits
   */
  struct SubmitHandle {
    uint32_t bufferIndex = 0;
    uint32_t submitId = 0;
    SubmitHandle() = default;

    /// @brief Creates a SubmitHandle object from an existing handle
    explicit SubmitHandle(uint64_t handle) :
      bufferIndex(static_cast<uint32_t>(handle & 0xffffffff)),
      submitId(static_cast<uint32_t>(handle >> 32)) {
      IGL_DEBUG_ASSERT(submitId);
    }

    /// @brief Checks whether the structure is empty and has not been associates with a command
    /// buffer submission yet
    [[nodiscard]] bool empty() const {
      return submitId == 0;
    }

    /// @brief Returns a unique identifiable handle, which is made of the `submitId_` and the
    /// `bufferIndex_` member variables
    [[nodiscard]] uint64_t handle() const {
      return (static_cast<uint64_t>(submitId) << 32) + bufferIndex;
    }

    [[nodiscard]] bool operator==(const SubmitHandle& rhs) const {
      return bufferIndex == rhs.bufferIndex && submitId == rhs.submitId;
    }
    [[nodiscard]] bool operator!=(const SubmitHandle& rhs) const {
      return !(*this == rhs);
    }
  };

  /// Ensures that the `SubmitHandle` structure size is not larger than a `uint64_t`
  static_assert(sizeof(SubmitHandle) == sizeof(uint64_t));
  static_assert(std::is_trivially_copyable_v<SubmitHandle>);

  /// @brief The CommandBufferWrapper structure encapsulates all the information needed to manage
  /// the synchronization of a command buffer along with a command buffer
  struct CommandBufferWrapper {
    CommandBufferWrapper(VulkanFence&& fence, VulkanSemaphore&& semaphore) :
      fence(std::move(fence)), semaphore(std::move(semaphore)) {}

    /// @brief The command buffer handle. It is initialied to VK_NULL_HANDLE. The command buffer
    /// handle stored in `cmdBufAllocated_` is copied into `cmdBuf_` when the command buffer is
    /// acquired for recording
    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    /// @brief Stores the command buffer handle allocated during initialization
    VkCommandBuffer cmdBufAllocated = VK_NULL_HANDLE;
    /// @brief the SubmitHandle object used to synchronize this command buffer
    SubmitHandle handle = {};
    /// @brief A VulkanFence object that is associated with the submission of the command buffer. It
    /// is used to check whether a command buffer is still executing or for waiting the command
    /// buffer to finish execution by the GPU
    VulkanFence fence;
    /// @brief A VulkanSemaphore object associated with the submission of the command buffer for
    /// execution.
    VulkanSemaphore semaphore;
    bool isEncoding = false;
    /// @brief The file descriptor for the underlying VkFence. It's only populated if an FD is set
    /// explicitly using VulkanImmediateCommands::storeFDInSubmitHandle(). It's reset in `acquire()`
    int fd = -1;
  };

  /// @brief Returns a `CommandBufferWrapper` object with the current command buffer (creates one if
  /// it does not exist) and its associated synchronization objects
  const CommandBufferWrapper& acquire();

  /** @brief Submits a command buffer (stored in a `CommandBufferWrapper` object) for submission and
   * returns the `SubmitHandle` associated with the command buffer. Caches the semaphore associated
   * with the command buffer bineg submitted as the last submitted semaphore
   * (`lastSubmitSemaphore_`). Caches the SubmitHandle associated with the command buffer being
   * submitted for execution in `lastSubmitHandle_`. Submitting a command buffer also marks the
   * `CommandBufferWrapper::isEncoding` variable to false and clears the injected wait- and
   * signal-semaphore queues.
   */
  SubmitHandle submit(const CommandBufferWrapper& wrapper);
  void discard(const CommandBufferWrapper& wrapper);
  /// @brief Returns `wrapper`, which was acquired and will never be submitted, to the available
  /// command buffers. The injected semaphores stay for the next submission, and a handle of
  /// `wrapper` no longer waits for anything.
  void release(const CommandBufferWrapper& wrapper);

  /// @brief Adds a semaphore to the waits for the next submission.
  [[nodiscard]] bool waitSemaphore(VkSemaphore semaphore, uint64_t waitValue = 0);
  /// @brief Adds a semaphore to the waits for the next submission at the specified destination
  /// pipeline stages.
  [[nodiscard]] bool waitSemaphore(VkSemaphore semaphore,
                                   uint64_t waitValue,
                                   VkPipelineStageFlags waitStageMask);
  /// @brief Adds a semaphore to the signals for the next submission.
  [[nodiscard]] bool signalSemaphore(VkSemaphore semaphore, uint64_t signalValue);

  [[nodiscard]] bool hasPendingInjectedSemaphores() const noexcept {
    return numWaitSemaphores_ != 0 || numSignalSemaphores_ != 0;
  }

  /// @brief Drops the most recently injected wait semaphore (see `waitSemaphore()`). Used to roll
  /// back a wait that was reserved before the operation meant to signal it ran, when that operation
  /// fails to signal it (e.g. a `vkAcquireNextImageKHR()` that returns `VK_ERROR_OUT_OF_DATE_KHR`,
  /// which leaves the acquire semaphore unsignaled).
  void cancelLastWaitSemaphore() noexcept {
    if (numWaitSemaphores_ > 0) {
      --numWaitSemaphores_;
    }
  }

  /// @brief Returns true if a command buffer has been acquired and not yet submitted or discarded
  [[nodiscard]] bool isAnyCommandBufferEncoding() const noexcept;

  /// @brief Returns the last SubmitHandle, which was submitted when `submit()` was last called
  [[nodiscard]] SubmitHandle getLastSubmitHandle() const;
  [[nodiscard]] SubmitHandle getNextSubmitHandle() const;

  /** @brief Checks whether a SubmitHandle is ready. A SubmitHandle is ready if it is recycled or
   * empty. If it has not been recycled and is not empty, a SubmitHandle is ready if the fence
   * associated with the command buffer referred by the SubmitHandle structure has been signaled.
   *  Note that this function does not wait for a fence to be signaled if it has not been signaled.
   * It merely checks the fence status
   */
  [[nodiscard]] bool isReady(SubmitHandle handle) const;

  /// @brief If the SubmitHandle is not ready, this function waits for the fence associated with the
  /// command buffer referred by the handle to become signaled. The default wait time is
  /// `UINT64_MAX` nanoseconds. Returns a result code if the wait was successful or not.
  VkResult wait(SubmitHandle handle, uint64_t timeoutNanoseconds = UINT64_MAX);

  /// @brief Wait for _all_ fences for all command buffers stored in `VulkanImmediateCommands` to
  /// become signaled. The maximum wait time is `UINT64_MAX` nanoseconds
  void waitAll();

  /// @brief Returns the fence associated with the handle if the handle has not been recycled.
  /// Returns `VK_NULL_HANDLE` otherwise.
  VkFence getVkFenceFromSubmitHandle(SubmitHandle handle);

  /// @brief Stores the file descriptor in the `CommandBufferWrapper` object associated with the
  /// handle. Chceks for bounds, but does not check for validity.
  void storeFDInSubmitHandle(SubmitHandle handle, int fd) noexcept;

  /// @brief Returns the file descriptor associated with the handle and its underlying VkFence. If
  /// the FD has not been explicitly set with `storeFDInSubmitHandle()`, it returns -1. This
  /// function DOES NOT retrieve the FD from the Vulkan implementation
  [[nodiscard]] int cachedFDFromSubmitHandle(SubmitHandle handle) const noexcept;

  /// @brief Returns true once a Vulkan call made by this object has returned
  /// VK_ERROR_DEVICE_LOST. Device loss is permanent, so it never becomes false again. Safe to call
  /// from any thread.
  [[nodiscard]] bool isDeviceLost() const noexcept {
    return deviceLost_.load(std::memory_order_relaxed);
  }

 private:
  friend class CommandQueue;
  friend class igl::tests::VulkanImmediateCommandsTest;
  /// @brief If `nextCmdBuf` is not null, it is a primary command buffer that the submission runs
  /// after the one of `wrapper`, such as the recording of a reusable command buffer.
  SubmitHandle submit(const CommandBufferWrapper& wrapper,
                      VkSemaphore presentationSemaphore,
                      VkCommandBuffer nextCmdBuf = VK_NULL_HANDLE);

  /// @brief Resets all commands buffers and their associated fences that are valid, are not being
  /// encoded, and have completed execution by the GPU (their fences have been signaled). Resets the
  /// number of available command buffers.
  void purge();
  /// @brief Checks whether the SubmitHandle is recycled. A recycled SubmitHandle is a handle that
  /// has a submit id greater than the submit id associated with the same command buffer stored
  /// internally in `VulkanImmediateCommands`. A SubmitHandle handle is also recycled if it's empty
  [[nodiscard]] bool isRecycled(SubmitHandle handle) const;

  /// @brief Returns true if `result` is VK_ERROR_DEVICE_LOST, after recording it in `deviceLost_`
  bool checkDeviceLost(VkResult result) const noexcept;

 private:
  const VulkanFunctionTable& vf_;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  std::string debugName_;
  std::vector<CommandBufferWrapper> buffers_;

  /// @brief The last submitted handle. Updated on `submit()`
  SubmitHandle lastSubmitHandle_ = SubmitHandle();
  SubmitHandle nextSubmitHandle_ = SubmitHandle();

  /// @brief The semaphore submitted with the last command buffer. Updated on `submit()`
  VkSemaphoreSubmitInfo lastSubmitSemaphore_{};
  // @lint-ignore CLANGTIDY
  VkSemaphoreSubmitInfo waitSemaphores_[kMaxInjectedSemaphores] = {};
  // @lint-ignore CLANGTIDY
  VkSemaphoreSubmitInfo signalSemaphores_[kMaxInjectedSemaphores] = {};
  uint32_t numWaitSemaphores_ = 0;
  uint32_t numSignalSemaphores_ = 0;

  uint32_t numAvailableCommandBuffers_ = kMaxCommandBuffers;

  // @brief The submission counter. Incremented on `submit()`
  uint32_t submitCounter_ = 1;

  bool useTimelineSemaphore_ = false;
  bool useTimelineSemaphoreAndSynchronization2_ = false;
  bool chainSubmitsWithSemaphores_ = true;

  /// @brief Set once a Vulkan call returned VK_ERROR_DEVICE_LOST. Updated from const methods
  /// (`isReady()`), and read from any thread by `isDeviceLost()`
  mutable std::atomic<bool> deviceLost_{false};
};

} // namespace igl::vulkan
