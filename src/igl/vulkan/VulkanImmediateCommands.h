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

  /// @brief Whether the command pool and all `kMaxCommandBuffers` rows (command buffer, fence and
  /// semaphore) were created. A pool that is not valid has no usable rows: only its destructor may
  /// run (it destroys what was created).
  [[nodiscard]] bool valid() const noexcept {
    return initResult_ == VK_SUCCESS;
  }
  /// @brief The `VkResult` of the first creation that failed, `VK_SUCCESS` for a valid pool.
  [[nodiscard]] VkResult initResult() const noexcept {
    return initResult_;
  }
  VulkanImmediateCommands(const VulkanImmediateCommands&) = delete;
  VulkanImmediateCommands& operator=(const VulkanImmediateCommands&) = delete;
  VulkanImmediateCommands(VulkanImmediateCommands&&) = delete;
  VulkanImmediateCommands& operator=(VulkanImmediateCommands&&) = delete;

  /** @brief A structure that encapsulates synchronization information about command buffers and
   * that is used by the `VulkanImmediateCommands` class to manage command buffer acquisition and
   * reuse.
   * A `SubmitHandle` is composed of two 32-bit integers, a buffer index (`bufferIndex_`) and a
   * submit id (`submitId_`). The buffer index is associated with the location of the command buffer
   * in the vector in which they are stored in `VulkanImmediateCommands` class. The submit id is the
   * low 32 bits of the recording's acquisition ordinal, a 64-bit count of `acquire()` calls that
   * skips the ordinals whose low 32 bits are 0. It identifies one recording: differences between
   * submit ids do not count submissions. A handle is a combination of those two values into a
   * 64-bit integer: the submit id is shifted and occupies the 32 most significant bits of the
   * handle, while the buffer index occupies the least significant 32 bits
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

 private:
  /// @brief Where a command buffer row is in its life. `acquire()` takes a `Free` row and makes it
  /// `Encoding`; `submit()` makes it `Submitted`; `discard()` and a completed `purge()` free it.
  enum class RowState : uint8_t { Free, Encoding, Submitted };
  /// @brief Retirement bookkeeping of a row: the identity of its current (or last) recording and
  /// what proves that recording's work finished.
  struct RowRetirement {
    /// @brief The acquisition ordinal of the row's current or last recording.
    uint64_t acquisition = 0;
    /// @brief `submitAttempts_` when that recording was acquired.
    uint64_t submitAttemptsAtAcquisition = 0;
    /// @brief What the recording waits for once it is closed: its own submission ordinal, or the
    /// submission counter when it was closed without a submission.
    uint64_t prefix = 0;
    /// @brief `prefix` of the row's previous recording, which bounds every older recording's.
    uint64_t previousPrefix = 0;
    /// @brief The oldest acquisition on this row that the current recording answers for: a
    /// recording that was closed without a submission, with no `submit()` call from its
    /// acquisition until the row was acquired again (the two shared one submit id), waits for the
    /// recording that took the row as well as for its own prefix.
    uint64_t aliasFloor = 0;
    /// @brief Whether the recording was closed without a submission (discarded).
    bool closedUnsubmitted = false;
  };

 public:
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
    /// @brief Where this row is in its life. `setRowState()` is the only writer of this, `cmdBuf`
    /// and `isEncoding`, so the three cannot drift.
    RowState state = RowState::Free;
    /// @brief The file descriptor for the underlying VkFence. It's only populated if an FD is set
    /// explicitly using VulkanImmediateCommands::storeFDInSubmitHandle(). It's reset in `acquire()`
    int fd = -1;

   private:
    friend class VulkanImmediateCommands;
    friend class igl::tests::VulkanImmediateCommandsTest;

    /// @brief What this row's current or last recording waits for (see `RowRetirement`).
    RowRetirement retirement_;
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
  /// command buffers. The injected semaphores stay for the next submission. As a discarded
  /// recording's, a handle of `wrapper` waits only for the work submitted before it (see
  /// `isReady()`).
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

  /** @brief Checks whether a SubmitHandle is ready: whether the GPU has finished everything
   * submitted to this pool up to the handle's recording. For a submitted recording that is all
   * submissions up to and including its own. For a discarded recording it is all submissions made
   * before it was discarded; recycling its command buffer does not make it ready. A recording that
   * is still open is not ready. An empty handle is ready.
   * This function never waits and never resets anything: it polls at most one fence, the lowest
   * one whose signal covers the work.
   */
  [[nodiscard]] bool isReady(SubmitHandle handle) const;

  /// @brief If the SubmitHandle is not ready, this function waits until the GPU has finished
  /// everything submitted to this pool up to the handle's recording (see `isReady()`), for at most
  /// `timeoutNanoseconds`, on the lowest fence whose signal covers that work. A zero timeout only
  /// polls. Returns `VK_SUCCESS`, `VK_ERROR_DEVICE_LOST` (which also means the work no longer
  /// runs), `VK_TIMEOUT`, the error of the fence wait, or `VK_ERROR_UNKNOWN` when the handle's
  /// recording is still open.
  VkResult wait(SubmitHandle handle, uint64_t timeoutNanoseconds = UINT64_MAX);

  /// @brief Wait for _all_ fences for all command buffers stored in `VulkanImmediateCommands` to
  /// become signaled. The maximum wait time is `UINT64_MAX` nanoseconds. A recording that is still
  /// open is not waited for.
  void waitAll();

  /// @brief Returns the fence associated with the handle until its command buffer is acquired
  /// again. Returns `VK_NULL_HANDLE` for an empty handle and for a recording whose row was reused.
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
  /// @brief Moves `wrapper` to `state` and updates its `cmdBuf` and `isEncoding` fields that mirror
  /// it, so they cannot drift. The only writer of the three.
  void setRowState(CommandBufferWrapper& wrapper, RowState state) noexcept;

  /// @brief What a handle waits for: nothing, the close of a recording that is still open, or a
  /// submission ordinal that `retiredThrough_` must reach.
  struct Ticket {
    enum class Kind : uint8_t { Settled, Open, Prefix } kind = Kind::Settled;
    uint64_t prefix = 0;
    /// @brief The row whose current recording the handle names, when it does.
    const CommandBufferWrapper* IGL_NULLABLE own = nullptr;
  };
  [[nodiscard]] Ticket ticketOf(SubmitHandle handle) const noexcept;
  [[nodiscard]] Ticket ticketOfRow(const CommandBufferWrapper& wrapper) const noexcept;

  /// @brief The lowest fence whose signal proves that submission ordinal `target` finished: the
  /// fence of a submitted row. `own` is the handle's row, checked first. `ordinal` receives what
  /// the fence covers.
  [[nodiscard]] const VkFence* IGL_NULLABLE
  lowestSourceFor(uint64_t target,
                  const CommandBufferWrapper* IGL_NULLABLE own,
                  uint64_t& ordinal) const noexcept;

  /// @brief Records that every submission up to `ordinal` finished, or no longer runs after a
  /// device loss (`lost`).
  void certify(uint64_t ordinal, bool lost = false) const noexcept;

  /// @brief Polls, at most once, the lowest fence that covers `ticket` when it is not yet
  /// known to have finished, and records what the answer proves.
  void pollFor(const Ticket& ticket) const;

  /// @brief What a wait for the finished `target` returns: `VK_SUCCESS`, or `VK_ERROR_DEVICE_LOST`
  /// when only a device loss ended it.
  [[nodiscard]] VkResult certifiedResult(uint64_t target) const noexcept;

  /// @brief Waits on `fence`, a row fence that covers `ordinal`, for at most `timeoutNanoseconds`,
  /// and records what a success or a device loss proves.
  VkResult waitFence(const VkFence* fence, uint64_t ordinal, uint64_t timeoutNanoseconds) const;

  /// @brief Whether `handle`, an older recording of its row, answers as the row's current
  /// recording (see `RowRetirement::aliasFloor`).
  [[nodiscard]] bool answersAsCurrentRecording(SubmitHandle handle) const noexcept;

  /// @brief Resets the command buffer of `wrapper`, whose recording will not be submitted, and
  /// frees its row. A handle of the recording then waits for everything submitted before it, and
  /// for the recording that takes the row if the row is acquired again before the next submission.
  void freeUnsubmittedRow(CommandBufferWrapper& wrapper);

  /// @brief Closes the recording of `wrapper` without a submission and frees its row.
  void closeUnsubmitted(CommandBufferWrapper& wrapper);

  /// @brief Whether the `cmdBuf` and `isEncoding` fields of `wrapper` agree with its `state`. For
  /// debug assertions.
  [[nodiscard]] static bool rowAgreesWithWrapper(const CommandBufferWrapper& wrapper) noexcept;

  /// @brief Checks whether the SubmitHandle is recycled. A recycled SubmitHandle is a handle that
  /// has a submit id different from the submit id associated with the same command buffer stored
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

  /// @brief The result of the first failed creation in the constructor (see `valid()`)
  VkResult initResult_ = VK_SUCCESS;

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

  /// @brief The last acquisition ordinal handed out (skipping those whose low 32 bits are 0).
  uint64_t acquisitionCounter_ = 0;
  /// @brief The last submission ordinal handed out, one per `submit()`.
  uint64_t submissionCounter_ = 0;
  /// @brief How many times `submit()` was called, whatever the outcome. Recordings acquired with no
  /// call in between shared one submit id; it is never a submission ordinal.
  uint64_t submitAttempts_ = 0;
  /// @brief Every submission ordinal up to this one is known to have finished (or no longer runs,
  /// after device loss). Advanced by fence answers from const queries too.
  mutable uint64_t retiredThrough_ = 0;
  /// @brief The part of `retiredThrough_` proven by fences that signaled, not by device loss.
  mutable uint64_t succeededThrough_ = 0;

  bool useTimelineSemaphore_ = false;
  bool useTimelineSemaphoreAndSynchronization2_ = false;
  bool chainSubmitsWithSemaphores_ = true;

  /// @brief Set once a Vulkan call returned VK_ERROR_DEVICE_LOST. Updated from const methods
  /// (`isReady()`), and read from any thread by `isDeviceLost()`
  mutable std::atomic<bool> deviceLost_{false};
};

} // namespace igl::vulkan
