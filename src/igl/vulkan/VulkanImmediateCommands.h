/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
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
  /// `Encoding`; `submit()` makes it `Submitted`, or `Dropped` when it does not submit a recording
  /// that cannot run; `discard()` and a completed `purge()` free it. A `Dropped` row stays out of
  /// circulation until a device loss is certified.
  enum class RowState : uint8_t { Free, Encoding, Submitted, Dropped };
  /// @brief Retirement bookkeeping of a row: the identity of its current (or last) recording and
  /// what proves that recording's work finished.
  struct RowRetirement {
    /// @brief The acquisition ordinal of the row's current or last recording.
    uint64_t acquisition = 0;
    /// @brief `submitAttempts_` when that recording was acquired.
    uint64_t submitAttemptsAtAcquisition = 0;
    /// @brief What the recording's prefix ticket waits for once it is closed: its own submission
    /// ordinal, or the submission counter when it was closed without a submission.
    uint64_t prefix = 0;
    /// @brief `prefix` of the row's previous recording, which bounds every older recording's.
    uint64_t previousPrefix = 0;
    /// @brief The oldest acquisition on this row that the current recording answers for: a
    /// recording that was closed without a submission, with no `submit()` call from its
    /// acquisition until the row was acquired again (the two shared one submit id), waits for the
    /// recording that took the row as well as for its own prefix.
    uint64_t aliasFloor = 0;
    /// @brief Whether the row fence may be polled or waited on: false once it was exported, or
    /// after a submit that failed with a result other than out of memory or device loss.
    bool fenceTrusted = true;
    /// @brief Whether the recording was closed without a submission (discarded, or nothing could
    /// be submitted for lack of memory).
    bool closedUnsubmitted = false;
    /// @brief Whether the row's command buffer or fence could not be reset after its submission
    /// finished: the row is kept out of circulation until `recoverQuarantinedRow()` frees it.
    bool quarantined = false;
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
    /// @brief The acquisition ordinal of the newest recording of this row whose submission was
    /// certified successful (before a device loss stopped certifying successes), or 0. Written
    /// when such a row is recycled, read only for a handle of an older recording of the row (see
    /// `ticketOf()`).
    uint64_t succeededAcquisition_ = 0;
    /// @brief Set when the row of a recording that was not submitted is reused after a device loss
    /// (see `recycleDroppedRows()`): from then on every handle of this row reads as lost, except a
    /// handle certified successful before (see `succeededAcquisition_`), so such a recording never
    /// reads as finished.
    bool outcomesLost_ = false;
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
   * signal-semaphore queues. A recording that cannot run is not submitted (see the private
   * overload): its handle is returned and never completes, as the fence of a submission that never
   * reached the queue.
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
   * submissions up to and including its own. For a recording that was discarded or not submitted,
   * it is all submissions made before it was closed; recycling its command buffer does not make
   * it ready. A recording that is still open is not ready. An empty handle is ready. A recording
   * that was not submitted because it could not run (see `submit()`) is not ready until a device
   * loss is certified.
   * This function never waits and never resets anything: it polls at most one fence, the lowest
   * one whose signal covers the work.
   */
  [[nodiscard]] bool isReady(SubmitHandle handle) const;

  /// @brief If the SubmitHandle is not ready, this function waits until the GPU has finished
  /// everything submitted to this pool up to the handle's recording (see `isReady()`), for at most
  /// `timeoutNanoseconds` of fence waiting. It waits on the lowest fence whose signal covers that
  /// work; when none can be trusted (an exported fence, or the fence of a submission that failed
  /// with an unknown result) it submits a checkpoint, an empty batch with a fence of its own. A
  /// zero timeout only polls. Returns `VK_SUCCESS`, `VK_ERROR_DEVICE_LOST` (which also means the
  /// work no longer runs), `VK_TIMEOUT`, the out-of-memory code of a checkpoint that could not be
  /// submitted, or `VK_ERROR_UNKNOWN` (the handle's recording is still open, or a checkpoint
  /// submission failed otherwise). The result is the handle's own: a finished row whose command
  /// buffer or fence cannot be reset while rows are freed stays out of circulation (see
  /// `recycleCertifiedRow()`). For a recording that was not submitted because it could not run, it
  /// waits on that recording's fence, which nothing signals: `VK_TIMEOUT` once the time is up, or
  /// `VK_ERROR_DEVICE_LOST` after a device loss.
  ///
  /// A handle whose submission was certified successful keeps answering `VK_SUCCESS` once its row
  /// is reused, also after a later device loss. That reports the completion of its work, not that
  /// the device or its resources are usable: `isDeviceLost()` stays true, work not certified
  /// successful stays lost, and nothing more is submitted. Exact for handles fewer than 2^32 - 1
  /// pool acquisitions older than their row's current acquisition (ids wrapping through 0
  /// included). Submit ids skip 0, so a handle exactly that many acquisitions older has the current
  /// recording's id and answers as it, and an older one may answer for a later acquisition: a
  /// proven success may then be reported as lost. Work not certified finished before the first
  /// loss is never reported as success; a submission whose result was unknown counts as certified
  /// once a later fence or checkpoint retires it before the loss.
  VkResult wait(SubmitHandle handle, uint64_t timeoutNanoseconds = UINT64_MAX);

  /// @brief Waits until the GPU has finished every submission made to this pool, with no time
  /// limit. While the work cannot be proven finished (a checkpoint cannot be submitted, or a fence
  /// wait fails), it keeps trying, waiting a little longer each time (see `backoff()`), so it
  /// blocks as long as that lasts; after a device loss that a fence, a checkpoint or that wait
  /// reported, it returns. A recording that is still open is not waited for.
  /// A recording that was not submitted because it could not run is waited for on its fence, which
  /// nothing signals: then it returns only once a device loss is certified (a success with no loss
  /// reported proves nothing, and it keeps backing off), or when that wait fails, which debug
  /// builds assert, as on master.
  /// For a wait with a time limit that reports its result, use `VulkanContext::certifyAll()`.
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
  friend class DescriptorPoolsArena;
  friend class VulkanContext;
  friend class VulkanStagingDevice;
  friend class igl::tests::VulkanImmediateCommandsTest;
  /// @brief What the pools of a context share: they submit to the same queue. A context's staging
  /// pool uses its main pool's, which outlives it; any other pool uses its own (`ownQueueState_`).
  /// This relies on `VulkanContext::initContext()` running once per context and on the context's
  /// main pool never being replaced: either would leave the staging pool using a destroyed state.
  struct QueueState {
    /// @brief Once set, every submission of every pool sharing it is dropped (see `submit()`): as
    /// on a queue where a submission waits on a semaphore that nothing signals, nothing submitted
    /// after it can run.
    bool stopped = false;
    /// @brief How many fence waits, checkpoint submissions and back-offs (counted once per pool)
    /// of the pools sharing it answered that the device is lost: each such answer is fresh
    /// evidence that nothing submitted to the device runs any more (see
    /// `VulkanContext::certifyAll()`). A loss only remembered from another call
    /// (`isDeviceLost()`), or derived from such an answer, is not one.
    uint64_t lossAnswers = 0;
    /// @brief The answers that `VulkanContext::certifyAll()` already applied to both pools.
    uint64_t certifiedLossAnswers = 0;
  };
  /// @brief If `nextCmdBuf` is not null, it is a primary command buffer that the submission runs
  /// after the one of `wrapper`, such as the recording of a reusable command buffer. A recording
  /// that cannot run is not submitted, and its row becomes `Dropped`: its command buffer could not
  /// be ended, two tries to submit it ran out of memory, the queue stopped (see `QueueState`),
  /// or this pool saw a device loss. Its injected semaphores are dropped, the semaphore chain stays
  /// at the last submission and no submission ordinal is taken; its handle is returned, and
  /// `lastSubmitDroppedTheRecording()` is true. As before, a failed end or submit asserts.
  SubmitHandle submit(const CommandBufferWrapper& wrapper,
                      VkSemaphore presentationSemaphore,
                      VkCommandBuffer nextCmdBuf = VK_NULL_HANDLE);

  /// @brief Waits, for at most `remainingNanoseconds` of fence waiting (`UINT64_MAX` waits without
  /// a limit; the time spent is subtracted), until the GPU has finished every submission made to
  /// this pool, including submissions whose fences were exported or whose submit result was unknown
  /// (through a checkpoint), then recycles the finished command buffers.
  /// A recording that is still open is not waited for. Returns `VK_SUCCESS` when everything
  /// submitted has finished, `VK_ERROR_DEVICE_LOST` when the device was lost (nothing submitted
  /// runs any more), `VK_TIMEOUT`, the out-of-memory code of a checkpoint that could not be
  /// submitted, or `VK_ERROR_UNKNOWN` when the work cannot be proven finished. Only `VK_SUCCESS`
  /// and `VK_ERROR_DEVICE_LOST` allow destroying objects the submitted work uses. A finished row
  /// whose command buffer or fence cannot be reset does not change the result: it stays out of
  /// circulation (see `recycleCertifiedRow()`).
  [[nodiscard]] VkResult certifyAll(uint64_t& remainingNanoseconds);

  /// @brief Exports the fence of the handle's submission as a sync file descriptor
  /// (`VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT`) and returns it, or -1 when `vkGetFenceFdKHR()`
  /// fails, is not available, or the handle no longer has a fence. A successful export resets the
  /// fence, so the pool no longer waits on it or polls it: the submission is proven finished by a
  /// later fence or a checkpoint.
  [[nodiscard]] int exportFenceFd(SubmitHandle handle);

  /// @brief Frees, without waiting, every row whose submission finished. First it learns what is
  /// certified: it polls each pending checkpoint and the fence of each trusted submission that no
  /// answer covers yet. Then it resets every submitted row that those answers cover, whatever its
  /// fence, so that a submission whose fence cannot be used is freed in the same call as the later
  /// fence that proves it finished. A row whose command buffer or fence cannot be reset is
  /// quarantined (see `recycleCertifiedRow()`).
  void purge();
  /// @brief Polls the fence of each pending checkpoint once, and records what it certifies.
  void pollPendingCheckpoints() const;

  /// @brief Resets the command buffer and the fence of a row whose submission finished, and frees
  /// the row. When either cannot be reset, the row is quarantined and the result returned.
  VkResult recycleCertifiedRow(CommandBufferWrapper& wrapper);

  /// @brief Frees the row of the oldest submission once it finished, or was lost with the device:
  /// waits, without a time limit, on its own fence, or, for a fence that cannot be trusted, for a
  /// later fence or a checkpoint. Returns whether a row was freed.
  [[nodiscard]] bool recycleOldest();

  /// @brief Frees one quarantined row whose command buffer and fence can now be reset, replacing
  /// the fence (with the same exportability) when it still cannot be reset. Called only when no row
  /// is free, right before a row is acquired. Returns whether a row was freed.
  [[nodiscard]] bool recoverQuarantinedRow();
  /// @brief One try of an exhausted `acquire()` to free a row, after `purge()` freed none: first,
  /// without waiting, a quarantined row that can now be reset, else, once a device loss is
  /// certified, the rows of recordings that were not submitted, else the oldest submission's row,
  /// waiting for it without a time limit. Returns whether a row was freed.
  [[nodiscard]] bool freeRow();
  /// @brief Waits a little before `acquire()`, `waitAll()` or the destructor tries again, when no
  /// row can be freed or the submitted work cannot be proven finished: on `backoffFence_`, for 1,
  /// 2, 4, 8, then 16 ms (`attempt` counts the tries of one call). VK_ERROR_DEVICE_LOST from that
  /// fence, which nothing signals, ends every submission of the pool, as lost; so does VK_SUCCESS
  /// once a device loss was already reported (`isDeviceLost()`). Without such a report VK_SUCCESS
  /// proves nothing and is logged once. When the fence wait fails or answers VK_SUCCESS that proves
  /// nothing, it waits that long on the host instead (`hostBackoffDelay()`). Returns the result of
  /// the fence wait.
  VkResult backoff(uint32_t& attempt);
  /// @brief Creates `backoffFence_` unless it exists. Returns the result of its creation.
  VkResult ensureBackoffFence();
  /// @brief A bounded host back-off delay, used only when the back-off fence wait itself fails or
  /// answers at once what proves nothing: a sleep of `nanoseconds`.
  void hostBackoffDelay(uint64_t nanoseconds);
  /// @brief Takes `result`, the answer of a wait on fences that nothing signals on a working device
  /// (`backoffFence_`, the fence of a recording that was not submitted): VK_ERROR_DEVICE_LOST, or
  /// VK_SUCCESS once a device loss was already reported (`isDeviceLost()`), ends every submission
  /// of the pool, as lost (`certifyLoss()`). Without such a report VK_SUCCESS proves nothing and is
  /// logged once. Returns whether the loss was certified.
  bool certifyLossFromUnsignaledFence(VkResult result) const noexcept;
  /// @brief Moves `wrapper` to `state` and updates its `cmdBuf` and `isEncoding` fields that mirror
  /// it, so they cannot drift. The only writer of the three.
  void setRowState(CommandBufferWrapper& wrapper, RowState state) noexcept;

  /// @brief What a handle's prefix ticket needs: nothing (a null handle, or an older recording of
  /// its row whose submission was certified successful), the close of a recording that is still
  /// open, or a submission ordinal that `retiredThrough_` must reach.
  struct Ticket {
    /// @brief `Dropped`: a recording that was not submitted, which never finishes and retires with
    /// its prefix. `Lost`: a handle of a row whose outcomes were lost with the device.
    enum class Kind : uint8_t { Settled, Open, Prefix, Dropped, Lost } kind = Kind::Settled;
    uint64_t prefix = 0;
    /// @brief The row whose current recording the handle names, when it does.
    const CommandBufferWrapper* IGL_NULLABLE own = nullptr;
  };
  [[nodiscard]] Ticket ticketOf(SubmitHandle handle) const noexcept;
  [[nodiscard]] Ticket ticketOfRow(const CommandBufferWrapper& wrapper) const noexcept;

  /// @brief The lowest fence whose signal certifies submission ordinal `target`: the fence of a
  /// trusted submitted row or of a pending checkpoint. `own` is the handle's row, checked first.
  /// `ordinal` receives what the fence certifies.
  [[nodiscard]] const VkFence* IGL_NULLABLE
  lowestSourceFor(uint64_t target,
                  const CommandBufferWrapper* IGL_NULLABLE own,
                  uint64_t& ordinal) const noexcept;

  /// @brief Records that every submission up to `ordinal` finished, or no longer runs after a
  /// device loss (`lost`). After a certified loss a success answer certifies no success: Vulkan
  /// allows a lost device to answer success, so it cannot make lost work succeed. Before a success,
  /// the losses that a pool sharing the queue state answered are applied (`applyLossAnswers()`).
  void certify(uint64_t ordinal, bool lost = false) const noexcept;
  /// @brief `certify()` without applying the losses answered by a pool sharing the queue state.
  void recordCertification(uint64_t ordinal, bool lost) const noexcept;
  /// @brief Marks every pending or abandoned checkpoint signaled, once the queue is idle or the
  /// device is lost: no checkpoint fence is pending any more.
  void markCheckpointsIdle() const noexcept;

  /// @brief Records a fresh answer that the device is lost (from a fence wait or poll, a checkpoint
  /// submission or the back-off fence): nothing submitted runs any more, so every submission of the
  /// pool ends, as lost (see `certifyQueueIdle()`). The first such answer of the pool is counted
  /// for the pools sharing its queue state (see `noteLossAnswer()`).
  void certifyLoss() const noexcept;
  /// @brief Counts a device-loss answer of this pool in its queue state (see
  /// `QueueState::lossAnswers`).
  void noteLossAnswer() const noexcept;
  /// @brief Ends every submission of this pool, as lost, when a pool sharing its queue state
  /// counted a device-loss answer that this pool has not applied yet; returns whether it did.
  /// Called before a submission, a success is certified, or a back-off, so that the other pool's
  /// answer stops submissions, freezes successes and ends a wait without one of its own.
  bool applyLossAnswers() const noexcept;

  /// @brief Polls, at most once, the lowest fence that covers `ticket` when it is not yet
  /// certified, and records what the answer certifies.
  void pollFor(const Ticket& ticket) const;

  /// @brief What a wait for the certified `target` returns: `VK_SUCCESS`, or `VK_ERROR_DEVICE_LOST`
  /// when only a device loss certified it.
  [[nodiscard]] VkResult certifiedResult(uint64_t target) const noexcept;

  /// @brief Waits until `retiredThrough_` reaches `target`, on the lowest fence that covers it or,
  /// when none does, on a checkpoint. Returns `VK_SUCCESS`, `VK_ERROR_DEVICE_LOST` (certified too),
  /// `VK_TIMEOUT` (also when no time is left to wait for a checkpoint, which is then not
  /// submitted) or the error of a fence wait or of a checkpoint submission. `own` is the row of the
  /// handle being waited for, if any. The time spent is subtracted from `remainingNanoseconds`.
  VkResult certifyThrough(uint64_t target,
                          uint64_t& remainingNanoseconds,
                          const CommandBufferWrapper* IGL_NULLABLE own);

  /// @brief Submits a checkpoint covering every submission made so far. It reuses a checkpoint
  /// whose fence was seen signaled, creates one, or waits for the oldest pending one first.
  VkResult submitCheckpoint(uint64_t& remainingNanoseconds);

  /// @brief Waits on `fence`, a row or checkpoint fence that certifies `ordinal`, within
  /// `remainingNanoseconds`, and records what a success or a device loss certifies.
  VkResult waitFence(const VkFence* fence, uint64_t ordinal, uint64_t& remainingNanoseconds) const;

  /// @brief Records that `fence`, which certifies `ordinal`, answered success or device loss.
  void noteFenceSignaled(const VkFence* fence, uint64_t ordinal, bool lost) const noexcept;

  /// @brief Whether `handle`, an older recording of its row, answers as the row's current
  /// recording (see `RowRetirement::aliasFloor`).
  [[nodiscard]] bool answersAsCurrentRecording(SubmitHandle handle) const noexcept;

  /// @brief Whether `handle`, an older recording of the row `buf`, is at or before the row's
  /// newest recording certified successful (`CommandBufferWrapper::succeededAcquisition_`). The
  /// handle's acquisition is taken as the newest one at or before the row's current acquisition
  /// with its submit id: exact for handles fewer than 2^32 - 1 acquisitions older (see `wait()`).
  [[nodiscard]] static bool succeededBefore(const CommandBufferWrapper& buf,
                                            SubmitHandle handle) noexcept;

  /// @brief Resets the command buffer of `wrapper`, whose recording will not be submitted, and
  /// frees its row. A handle of the recording then waits for everything submitted before it, and
  /// for the recording that takes the row if the row is acquired again before the next submission.
  void freeUnsubmittedRow(CommandBufferWrapper& wrapper);

  /// @brief Closes the recording of `wrapper` without a submission and frees its row.
  void closeUnsubmitted(CommandBufferWrapper& wrapper);

  /// @brief Submits one batch with the given semaphores and the row's fence: the row's command
  /// buffer, and `nextCmdBuf` (if not null) after it.
  VkResult queueSubmit(const CommandBufferWrapper& wrapper,
                       VkCommandBuffer nextCmdBuf,
                       const VkSemaphoreSubmitInfo* waitSemaphores,
                       uint32_t numWaitSemaphores,
                       const VkSemaphoreSubmitInfo* signalSemaphores,
                       uint32_t numSignalSemaphores) const;

  /// @brief Whether the last `submit()` did not submit its recording, which never runs (see
  /// `submit()`). Its timestamps are never written, and its handle never completes.
  [[nodiscard]] bool lastSubmitDroppedTheRecording() const noexcept {
    return lastSubmitDroppedTheRecording_;
  }

  /// @brief The last submission the queue accepted, or may have: what internal bookkeeping (slots
  /// of reusable command buffers, ring buffers, staging blocks) waits for after a recording that
  /// was not submitted, whose own handle never completes. It bounds every earlier submission.
  [[nodiscard]] SubmitHandle lastAcceptedSubmitHandle() const noexcept {
    return lastAcceptedSubmitHandle_;
  }

  /// @brief Whether the objects the work of a handle uses can be destroyed: everything submitted up
  /// to its recording finished or was lost with the device. Unlike `isReady()`, it is true for a
  /// recording that was not submitted once the work submitted before it retired, as it never runs.
  /// For deferred destruction only.
  [[nodiscard]] bool isRetired(SubmitHandle handle) const;

  /// @brief Waits, for at most `timeoutNanoseconds` of fence waiting, until `isRetired(handle)`.
  /// Returns `VK_SUCCESS` once it is, or why that could not be proven: `VK_TIMEOUT`, or the error
  /// of a checkpoint or a fence wait (`VK_ERROR_UNKNOWN` for a recording that is still open).
  VkResult waitRetired(SubmitHandle handle, uint64_t timeoutNanoseconds);

  /// @brief Closes the recording of `wrapper` without submitting it (see `submit()`): its row
  /// becomes `Dropped`. `poisonChain`: the batch would have signaled the chain semaphore, which the
  /// next submission of this pool waits on, so that next submission stops the queue.
  SubmitHandle dropRecording(CommandBufferWrapper& wrapper, bool poisonChain);

  /// @brief Answers `wait()` for a recording that was not submitted.
  VkResult waitDropped(const CommandBufferWrapper& wrapper, uint64_t timeoutNanoseconds) const;

  /// @brief Once a device loss is certified, frees the rows of recordings that were not submitted,
  /// and marks them so that every handle of these rows reads as lost. Returns whether a row was
  /// freed.
  [[nodiscard]] bool recycleDroppedRows();

  /// @brief Waits, retrying, until everything submitted is proven finished or lost (see
  /// `waitAll()`), without waiting for recordings that were not submitted.
  void drainSubmitted();

  /// @brief Whether the last `submit()` failed with a result that leaves its batch's fate unknown:
  /// its recording may or may not have run, even once a later fence or a checkpoint proves that the
  /// submission retired.
  [[nodiscard]] bool lastSubmitOutcomeUnknown() const noexcept {
    return lastSubmitOutcomeUnknown_;
  }

  /// @brief Records that the queue was idle: everything submitted so far finished, or no longer
  /// runs after a device loss (`lost`).
  void certifyQueueIdle(bool lost) const noexcept;

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
  /// @brief The last submission ordinal handed out. A submission takes one when it reaches the
  /// queue, or may have (device loss, unknown result).
  uint64_t submissionCounter_ = 0;
  /// @brief How many times `submit()` was called, whatever the outcome. Recordings acquired with no
  /// call in between shared one submit id; it is never a submission ordinal.
  uint64_t submitAttempts_ = 0;
  /// @brief Every submission ordinal up to this one is known to have finished (or no longer runs,
  /// after device loss). Advanced by fence answers from const queries too.
  mutable uint64_t retiredThrough_ = 0;
  /// @brief The part of `retiredThrough_` certified by fences that signaled, not by device loss. It
  /// no longer grows once a loss is certified (see `certify()`).
  mutable uint64_t succeededThrough_ = 0;
  /// @brief The device-loss answers of the shared queue state that this pool applied (see
  /// `applyLossAnswers()`).
  mutable uint64_t appliedLossAnswers_ = 0;
  /// @brief Set once a fence, a checkpoint, the back-off fence, a queue idle, or the fence of a
  /// recording that was not submitted reported a device loss: nothing submitted runs any more.
  mutable bool lossCertified_ = false;
  /// @brief See `lastSubmitDroppedTheRecording()`.
  bool lastSubmitDroppedTheRecording_ = false;
  /// @brief See `lastAcceptedSubmitHandle()`.
  SubmitHandle lastAcceptedSubmitHandle_ = {};
  /// @brief A recording that was not submitted would have signaled the chain semaphore: the next
  /// submission of this pool stops the queue, where it would have waited on that semaphore forever.
  bool chainPoisoned_ = false;
  QueueState ownQueueState_;
  /// @brief The state this pool shares with the other pool of its context, else `ownQueueState_`.
  QueueState* queueState_ = &ownQueueState_;
  /// @brief See `lastSubmitOutcomeUnknown()`.
  bool lastSubmitOutcomeUnknown_ = false;

  /// @brief A checkpoint: an empty batch whose fence certifies every submission made before it.
  struct Checkpoint {
    /// @brief `Abandoned`: its submission failed with an unknown result. `Quarantined`: its fence
    /// could not be reset. Neither fence is used again.
    enum class State : uint8_t { Unused, Pending, Signaled, Abandoned, Quarantined };
    VulkanFence fence;
    /// @brief The submission counter when the checkpoint was submitted.
    uint64_t covers = 0;
    State state = State::Unused;
  };
  static constexpr uint32_t kMaxCheckpoints = 2;
  /// @brief Gives a checkpoint that signaled a new fence, and makes it `Unused`. Returns false when
  /// no fence can be created.
  [[nodiscard]] bool replaceCheckpointFence(Checkpoint& checkpoint);
  /// @brief Created on first use. `mutable` so that const queries can record a signaled fence.
  mutable std::vector<Checkpoint> checkpoints_;

  bool useTimelineSemaphore_ = false;
  bool useTimelineSemaphoreAndSynchronization2_ = false;
  bool chainSubmitsWithSemaphores_ = true;

  /// @brief Set once a Vulkan call returned VK_ERROR_DEVICE_LOST. Updated from const methods
  /// (`isReady()`), and read from any thread by `isDeviceLost()`
  mutable std::atomic<bool> deviceLost_{false};

  /// @brief A fence that is never submitted, reset, exported or handed out, for `backoff()`. On a
  /// working device nothing signals it, so a wait on it ends only at its time limit; once the
  /// device is lost, a wait on it returns VK_SUCCESS or VK_ERROR_DEVICE_LOST in finite time.
  /// Created by the first back-off (see `ensureBackoffFence()`).
  std::optional<VulkanFence> backoffFence_;
  /// @brief For `hostBackoffDelay()`: never notified, so a wait on it is a sleep.
  std::mutex hostBackoffMutex_;
  std::condition_variable hostBackoffWakeup_;
};

} // namespace igl::vulkan
