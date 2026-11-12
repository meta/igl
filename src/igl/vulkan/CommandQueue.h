/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/CommandQueue.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>

namespace igl::vulkan {

class CommandBuffer;

/** @brief Implements the igl::ICommandQueue interface for Vulkan. Currently, this class only
 * supports one command buffer active at a time, tracked by an internal flag set to true in
 * `createCommandBuffer()` and reset in `endCommandBuffer()` (automatically called from `submit()`).
 */
class CommandQueue final : public ICommandQueue {
 public:
  CommandQueue(Device& device, const CommandQueueDesc& desc);

  ~CommandQueue() override {
    IGL_DEBUG_ASSERT(*numBuffersLeftToSubmit_ == 0);
  }

  CommandQueue(const CommandQueue&) = delete;
  CommandQueue& operator=(const CommandQueue&) = delete;
  CommandQueue(CommandQueue&&) = delete;
  CommandQueue& operator=(CommandQueue&&) = delete;

  /// @brief Create a new command buffer. Sets the internal flag that tracks an active command
  /// buffer has been created.
  std::shared_ptr<ICommandBuffer> createCommandBuffer(const CommandBufferDesc& desc,
                                                      Result* IGL_NULLABLE outResult) override;

  /// @brief Submits the `commandBuffer` for execution on the GPU. A reusable command buffer (see
  /// PlatformDevice::createReusableCommandBuffer()) runs the recording of its current slot after a
  /// command buffer acquired for the submission, which carries the fence and the semaphores of the
  /// submission. It returns 0, without running, while a render pass of the recording is being
  /// encoded, if the recording of the slot failed, or if the last run of the slot does not complete
  /// within VulkanContextConfig::fenceTimeoutNanoseconds. The draw statistics count the draws of
  /// the slot of a reusable command buffer, and of the slots that `commandBuffer` executes when the
  /// submission runs.
  /// @param cmdBuffer The command buffer to be submitted.
  /// @param endOfFrame Not used
  SubmitHandle submit(const ICommandBuffer& cmdBuffer, bool endOfFrame = false) override;

  /** @brief Ends the current command buffer and resets the internal flag tracking an active command
   * buffer. Determines if an image should be presented by (1) checking if this instance belongs to
   * a graphics queue, (2) the context has a swapchain object, (3) the command buffer is from a
   * swapchain (please refer to CommandBuffer::present), and (4) the present parameter is true. If
   * so, this function waits for the swapchain semaphore before submitting the command buffer for
   * execution. After the command buffer is submitted, this function calls VulkanContext::present()
   * if an image should be presented. Finally, it signals the context to process deferred tasks (for
   * more details about deferred tasks, please refer to the igl::vulkan::VulkanContext class).
   */
  SubmitHandle endCommandBuffer(VulkanContext& ctx, CommandBuffer* cmdBuffer, bool present);

 private:
  Device& device_;

  /// @brief Counter indicating whether or not there is an active command buffer. C
  /// Shared with the command buffers of the queue: one that executes slots of a reusable command
  /// buffer leaves the count when it is destroyed without being submitted.
  std::shared_ptr<int> numBuffersLeftToSubmit_ = std::make_shared<int>(0);
};

} // namespace igl::vulkan
