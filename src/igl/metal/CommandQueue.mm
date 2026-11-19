/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/metal/CommandQueue.h>

#include <Foundation/Foundation.h>
#include <utility>
#include <igl/Macros.h>
#include <igl/metal/BufferSynchronizationManager.h>
#include <igl/metal/CommandBuffer.h>
#include <igl/metal/DeviceStatistics.h>
#include <igl/metal/Timer.h>
#include <igl/metal/TimestampQueries.h>

// @brief Number of command buffers to be automatically captured for GPU debugging. Zero (0)
// means no command buffers will be recorded and the capture code is deactivated.
constexpr uint32_t kIGLMetalNumberCommandBuffersToCapture = 0;
// @brief Beginning command buffer to be captured
constexpr uint32_t kIGLMetalBeginCommandBufferToCapture = 0;
// @brief Ending command buffer. Only command buffers within [kIGLMetalBeginCommandBufferToCapture,
// kIGLMetalEndCommandBufferToCapture) will be captured.
constexpr uint32_t kIGLMetalEndCommandBufferToCapture =
    kIGLMetalBeginCommandBufferToCapture + kIGLMetalNumberCommandBuffersToCapture;

namespace igl::metal {

CommandQueue::CommandQueue(Device& device,
                           id<MTLCommandQueue> value,
                           const std::shared_ptr<BufferSynchronizationManager>& syncManager,
                           DeviceStatistics& deviceStatistics) noexcept :
  value_(value),
  bufferSyncManager_(syncManager),
  deviceStatistics_(deviceStatistics),
  device_(device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if constexpr (kIGLMetalNumberCommandBuffersToCapture > 0 &&
                kIGLMetalBeginCommandBufferToCapture == 0) {
    startCapture(value_);
  }
}

std::shared_ptr<ICommandBuffer> CommandQueue::createCommandBuffer(const CommandBufferDesc& desc,
                                                                  Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // If the previous frame ended, perform the triple-buffering wait here -- at the start of the
  // frame, before anything is recorded into the next in-flight buffer slot -- rather than at
  // frame end. The flag keeps this to one wait per frame even when a frame records multiple
  // command buffers; blit/upload command buffers are created directly off the MTLCommandQueue
  // and never reach this path.
  if (frameBoundaryPending_) {
    frameBoundaryPending_ = false;
    bufferSyncManager_->manageEndOfFrameSync();
  }

  id<MTLCommandBuffer> metalObject = [value_ commandBuffer];
  metalObject.label = [NSString stringWithUTF8String:desc.debugName.c_str()];
  auto resource = std::make_shared<CommandBuffer>(device_, metalObject, desc);
  Result::setOk(outResult);
  return resource;
}

SubmitHandle CommandQueue::submit(const ICommandBuffer& commandBuffer, bool endOfFrame) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_SUBMIT);
  incrementDrawCount(commandBuffer.getCurrentDrawCount());
  deviceStatistics_.incrementDrawCount(commandBuffer.getCurrentDrawCount());

  if (endOfFrame) {
    bufferSyncManager_->markCommandBufferAsEndOfFrame(commandBuffer);
  }

  const auto& metalCommandBuffer = static_cast<const CommandBuffer&>(commandBuffer);
  std::shared_ptr<ITimer> timer = metalCommandBuffer.desc.timer;
  if (timer) {
    [metalCommandBuffer.get() addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      CFTimeInterval gpuDuration = cb.GPUEndTime - cb.GPUStartTime;
      static_cast<Timer&>(*timer).executionTime_ =
          static_cast<uint64_t>(gpuDuration * 1e9); // Convert to nanoseconds
    }];
  }

  if (@available(macOS 10.15, iOS 14.0, *)) {
    std::shared_ptr<ITimestampQueries> tsQueries = metalCommandBuffer.desc.timestampQueries;
    if (tsQueries) {
      auto metalTsQueries = std::static_pointer_cast<TimestampQueries>(tsQueries);
      id<MTLCounterSampleBuffer> csb = metalTsQueries->sampleBuffer_;
      uint64_t gen = metalTsQueries->generation_.load(std::memory_order_seq_cst);
      [metalCommandBuffer.get() addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
        @try {
          if (metalTsQueries->generation_.load(std::memory_order_acquire) == gen) {
            metalTsQueries->resolveTimestamps(csb);
          }
        } @catch (NSException*) {
          // GPU reset or counter sample buffer invalidated -- silently skip.
        }
      }];
    }
  }

  [metalCommandBuffer.get() commit];

  if (endOfFrame) {
    // Defer the CPU wait to the start of the next frame (createCommandBuffer) instead of
    // stalling here at frame end.
    frameBoundaryPending_ = true;
  }

  if constexpr (kIGLMetalNumberCommandBuffersToCapture > 0) {
    static uint32_t currentCommandBuffer = 0;
    if ((currentCommandBuffer + 1) == kIGLMetalBeginCommandBufferToCapture) {
      // Start capturing one command buffer earlier
      startCapture(value_);
    } else if (std::cmp_greater_equal(currentCommandBuffer, kIGLMetalEndCommandBufferToCapture)) {
      stopCapture();
    }
    ++currentCommandBuffer;
  }

  return SubmitHandle{};
}

void CommandQueue::startCapture(id<MTLCommandQueue> queue) {
  IGL_PROFILER_FUNCTION();
  MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
  MTLCaptureDescriptor* captureDescriptor = [[MTLCaptureDescriptor alloc] init];
  captureDescriptor.captureObject = queue;

  NSError* error;
  if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
    NSLog(@"Failed to start capture, error %@", error);
  }
}

void CommandQueue::stopCapture() {
  IGL_PROFILER_FUNCTION();
  MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
  [captureManager stopCapture];
}

} // namespace igl::metal
