/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/metal/Buffer.h>

#include <igl/IGLSafeC.h>
#include <igl/Macros.h>
#include <igl/metal/BufferSynchronizationManager.h>

namespace {
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
igl::Result upload(const std::vector<id<MTLBuffer>>& buffers,
                   size_t bufferIdx,
                   const void* data,
                   const igl::BufferRange& range,
                   MTLResourceOptions mtlResourceOptions,
                   igl::BufferDesc::BufferAPIHint bufferAcceptedApiHints) {
  IGL_DEBUG_ASSERT(bufferIdx < buffers.size());
  const auto& buffer = buffers[bufferIdx];
  auto length = [buffer length];
  if (!IGL_DEBUG_VERIFY(range.offset + range.size <= length)) {
    return igl::Result(igl::Result::Code::ArgumentOutOfRange);
  }

  if (data == nullptr) {
    if (!(bufferAcceptedApiHints & igl::BufferDesc::BufferAPIHintBits::NoCopy)) {
      return igl::Result(igl::Result::Code::ArgumentInvalid);
    }
  } else {
    void* contents = [buffer contents];
    checked_memcpy_offset(contents, length, range.offset, data, range.size);
  }

#if IGL_PLATFORM_MACOSX
  if ((mtlResourceOptions & MTLResourceStorageModeMask) == MTLResourceStorageModeManaged) {
    [buffer didModifyRange:NSMakeRange(range.offset, range.size)];
  }
#else
  (void)mtlResourceOptions; // silence unused member warning
#endif
  return igl::Result();
}

void* map(const std::vector<id<MTLBuffer>>& buffers,
          size_t bufferIdx,
          const igl::BufferRange& range,
          igl::Result* outResult,
          MTLResourceOptions resourceOptions) {
  if ((resourceOptions & MTLResourceStorageModeMask) == MTLResourceStorageModePrivate) {
    igl::Result::setResult(outResult,
                           igl::Result::Code::InvalidOperation,
                           "Cannot map() buffer with private storage mode");
    return nullptr;
  }

  IGL_DEBUG_ASSERT(bufferIdx < buffers.size());
  const auto& buffer = buffers[bufferIdx];
  if ([buffer length] < (range.size + range.offset)) {
    igl::Result::setResult(outResult,
                           igl::Result::Code::ArgumentOutOfRange,
                           "map() size + offset must be less than buffer size");
    return nullptr;
  }

  igl::Result::setOk(outResult);
  return static_cast<uint8_t*>(buffer.contents) + range.offset;
}

igl::Result copyFromPreviousBufferInstance(std::vector<id<MTLBuffer>>& buffers,
                                           size_t bufferIdx,
                                           MTLResourceOptions resourceOptions,
                                           igl::BufferDesc::BufferAPIHint acceptedApiHints) {
  if (buffers.size() <= 1) {
    return igl::Result();
  }

  const size_t prevIdx = bufferIdx == 0 ? buffers.size() - 1 : bufferIdx - 1;
  IGL_DEBUG_ASSERT([buffers[bufferIdx] length] == [buffers[prevIdx] length]);

  auto length = [buffers[bufferIdx] length];
  auto* srcContents = [buffers[prevIdx] contents];
  return ::upload(buffers,
                  bufferIdx,
                  srcContents,
                  igl::BufferRange(length, 0),
                  resourceOptions,
                  acceptedApiHints);
}
} // namespace

namespace igl::metal {
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Buffer::Buffer(id<MTLBuffer> value,
               MTLResourceOptions options,
               BufferDesc::BufferAPIHint requestedApiHints,
               BufferDesc::BufferAPIHint acceptedApiHints,
               BufferDesc::BufferType bufferType,
               size_t sizeInBytes) :
  resourceOptions_(options),
  requestedApiHints_(requestedApiHints),
  acceptedApiHints_(acceptedApiHints),
  bufferType_(bufferType),
  sizeInBytes_(sizeInBytes) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  mtlBuffers_.push_back(value);
}

Result Buffer::upload(const void* data, const BufferRange& range) {
  IGL_PROFILER_FUNCTION();
  return ::upload(mtlBuffers_, 0, data, range, resourceOptions_, acceptedApiHints_);
}

void* Buffer::map(const BufferRange& range, Result* outResult) {
  IGL_PROFILER_FUNCTION();
  return ::map(mtlBuffers_, 0, range, outResult, resourceOptions_);
}

void Buffer::unmap() {}

BufferDesc::BufferAPIHint Buffer::requestedApiHints() const noexcept {
  return requestedApiHints_;
}

BufferDesc::BufferAPIHint Buffer::acceptedApiHints() const noexcept {
  return acceptedApiHints_;
}

ResourceStorage Buffer::storage() const noexcept {
#if IGL_PLATFORM_MACOSX
  if ((resourceOptions_ & MTLResourceStorageModeMask) == MTLResourceStorageModeManaged) {
    return ResourceStorage::Managed;
  }
#endif // IGL_PLATFORM_MACOSX
  if ((resourceOptions_ & MTLResourceStorageModeMask) == MTLResourceStorageModePrivate) {
    return ResourceStorage::Private;
  }
  if ((resourceOptions_ & MTLResourceStorageModeMask) == MTLResourceStorageModeShared) {
    return ResourceStorage::Shared;
  }
  if ((resourceOptions_ & MTLResourceStorageModeMask) == MTLResourceStorageModeMemoryless) {
    return ResourceStorage::Memoryless;
  }
  return ResourceStorage::Invalid;
}

size_t Buffer::getSizeInBytes() const {
  return sizeInBytes_ != 0 ? sizeInBytes_ : [mtlBuffers_[0] length];
}

uint64_t Buffer::gpuAddress(size_t /*offset*/) const {
  // TODO: implement via gpuResourceID
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  return 0;
}

RingBuffer::RingBuffer(std::vector<id<MTLBuffer>> ringBuffers,
                       MTLResourceOptions options,
                       std::shared_ptr<const BufferSynchronizationManager> syncManager,
                       BufferDesc::BufferAPIHint requestedApiHints,
                       BufferDesc::BufferType bufferType,
                       size_t sizeInBytes) :
  Buffer(nil,
         options,
         requestedApiHints,
         BufferDesc::BufferAPIHintBits::Ring,
         bufferType,
         sizeInBytes),
  syncManager_(std::move(syncManager)) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  mtlBuffers_ = std::move(ringBuffers);
}

/* In some scenarios, only part of the buffer could be upload, before the buffer is submitted;
 * To handle this case, we copy the previous instance of the buffer to this one
 */
Result RingBuffer::upload(const void* data, const BufferRange& range) {
  IGL_PROFILER_FUNCTION();
  auto bufferIdx = syncManager_->getCurrentInFlightBufferIndex();

  if (lastUpdatedBufferIdx_ != bufferIdx) {
    auto length = [mtlBuffers_[bufferIdx] length];
    if (range.offset != 0 || range.size != length) {
      // partial upload
      // Copy from the previous buffer first
      auto result = copyFromPreviousBufferInstance(
          mtlBuffers_, bufferIdx, resourceOptions_, acceptedApiHints_);
      if (!result.isOk()) {
        return result;
      }
    }
  }
  auto result = ::upload(mtlBuffers_, bufferIdx, data, range, resourceOptions_, acceptedApiHints_);
  if (result.isOk()) {
    lastUpdatedBufferIdx_ = bufferIdx;
  }
  return result;
}

void* RingBuffer::map(const BufferRange& /* unused */, Result* /* unused */) {
  IGL_DEBUG_ABORT("map() operation not supported for RingBuffer");
  return nullptr;
}

void RingBuffer::unmap() {
  IGL_DEBUG_ABORT("unmap() operation not supported for RingBuffer");
}

id<MTLBuffer> RingBuffer::get() {
  IGL_PROFILER_FUNCTION();
  auto bufferIdx = syncManager_->getCurrentInFlightBufferIndex();
  IGL_DEBUG_ASSERT(bufferIdx < mtlBuffers_.size());
  if (bufferIdx != lastUpdatedBufferIdx_) {
    // client hasn't updated the buffer at this idx; Update from the previous buffer instance
    auto result =
        copyFromPreviousBufferInstance(mtlBuffers_, bufferIdx, resourceOptions_, acceptedApiHints_);
    if (!result.isOk()) {
      IGL_DEBUG_ABORT("Failed to copy buffer");
      return nullptr;
    }

    lastUpdatedBufferIdx_ = bufferIdx;
  }
  return mtlBuffers_[bufferIdx];
}

} // namespace igl::metal
