/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Metal/MTLBuffer.h> // @donotremove
#import <Metal/MTLTypes.h>
#include <igl/Buffer.h>
#include <igl/metal/Device.h>

namespace igl::metal {

class BufferSynchronizationManager;

class Buffer : public IBuffer {
 public:
  Buffer(id<MTLBuffer> value,
         MTLResourceOptions options,
         BufferDesc::BufferAPIHint requestedApiHints,
         BufferDesc::BufferAPIHint acceptedApiHints,
         BufferDesc::BufferType bufferType,
         size_t sizeInBytes = 0);

  Result upload(const void* data, const BufferRange& range) override;

  void* map(const BufferRange& range, Result* outResult) override;
  void unmap() override;

  [[nodiscard]] BufferDesc::BufferAPIHint requestedApiHints() const noexcept override;
  [[nodiscard]] BufferDesc::BufferAPIHint acceptedApiHints() const noexcept override;
  [[nodiscard]] ResourceStorage storage() const noexcept override;

  [[nodiscard]] size_t getSizeInBytes() const override;
  [[nodiscard]] uint64_t gpuAddress(size_t offset) const override;

  [[nodiscard]] BufferDesc::BufferType getBufferType() const override {
    return bufferType_;
  }

  IGL_INLINE virtual id<MTLBuffer> get() {
    return mtlBuffers_[0];
  }

 protected:
  MTLResourceOptions resourceOptions_;
  std::vector<id<MTLBuffer>> mtlBuffers_;
  BufferDesc::BufferAPIHint requestedApiHints_;
  BufferDesc::BufferAPIHint acceptedApiHints_;
  BufferDesc::BufferType bufferType_;
  size_t sizeInBytes_ = 0;
};

// Manages a ring of buffers.
// At a given time frame, all upload, map and get methods operate on the buffer
// indexed by the current inFlight buffer index
class RingBuffer final : public Buffer {
 public:
  RingBuffer(std::vector<id<MTLBuffer>> ringBuffers,
             MTLResourceOptions options,
             std::shared_ptr<const BufferSynchronizationManager> syncManager,
             BufferDesc::BufferAPIHint requestedApiHints,
             BufferDesc::BufferType bufferType,
             size_t sizeInBytes = 0);

  Result upload(const void* data, const BufferRange& range) override;

  void* map(const BufferRange& range, Result* outResult) override;
  void unmap() override;

  IGL_INLINE id<MTLBuffer> get() override;

 private:
  int lastUpdatedBufferIdx_ = -1;
  std::shared_ptr<const BufferSynchronizationManager> syncManager_;
};

} // namespace igl::metal
