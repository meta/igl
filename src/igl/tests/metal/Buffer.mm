/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/metal/Buffer.h>

#include "../util/Common.h"
#include "../util/TestDevice.h"

#import <Metal/MTLBlitCommandEncoder.h>
#import <Metal/MTLCommandBuffer.h>
#import <Metal/MTLCommandQueue.h>
#import <Metal/MTLDevice.h>
#include <cstring>
#include <igl/IGL.h>
#include <igl/metal/Device.h>

namespace igl::tests {

//
// MetalBufferTest
//
// This test covers igl::metal::Buffer.
// Tests creation, upload, mapping, and storage modes of Metal buffers.
//
class MetalBufferTest : public ::testing::Test {
 public:
  MetalBufferTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(device_, cmdQueue_);
    ASSERT_NE(device_, nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> device_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

//
// CreateBufferWithData
//
// Test creating a buffer with initial data, verify size and content.
//
TEST_F(MetalBufferTest, CreateBufferWithData) {
  const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const size_t dataSize = sizeof(data);

  const BufferDesc desc{.type = BufferDesc::BufferTypeBits::Uniform,
                        .data = data,
                        .length = dataSize,
                        .storage = ResourceStorage::Shared,
                        .debugName = "testBufferWithData"};

  Result res;
  auto buffer = device_->createBuffer(desc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(buffer, nullptr);

  ASSERT_EQ(buffer->getSizeInBytes(), dataSize);

  // Map and verify data
  Result mapRes;
  auto* mapped = static_cast<float*>(buffer->map(BufferRange(dataSize, 0), &mapRes));
  ASSERT_TRUE(mapRes.isOk()) << mapRes.message;
  ASSERT_NE(mapped, nullptr);
  ASSERT_EQ(mapped[0], 1.0f);
  ASSERT_EQ(mapped[1], 2.0f);
  ASSERT_EQ(mapped[2], 3.0f);
  ASSERT_EQ(mapped[3], 4.0f);
  buffer->unmap();
}

//
// CreateBufferEmpty
//
// Test creating a buffer without initial data.
//
TEST_F(MetalBufferTest, CreateBufferEmpty) {
  const size_t bufferSize = 256;

  const BufferDesc desc{.type = BufferDesc::BufferTypeBits::Uniform,
                        .length = bufferSize,
                        .storage = ResourceStorage::Shared,
                        .debugName = "testBufferEmpty"};

  Result res;
  auto buffer = device_->createBuffer(desc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->getSizeInBytes(), bufferSize);
}

//
// BufferUploadAndVerify
//
// Test uploading data to a buffer, then mapping and verifying the data matches.
//
TEST_F(MetalBufferTest, BufferUploadAndVerify) {
  const size_t bufferSize = 4 * sizeof(float);

  const BufferDesc desc{.type = BufferDesc::BufferTypeBits::Uniform,
                        .length = bufferSize,
                        .storage = ResourceStorage::Shared,
                        .debugName = "testBufferUpload"};

  Result res;
  auto buffer = device_->createBuffer(desc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(buffer, nullptr);

  const float uploadData[] = {10.0f, 20.0f, 30.0f, 40.0f};
  res = buffer->upload(uploadData, BufferRange(sizeof(uploadData), 0));
  ASSERT_TRUE(res.isOk()) << res.message;

  Result mapRes;
  auto* mapped = static_cast<float*>(buffer->map(BufferRange(bufferSize, 0), &mapRes));
  ASSERT_TRUE(mapRes.isOk()) << mapRes.message;
  ASSERT_NE(mapped, nullptr);
  ASSERT_EQ(mapped[0], 10.0f);
  ASSERT_EQ(mapped[1], 20.0f);
  ASSERT_EQ(mapped[2], 30.0f);
  ASSERT_EQ(mapped[3], 40.0f);
  buffer->unmap();
}

//
// BufferMapUnmap
//
// Test mapping, writing, unmapping, then re-mapping to verify written data.
//
TEST_F(MetalBufferTest, BufferMapUnmap) {
  const size_t bufferSize = 4 * sizeof(float);

  const BufferDesc desc{.type = BufferDesc::BufferTypeBits::Uniform,
                        .length = bufferSize,
                        .storage = ResourceStorage::Shared,
                        .debugName = "testMapUnmap"};

  Result res;
  auto buffer = device_->createBuffer(desc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(buffer, nullptr);

  // Map and write data
  Result mapRes;
  auto* mapped = static_cast<float*>(buffer->map(BufferRange(bufferSize, 0), &mapRes));
  ASSERT_TRUE(mapRes.isOk()) << mapRes.message;
  ASSERT_NE(mapped, nullptr);
  mapped[0] = 100.0f;
  mapped[1] = 200.0f;
  mapped[2] = 300.0f;
  mapped[3] = 400.0f;
  buffer->unmap();

  // Re-map and verify
  auto* mapped2 = static_cast<float*>(buffer->map(BufferRange(bufferSize, 0), &mapRes));
  ASSERT_TRUE(mapRes.isOk()) << mapRes.message;
  ASSERT_NE(mapped2, nullptr);
  ASSERT_EQ(mapped2[0], 100.0f);
  ASSERT_EQ(mapped2[1], 200.0f);
  ASSERT_EQ(mapped2[2], 300.0f);
  ASSERT_EQ(mapped2[3], 400.0f);
  buffer->unmap();
}

//
// StorageModeShared
//
// Verify that a buffer created with Shared storage mode reports Shared.
//
TEST_F(MetalBufferTest, StorageModeShared) {
  const size_t bufferSize = 64;

  const BufferDesc desc{.type = BufferDesc::BufferTypeBits::Uniform,
                        .length = bufferSize,
                        .storage = ResourceStorage::Shared,
                        .debugName = "testSharedStorage"};

  Result res;
  auto buffer = device_->createBuffer(desc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->storage(), ResourceStorage::Shared);
}

//
// RingBufferCreation
//
// Create a ring buffer and verify it is non-null.
//
TEST_F(MetalBufferTest, RingBufferCreation) {
  const size_t bufferSize = 256;

  const BufferDesc desc{.type = BufferDesc::BufferTypeBits::Uniform,
                        .length = bufferSize,
                        .storage = ResourceStorage::Shared,
                        .hint = BufferDesc::BufferAPIHintBits::Ring,
                        .debugName = "testRingBuffer"};

  Result res;
  auto buffer = device_->createBuffer(desc, &res);
  ASSERT_TRUE(res.isOk()) << res.message;
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(buffer->getSizeInBytes(), bufferSize);
}

TEST_F(MetalBufferTest, CreatePrivateBufferWithInitialData) {
  const uint16_t data[] = {0x1234, 0xabcd, 0x5678};
  const BufferDesc desc{.type = BufferDesc::BufferTypeBits::Index,
                        .data = data,
                        .length = sizeof(data),
                        .storage = ResourceStorage::Private};
  Result result;
  auto buffer = device_->createBuffer(desc, &result);
  ASSERT_TRUE(result.isOk()) << result.message;
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->getSizeInBytes(), sizeof(data));
  EXPECT_EQ(buffer->storage(), ResourceStorage::Private);
  EXPECT_EQ(buffer->getBufferType(), BufferDesc::BufferTypeBits::Index);
  EXPECT_EQ(buffer->requestedApiHints(), 0);
  EXPECT_EQ(buffer->acceptedApiHints(), 0);

  id<MTLBuffer> nativeBuffer = static_cast<metal::Buffer*>(buffer.get())->get();
  ASSERT_NE(nativeBuffer, nil);
  constexpr size_t kCopySize = 8;
  ASSERT_GE(nativeBuffer.length, kCopySize);
  id<MTLBuffer> readback = [nativeBuffer.device newBufferWithLength:kCopySize
                                                            options:MTLResourceStorageModeShared];
  ASSERT_NE(readback, nil);
  id<MTLCommandQueue> queue = [nativeBuffer.device newCommandQueue];
  ASSERT_NE(queue, nil);
  id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
  ASSERT_NE(commandBuffer, nil);
  id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
  ASSERT_NE(encoder, nil);
  [encoder copyFromBuffer:nativeBuffer
             sourceOffset:0
                 toBuffer:readback
        destinationOffset:0
                     size:kCopySize];
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  ASSERT_EQ(commandBuffer.status, MTLCommandBufferStatusCompleted);
  ASSERT_NE(readback.contents, nullptr);
  EXPECT_EQ(std::memcmp(readback.contents, data, sizeof(data)), 0);
  const auto* bytes = static_cast<const uint8_t*>(readback.contents);
  EXPECT_EQ(bytes[sizeof(data)], 0);
  EXPECT_EQ(bytes[sizeof(data) + 1], 0);
}

} // namespace igl::tests
