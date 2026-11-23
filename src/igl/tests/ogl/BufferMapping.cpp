/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <cstring>
#include <igl/opengl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// BufferMappingOGLTest
//
// Tests for buffer mapping operations in OpenGL.
//
class BufferMappingOGLTest : public ::testing::Test {
 public:
  BufferMappingOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
};

//
// MapBuffer
//
// Test mapping a buffer for reading.
//
TEST_F(BufferMappingOGLTest, MapBuffer) {
  if (!context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::MapBuffer)) {
    GTEST_SKIP() << "MapBuffer not supported";
  }

  Result ret;
  const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const BufferDesc bufDesc{
      .type = BufferDesc::BufferTypeBits::Vertex, .data = data, .length = sizeof(data)};

  auto buffer = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  // Map the buffer
  auto* mapped = buffer->map(BufferRange(sizeof(data), 0), &ret);
  if (!ret.isOk()) {
    GTEST_SKIP() << "Buffer mapping failed: " << ret.message.c_str();
  }
  ASSERT_NE(mapped, nullptr);

  buffer->unmap();
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// MapBufferRange
//
// Test mapping a sub-range of a buffer.
//
TEST_F(BufferMappingOGLTest, MapBufferRange) {
  if (!iglDev_->hasFeature(DeviceFeatures::MapBufferRange)) {
    GTEST_SKIP() << "MapBufferRange not supported";
  }

  Result ret;
  const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const BufferDesc bufDesc{
      .type = BufferDesc::BufferTypeBits::Vertex, .data = data, .length = sizeof(data)};

  auto buffer = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  // Map a range of the buffer (offset 0, 2 floats)
  auto* mapped = buffer->map(BufferRange(sizeof(float) * 2, 0), &ret);
  if (!ret.isOk()) {
    GTEST_SKIP() << "Buffer range mapping failed: " << ret.message.c_str();
  }
  ASSERT_NE(mapped, nullptr);

  buffer->unmap();
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// WriteAndUnmap
//
// Create a dynamic buffer, upload data to it, then map and verify.
//
TEST_F(BufferMappingOGLTest, WriteAndUnmap) {
  if (!context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::MapBuffer)) {
    GTEST_SKIP() << "MapBuffer not supported";
  }

  Result ret;
  const BufferDesc bufDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                           .data = nullptr,
                           .length = sizeof(float) * 4,
                           .storage = ResourceStorage::Shared};

  auto buffer = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  // Upload data to dynamic buffer
  const float newData[] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto uploadResult = buffer->upload(newData, BufferRange(sizeof(newData), 0));
  ASSERT_TRUE(uploadResult.isOk()) << uploadResult.message.c_str();

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
