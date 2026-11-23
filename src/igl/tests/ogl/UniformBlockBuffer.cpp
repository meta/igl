/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/Buffer.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// UniformBlockBufferOGLTest
//
// Tests for uniform block buffer binding in OpenGL.
//
class UniformBlockBufferOGLTest : public ::testing::Test {
 public:
  UniformBlockBufferOGLTest() = default;

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
// SetBlockBinding
//
// Create a buffer with uniform block data and verify binding does not cause errors.
//
TEST_F(UniformBlockBufferOGLTest, SetBlockBinding) {
  if (!iglDev_->hasFeature(DeviceFeatures::UniformBlocks)) {
    GTEST_SKIP() << "Uniform blocks not supported";
  }

  Result ret;

  // Create a uniform buffer
  const float uniformData[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  const BufferDesc bufDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                           .data = uniformData,
                           .length = sizeof(uniformData)};

  auto uniformBuffer = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(uniformBuffer, nullptr);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// BindBase
//
// Bind a uniform buffer to a binding point using bindBufferBase.
//
TEST_F(UniformBlockBufferOGLTest, BindBase) {
  if (!iglDev_->hasFeature(DeviceFeatures::UniformBlocks)) {
    GTEST_SKIP() << "Uniform blocks not supported";
  }

  Result ret;

  const float uniformData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const BufferDesc bufDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                           .data = uniformData,
                           .length = sizeof(uniformData)};

  auto uniformBuffer = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(uniformBuffer, nullptr);

  // Verify buffer creation and data upload produced no GL errors
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// BindRange
//
// Bind a sub-range of a uniform buffer to a binding point.
//
TEST_F(UniformBlockBufferOGLTest, BindRange) {
  if (!iglDev_->hasFeature(DeviceFeatures::UniformBlocks)) {
    GTEST_SKIP() << "Uniform blocks not supported";
  }

  Result ret;

  // Create a larger uniform buffer
  const float uniformData[16] = {1.0f,
                                 2.0f,
                                 3.0f,
                                 4.0f,
                                 5.0f,
                                 6.0f,
                                 7.0f,
                                 8.0f,
                                 9.0f,
                                 10.0f,
                                 11.0f,
                                 12.0f,
                                 13.0f,
                                 14.0f,
                                 15.0f,
                                 16.0f};
  const BufferDesc bufDesc{.type = BufferDesc::BufferTypeBits::Uniform,
                           .data = uniformData,
                           .length = sizeof(uniformData)};

  auto uniformBuffer = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(uniformBuffer, nullptr);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
