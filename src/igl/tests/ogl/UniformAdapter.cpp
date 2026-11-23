/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/UniformAdapter.h>

#include "../util/Common.h"

#include <igl/Uniform.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

#define OFFSCREEN_TEX_WIDTH 2
#define OFFSCREEN_TEX_HEIGHT 2

//
// UniformAdapterOGLTest
//
// Tests for the OpenGL UniformAdapter.
//
class UniformAdapterOGLTest : public ::testing::Test {
 public:
  UniformAdapterOGLTest() = default;

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
// SetUniform
//
// Store uniform data in the adapter.
//
TEST_F(UniformAdapterOGLTest, SetUniform) {
  opengl::UniformAdapter adapter(*context_, opengl::UniformAdapter::PipelineType::Render);

  // Create a uniform descriptor for a float uniform
  const UniformDesc desc = {
      .location = 0,
      .type = UniformType::Float,
      .numElements = 1,
  };

  float value = 42.0f;
  Result ret;
  adapter.setUniform(desc, &value, &ret);
  // setUniform should succeed
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
}

//
// ClearUniformBuffers
//
// Verify clearUniformBuffers resets the adapter state.
//
TEST_F(UniformAdapterOGLTest, ClearUniformBuffers) {
  opengl::UniformAdapter adapter(*context_, opengl::UniformAdapter::PipelineType::Render);

  // Set a uniform
  const UniformDesc desc = {
      .location = 0,
      .type = UniformType::Float,
      .numElements = 1,
  };

  float value = 1.0f;
  Result ret;
  adapter.setUniform(desc, &value, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Clear and verify no crash
  adapter.clearUniformBuffers();

  // Set again after clear should also work
  adapter.setUniform(desc, &value, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
}

} // namespace igl::tests
