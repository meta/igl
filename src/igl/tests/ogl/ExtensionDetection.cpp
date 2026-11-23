/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/opengl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// ExtensionDetectionOGLTest
//
// Tests for OpenGL extension detection.
//
class ExtensionDetectionOGLTest : public ::testing::Test {
 public:
  ExtensionDetectionOGLTest() = default;

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
// HasExtensionReturnsValue
//
// Verify that querying for extensions returns consistent boolean values
// and does not crash.
//
TEST_F(ExtensionDetectionOGLTest, HasExtensionReturnsValue) {
  const auto& features = context_->deviceFeatures();

  // Query a variety of extensions and verify they return consistently
  // (calling twice should give the same result)
  bool timerQuery1 = features.hasExtension(opengl::Extensions::TimerQuery);
  bool timerQuery2 = features.hasExtension(opengl::Extensions::TimerQuery);
  ASSERT_EQ(timerQuery1, timerQuery2);

  bool vao1 = features.hasExtension(opengl::Extensions::VertexArrayObject);
  bool vao2 = features.hasExtension(opengl::Extensions::VertexArrayObject);
  ASSERT_EQ(vao1, vao2);

  bool fbBlit1 = features.hasExtension(opengl::Extensions::FramebufferBlit);
  bool fbBlit2 = features.hasExtension(opengl::Extensions::FramebufferBlit);
  ASSERT_EQ(fbBlit1, fbBlit2);

  bool mapBuffer1 = features.hasExtension(opengl::Extensions::MapBuffer);
  bool mapBuffer2 = features.hasExtension(opengl::Extensions::MapBuffer);
  ASSERT_EQ(mapBuffer1, mapBuffer2);

  bool depth241 = features.hasExtension(opengl::Extensions::Depth24);
  bool depth242 = features.hasExtension(opengl::Extensions::Depth24);
  ASSERT_EQ(depth241, depth242);
}

} // namespace igl::tests
