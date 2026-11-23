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
// DebugOutputOGLTest
//
// Tests for OpenGL debug output functionality.
//
class DebugOutputOGLTest : public ::testing::Test {
 public:
  DebugOutputOGLTest() = default;

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
// DebugMessageLogNoError
//
// Query debug message log and verify no errors.
//
TEST_F(DebugOutputOGLTest, DebugMessageLogNoError) {
  if (!context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::DebugMessage)) {
    GTEST_SKIP() << "Debug messages not supported";
  }

  // Try to get debug messages - there may not be any
  GLenum sources[1];
  GLenum types[1];
  GLuint ids[1];
  GLenum severities[1];
  GLsizei lengths[1];
  GLchar messageLog[256];

  // getDebugMessageLog should not crash, even when no messages are available
  GLuint count = context_->getDebugMessageLog(
      1, sizeof(messageLog), sources, types, ids, severities, lengths, messageLog);

  // Count should be 0 or 1 (whatever messages may be in the log)
  ASSERT_LE(count, 1u);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
