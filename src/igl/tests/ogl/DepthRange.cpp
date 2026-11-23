/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// DepthRangeOGLTest
//
// Tests for glDepthRangef operations in OpenGL.
//
class DepthRangeOGLTest : public ::testing::Test {
 public:
  DepthRangeOGLTest() = default;

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
// DepthRangeNoError
//
// Set the depth range and verify no GL errors.
//
TEST_F(DepthRangeOGLTest, DepthRangeNoError) {
  // Set depth range to standard [0, 1]
  context_->depthRangef(0.0f, 1.0f);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  // Set depth range to a custom range [0.25, 0.75]
  context_->depthRangef(0.25f, 0.75f);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  // Restore default
  context_->depthRangef(0.0f, 1.0f);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
