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
// FeatureLimitsOGLTest
//
// Tests for querying OpenGL feature limits.
//
class FeatureLimitsOGLTest : public ::testing::Test {
 public:
  FeatureLimitsOGLTest() = default;

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
// MaxTextureSize
//
// Verify that the maximum texture size is at least a reasonable minimum.
//
TEST_F(FeatureLimitsOGLTest, MaxTextureSize) {
  size_t maxTextureSize = 0;
  bool hasLimit =
      iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxTextureDimension1D2D, maxTextureSize);
  ASSERT_TRUE(hasLimit);
  // OpenGL spec guarantees at least 64, but practically all hardware supports much more
  ASSERT_GE(maxTextureSize, 64u);
}

//
// MaxRenderTargets
//
// Verify the number of render targets is at least 1.
//
TEST_F(FeatureLimitsOGLTest, MaxRenderTargets) {
  size_t maxRenderTargets = 0;
  bool hasLimit =
      iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxColorAttachments, maxRenderTargets);
  if (!hasLimit) {
    // Some backends might not report this limit
    GTEST_SKIP() << "MaxColorAttachments limit not reported";
  }
  // Must support at least 1 render target
  ASSERT_GE(maxRenderTargets, 1u);
}

} // namespace igl::tests
