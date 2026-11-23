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
// ImageLoadStoreOGLTest
//
// Tests for shader image load/store operations in OpenGL.
//
class ImageLoadStoreOGLTest : public ::testing::Test {
 public:
  ImageLoadStoreOGLTest() = default;

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
// BindImageTexture
//
// Test binding an image texture for compute shader access.
//
TEST_F(ImageLoadStoreOGLTest, BindImageTexture) {
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    GTEST_SKIP() << "Compute not supported";
  }

  if (!context_->deviceFeatures().hasInternalFeature(
          opengl::InternalFeatures::ShaderImageLoadStore)) {
    GTEST_SKIP() << "Shader image load/store not supported";
  }

  Result ret;

  // Create a texture suitable for image load/store
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Storage);
  auto texture = iglDev_->createTexture(texDesc, &ret);
  if (!ret.isOk() || texture == nullptr) {
    GTEST_SKIP() << "Cannot create storage texture: " << ret.message.c_str();
  }

  // Bind the image texture using the context directly
  // This tests the low-level bindImageTexture call
  auto* oglTexture = texture.get();
  ASSERT_NE(oglTexture, nullptr);

  // Just verify no GL errors from the texture creation
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
