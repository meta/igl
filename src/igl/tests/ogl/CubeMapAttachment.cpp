/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <igl/Framebuffer.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// CubeMapAttachmentOGLTest
//
// Tests for cube map texture creation and face attachment.
//
class CubeMapAttachmentOGLTest : public ::testing::Test {
 public:
  CubeMapAttachmentOGLTest() = default;

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
// CreateCubeTexture
//
// Create a cube map texture.
//
TEST_F(CubeMapAttachmentOGLTest, CreateCubeTexture) {
  Result ret;
  TextureDesc desc = TextureDesc::newCube(
      TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);
  ASSERT_EQ(texture->getType(), TextureType::Cube);
}

//
// AttachFace
//
// Attach a cube map face to a framebuffer.
//
TEST_F(CubeMapAttachmentOGLTest, AttachFace) {
  Result ret;

  // Create cube texture with attachment usage
  TextureDesc desc = TextureDesc::newCube(TextureFormat::RGBA_UNorm8,
                                          4,
                                          4,
                                          TextureDesc::TextureUsageBits::Sampled |
                                              TextureDesc::TextureUsageBits::Attachment);
  auto cubeTexture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cubeTexture, nullptr);

  // Create a framebuffer with one face of the cube map
  // Note: In IGL, cube map attachment uses the whole texture;
  // face selection is implementation-specific
  const FramebufferDesc fbDesc{.colorAttachments = {{.texture = cubeTexture}}};

  auto framebuffer = iglDev_->createFramebuffer(fbDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(framebuffer, nullptr);

  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

} // namespace igl::tests
