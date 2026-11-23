/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/TextureTarget.h>

#include "../util/TestDevice.h"

#include <string>
#include <igl/opengl/Device.h>

namespace igl::tests {

// Picking this just to match the texture we will use. If you use a different
// size texture, then you will have to either create a new offscreenTexture_
// and the framebuffer object in your test, so know exactly what the end result
// would be after sampling
#define OFFSCREEN_TEX_WIDTH 2
#define OFFSCREEN_TEX_HEIGHT 2

// Picking this to check mipmap validity, max mipmap level = log(16) - 1 = 3
#define MIPMAP_TEX_WIDTH_16 16
#define MIPMAP_TEX_HEIGHT_16 16
#define MIPMAP_TEX_WIDTH_1023 1023
#define MIPMAP_TEX_HEIGHT_1023 1023

//
// TextureTargetOGLTest
//
// Unit tests for igl::opengl::TextureTarget.
// Covers code paths that may not be hit by top level texture calls from device.
//
class TextureTargetOGLTest : public ::testing::Test {
 private:
 public:
  TextureTargetOGLTest() = default;

  void SetUp() override {
    // Turn off debug breaks, only use in debug mode
    setDebugBreakEnabled(false);

    device_ = util::createTestDevice();
    ASSERT_TRUE(device_ != nullptr);
    context_ = &static_cast<opengl::Device&>(*device_).getContext();
    ASSERT_TRUE(context_ != nullptr);
  }

  void TearDown() override {}

  // Member variables
 protected:
  opengl::IContext* context_{};
  std::shared_ptr<IDevice> device_;
};

//
// Specifications Test
//
// This is a sanity test that override specs are defined correctly.
//
TEST_F(TextureTargetOGLTest, Specifications) {
  std::unique_ptr<opengl::TextureTarget> textureTarget =
      std::make_unique<opengl::TextureTarget>(*context_, TextureFormat::RGBA_UNorm8);
  ASSERT_EQ(textureTarget->getType(), TextureType::TwoD);
  ASSERT_EQ(textureTarget->getUsage(), TextureDesc::TextureUsageBits::Attachment);
}

//
// Texture Creation Paths Test
//
// This tests all failure and success paths for TextureTarget::create.
// Also covers private functions createRenderBuffer and toRenderBufferFormatGL
// which are called within create.
// See tests for Texture.cpp for cases covering cases specific to base class.
//
TEST_F(TextureTargetOGLTest, TextureCreation) {
  std::shared_ptr<opengl::TextureTarget> textureTarget;
  Result ret;
  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           OFFSCREEN_TEX_WIDTH,
                                           OFFSCREEN_TEX_HEIGHT,
                                           TextureDesc::TextureUsageBits::Sampled);

  // kShaderRead not supported by TextureTarget
  textureTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  ret = textureTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Unsupported);

  texDesc.usage = TextureDesc::TextureUsageBits::Attachment;

  // TextureTarget only supports TwoD
  textureTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  texDesc.type = TextureType::ThreeD;
  ret = textureTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Unsupported);

  // TextureTarget only supports a single mip level
  textureTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  texDesc.type = TextureType::TwoD;
  texDesc.numMipLevels = 2;
  ret = textureTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Unsupported);

  // TextureTarget only supports a layer
  textureTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  texDesc.type = TextureType::TwoD;
  texDesc.numMipLevels = 1;
  texDesc.numLayers = 2;
  ret = textureTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Unsupported);

  // Unsupported texture format
  texDesc.format = TextureFormat::Invalid;
  textureTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  texDesc.numLayers = 1;
  ret = textureTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::ArgumentInvalid);

  // Correct usage of TextureTarget::create with > 1 samples
  texDesc.format = TextureFormat::RGBA_UNorm8;
  texDesc.numSamples = 2;
  textureTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  ret = textureTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Ok);
}

//
// Bind and Attach/Detach function Tests.
//
// This tests TextureTarget::bind(),
//                           unbind(),
//                           attachAsColor(),
//                           detachAsColor(),
//                           attachAsDepth(),
//                           attachAsStencil()
//
TEST_F(TextureTargetOGLTest, TextureBindAndAttachAndDetach) {
  const std::unique_ptr<opengl::TextureTarget> textureTarget;
  Result ret;
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 OFFSCREEN_TEX_WIDTH,
                                                 OFFSCREEN_TEX_HEIGHT,
                                                 TextureDesc::TextureUsageBits::Attachment);

  // Create 3 types of targets
  auto colorTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  auto depthTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);
  auto stencilTarget = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);

  // calling create() so that renderBufferID_ is set
  ret = colorTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  ret = depthTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  ret = stencilTarget->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Ok);

  // Since the default framebuffer already comes with unattachable color,
  // depth and stencil, we have to create a new framebuffer before trying
  // to attach our renderbuffers
  GLuint tmpFb = 0;
  context_->genFramebuffers(1, &tmpFb);
  context_->bindFramebuffer(GL_FRAMEBUFFER, tmpFb);

  //--------------------------------------------------------------------------
  // Test Renderbuffer as Color
  //--------------------------------------------------------------------------
  GLint colorType = -1, colorRid = -1;

  colorTarget->attachAsColor(0, opengl::Texture::AttachmentParams{});
  context_->getFramebufferAttachmentParameteriv(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &colorType);
  // Check here that colorType is GL_RENDERBUFFER
  ASSERT_EQ(colorType, GL_RENDERBUFFER);

  context_->getFramebufferAttachmentParameteriv(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &colorRid);
  // Check here that colorRid is anything other than -1
  ASSERT_NE(colorRid, -1);

  colorTarget->detachAsColor(0, false);
  // Nothing to test

  //--------------------------------------------------------------------------
  // Test Renderbuffer as Depth
  //--------------------------------------------------------------------------
  GLint depthType = -1, depthRid = -1;

  depthTarget->attachAsDepth(opengl::Texture::AttachmentParams{});
  context_->getFramebufferAttachmentParameteriv(
      GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &depthType);
  // Check here that depthType is GL_RENDERBUFFER
  ASSERT_EQ(depthType, GL_RENDERBUFFER);

  context_->getFramebufferAttachmentParameteriv(
      GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &depthRid);
  // Check here that depthRid != -1 and depthRid != colorRid
  ASSERT_NE(depthRid, -1);
  ASSERT_NE(depthRid, colorRid);

  //--------------------------------------------------------------------------
  // Test Renderbuffer as Stencil
  //--------------------------------------------------------------------------
  GLint stencilType = -1, stencilRid = -1;

  stencilTarget->attachAsStencil(opengl::Texture::AttachmentParams{});
  context_->getFramebufferAttachmentParameteriv(
      GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &stencilType);
  // Check here that stencilType is GL_RENDERBUFFER
  ASSERT_EQ(stencilType, GL_RENDERBUFFER);

  context_->getFramebufferAttachmentParameteriv(
      GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &stencilRid);
  // Check here that stencilRid != -1 and stencilRid != colorRid and stencilRid != depthRid
  ASSERT_NE(stencilRid, -1);
  ASSERT_NE(stencilRid, colorRid);
  ASSERT_NE(stencilRid, depthRid);

  //--------------------------------------------------------------------------
  // Test bind and unbind
  //--------------------------------------------------------------------------
  GLint value = 0;

  colorTarget->bind();
  // Get renderBuffer binding and check it is non-zero
  context_->getIntegerv(GL_RENDERBUFFER_BINDING, &value);
  ASSERT_NE(value, GL_ZERO);

  colorTarget->unbind();
  // Get renderBuffer binding and check it is zero
  context_->getIntegerv(GL_RENDERBUFFER_BINDING, &value);
  ASSERT_EQ(value, GL_ZERO);

  depthTarget->bind();
  // Get renderBuffer binding and check it is non-zero
  context_->getIntegerv(GL_RENDERBUFFER_BINDING, &value);
  ASSERT_NE(value, GL_ZERO);

  depthTarget->unbind();
  // Get renderBuffer binding and check it is zero
  context_->getIntegerv(GL_RENDERBUFFER_BINDING, &value);
  ASSERT_EQ(value, GL_ZERO);

  stencilTarget->bind();
  // Get renderBuffer binding and check it is non-zero
  context_->getIntegerv(GL_RENDERBUFFER_BINDING, &value);
  ASSERT_NE(value, GL_ZERO);

  stencilTarget->unbind();
  // Get renderBuffer binding and check it is zero
  context_->getIntegerv(GL_RENDERBUFFER_BINDING, &value);
  ASSERT_EQ(value, GL_ZERO);
}

TEST_F(TextureTargetOGLTest, CreateWithDebugName) {
  const std::unique_ptr<opengl::TextureTarget> textureTarget;
  Result ret;
  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           OFFSCREEN_TEX_WIDTH,
                                           OFFSCREEN_TEX_HEIGHT,
                                           TextureDesc::TextureUsageBits::Attachment);
  texDesc.debugName = "test";

  // Create 3 types of targets
  auto target = std::make_unique<opengl::TextureTarget>(*context_, texDesc.format);

  // calling create() so that renderBufferID_ is set
  ret = target->create(texDesc, false);
  ASSERT_EQ(ret.code, Result::Code::Ok);
}
} // namespace igl::tests
