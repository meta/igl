/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/SamplerState.h>

#include "../util/TestDevice.h"

#include <string>
#include <igl/opengl/Device.h>

namespace igl::tests {

#ifndef GL_CLAMP_TO_BORDER
#define GL_CLAMP_TO_BORDER 0x812D
#endif

//
// SamplerStateOGLTest
//
// Unit tests for igl::opengl::SamplerState.
//
class SamplerStateOGLTest : public ::testing::Test {
 public:
  // Set up common resources.
  void SetUp() override {
    // Turn off debug break so unit tests can run
    setDebugBreakEnabled(false);

    device_ = util::createTestDevice();
    context_ = &static_cast<opengl::Device&>(*device_).getContext();

    ASSERT_TRUE(context_ != nullptr);
  }

 protected:
  std::shared_ptr<IDevice> device_;
  opengl::IContext* context_{};
};

//
// convertGLMinFilter tests
//
// This tests conversions from glMinFilter to
// SamplerMinMagFilter
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertGLMinFilter) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertGLMinFilter(GL_NEAREST), SamplerMinMagFilter::Nearest);
  ASSERT_EQ(dummySamplerState->convertGLMinFilter(GL_NEAREST_MIPMAP_NEAREST),
            SamplerMinMagFilter::Nearest);
  ASSERT_EQ(dummySamplerState->convertGLMinFilter(GL_NEAREST_MIPMAP_LINEAR),
            SamplerMinMagFilter::Nearest);
  ASSERT_EQ(dummySamplerState->convertGLMinFilter(GL_LINEAR), SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMinFilter(GL_LINEAR_MIPMAP_NEAREST),
            SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMinFilter(GL_LINEAR_MIPMAP_LINEAR),
            SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMinFilter(GL_NONE), SamplerMinMagFilter::Nearest);
}

//
// convertGLMipFilter tests
//
// This test checks the conversion from OpenGL Mip filter enums to their
// corresponding IGL equivalent.
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertGLMipFilter) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertGLMipFilter(GL_NEAREST), SamplerMipFilter::Disabled);
  ASSERT_EQ(dummySamplerState->convertGLMipFilter(GL_NEAREST_MIPMAP_NEAREST),
            SamplerMipFilter::Nearest);
  ASSERT_EQ(dummySamplerState->convertGLMipFilter(GL_NEAREST_MIPMAP_LINEAR),
            SamplerMipFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMipFilter(GL_LINEAR), SamplerMipFilter::Disabled);
  ASSERT_EQ(dummySamplerState->convertGLMipFilter(GL_LINEAR_MIPMAP_NEAREST),
            SamplerMipFilter::Nearest);
  ASSERT_EQ(dummySamplerState->convertGLMipFilter(GL_LINEAR_MIPMAP_LINEAR),
            SamplerMipFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMipFilter(GL_NONE), SamplerMipFilter::Disabled);
}

//
// convertMinMipFilter tests
//
// This test checks the conversion from IGL Min/Mip filter enums to their
// corresponding OpenGL equivalent.
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertMinMipFilter) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertMinMipFilter(SamplerMinMagFilter::Nearest,
                                                   SamplerMipFilter::Disabled),
            GL_NEAREST);
  ASSERT_EQ(dummySamplerState->convertMinMipFilter(SamplerMinMagFilter::Nearest,
                                                   SamplerMipFilter::Linear),
            GL_NEAREST_MIPMAP_LINEAR);
  ASSERT_EQ(dummySamplerState->convertMinMipFilter(SamplerMinMagFilter::Nearest,
                                                   SamplerMipFilter::Nearest),
            GL_NEAREST_MIPMAP_NEAREST);
}

//
// convertMinMipFilter (Linear min filter) tests
//
// The SamplerStateConvertMinMipFilter test above only exercises the Nearest
// minFilter branch. This test covers the Linear minFilter branch of
// convertMinMipFilter() across all three mip filter modes.
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertMinMipFilterLinear) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertMinMipFilter(SamplerMinMagFilter::Linear,
                                                   SamplerMipFilter::Disabled),
            GL_LINEAR);
  ASSERT_EQ(dummySamplerState->convertMinMipFilter(SamplerMinMagFilter::Linear,
                                                   SamplerMipFilter::Nearest),
            GL_LINEAR_MIPMAP_NEAREST);
  ASSERT_EQ(
      dummySamplerState->convertMinMipFilter(SamplerMinMagFilter::Linear, SamplerMipFilter::Linear),
      GL_LINEAR_MIPMAP_LINEAR);
}

//
// convertGLMagFilter tests
//
// This test checks the conversion from OpenGL Min/Mag filter enums to their
// corresponding IGL equivalent.
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertGLMagFilter) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertGLMagFilter(GL_NEAREST), SamplerMinMagFilter::Nearest);
  ASSERT_EQ(dummySamplerState->convertGLMagFilter(GL_NEAREST_MIPMAP_NEAREST),
            SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMagFilter(GL_NEAREST_MIPMAP_LINEAR),
            SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMagFilter(GL_LINEAR), SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMagFilter(GL_LINEAR_MIPMAP_NEAREST),
            SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMagFilter(GL_LINEAR_MIPMAP_LINEAR),
            SamplerMinMagFilter::Linear);
  ASSERT_EQ(dummySamplerState->convertGLMagFilter(GL_NONE), SamplerMinMagFilter::Linear);
}

//
// convertMagFilter tests
//
// This test checks the conversion from IGL Mag filter enums to their
// corresponding OpenGL equivalent.
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertMagFilter) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertMagFilter(SamplerMinMagFilter::Nearest), GL_NEAREST);
  ASSERT_EQ(dummySamplerState->convertMagFilter(SamplerMinMagFilter::Linear), GL_LINEAR);
}

//
// convertGLAddressMode tests
//
// This test checks the conversion from OpenGL Address Mode enums to their
// corresponding IGL equivalent.
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertGLAddressMode) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertGLAddressMode(GL_REPEAT), SamplerAddressMode::Repeat);
  ASSERT_EQ(dummySamplerState->convertGLAddressMode(GL_CLAMP_TO_EDGE), SamplerAddressMode::Clamp);
  ASSERT_EQ(dummySamplerState->convertGLAddressMode(GL_MIRRORED_REPEAT),
            SamplerAddressMode::MirrorRepeat);
  ASSERT_EQ(dummySamplerState->convertGLAddressMode(GL_CLAMP_TO_BORDER),
            SamplerAddressMode::ClampToBorder);
}

//
// convertAddressMode tests
//
// This test checks the conversion from IGL Min/Mag filter enums to their
// corresponding OpenGL equivalent.
//
TEST_F(SamplerStateOGLTest, SamplerStateConvertAddressMode) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  ASSERT_EQ(dummySamplerState->convertAddressMode(SamplerAddressMode::Repeat), GL_REPEAT);
  ASSERT_EQ(dummySamplerState->convertAddressMode(SamplerAddressMode::Clamp), GL_CLAMP_TO_EDGE);
  ASSERT_EQ(dummySamplerState->convertAddressMode(SamplerAddressMode::MirrorRepeat),
            GL_MIRRORED_REPEAT);
  ASSERT_EQ(dummySamplerState->convertAddressMode(SamplerAddressMode::ClampToBorder),
            GL_CLAMP_TO_BORDER);
}

TEST_F(SamplerStateOGLTest, BindNullTexture) {
  auto dummySamplerState =
      std::make_unique<opengl::SamplerState>(*context_, SamplerStateDesc::newLinear());
  dummySamplerState->bind(nullptr);
  EXPECT_TRUE(true); // Expect no crash
}

} // namespace igl::tests
