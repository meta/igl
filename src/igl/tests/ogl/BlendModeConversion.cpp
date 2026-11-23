/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <utility>
#include <vector>
#include <igl/opengl/RenderPipelineState.h>

namespace igl::tests {

//
// BlendModeConversionOGLTest
//
// Tests for BlendOp and BlendFactor conversion in the OpenGL RenderPipelineState.
//
class BlendModeConversionOGLTest : public ::testing::Test {
 public:
  BlendModeConversionOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
  }

  void TearDown() override {}
};

//
// ConvertBlendOp
//
// Test all 5 blend operations map to their GL equivalents.
//
TEST_F(BlendModeConversionOGLTest, ConvertBlendOp) {
  const std::vector<std::pair<BlendOp, GLenum>> conversions = {
      {BlendOp::Add, GL_FUNC_ADD},
      {BlendOp::Subtract, GL_FUNC_SUBTRACT},
      {BlendOp::ReverseSubtract, GL_FUNC_REVERSE_SUBTRACT},
      {BlendOp::Min, GL_MIN},
      {BlendOp::Max, GL_MAX},
  };

  for (const auto& [iglOp, glOp] : conversions) {
    ASSERT_EQ(igl::opengl::RenderPipelineState::convertBlendOp(iglOp), glOp);
  }
}

//
// ConvertBlendFactor
//
// Test all blend factors map to their GL equivalents.
//
TEST_F(BlendModeConversionOGLTest, ConvertBlendFactor) {
  const std::vector<std::pair<BlendFactor, GLenum>> conversions = {
      {BlendFactor::Zero, GL_ZERO},
      {BlendFactor::One, GL_ONE},
      {BlendFactor::SrcColor, GL_SRC_COLOR},
      {BlendFactor::OneMinusSrcColor, GL_ONE_MINUS_SRC_COLOR},
      {BlendFactor::DstColor, GL_DST_COLOR},
      {BlendFactor::OneMinusDstColor, GL_ONE_MINUS_DST_COLOR},
      {BlendFactor::SrcAlpha, GL_SRC_ALPHA},
      {BlendFactor::OneMinusSrcAlpha, GL_ONE_MINUS_SRC_ALPHA},
      {BlendFactor::DstAlpha, GL_DST_ALPHA},
      {BlendFactor::OneMinusDstAlpha, GL_ONE_MINUS_DST_ALPHA},
      {BlendFactor::BlendColor, GL_CONSTANT_COLOR},
      {BlendFactor::OneMinusBlendColor, GL_ONE_MINUS_CONSTANT_COLOR},
      {BlendFactor::BlendAlpha, GL_CONSTANT_ALPHA},
      {BlendFactor::OneMinusBlendAlpha, GL_ONE_MINUS_CONSTANT_ALPHA},
      {BlendFactor::SrcAlphaSaturated, GL_SRC_ALPHA_SATURATE},
      // Unsupported values default to GL_ONE
      {BlendFactor::Src1Color, GL_ONE},
      {BlendFactor::OneMinusSrc1Color, GL_ONE},
      {BlendFactor::Src1Alpha, GL_ONE},
      {BlendFactor::OneMinusSrc1Alpha, GL_ONE},
  };

  for (const auto& [iglFactor, glFactor] : conversions) {
    ASSERT_EQ(igl::opengl::RenderPipelineState::convertBlendFactor(iglFactor), glFactor);
  }
}

} // namespace igl::tests
