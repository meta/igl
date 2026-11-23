/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <shell/shared/renderSession/QuadLayerParams.h>

namespace igl::shell::tests {

// ---------------------------------------------------------------------------
// LayerBlendMode enum values
// ---------------------------------------------------------------------------

TEST(LayerBlendModeTest, OpaqueValue) {
  EXPECT_EQ(static_cast<uint8_t>(LayerBlendMode::Opaque), 0u);
}

TEST(LayerBlendModeTest, AlphaBlendValue) {
  EXPECT_EQ(static_cast<uint8_t>(LayerBlendMode::AlphaBlend), 1u);
}

TEST(LayerBlendModeTest, CustomValue) {
  EXPECT_EQ(static_cast<uint8_t>(LayerBlendMode::Custom), 2u);
}

// ---------------------------------------------------------------------------
// QuadLayerInfo default member initializers
// ---------------------------------------------------------------------------

TEST(QuadLayerInfoTest, DefaultPosition) {
  const QuadLayerInfo info;
  EXPECT_FLOAT_EQ(info.position.x, 0.0f);
  EXPECT_FLOAT_EQ(info.position.y, 0.0f);
  EXPECT_FLOAT_EQ(info.position.z, 0.0f);
}

TEST(QuadLayerInfoTest, DefaultSize) {
  const QuadLayerInfo info;
  EXPECT_FLOAT_EQ(info.size.x, 1.0f);
  EXPECT_FLOAT_EQ(info.size.y, 1.0f);
}

TEST(QuadLayerInfoTest, DefaultBlendMode) {
  const QuadLayerInfo info;
  EXPECT_EQ(info.blendMode, LayerBlendMode::Opaque);
}

TEST(QuadLayerInfoTest, DefaultImageWidth) {
  const QuadLayerInfo info;
  EXPECT_EQ(info.imageWidth, 1024u);
}

TEST(QuadLayerInfoTest, DefaultImageHeight) {
  const QuadLayerInfo info;
  EXPECT_EQ(info.imageHeight, 1024u);
}

TEST(QuadLayerInfoTest, DefaultCustomSrcRGBBlendFactor) {
  const QuadLayerInfo info;
  EXPECT_EQ(info.customSrcRGBBlendFactor, igl::BlendFactor::One);
}

TEST(QuadLayerInfoTest, DefaultCustomSrcAlphaBlendFactor) {
  const QuadLayerInfo info;
  EXPECT_EQ(info.customSrcAlphaBlendFactor, igl::BlendFactor::One);
}

TEST(QuadLayerInfoTest, DefaultCustomDstRGBBlendFactor) {
  const QuadLayerInfo info;
  EXPECT_EQ(info.customDstRGBBlendFactor, igl::BlendFactor::Zero);
}

TEST(QuadLayerInfoTest, DefaultCustomDstAlphaBlendFactor) {
  const QuadLayerInfo info;
  EXPECT_EQ(info.customDstAlphaBlendFactor, igl::BlendFactor::Zero);
}

// ---------------------------------------------------------------------------
// QuadLayerInfo with non-default values
// ---------------------------------------------------------------------------

TEST(QuadLayerInfoTest, CustomPositionSticks) {
  QuadLayerInfo info;
  info.position = glm::vec3(1.5f, -2.5f, 3.5f);
  EXPECT_FLOAT_EQ(info.position.x, 1.5f);
  EXPECT_FLOAT_EQ(info.position.y, -2.5f);
  EXPECT_FLOAT_EQ(info.position.z, 3.5f);
}

TEST(QuadLayerInfoTest, CustomSizeSticks) {
  QuadLayerInfo info;
  info.size = glm::vec2(2.0f, 4.0f);
  EXPECT_FLOAT_EQ(info.size.x, 2.0f);
  EXPECT_FLOAT_EQ(info.size.y, 4.0f);
}

TEST(QuadLayerInfoTest, CustomImageDimensionsStick) {
  QuadLayerInfo info;
  info.imageWidth = 512u;
  info.imageHeight = 256u;
  EXPECT_EQ(info.imageWidth, 512u);
  EXPECT_EQ(info.imageHeight, 256u);
}

TEST(QuadLayerInfoTest, BlendModeAlphaBlendSticks) {
  QuadLayerInfo info;
  info.blendMode = LayerBlendMode::AlphaBlend;
  EXPECT_EQ(info.blendMode, LayerBlendMode::AlphaBlend);
}

TEST(QuadLayerInfoTest, BlendModeCustomSticks) {
  QuadLayerInfo info;
  info.blendMode = LayerBlendMode::Custom;
  EXPECT_EQ(info.blendMode, LayerBlendMode::Custom);
}

TEST(QuadLayerInfoTest, CustomBlendFactorsStick) {
  QuadLayerInfo info;
  info.blendMode = LayerBlendMode::Custom;
  info.customSrcRGBBlendFactor = BlendFactor::SrcAlpha;
  info.customSrcAlphaBlendFactor = BlendFactor::OneMinusSrcAlpha;
  info.customDstRGBBlendFactor = BlendFactor::DstColor;
  info.customDstAlphaBlendFactor = BlendFactor::OneMinusDstColor;

  EXPECT_EQ(info.customSrcRGBBlendFactor, igl::BlendFactor::SrcAlpha);
  EXPECT_EQ(info.customSrcAlphaBlendFactor, igl::BlendFactor::OneMinusSrcAlpha);
  EXPECT_EQ(info.customDstRGBBlendFactor, igl::BlendFactor::DstColor);
  EXPECT_EQ(info.customDstAlphaBlendFactor, igl::BlendFactor::OneMinusDstColor);
}

TEST(QuadLayerInfoTest, AggregateInitializationOverridesAllFields) {
  const QuadLayerInfo info{
      .position = glm::vec3(10.0f, 20.0f, 30.0f),
      .size = glm::vec2(5.0f, 6.0f),
      .blendMode = LayerBlendMode::Custom,
      .imageWidth = 128u,
      .imageHeight = 64u,
      .customSrcRGBBlendFactor = BlendFactor::BlendColor,
      .customSrcAlphaBlendFactor = BlendFactor::OneMinusBlendColor,
      .customDstRGBBlendFactor = BlendFactor::BlendAlpha,
      .customDstAlphaBlendFactor = BlendFactor::OneMinusBlendAlpha,
  };

  EXPECT_FLOAT_EQ(info.position.x, 10.0f);
  EXPECT_FLOAT_EQ(info.position.y, 20.0f);
  EXPECT_FLOAT_EQ(info.position.z, 30.0f);
  EXPECT_FLOAT_EQ(info.size.x, 5.0f);
  EXPECT_FLOAT_EQ(info.size.y, 6.0f);
  EXPECT_EQ(info.blendMode, LayerBlendMode::Custom);
  EXPECT_EQ(info.imageWidth, 128u);
  EXPECT_EQ(info.imageHeight, 64u);
  EXPECT_EQ(info.customSrcRGBBlendFactor, igl::BlendFactor::BlendColor);
  EXPECT_EQ(info.customSrcAlphaBlendFactor, igl::BlendFactor::OneMinusBlendColor);
  EXPECT_EQ(info.customDstRGBBlendFactor, igl::BlendFactor::BlendAlpha);
  EXPECT_EQ(info.customDstAlphaBlendFactor, igl::BlendFactor::OneMinusBlendAlpha);
}

// ---------------------------------------------------------------------------
// QuadLayerParams::numQuads()
// ---------------------------------------------------------------------------

TEST(QuadLayerParamsTest, DefaultLayerInfoIsEmpty) {
  const QuadLayerParams params;
  EXPECT_TRUE(params.layerInfo.empty());
}

TEST(QuadLayerParamsTest, NumQuadsEmpty) {
  const QuadLayerParams params;
  EXPECT_EQ(params.numQuads(), 0u);
}

TEST(QuadLayerParamsTest, NumQuadsAfterSingleInsert) {
  QuadLayerParams params;
  params.layerInfo.emplace_back();
  EXPECT_EQ(params.numQuads(), 1u);
}

TEST(QuadLayerParamsTest, NumQuadsAfterMultipleInserts) {
  QuadLayerParams params;
  params.layerInfo.emplace_back();
  params.layerInfo.emplace_back();
  params.layerInfo.emplace_back();
  EXPECT_EQ(params.numQuads(), 3u);
}

TEST(QuadLayerParamsTest, NumQuadsTracksVectorAfterPopBack) {
  QuadLayerParams params;
  params.layerInfo.emplace_back();
  params.layerInfo.emplace_back();
  EXPECT_EQ(params.numQuads(), 2u);

  params.layerInfo.pop_back();
  EXPECT_EQ(params.numQuads(), 1u);
}

TEST(QuadLayerParamsTest, NumQuadsMatchesVectorSizeAfterClear) {
  QuadLayerParams params;
  params.layerInfo.emplace_back();
  params.layerInfo.emplace_back();
  params.layerInfo.clear();
  EXPECT_EQ(params.numQuads(), 0u);
}

TEST(QuadLayerParamsTest, LayerInfoPreservesInsertedValues) {
  QuadLayerParams params;
  QuadLayerInfo first;
  first.imageWidth = 111u;
  QuadLayerInfo second;
  second.imageWidth = 222u;

  params.layerInfo.push_back(first);
  params.layerInfo.push_back(second);

  ASSERT_EQ(params.numQuads(), 2u);
  EXPECT_EQ(params.layerInfo[0].imageWidth, 111u);
  EXPECT_EQ(params.layerInfo[1].imageWidth, 222u);
}

} // namespace igl::shell::tests
