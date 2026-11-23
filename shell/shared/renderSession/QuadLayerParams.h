/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <igl/RenderPipelineState.h>

namespace igl::shell {

enum class LayerBlendMode : uint8_t {
  Opaque = 0,
  AlphaBlend = 1,
  Custom = 2,
};

struct QuadLayerInfo {
  glm::vec3 position{0.0f, 0.0f, 0.0f};
  glm::vec2 size{1.0f, 1.0f};
  LayerBlendMode blendMode = LayerBlendMode::Opaque;
  uint32_t imageWidth = 1024;
  uint32_t imageHeight = 1024;
  BlendFactor customSrcRGBBlendFactor = BlendFactor::One;
  BlendFactor customSrcAlphaBlendFactor = BlendFactor::One;
  BlendFactor customDstRGBBlendFactor = BlendFactor::Zero;
  BlendFactor customDstAlphaBlendFactor = BlendFactor::Zero;
};

struct QuadLayerParams {
  std::vector<QuadLayerInfo> layerInfo;

  [[nodiscard]] size_t numQuads() const noexcept {
    return layerInfo.size();
  }
};
} // namespace igl::shell
