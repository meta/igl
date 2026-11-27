/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <utility>
#include <igl/DepthStencilState.h>

namespace igl::webgpu {

/// @brief Depth/stencil state is part of a WebGPU render pipeline, so this only keeps the
/// descriptor; RenderPipelineState picks a pipeline variant for it at draw time.
class DepthStencilState final : public IDepthStencilState {
 public:
  explicit DepthStencilState(DepthStencilStateDesc desc) : desc_(std::move(desc)) {}

  [[nodiscard]] const DepthStencilStateDesc& getDesc() const noexcept;

 private:
  DepthStencilStateDesc desc_;
};

[[nodiscard]] inline const DepthStencilStateDesc& DepthStencilState::getDesc() const noexcept {
  return desc_;
}

} // namespace igl::webgpu
