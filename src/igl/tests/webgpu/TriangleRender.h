/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <igl/CommandQueue.h>
#include <igl/Device.h>
#include <igl/RenderPipelineState.h>

namespace igl::tests::webgpu {

/// @brief Draws the reference triangle (red, green and blue corners on a fixed clear color).
class TriangleRenderer final {
 public:
  /// Creates the command queue and the pipeline for targets of `format`.
  [[nodiscard]] Result initialize(IDevice& device, TextureFormat format);
  /// Clears `target` and draws the triangle into it; submits without waiting.
  [[nodiscard]] Result render(const std::shared_ptr<ITexture>& target);

  [[nodiscard]] ICommandQueue& getQueue() const;

 private:
  IDevice* device_ = nullptr;
  std::shared_ptr<ICommandQueue> queue_;
  std::shared_ptr<IRenderPipelineState> pipeline_;
};

[[nodiscard]] inline ICommandQueue& TriangleRenderer::getQueue() const {
  return *queue_;
}

/// Renders the reference triangle into a `size` x `size` RGBA8 target and reads it back with rows
/// top to bottom. Shared by the native test and the wasm sample, so their frames can be compared
/// byte for byte.
[[nodiscard]] Result renderTriangle(IDevice& device, uint32_t size, std::vector<uint8_t>& outRgba);

} // namespace igl::tests::webgpu
