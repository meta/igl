/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#pragma once

#include <IGLU/imgui/Session.h>
#include <glm/ext/matrix_float4x4.hpp>
#include <shell/shared/platform/Platform.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <igl/FPSCounter.h>
#include <igl/RenderPass.h>

namespace igl::shell {

struct VertexFormat {
  glm::mat4 mvpMatrix;
};

class BindGroupSession : public RenderSession {
 public:
  explicit BindGroupSession(std::shared_ptr<Platform> platform);
  void initialize() noexcept override;
  void update(SurfaceTextures surfaceTextures) noexcept override;
  void releaseFramebuffer() override;

 private:
  RenderPassDesc renderPass_;
  FramebufferDesc framebufferDesc_;
  std::shared_ptr<IRenderPipelineState> pipelineState_;
  std::shared_ptr<IVertexInputState> vertexInput0_;
  std::shared_ptr<IShaderStages> shaderStages_;
  std::shared_ptr<IBuffer> vb0_, ib0_; // Buffers for vertices and indices (or constants)
  std::shared_ptr<IFramebuffer> framebuffer_;
  Holder<BindGroupTextureHandle> bindGroupTextures_;
  std::unique_ptr<iglu::imgui::Session> imguiSession_;

  VertexFormat vertexParameters_;
  FPSCounter fps_;
  float angle_ = 0;

  void createSamplerAndTextures(const IDevice& /*device*/);
};

inline void BindGroupSession::releaseFramebuffer() {
  framebuffer_ = nullptr;
}

} // namespace igl::shell
