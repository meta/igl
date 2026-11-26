/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#pragma once

#include <shell/renderSessions/ShellRenderSession.h>
#include <shell/shared/platform/Platform.h>
#include <igl/IGL.h>

namespace igl::shell {

class ClothSimulationSession : public ShellRenderSession {
 public:
  explicit ClothSimulationSession(std::shared_ptr<Platform> platform) :
    ShellRenderSession(std::move(platform)) {}
  void initialize() noexcept override;
  void update(SurfaceTextures surfaceTextures) noexcept override;
  void releaseFramebuffer() override;

 private:
  void createOrUpdateDefaultFramebuffer(const SurfaceTextures& surfaceTextures);
  std::string shaderForStage(ShaderStage stage, int computePipelineIndex = 0);

 private:
  RenderPassDesc renderPass_;
  std::shared_ptr<IFramebuffer> framebuffer_;

  std::shared_ptr<IRenderPipelineState> clothRenderPipelineState_;
  std::shared_ptr<IRenderPipelineState> obstacleRenderPipelineState_;

  std::shared_ptr<IComputePipelineState> updateVelocityPipelineState_, updatePositionPipelineState_,
      updateNormalPipelineState_;

  std::shared_ptr<IVertexInputState> clothVertexInput_;
  std::shared_ptr<IVertexInputState> obstacleVertexInput_;

  std::shared_ptr<IShaderStages> clothShaderStages_;
  std::shared_ptr<IShaderStages> obstacleShaderStages_;

  std::shared_ptr<IShaderStages> updateVelocityStages_, updatePositionStages_, updateNormalStages_;
  std::shared_ptr<IBuffer> clothVertexBuffer_, clothIndexBuffer_;
  std::shared_ptr<IBuffer> obstacleVertexBuffer_, obstacleIndexBuffer_;
  std::shared_ptr<IBuffer> uniformBuffer_;

  std::shared_ptr<IDepthStencilState> depthStencilState_;
};

inline void ClothSimulationSession::releaseFramebuffer() {
  framebuffer_ = nullptr;
}

} // namespace igl::shell
