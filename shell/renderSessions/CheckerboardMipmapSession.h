/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#pragma once

#include <cmath> // IWYU pragma: keep
#include <memory>
#include <utility>
#include <shell/shared/platform/Platform.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <igl/IGL.h>

namespace igl::shell {

#define ROTATE_PLANE 1

class CheckerboardMipmapSession : public RenderSession {
 public:
  explicit CheckerboardMipmapSession(std::shared_ptr<Platform> platform) :
    RenderSession(std::move(platform)) {}
  void initialize() noexcept override;
  void update(SurfaceTextures surfaceTextures) noexcept override;
  void releaseFramebuffer() override;

 private:
  RenderPassDesc renderPass_;
  std::shared_ptr<IRenderPipelineState> pipelineState_;
  std::shared_ptr<IVertexInputState> vertexInput0_;
  std::shared_ptr<IShaderStages> shaderStages_;
  std::shared_ptr<IBuffer> vb0_, ib0_;
  std::shared_ptr<ITexture> tex0_;
  std::shared_ptr<ISamplerState> samp0_;
  std::shared_ptr<IFramebuffer> framebuffer_;
  std::shared_ptr<IBuffer> mvpUniformBuffer_;

  // Initial angle of the plane, so that it starts at an angle
  float planeAngle_ = M_PI / 4.0f;
};

inline void CheckerboardMipmapSession::releaseFramebuffer() {
  framebuffer_ = nullptr;
}

} // namespace igl::shell
