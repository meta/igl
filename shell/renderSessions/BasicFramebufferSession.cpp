/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/BasicFramebufferSession.h>

#include <shell/shared/platform/Platform.h>
#include <shell/shared/renderSession/ShellParams.h>
#include <igl/Common.h>

namespace igl::shell {

void BasicFramebufferSession::initialize() noexcept {
  // Create commandQueue
  commandQueue_ = getPlatform().getDevice().createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  renderPass_ = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = getPreferredClearColor(),
      }},
  };
}

void BasicFramebufferSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }

  // Create/update framebuffer
  if (framebuffer_ == nullptr) {
    framebuffer_ = getPlatform().getDevice().createFramebuffer(
        {
            .colorAttachments = {{.texture = surfaceTextures.color}},
        },
        nullptr);
    IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
  } else {
    framebuffer_->updateDrawable(surfaceTextures.color);
  }

  // Create/submit command buffer
  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  const auto commands = buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(commands != nullptr);
  commands->endEncoding();
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  if (shellParams().shouldPresent) {
    buffer->present(framebuffer_->getColorAttachment(0));
  }
  commandQueue_->submit(*buffer);
}

} // namespace igl::shell
