/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/CommandBuffer.h>

#include <igl/Macros.h>
#include <igl/RenderPass.h>
#include <igl/opengl/Buffer.h>
#include <igl/opengl/ComputeCommandEncoder.h>
#include <igl/opengl/Framebuffer.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/RenderCommandEncoder.h>

namespace igl::opengl {

CommandBuffer::CommandBuffer(std::shared_ptr<IContext> context, CommandBufferDesc desc) :
  ICommandBuffer(std::move(desc)), context_(std::move(context)) {}

CommandBuffer::~CommandBuffer() = default;

std::unique_ptr<IRenderCommandEncoder> CommandBuffer::createRenderCommandEncoder(
    const RenderPassDesc& renderPass,
    const std::shared_ptr<IFramebuffer>& framebuffer,
    const Dependencies& dependencies,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const bool hasSubpasses = !renderPass.subpasses.empty();
  if (hasSubpasses) {
    if (!getContext().deviceFeatures().hasFeature(DeviceFeatures::Subpasses)) {
      Result::setResult(outResult, Result::Code::Unsupported, "Subpasses are not supported");
      return nullptr;
    }
    if (!framebuffer) {
      Result::setResult(outResult, Result::Code::ArgumentNull, "The framebuffer is null");
      return nullptr;
    }
    // GL_EXT_shader_framebuffer_fetch does not define reads from the framebuffers of
    // GL_OVR_multiview
    if (framebuffer->getMode() != FramebufferMode::Mono) {
      Result::setResult(
          outResult, Result::Code::Unsupported, "Subpasses support only mono framebuffers");
      return nullptr;
    }
    // The color attachment of CurrentFramebuffer is a placeholder that validate() cannot query,
    // and the window system's framebuffer ignores every attachment after the first
    if (!static_cast<const Framebuffer&>(*framebuffer).hasTextureAttachments()) {
      Result::setResult(outResult,
                        Result::Code::Unsupported,
                        "Subpasses need a framebuffer whose color attachments are textures");
      return nullptr;
    }
    const Result validation = renderPass.validate(*framebuffer);
    if (!validation.isOk()) {
      Result::setResult(outResult, validation);
      return nullptr;
    }
  }
#if IGL_DEBUG_ABORT_ENABLED
  if (!hasSubpasses && framebuffer) {
    const Result validation = renderPass.validate(*framebuffer);
    if (!validation.isOk()) {
      IGL_LOG_ERROR_ONCE("Invalid RenderPassDesc: %s\n", validation.message.c_str());
    }
  }
#endif
  return RenderCommandEncoder::create(
      shared_from_this(), renderPass, framebuffer, dependencies, outResult);
}

std::unique_ptr<IComputeCommandEncoder> CommandBuffer::createComputeCommandEncoder() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  return std::make_unique<ComputeCommandEncoder>(shared_from_this()->getContext());
}

void CommandBuffer::present(const std::shared_ptr<ITexture>& surface) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_PRESENT);
  context_->present(surface);
}

void CommandBuffer::waitUntilScheduled() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
  context_->flush();
}

void CommandBuffer::waitUntilCompleted() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
  context_->finish();
}

void CommandBuffer::pushDebugGroupLabel(const char* label, const Color& /*color*/) const {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(label != nullptr && *label);
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DebugMessage)) {
    getContext().pushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, label);
  } else {
    IGL_LOG_ERROR_ONCE("CommandBuffer::pushDebugGroupLabel not supported in this context!\n");
  }
}

void CommandBuffer::popDebugGroupLabel() const {
  IGL_PROFILER_FUNCTION();
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DebugMessage)) {
    getContext().popDebugGroup();
  } else {
    IGL_LOG_ERROR_ONCE("CommandBuffer::popDebugGroupLabel not supported in this context!\n");
  }
}

void CommandBuffer::copyBuffer(IBuffer& src,
                               IBuffer& dst,
                               uint64_t srcOffset,
                               uint64_t dstOffset,
                               uint64_t size) {
  IGL_PROFILER_FUNCTION();
  IContext& ctx = getContext();

  if (!ctx.deviceFeatures().hasFeature(DeviceFeatures::CopyBuffer)) {
    IGL_LOG_ERROR_ONCE("CommandBuffer::copyBuffer() not supported in this context!\n");
    return;
  }

  auto& srcBuffer = static_cast<ArrayBuffer&>(src);
  auto& dstBuffer = static_cast<ArrayBuffer&>(dst);

  ctx.bindBuffer(GL_COPY_READ_BUFFER, srcBuffer.getId());
  ctx.bindBuffer(GL_COPY_WRITE_BUFFER, dstBuffer.getId());
  ctx.copyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, srcOffset, dstOffset, size);
  ctx.bindBuffer(GL_COPY_READ_BUFFER, 0);
  ctx.bindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void CommandBuffer::copyTextureToBuffer(ITexture& src,
                                        IBuffer& dst,
                                        uint64_t dstOffset,
                                        uint32_t level,
                                        uint32_t layer) {
  IGL_PROFILER_FUNCTION();
  (void)src;
  (void)dst;
  (void)dstOffset;
  (void)level;
  (void)layer;
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

IContext& CommandBuffer::getContext() const {
  return *context_;
}

} // namespace igl::opengl
