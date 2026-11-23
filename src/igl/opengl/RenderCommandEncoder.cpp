/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/RenderCommandEncoder.h>

#include <igl/DepthStencilState.h>
#include <igl/Macros.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/opengl/Buffer.h>
#include <igl/opengl/CommandBuffer.h>
#include <igl/opengl/Framebuffer.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/PlatformDevice.h>
#include <igl/opengl/RenderCommandAdapter.h>
#include <igl/opengl/TimestampQueries.h>
#include <igl/opengl/UniformAdapter.h>

namespace igl::opengl {

namespace {
GLenum toGlPrimitive(PrimitiveType primitiveType) {
  GLenum result = GL_TRIANGLES;
  switch (primitiveType) {
  case PrimitiveType::Point:
    result = GL_POINTS;
    break;
  case PrimitiveType::Line:
    result = GL_LINES;
    break;
  case PrimitiveType::LineStrip:
    result = GL_LINE_STRIP;
    break;
  case PrimitiveType::Triangle:
    break;
  case PrimitiveType::TriangleStrip:
    result = GL_TRIANGLE_STRIP;
    break;
  }
  return result;
}

GLenum toGlType(IndexFormat format) {
  switch (format) {
  case IndexFormat::UInt8:
    return GL_UNSIGNED_BYTE;
  case IndexFormat::UInt16:
    return GL_UNSIGNED_SHORT;
  case IndexFormat::UInt32:
    return GL_UNSIGNED_INT;
  }
  IGL_UNREACHABLE_RETURN(GL_UNSIGNED_INT)
}

uint8_t getIndexByteSize(GLenum indexType) {
  switch (indexType) {
  case GL_UNSIGNED_BYTE:
    return 1u;
  case GL_UNSIGNED_SHORT:
    return 2u;
  case GL_UNSIGNED_INT:
    return 4u;
  default:
    break;
  }
  IGL_UNREACHABLE_RETURN(4u)
}

} // namespace

RenderCommandEncoder::RenderCommandEncoder(const std::shared_ptr<CommandBuffer>& commandBuffer) :
  IRenderCommandEncoder(commandBuffer),
  WithContext(static_cast<CommandBuffer&>(getCommandBuffer()).getContext()) {}

std::unique_ptr<RenderCommandEncoder> RenderCommandEncoder::create(
    const std::shared_ptr<CommandBuffer>& commandBuffer,
    const RenderPassDesc& renderPass,
    const std::shared_ptr<IFramebuffer>& framebuffer,
    const Dependencies& /*dependencies*/,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (!commandBuffer) {
    Result::setResult(outResult, Result::Code::ArgumentNull, "commandBuffer was null");
    return {};
  }

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  std::unique_ptr<RenderCommandEncoder> newEncoder(new RenderCommandEncoder(commandBuffer));
  newEncoder->beginEncoding(renderPass, framebuffer, outResult);
  return newEncoder;
}

RenderCommandEncoder::~RenderCommandEncoder() = default;

void RenderCommandEncoder::beginEncoding(const RenderPassDesc& renderPass,
                                         const std::shared_ptr<IFramebuffer>& framebuffer,
                                         Result* outResult) {
  IGL_PROFILER_FUNCTION();
  // Save caller state
  auto& context = getContext();

  scissorEnabled_ = (context.isEnabled(GL_SCISSOR_TEST) != 0u);
  context.disable(GL_SCISSOR_TEST); // only turn on if bindScissorRect is called

  auto& pool = context.getAdapterPool();
  if (pool.empty()) {
    Result result;
    adapter_ = RenderCommandAdapter::create(context, renderPass, framebuffer, &result);
    if (!result.isOk()) {
      if (outResult) {
        *outResult = result;
      }
      return;
    }
  } else {
    Result result;
    adapter_ = std::move(pool[pool.size() - 1]);
    pool.pop_back();
    adapter_->initialize(renderPass, framebuffer, &result);
    if (!result.isOk()) {
      if (outResult) {
        *outResult = result;
      }
      return;
    }
  }
  framebuffer_ = std::static_pointer_cast<Framebuffer>(framebuffer);
  resolveFramebuffer_ = framebuffer_->getResolveFramebuffer();

  // CommandBuffer has validated the subpasses
  if (!renderPass.subpasses.empty()) {
    numSubpasses_ = static_cast<uint32_t>(renderPass.subpasses.size());
    for (size_t i = 0; i != renderPass.subpasses.size(); ++i) {
      subpassColorMasks_[i] = renderPass.subpasses[i].colorAttachmentMask;
    }
    adapter_->beginSubpasses(static_cast<uint32_t>(framebuffer->getColorAttachmentIndices().size()),
                             subpassColorMasks_[0]);
  }

  // GPU timing: if the render pass descriptor contains a timestamp query,
  // start a GL_TIME_ELAPSED query to measure this render pass's GPU time.
  if (renderPass.timestampQuery.queries) {
    timestampQueries_ = renderPass.timestampQuery.queries;
    static_cast<TimestampQueries&>(*timestampQueries_)
        .beginElapsedQuery(renderPass.timestampQuery.slotIndex, framebuffer);
  }

  Result::setOk(outResult);
}

/**
 * @brief Ends render command encoding, restores GL state,
 *        and resolves MSAA framebuffers.
 *
 * Restores the GL_SCISSOR_TEST state saved during
 * beginEncoding(), disables GL_POLYGON_OFFSET_FILL, resets
 * depth bias, and delegates to the adapter's endEncoding().
 * The adapter is then returned to the context's pool for
 * reuse. If a GPU timestamp query was started, it is ended
 * here. If a resolve framebuffer is present, the primary
 * framebuffer is blitted to it for MSAA resolve.
 */
void RenderCommandEncoder::endEncoding() {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(currentSubpass_ + 1 == numSubpasses_,
                   "nextSubpass() must be called RenderPassDesc::subpasses.size() - 1 times before "
                   "endEncoding()");
  if (IGL_DEBUG_VERIFY(adapter_)) {
    // Restore caller state
    getContext().setEnabled(scissorEnabled_, GL_SCISSOR_TEST);

    // Disable depthBias
    getContext().setEnabled(false, GL_POLYGON_OFFSET_FILL);
    adapter_->setDepthBias(0.0f, 0.0f, 0.0f);

    adapter_->endEncoding();
    getContext().getAdapterPool().push_back(std::move(adapter_));

    // GPU timing: end the GL_TIME_ELAPSED query after all draw calls.
    if (timestampQueries_) {
      static_cast<TimestampQueries&>(*timestampQueries_).endElapsedQuery();
      timestampQueries_ = nullptr;
    }

    if (resolveFramebuffer_) {
      Result outResult;
      auto width = 0;
      auto height = 0;
      auto mask = 0;
      auto sizeMatch = true;
      if (resolveFramebuffer_->getColorAttachment(0)) {
        const auto colorDimensions = resolveFramebuffer_->getColorAttachment(0)->getDimensions();
        mask |= GL_COLOR_BUFFER_BIT;
        width = static_cast<int>(colorDimensions.width);
        height = static_cast<int>(colorDimensions.height);
      }
      if (resolveFramebuffer_->getDepthAttachment()) {
        mask |= GL_DEPTH_BUFFER_BIT;
        const auto depthDimensions = resolveFramebuffer_->getDepthAttachment()->getDimensions();
        if (width != 0 && static_cast<int>(depthDimensions.width) != width) {
          sizeMatch = false;
        }
        if (height != 0 && static_cast<int>(depthDimensions.height) != height) {
          sizeMatch = false;
        }
        width = static_cast<int>(depthDimensions.width);
        height = static_cast<int>(depthDimensions.height);
      }
      if (resolveFramebuffer_->getStencilAttachment()) {
        const auto stencilDimensions = resolveFramebuffer_->getStencilAttachment()->getDimensions();
        mask |= GL_STENCIL_BUFFER_BIT;
        if (width != 0 && stencilDimensions.width != width) {
          sizeMatch = false;
        }
        if (height != 0 && static_cast<int>(stencilDimensions.height) != height) {
          sizeMatch = false;
        }
        width = static_cast<int>(stencilDimensions.width);
        height = static_cast<int>(stencilDimensions.height);
      }
      IGL_DEBUG_ASSERT(mask != 0);

      if (sizeMatch) {
        igl::opengl::PlatformDevice::blitFramebuffer(framebuffer_,
                                                     0,
                                                     0,
                                                     width,
                                                     height,
                                                     resolveFramebuffer_,
                                                     0,
                                                     0,
                                                     width,
                                                     height,
                                                     mask,
                                                     getContext(),
                                                     &outResult);

        bool hasMRT = false;
        for (size_t i = 1; i < IGL_COLOR_ATTACHMENTS_MAX && !hasMRT; ++i) {
          if (!framebuffer_->getColorAttachment(i) || !resolveFramebuffer_->getColorAttachment(i)) {
            continue;
          }
          hasMRT = true;
        }

        if (hasMRT) {
          const FramebufferBindingGuard g(getContext());
          const GLuint readFboId = static_cast<Framebuffer&>(*framebuffer_).getId();
          const GLuint drawFboId = static_cast<Framebuffer&>(*resolveFramebuffer_).getId();

          getContext().bindFramebuffer(GL_READ_FRAMEBUFFER, readFboId);
          getContext().bindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFboId);

          for (size_t i = 1; i < IGL_COLOR_ATTACHMENTS_MAX; ++i) {
            if (!framebuffer_->getColorAttachment(i) ||
                !resolveFramebuffer_->getColorAttachment(i)) {
              continue;
            }

            const GLenum attachment = static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + i);
            getContext().readBuffer(attachment);

            // Note on glDrawBuffers semantics (GLES 3.0+): the i-th entry in bufs
            // must be either GL_COLOR_ATTACHMENTi or GL_NONE. We can NOT pass
            // {GL_COLOR_ATTACHMENT1} with n=1 to enable only attachment 1 — that
            // places ATTACHMENT1 at index 0 and triggers GL_INVALID_OPERATION.
            // Instead, pad with GL_NONE so the desired attachment lives at its
            // own index, e.g. {GL_NONE, ATTACHMENT1}.
            GLenum dbs[IGL_COLOR_ATTACHMENTS_MAX] = {};
            for (size_t k = 0; k < i; ++k) {
              dbs[k] = GL_NONE;
            }
            dbs[i] = attachment;
            getContext().drawBuffers(static_cast<GLsizei>(i + 1), dbs);
            igl::opengl::PlatformDevice::blitFramebuffer(framebuffer_,
                                                         0,
                                                         0,
                                                         width,
                                                         height,
                                                         resolveFramebuffer_,
                                                         0,
                                                         0,
                                                         width,
                                                         height,
                                                         GL_COLOR_BUFFER_BIT,
                                                         getContext(),
                                                         &outResult);
            if (!outResult.isOk()) {
              break;
            }
          }

          // Restore per-FBO READ_BUFFER / DRAW_BUFFERS so subsequent passes
          // that reuse these framebuffers (especially resolveFramebuffer_ as
          // an MRT render target) see all attachments enabled again.
          getContext().readBuffer(GL_COLOR_ATTACHMENT0);
          GLenum dbs[IGL_COLOR_ATTACHMENTS_MAX] = {};
          GLsizei dbCount = 0;
          for (size_t i = 0; i < IGL_COLOR_ATTACHMENTS_MAX; ++i) {
            if (resolveFramebuffer_->getColorAttachment(i)) {
              dbs[i] = static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + i);
              dbCount = static_cast<GLsizei>(i + 1);
            } else {
              dbs[i] = GL_NONE;
            }
          }
          if (dbCount > 0) {
            getContext().drawBuffers(dbCount, dbs);
          }
        }

      } else {
        IGL_DEBUG_ASSERT_NOT_REACHED();
      }
    }
    framebuffer_->unbind();
  }
}

void RenderCommandEncoder::nextSubpass() {
  IGL_PROFILER_FUNCTION();
  if (!IGL_DEBUG_VERIFY(adapter_)) {
    return;
  }
  if (currentSubpass_ + 1 >= numSubpasses_) {
    IGL_DEBUG_ABORT(
        "nextSubpass() was called more than RenderPassDesc::subpasses.size() - 1 times");
    return;
  }
  ++currentSubpass_;
  // No GL command is needed: GL_EXT_shader_framebuffer_fetch is coherent, so a fragment reads
  // everything that the earlier subpasses wrote at its pixel. Only the attachments to write change.
  adapter_->setSubpassColorAttachmentMask(subpassColorMasks_[currentSubpass_]);
}

void RenderCommandEncoder::pushDebugGroupLabel(const char* label, const Color& /*color*/) const {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(adapter_);
  IGL_DEBUG_ASSERT(label != nullptr && *label);
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DebugMessage)) {
    getContext().pushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, label);
  } else {
    IGL_LOG_ERROR_ONCE(
        "RenderCommandEncoder::pushDebugGroupLabel not supported in this context!\n");
  }
}

void RenderCommandEncoder::insertDebugEventLabel(const char* label, const Color& /*color*/) const {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(adapter_);
  IGL_DEBUG_ASSERT(label != nullptr && *label);
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DebugMessage)) {
    getContext().debugMessageInsert(
        GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 0, GL_DEBUG_SEVERITY_LOW, -1, label);
  } else {
    IGL_LOG_ERROR_ONCE(
        "RenderCommandEncoder::insertDebugEventLabel not supported in this context!\n");
  }
}

void RenderCommandEncoder::popDebugGroupLabel() const {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(adapter_);
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DebugMessage)) {
    getContext().popDebugGroup();
  } else {
    IGL_LOG_ERROR_ONCE("RenderCommandEncoder::popDebugGroupLabel not supported in this context!\n");
  }
}

void RenderCommandEncoder::bindViewport(const Viewport& viewport) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setViewport(viewport);
  }
}

void RenderCommandEncoder::bindScissorRect(const ScissorRect& rect) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setScissorRect(rect);
  }
}

void RenderCommandEncoder::bindRenderPipelineState(
    const std::shared_ptr<IRenderPipelineState>& pipelineState) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setPipelineState(pipelineState);
  }
}

void RenderCommandEncoder::bindDepthStencilState(
    const std::shared_ptr<IDepthStencilState>& depthStencilState) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setDepthStencilState(depthStencilState);
  }
}

void RenderCommandEncoder::bindUniform(const UniformDesc& uniformDesc, const void* data) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(uniformDesc.location >= 0,
                   "Invalid location passed to bindUniformBuffer: %d",
                   uniformDesc.location);
  IGL_DEBUG_ASSERT(data != nullptr, "Data cannot be null");
  if (IGL_DEBUG_VERIFY(adapter_) && data) {
    adapter_->setUniform(uniformDesc, data);
  }
}

void RenderCommandEncoder::bindBuffer(uint32_t index,
                                      uint8_t bindTarget,
                                      IBuffer* buffer,
                                      size_t offset,
                                      size_t bufferSize) {
  bindBuffer(index, buffer, offset, bufferSize);
}

void RenderCommandEncoder::bindBuffer(uint32_t index,
                                      IBuffer* buffer,
                                      size_t offset,
                                      size_t bufferSize) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_) && buffer) {
    auto* glBuffer = static_cast<Buffer*>(buffer);
    auto bufferType = glBuffer->getType();

    if (bufferType == Buffer::Type::Uniform) {
      IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    } else if (bufferType == Buffer::Type::UniformBlock) {
      adapter_->setUniformBuffer(glBuffer, offset, bufferSize, index);
    } else if (glBuffer->getBufferType() & BufferDesc::BufferTypeBits::Storage) {
      adapter_->setStorageBuffer(glBuffer, offset, index);
    }
  }
}

void RenderCommandEncoder::bindVertexBuffer(uint32_t index,
                                            IBuffer& buffer,
                                            size_t bufferOffset,
                                            size_t attributeStride) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    Buffer& glBuffer = static_cast<Buffer&>(buffer);

    IGL_DEBUG_ASSERT(glBuffer.getType() == Buffer::Type::Attribute);

    adapter_->setVertexBuffer(glBuffer, bufferOffset, static_cast<int>(index), attributeStride);
  }
}

void RenderCommandEncoder::bindIndexBuffer(IBuffer& buffer,
                                           IndexFormat format,
                                           size_t bufferOffset) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    indexType_ = toGlType(format);
    indexBufferOffset_ = reinterpret_cast<void*>(bufferOffset); // NOLINT(performance-no-int-to-ptr)
    adapter_->setIndexBuffer(static_cast<Buffer&>(buffer));
  }
}

void RenderCommandEncoder::bindBytes(size_t /*index*/,
                                     uint8_t /*target*/,
                                     const void* /*data*/,
                                     size_t /*length*/) {
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

void RenderCommandEncoder::bindPushConstants(const void* /*data*/,
                                             size_t /*length*/,
                                             size_t /*offset*/) {
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void RenderCommandEncoder::bindSamplerState(size_t index,
                                            uint8_t bindTarget,
                                            ISamplerState* samplerState) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    if ((bindTarget & BindTarget::kVertex) != 0) {
      adapter_->setVertexSamplerState(samplerState, index);
    }
    if ((bindTarget & BindTarget::kFragment) != 0) {
      adapter_->setFragmentSamplerState(samplerState, index);
    }
  }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void RenderCommandEncoder::bindTexture(size_t index, uint8_t bindTarget, ITexture* texture) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    if ((bindTarget & BindTarget::kVertex) != 0) {
      adapter_->setVertexTexture(texture, index);
    }
    if ((bindTarget & BindTarget::kFragment) != 0) {
      adapter_->setFragmentTexture(texture, index);
    }
  }
}

void RenderCommandEncoder::bindTexture(size_t index, ITexture* texture) {
  bindTexture(index, BindTarget::kFragment, texture);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void RenderCommandEncoder::draw(size_t vertexCount,
                                uint32_t instanceCount,
                                uint32_t firstVertex,
                                uint32_t baseInstance) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  IGL_PROFILER_FUNCTION();
  (void)baseInstance;

  IGL_DEBUG_ASSERT(baseInstance == 0, "Instancing is not implemented");

  if (IGL_DEBUG_VERIFY(adapter_)) {
    getCommandBuffer().incrementCurrentDrawCount();
    auto mode = toGlPrimitive(adapter_->pipelineState().getRenderPipelineDesc().topology);
    if (instanceCount > 1) {
      adapter_->drawArraysInstanced(mode,
                                    static_cast<GLsizei>(firstVertex),
                                    static_cast<GLsizei>(vertexCount),
                                    static_cast<GLsizei>(instanceCount));
    } else {
      adapter_->drawArrays(
          mode, static_cast<GLsizei>(firstVertex), static_cast<GLsizei>(vertexCount));
    }
  }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void RenderCommandEncoder::drawIndexed(size_t indexCount,
                                       uint32_t instanceCount,
                                       uint32_t firstIndex,
                                       int32_t vertexOffset,
                                       uint32_t baseInstance) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  IGL_PROFILER_FUNCTION();
  (void)vertexOffset;
  (void)baseInstance;

  IGL_DEBUG_ASSERT(vertexOffset == 0, "vertexOffset is not implemented");
  IGL_DEBUG_ASSERT(baseInstance == 0, "Instancing is not implemented");
  IGL_DEBUG_ASSERT(indexType_, "No index buffer bound");

  const size_t indexOffsetBytes = static_cast<size_t>(firstIndex) * getIndexByteSize(indexType_);

  if (IGL_DEBUG_VERIFY(adapter_ && indexType_)) {
    getCommandBuffer().incrementCurrentDrawCount();
    auto mode = toGlPrimitive(adapter_->pipelineState().getRenderPipelineDesc().topology);
    if (instanceCount > 1) {
      adapter_->drawElementsInstanced(mode,
                                      static_cast<GLsizei>(indexCount),
                                      indexType_,
                                      (uint8_t*)indexBufferOffset_ + indexOffsetBytes,
                                      instanceCount);
    } else {
      adapter_->drawElements(mode,
                             static_cast<GLsizei>(indexCount),
                             indexType_,
                             (uint8_t*)indexBufferOffset_ + indexOffsetBytes);
    }
  }
}

void RenderCommandEncoder::drawMeshTasks(const Dimensions& threadgroupsPerGrid,
                                         const Dimensions& threadsPerTaskThreadgroup,
                                         const Dimensions& threadsPerMeshThreadgroup) {
  (void)threadgroupsPerGrid;
  (void)threadsPerTaskThreadgroup;
  (void)threadsPerMeshThreadgroup;

  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void RenderCommandEncoder::multiDrawIndirect(IBuffer& indirectBuffer,
                                             size_t indirectBufferOffset,
                                             uint32_t drawCount,
                                             uint32_t stride) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    getCommandBuffer().incrementCurrentDrawCount();
    const auto mode = toGlPrimitive(adapter_->pipelineState().getRenderPipelineDesc().topology);
    const auto* indirectBufferOffsetPtr =
        reinterpret_cast<uint8_t*>(indirectBufferOffset); // NOLINT(performance-no-int-to-ptr)
    const GLsizei effectiveStride = stride ? stride : 16u; // sizeof(DrawArraysIndirectCommand)
    if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::MultiDrawIndirect)) {
      adapter_->multiDrawArraysIndirect(mode,
                                        static_cast<Buffer&>(indirectBuffer),
                                        indirectBufferOffsetPtr,
                                        drawCount,
                                        effectiveStride);
    } else {
      for (uint32_t i = 0; i != drawCount; i++) {
        adapter_->drawArraysIndirect(
            mode, static_cast<Buffer&>(indirectBuffer), indirectBufferOffsetPtr);
        indirectBufferOffsetPtr += effectiveStride;
      }
    }
  }
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void RenderCommandEncoder::multiDrawIndexedIndirect(IBuffer& indirectBuffer,
                                                    size_t indirectBufferOffset,
                                                    uint32_t drawCount,
                                                    uint32_t stride) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(indexType_, "No index buffer bound");

  if (IGL_DEBUG_VERIFY(adapter_ && indexType_)) {
    getCommandBuffer().incrementCurrentDrawCount();
    const auto mode = toGlPrimitive(adapter_->pipelineState().getRenderPipelineDesc().topology);
    const auto* indirectBufferOffsetPtr =
        reinterpret_cast<uint8_t*>(indirectBufferOffset); // NOLINT(performance-no-int-to-ptr)
    const GLsizei effectiveStride = stride ? stride : 20u; // sizeof(DrawElementsIndirectCommand)
    if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::MultiDrawIndirect)) {
      adapter_->multiDrawElementsIndirect(mode,
                                          indexType_,
                                          static_cast<Buffer&>(indirectBuffer),
                                          indirectBufferOffsetPtr,
                                          drawCount,
                                          effectiveStride);
    } else {
      for (uint32_t i = 0; i != drawCount; i++) {
        adapter_->drawElementsIndirect(
            mode, indexType_, static_cast<Buffer&>(indirectBuffer), indirectBufferOffsetPtr);
        indirectBufferOffsetPtr += effectiveStride;
      }
    }
  }
}

void RenderCommandEncoder::setStencilReferenceValue(uint32_t value) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setStencilReferenceValue(value);
  }
}

void RenderCommandEncoder::setBlendColor(const Color& color) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setBlendColor(color);
  }
}

void RenderCommandEncoder::setCullMode(CullMode cullMode) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setCullMode(cullMode);
  }
}

void RenderCommandEncoder::setDepthBias(float depthBias, float slopeScale, float clamp) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setDepthBias(depthBias, slopeScale, clamp);
  }
}

void RenderCommandEncoder::setFrontFacingWinding(WindingMode frontFaceWinding) {
  IGL_PROFILER_FUNCTION();
  if (IGL_DEBUG_VERIFY(adapter_)) {
    adapter_->setFrontFacingWinding(frontFaceWinding);
  }
}

void RenderCommandEncoder::bindBindGroup(BindGroupTextureHandle handle) {
  IGL_PROFILER_FUNCTION();
  if (handle.empty()) {
    return;
  }

  const BindGroupTextureDesc* desc = getContext().bindGroupTexturesPool.get(handle);

  for (uint32_t i = 0; i != IGL_TEXTURE_SAMPLERS_MAX; i++) {
    if (desc->textures[i]) {
      IGL_DEBUG_ASSERT(desc->samplers[i]);
      bindTexture(i, BindTarget::kAllGraphics, desc->textures[i].get());
      bindSamplerState(i, BindTarget::kAllGraphics, desc->samplers[i].get());
    }
  }
}

void RenderCommandEncoder::bindBindGroup(BindGroupBufferHandle handle,
                                         uint32_t numDynamicOffsets,
                                         const uint32_t* dynamicOffsets) {
  IGL_PROFILER_FUNCTION();
  if (handle.empty()) {
    return;
  }

  const BindGroupBufferDesc* desc = getContext().bindGroupBuffersPool.get(handle);

  uint32_t dynamicOffset = 0;

  for (uint32_t i = 0; i != IGL_UNIFORM_BLOCKS_BINDING_MAX; i++) {
    if (desc->buffers[i]) {
      if (desc->isDynamicBufferMask & (1 << i)) {
        IGL_DEBUG_ASSERT(dynamicOffsets, "No dynamic offsets provided");
        IGL_DEBUG_ASSERT(dynamicOffset < numDynamicOffsets, "Not enough dynamic offsets provided");
        bindBuffer(i,
                   desc->buffers[i].get(),
                   desc->offset[i] + dynamicOffsets[dynamicOffset++],
                   desc->size[i]);
      } else {
        bindBuffer(i, desc->buffers[i].get(), desc->offset[i], desc->size[i]);
      }
    }
  }

  IGL_DEBUG_ASSERT(dynamicOffset == numDynamicOffsets, "Not all dynamic offsets were consumed");
}

} // namespace igl::opengl
