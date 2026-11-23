/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/RenderCommandAdapter.h>

#include <algorithm>
#include <igl/Common.h>
#include <igl/Macros.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/opengl/Buffer.h>
#include <igl/opengl/DepthStencilState.h>
#include <igl/opengl/Framebuffer.h>
#include <igl/opengl/GLIncludes.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/RenderPipelineState.h>
#include <igl/opengl/SamplerState.h>
#include <igl/opengl/Shader.h>
#include <igl/opengl/Texture.h>
#include <igl/opengl/UniformAdapter.h>
#include <igl/opengl/VertexArrayObject.h>

#define SET_DIRTY(dirtyMap, index) dirtyMap.set(index)
#define CLEAR_DIRTY(dirtyMap, index) dirtyMap.reset(index)
#define IS_DIRTY(dirtyMap, index) dirtyMap[index]

namespace igl::opengl {
RenderCommandAdapter::RenderCommandAdapter(IContext& context) :
  WithContext(context),
  uniformAdapter_(UniformAdapter(context, UniformAdapter::PipelineType::Render)) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  useVAO_ = context.deviceFeatures().hasInternalFeature(InternalFeatures::VertexArrayObject);
  if (useVAO_) {
    activeVAO_ = std::make_shared<VertexArrayObject>(getContext());
    activeVAO_->create();
  }
}

std::unique_ptr<RenderCommandAdapter> RenderCommandAdapter::create(
    IContext& context,
    const RenderPassDesc& renderPass,
    const std::shared_ptr<IFramebuffer>& framebuffer,
    Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  std::unique_ptr<RenderCommandAdapter> newAdapter(new RenderCommandAdapter(context));
  newAdapter->initialize(renderPass, framebuffer, outResult);
  return newAdapter;
}

void RenderCommandAdapter::initialize(const RenderPassDesc& renderPass,
                                      const std::shared_ptr<IFramebuffer>& framebuffer,
                                      Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  numSubpassColorAttachments_ = 0;
  subpassColorAttachmentMask_ = 0;
  subpassColorAttachmentsApplied_ = false;
  if (!IGL_DEBUG_VERIFY(framebuffer)) {
    Result::setResult(outResult, Result::Code::ArgumentNull, "framebuffer is null");
    return;
  }
  if (activeVAO_) {
    if (!IGL_DEBUG_VERIFY(activeVAO_->isValid())) {
      Result::setResult(outResult, Result::Code::RuntimeError, "Vertex array object is invalid");
      return;
    }
    activeVAO_->bind();
  }
  const auto& openglFramebuffer = static_cast<const Framebuffer&>(*framebuffer);
  openglFramebuffer.bind(renderPass);

  auto viewport = openglFramebuffer.getViewport();
  IGL_DEBUG_ASSERT(!(viewport.width < 0.f) && !(viewport.height < 0.f));
  setViewport(viewport);
  Result::setOk(outResult);
}

void RenderCommandAdapter::setViewport(const Viewport& viewport) {
  IGL_PROFILER_FUNCTION();
  getContext().viewport(static_cast<GLint>(viewport.x),
                        static_cast<GLint>(viewport.y),
                        static_cast<GLint>(viewport.width),
                        static_cast<GLint>(viewport.height));
}

void RenderCommandAdapter::setScissorRect(const ScissorRect& rect) {
  IGL_PROFILER_FUNCTION();
  const bool scissorEnabled = !rect.isNull();
  getContext().setEnabled(scissorEnabled, GL_SCISSOR_TEST);
  if (scissorEnabled) {
    getContext().scissor(rect.x, rect.y, rect.width, rect.height);
  }
}

void RenderCommandAdapter::setDepthStencilState(
    const std::shared_ptr<IDepthStencilState>& newValue) {
  IGL_PROFILER_FUNCTION();
  depthStencilState_ = newValue;
  setDirty(StateMask::DepthStencil);
}

void RenderCommandAdapter::setStencilReferenceValue(uint32_t value) {
  IGL_PROFILER_FUNCTION();
  frontStencilReferenceValue_ = value;
  backStencilReferenceValue_ = value;

  setDirty(StateMask::DepthStencil);
}

void RenderCommandAdapter::setBlendColor(const Color& color) {
  IGL_PROFILER_FUNCTION();
  getContext().blendColor(color.r, color.g, color.b, color.a);
}

void RenderCommandAdapter::setCullMode(CullMode cullMode) {
  IGL_PROFILER_FUNCTION();
  cullMode_ = cullMode;
  setDirty(StateMask::CullMode);
}

void RenderCommandAdapter::setFrontFacingWinding(WindingMode mode) {
  IGL_PROFILER_FUNCTION();
  windingMode_ = mode;
  setDirty(StateMask::FrontFaceWinding);
}

void RenderCommandAdapter::setDepthBias(float depthBias, float slopeScale, float clamp) {
  IGL_PROFILER_FUNCTION();
  getContext().setEnabled(true, GL_POLYGON_OFFSET_FILL);
  getContext().polygonOffsetClamp(slopeScale, depthBias, clamp);
}

void RenderCommandAdapter::clearVertexBuffers() {
  vertexBuffersDirty_.reset();
}

void RenderCommandAdapter::setVertexBuffer(Buffer& buffer,
                                           size_t offset,
                                           size_t index,
                                           size_t stride,
                                           Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(index < IGL_BUFFER_BINDINGS_MAX,
                   "Buffer index is beyond max, may want to increase limit");
  if (index < IGL_BUFFER_BINDINGS_MAX) {
    vertexBuffers_[index] = {.resource = &buffer, .offset = offset, .stride = stride};
    SET_DIRTY(vertexBuffersDirty_, index);
    Result::setOk(outResult);
  } else {
    Result::setResult(outResult, Result::Code::ArgumentInvalid);
  }
}

void RenderCommandAdapter::setIndexBuffer(Buffer& buffer) {
  bindBufferWithShaderStorageBufferOverride(buffer, GL_ELEMENT_ARRAY_BUFFER);
}

void RenderCommandAdapter::clearUniformBuffers() {
  uniformAdapter_.clearUniformBuffers();
}

void RenderCommandAdapter::setUniform(const UniformDesc& uniformDesc,
                                      const void* IGL_NONNULL data,
                                      Result* IGL_NULLABLE outResult) {
  // `outResult` is forwarded to UniformAdapter::setUniform(), which null-checks it via
  // Result::setResult()/setOk(); the dereference is guarded, so this is a false positive.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  uniformAdapter_.setUniform(uniformDesc, data, outResult);
}

void RenderCommandAdapter::setUniformBuffer(Buffer* IGL_NULLABLE buffer,
                                            size_t offset,
                                            size_t size,
                                            uint32_t index,
                                            Result* IGL_NULLABLE outResult) {
  // `buffer` and `outResult` are forwarded to UniformAdapter::setUniformBuffer(), which guards
  // both (`if (... && buffer)` and Result::setResult()/setOk()); these are false positives.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  uniformAdapter_.setUniformBuffer(buffer, offset, size, index, outResult);
}

void RenderCommandAdapter::setStorageBuffer(Buffer* buffer, size_t offset, uint32_t index) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(index < IGL_BUFFER_BINDINGS_MAX,
                   "Buffer index is beyond max, may want to increase limit");
  if (index < IGL_BUFFER_BINDINGS_MAX) {
    storageBuffers_[index] = {.resource = buffer, .offset = offset};
    SET_DIRTY(storageBuffersDirty_, index);
  }
}

void RenderCommandAdapter::clearVertexTexture() {
  IGL_PROFILER_FUNCTION();
  vertexTextureStates_ = TextureStates();
  vertexTextureStatesDirty_.reset();
}

void RenderCommandAdapter::setVertexTexture(ITexture* IGL_NULLABLE texture,
                                            size_t index,
                                            Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if (!IGL_DEBUG_VERIFY(index < IGL_TEXTURE_SAMPLERS_MAX)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid);
    return;
  }
  if (vertexTextureStates_[index].first != texture) {
    vertexTextureStates_[index].first = texture;
    SET_DIRTY(vertexTextureStatesDirty_, index);
  }
  Result::setOk(outResult);
}

void RenderCommandAdapter::setVertexSamplerState(ISamplerState* IGL_NULLABLE samplerState,
                                                 size_t index,
                                                 Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if (!IGL_DEBUG_VERIFY(index < IGL_TEXTURE_SAMPLERS_MAX)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid);
    return;
  }
  if (vertexTextureStates_[index].second != samplerState) {
    vertexTextureStates_[index].second = samplerState;
    SET_DIRTY(vertexTextureStatesDirty_, index);
  }
  Result::setOk(outResult);
}

void RenderCommandAdapter::clearFragmentTexture() {
  IGL_PROFILER_FUNCTION();
  fragmentTextureStates_ = TextureStates();
  fragmentTextureStatesDirty_.reset();
}

void RenderCommandAdapter::setFragmentTexture(ITexture* IGL_NULLABLE texture,
                                              size_t index,
                                              Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if (!IGL_DEBUG_VERIFY(index < IGL_TEXTURE_SAMPLERS_MAX)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid);
    return;
  }
  if (fragmentTextureStates_[index].first != texture) {
    fragmentTextureStates_[index].first = texture;
    SET_DIRTY(fragmentTextureStatesDirty_, index);
  }
  Result::setOk(outResult);
}

void RenderCommandAdapter::setFragmentSamplerState(ISamplerState* IGL_NULLABLE samplerState,
                                                   size_t index,
                                                   Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if (!IGL_DEBUG_VERIFY(index < IGL_TEXTURE_SAMPLERS_MAX)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid);
    return;
  }
  if (fragmentTextureStates_[index].second != samplerState) {
    fragmentTextureStates_[index].second = samplerState;
    SET_DIRTY(fragmentTextureStatesDirty_, index);
  }
  Result::setOk(outResult);
}

// When pipelineState is modified, all dependent resources are cleared
void RenderCommandAdapter::clearDependentResources(
    const std::shared_ptr<IRenderPipelineState>& newValue,
    Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  auto* curStateOpenGL = static_cast<RenderPipelineState*>(pipelineState_.get());
  if (!IGL_DEBUG_VERIFY(curStateOpenGL)) {
    Result::setResult(outResult, Result::Code::RuntimeError, "pipeline state is null");
    return;
  }

  auto* newStateOpenGL = static_cast<RenderPipelineState*>(newValue.get());

  if (!newStateOpenGL || !curStateOpenGL->matchesShaderProgram(*newStateOpenGL)) {
    // Don't use previously set resources. Uniforms/texture locations not same between programs
    uniformAdapter_.clearUniformBuffers();
    clearVertexTexture();
    clearFragmentTexture();
  }

  if (curStateOpenGL && newStateOpenGL) {
    newStateOpenGL->savePrevPipelineStateAttributesLocations(*curStateOpenGL);
  }

  if (!newStateOpenGL || !curStateOpenGL->matchesVertexInputState(*newStateOpenGL)) {
    // Don't reuse previously set vertex buffers.
    clearVertexBuffers();
  }
  Result::setOk(outResult);
}

void RenderCommandAdapter::setPipelineState(const std::shared_ptr<IRenderPipelineState>& newValue,
                                            Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  Result::setOk(outResult);
  if (pipelineState_) {
    clearDependentResources(newValue, outResult); // Only clear if pipeline state was previously set
  }
  pipelineState_ = newValue;
  // Reset the effective cull mode / front-face to what the new pipeline bakes in,
  // matching Metal's bindRenderPipelineState() semantics. willDraw() will apply
  // them once.
  if (auto* newStateOpenGL = static_cast<RenderPipelineState*>(pipelineState_.get())) {
    cullMode_ = newStateOpenGL->getCullMode();
    setDirty(StateMask::CullMode);
    windingMode_ = newStateOpenGL->getWindingMode();
    setDirty(StateMask::FrontFaceWinding);
  }
  setDirty(StateMask::PIPELINE);
}

void RenderCommandAdapter::beginSubpasses(uint32_t numColorAttachments,
                                          uint8_t colorAttachmentMask) {
  IGL_PROFILER_FUNCTION();
  numSubpassColorAttachments_ = numColorAttachments;
  setSubpassColorAttachmentMask(colorAttachmentMask);
}

void RenderCommandAdapter::setSubpassColorAttachmentMask(uint8_t colorAttachmentMask) {
  IGL_PROFILER_FUNCTION();
  subpassColorAttachmentMask_ = colorAttachmentMask;
  setDirty(StateMask::SubpassColorAttachments);
}

void RenderCommandAdapter::drawArrays(GLenum mode, GLint first, GLsizei count) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  getContext().drawArrays(toMockWireframeMode(mode), first, count);
  didDraw();
}

void RenderCommandAdapter::drawArraysIndirect(GLenum mode,
                                              Buffer& indirectBuffer,
                                              const GLvoid* IGL_NULLABLE indirectBufferOffset) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DrawArraysIndirect)) {
    bindBufferWithShaderStorageBufferOverride(indirectBuffer, GL_DRAW_INDIRECT_BUFFER);
    getContext().drawArraysIndirect(toMockWireframeMode(mode), indirectBufferOffset);
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }
  didDraw();
}

void RenderCommandAdapter::drawArraysInstanced(GLenum mode,
                                               GLint first,
                                               GLsizei count,
                                               GLsizei instancecount) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  if (getContext().deviceFeatures().hasFeature(DeviceFeatures::DrawInstanced)) {
    getContext().drawArraysInstanced(toMockWireframeMode(mode), first, count, instancecount);
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }
  didDraw();
}

void RenderCommandAdapter::drawElements(GLenum mode,
                                        GLsizei indexCount,
                                        GLenum indexType,
                                        const GLvoid* IGL_NULLABLE indexOffset) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  getContext().drawElements(toMockWireframeMode(mode), indexCount, indexType, indexOffset);
  didDraw();
}

void RenderCommandAdapter::drawElementsInstanced(GLenum mode,
                                                 GLsizei indexCount,
                                                 GLenum indexType,
                                                 const GLvoid* IGL_NULLABLE indexOffset,
                                                 GLsizei instancecount) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  if (getContext().deviceFeatures().hasFeature(DeviceFeatures::DrawInstanced)) {
    getContext().drawElementsInstanced(
        toMockWireframeMode(mode), indexCount, indexType, indexOffset, instancecount);
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }
  didDraw();
}

void RenderCommandAdapter::drawElementsIndirect(GLenum mode,
                                                GLenum indexType,
                                                Buffer& indirectBuffer,
                                                const GLvoid* IGL_NULLABLE indirectBufferOffset) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  if (getContext().deviceFeatures().hasFeature(DeviceFeatures::DrawIndexedIndirect)) {
    bindBufferWithShaderStorageBufferOverride(indirectBuffer, GL_DRAW_INDIRECT_BUFFER);
    getContext().drawElementsIndirect(toMockWireframeMode(mode), indexType, indirectBufferOffset);
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }
  didDraw();
}

void RenderCommandAdapter::multiDrawArraysIndirect(GLenum mode,
                                                   Buffer& indirectBuffer,
                                                   const GLvoid* IGL_NULLABLE indirectBufferOffset,
                                                   GLsizei drawcount,
                                                   GLsizei stride) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::MultiDrawIndirect)) {
    bindBufferWithShaderStorageBufferOverride(indirectBuffer, GL_DRAW_INDIRECT_BUFFER);
    getContext().multiDrawArraysIndirect(
        toMockWireframeMode(mode), indirectBufferOffset, drawcount, stride);
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }
  didDraw();
}

void RenderCommandAdapter::multiDrawElementsIndirect(GLenum mode,
                                                     GLenum indexType,
                                                     Buffer& indirectBuffer,
                                                     const GLvoid* IGL_NULLABLE
                                                         indirectBufferOffset,
                                                     GLsizei drawcount,
                                                     GLsizei stride) {
  IGL_PROFILER_FUNCTION();
  willDraw();
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::MultiDrawIndirect)) {
    bindBufferWithShaderStorageBufferOverride(indirectBuffer, GL_DRAW_INDIRECT_BUFFER);
    getContext().multiDrawElementsIndirect(
        toMockWireframeMode(mode), indexType, indirectBufferOffset, drawcount, stride);
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }
  didDraw();
}

void RenderCommandAdapter::endEncoding() {
  IGL_PROFILER_FUNCTION();
  // Some minimal cleanup needs to occur in order. Otherwise, OpenGL can end in a bad state
  // with complex rendering.
  if (subpassColorAttachmentsApplied_) {
    // Unlike their indexed variants, glColorMask() and glDisable() set the state of every draw
    // buffer
    getContext().colorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    getContext().disable(GL_BLEND);
    subpassColorAttachmentsApplied_ = false;
  }
  if (pipelineState_) {
    unbindVertexAttributes();
  }

  pipelineState_ = nullptr;
  depthStencilState_ = nullptr;

  uniformAdapter_.shrinkUniformUsage();
  uniformAdapter_.clearUniformBuffers();
  vertexTextureStates_ = TextureStates();
  fragmentTextureStates_ = TextureStates();

  vertexBuffersDirty_.reset();
  vertexTextureStatesDirty_.reset();
  fragmentTextureStatesDirty_.reset();
  dirtyStateBits_ = EnumToValue(StateMask::NONE);
}

/**
 * @brief Binds all dirty OpenGL state before issuing a draw call.
 *
 * Prepares the rendering pipeline by flushing pending state changes:
 * dirty vertex buffers and their attributes, the render pipeline
 * state, depth/stencil state with stencil reference values, queued
 * uniforms, and dirty vertex/fragment texture-sampler pairs. Also
 * validates shader stages when shader validation is enabled.
 */
void RenderCommandAdapter::willDraw() {
  IGL_PROFILER_FUNCTION();
  Result ret;
  auto* pipelineState = static_cast<RenderPipelineState*>(pipelineState_.get());

  // Vertex Buffers must be bound before pipelineState->bind()
  if (pipelineState) {
    pipelineState->clearActiveAttributesLocations();
    for (size_t bufferIndex = 0; bufferIndex < IGL_BUFFER_BINDINGS_MAX; ++bufferIndex) {
      if (IS_DIRTY(vertexBuffersDirty_, bufferIndex)) {
        auto& bufferState = vertexBuffers_[bufferIndex];
        bindBufferWithShaderStorageBufferOverride((*bufferState.resource), GL_ARRAY_BUFFER);
        // now bind the vertex attributes corresponding to this vertex buffer
        pipelineState->bindVertexAttributes(bufferIndex, bufferState.offset, bufferState.stride);
        CLEAR_DIRTY(vertexBuffersDirty_, bufferIndex);
      }
    }
    pipelineState->unbindPrevPipelineVertexAttributes();
    if (isDirty(StateMask::PIPELINE)) {
      pipelineState->bind();
      clearDirty(StateMask::PIPELINE);
      if (numSubpassColorAttachments_ != 0) {
        // bind() sets the color mask and the blending of every draw buffer from the first
        // color attachment
        setDirty(StateMask::SubpassColorAttachments);
      }
    }
    if (isDirty(StateMask::SubpassColorAttachments)) {
      applySubpassColorAttachments(*pipelineState);
      clearDirty(StateMask::SubpassColorAttachments);
    }
    if (isDirty(StateMask::CullMode)) {
      if (cullMode_ == CullMode::Disabled) {
        getContext().disable(GL_CULL_FACE);
      } else {
        getContext().enable(GL_CULL_FACE);
        getContext().cullFace(cullMode_ == CullMode::Front ? GL_FRONT : GL_BACK);
      }
      clearDirty(StateMask::CullMode);
    }
    if (isDirty(StateMask::FrontFaceWinding)) {
      getContext().frontFace(windingMode_ == WindingMode::Clockwise ? GL_CW : GL_CCW);
      clearDirty(StateMask::FrontFaceWinding);
    }
  }

  auto* depthStencilState = static_cast<DepthStencilState*>(depthStencilState_.get());
  if (depthStencilState && isDirty(StateMask::DepthStencil)) {
    depthStencilState->bind(frontStencilReferenceValue_, backStencilReferenceValue_);
    clearDirty(StateMask::DepthStencil);
  }

  // We store 2 parallel vectors (one for uniforms and one for uniform blocks)
  // that behave as a queue of uniforms waiting to be issued to the GPU.
  // These were set between the previous draw call and the current draw call.
  // At the end of the draw call, the queue can be treated as empty since we
  // don't need to re-issue them (they'll carry over to the next draw call unless
  // a subsequent uniform at the same location is bound).
  //
  // Various error-checking does _not_ happen in production, since this is the "inner loop":
  //
  // * Duplicate uniform and uniform block at same location (one in each vector)
  // * Duplicate uniforms in the vector with the same location (same for uniform blocks)
  //
  // These should be considered client bugs, so an assert fires in local dev builds.

  // this is actually compile time defined and doesn't change, cached these statically.
  static const size_t kVertexTextureStatesSize = vertexTextureStates_.size();
  static const size_t kFragmentTextureStatesSize = fragmentTextureStates_.size();
  if (pipelineState) {
    // Bind uniforms to be used for render
    uniformAdapter_.bindToPipeline(getContext());
    // Bind storage buffers
    for (size_t bufferIndex = 0; bufferIndex < IGL_BUFFER_BINDINGS_MAX; ++bufferIndex) {
      if (IS_DIRTY(storageBuffersDirty_, bufferIndex)) {
        auto& bufferState = storageBuffers_[bufferIndex];
        auto& arrayBuffer = static_cast<ArrayBuffer&>(*bufferState.resource);
        Result result;
        arrayBuffer.bindBase(bufferIndex, &result);
        IGL_DEBUG_ASSERT(result.isOk());
        CLEAR_DIRTY(storageBuffersDirty_, bufferIndex);
      }
    }
    for (size_t index = 0; index < kVertexTextureStatesSize; index++) {
      if (!IS_DIRTY(vertexTextureStatesDirty_, index)) {
        continue;
      }
      auto& textureState = vertexTextureStates_[index];
      if (auto* texture = static_cast<Texture*>(textureState.first)) {
        if (IGL_DEBUG_VERIFY_NOT(texture == nullptr)) {
          continue;
        }
        ret = pipelineState->bindTextureUnit(index, BindTarget::kVertex, *texture);

        if (!ret.isOk()) {
          IGL_LOG_INFO_ONCE(ret.message.c_str());
          continue;
        }

        if (auto* samplerState = static_cast<SamplerState*>(textureState.second)) {
          samplerState->bind(texture);
        }
        CLEAR_DIRTY(vertexTextureStatesDirty_, index);
      }
    }
    for (size_t index = 0; index < kFragmentTextureStatesSize; index++) {
      if (!IS_DIRTY(fragmentTextureStatesDirty_, index)) {
        continue;
      }
      auto& textureState = fragmentTextureStates_[index];
      if (auto* texture = static_cast<Texture*>(textureState.first)) {
        if (IGL_DEBUG_VERIFY_NOT(texture == nullptr)) {
          continue;
        }
        ret = pipelineState->bindTextureUnit(index, BindTarget::kFragment, *texture);

        if (!ret.isOk()) {
          IGL_LOG_INFO_ONCE(ret.message.c_str());
          continue;
        }

        if (auto* samplerState = static_cast<SamplerState*>(textureState.second)) {
          samplerState->bind(texture);
        }
        CLEAR_DIRTY(fragmentTextureStatesDirty_, index);
      }
    }

    if (getContext().shouldValidateShaders()) {
      const auto* stages = pipelineState->getShaderStages();
      if (stages) {
        const auto result = stages->validate();
        IGL_DEBUG_ASSERT(result.isOk(), result.message.c_str());
      }
    }
  }
}

GLenum RenderCommandAdapter::toMockWireframeMode(GLenum mode) const {
#if defined(IGL_OPENGL_ES)
  auto* const pipelineState = static_cast<RenderPipelineState*>(pipelineState_.get());
  const bool modeNeedsConversion = mode == GL_TRIANGLES || mode != GL_TRIANGLE_STRIP;
  if (pipelineState->getPolygonFillMode() == PolygonFillMode::Line && modeNeedsConversion) {
    return GL_LINE_STRIP;
  }
#endif

  return mode;
}

void RenderCommandAdapter::didDraw() {
  // Placeholder stub in case we want to add something later
}

void RenderCommandAdapter::unbindVertexAttributes() {
  IGL_PROFILER_FUNCTION();
  auto* pipelineState = static_cast<RenderPipelineState*>(pipelineState_.get());
  if (pipelineState) {
    pipelineState->unbindVertexAttributes();
  }
}

void RenderCommandAdapter::applySubpassColorAttachments(const RenderPipelineState& pipelineState) {
  IGL_PROFILER_FUNCTION();
  const auto& colorAttachments = pipelineState.getRenderPipelineDesc().targetDesc.colorAttachments;
  for (uint32_t i = 0; i != numSubpassColorAttachments_; ++i) {
    // A pipeline that does not describe an attachment writes all of its channels without blending
    const bool isDescribed = i < colorAttachments.size();
    ColorWriteMask writeMask = 0;
    if ((subpassColorAttachmentMask_ & (1u << i)) != 0) {
      writeMask = isDescribed ? colorAttachments[i].colorWriteMask : kColorWriteBitsAll;
    }
    getContext().colorMaski(i,
                            static_cast<GLboolean>((writeMask & kColorWriteBitsRed) != 0),
                            static_cast<GLboolean>((writeMask & kColorWriteBitsGreen) != 0),
                            static_cast<GLboolean>((writeMask & kColorWriteBitsBlue) != 0),
                            static_cast<GLboolean>((writeMask & kColorWriteBitsAlpha) != 0));

    if (isDescribed && colorAttachments[i].blendEnabled) {
      const auto& attachment = colorAttachments[i];
      getContext().enablei(GL_BLEND, i);
      getContext().blendEquationSeparatei(
          i,
          RenderPipelineState::convertBlendOp(attachment.rgbBlendOp),
          RenderPipelineState::convertBlendOp(attachment.alphaBlendOp));
      getContext().blendFuncSeparatei(
          i,
          RenderPipelineState::convertBlendFactor(attachment.srcRGBBlendFactor),
          RenderPipelineState::convertBlendFactor(attachment.dstRGBBlendFactor),
          RenderPipelineState::convertBlendFactor(attachment.srcAlphaBlendFactor),
          RenderPipelineState::convertBlendFactor(attachment.dstAlphaBlendFactor));
    } else {
      getContext().disablei(GL_BLEND, i);
    }
  }
  subpassColorAttachmentsApplied_ = true;
}

void RenderCommandAdapter::bindBufferWithShaderStorageBufferOverride(
    Buffer& buffer,
    GLenum overrideTargetForShaderStorageBuffer) {
  IGL_PROFILER_FUNCTION();
  auto& arrayBuffer = static_cast<ArrayBuffer&>(buffer);
  if (arrayBuffer.getTarget() == GL_SHADER_STORAGE_BUFFER) {
    arrayBuffer.bindForTarget(overrideTargetForShaderStorageBuffer);
  } else {
    arrayBuffer.bind();
  }
}
} // namespace igl::opengl
