/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/metal/RenderCommandEncoder.h>

#import <Foundation/Foundation.h>
#import <Metal/MTLBuffer.h>
#import <Metal/MTLRenderCommandEncoder.h>
#import <Metal/MTLRenderPass.h>
#import <Metal/MTLTypes.h>
#include <igl/Macros.h>
#include <igl/RenderPass.h>
#include <igl/metal/Buffer.h>
#include <igl/metal/DepthStencilState.h>
#include <igl/metal/Device.h>
#include <igl/metal/Framebuffer.h>
#include <igl/metal/RenderPipelineState.h>
#include <igl/metal/SamplerState.h>
#include <igl/metal/Texture.h>
#include <igl/metal/TimestampQueries.h>

namespace igl::metal {
RenderCommandEncoder::RenderCommandEncoder(const std::shared_ptr<CommandBuffer>& commandBuffer) :
  IRenderCommandEncoder::IRenderCommandEncoder(commandBuffer), device_(commandBuffer->device()) {}

/**
 * @brief Builds a Metal render pass descriptor from IGL
 *        abstractions and creates the render command encoder.
 *
 * Translates the IGL RenderPassDesc and IFramebuffer into a
 * MTLRenderPassDescriptor, configuring color, depth, and stencil
 * attachments with load/store actions, clear values, MSAA resolve
 * textures, and slice/mip levels. On macOS 11.0+/iOS 14.0+, also
 * configures GPU timestamp query sample buffer attachments.
 * Stores the resulting encoder in the encoder_ member.
 *
 * @param[in] commandBuffer Metal command buffer from which the
 *        render command encoder is created.
 * @param[in] renderPass Specifies per-attachment load/store
 *        actions, clear values, and timestamp query configuration.
 * @param[in] framebuffer Provides color, depth, and stencil
 *        texture attachments; cast internally to metal::Framebuffer.
 * @param[out] outResult Set to OK on success, ArgumentNull if
 *        framebuffer is null, ArgumentInvalid on attachment count
 *        mismatch, or RuntimeError if descriptor creation fails.
 */
void RenderCommandEncoder::initialize(const std::shared_ptr<CommandBuffer>& commandBuffer,
                                      const RenderPassDesc& renderPass,
                                      const std::shared_ptr<IFramebuffer>& framebuffer,
                                      Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  Result::setOk(outResult);
  if (!IGL_DEBUG_VERIFY(framebuffer)) {
    Result::setResult(outResult, Result::Code::ArgumentNull);
    return;
  }
  MTLRenderPassDescriptor* metalRenderPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];
  if (!metalRenderPassDesc) {
    static const char* kFailedToCreateRenderPassDesc =
        "Failed to create Metal render pass descriptor";
    IGL_DEBUG_ABORT(kFailedToCreateRenderPassDesc);
    Result::setResult(outResult, Result::Code::RuntimeError, kFailedToCreateRenderPassDesc);
    return;
  }

  const FramebufferDesc& desc = static_cast<const Framebuffer&>(*framebuffer).get();

  // Colors
  for (size_t index = 0; index != IGL_COLOR_ATTACHMENTS_MAX; index++) {
    const auto& attachment = desc.colorAttachments[index];

    if (!attachment.texture) {
      continue;
    }

    if (index >= renderPass.colorAttachments.size() || index >= IGL_COLOR_ATTACHMENTS_MAX) {
      static const char* kNotEnoughRenderPassColorAttachments =
          "Framebuffer color attachment count larger than renderPass color attachment count";
      IGL_DEBUG_ABORT(kNotEnoughRenderPassColorAttachments);
      Result::setResult(
          outResult, Result::Code::ArgumentInvalid, kNotEnoughRenderPassColorAttachments);
      break;
    }

    const auto& iglTexture = attachment.texture;
    MTLRenderPassColorAttachmentDescriptor* metalColorAttachment =
        metalRenderPassDesc.colorAttachments[index];

    static const char* kNullColorAttachmentMsg = "Render pass color attachment cannot be null";
    IGL_DEBUG_ASSERT(iglTexture, kNullColorAttachmentMsg);
    if (iglTexture) {
      metalColorAttachment.texture = static_cast<Texture&>(*iglTexture).get();
    } else {
      Result::setResult(outResult, Result::Code::ArgumentNull, kNullColorAttachmentMsg);
    }

    const auto& iglResolveTexture = attachment.resolveTexture;
    if (iglResolveTexture &&
        renderPass.colorAttachments[index].storeAction == StoreAction::MsaaResolve) {
      metalColorAttachment.resolveTexture = static_cast<Texture&>(*iglResolveTexture).get();
    }

    const auto& iglColorAttachment = renderPass.colorAttachments[index];
    metalColorAttachment.loadAction = convertLoadAction(iglColorAttachment.loadAction);
    metalColorAttachment.storeAction = convertStoreAction(iglColorAttachment.storeAction);
    metalColorAttachment.clearColor = convertClearColor(iglColorAttachment.clearColor);
    metalColorAttachment.slice = iglTexture ? Texture::getMetalSlice(iglTexture->getType(),
                                                                     iglColorAttachment.face,
                                                                     iglColorAttachment.layer)
                                            : 0;
    metalColorAttachment.level = iglColorAttachment.mipLevel;
  }

  // Depth
  if (desc.depthAttachment.texture) {
    metalRenderPassDesc.depthAttachment.texture =
        static_cast<Texture&>(*desc.depthAttachment.texture).get();
    metalRenderPassDesc.depthAttachment.loadAction =
        convertLoadAction(renderPass.depthAttachment.loadAction);
    metalRenderPassDesc.depthAttachment.storeAction =
        convertStoreAction(renderPass.depthAttachment.storeAction);
    metalRenderPassDesc.depthAttachment.clearDepth = renderPass.depthAttachment.clearDepth;

    if (desc.depthAttachment.resolveTexture &&
        renderPass.depthAttachment.storeAction == StoreAction::MsaaResolve) {
      metalRenderPassDesc.depthAttachment.resolveTexture =
          static_cast<Texture&>(*desc.depthAttachment.resolveTexture).get();
    }
  }

  // Stencil
  if (desc.stencilAttachment.texture) {
    metalRenderPassDesc.stencilAttachment.texture =
        static_cast<Texture&>(*desc.stencilAttachment.texture).get();
    metalRenderPassDesc.stencilAttachment.loadAction =
        convertLoadAction(renderPass.stencilAttachment.loadAction);
    metalRenderPassDesc.stencilAttachment.storeAction =
        convertStoreAction(renderPass.stencilAttachment.storeAction);
    metalRenderPassDesc.stencilAttachment.clearStencil = renderPass.stencilAttachment.clearStencil;

    if (desc.stencilAttachment.resolveTexture &&
        renderPass.stencilAttachment.storeAction == StoreAction::MsaaResolve) {
      metalRenderPassDesc.stencilAttachment.resolveTexture =
          static_cast<Texture&>(*desc.stencilAttachment.resolveTexture).get();
    }
  }

  // Attach counter sample buffer for GPU timestamp queries.
  // Configure sampleBufferAttachments with sample indices for descriptor-based
  // sampling (Tracy/Dawn pattern). Metal auto-samples at vertex start and fragment
  // end.
  // This is the ONLY approach that works on Apple GPUs (A-series, M-series).
  if (@available(macOS 11.0, iOS 14.0, *)) {
    if (renderPass.timestampQuery.queries) {
      auto metalTsQueries =
          std::static_pointer_cast<TimestampQueries>(renderPass.timestampQuery.queries);
      if (metalTsQueries && metalTsQueries->sampleBuffer_ != nil) {
        const uint32_t startSampleIdx =
            renderPass.timestampQuery.slotIndex * TimestampQueries::kSamplesPerTimingSlot;
        const uint32_t endSampleIdx = startSampleIdx + 1;
        const uint32_t bufferSize =
            metalTsQueries->maxTimestamps_ * TimestampQueries::kSamplesPerTimingSlot;
        if (endSampleIdx < bufferSize) {
          metalRenderPassDesc.sampleBufferAttachments[0].sampleBuffer =
              metalTsQueries->sampleBuffer_;
          metalRenderPassDesc.sampleBufferAttachments[0].startOfVertexSampleIndex = startSampleIdx;
          metalRenderPassDesc.sampleBufferAttachments[0].endOfVertexSampleIndex =
              MTLCounterDontSample;
          metalRenderPassDesc.sampleBufferAttachments[0].startOfFragmentSampleIndex =
              MTLCounterDontSample;
          metalRenderPassDesc.sampleBufferAttachments[0].endOfFragmentSampleIndex = endSampleIdx;

          // Advance currentIndex_ so resolveTimestamps knows how many samples to resolve.
          uint32_t requiredCount = endSampleIdx + 1;
          uint32_t current = metalTsQueries->currentIndex_.load(std::memory_order_relaxed);
          while (current < requiredCount) {
            if (metalTsQueries->currentIndex_.compare_exchange_weak(
                    current, requiredCount, std::memory_order_relaxed)) {
              break;
            }
          }
        }
      }
    }
  }

  encoder_ = [commandBuffer->get() renderCommandEncoderWithDescriptor:metalRenderPassDesc];

  // CommandBuffer has validated the subpasses
  if (!renderPass.subpasses.empty()) {
    numSubpasses_ = static_cast<uint32_t>(renderPass.subpasses.size());
  }
}

std::unique_ptr<RenderCommandEncoder> RenderCommandEncoder::create(
    const std::shared_ptr<CommandBuffer>& commandBuffer,
    const RenderPassDesc& renderPass,
    const std::shared_ptr<IFramebuffer>& framebuffer,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  std::unique_ptr<RenderCommandEncoder> encoder(new RenderCommandEncoder(commandBuffer));
  encoder->initialize(commandBuffer, renderPass, framebuffer, outResult);
  return encoder;
}

void RenderCommandEncoder::endEncoding() {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(currentSubpass_ + 1 == numSubpasses_,
                   "nextSubpass() must be called RenderPassDesc::subpasses.size() - 1 times before "
                   "endEncoding()");
  // @fb-only
  // @fb-only
  [encoder_ endEncoding];
  encoder_ = nil;
}

void RenderCommandEncoder::nextSubpass() {
  IGL_PROFILER_FUNCTION();
  if (!IGL_DEBUG_VERIFY(encoder_)) {
    return;
  }
  if (currentSubpass_ + 1 >= numSubpasses_) {
    IGL_DEBUG_ABORT(
        "nextSubpass() was called more than RenderPassDesc::subpasses.size() - 1 times");
    return;
  }
  // No Metal command is needed: the subpasses share this encoder, and programmable blending orders
  // the color attachment reads and writes of its draws at each pixel.
  ++currentSubpass_;
}

void RenderCommandEncoder::pushDebugGroupLabel(const char* label, const Color& /*color*/) const {
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(label != nullptr && *label);
  [encoder_ pushDebugGroup:[NSString stringWithUTF8String:label] ?: @""];
}

void RenderCommandEncoder::insertDebugEventLabel(const char* label, const Color& /*color*/) const {
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(label != nullptr && *label);
  [encoder_ insertDebugSignpost:[NSString stringWithUTF8String:label] ?: @""];
}

void RenderCommandEncoder::popDebugGroupLabel() const {
  IGL_DEBUG_ASSERT(encoder_);
  [encoder_ popDebugGroup];
}

void RenderCommandEncoder::bindViewport(const Viewport& viewport) {
  IGL_DEBUG_ASSERT(encoder_);
  const MTLViewport metalViewport = {.originX = viewport.x,
                                     .originY = viewport.y,
                                     .width = viewport.width,
                                     .height = viewport.height,
                                     .znear = viewport.minDepth,
                                     .zfar = viewport.maxDepth};
  [encoder_ setViewport:metalViewport];
}

void RenderCommandEncoder::bindScissorRect(const ScissorRect& rect) {
  IGL_DEBUG_ASSERT(encoder_);
  const MTLScissorRect scissorRect = {
      .x = rect.x, .y = rect.y, .width = rect.width, .height = rect.height};
  [encoder_ setScissorRect:scissorRect];
}

void RenderCommandEncoder::setCullMode(CullMode cullMode) {
  IGL_DEBUG_ASSERT(encoder_);
  MTLCullMode mode = MTLCullModeNone;
  switch (cullMode) {
  case CullMode::Disabled:
    mode = MTLCullModeNone;
    break;
  case CullMode::Front:
    mode = MTLCullModeFront;
    break;
  case CullMode::Back:
    mode = MTLCullModeBack;
    break;
  default:
    break;
  }
  [encoder_ setCullMode:mode];
}

void RenderCommandEncoder::setFrontFacingWinding(WindingMode frontFaceWinding) {
  IGL_DEBUG_ASSERT(encoder_);
  const MTLWinding mode = (frontFaceWinding == WindingMode::Clockwise) ? MTLWindingClockwise
                                                                       : MTLWindingCounterClockwise;

  [encoder_ setFrontFacingWinding:mode];
}

void RenderCommandEncoder::bindPolygonFillMode(const PolygonFillMode& polygonFillMode) {
  IGL_DEBUG_ASSERT(encoder_);

  if (polygonFillMode == PolygonFillMode::Fill) {
    return;
  }

  [encoder_ setTriangleFillMode:MTLTriangleFillModeLines];
}

void RenderCommandEncoder::bindRenderPipelineState(
    const std::shared_ptr<IRenderPipelineState>& pipelineState) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(pipelineState);
  if (!pipelineState) {
    return;
  }
  auto& metalPipelineState = static_cast<RenderPipelineState&>(*pipelineState);

  [encoder_ setRenderPipelineState:metalPipelineState.get()];

  setCullMode(metalPipelineState.getCullMode());
  setFrontFacingWinding(metalPipelineState.getWindingMode());
  bindPolygonFillMode(metalPipelineState.getPolygonFillMode());

  metalPrimitive_ = convertPrimitiveType(pipelineState->getRenderPipelineDesc().topology);
}

void RenderCommandEncoder::bindDepthStencilState(
    const std::shared_ptr<IDepthStencilState>& depthStencilState) {
  IGL_DEBUG_ASSERT(encoder_);
  if (depthStencilState) {
    [encoder_ setDepthStencilState:static_cast<DepthStencilState&>(*depthStencilState).get()];
  }
}

void RenderCommandEncoder::setBlendColor(const Color& color) {
  IGL_DEBUG_ASSERT(encoder_);
  [encoder_ setBlendColorRed:color.r green:color.g blue:color.b alpha:color.a];
}

void RenderCommandEncoder::setDepthBias(float depthBias, float slopeScale, float clamp) {
  IGL_DEBUG_ASSERT(encoder_);
  [encoder_ setDepthBias:depthBias slopeScale:slopeScale clamp:clamp];
}

void RenderCommandEncoder::setStencilReferenceValue(uint32_t value) {
  IGL_DEBUG_ASSERT(encoder_);
  [encoder_ setStencilReferenceValue:value];
}

void RenderCommandEncoder::bindBuffer(uint32_t index,
                                      uint8_t bindTarget,
                                      IBuffer* buffer,
                                      size_t offset,
                                      size_t bufferSize) {
  IGL_PROFILER_FUNCTION();
  (void)bufferSize;

  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(index < IGL_BUFFER_BINDINGS_MAX);

  auto* iglBuffer = static_cast<Buffer*>(buffer);
  auto metalBuffer = iglBuffer ? iglBuffer->get() : nil;

  if ((bindTarget & BindTarget::kVertex) != 0) {
    [encoder_ setVertexBuffer:metalBuffer offset:offset atIndex:index];
  }
  if ((bindTarget & BindTarget::kFragment) != 0) {
    [encoder_ setFragmentBuffer:metalBuffer offset:offset atIndex:index];
  }
  if (@available(iOS 16, macOS 13, *)) {
    if (device_.hasFeature(DeviceFeatures::MeshShaders)) {
      if ((bindTarget & BindTarget::kTask) != 0) {
        [encoder_ setObjectBuffer:metalBuffer offset:offset atIndex:index];
      }
      if ((bindTarget & BindTarget::kMesh) != 0) {
        [encoder_ setMeshBuffer:metalBuffer offset:offset atIndex:index];
      }
    }
  }
}

void RenderCommandEncoder::bindBuffer(uint32_t index,
                                      IBuffer* buffer,
                                      size_t offset,
                                      size_t bufferSize) {
  IGL_PROFILER_FUNCTION();
  (void)bufferSize;

  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(index < IGL_BUFFER_BINDINGS_MAX);

  if (buffer) {
    auto& metalBuffer = static_cast<Buffer&>(*buffer);
    [encoder_ setVertexBuffer:metalBuffer.get() offset:offset atIndex:index];
    [encoder_ setFragmentBuffer:metalBuffer.get() offset:offset atIndex:index];
  } else {
    [encoder_ setVertexBuffer:nil offset:0 atIndex:index];
    [encoder_ setFragmentBuffer:nil offset:0 atIndex:index];
  }
}

void RenderCommandEncoder::bindVertexBuffer(uint32_t index,
                                            IBuffer& buffer,
                                            size_t bufferOffset,
                                            size_t attributeStride) {
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(index < IGL_BUFFER_BINDINGS_MAX);
  if (index >= IGL_BUFFER_BINDINGS_MAX) {
    IGL_LOG_ERROR("bindVertexBuffer: index %u exceeds max %u\n", index, IGL_BUFFER_BINDINGS_MAX);
    return;
  }

  auto& metalBuffer = static_cast<Buffer&>(buffer);

  if (attributeStride == 0) {
    [encoder_ setVertexBuffer:metalBuffer.get() offset:bufferOffset atIndex:index];
  } else {
    if (device_.hasFeature(DeviceFeatures::DynamicVertexBufferStride)) {
      if (@available(macos 14, ios 17, *)) {
        [encoder_ setVertexBuffer:metalBuffer.get()
                           offset:bufferOffset
                  attributeStride:attributeStride
                          atIndex:index];
      }
    } else {
      IGL_DEBUG_ASSERT_NOT_REACHED();
    }
  }
}

void RenderCommandEncoder::bindIndexBuffer(IBuffer& buffer,
                                           IndexFormat format,
                                           size_t bufferOffset) {
  auto& metalBuffer = static_cast<Buffer&>(buffer);
  indexBuffer_ = metalBuffer.get();
  indexType_ = convertIndexType(format);
  indexBufferOffset_ = bufferOffset;
}

void RenderCommandEncoder::bindBytes(size_t index,
                                     uint8_t bindTarget,
                                     const void* data,
                                     size_t length) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(bindTarget == BindTarget::kVertex || bindTarget == BindTarget::kFragment ||
                       bindTarget == BindTarget::kTask || bindTarget == BindTarget::kMesh ||
                       bindTarget == BindTarget::kAllGraphics,
                   "Bind target is not valid: %d",
                   bindTarget);
  if (data) {
    if (length > kMaxRecommendedBytes) {
      IGL_LOG_INFO(
          "It is recommended to use bindBuffer instead of bindBytes when binding > 4kb: %u",
          length);
    }
    if ((bindTarget & BindTarget::kVertex) != 0) {
      [encoder_ setVertexBytes:data length:length atIndex:index];
    }
    if ((bindTarget & BindTarget::kFragment) != 0) {
      [encoder_ setFragmentBytes:data length:length atIndex:index];
    }
    if (@available(iOS 16, macOS 13, *)) {
      if (device_.hasFeature(DeviceFeatures::MeshShaders)) {
        if ((bindTarget & BindTarget::kTask) != 0) {
          [encoder_ setObjectBytes:data length:length atIndex:index];
        }
        if ((bindTarget & BindTarget::kMesh) != 0) {
          [encoder_ setMeshBytes:data length:length atIndex:index];
        }
      }
    }
  }
}

void RenderCommandEncoder::bindPushConstants(const void* /*data*/,
                                             size_t /*length*/,
                                             size_t /*offset*/) {
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

void RenderCommandEncoder::bindTexture(size_t index, uint8_t bindTarget, ITexture* texture) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(bindTarget == BindTarget::kVertex || bindTarget == BindTarget::kFragment ||
                       bindTarget == BindTarget::kTask || bindTarget == BindTarget::kMesh ||
                       bindTarget == BindTarget::kAllGraphics,
                   "Bind target is not valid: %d",
                   bindTarget);

  auto* iglTexture = static_cast<Texture*>(texture);
  auto metalTexture = iglTexture ? iglTexture->get() : nil;

  if ((bindTarget & BindTarget::kVertex) != 0) {
    [encoder_ setVertexTexture:metalTexture atIndex:index];
  }
  if ((bindTarget & BindTarget::kFragment) != 0) {
    [encoder_ setFragmentTexture:metalTexture atIndex:index];
  }
  if (@available(iOS 16, macOS 13, *)) {
    if (device_.hasFeature(DeviceFeatures::MeshShaders)) {
      if ((bindTarget & BindTarget::kTask) != 0) {
        [encoder_ setObjectTexture:metalTexture atIndex:index];
      }
      if ((bindTarget & BindTarget::kMesh) != 0) {
        [encoder_ setMeshTexture:metalTexture atIndex:index];
      }
    }
  }
}

void RenderCommandEncoder::bindTexture(size_t index, ITexture* texture) {
  bindTexture(index, BindTarget::kFragment, texture);
}

void RenderCommandEncoder::bindUniform(const UniformDesc& /*uniformDesc*/, const void* /*data*/) {
  // DO NOT IMPLEMENT!
  // This is only for backends that MUST use single uniforms in some situations.
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
}

void RenderCommandEncoder::bindSamplerState(size_t index,
                                            uint8_t bindTarget,
                                            ISamplerState* samplerState) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(bindTarget == BindTarget::kVertex || bindTarget == BindTarget::kFragment ||
                       bindTarget == BindTarget::kTask || bindTarget == BindTarget::kMesh ||
                       bindTarget == BindTarget::kAllGraphics,
                   "Bind target is not valid: %d",
                   bindTarget);

  auto* iglSamplerState = static_cast<SamplerState*>(samplerState);
  auto metalSamplerState = iglSamplerState ? iglSamplerState->get() : nil;

  if ((bindTarget & BindTarget::kVertex) != 0) {
    [encoder_ setVertexSamplerState:metalSamplerState atIndex:index];
  }
  if ((bindTarget & BindTarget::kFragment) != 0) {
    [encoder_ setFragmentSamplerState:metalSamplerState atIndex:index];
  }
  if (@available(iOS 16, macOS 13, *)) {
    if (device_.hasFeature(DeviceFeatures::MeshShaders)) {
      if ((bindTarget & BindTarget::kTask) != 0) {
        [encoder_ setObjectSamplerState:metalSamplerState atIndex:index];
      }
      if ((bindTarget & BindTarget::kMesh) != 0) {
        [encoder_ setMeshSamplerState:metalSamplerState atIndex:index];
      }
    }
  }
}

void RenderCommandEncoder::draw(size_t vertexCount,
                                uint32_t instanceCount,
                                uint32_t firstVertex,
                                uint32_t baseInstance) {
  IGL_PROFILER_FUNCTION();
  getCommandBuffer().incrementCurrentDrawCount();
  IGL_DEBUG_ASSERT(encoder_);
#if IGL_PLATFORM_IOS
  if (@available(iOS 16, *)) {
#endif // IGL_PLATFORM_IOS
    [encoder_ drawPrimitives:metalPrimitive_
                 vertexStart:firstVertex
                 vertexCount:vertexCount
               instanceCount:instanceCount
                baseInstance:baseInstance];
#if IGL_PLATFORM_IOS
  } else {
    IGL_DEBUG_ASSERT(!baseInstance);
    if (device_.hasFeature(DeviceFeatures::DrawInstanced)) {
      [encoder_ drawPrimitives:metalPrimitive_
                   vertexStart:firstVertex
                   vertexCount:vertexCount
                 instanceCount:instanceCount];
    } else {
      IGL_DEBUG_ASSERT(instanceCount <= 1);
      [encoder_ drawPrimitives:metalPrimitive_ vertexStart:firstVertex vertexCount:vertexCount];
    }
  }
#endif // IGL_PLATFORM_IOS
}

void RenderCommandEncoder::drawIndexed(size_t indexCount,
                                       uint32_t instanceCount,
                                       uint32_t firstIndex,
                                       int32_t vertexOffset,
                                       uint32_t baseInstance) {
  IGL_PROFILER_FUNCTION();
  getCommandBuffer().incrementCurrentDrawCount();
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(indexBuffer_, "No index buffer bound");
  if (!IGL_DEBUG_VERIFY(encoder_ && indexBuffer_)) {
    return;
  }

  const size_t indexOffsetBytes =
      static_cast<size_t>(firstIndex) * (indexType_ == MTLIndexTypeUInt32 ? 4u : 2u);

#if IGL_PLATFORM_IOS
  if (@available(iOS 16, *)) {
#endif // IGL_PLATFORM_IOS
    [encoder_ drawIndexedPrimitives:metalPrimitive_
                         indexCount:indexCount
                          indexType:indexType_
                        indexBuffer:indexBuffer_
                  indexBufferOffset:indexBufferOffset_ + indexOffsetBytes
                      instanceCount:instanceCount
                         baseVertex:vertexOffset
                       baseInstance:baseInstance];
#if IGL_PLATFORM_IOS
  } else {
    if (baseInstance) {
      IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    } else {
      if (device_.hasFeature(DeviceFeatures::DrawInstanced)) {
        [encoder_ drawIndexedPrimitives:metalPrimitive_
                             indexCount:indexCount
                              indexType:indexType_
                            indexBuffer:indexBuffer_
                      indexBufferOffset:indexBufferOffset_ + indexOffsetBytes
                          instanceCount:instanceCount];
      } else {
        IGL_DEBUG_ASSERT(instanceCount <= 1);
        [encoder_ drawIndexedPrimitives:metalPrimitive_
                             indexCount:indexCount
                              indexType:indexType_
                            indexBuffer:indexBuffer_
                      indexBufferOffset:indexBufferOffset_ + indexOffsetBytes];
      }
    }
  }
#endif // IGL_PLATFORM_IOS
}

void RenderCommandEncoder::drawMeshTasks(const Dimensions& threadgroupsPerGrid,
                                         const Dimensions& threadsPerTaskThreadgroup,
                                         const Dimensions& threadsPerMeshThreadgroup) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(encoder_);

  if (!device_.hasFeature(DeviceFeatures::MeshShaders)) {
    IGL_DEBUG_ASSERT(false, "Mesh shaders require Apple GPU Family 7 or higher (A14/M1 and later)");
    return;
  }

  if (@available(iOS 16, macOS 13, *)) {
    MTLSize tgg;
    tgg.width = threadgroupsPerGrid.width;
    tgg.height = threadgroupsPerGrid.height;
    tgg.depth = threadgroupsPerGrid.depth;

    MTLSize tgt;
    tgt.width = threadsPerTaskThreadgroup.width;
    tgt.height = threadsPerTaskThreadgroup.height;
    tgt.depth = threadsPerTaskThreadgroup.depth;

    MTLSize tgm;
    tgm.width = threadsPerMeshThreadgroup.width;
    tgm.height = threadsPerMeshThreadgroup.height;
    tgm.depth = threadsPerMeshThreadgroup.depth;

    [encoder_ drawMeshThreadgroups:tgg
        threadsPerObjectThreadgroup:tgt
          threadsPerMeshThreadgroup:tgm];
  }
}

void RenderCommandEncoder::multiDrawIndirect(IBuffer& indirectBuffer,
                                             // Ignore bugprone-easily-swappable-parameters
                                             // @lint-ignore CLANGTIDY
                                             size_t indirectBufferOffset,
                                             uint32_t drawCount,
                                             uint32_t stride) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(encoder_);
  stride = stride ? stride : sizeof(MTLDrawPrimitivesIndirectArguments);
  auto& indirectBufferRef = static_cast<Buffer&>(indirectBuffer);

  for (uint32_t drawIndex = 0; drawIndex < drawCount; drawIndex++) {
    getCommandBuffer().incrementCurrentDrawCount();
    [encoder_ drawPrimitives:metalPrimitive_
              indirectBuffer:indirectBufferRef.get()
        indirectBufferOffset:indirectBufferOffset + static_cast<size_t>(stride) * drawIndex];
  }
}

void RenderCommandEncoder::multiDrawIndexedIndirect(IBuffer& indirectBuffer,
                                                    // Ignore bugprone-easily-swappable-parameters
                                                    // @lint-ignore CLANGTIDY
                                                    size_t indirectBufferOffset,
                                                    uint32_t drawCount,
                                                    uint32_t stride) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(encoder_);
  IGL_DEBUG_ASSERT(indexBuffer_, "No index buffer bound");
  if (!IGL_DEBUG_VERIFY(encoder_ && indexBuffer_)) {
    return;
  }
  stride = stride ? stride : sizeof(MTLDrawIndexedPrimitivesIndirectArguments);
  auto& indirectBufferRef = static_cast<Buffer&>(indirectBuffer);

  for (uint32_t drawIndex = 0; drawIndex < drawCount; drawIndex++) {
    getCommandBuffer().incrementCurrentDrawCount();
    [encoder_ drawIndexedPrimitives:metalPrimitive_
                          indexType:indexType_
                        indexBuffer:indexBuffer_
                  indexBufferOffset:indexBufferOffset_
                     indirectBuffer:indirectBufferRef.get()
               indirectBufferOffset:indirectBufferOffset +
                                    (stride ? static_cast<size_t>(stride)
                                            : sizeof(MTLDrawIndexedPrimitivesIndirectArguments)) *
                                        drawIndex];
  }
}

MTLPrimitiveType RenderCommandEncoder::convertPrimitiveType(PrimitiveType value) {
  switch (value) {
  case PrimitiveType::Point:
    return MTLPrimitiveTypePoint;
  case PrimitiveType::Line:
    return MTLPrimitiveTypeLine;
  case PrimitiveType::LineStrip:
    return MTLPrimitiveTypeLineStrip;
  case PrimitiveType::Triangle:
    return MTLPrimitiveTypeTriangle;
  case PrimitiveType::TriangleStrip:
    return MTLPrimitiveTypeTriangleStrip;
  default:
    break;
  }
  IGL_UNREACHABLE_RETURN(MTLPrimitiveTypeTriangle)
}

MTLIndexType RenderCommandEncoder::convertIndexType(IndexFormat value) {
  switch (value) {
  case IndexFormat::UInt8:
    IGL_DEBUG_ASSERT(false, "8-bit indices are not supported in Metal");
    [[fallthrough]];
  case IndexFormat::UInt16:
    return MTLIndexTypeUInt16;
  case IndexFormat::UInt32:
    return MTLIndexTypeUInt32;
  default:
    break;
  }
  IGL_UNREACHABLE_RETURN(MTLIndexTypeUInt16)
}

MTLLoadAction RenderCommandEncoder::convertLoadAction(LoadAction value) {
  switch (value) {
  case LoadAction::DontCare:
    return MTLLoadActionDontCare;
  case LoadAction::Clear:
    return MTLLoadActionClear;
  case LoadAction::Load:
    return MTLLoadActionLoad;
  default:
    break;
  }
  IGL_UNREACHABLE_RETURN(MTLLoadActionDontCare)
}

MTLStoreAction RenderCommandEncoder::convertStoreAction(StoreAction value) {
  switch (value) {
  case StoreAction::DontCare:
    return MTLStoreActionDontCare;
  case StoreAction::Store:
    return MTLStoreActionStore;
  case StoreAction::MsaaResolve:
    return MTLStoreActionMultisampleResolve;
  default:
    break;
  }
  IGL_UNREACHABLE_RETURN(MTLStoreActionDontCare)
}

MTLClearColor RenderCommandEncoder::convertClearColor(Color value) {
  return MTLClearColorMake(value.r, value.g, value.b, value.a);
}

void RenderCommandEncoder::bindBindGroup(BindGroupTextureHandle handle) {
  IGL_PROFILER_FUNCTION();
  if (handle.empty()) {
    return;
  }

  const BindGroupTextureDesc* desc = device_.bindGroupTexturesPool.get(handle);

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

  const BindGroupBufferDesc* desc = device_.bindGroupBuffersPool.get(handle);

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

} // namespace igl::metal
