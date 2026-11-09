/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <igl/RenderPipelineState.h>
#include <igl/webgpu/BindLayouts.h>
#include <igl/webgpu/Common.h>

namespace igl::webgpu {

class DeviceFeatureSet;
class RenderPipelineReflection;
class ShaderModule;
class WebGPUContext;

/// @brief The render state IGL sets on an encoder that WebGPU bakes into the pipeline. Every
/// field is a plain value with no padding, so keys compare and hash as bytes.
struct RenderPipelineDynamicState {
  // Depth/stencil state.
  uint8_t depthCompare = 0;
  uint8_t depthWriteEnabled = 0;
  uint8_t frontStencilCompare = 0;
  uint8_t frontStencilFailOp = 0;
  uint8_t frontDepthFailOp = 0;
  uint8_t frontStencilPassOp = 0;
  uint8_t backStencilCompare = 0;
  uint8_t backStencilFailOp = 0;
  uint8_t backDepthFailOp = 0;
  uint8_t backStencilPassOp = 0;
  // Rasterization.
  uint8_t cullMode = 0;
  uint8_t frontFaceWinding = 0;
  // IndexFormat + 1 of the bound index buffer for indexed strip draws, 0 otherwise.
  uint8_t stripIndexFormat = 0;
  uint8_t padding[3] = {};
  uint32_t stencilReadMask = ~0u;
  uint32_t stencilWriteMask = ~0u;
  float depthBias = 0.0f;
  float depthBiasSlopeScale = 0.0f;
  float depthBiasClamp = 0.0f;
  uint32_t padding2 = 0;
  SampleClasses sampleClasses = 0;

  void setDepthStencilState(const DepthStencilStateDesc& desc);
  [[nodiscard]] DepthStencilStateDesc getDepthStencilState() const;

  bool operator==(const RenderPipelineDynamicState& other) const;
};

struct RenderPipelineDynamicStateHash {
  size_t operator()(const RenderPipelineDynamicState& key) const;
};

/// @brief Implements the igl::IRenderPipelineState interface for WebGPU.
///
/// A WebGPU render pipeline also fixes depth/stencil state, culling, winding, depth bias, the
/// strip index format and the bind group layouts (which depend on the formats of the bound
/// textures). Pipelines are therefore created lazily, one per RenderPipelineDynamicState variant,
/// with explicit layouts built from the shaders' reflection.
class RenderPipelineState final : public IRenderPipelineState, public PipelineLayoutSource {
 public:
  /// Soft limit on variants per pipeline; exceeding it logs a warning.
  static constexpr size_t kVariantWarningThreshold = 64;

  [[nodiscard]] static std::shared_ptr<RenderPipelineState> create(WebGPUContext& ctx,
                                                                   const DeviceFeatureSet& features,
                                                                   BindLayoutCache& layoutCache,
                                                                   const RenderPipelineDesc& desc,
                                                                   Result* IGL_NULLABLE outResult);
  ~RenderPipelineState() override;

  RenderPipelineState(const RenderPipelineState&) = delete;
  RenderPipelineState& operator=(const RenderPipelineState&) = delete;
  RenderPipelineState(RenderPipelineState&&) = delete;
  RenderPipelineState& operator=(RenderPipelineState&&) = delete;

  [[nodiscard]] std::shared_ptr<IRenderPipelineReflection> renderPipelineReflection() override;
  void setRenderPipelineReflection(const IRenderPipelineReflection& reflection) override;
  [[nodiscard]] int getIndexByName(const NameHandle& name, ShaderStage stage) const override;
  [[nodiscard]] int getIndexByName(const std::string& name, ShaderStage stage) const override;

  /// The state the pipeline description implies before any encoder state is applied.
  [[nodiscard]] RenderPipelineDynamicState getDefaultDynamicState() const;

  /// The pipeline for `state`, created on first use.
  [[nodiscard]] WGPURenderPipeline IGL_NULLABLE getPipeline(const RenderPipelineDynamicState& state,
                                                            Result* IGL_NULLABLE outResult);
  [[nodiscard]] WGPUBindGroupLayout IGL_NULLABLE
  getBindGroupLayout(uint32_t group,
                     SampleClasses classes,
                     Result* IGL_NULLABLE outResult) override;
  [[nodiscard]] const PipelineBindings& getBindings() const noexcept override {
    return bindings_;
  }
  /// Vertex buffer slots with attributes; draws need a buffer bound to each.
  [[nodiscard]] uint32_t getRequiredVertexBufferMask() const noexcept {
    return requiredVertexBufferMask_;
  }
  [[nodiscard]] bool hasDepthStencilAttachment() const noexcept {
    return depthStencilFormat_ != WGPUTextureFormat_Undefined;
  }
  [[nodiscard]] size_t getVariantCount() const noexcept {
    return variants_.size();
  }
  /// Number of WGPURenderPipelines created so far.
  [[nodiscard]] size_t getPipelineCreationCount() const noexcept {
    return pipelineCreationCount_;
  }

 private:
  RenderPipelineState(WebGPUContext& ctx,
                      const DeviceFeatureSet& features,
                      BindLayoutCache& layoutCache,
                      const RenderPipelineDesc& desc);

  [[nodiscard]] Result init();
  [[nodiscard]] Result createPipeline(const RenderPipelineDynamicState& state,
                                      Handle<WGPURenderPipeline>& outPipeline);

  WebGPUContext& ctx_;
  const DeviceFeatureSet& features_;
  BindLayoutCache& layoutCache_;
  std::shared_ptr<ShaderModule> vertexModule_;
  std::shared_ptr<ShaderModule> fragmentModule_;
  PipelineBindings bindings_;
  uint32_t numBindGroups_ = 0;
  uint32_t requiredVertexBufferMask_ = 0;
  WGPUTextureFormat depthStencilFormat_ = WGPUTextureFormat_Undefined;
  std::shared_ptr<RenderPipelineReflection> reflection_;
  std::unordered_map<RenderPipelineDynamicState,
                     Handle<WGPURenderPipeline>,
                     RenderPipelineDynamicStateHash>
      variants_;
  size_t pipelineCreationCount_ = 0;
};

} // namespace igl::webgpu
