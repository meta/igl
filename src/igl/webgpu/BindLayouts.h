/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <igl/webgpu/Common.h>
#include <igl/webgpu/WgslReflection.h>

namespace igl::webgpu {

class WebGPUContext;

/// Maximum number of texture units: bindings 2i and 2i+1 of group 0 for i < kMaxTextureUnits.
inline constexpr uint32_t kMaxTextureUnits = 16;
/// Storage texture i is binding i of group 2 for i < kMaxStorageTextures. This bounds the binding
/// numbers only: how many storage textures one stage may use is the device's
/// maxStorageTexturesPerShaderStage limit (4 by default), which WebGPU checks at pipeline creation.
inline constexpr uint32_t kMaxStorageTextures = 16;

/// @brief How a texture binding is sampled, which fixes its bind group layout entry and the kind
/// of sampler it pairs with. The same WGSL declaration can need different classes depending on
/// the bound texture: `texture_2d<f32>` is Float for filterable formats and UnfilterableFloat for
/// r32float without `float32-filterable` or for depth formats.
enum class SampleClass : uint8_t { Float = 0, UnfilterableFloat, Depth, Sint, Uint };

/// Sample classes of the texture units of a draw, 4 bits per unit.
using SampleClasses = uint64_t;

[[nodiscard]] inline SampleClass getSampleClass(SampleClasses classes, uint32_t unit) {
  return static_cast<SampleClass>((classes >> (4 * unit)) & 0xf);
}

[[nodiscard]] inline SampleClasses setSampleClass(SampleClasses classes,
                                                  uint32_t unit,
                                                  SampleClass sampleClass) {
  const auto shift = 4 * unit;
  return (classes & ~(SampleClasses{0xf} << shift)) |
         (static_cast<SampleClasses>(sampleClass) << shift);
}

/// The sample-type table: the class a texture of `format` needs when bound to `declaration`, or
/// std::nullopt when WebGPU cannot bind it there (an integer format to an f32 declaration, a
/// color format to a depth declaration, ...).
[[nodiscard]] std::optional<SampleClass> getSampleClass(const WgslBinding& declaration,
                                                        TextureFormat format,
                                                        bool float32Filterable);

/// The class of a declaration when nothing is bound to it (a dummy texture is bound instead).
[[nodiscard]] SampleClass getDefaultSampleClass(const WgslBinding& declaration);

/// @brief One binding of a pipeline's bind group layout, merged from all stages that declare it.
struct PipelineBinding {
  WgslBinding declaration;
  WGPUShaderStage visibility = WGPUShaderStage_None;
  /// Uniform buffers of group 1 use dynamic offsets, up to the device limit.
  bool hasDynamicOffset = false;
};

/// @brief The bindings a render or compute pipeline declares, per bind group, sorted by binding.
struct PipelineBindings {
  std::array<std::vector<PipelineBinding>, kNumBindGroups> groups;

  /// Merges the declarations of `stage`. Returns ArgumentInvalid when two stages declare the same
  /// binding differently, and Unsupported for bindings outside the bind convention.
  [[nodiscard]] Result add(const WgslReflection& reflection, WGPUShaderStage stage);
  /// Marks the push-constant buffer and then group 1 uniform buffers (in binding order) dynamic, up
  /// to `maxDynamicUniformBuffers` in total.
  void assignDynamicOffsets(uint32_t maxDynamicUniformBuffers);
  /// Declared texture units (group 0 even bindings), as a bit mask.
  [[nodiscard]] uint32_t getTextureUnitMask() const;
  [[nodiscard]] const PipelineBinding* IGL_NULLABLE find(uint32_t group, uint32_t binding) const;
  /// Classes of every declared texture unit when nothing is bound.
  [[nodiscard]] SampleClasses getDefaultSampleClasses() const;
};

/// @brief What ResourcesBinder needs from a render or compute pipeline: its declared bindings and
/// the bind group layouts for a set of sample classes.
class PipelineLayoutSource {
 public:
  virtual ~PipelineLayoutSource() = default;
  [[nodiscard]] virtual const PipelineBindings& getBindings() const noexcept = 0;
  /// The bind group layout of `group` for texture units sampled as `classes`.
  [[nodiscard]] virtual WGPUBindGroupLayout IGL_NULLABLE
  getBindGroupLayout(uint32_t group, SampleClasses classes, Result* IGL_NULLABLE outResult) = 0;
};

/// @brief Deduplicates bind group layouts and pipeline layouts structurally.
class BindLayoutCache final {
 public:
  explicit BindLayoutCache(const WebGPUContext& ctx) : ctx_(ctx) {}

  /// The layout of `group` of `bindings` for texture units sampled as `classes`. Groups without
  /// bindings share one empty layout.
  [[nodiscard]] WGPUBindGroupLayout IGL_NULLABLE
  getBindGroupLayout(const PipelineBindings& bindings,
                     uint32_t group,
                     SampleClasses classes,
                     Result* IGL_NULLABLE outResult);
  /// The pipeline layout over groups 0..(number of groups - 1).
  [[nodiscard]] WGPUPipelineLayout IGL_NULLABLE
  getPipelineLayout(const std::vector<WGPUBindGroupLayout>& groups, Result* IGL_NULLABLE outResult);

  [[nodiscard]] size_t getBindGroupLayoutCount() const noexcept {
    return bindGroupLayouts_.size();
  }
  [[nodiscard]] size_t getPipelineLayoutCount() const noexcept {
    return pipelineLayouts_.size();
  }

 private:
  const WebGPUContext& ctx_;
  std::map<std::string, Handle<WGPUBindGroupLayout>> bindGroupLayouts_;
  std::map<std::vector<WGPUBindGroupLayout>, Handle<WGPUPipelineLayout>> pipelineLayouts_;
};

/// Bind group layout entries of `group` (exposed for tests).
[[nodiscard]] std::vector<WGPUBindGroupLayoutEntry>
makeBindGroupLayoutEntries(const PipelineBindings& bindings, uint32_t group, SampleClasses classes);

} // namespace igl::webgpu
