/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/webgpu/BindLayouts.h>

#include <algorithm>
#include <igl/webgpu/WebGPUContext.h>

namespace igl::webgpu {

namespace {

bool isSampledTexture(WgslBindingKind kind) {
  return kind == WgslBindingKind::Texture || kind == WgslBindingKind::DepthTexture ||
         kind == WgslBindingKind::MultisampledTexture;
}

bool isSampler(WgslBindingKind kind) {
  return kind == WgslBindingKind::Sampler || kind == WgslBindingKind::ComparisonSampler;
}

bool isBuffer(WgslBindingKind kind) {
  return kind == WgslBindingKind::UniformBuffer || kind == WgslBindingKind::StorageBuffer ||
         kind == WgslBindingKind::ReadOnlyStorageBuffer;
}

bool isFloat32(TextureFormat format) {
  return format == TextureFormat::R_F32 || format == TextureFormat::RG_F32 ||
         format == TextureFormat::RGBA_F32;
}

WGPUTextureSampleType toWGPUSampleType(SampleClass sampleClass) {
  switch (sampleClass) {
  case SampleClass::Float:
    return WGPUTextureSampleType_Float;
  case SampleClass::UnfilterableFloat:
    return WGPUTextureSampleType_UnfilterableFloat;
  case SampleClass::Depth:
    return WGPUTextureSampleType_Depth;
  case SampleClass::Sint:
    return WGPUTextureSampleType_Sint;
  case SampleClass::Uint:
    return WGPUTextureSampleType_Uint;
  }
  IGL_UNREACHABLE_RETURN(WGPUTextureSampleType_Float)
}

Result checkConvention(const WgslBinding& binding) {
  const auto violation = [&binding](const std::string& what) {
    return Result(Result::Code::ArgumentInvalid,
                  "@group(" + std::to_string(binding.group) + ") @binding(" +
                      std::to_string(binding.binding) + ") " + binding.name + ": " + what);
  };
  switch (binding.group) {
  case kTextureGroup:
    if (binding.binding >= 2 * kMaxTextureUnits) {
      return violation("texture units are 0-15 (bindings 0-31)");
    }
    if (binding.binding % 2 == 0 && !isSampledTexture(binding.kind)) {
      return violation("even bindings of group 0 are textures");
    }
    if (binding.binding % 2 == 1 && !isSampler(binding.kind)) {
      return violation("odd bindings of group 0 are samplers");
    }
    return Result();
  case kBufferGroup:
    return isBuffer(binding.kind) ? Result() : violation("group 1 holds buffers");
  case kStorageTextureGroup:
    if (binding.binding >= kMaxStorageTextures) {
      return violation("storage textures are bindings 0-" +
                       std::to_string(kMaxStorageTextures - 1));
    }
    return binding.kind == WgslBindingKind::StorageTexture
               ? Result()
               : violation("group 2 holds storage textures");
  case kPushConstantGroup:
    if (binding.binding != 0 || binding.kind != WgslBindingKind::UniformBuffer) {
      return violation("group 3 holds the push-constant uniform buffer at binding 0");
    }
    return binding.bufferSize <= kMaxPushConstantBytes
               ? Result()
               : violation("push constants are at most 128 bytes");
  default:
    return violation("the WebGPU backend uses bind groups 0-3");
  }
}

void appendKey(std::string& key, uint32_t value) {
  key.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

} // namespace

std::optional<SampleClass> getSampleClass(const WgslBinding& declaration,
                                          TextureFormat format,
                                          bool float32Filterable) {
  const TextureFormatProperties props = TextureFormatProperties::fromTextureFormat(format);
  const bool isUint = props.isInteger() || props.isStencilOnly();
  switch (declaration.kind) {
  case WgslBindingKind::DepthTexture:
    if (props.hasDepth()) {
      return SampleClass::Depth;
    }
    return std::nullopt;
  case WgslBindingKind::Texture:
  case WgslBindingKind::MultisampledTexture:
    switch (declaration.sampledType) {
    case WgslSampledType::Float:
      if (isUint) {
        return std::nullopt;
      }
      // Depth formats are sampled through their depth aspect, which is never filterable.
      if (props.hasDepth() || declaration.kind == WgslBindingKind::MultisampledTexture ||
          (isFloat32(format) && !float32Filterable)) {
        return SampleClass::UnfilterableFloat;
      }
      return SampleClass::Float;
    case WgslSampledType::Uint:
      return isUint ? std::optional(SampleClass::Uint) : std::nullopt;
    case WgslSampledType::Sint:
      // IGL has no signed integer texture formats.
      return std::nullopt;
    }
    return std::nullopt;
  case WgslBindingKind::UniformBuffer:
  case WgslBindingKind::StorageBuffer:
  case WgslBindingKind::ReadOnlyStorageBuffer:
  case WgslBindingKind::StorageTexture:
  case WgslBindingKind::Sampler:
  case WgslBindingKind::ComparisonSampler:
    return std::nullopt;
  }
  IGL_UNREACHABLE_RETURN(std::nullopt)
}

SampleClass getDefaultSampleClass(const WgslBinding& declaration) {
  if (declaration.kind == WgslBindingKind::DepthTexture) {
    return SampleClass::Depth;
  }
  switch (declaration.sampledType) {
  case WgslSampledType::Float:
    return declaration.kind == WgslBindingKind::MultisampledTexture ? SampleClass::UnfilterableFloat
                                                                    : SampleClass::Float;
  case WgslSampledType::Sint:
    return SampleClass::Sint;
  case WgslSampledType::Uint:
    return SampleClass::Uint;
  }
  IGL_UNREACHABLE_RETURN(SampleClass::Float)
}

Result PipelineBindings::add(const WgslReflection& reflection, WGPUShaderStage stage) {
  for (const WgslBinding& binding : reflection.bindings) {
    Result result = checkConvention(binding);
    if (!result.isOk()) {
      return result;
    }
    // Vertex shaders cannot write storage resources; a declaration shared with other stages
    // through a library module must not make the layout invalid.
    const bool writable = binding.kind == WgslBindingKind::StorageBuffer ||
                          (binding.kind == WgslBindingKind::StorageTexture &&
                           binding.storageAccess != WGPUStorageTextureAccess_ReadOnly);
    const WGPUShaderStage visibility =
        writable ? (stage & ~static_cast<WGPUShaderStage>(WGPUShaderStage_Vertex)) : stage;
    auto& group = groups[binding.group];
    const auto it = std::find_if(group.begin(), group.end(), [&binding](const auto& b) {
      return b.declaration.binding == binding.binding;
    });
    if (it == group.end()) {
      group.push_back({.declaration = binding, .visibility = visibility});
      continue;
    }
    const WgslBinding& existing = it->declaration;
    if (existing.kind != binding.kind || existing.viewDimension != binding.viewDimension ||
        existing.sampledType != binding.sampledType ||
        existing.storageFormat != binding.storageFormat ||
        existing.storageAccess != binding.storageAccess) {
      return Result(Result::Code::ArgumentInvalid,
                    "@group(" + std::to_string(binding.group) + ") @binding(" +
                        std::to_string(binding.binding) +
                        ") is declared differently by two shader stages");
    }
    it->visibility |= visibility;
    // Stages may declare buffer structs of different sizes (e.g. per-stage push constant blocks);
    // the binding must cover the largest.
    it->declaration.bufferSize = std::max(it->declaration.bufferSize, binding.bufferSize);
  }
  for (auto& group : groups) {
    std::sort(group.begin(), group.end(), [](const auto& a, const auto& b) {
      return a.declaration.binding < b.declaration.binding;
    });
  }
  return Result();
}

void PipelineBindings::assignDynamicOffsets(uint32_t maxDynamicUniformBuffers) {
  uint32_t count = 0;
  for (PipelineBinding& binding : groups[kPushConstantGroup]) {
    binding.hasDynamicOffset = true;
    ++count;
  }
  for (PipelineBinding& binding : groups[kBufferGroup]) {
    binding.hasDynamicOffset = binding.declaration.kind == WgslBindingKind::UniformBuffer &&
                               count < maxDynamicUniformBuffers;
    count += binding.hasDynamicOffset ? 1 : 0;
  }
}

uint32_t PipelineBindings::getTextureUnitMask() const {
  uint32_t mask = 0;
  for (const PipelineBinding& binding : groups[kTextureGroup]) {
    if (isSampledTexture(binding.declaration.kind)) {
      mask |= 1u << (binding.declaration.binding / 2);
    }
  }
  return mask;
}

const PipelineBinding* IGL_NULLABLE PipelineBindings::find(uint32_t group, uint32_t binding) const {
  if (group >= kNumBindGroups) {
    return nullptr;
  }
  const auto it =
      std::find_if(groups[group].begin(), groups[group].end(), [binding](const auto& b) {
        return b.declaration.binding == binding;
      });
  return it != groups[group].end() ? &*it : nullptr;
}

SampleClasses PipelineBindings::getDefaultSampleClasses() const {
  SampleClasses classes = 0;
  for (const PipelineBinding& binding : groups[kTextureGroup]) {
    if (isSampledTexture(binding.declaration.kind)) {
      classes = setSampleClass(
          classes, binding.declaration.binding / 2, getDefaultSampleClass(binding.declaration));
    }
  }
  return classes;
}

std::vector<WGPUBindGroupLayoutEntry> makeBindGroupLayoutEntries(const PipelineBindings& bindings,
                                                                 uint32_t group,
                                                                 SampleClasses classes) {
  std::vector<WGPUBindGroupLayoutEntry> entries;
  for (const PipelineBinding& binding : bindings.groups[group]) {
    const WgslBinding& declaration = binding.declaration;
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = declaration.binding;
    entry.visibility = binding.visibility;
    switch (declaration.kind) {
    case WgslBindingKind::UniformBuffer:
      entry.buffer.type = WGPUBufferBindingType_Uniform;
      entry.buffer.hasDynamicOffset = binding.hasDynamicOffset ? 1u : 0u;
      break;
    case WgslBindingKind::StorageBuffer:
      entry.buffer.type = WGPUBufferBindingType_Storage;
      break;
    case WgslBindingKind::ReadOnlyStorageBuffer:
      entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
      break;
    case WgslBindingKind::Texture:
    case WgslBindingKind::DepthTexture:
    case WgslBindingKind::MultisampledTexture:
      entry.texture.sampleType = toWGPUSampleType(getSampleClass(classes, declaration.binding / 2));
      entry.texture.viewDimension = declaration.viewDimension;
      entry.texture.multisampled = declaration.kind == WgslBindingKind::MultisampledTexture ? 1u
                                                                                            : 0u;
      break;
    case WgslBindingKind::Sampler: {
      // A sampler filters only when the texture of its unit is filterable.
      const PipelineBinding* texture = bindings.find(group, declaration.binding - 1);
      const bool filtering = texture == nullptr ||
                             getSampleClass(classes, declaration.binding / 2) == SampleClass::Float;
      entry.sampler.type = filtering ? WGPUSamplerBindingType_Filtering
                                     : WGPUSamplerBindingType_NonFiltering;
      break;
    }
    case WgslBindingKind::ComparisonSampler:
      entry.sampler.type = WGPUSamplerBindingType_Comparison;
      break;
    case WgslBindingKind::StorageTexture:
      entry.storageTexture.access = declaration.storageAccess;
      entry.storageTexture.format = declaration.storageFormat;
      entry.storageTexture.viewDimension = declaration.viewDimension;
      break;
    }
    entries.push_back(entry);
  }
  return entries;
}

WGPUBindGroupLayout IGL_NULLABLE
BindLayoutCache::getBindGroupLayout(const PipelineBindings& bindings,
                                    uint32_t group,
                                    SampleClasses classes,
                                    Result* IGL_NULLABLE outResult) {
  const std::vector<WGPUBindGroupLayoutEntry> entries =
      makeBindGroupLayoutEntries(bindings, group, classes);
  std::string key;
  for (const WGPUBindGroupLayoutEntry& entry : entries) {
    for (const uint32_t value : {entry.binding,
                                 static_cast<uint32_t>(entry.visibility),
                                 static_cast<uint32_t>(entry.buffer.type),
                                 static_cast<uint32_t>(entry.buffer.hasDynamicOffset),
                                 static_cast<uint32_t>(entry.sampler.type),
                                 static_cast<uint32_t>(entry.texture.sampleType),
                                 static_cast<uint32_t>(entry.texture.viewDimension),
                                 static_cast<uint32_t>(entry.texture.multisampled),
                                 static_cast<uint32_t>(entry.storageTexture.access),
                                 static_cast<uint32_t>(entry.storageTexture.format),
                                 static_cast<uint32_t>(entry.storageTexture.viewDimension)}) {
      appendKey(key, value);
    }
  }
  if (const auto it = bindGroupLayouts_.find(key); it != bindGroupLayouts_.end()) {
    Result::setOk(outResult);
    return it->second.get();
  }
  WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  desc.entryCount = entries.size();
  desc.entries = entries.data();
  ctx_.pushErrorScope(WGPUErrorFilter_Validation);
  Handle<WGPUBindGroupLayout> layout(wgpuDeviceCreateBindGroupLayout(ctx_.getDevice(), &desc));
  Result result = ctx_.popErrorScope();
  if (!result.isOk() || !layout) {
    Result::setResult(outResult, std::move(result));
    return nullptr;
  }
  Result::setOk(outResult);
  return bindGroupLayouts_.emplace(std::move(key), std::move(layout)).first->second.get();
}

WGPUPipelineLayout IGL_NULLABLE
BindLayoutCache::getPipelineLayout(const std::vector<WGPUBindGroupLayout>& groups,
                                   Result* IGL_NULLABLE outResult) {
  if (const auto it = pipelineLayouts_.find(groups); it != pipelineLayouts_.end()) {
    Result::setOk(outResult);
    return it->second.get();
  }
  WGPUPipelineLayoutDescriptor desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  desc.bindGroupLayoutCount = groups.size();
  desc.bindGroupLayouts = groups.data();
  ctx_.pushErrorScope(WGPUErrorFilter_Validation);
  Handle<WGPUPipelineLayout> layout(wgpuDeviceCreatePipelineLayout(ctx_.getDevice(), &desc));
  Result result = ctx_.popErrorScope();
  if (!result.isOk() || !layout) {
    Result::setResult(outResult, std::move(result));
    return nullptr;
  }
  Result::setOk(outResult);
  return pipelineLayouts_.emplace(groups, std::move(layout)).first->second.get();
}

} // namespace igl::webgpu
