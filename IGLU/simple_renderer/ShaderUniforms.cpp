/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ShaderUniforms.h"

#include <igl/Buffer.h>
#include <igl/Device.h>
#include <igl/Macros.h>
#include <igl/Uniform.h>
#if IGL_BACKEND_OPENGL
// RenderPipelineState.h is included below for OpenGL-specific functionality
#include <igl/opengl/RenderPipelineState.h>
#endif
#if defined(IGL_CMAKE_BUILD)
#include <igl/IGLSafeC.h>
#else
#include <secure_lib/secure_string.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

namespace {

// For Suballocated uniform buffers, try to allocate at most a buffer of size 64K.
// We will clamp the size to the limits of the device.
// For example, on the Quest 2 GPU, maxUniformBufferSize is 64k, so we are using it all.
constexpr size_t kMaxSuballocatedBufferSizeBytes = 65536;

uint8_t bindTargetForShaderStage(igl::ShaderStage stage) {
  switch (stage) {
  case igl::ShaderStage::Vertex:
    return igl::BindTarget::kVertex;
  case igl::ShaderStage::Fragment:
    return igl::BindTarget::kFragment;
  case igl::ShaderStage::Compute:
  case igl::ShaderStage::Task:
  case igl::ShaderStage::Mesh:
  default:
    IGL_DEBUG_ABORT("invalid shader stage for rendering: %d", static_cast<int>(stage));
    return 0;
  }
}

} // namespace

namespace iglu::material {

ShaderUniforms::ShaderUniforms(igl::IDevice& device,
                               const igl::IRenderPipelineReflection& reflection,
                               bool enableSuballocationforVulkan,
                               UploadMode uploadMode) :
  device_(device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  bool hasBindBytesFeature = device.hasFeature(igl::DeviceFeatures::BindBytes);
  size_t bindBytesLimit = 0;
  if (!device.getFeatureLimits(igl::DeviceFeatureLimits::MaxBindBytesBytes, bindBytesLimit)) {
    IGL_LOG_ERROR("[IGL][Warning] Failed to get MaxBindBytesBytes value. Turning off bind bytes");
    hasBindBytesFeature = false;
  }

  size_t uniformBufferLimit = 0;
  device.getFeatureLimits(igl::DeviceFeatureLimits::MaxUniformBufferBytes, uniformBufferLimit);

  const bool isSuballocated = enableSuballocationforVulkan &&
                              device_.getBackendType() == igl::BackendType::Vulkan;
  // Sub-allocations are written between draws of one encoder, where a command buffer write is not
  // allowed.
  IGL_DEBUG_ASSERT(uploadMode == UploadMode::AtBind || !isSuballocated,
                   "UploadMode::CommandBuffer does not support sub-allocation");
  uploadWithCommandBuffer_ = uploadMode == UploadMode::CommandBuffer && !isSuballocated &&
                             device.hasFeature(igl::DeviceFeatures::UpdateBuffer);
  for (const igl::BufferArgDesc& iglDesc : reflection.allUniformBuffers()) {
    const size_t length = iglDesc.bufferDataSize;
    IGL_DEBUG_ASSERT(length > 0, "unexpected buffer with size 0");
    IGL_DEBUG_ASSERT(length <= kMaxSuballocatedBufferSizeBytes &&
                         (uniformBufferLimit == 0 || length <= uniformBufferLimit),
                     "buffer size exceeds limits");
    const size_t bufferAllocationLength =
        std::min(isSuballocated ? kMaxSuballocatedBufferSizeBytes : length,
                 uniformBufferLimit != 0 ? uniformBufferLimit : std::numeric_limits<size_t>::max());
    const std::string vertexBufferPrefix = "vertexBuffer.";
    if (device.getBackendType() == igl::BackendType::Metal &&
        iglDesc.name.toString().starts_with(vertexBufferPrefix)) {
      continue;
    }

    bool createBuffer = false;
    if (device_.getBackendType() == igl::BackendType::OpenGL) {
      // On OpenGL, create buffers only when dealing with uniform blocks (and not single uniforms)
      createBuffer = iglDesc.isUniformBlock;
    } else if (device_.getBackendType() == igl::BackendType::Vulkan) {
      createBuffer = true;
    } else if (device_.getBackendType() == igl::BackendType::Metal ||
               device_.getBackendType() == igl::BackendType::WebGPU) {
      // On Metal and WebGPU, need to create buffers only when data exceeds the bindBytes limit
      createBuffer = !hasBindBytesFeature || length > bindBytesLimit;
    } else if (device_.getBackendType() == igl::BackendType::D3D12) {
      // D3D12 does not support bindBytes, always create buffers
      createBuffer = true;
    }

    std::shared_ptr<igl::IBuffer> buffer = nullptr;
    if (createBuffer) {
      // Per IGL Error Handling rule #24, every resource creation call must
      // pass a Result* and check it; passing nullptr silently swallows errors.
      igl::Result result;
      const auto backendType = device_.getBackendType();
      // Writes recorded in command buffers are ordered with the GPU reads, so those buffers need
      // neither a Ring nor CPU access.
      const bool useRing = !uploadWithCommandBuffer_ && (backendType == igl::BackendType::Metal ||
                                                         backendType == igl::BackendType::Vulkan ||
                                                         backendType == igl::BackendType::D3D12);
      const igl::BufferDesc desc{
          .type = igl::BufferDesc::BufferTypeBits::Uniform,
          .length = bufferAllocationLength,
          .storage = uploadWithCommandBuffer_ ? igl::ResourceStorage::Private
                                              : igl::ResourceStorage::Shared,
          .hint = static_cast<igl::BufferDesc::BufferAPIHint>(
              igl::BufferDesc::BufferAPIHintBits::UniformBlock |
              (useRing ? igl::BufferDesc::BufferAPIHintBits::Ring : 0)),
      };
      buffer = device.createBuffer(desc, &result);
      IGL_DEBUG_ASSERT(result.isOk(), "createBuffer(uniform) failed: %s", result.message.c_str());
      IGL_DEBUG_ASSERT(buffer != nullptr);
    }

    // All uniform updates will be made to this malloc'ed data block,
    // which will later be uploaded to the buffer (if using buffer)
    void* data = std::malloc(bufferAllocationLength);
    if (data == nullptr) {
      continue;
    }
    auto allocation = std::make_shared<BufferAllocation>(data, bufferAllocationLength, buffer);
    // The buffer has not been written yet; bind() checks that update() ran first.
    allocation->dirty = true;
    allocations_.push_back(allocation);

    std::shared_ptr<BufferDesc> bufferDesc = std::make_shared<BufferDesc>();
    bufferDesc->iglBufferDesc = iglDesc;
    bufferDesc->allocation = allocation;

    if (isSuballocated) {
      bufferDesc->isSuballocated = true;
      // In Vulkan, Uniform buffers must have offets that are a multiple of
      // VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment
      size_t alignement = 0;
      device.getFeatureLimits(igl::DeviceFeatureLimits::BufferAlignment, alignement);

      // Align the suballocation size to the physical device alignment
      bufferDesc->suballocationsSize = (length + alignement - 1) & ~(alignement - 1);
    }

    for (int i = 0; i < static_cast<int>(iglDesc.members.size()); ++i) {
      const auto& uniformDesc = iglDesc.members[i];
      const UniformDesc uniform{.iglMemberDesc = uniformDesc, .buffer = bufferDesc};
      allUniformsByName_.insert({uniformDesc.name, uniform});
      bufferDesc->uniforms.push_back(uniform);
      bufferDesc->memberIndices[uniformDesc.name] = i;
    }

    bufferDescs_.insert({iglDesc.name, std::move(bufferDesc)});
  }

  for (const igl::TextureArgDesc& iglDesc : reflection.allTextures()) {
    textureDescs_.push_back(iglDesc);
    allTexturesByName_[iglDesc.name] = TextureSlot{.texture = nullptr, .rawTexture = nullptr};
  }
}

ShaderUniforms::~ShaderUniforms() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  for (auto& allocation : allocations_) {
    std::free(allocation->ptr);
  }
}

igl::NameHandle ShaderUniforms::MemoizedQualifiedMemberNameCalculator::getQualifiedMemberName(
    const igl::NameHandle& /*blockTypeName*/,
    const igl::NameHandle& blockInstanceName,
    const igl::NameHandle& memberName) const {
  IGL_PROFILER_FUNCTION();
  const std::pair<igl::NameHandle, igl::NameHandle> key = {blockInstanceName, memberName};
  auto it = qualifiedMemberNameCache_.find(key);
  if (it != qualifiedMemberNameCache_.end()) {
    return it->second;
  }
  auto qualifiedMemberName =
      igl::genNameHandle(blockInstanceName.toString() + "." + memberName.toString());
  qualifiedMemberNameCache_.insert({key, qualifiedMemberName});
  return qualifiedMemberName;
}

igl::NameHandle ShaderUniforms::getQualifiedMemberName(const igl::NameHandle& blockTypeName,
                                                       const igl::NameHandle& blockInstanceName,
                                                       const igl::NameHandle& memberName) {
  return memoizedQualifiedMemberNameCalculator_.getQualifiedMemberName(
      blockTypeName, blockInstanceName, memberName);
}

std::vector<std::pair<igl::NameHandle, igl::NameHandle>>
ShaderUniforms::getPossibleBufferAndMemberNames(const igl::NameHandle& blockTypeName,
                                                const igl::NameHandle& blockInstanceName,
                                                const igl::NameHandle& memberName) {
  IGL_PROFILER_FUNCTION();
  /**
    Given an SparkSL/GLSL3 interface block:
    ```
      uniform BlockTypeName {
        float f;
      } blockInstanceName;
    ```

    The corresponding Legacy GLSL code:
    ```
      struct BlockTypeName {
        float f;
      }
      uniform BlockTypeName blockInstanceName;
    ```

    The corresponding Metal code:
    ```
      struct BlockTypeName {
        float f;
      };
      main(BlockTypeName& blockInstanceName) {
        ...
      }
    ```

    In OpenGL3, the name of the buffer block is `BlockTypeName` and the member name is 'memberName'.

    In legacy OpenGL, we treat each member of the struct an individual uniform, so both the buffer
    name and member name are `blockInstanceName.f`

    In Metal, the name of the block is `blockInstanceName` and the member name is 'memberName'.
  */
  if (device_.getBackendType() == igl::BackendType::Metal) {
    return {{blockInstanceName, memberName}};
  } else {
    if (device_.getBackendType() == igl::BackendType::OpenGL) {
      auto qualifiedName =
          ShaderUniforms::getQualifiedMemberName(blockTypeName, blockInstanceName, memberName);
      return {{blockTypeName, memberName}, {qualifiedName, qualifiedName}};
    }
    return {{blockTypeName, memberName}};
  }
}

void ShaderUniforms::setUniformBytes(const UniformDesc& uniformDesc,
                                     const void* data,
                                     size_t elementSize,
                                     size_t count,
                                     size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  if (arrayIndex + count > uniformDesc.iglMemberDesc.arrayLength) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Invalid range for uniform %s:  %zu,%zu,%zu\n",
                       uniformDesc.iglMemberDesc.name.c_str(),
                       arrayIndex,
                       count,
                       uniformDesc.iglMemberDesc.arrayLength);
    return;
  }
  auto strongBuffer = uniformDesc.buffer.lock();
  if (!strongBuffer) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] null uniform buffer %s!\n",
                       uniformDesc.iglMemberDesc.name.c_str());
    return;
  }

  uintptr_t subAllocatedOffset = 0;
  if (strongBuffer->isSuballocated && strongBuffer->currentAllocation >= 0) {
    subAllocatedOffset = strongBuffer->currentAllocation * strongBuffer->suballocationsSize;
  }
  const uintptr_t offset =
      uniformDesc.iglMemberDesc.offset + elementSize * arrayIndex + subAllocatedOffset;

  auto err = try_checked_memcpy(
      static_cast<uint8_t*>(strongBuffer->allocation->ptr) + offset, // destination
      strongBuffer->allocation->size - offset, // max destination size
      data, // source
      elementSize * count // num bytes to copy
  );
  if (err != 0) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Failed to update uniform buffer\n");
    return;
  }
  strongBuffer->allocation->dirty = true;
}

void ShaderUniforms::setUniformBytes(const igl::NameHandle& blockTypeName,
                                     const igl::NameHandle& blockInstanceName,
                                     const igl::NameHandle& memberName,
                                     const void* data,
                                     size_t elementSize,
                                     size_t count,
                                     size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  auto possibleBufferNames =
      getPossibleBufferAndMemberNames(blockTypeName, blockInstanceName, memberName);

  for (auto& [bufferName, bufferMemberName] : possibleBufferNames) {
    auto range = bufferDescs_.equal_range(bufferName);
    if (range.first == range.second) {
      continue;
    }

    for (auto bufferDescIt = range.first; bufferDescIt != range.second; ++bufferDescIt) {
      auto& bufferDesc = bufferDescIt->second;
      auto memberIndexIt = bufferDesc->memberIndices.find(bufferMemberName);
      if (memberIndexIt == bufferDesc->memberIndices.end()) {
        IGL_LOG_ERROR_ONCE(
            "Member %s not found in buffer %s", bufferMemberName.c_str(), bufferName.c_str());
        continue;
      }
      auto& uniformDesc = bufferDesc->uniforms[memberIndexIt->second];
      setUniformBytes(uniformDesc, data, elementSize, count, arrayIndex);
    }
    return;
  }
  IGL_LOG_ERROR_ONCE("Buffer block not found: %s", blockTypeName.c_str());
}

void ShaderUniforms::setUniformBytes(const igl::NameHandle& name,
                                     const void* data,
                                     size_t elementSize,
                                     size_t count,
                                     size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  auto range = allUniformsByName_.equal_range(name);
  if (range.first == range.second) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Invalid uniform name: %s\n", name.c_str());
    return;
  }
  for (auto it = range.first; it != range.second; ++it) {
    auto& uniformDesc = it->second;
    setUniformBytes(uniformDesc, data, elementSize, count, arrayIndex);
  }
}

void ShaderUniforms::setBool(const igl::NameHandle& uniformName,
                             const bool& value,
                             size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(bool), 1, arrayIndex);
}

void ShaderUniforms::setBool(const igl::NameHandle& blockTypeName,
                             const igl::NameHandle& blockInstanceName,
                             const igl::NameHandle& memberName,
                             const bool& value,
                             size_t arrayIndex) {
  setUniformBytes(
      blockTypeName, blockInstanceName, memberName, &value, sizeof(bool), 1, arrayIndex);
}

void ShaderUniforms::setBoolArray(const igl::NameHandle& uniformName,
                                  const bool* value,
                                  size_t count,
                                  size_t arrayIndex) {
  setUniformBytes(uniformName, value, sizeof(bool), count, arrayIndex);
}

void ShaderUniforms::setBoolArray(const igl::NameHandle& blockTypeName,
                                  const igl::NameHandle& blockInstanceName,
                                  const igl::NameHandle& memberName,
                                  const bool* value,
                                  size_t count,
                                  size_t arrayIndex) {
  setUniformBytes(
      blockTypeName, blockInstanceName, memberName, value, sizeof(bool), count, arrayIndex);
}

void ShaderUniforms::setFloat(const igl::NameHandle& uniformName,
                              const simdtypes::float1& value,
                              size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(simdtypes::float1), 1, arrayIndex);
}

void ShaderUniforms::setFloat(const igl::NameHandle& blockTypeName,
                              const igl::NameHandle& blockInstanceName,
                              const igl::NameHandle& memberName,
                              const simdtypes::float1& value,
                              size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  &value,
                  sizeof(simdtypes::float1),
                  1,
                  arrayIndex);
}

void ShaderUniforms::setFloatArray(const igl::NameHandle& uniformName,
                                   const simdtypes::float1* value,
                                   size_t count,
                                   size_t arrayIndex) {
  setUniformBytes(uniformName, value, sizeof(simdtypes::float1), count, arrayIndex);
}

void ShaderUniforms::setFloatArray(const igl::NameHandle& blockTypeName,
                                   const igl::NameHandle& blockInstanceName,
                                   const igl::NameHandle& memberName,
                                   const simdtypes::float1* value,
                                   size_t count,
                                   size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  value,
                  sizeof(simdtypes::float1),
                  count,
                  arrayIndex);
}

void ShaderUniforms::setFloat2(const igl::NameHandle& uniformName,
                               const simdtypes::float2& value,
                               size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(simdtypes::float2), 1, arrayIndex);
}

void ShaderUniforms::setFloat2(const igl::NameHandle& blockTypeName,
                               const igl::NameHandle& blockInstanceName,
                               const igl::NameHandle& memberName,
                               const simdtypes::float2& value,
                               size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  &value,
                  sizeof(simdtypes::float2),
                  1,
                  arrayIndex);
}

void ShaderUniforms::setFloat2Array(const igl::NameHandle& uniformName,
                                    const simdtypes::float2* value,
                                    size_t count,
                                    size_t arrayIndex) {
  setUniformBytes(uniformName, value, sizeof(simdtypes::float2), count, arrayIndex);
}

void ShaderUniforms::setFloat2Array(const igl::NameHandle& blockTypeName,
                                    const igl::NameHandle& blockInstanceName,
                                    const igl::NameHandle& memberName,
                                    const simdtypes::float2* value,
                                    size_t count,
                                    size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  value,
                  sizeof(simdtypes::float2),
                  count,
                  arrayIndex);
}

void ShaderUniforms::setFloat3(const igl::NameHandle& uniformName,
                               const simdtypes::float3& value,
                               size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(float[3]), 1, arrayIndex);
}

void ShaderUniforms::setFloat3Array(const igl::NameHandle& uniformName,
                                    const simdtypes::float3* value,
                                    size_t count,
                                    size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  if (device_.getBackendType() == igl::BackendType::Metal) {
    setUniformBytes(uniformName, value, sizeof(simdtypes::float3), count, arrayIndex);
  } else {
    // simdtypes::float3 is padded to have an extra float.
    // This code path should not be used for Vulkan. (it should only be used for OpenGL when uniform
    // blocks are not used).
    const size_t size = sizeof(float) * 3u * count;
    IGL_DEBUG_ASSERT(size <= 65536);
    float* IGL_RESTRICT packedArray = reinterpret_cast<float*>(alloca(size));
    float* IGL_RESTRICT packedArrayPtr = packedArray;
    const float* paddedArrayPtr = reinterpret_cast<const float*>(value);
    for (size_t i = 0; i < count; i++) {
      for (int j = 0; j < 3; j++) {
        *packedArrayPtr++ = *paddedArrayPtr++;
      }
      paddedArrayPtr++; // padded float
    }
    setUniformBytes(uniformName, packedArray, size, 1, arrayIndex);
  }
}

void ShaderUniforms::setFloat4(const igl::NameHandle& uniformName,
                               const simdtypes::float4& value,
                               size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(simdtypes::float4), 1, arrayIndex);
}

void ShaderUniforms::setFloat4(const igl::NameHandle& blockTypeName,
                               const igl::NameHandle& blockInstanceName,
                               const igl::NameHandle& memberName,
                               const simdtypes::float4& value,
                               size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  &value,
                  sizeof(simdtypes::float4),
                  1,
                  arrayIndex);
}

void ShaderUniforms::setFloat4Array(const igl::NameHandle& uniformName,
                                    const simdtypes::float4* value,
                                    size_t count,
                                    size_t arrayIndex) {
  setUniformBytes(uniformName, value, sizeof(simdtypes::float4), count, arrayIndex);
}

void ShaderUniforms::setFloat4Array(const igl::NameHandle& blockTypeName,
                                    const igl::NameHandle& blockInstanceName,
                                    const igl::NameHandle& memberName,
                                    const simdtypes::float4* value,
                                    size_t count,
                                    size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  value,
                  sizeof(simdtypes::float4),
                  count,
                  arrayIndex);
}

void ShaderUniforms::setFloat2x2(const igl::NameHandle& uniformName,
                                 const simdtypes::float2x2& value,
                                 size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(simdtypes::float2x2), 1, arrayIndex);
}

void ShaderUniforms::setFloat2x2(const igl::NameHandle& blockTypeName,
                                 const igl::NameHandle& blockInstanceName,
                                 const igl::NameHandle& memberName,
                                 const simdtypes::float2x2& value,
                                 size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  &value,
                  sizeof(simdtypes::float2x2),
                  1,
                  arrayIndex);
}

void ShaderUniforms::setFloat2x2Array(const igl::NameHandle& uniformName,
                                      const simdtypes::float2x2* value,
                                      size_t count,
                                      size_t arrayIndex) {
  setUniformBytes(uniformName, value, sizeof(simdtypes::float2x2), count, arrayIndex);
}

void ShaderUniforms::setFloat2x2Array(const igl::NameHandle& blockTypeName,
                                      const igl::NameHandle& blockInstanceName,
                                      const igl::NameHandle& memberName,
                                      const simdtypes::float2x2* value,
                                      size_t count,
                                      size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  value,
                  sizeof(simdtypes::float2x2),
                  count,
                  arrayIndex);
}

void ShaderUniforms::setFloat3x3(const igl::NameHandle& uniformName,
                                 const simdtypes::float3x3& value,
                                 size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  if (device_.getBackendType() == igl::BackendType::Metal ||
      device_.getBackendType() == igl::BackendType::Vulkan ||
      device_.getBackendType() == igl::BackendType::WebGPU) {
    setUniformBytes(uniformName, &value, sizeof(simdtypes::float3x3), 1, arrayIndex);
  } else {
    // simdtypes::float3x3 has an extra float per float-vector.
    // Remove it so we can send the packed version to OpenGL
    float packedMatrix[9] = {0.0f};
    const auto* paddedMatrixPtr = reinterpret_cast<const float*>(&value);
    float* packedMatrixPtr = packedMatrix;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        *packedMatrixPtr++ = *paddedMatrixPtr++;
      }
      paddedMatrixPtr++; // padded float
    }
    setUniformBytes(uniformName, &packedMatrix, sizeof(packedMatrix), 1, arrayIndex);
  }
}

void ShaderUniforms::setFloat3x3(const igl::NameHandle& blockTypeName,
                                 const igl::NameHandle& blockInstanceName,
                                 const igl::NameHandle& uniformName,
                                 const simdtypes::float3x3& value,
                                 size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  const bool isOglBlock = device_.getBackendType() == igl::BackendType::OpenGL &&
                          bufferDescs_.find(blockTypeName) != bufferDescs_.end();
  if (device_.getBackendType() == igl::BackendType::Metal ||
      device_.getBackendType() == igl::BackendType::Vulkan ||
      device_.getBackendType() == igl::BackendType::WebGPU || isOglBlock) {
    setUniformBytes(blockTypeName,
                    blockInstanceName,
                    uniformName,
                    &value,
                    sizeof(simdtypes::float3x3),
                    1,
                    arrayIndex);
  } else {
    // simdtypes::float3x3 has an extra float per float-vector.
    // Remove it so we can send the packed version to OpenGL
    std::array<float, 9> packedMatrix{};
    const auto* paddedMatrixPtr = reinterpret_cast<const float*>(&value);
    float* packedMatrixPtr = packedMatrix.data();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        *packedMatrixPtr++ = *paddedMatrixPtr++;
      }
      paddedMatrixPtr++; // padded float
    }
    setUniformBytes(blockTypeName,
                    blockInstanceName,
                    uniformName,
                    &packedMatrix,
                    sizeof(packedMatrix),
                    1,
                    arrayIndex);
  }
}

void ShaderUniforms::setFloat3x3Array(const igl::NameHandle& uniformName,
                                      const simdtypes::float3x3* value,
                                      size_t count,
                                      size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  if (device_.getBackendType() == igl::BackendType::Metal ||
      device_.getBackendType() == igl::BackendType::Vulkan ||
      device_.getBackendType() == igl::BackendType::WebGPU) {
    setUniformBytes(uniformName, value, sizeof(simdtypes::float3x3), count, arrayIndex);
  } else {
    // simdtypes::float3x3 has an extra float per float-vector.
    // Remove it so we can send the packed version to OpenGL
    const auto* paddedMatrixPtr = reinterpret_cast<const float*>(value);
    auto* packedMatrix = new float[9 * count];
    auto* packedMatrixPtr = packedMatrix;
    for (size_t n = 0; n < count; n++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          *packedMatrixPtr++ = *paddedMatrixPtr++;
        }
        paddedMatrixPtr++; // skip over padded float
      }
    }
    setUniformBytes(uniformName, packedMatrix, sizeof(float) * 9, count, arrayIndex);
    delete[] packedMatrix;
  }
}

void ShaderUniforms::setFloat3x3Array(const igl::NameHandle& blockTypeName,
                                      const igl::NameHandle& blockInstanceName,
                                      const igl::NameHandle& memberName,
                                      const simdtypes::float3x3* value,
                                      size_t count,
                                      size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  auto isOglBlock = device_.getBackendType() == igl::BackendType::OpenGL &&
                    bufferDescs_.find(blockTypeName) != bufferDescs_.end();

  if (device_.getBackendType() == igl::BackendType::Metal ||
      device_.getBackendType() == igl::BackendType::Vulkan ||
      device_.getBackendType() == igl::BackendType::WebGPU || isOglBlock) {
    setUniformBytes(blockTypeName,
                    blockInstanceName,
                    memberName,
                    value,
                    sizeof(simdtypes::float3x3),
                    count,
                    arrayIndex);
  } else {
    // simdtypes::float3x3 has an extra float per float-vector.
    // Remove it so we can send the packed version to OpenGL
    const size_t size = sizeof(float) * 9u * count;
    IGL_DEBUG_ASSERT(size <= 65536);
    float* IGL_RESTRICT packedMatrix = reinterpret_cast<float*>(alloca(size));
    float* IGL_RESTRICT packedMatrixPtr = packedMatrix;

    const auto* paddedMatrixPtr = reinterpret_cast<const float*>(value);
    for (size_t n = 0; n < count; n++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          *packedMatrixPtr++ = *paddedMatrixPtr++;
        }
        paddedMatrixPtr++; // skip over padded float
      }
    }

    setUniformBytes(blockTypeName,
                    blockInstanceName,
                    memberName,
                    packedMatrix,
                    sizeof(float) * 9,
                    count,
                    arrayIndex);
  }
}

void ShaderUniforms::setFloat4x4(const igl::NameHandle& uniformName,
                                 const simdtypes::float4x4& value,
                                 size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(simdtypes::float4x4), 1, arrayIndex);
}

void ShaderUniforms::setFloat4x4(const igl::NameHandle& blockTypeName,
                                 const igl::NameHandle& blockInstanceName,
                                 const igl::NameHandle& memberName,
                                 const simdtypes::float4x4& value,
                                 size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  &value,
                  sizeof(simdtypes::float4x4),
                  1,
                  arrayIndex);
}

void ShaderUniforms::setFloat4x4Array(const igl::NameHandle& uniformName,
                                      const simdtypes::float4x4* value,
                                      size_t count,
                                      size_t arrayIndex) {
  setUniformBytes(uniformName, value, sizeof(simdtypes::float4x4), count, arrayIndex);
}

void ShaderUniforms::setFloat4x4Array(const igl::NameHandle& blockTypeName,
                                      const igl::NameHandle& blockInstanceName,
                                      const igl::NameHandle& memberName,
                                      const simdtypes::float4x4* value,
                                      size_t count,
                                      size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  value,
                  sizeof(simdtypes::float4x4),
                  count,
                  arrayIndex);
}

void ShaderUniforms::setInt(const igl::NameHandle& uniformName,
                            const simdtypes::int1& value,
                            size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(simdtypes::int1), 1, arrayIndex);
}

void ShaderUniforms::setInt(const igl::NameHandle& blockTypeName,
                            const igl::NameHandle& blockInstanceName,
                            const igl::NameHandle& memberName,
                            const simdtypes::int1& value,
                            size_t arrayIndex) {
  setUniformBytes(
      blockTypeName, blockInstanceName, memberName, &value, sizeof(simdtypes::int1), 1, arrayIndex);
}

void ShaderUniforms::setInt2(const igl::NameHandle& uniformName,
                             const simdtypes::int2& value,
                             size_t arrayIndex) {
  setUniformBytes(uniformName, &value, sizeof(simdtypes::int2), 1, arrayIndex);
}

void ShaderUniforms::setInt2(const igl::NameHandle& blockTypeName,
                             const igl::NameHandle& blockInstanceName,
                             const igl::NameHandle& memberName,
                             const simdtypes::int2& value,
                             size_t arrayIndex) {
  setUniformBytes(
      blockTypeName, blockInstanceName, memberName, &value, sizeof(simdtypes::int2), 1, arrayIndex);
}

void ShaderUniforms::setIntArray(const igl::NameHandle& uniformName,
                                 const simdtypes::int1* value,
                                 size_t count,
                                 size_t arrayIndex) {
  setUniformBytes(uniformName, value, sizeof(simdtypes::int1), count, arrayIndex);
}

void ShaderUniforms::setIntArray(const igl::NameHandle& blockTypeName,
                                 const igl::NameHandle& blockInstanceName,
                                 const igl::NameHandle& memberName,
                                 const simdtypes::int1* value,
                                 size_t count,
                                 size_t arrayIndex) {
  setUniformBytes(blockTypeName,
                  blockInstanceName,
                  memberName,
                  value,
                  sizeof(simdtypes::int1),
                  count,
                  arrayIndex);
}

void ShaderUniforms::setTexture(const std::string& name,
                                const std::shared_ptr<igl::ITexture>& value,
                                const std::shared_ptr<igl::ISamplerState>& sampler,
                                IGL_MAYBE_UNUSED size_t arrayIndex) {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(arrayIndex == 0, "texture arrays not supported");
  auto it = allTexturesByName_.find(name);
  if (it == allTexturesByName_.end()) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Invalid texture name: %s\n", name.c_str());
    return;
  }
  allTexturesByName_[name] = TextureSlot{.texture = value, .rawTexture = value.get()};
  allSamplersByName_[name] = SamplerSlot{.sampler = sampler, .rawSampler = sampler.get()};
}

void ShaderUniforms::setTexture(const std::string& name,
                                igl::ITexture* value,
                                const std::shared_ptr<igl::ISamplerState>& sampler) {
  IGL_PROFILER_FUNCTION();
  auto it = allTexturesByName_.find(name);
  if (it == allTexturesByName_.end()) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Invalid texture name: %s\n", name.c_str());
    return;
  }
  allTexturesByName_[name] = TextureSlot{.texture = nullptr, .rawTexture = value}; // non-owning
  allSamplersByName_[name] = SamplerSlot{.sampler = sampler, .rawSampler = sampler.get()}; // owning
}

void ShaderUniforms::setTexture(const std::string& name,
                                igl::ITexture* value,
                                igl::ISamplerState* sampler) {
  IGL_PROFILER_FUNCTION();
  auto it = allTexturesByName_.find(name);
  if (it == allTexturesByName_.end()) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Invalid texture name: %s\n", name.c_str());
    return;
  }
  allTexturesByName_[name] = TextureSlot{.texture = nullptr, .rawTexture = value}; // non-owning
  allSamplersByName_[name] = SamplerSlot{.sampler = nullptr, .rawSampler = sampler}; // non-owning
}

#if IGL_BACKEND_OPENGL
void ShaderUniforms::bindUniformOpenGL(const igl::NameHandle& uniformName,
                                       const UniformDesc& uniformDesc,
                                       const igl::IRenderPipelineState& pipelineState,
                                       igl::IRenderCommandEncoder& encoder) {
  IGL_PROFILER_FUNCTION();
  const igl::BufferArgDesc::BufferMemberDesc& iglMemberDesc = uniformDesc.iglMemberDesc;
  igl::UniformDesc desc{
      .location = pipelineState.getIndexByName(uniformName, igl::ShaderStage::Fragment),
      .type = iglMemberDesc.type,
      .numElements = iglMemberDesc.arrayLength,
      .offset = iglMemberDesc.offset,
      .elementStride = igl::sizeForUniformType(iglMemberDesc.type),
  };

  if (desc.location >= 0) {
    auto strongBuffer = uniformDesc.buffer.lock();
    if (strongBuffer) {
      // We are binding individual uniforms. Confirm that iglBuffer is null
      IGL_DEBUG_ASSERT(strongBuffer->allocation->iglBuffer == nullptr);
      encoder.bindUniform(desc, strongBuffer->allocation->ptr);
    }
  } else {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Uniform not found in shader: %s\n", uniformName.c_str());
  }
}
#endif

void ShaderUniforms::bindBuffer(igl::IDevice& device,
                                const igl::IRenderPipelineState& pipelineState,
                                igl::IRenderCommandEncoder& encoder,
                                BufferDesc* buffer) {
  IGL_PROFILER_FUNCTION();
  if (!buffer) {
    return;
  }
  if (device.getBackendType() == igl::BackendType::OpenGL) {
#if IGL_BACKEND_OPENGL
    const auto& uniformName = buffer->iglBufferDesc.name;
    if (buffer->iglBufferDesc.isUniformBlock) {
      IGL_DEBUG_ASSERT(buffer->allocation->iglBuffer != nullptr);
      buffer->allocation->iglBuffer->upload(buffer->allocation->ptr,
                                            igl::BufferRange(buffer->allocation->size, 0));
      const auto& glPipelineState =
          static_cast<const igl::opengl::RenderPipelineState&>(pipelineState);
      encoder.bindBuffer(glPipelineState.getUniformBlockBindingPoint(uniformName),
                         buffer->allocation->iglBuffer.get());
    } else {
      // not a uniform block
      IGL_DEBUG_ASSERT(buffer->iglBufferDesc.name == buffer->iglBufferDesc.members[0].name);
      IGL_DEBUG_ASSERT(buffer->uniforms.size() == 1);
      IGL_DEBUG_ASSERT(buffer->iglBufferDesc.name == buffer->uniforms[0].iglMemberDesc.name);
      auto& uniformDesc = buffer->uniforms[0];

      bindUniformOpenGL(uniformName, uniformDesc, pipelineState, encoder);
    }
#endif
  } else {
    if (buffer->allocation->iglBuffer && uploadWithCommandBuffer_) {
      // A change made after update() has not reached the buffer; the draw reads the older values.
      if (!IGL_DEBUG_VERIFY(!buffer->allocation->dirty,
                            "uniform buffer changed after update(): call update() before "
                            "creating the encoder")) {
        IGL_LOG_ERROR_ONCE("[IGL][Error] Uniform buffer %s changed after update()\n",
                           buffer->iglBufferDesc.name.c_str());
      }
      encoder.bindBuffer(buffer->iglBufferDesc.bufferIndex, buffer->allocation->iglBuffer.get());
    } else if (buffer->allocation->iglBuffer) {
      uintptr_t subAllocatedOffset = 0;
      size_t uploadSize = buffer->allocation->size;
      if (buffer->isSuballocated && buffer->currentAllocation >= 0) {
        subAllocatedOffset = buffer->currentAllocation * buffer->suballocationsSize;
        uploadSize = buffer->suballocationsSize;
      }

      buffer->allocation->iglBuffer->upload(
          static_cast<uint8_t*>(buffer->allocation->ptr) + subAllocatedOffset,
          igl::BufferRange(uploadSize, subAllocatedOffset));
      encoder.bindBuffer(buffer->iglBufferDesc.bufferIndex,
                         buffer->allocation->iglBuffer.get(),
                         subAllocatedOffset);
    } else {
      encoder.bindBytes(buffer->iglBufferDesc.bufferIndex,
                        bindTargetForShaderStage(buffer->iglBufferDesc.shaderStage),
                        buffer->allocation->ptr,
                        buffer->iglBufferDesc.bufferDataSize);
    }
  }
}

void ShaderUniforms::update(igl::ICommandBuffer& commandBuffer) {
  IGL_PROFILER_FUNCTION();
  if (!uploadWithCommandBuffer_) {
    return;
  }
  for (const auto& allocation : allocations_) {
    // Every buffer is written on every call, so a command buffer that is dropped before submission
    // cannot leave a buffer behind its CPU copy.
    if (!allocation->iglBuffer) {
      continue;
    }
    commandBuffer.updateBuffer(
        *allocation->iglBuffer, igl::BufferRange(allocation->size, 0), allocation->ptr);
    allocation->dirty = false;
  }
}

// Bind the block which the specified uniform belongs to.
void ShaderUniforms::bind(igl::IDevice& device,
                          const igl::IRenderPipelineState& pipelineState,
                          igl::IRenderCommandEncoder& encoder,
                          const igl::NameHandle& uniformName) {
  IGL_PROFILER_FUNCTION();
  auto range = allUniformsByName_.equal_range(uniformName);
  if (range.first == range.second) {
    IGL_LOG_ERROR_ONCE("[IGL][Error] Invalid uniform name: %s\n", uniformName.c_str());
    return;
  }

  for (auto it = range.first; it != range.second; ++it) {
    auto strongBuffer = it->second.buffer.lock();
    bindBuffer(device, pipelineState, encoder, strongBuffer.get());
  }
}

void ShaderUniforms::bind(igl::IDevice& device,
                          const igl::IRenderPipelineState& pipelineState,
                          igl::IRenderCommandEncoder& encoder,
                          const igl::NameHandle& blockName,
                          const igl::NameHandle& blockInstanceName,
                          const igl::NameHandle& memberName) {
  IGL_PROFILER_FUNCTION();
  auto possibleBufferNames =
      getPossibleBufferAndMemberNames(blockName, blockInstanceName, memberName);
  for (auto& [bufferName, bufferMemberName] : possibleBufferNames) {
    auto range = bufferDescs_.equal_range(bufferName);
    for (auto bufferDescIt = range.first; bufferDescIt != range.second; ++bufferDescIt) {
      bindBuffer(device, pipelineState, encoder, bufferDescIt->second.get());
    }
  }
}

void ShaderUniforms::bind(igl::IDevice& device,
                          const igl::IRenderPipelineState& pipelineState,
                          igl::IRenderCommandEncoder& encoder) {
  IGL_PROFILER_FUNCTION();
  for (auto& [name, bufferDesc] : bufferDescs_) {
    bindBuffer(device, pipelineState, encoder, bufferDesc.get());
  }

  for (auto& textureDesc : textureDescs_) {
    auto textureIt = allTexturesByName_.find(textureDesc.name);
    auto samplerIt = allSamplersByName_.find(textureDesc.name);
    if (textureIt == allTexturesByName_.end() || samplerIt == allSamplersByName_.end()) {
      IGL_LOG_ERROR_ONCE("[IGL][Warning] No texture set for sampler: %s\n",
                         textureDesc.name.c_str());
      continue;
    }
    encoder.bindTexture(textureDesc.textureIndex,
                        bindTargetForShaderStage(textureDesc.shaderStage),
                        textureIt->second.rawTexture ? textureIt->second.rawTexture
                                                     : textureIt->second.texture.get());

    // Assumption: each texture has an associated sampler at the same index in Metal
    encoder.bindSamplerState(textureDesc.textureIndex,
                             bindTargetForShaderStage(textureDesc.shaderStage),
                             samplerIt->second.rawSampler ? samplerIt->second.rawSampler
                                                          : samplerIt->second.sampler.get());
  }
}

igl::Result ShaderUniforms::setSuballocationIndex(const igl::NameHandle& name, int index) {
  IGL_PROFILER_FUNCTION();
  if (device_.getBackendType() != igl::BackendType::Vulkan) {
    return igl::Result(igl::Result::Code::Unsupported,
                       "Suballocation is only available for Vulkan for now");
  }

  if (index < 0) {
    return igl::Result(igl::Result::Code::ArgumentOutOfRange,
                       "Invalid argument, index cannot be < 0");
  }

  auto range = allUniformsByName_.equal_range(name);
  if (range.first == range.second) {
    return igl::Result(igl::Result::Code::RuntimeError,
                       "Could not find uniform " + name.toString());
  }

  // At least one of the uniforms should be updated
  bool setIndexSuccess = false;

  for (auto it = range.first; it != range.second; ++it) {
    auto& uniformDesc = it->second;

    auto strongBuffer = uniformDesc.buffer.lock();

    if (!strongBuffer || !strongBuffer->isSuballocated) {
      continue;
    }

    // if index already exists, just update the allocation index
    if (std::find(strongBuffer->suballocations.begin(),
                  strongBuffer->suballocations.end(),
                  index) != strongBuffer->suballocations.end()) {
      strongBuffer->currentAllocation = index;
    } else {
      // Add new allocation
      // Make sure we have enough space
      auto currentSize = strongBuffer->suballocations.size() * strongBuffer->suballocationsSize;
      if (currentSize + strongBuffer->suballocationsSize > strongBuffer->allocation->size) {
        return igl::Result(igl::Result::Code::ArgumentOutOfRange,
                           "Cannot add new suballocation, exceeding buffer size of " +
                               std::to_string(strongBuffer->allocation->size));
      }

      strongBuffer->currentAllocation = index;
      strongBuffer->suballocations.push_back(index);
    }

    setIndexSuccess = true;
  }

  if (setIndexSuccess) {
    return igl::Result();
  } else {
    return igl::Result(igl::Result::Code::RuntimeError,
                       "Could not update suballocation index for " + name.toString());
  }
}

bool ShaderUniforms::containsUniform(const igl::NameHandle& blockTypeName,
                                     const igl::NameHandle& blockInstanceName,
                                     const igl::NameHandle& memberName) {
  IGL_PROFILER_FUNCTION();
  auto possibleBufferNames =
      getPossibleBufferAndMemberNames(blockTypeName, blockInstanceName, memberName);

  for (auto& [bufferName, bufferMemberName] : possibleBufferNames) {
    auto bufferDescIt = bufferDescs_.find(bufferName);
    if (bufferDescIt == bufferDescs_.end()) {
      continue;
    }
    auto& bufferDesc = bufferDescIt->second;
    if (bufferDesc->memberIndices.find(bufferMemberName) != bufferDesc->memberIndices.end()) {
      return true;
    }
  }
  return false;
}
} // namespace iglu::material
