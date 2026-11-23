/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <IGLU/simdtypes/SimdTypes.h>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <igl/Common.h>
#include <igl/IGL.h>
#include <igl/NameHandle.h>
#include <igl/Shader.h>

namespace iglu::material {

/// Handles allocation, updating and binding of shader uniforms. It uses reflection
/// information to generate the underlying data and provides a simple API to manipulate it.
class ShaderUniforms final {
 public:
  /// How uniform buffer contents reach the GPU.
  enum class UploadMode : uint8_t {
    /// bind() uploads each buffer (into a Ring buffer on Metal, Vulkan and D3D12).
    AtBind,
    /// update() records the changed buffers into a command buffer with
    /// ICommandBuffer::updateBuffer(), and bind() only binds. Used only when the device has
    /// DeviceFeatures::UpdateBuffer and sub-allocation is off; otherwise behaves as AtBind.
    CommandBuffer,
  };

  // Setters: use these to update uniforms, individually or in bulk.
  void setBool(const igl::NameHandle& uniformName, const bool& value, size_t arrayIndex = 0);
  void setBool(const igl::NameHandle& blockTypeName,
               const igl::NameHandle& blockInstanceName,
               const igl::NameHandle& memberName,
               const bool& value,
               size_t arrayIndex = 0);
  void setBoolArray(const igl::NameHandle& uniformName,
                    const bool* value,
                    size_t count = 1,
                    size_t arrayIndex = 0);
  void setBoolArray(const igl::NameHandle& blockTypeName,
                    const igl::NameHandle& blockInstanceName,
                    const igl::NameHandle& memberName,
                    const bool* value,
                    size_t count = 1,
                    size_t arrayIndex = 0);

  void setFloat(const igl::NameHandle& uniformName,
                const simdtypes::float1& value,
                size_t arrayIndex = 0);
  void setFloat(const igl::NameHandle& blockTypeName,
                const igl::NameHandle& blockInstanceName,
                const igl::NameHandle& memberName,
                const simdtypes::float1& value,
                size_t arrayIndex = 0);
  void setFloatArray(const igl::NameHandle& uniformName,
                     const simdtypes::float1* value,
                     size_t count = 1,
                     size_t arrayIndex = 0);
  void setFloatArray(const igl::NameHandle& blockTypeName,
                     const igl::NameHandle& blockInstanceName,
                     const igl::NameHandle& memberName,
                     const simdtypes::float1* value,
                     size_t count = 1,
                     size_t arrayIndex = 0);

  void setFloat2(const igl::NameHandle& uniformName,
                 const simdtypes::float2& value,
                 size_t arrayIndex = 0);
  void setFloat2(const igl::NameHandle& blockTypeName,
                 const igl::NameHandle& blockInstanceName,
                 const igl::NameHandle& memberName,
                 const simdtypes::float2& value,
                 size_t arrayIndex = 0);
  void setFloat2Array(const igl::NameHandle& uniformName,
                      const simdtypes::float2* value,
                      size_t count = 1,
                      size_t arrayIndex = 0);
  void setFloat2Array(const igl::NameHandle& blockTypeName,
                      const igl::NameHandle& blockInstanceName,
                      const igl::NameHandle& memberName,
                      const simdtypes::float2* value,
                      size_t count = 1,
                      size_t arrayIndex = 0);

  void setFloat3(const igl::NameHandle& uniformName,
                 const simdtypes::float3& value,
                 size_t arrayIndex = 0);
  void setFloat3Array(const igl::NameHandle& uniformName,
                      const simdtypes::float3* value,
                      size_t count = 1,
                      size_t arrayIndex = 0);

  void setFloat4(const igl::NameHandle& uniformName,
                 const simdtypes::float4& value,
                 size_t arrayIndex = 0);
  void setFloat4(const igl::NameHandle& blockTypeName,
                 const igl::NameHandle& blockInstanceName,
                 const igl::NameHandle& memberName,
                 const simdtypes::float4& value,
                 size_t arrayIndex = 0);
  void setFloat4Array(const igl::NameHandle& uniformName,
                      const simdtypes::float4* value,
                      size_t count = 1,
                      size_t arrayIndex = 0);
  void setFloat4Array(const igl::NameHandle& blockTypeName,
                      const igl::NameHandle& blockInstanceName,
                      const igl::NameHandle& memberName,
                      const simdtypes::float4* value,
                      size_t count = 1,
                      size_t arrayIndex = 0);

  void setFloat2x2(const igl::NameHandle& uniformName,
                   const simdtypes::float2x2& value,
                   size_t arrayIndex = 0);
  void setFloat2x2(const igl::NameHandle& blockTypeName,
                   const igl::NameHandle& blockInstanceName,
                   const igl::NameHandle& memberName,
                   const simdtypes::float2x2& value,
                   size_t arrayIndex = 0);
  void setFloat2x2Array(const igl::NameHandle& uniformName,
                        const simdtypes::float2x2* value,
                        size_t count = 1,
                        size_t arrayIndex = 0);
  void setFloat2x2Array(const igl::NameHandle& blockTypeName,
                        const igl::NameHandle& blockInstanceName,
                        const igl::NameHandle& memberName,
                        const simdtypes::float2x2* value,
                        size_t count = 1,
                        size_t arrayIndex = 0);

  void setFloat3x3(const igl::NameHandle& uniformName,
                   const simdtypes::float3x3& value,
                   size_t arrayIndex = 0);
  void setFloat3x3(const igl::NameHandle& blockTypeName,
                   const igl::NameHandle& blockInstanceName,
                   const igl::NameHandle& uniformName,
                   const simdtypes::float3x3& value,
                   size_t arrayIndex = 0);
  void setFloat3x3Array(const igl::NameHandle& uniformName,
                        const simdtypes::float3x3* value,
                        size_t count = 1,
                        size_t arrayIndex = 0);
  void setFloat3x3Array(const igl::NameHandle& blockTypeName,
                        const igl::NameHandle& blockInstanceName,
                        const igl::NameHandle& memberName,
                        const simdtypes::float3x3* value,
                        size_t count = 1,
                        size_t arrayIndex = 0);

  void setFloat4x4(const igl::NameHandle& uniformName,
                   const simdtypes::float4x4& value,
                   size_t arrayIndex = 0);
  void setFloat4x4(const igl::NameHandle& blockTypeName,
                   const igl::NameHandle& blockInstanceName,
                   const igl::NameHandle& memberName,
                   const simdtypes::float4x4& value,
                   size_t arrayIndex = 0);
  void setFloat4x4Array(const igl::NameHandle& uniformName,
                        const simdtypes::float4x4* value,
                        size_t count = 1,
                        size_t arrayIndex = 0);
  void setFloat4x4Array(const igl::NameHandle& blockTypeName,
                        const igl::NameHandle& blockInstanceName,
                        const igl::NameHandle& memberName,
                        const simdtypes::float4x4* value,
                        size_t count = 1,
                        size_t arrayIndex = 0);

  void setInt(const igl::NameHandle& uniformName,
              const simdtypes::int1& value,
              size_t arrayIndex = 0);
  void setInt(const igl::NameHandle& blockTypeName,
              const igl::NameHandle& blockInstanceName,
              const igl::NameHandle& memberName,
              const simdtypes::int1& value,
              size_t arrayIndex = 0);
  void setIntArray(const igl::NameHandle& uniformName,
                   const simdtypes::int1* value,
                   size_t count = 1,
                   size_t arrayIndex = 0);
  void setIntArray(const igl::NameHandle& blockTypeName,
                   const igl::NameHandle& blockInstanceName,
                   const igl::NameHandle& memberName,
                   const simdtypes::int1* value,
                   size_t count = 1,
                   size_t arrayIndex = 0);

  void setInt2(const igl::NameHandle& uniformName,
               const simdtypes::int2& value,
               size_t arrayIndex = 0);
  void setInt2(const igl::NameHandle& blockTypeName,
               const igl::NameHandle& blockInstanceName,
               const igl::NameHandle& memberName,
               const simdtypes::int2& value,
               size_t arrayIndex = 0);

  void setTexture(const std::string& name,
                  const std::shared_ptr<igl::ITexture>& value,
                  const std::shared_ptr<igl::ISamplerState>& sampler,
                  size_t arrayIndex = 0);

  void setTexture(const std::string& name,
                  igl::ITexture* value,
                  const std::shared_ptr<igl::ISamplerState>& sampler);

  void setTexture(const std::string& name, igl::ITexture* value, igl::ISamplerState* sampler);

  /// With UploadMode::CommandBuffer, records the current contents of every uniform buffer into
  /// 'commandBuffer'. Call it after the last uniform change and before creating the encoder that
  /// binds them. Does nothing in AtBind mode.
  void update(igl::ICommandBuffer& commandBuffer);

  /// Binds all relevant states in 'encoder' in preparation for drawing.
  void bind(igl::IDevice& device,
            const igl::IRenderPipelineState& pipelineState,
            igl::IRenderCommandEncoder& encoder);

  void bind(igl::IDevice& device,
            const igl::IRenderPipelineState& pipelineState,
            igl::IRenderCommandEncoder& encoder,
            const igl::NameHandle& uniformName);

  void bind(igl::IDevice& device,
            const igl::IRenderPipelineState& pipelineState,
            igl::IRenderCommandEncoder& encoder,
            const igl::NameHandle& blockName,
            const igl::NameHandle& blockInstanceName,
            const igl::NameHandle& memberName);

  /**
   * Uniform/Storage buffers can be suballocated, for scenarios where
   * we only want to update a portion of a buffer
   *
   * Each allocation has the same size, so we don't need to track
   * the size per allocation. When setSubAllocationIndex is called followed
   * by the uniform being updated, it will only update with the offset = index * allocationSize
   *
   * @param name The uniform name
   * @param index The index within the buffer
   * @return indices whether setting the index was successful or not
   */
  igl::Result setSuballocationIndex(const igl::NameHandle& name, int index);

  inline bool containsUniform(const igl::NameHandle& uniformName) const {
    return allUniformsByName_.count(uniformName) > 0;
  }

  bool containsUniform(const igl::NameHandle& blockTypeName,
                       const igl::NameHandle& blockInstanceName,
                       const igl::NameHandle& memberName);

  class MemoizedQualifiedMemberNameCalculator {
   public:
    igl::NameHandle getQualifiedMemberName(const igl::NameHandle& blockTypeName,
                                           const igl::NameHandle& blockInstanceName,
                                           const igl::NameHandle& memberName) const;

   private:
    mutable std::unordered_map<std::pair<igl::NameHandle, igl::NameHandle>, igl::NameHandle>
        qualifiedMemberNameCache_;
  };

  igl::NameHandle getQualifiedMemberName(const igl::NameHandle& blockTypeName,
                                         const igl::NameHandle& blockInstanceName,
                                         const igl::NameHandle& memberName);

  ShaderUniforms(igl::IDevice& device,
                 const igl::IRenderPipelineReflection& reflection,
                 bool enableSuballocationforVulkan = true,
                 UploadMode uploadMode = UploadMode::AtBind);
  ~ShaderUniforms();

 private:
  struct BufferAllocation {
    void* ptr = nullptr;
    size_t size = 0;
    std::shared_ptr<igl::IBuffer> iglBuffer;
    // ptr changed since the last update(); checked by bind() (UploadMode::CommandBuffer only)
    bool dirty = false;

    BufferAllocation(void* ptr, size_t size, std::shared_ptr<igl::IBuffer> buffer) :
      ptr(ptr), size(size), iglBuffer(std::move(buffer)) {}
  };

  struct BufferDesc;
  struct UniformDesc {
    igl::BufferArgDesc::BufferMemberDesc iglMemberDesc;
    std::weak_ptr<BufferDesc> buffer;
  };
  struct BufferDesc {
    igl::BufferArgDesc iglBufferDesc;
    std::shared_ptr<BufferAllocation> allocation;
    std::vector<UniformDesc> uniforms;
    std::unordered_map<igl::NameHandle, int> memberIndices;

    // For suballocation:
    bool isSuballocated = false;
    size_t suballocationsSize = 0; // this is a fixed size
    int currentAllocation = -1; // Which allocation are we updating/binding?
    std::vector<int> suballocations;
  };

  igl::IDevice& device_;
  // UploadMode::CommandBuffer is in effect: update() uploads, bind() only binds
  bool uploadWithCommandBuffer_ = false;

  std::vector<std::shared_ptr<BufferAllocation>> allocations_;

  std::unordered_multimap<igl::NameHandle, std::shared_ptr<BufferDesc>> bufferDescs_;

  std::unordered_multimap<igl::NameHandle, UniformDesc> allUniformsByName_;

  MemoizedQualifiedMemberNameCalculator memoizedQualifiedMemberNameCalculator_;

  struct TextureSlot {
    std::shared_ptr<igl::ITexture> texture;
    igl::ITexture* rawTexture = nullptr;
  };

  struct SamplerSlot {
    std::shared_ptr<igl::ISamplerState> sampler;
    igl::ISamplerState* rawSampler = nullptr;
  };

  std::vector<igl::TextureArgDesc> textureDescs_;
  std::unordered_map<std::string, TextureSlot> allTexturesByName_;
  std::unordered_map<std::string, SamplerSlot> allSamplersByName_;

  std::vector<std::pair<igl::NameHandle, igl::NameHandle>> getPossibleBufferAndMemberNames(
      const igl::NameHandle& blockTypeName,
      const igl::NameHandle& blockInstanceName,
      const igl::NameHandle& memberName);

  void setUniformBytes(const UniformDesc& uniformDesc,
                       const void* data,
                       size_t elementSize,
                       size_t count,
                       size_t arrayIndex);

  void setUniformBytes(const igl::NameHandle& blockTypeName,
                       const igl::NameHandle& blockInstanceName,
                       const igl::NameHandle& memberName,
                       const void* data,
                       size_t elementSize,
                       size_t count,
                       size_t arrayIndex);

  void setUniformBytes(const igl::NameHandle& name,
                       const void* data,
                       size_t elementSize,
                       size_t count,
                       size_t arrayIndex);

  void bindUniformOpenGL(const igl::NameHandle& uniformName,
                         const UniformDesc& uniformDesc,
                         const igl::IRenderPipelineState& pipelineState,
                         igl::IRenderCommandEncoder& encoder);

  void bindBuffer(igl::IDevice& device,
                  const igl::IRenderPipelineState& pipelineState,
                  igl::IRenderCommandEncoder& encoder,
                  BufferDesc* buffer);
};

} // namespace iglu::material
