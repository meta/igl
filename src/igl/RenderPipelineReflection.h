/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>
#include <igl/NameHandle.h>
#include <igl/Shader.h>
// NOLINTNEXTLINE(facebook-unused-include-check)
#include <igl/Texture.h>
#include <igl/Uniform.h>

namespace igl {

/**
 * @brief A structure that describes a buffer argument to a shader.
 */
struct BufferArgDesc {
  /**
   * @brief A structure that describes a member of a buffer argument.
   */
  struct BufferMemberDesc {
    NameHandle name; ///< The name of the member
    UniformType type = UniformType::Invalid; ///< The type of the member
    size_t offset = 0; ///< The offset from the beginning of the structure
    size_t arrayLength = 1; ///< The number of elements if the member is an array
    size_t arrayStride = 0; ///< The per-element byte stride reported by the backend for an
                            ///< array member. 0 means "not reported"; callers should derive a
                            ///< tight stride from sizeForUniformType (correct for scalars and for
                            ///< OpenGL/D3D12, which do not report it). Metal reports the real MSL
                            ///< stride here (e.g. 16 for vec3[]/mat3[]).
  };

  NameHandle name; ///< The name of the buffer argument
  size_t bufferAlignment = 0; ///< The required byte alignment in memory
  size_t bufferDataSize = 0; ///< The size of the buffer argument in bytes
  int bufferIndex = -1; ///< The index of the buffer argument

  ShaderStage shaderStage = ShaderStage::Fragment; ///< The shader stage the argument belongs to.
  bool isUniformBlock = false; // used in the OpenGL backend only

  std::vector<BufferMemberDesc> members; ///< An array of BufferMemberDesc that describes
                                         ///< each members of a buffer argument.
};

/**
 * @brief A structure that describes the texture argument to a shader.
 */
struct TextureArgDesc {
  std::string name; ///< The name of the texture argument
  TextureType type; ///< The type of the texture argument
  int textureIndex = -1; ///< The index in argument table
  ShaderStage shaderStage =
      ShaderStage::Fragment; ///< The shader stage that the texture argument belongs to.
};

/**
 * @brief A structure that describes the texture sampler argument to a shader.
 */
struct SamplerArgDesc {
  std::string name; ///< The name of the sampler argument
  int samplerIndex = -1; ///< The index in the argument table
  ShaderStage shaderStage =
      ShaderStage::Fragment; ///< The shader stage that the sampler argument belongs to
};

/**
 * @brief An interface that provides access to the information of different types of arguments to
 * a RenderPipelineState object. The RenderPipelineReflection object can be obtained by calling
 * renderPipelineReflection on RenderPipelineState.
 */
class IRenderPipelineReflection {
 public:
  /**
   * @brief A function that returns an array of BufferArgDesc that describe buffer arguments to
   * the RenderPipelineState.
   *
   * @return An array of BufferArgDesc that describes the buffer argument.
   */
  [[nodiscard]] virtual const std::vector<BufferArgDesc>& allUniformBuffers() const = 0;
  /**
   * @brief A function that returns an array of SamplerArgDesc that describe the sampler arguments
   * to the RenderPipelineState.
   *
   * @return An array of SamplerArgDesc that describes the sampler argument.
   */
  [[nodiscard]] virtual const std::vector<SamplerArgDesc>& allSamplers() const = 0;

  /**
   * @brief A function that returns an array of TextureArgDesc that describe the texture arguments
   * to the RenderPipelineState.
   *
   * @return An array of TextureArgDesc that describes the texture argument.
   */
  [[nodiscard]] virtual const std::vector<TextureArgDesc>& allTextures() const = 0;

  /**
   * @brief Destroy the IRenderPipelineReflection object
   */
  virtual ~IRenderPipelineReflection() = default;
};

} // namespace igl
