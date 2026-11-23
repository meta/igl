/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RenderPipelineReflection.h"

#include <algorithm>
#include <cstring>
#include <igl/Macros.h>
#include <igl/opengl/GLIncludes.h>

namespace {

igl::UniformType toIGLUniformType(GLenum type) {
  switch (type) {
  case GL_FLOAT:
    return igl::UniformType::Float;
  case GL_FLOAT_VEC2:
    return igl::UniformType::Float2;
  case GL_FLOAT_VEC3:
    return igl::UniformType::Float3;
  case GL_FLOAT_VEC4:
    return igl::UniformType::Float4;
  case GL_BOOL:
    return igl::UniformType::Boolean;
  case GL_INT:
    return igl::UniformType::Int;
  case GL_INT_VEC2:
    return igl::UniformType::Int2;
  case GL_INT_VEC3:
    return igl::UniformType::Int3;
  case GL_INT_VEC4:
    return igl::UniformType::Int4;
  case GL_FLOAT_MAT2:
    return igl::UniformType::Mat2x2;
  case GL_FLOAT_MAT3:
    return igl::UniformType::Mat3x3;
  case GL_FLOAT_MAT4:
    return igl::UniformType::Mat4x4;
  default:
    IGL_LOG_ERROR("Unsupported uniform type: 0x%04x\n", type);
    return igl::UniformType::Invalid;
  }
}

igl::TextureType toIGLTextureType(GLenum type) {
  switch (type) {
  case GL_SAMPLER_2D:

#if (defined(GL_VERSION_2_0) && GL_VERSION_2_0) || (defined(GL_ES_VERSION_3_0) && GL_ES_VERSION_3_0)
  case GL_SAMPLER_2D_SHADOW:
#elif defined(GL_ES_VERSION_2_0) && GL_ES_VERSION_2_0
  case GL_SAMPLER_2D_SHADOW_EXT:
#elif defined(GL_ARB_shader_objects) && GL_ARB_shader_objects
  case GL_SAMPLER_2D_SHADOW_ARB:
#endif
  case GL_IMAGE_2D:
  case GL_IMAGE_2D_MULTISAMPLE:
    return igl::TextureType::TwoD;
  case GL_SAMPLER_2D_ARRAY:
  case GL_IMAGE_2D_ARRAY:
  case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
    return igl::TextureType::TwoDArray;
  case GL_SAMPLER_3D:
  case GL_IMAGE_3D:
    return igl::TextureType::ThreeD;
  case GL_SAMPLER_CUBE:
  case GL_IMAGE_CUBE:
    return igl::TextureType::Cube;
  case GL_SAMPLER_EXTERNAL_OES:
#if defined(GL_EXT_YUV_target)
  case GL_SAMPLER_EXTERNAL_2D_Y2Y_EXT:
#endif
    return igl::TextureType::ExternalImage;
  default:
    return igl::TextureType::Invalid;
  }
}

} // namespace

namespace igl::opengl {

RenderPipelineReflection::RenderPipelineReflection(IContext& context, const ShaderStages& stages) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (context.deviceFeatures().hasFeature(DeviceFeatures::UniformBlocks)) {
    generateUniformBlocksDictionary(context, stages.getProgramID());
  }
  generateUniformDictionary(context, stages.getProgramID());
  generateAttributeDictionary(context, stages.getProgramID());
  generateShaderStorageBufferObjectDictionary(context, stages.getProgramID());
  cacheDescriptors();
}

RenderPipelineReflection::~RenderPipelineReflection() = default;

void RenderPipelineReflection::generateUniformDictionary(IContext& context, GLuint pid) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(pid != 0);
  uniformDictionary_.clear();

  GLint count = 0;
  context.getProgramiv(pid, GL_ACTIVE_UNIFORMS, &count);

  // We compute max uniform length by querying GL_ACTIVE_UNIFORM_MAX_LENGTH, and then taking the max
  // of that with every GL_UNIFORM_NAME_LENGTH of each of the uniforms. This is needed because we
  // observed that OpenGL drivers are sometimes unreliable with these values:
  //
  // 1. Android devices with old Mali GPUs (e.g. Mali-T860MP2) sometimes incorrectly return 0 for
  // GL_ACTIVE_UNIFORM_MAX_LENGTH
  //
  // 2. When running macOS unit tests, sometimes GL_UNIFORM_NAME_LENGTH always return 0
  //
  // So the safe thing to do here is to take the max of the two.

  GLint maxUniformNameLength = 0;
  context.getProgramiv(pid, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxUniformNameLength);

#ifdef GL_UNIFORM_NAME_LENGTH
  auto glVersion = context.deviceFeatures().getGLVersion();
  const bool supportsGetActiveUniformsiv =
      glVersion == GLVersion::v3_0_ES || glVersion == GLVersion::v3_1_ES ||
      glVersion == GLVersion::v3_2_ES || glVersion >= GLVersion::v3_1;
  if (supportsGetActiveUniformsiv && count > 0) {
    std::vector<GLuint> indices(count);
    for (int i = 0; i < count; ++i) {
      indices[i] = static_cast<GLuint>(i);
    }
    std::vector<GLint> nameLengths(count);
    context.getActiveUniformsiv(
        pid, count, indices.data(), GL_UNIFORM_NAME_LENGTH, nameLengths.data());

    for (int i = 0; i < count; ++i) {
      maxUniformNameLength = std::max(maxUniformNameLength, nameLengths[i]);
    }
  }
#endif

  std::vector<GLchar> cname(maxUniformNameLength);
  for (int i = 0; i < count; i++) {
    GLsizei length = 0;
    GLsizei size = 0;
    GLenum type = GL_NONE;

    context.getActiveUniform(pid, i, maxUniformNameLength, &length, &size, &type, cname.data());
    const GLint location = context.getUniformLocation(pid, cname.data());
    if (location < 0) {
      // this uniform belongs to a block;
      continue;
    }

    if (length >= 4 && std::strcmp(cname.data() + length - 3, "[0]") == 0) {
      length = length - 3; // remove '[0]' for arrays
    }
    auto name = std::string(cname.data(), cname.data() + length);
    const UniformDesc u(size, location, type);
    uniformDictionary_.insert(std::make_pair(igl::genNameHandle(name), u));
  }
}

void RenderPipelineReflection::generateUniformBlocksDictionary(IContext& context, GLuint pid) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(pid != 0);
  uniformBlocksDictionary_.clear();

  GLint numBlocks = 0;
  GLint maxBlockNameLength = 0;
  context.getProgramiv(pid, GL_ACTIVE_UNIFORM_BLOCKS, &numBlocks);
  if (numBlocks <= 0) {
    return;
  }
  context.getProgramiv(pid, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &maxBlockNameLength);
  if (maxBlockNameLength <= 0) {
    return;
  }

  GLint maxUniformNameLength = 0;
  context.getProgramiv(pid, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxUniformNameLength);

  std::vector<GLchar> uniformBlockNameData;
  for (int i = 0; i < numBlocks; i++) {
    UniformBlockDesc blockDesc;
    blockDesc.blockIndex = i;

    // Get uniform block name
    uniformBlockNameData.assign(maxBlockNameLength, '\0');
    GLsizei blockNameLength = 0;
    context.getActiveUniformBlockName(pid,
                                      blockDesc.blockIndex,
                                      maxBlockNameLength,
                                      &blockNameLength,
                                      uniformBlockNameData.data());
    const std::string uniformBlockName(uniformBlockNameData.begin(),
                                       uniformBlockNameData.begin() + blockNameLength);

    context.getActiveUniformBlockiv(
        pid, blockDesc.blockIndex, GL_UNIFORM_BLOCK_DATA_SIZE, &blockDesc.size);
    context.getActiveUniformBlockiv(
        pid, blockDesc.blockIndex, GL_UNIFORM_BLOCK_BINDING, &blockDesc.bindingIndex);

    // get the number of uniforms in the block
    GLint numActiveUniforms = 0;
    context.getActiveUniformBlockiv(
        pid, blockDesc.blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &numActiveUniforms);

    // Get the indices of the uniforms in the block
    std::vector<GLint> indices(numActiveUniforms);
    context.getActiveUniformBlockiv(
        pid, blockDesc.blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, indices.data());

    std::vector<GLchar> nameData;
    for (size_t k = 0; k < numActiveUniforms; k++) {
      const GLuint index = indices[k];
      UniformBlockDesc::UniformBlockMemberDesc memberDesc;
      // Get uniform block name, type, size
      nameData.assign(maxUniformNameLength, '\0');
      GLsizei nameLength = 0;
      // size  will be 1 for non-arrays; For arrays, it is the number of elements
      context.getActiveUniform(pid,
                               indices[k],
                               maxUniformNameLength,
                               &nameLength,
                               &memberDesc.size,
                               &memberDesc.type,
                               nameData.data());
      context.getActiveUniformsiv(pid, 1, &index, GL_UNIFORM_OFFSET, &memberDesc.offset);

      // Fix the name by removing [0] suffix for arrays
      // and by stripping the block name from the uniform name
      if (nameLength >= 4 && nameData[nameLength - 3] == '[' && nameData[nameLength - 2] == '0' &&
          nameData[nameLength - 1] == ']') {
        nameLength = nameLength - 3; // remove '[0]' for arrays
      }

      auto start = nameData.begin();
      auto end = nameData.begin() + nameLength;
      // strip the block name from the uniform name
      auto it = std::find(start, end, '.');
      if (it != end) {
        start = it + 1;
      }
      const std::string uniformName(start, end);

      blockDesc.members.insert(std::make_pair(igl::genNameHandle(uniformName), memberDesc));
    }
    uniformBlocksDictionary_[igl::genNameHandle(uniformBlockName)] = blockDesc;
  }
}

void RenderPipelineReflection::generateAttributeDictionary(IContext& context, GLuint pid) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(pid != 0);

  attributeDictionary_.clear();
  GLint maxAttributeNameLength = 0;
  context.getProgramiv(pid, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxAttributeNameLength);
  GLint count = 0;
  context.getProgramiv(pid, GL_ACTIVE_ATTRIBUTES, &count);

  std::vector<GLchar> attribName(maxAttributeNameLength);
  for (int i = 0; i < count; i++) {
    GLsizei length = 0;
    GLsizei size = 0;
    GLenum type = GL_NONE;

    context.getActiveAttrib(
        pid, i, maxAttributeNameLength, &length, &size, &type, attribName.data());
    auto name = std::string(attribName.data(), attribName.data() + length);
    const GLint location = context.getAttribLocation(pid, name.c_str());

    attributeDictionary_.insert(std::make_pair(name, location));
  }
}

void RenderPipelineReflection::generateShaderStorageBufferObjectDictionary(IContext& context,
                                                                           GLuint pid) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (context.deviceFeatures().hasFeature(DeviceFeatures::Compute)) {
    IGL_DEBUG_ASSERT(pid != 0);
    shaderStorageBufferObjectDictionary_.clear();

    GLint maxSSBONameLength = 0;
    context.getProgramInterfaceiv(
        pid, GL_SHADER_STORAGE_BLOCK, GL_MAX_NAME_LENGTH, &maxSSBONameLength);
    GLint count = 0;
    context.getProgramInterfaceiv(pid, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &count);

    std::vector<GLchar> cname(maxSSBONameLength);
    for (int i = 0; i < count; i++) {
      GLsizei length = 0;
      context.getProgramResourceName(
          pid, GL_SHADER_STORAGE_BLOCK, i, maxSSBONameLength, &length, cname.data());
      auto name = std::string(cname.data(), cname.data() + length);
      const GLuint index =
          context.getProgramResourceIndex(pid, GL_SHADER_STORAGE_BLOCK, cname.data());
      GLint location{-1};
      const GLenum prop = GL_BUFFER_BINDING;
      context.getProgramResourceiv(
          pid, GL_SHADER_STORAGE_BLOCK, index, 1, &prop, 1, &length, &location);

      shaderStorageBufferObjectDictionary_.insert(
          std::make_pair(igl::genNameHandle(name), location));
    }
  }
}

int RenderPipelineReflection::getIndexByName(const NameHandle& name) const {
  IGL_PROFILER_FUNCTION();
  // Search through list of uniforms
  const auto uniformEntry = uniformDictionary_.find(name);

  if (uniformEntry != uniformDictionary_.end()) {
    return uniformEntry->second.location;
  }

  // search through the list of uniform blocks
  const auto uniformBlockEntry = uniformBlocksDictionary_.find(name);
  if (uniformBlockEntry != uniformBlocksDictionary_.end()) {
    return uniformBlockEntry->second.bindingIndex;
  }

  // Search through list of attributes
  auto attribEntry = attributeDictionary_.find(name.toString());
  if (attribEntry != attributeDictionary_.end()) {
    return attribEntry->second;
  }

  // Search through list of SSBOs
  auto ssboEntry = shaderStorageBufferObjectDictionary_.find(name);
  if (ssboEntry != shaderStorageBufferObjectDictionary_.end()) {
    return ssboEntry->second;
  }

  return -1;
}

void RenderPipelineReflection::cacheDescriptors() {
  IGL_PROFILER_FUNCTION();
  bufferArguments_.clear();
  samplerArguments_.clear();
  textureArguments_.clear();

  for (const auto& entry : uniformDictionary_) {
    const UniformDesc& glDesc = entry.second;
    const TextureType textureType = toIGLTextureType(glDesc.type);

    // buffers
    if (textureType == TextureType::Invalid) {
      const UniformType uniformType = toIGLUniformType(glDesc.type);

      BufferArgDesc bufferDesc;
      bufferDesc.name = entry.first;
      bufferDesc.bufferAlignment = 1;
      bufferDesc.bufferDataSize = glDesc.size * igl::sizeForUniformType(uniformType);
      bufferDesc.bufferIndex = glDesc.location;
      bufferDesc.shaderStage = ShaderStage::Fragment;

      BufferArgDesc::BufferMemberDesc iglMemberDesc{
          .name = entry.first,
          .type = uniformType,
          .offset = 0,
          .arrayLength = static_cast<size_t>(glDesc.size),
      };
      bufferDesc.members.push_back(std::move(iglMemberDesc));

      bufferArguments_.push_back(std::move(bufferDesc));
    }
    // textures & samplers
    else {
      TextureArgDesc textureDesc;
      textureDesc.name = entry.first;
      textureDesc.type = textureType;
      textureDesc.textureIndex = glDesc.location;
      textureDesc.shaderStage = ShaderStage::Fragment;

      // Create one artificial sampler for each texture
      SamplerArgDesc samplerDesc;
      samplerDesc.name = textureDesc.name;
      samplerDesc.samplerIndex = textureDesc.textureIndex;
      samplerDesc.shaderStage = textureDesc.shaderStage;

      textureArguments_.push_back(std::move(textureDesc));
      samplerArguments_.push_back(std::move(samplerDesc));
    }
  }

  // uniform blocks
  for (const auto& blockEntry : uniformBlocksDictionary_) {
    const auto& blockDesc = blockEntry.second;
    BufferArgDesc bufferDesc;
    bufferDesc.isUniformBlock = true;
    bufferDesc.name = blockEntry.first;
    bufferDesc.bufferAlignment = 1;
    bufferDesc.bufferDataSize = blockDesc.size;
    bufferDesc.bufferIndex = blockDesc.blockIndex;
    bufferDesc.shaderStage = ShaderStage::Fragment;

    for (const auto& uniformEntry : blockDesc.members) {
      const auto& uniformDesc = uniformEntry.second;
      const UniformType uniformType = toIGLUniformType(uniformDesc.type);

      BufferArgDesc::BufferMemberDesc iglMemberDesc{
          .name = uniformEntry.first,
          .type = uniformType,
          .offset = static_cast<size_t>(uniformDesc.offset),
          .arrayLength = static_cast<size_t>(uniformDesc.size),
      };
      bufferDesc.members.push_back(std::move(iglMemberDesc));
    }
    bufferArguments_.push_back(std::move(bufferDesc));
  }
}

const std::vector<BufferArgDesc>& RenderPipelineReflection::allUniformBuffers() const {
  return bufferArguments_;
}

const std::vector<SamplerArgDesc>& RenderPipelineReflection::allSamplers() const {
  return samplerArguments_;
}

const std::vector<TextureArgDesc>& RenderPipelineReflection::allTextures() const {
  return textureArguments_;
}

} // namespace igl::opengl
