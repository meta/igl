/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/ComputePipelineState.h>
#include <igl/NameHandle.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/RenderPipelineReflection.h>
#include <igl/opengl/Shader.h>

namespace igl::opengl {
class Texture;
class Buffer;

class ComputePipelineState final : public WithContext, public IComputePipelineState {
  friend class Device;

 public:
  explicit ComputePipelineState(IContext& context);
  ~ComputePipelineState() override;

  Result create(const ComputePipelineDesc& desc);
  void bind();
  void unbind();
  Result bindImageTextureUnit(size_t unit, Texture* texture);
  Result bindSamplerTextureUnit(size_t unit, Texture* texture);
  Result bindBuffer(size_t unit, Buffer* buffer);

  std::shared_ptr<IComputePipelineReflection> computePipelineReflection() override;

  [[nodiscard]] int getIndexByName(const NameHandle& name) const override;

  bool getIsUsingShaderStorageBuffers();

 private:
  using ComputePipelineReflection = RenderPipelineReflection;

  std::array<GLint, IGL_BUFFER_BINDINGS_MAX> bufferUnitMap_{};
  std::array<GLint, IGL_TEXTURE_SAMPLERS_MAX> imageUnitMap_{};
  std::array<GLint, IGL_TEXTURE_SAMPLERS_MAX> samplerUnitMap_{};

  std::shared_ptr<ShaderStages> shaderStages_;
  std::shared_ptr<ComputePipelineReflection> reflection_;

  bool usingShaderStorageBuffers_ = false;
};

inline bool ComputePipelineState::getIsUsingShaderStorageBuffers() {
  return usingShaderStorageBuffers_;
}

} // namespace igl::opengl
