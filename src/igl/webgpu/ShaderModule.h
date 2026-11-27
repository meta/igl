/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <igl/Shader.h>
#include <igl/webgpu/Common.h>
#include <igl/webgpu/WgslReflection.h>

namespace igl::webgpu {

class WebGPUContext;

/// @brief A compiled WGSL module and the reflection of its declarations.
struct WgslModule {
  Handle<WGPUShaderModule> module;
  std::shared_ptr<const WgslReflection> reflection;
};

/// Compiles and reflects WGSL `source`. Compilation errors are returned with their line numbers.
[[nodiscard]] Result compileWgsl(const WebGPUContext& ctx,
                                 const char* IGL_NULLABLE source,
                                 const std::string& debugName,
                                 WgslModule& outModule);

/// @brief WGSL override constants for a pipeline stage; `entries` point into `keys`.
struct PipelineConstants {
  std::vector<std::string> keys;
  std::vector<WGPUConstantEntry> entries;

  PipelineConstants() = default;
  PipelineConstants(const PipelineConstants&) = delete;
  PipelineConstants& operator=(const PipelineConstants&) = delete;
  PipelineConstants(PipelineConstants&&) = default;
  PipelineConstants& operator=(PipelineConstants&&) = default;
};

/// @brief A WGSL shader module and the entry point IGL selected from it.
class ShaderModule final : public IShaderModule {
 public:
  /// Compiles the WGSL source in `desc.input`. Compilation errors are returned in `outResult`.
  [[nodiscard]] static std::shared_ptr<ShaderModule> create(const WebGPUContext& ctx,
                                                            const ShaderModuleDesc& desc,
                                                            Result* IGL_NULLABLE outResult);
  /// Selects the entry point `info` from an already compiled module.
  [[nodiscard]] static std::shared_ptr<ShaderModule> create(const WgslModule& module,
                                                            const ShaderModuleInfo& info,
                                                            Result* IGL_NULLABLE outResult);

  ShaderModule(ShaderModuleInfo info, WgslModule module);

  [[nodiscard]] WGPUShaderModule IGL_NULLABLE getWGPUShaderModule() const noexcept {
    return module_.module.get();
  }
  [[nodiscard]] const WgslReflection& getReflection() const noexcept;
  /// The selected entry point; never null for a successfully created module.
  [[nodiscard]] const WgslEntryPoint* IGL_NULLABLE getEntryPoint() const noexcept {
    return module_.reflection->findEntryPoint(info().entryPoint);
  }
  /// The function constants of info() as override constants: constant i sets the `override` with
  /// `@id(i)`. Only scalar constants map; constants the module does not declare are dropped.
  [[nodiscard]] PipelineConstants getPipelineConstants() const;

 private:
  WgslModule module_;
};

[[nodiscard]] inline const WgslReflection& ShaderModule::getReflection() const noexcept {
  return *module_.reflection;
}

class ShaderLibrary final : public IShaderLibrary {
 public:
  explicit ShaderLibrary(std::vector<std::shared_ptr<IShaderModule>> modules) :
    IShaderLibrary(std::move(modules)) {}
};

class ShaderStages final : public IShaderStages {
 public:
  explicit ShaderStages(ShaderStagesDesc desc) : IShaderStages(std::move(desc)) {}
};

} // namespace igl::webgpu
