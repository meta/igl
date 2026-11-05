/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/Device.h>

#include <algorithm>
#include <cstring>
#include <igl/FramebufferWrapper.h>
#include <igl/glslang/GlslCompiler.h>
#include <igl/glslang/GlslangHelpers.h>
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/CommandQueue.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/ComputePipelineState.h>
#include <igl/vulkan/Framebuffer.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/RenderPipelineState.h>
#include <igl/vulkan/SamplerState.h>
#include <igl/vulkan/ShaderModule.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/Timer.h>
#include <igl/vulkan/TimestampQueries.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanShaderModule.h>

// Writes the shader code to disk for debugging. Used in `Device::createShaderModule()`
#if IGL_SHADER_DUMP && IGL_DEBUG
#include <filesystem>
#include <fstream>
#endif // IGL_SHADER_DUMP && IGL_DEBUG

namespace {

#if IGL_SHADER_DUMP && IGL_DEBUG
std::string sanitizeFileName(const std::string& fileName) {
  std::string result;
  for (const char c : fileName) {
    if (std::isalnum(c) || c == '.' || c == '_' || c == '-') {
      result += c;
    } else {
      result += '_';
    }
  }
  return result;
}
#endif // IGL_SHADER_DUMP && IGL_DEBUG

bool supportsFormat(const VulkanFunctionTable& vf,
                    VkPhysicalDevice physicalDevice,
                    VkFormat format) {
  VkFormatProperties properties;
  vf.vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
  return properties.bufferFeatures != 0 || properties.linearTilingFeatures != 0 ||
         properties.optimalTilingFeatures != 0;
}

VkShaderStageFlagBits shaderStageToVkShaderStage(igl::ShaderStage stage) {
  switch (stage) {
  case igl::ShaderStage::Vertex:
    return VK_SHADER_STAGE_VERTEX_BIT;
  case igl::ShaderStage::Fragment:
    return VK_SHADER_STAGE_FRAGMENT_BIT;
  case igl::ShaderStage::Compute:
    return VK_SHADER_STAGE_COMPUTE_BIT;
  case igl::ShaderStage::Task:
    return VK_SHADER_STAGE_TASK_BIT_EXT;
  case igl::ShaderStage::Mesh:
    return VK_SHADER_STAGE_MESH_BIT_EXT;
  };
  return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

const char* shaderStageToString(igl::ShaderStage stage) {
  switch (stage) {
  case igl::ShaderStage::Vertex:
    return "vertex";
  case igl::ShaderStage::Fragment:
    return "fragment";
  case igl::ShaderStage::Compute:
    return "compute";
  case igl::ShaderStage::Task:
    return "task";
  case igl::ShaderStage::Mesh:
    return "mesh";
  };
  return "unknown";
}

// Checks the bindings of a shader module that the encoders and the descriptor sets rely on. The
// input attachments are legal only in a fragment shader. A library can have the shader modules of
// several stages that share one SPIR-V module, in which case `stage` has to be the fragment stage
// when one of them is a fragment shader.
igl::Result validateShaderModule(const igl::vulkan::util::SpvModuleInfo& info,
                                 igl::ShaderStage stage,
                                 const std::string& debugName) {
  if (!info.inputAttachments.empty() && stage != igl::ShaderStage::Fragment) {
    return igl::Result(igl::Result::Code::ArgumentInvalid,
                       IGL_FORMAT("Shader module '{}': input attachments (subpassInput) can only "
                                  "be used in a fragment shader, but the stage is {}",
                                  debugName,
                                  shaderStageToString(stage)));
  }
  return igl::vulkan::validateShaderModuleBindings(info, debugName);
}

} // namespace

namespace igl::vulkan {

Device::Device(std::unique_ptr<VulkanContext> ctx) : ctx_(std::move(ctx)), platformDevice_(*this) {}

std::shared_ptr<ICommandQueue> Device::createCommandQueueInternal(const CommandQueueDesc& desc,
                                                                  Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  Result::setOk(outResult);
  auto resource = std::make_shared<CommandQueue>(*this, desc);
  return resource;
}

std::unique_ptr<IBuffer> Device::createBufferInternal( // NOLINT(bugprone-exception-escape)
    const BufferDesc& desc,
    Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  auto buffer = std::make_unique<Buffer>(*this);

  const auto result = buffer->create(desc);

  if (!IGL_DEBUG_VERIFY(result.isOk())) {
    return nullptr;
  }

  if (!desc.data) {
    return buffer;
  }

  const auto uploadResult = buffer->upload(desc.data, BufferRange(desc.length, 0u));
  IGL_DEBUG_ASSERT(uploadResult.isOk());
  Result::setResult(outResult, uploadResult);

  if (hasResourceTracker()) {
    buffer->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  return buffer;
}

std::shared_ptr<IDepthStencilState> Device::createDepthStencilStateInternal(
    const DepthStencilStateDesc& desc,
    Result* IGL_NULLABLE outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  Result::setOk(outResult);
  return std::make_shared<DepthStencilState>(desc);
}

std::unique_ptr<IShaderStages> Device::createShaderStagesInternal(const ShaderStagesDesc& desc,
                                                                  Result* IGL_NULLABLE
                                                                      outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  auto shaderStages = std::make_unique<ShaderStages>(desc);
  if (!shaderStages) {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "Could not instantiate shader stages.");
  } else if (!shaderStages->isValid()) {
    Result::setResult(
        outResult, Result::Code::ArgumentInvalid, "Missing required shader module(s).");
  } else {
    Result::setOk(outResult);
  }

  if (hasResourceTracker()) {
    shaderStages->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  return shaderStages;
}

std::shared_ptr<ISamplerState> Device::createSamplerStateInternal(const SamplerStateDesc& desc,
                                                                  Result* IGL_NULLABLE
                                                                      outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  auto samplerState = std::make_shared<SamplerState>(const_cast<Device&>(*this));

  Result::setResult(outResult, samplerState->create(desc));

  if (hasResourceTracker()) {
    samplerState->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  return samplerState;
}

std::shared_ptr<ITexture> Device::createTextureInternal( // NOLINT(bugprone-exception-escape)
    const TextureDesc& desc,
    Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  const Result validation = validateTextureDesc(desc);
  if (!validation.isOk()) {
    Result::setResult(outResult, validation);
    return nullptr;
  }

  const auto sanitized = sanitize(desc);

  auto texture = std::make_shared<Texture>(const_cast<Device&>(*this), desc.format);

  const Result res = texture->create(sanitized);

  if (hasResourceTracker()) {
    texture->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  Result::setResult(outResult, res);

  return res.isOk() ? texture : nullptr;
}

std::shared_ptr<ITexture> Device::createTextureView( // NOLINT(bugprone-exception-escape)
    std::shared_ptr<ITexture> texture,
    const TextureViewDesc& desc,
    Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  if (!IGL_DEBUG_VERIFY(texture)) {
    Result::setResult(outResult,
                      Result(Result::Code::ArgumentInvalid, "A base texture should be specified"));
    return {};
  }

  const Texture& baseTexture = static_cast<Texture&>(*texture);

  auto newTexture = std::make_shared<Texture>(
      const_cast<Device&>(*this),
      desc.format == TextureFormat::Invalid ? baseTexture.getFormat() : desc.format);

  const Result res = newTexture->createView(baseTexture, desc);

  if (hasResourceTracker()) {
    newTexture->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  Result::setResult(outResult, res);

  return res.isOk() ? newTexture : nullptr;
}

std::shared_ptr<IVertexInputState> Device::createVertexInputStateInternal(
    const VertexInputStateDesc& desc,
    Result* IGL_NULLABLE outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  // VertexInputState is compiled into the RenderPipelineState at a later stage. For now, we just
  // have to store the description.
  Result::setOk(outResult);

  return std::make_shared<VertexInputState>(desc);
}

std::shared_ptr<IComputePipelineState> Device::createComputePipelineInternal(
    const ComputePipelineDesc& desc,
    Result* IGL_NULLABLE outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  if (!IGL_DEBUG_VERIFY(desc.shaderStages)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing shader stages");
    return nullptr;
  }
  if (!IGL_DEBUG_VERIFY(desc.shaderStages->getType() == ShaderStagesType::Compute)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Shader stages not for compute");
    return nullptr;
  }
  if (!IGL_DEBUG_VERIFY(desc.shaderStages->getComputeModule())) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing compute shader");
    return nullptr;
  }

  Result::setOk(outResult);
  return std::make_shared<ComputePipelineState>(*this, desc);
}

std::shared_ptr<IRenderPipelineState> Device::createRenderPipelineInternal(
    const RenderPipelineDesc& desc,
    Result* IGL_NULLABLE outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  if (!IGL_DEBUG_VERIFY(desc.shaderStages)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing shader stages");
    return nullptr;
  }
  if (!IGL_DEBUG_VERIFY(desc.shaderStages->getType() == ShaderStagesType::Render ||
                        desc.shaderStages->getType() == ShaderStagesType::RenderMeshShader)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Shader stages not for render");
    return nullptr;
  }

  const bool hasColorAttachments = !desc.targetDesc.colorAttachments.empty();
  const bool hasDepthAttachment = desc.targetDesc.depthAttachmentFormat != TextureFormat::Invalid;
  const bool hasAnyAttachments = hasColorAttachments || hasDepthAttachment;
  if (!IGL_DEBUG_VERIFY(hasAnyAttachments)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Need at least one attachment");
    return nullptr;
  }

  if (desc.shaderStages->getType() == ShaderStagesType::Render &&
      !IGL_DEBUG_VERIFY(desc.shaderStages->getVertexModule())) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing vertex shader");
    return nullptr;
  }

  if (desc.shaderStages->getType() == ShaderStagesType::RenderMeshShader &&
      !IGL_DEBUG_VERIFY(desc.shaderStages->getMeshModule())) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing mesh shader");
    return nullptr;
  }

  if (!IGL_DEBUG_VERIFY(desc.shaderStages->getFragmentModule())) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing fragment shader");
    return nullptr;
  }

  if (ctx_->features().has_VK_EXT_descriptor_buffer) {
    const auto& fragmentModule =
        static_cast<const ShaderModule&>(*desc.shaderStages->getFragmentModule());
    if (!fragmentModule.getVulkanShaderModule().getSpvModuleInfo().inputAttachments.empty()) {
      Result::setResult(
          outResult,
          Result::Code::Unsupported,
          IGL_FORMAT("Render pipeline '{}': input attachments (subpassInput) are not supported "
                     "with VK_EXT_descriptor_buffer",
                     desc.debugName.c_str()));
      return nullptr;
    }
  }

  return std::make_shared<RenderPipelineState>(*this, desc);
}

std::shared_ptr<IShaderModule> Device::createShaderModuleInternal(const ShaderModuleDesc& desc,
                                                                  Result* IGL_NULLABLE
                                                                      outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  std::shared_ptr<VulkanShaderModule> vulkanShaderModule;
  Result result;
  if (desc.input.type == ShaderInputType::Binary) {
    vulkanShaderModule =
        createShaderModule(desc.input.data, desc.input.length, desc.debugName, &result);
  } else {
    vulkanShaderModule = createShaderModule(
        desc.info.stage, desc.input.source, desc.input.options, desc.debugName, &result);
  }

  if (!result.isOk()) {
    Result::setResult(outResult, std::move(result));
    return nullptr;
  }

  result =
      validateShaderModule(vulkanShaderModule->getSpvModuleInfo(), desc.info.stage, desc.debugName);
  if (!result.isOk()) {
    Result::setResult(outResult, std::move(result));
    return nullptr;
  }

  ctx_->shaderCompilationCount_++;

  Result::setResult(outResult, std::move(result));
  auto shaderModule = std::make_shared<ShaderModule>(desc.info, std::move(vulkanShaderModule));

  if (hasResourceTracker()) {
    shaderModule->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  return shaderModule;
}

/**
 * @brief Creates a VkShaderModule from pre-compiled
 *        SPIR-V binary data.
 *
 * Optionally dumps the binary to disk when IGL_SHADER_DUMP
 * is enabled. Assigns a Vulkan debug name if provided.
 *
 * @param[in] data SPIR-V binary (as uint32_t words).
 * @param[in] length Size of the binary data in bytes.
 * @param[in] debugName Label for Vulkan debug utilities.
 * @param[out] outResult Receives the operation status.
 * @return The created module with reflection data, or
 *         nullptr on failure.
 */
std::shared_ptr<VulkanShaderModule> Device::createShaderModule(const void* IGL_NULLABLE data,
                                                               size_t length,
                                                               const std::string& debugName,
                                                               Result* IGL_NULLABLE
                                                                   outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  if (!data || length == 0) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Shader data is null or empty");
    return nullptr;
  }

#if IGL_SHADER_DUMP && IGL_DEBUG
  uint64_t hash = 0;
  IGL_DEBUG_ASSERT(length % sizeof(uint32_t) == 0);
  auto words = reinterpret_cast<const uint32_t*>(data);
  for (size_t i = 0; i < length / sizeof(uint32_t); i++) {
    hash ^= std::hash<uint32_t>()(words[i]);
  }
  const std::string filename = IGL_FORMAT(
      "{}{}{}.spv", IGL_SHADER_DUMP_PATH, sanitizeFileName(debugName), std::to_string(hash));
  IGL_LOG_INFO("Dumping shader to: %s", filename.c_str());
  if (!std::filesystem::exists(filename)) {
    std::ofstream spirvFile;
    spirvFile.open(filename, std::ios::out | std::ios::binary);
    for (size_t i = 0; i < length / sizeof(uint32_t); i++) {
      spirvFile.write(reinterpret_cast<const char*>(&words[i]), sizeof(uint32_t));
    }
    spirvFile.close();
  }
#endif // IGL_SHADER_DUMP && IGL_DEBUG

  VkShaderModule vkShaderModule = VK_NULL_HANDLE;
  const VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = length,
      .pCode = static_cast<const uint32_t*>(data),
  };
  const VkResult result =
      ctx_->vf_.vkCreateShaderModule(ctx_->getVkDevice(), &ci, nullptr, &vkShaderModule);

  setResultFrom(outResult, result);

  if (result != VK_SUCCESS) {
    return nullptr;
  }

  if (!debugName.empty()) {
    // set debug name
    VK_ASSERT(ivkSetDebugObjectName(&ctx_->vf_,
                                    ctx_->getVkDevice(),
                                    VK_OBJECT_TYPE_SHADER_MODULE,
                                    (uint64_t)vkShaderModule,
                                    debugName.c_str()));
  }

  IGL_DEBUG_ASSERT(vkShaderModule != VK_NULL_HANDLE);
  return std::make_shared<VulkanShaderModule>(
      ctx_->vf_,
      ctx_->getVkDevice(),
      vkShaderModule,
      util::getReflectionData(reinterpret_cast<const uint32_t*>(data), length));
}

/**
 * @brief Compiles GLSL source into SPIR-V and creates a
 *        Vulkan shader module.
 *
 * If the source lacks a `#version` directive, a GLSL 460
 * header with device-appropriate extensions is prepended.
 * For fragment shaders, bindless texture/sampler layout
 * declarations are also injected when descriptor indexing
 * is enabled.
 *
 * @param[in] stage  Shader stage (Vertex, Fragment,
 *                   Compute, Task, or Mesh).
 * @param[in] source GLSL source code to compile.
 * @param[in] options Shader compiler options selecting the optimization strategy.
 * @param[in] debugName Label assigned to the Vulkan
 *                      shader module for debug tooling.
 * @param[out] outResult Receives the compilation result.
 * @return The compiled shader module, or nullptr on
 *         failure.
 */
std::shared_ptr<VulkanShaderModule> Device::createShaderModule(ShaderStage stage,
                                                               const char* IGL_NULLABLE source,
                                                               const ShaderCompilerOptions& options,
                                                               const std::string& debugName,
                                                               Result* IGL_NULLABLE
                                                                   outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  const VkShaderStageFlagBits vkStage = shaderStageToVkShaderStage(stage);
  IGL_DEBUG_ASSERT(vkStage != VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM);
  IGL_DEBUG_ASSERT(source);

  std::string sourcePatched;

  if (!source || !*source) {
    Result::setResult(outResult, Result::Code::ArgumentNull, "Shader source is empty");
    return nullptr;
  }

  if (!std::strstr(source, "#version ")) {
    std::string extraExtensions = ctx_->config_.enableDescriptorIndexing
                                      ? "#extension GL_EXT_nonuniform_qualifier : require\n"
                                      : "";

    // GL_EXT_debug_printf extension
    if (ctx_->features_.has_VK_KHR_shader_non_semantic_info) {
      extraExtensions += "#extension GL_EXT_debug_printf : enable\n";
    }

    if (ctx_->features_.featuresShaderFloat16Int8.shaderFloat16 == VK_TRUE) {
      extraExtensions += "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    }

    if (ctx_->features_.has_VK_KHR_buffer_device_address) {
      extraExtensions += "#extension GL_EXT_buffer_reference : require\n";
      extraExtensions += "#extension GL_EXT_buffer_reference_uvec2 : require\n";
    }

    const std::string bindlessTexturesSource = ctx_->config_.enableDescriptorIndexing ?
                                                                                      R"(
      // everything - indexed by global texture/sampler id
      layout (set = 3, binding = 0) uniform texture2D kTextures2D[];
      layout (set = 3, binding = 1) uniform texture2DArray kTextures2DArray[];
      layout (set = 3, binding = 2) uniform texture3D kTextures3D[];
      layout (set = 3, binding = 3) uniform textureCube kTexturesCube[];
      layout (set = 3, binding = 4) uniform sampler kSamplers[];
      layout (set = 3, binding = 5) uniform samplerShadow kSamplersShadow[];
      // binding #6 is reserved for STORAGE_IMAGEs: check VulkanContext.cpp
      )"
                                                                                      : "";

    // there's no header provided in the shader source, let's insert our own header
    if (vkStage == VK_SHADER_STAGE_VERTEX_BIT || vkStage == VK_SHADER_STAGE_COMPUTE_BIT) {
      sourcePatched += R"(
      #version 460
      )" + extraExtensions;
    }
    if (vkStage == VK_SHADER_STAGE_FRAGMENT_BIT) {
      sourcePatched += R"(
      #version 460
      )" + extraExtensions +
                       bindlessTexturesSource;
    }
    sourcePatched.append(source, std::strlen(source));
    source = sourcePatched.c_str();
  }

  glslang_resource_t glslangResource = {};
  glslangGetDefaultResource(&glslangResource);
  ivkUpdateGlslangResource(&glslangResource,
                           &ctx_->getVkPhysicalDeviceProperties(),
                           &ctx_->getvkPhysicalDeviceMeshShaderPropertiesEXT());

  // Clamp the SPIR-V target to what the device's Vulkan API version accepts; generating
  // the default 1.5 for older devices fails VUID-VkShaderModuleCreateInfo-pCode-08737.
  // Newer devices keep the default.
  const uint32_t apiVersion = ctx_->getVkPhysicalDeviceProperties().apiVersion;
  const glslang::GlslTargetVersion target = glslang::glslTargetForVulkanVersion(
      VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));

  std::vector<uint32_t> spirv;
  const Result result =
      glslang::compileShader(stage, source, spirv, &glslangResource, options, target);

  if (!result.isOk()) {
    Result::setResult(outResult, result);
    return nullptr;
  }

  VkShaderModule vkShaderModule = VK_NULL_HANDLE;
  const VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size() * sizeof(uint32_t),
      .pCode = spirv.data(),
  };
  VK_ASSERT(ctx_->vf_.vkCreateShaderModule(ctx_->getVkDevice(), &ci, nullptr, &vkShaderModule));

  Result::setResult(outResult, result);

  IGL_DEBUG_ASSERT(vkShaderModule != VK_NULL_HANDLE);

  if (!debugName.empty()) {
    // set debug name
    VK_ASSERT(ivkSetDebugObjectName(&ctx_->vf_,
                                    ctx_->getVkDevice(),
                                    VK_OBJECT_TYPE_SHADER_MODULE,
                                    (uint64_t)vkShaderModule,
                                    debugName.c_str()));
  }

  return std::make_shared<VulkanShaderModule>(
      ctx_->vf_,
      ctx_->getVkDevice(),
      vkShaderModule,
      util::getReflectionData(spirv.data(), spirv.size() * sizeof(uint32_t)));
}

std::shared_ptr<IFramebuffer> Device::createFramebufferInternal(const FramebufferDesc& desc,
                                                                Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  auto resource = std::make_shared<Framebuffer>(*this, desc);
  Result::setOk(outResult);

  if (hasResourceTracker()) {
    resource->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  return resource;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
std::shared_ptr<ITimestampQueries> Device::createTimestampQueriesInternal(
    uint32_t maxTimestamps,
    Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  if (!hasFeatureInternal(DeviceFeatures::TimestampQueries)) {
    Result::setResult(
        outResult, Result::Code::Unsupported, "TimestampQueries unsupported by this Vulkan device");
    return nullptr;
  }

  auto queries = std::make_shared<TimestampQueries>(*ctx_, maxTimestamps);
  if (!queries->isValid()) {
    Result::setResult(outResult,
                      Result::Code::RuntimeError,
                      "TimestampQueries: failed to create Vulkan query pool");
    return nullptr;
  }

  Result::setOk(outResult);
  return queries;
}

base::IFramebufferInterop* IGL_NULLABLE
Device::createFramebufferInterop(const base::FramebufferInteropDesc& desc) {
  auto framebuffer = createFramebufferFromBaseDesc(desc);
  if (!framebuffer) {
    return nullptr;
  }
  return new (std::nothrow) FramebufferWrapper(std::move(framebuffer));
}

const PlatformDevice& Device::getPlatformDeviceInternal() const noexcept {
  return platformDevice_;
}

size_t Device::getCurrentDrawCountInternal() const {
  return ctx_->drawCallCount_;
}

size_t Device::getShaderCompilationCountInternal() const {
  return ctx_->shaderCompilationCount_;
}

std::unique_ptr<IShaderLibrary> Device::createShaderLibraryInternal(const ShaderLibraryDesc& desc,
                                                                    Result* IGL_NULLABLE
                                                                        outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  ctx_->shaderCompilationCount_++;

  if (IGL_DEBUG_VERIFY_NOT(desc.moduleInfo.empty())) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid);
    return nullptr;
  }
  Result result;
  std::shared_ptr<VulkanShaderModule> vulkanShaderModule;
  if (desc.input.type == ShaderInputType::Binary) {
    vulkanShaderModule =
        createShaderModule(desc.input.data, desc.input.length, desc.debugName, &result);
  } else {
    if (desc.moduleInfo.size() > 1) {
      IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
      Result::setResult(outResult, Result::Code::Unsupported);
      return nullptr;
    }
    if (!IGL_DEBUG_VERIFY(!desc.moduleInfo.empty())) {
      Result::setResult(outResult, Result::Code::ArgumentInvalid, "moduleInfo is empty");
      return nullptr;
    }
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    vulkanShaderModule = createShaderModule(desc.moduleInfo.front().stage,
                                            desc.input.source,
                                            desc.input.options,
                                            desc.debugName,
                                            &result);
  }

  if (!result.isOk()) {
    Result::setResult(outResult, std::move(result));
    return nullptr;
  }

  // the modules share the SPIR-V module, so it is the one of the fragment shader when there is one
  ShaderStage stage = ShaderStage::Fragment;
  for (const auto& moduleInfo : desc.moduleInfo) {
    stage = moduleInfo.stage;
    if (stage == ShaderStage::Fragment) {
      break;
    }
  }
  result = validateShaderModule(vulkanShaderModule->getSpvModuleInfo(), stage, desc.debugName);
  if (!result.isOk()) {
    Result::setResult(outResult, std::move(result));
    return nullptr;
  }

  std::vector<std::shared_ptr<IShaderModule>> modules;
  modules.reserve(desc.moduleInfo.size());
  for (const auto& info : desc.moduleInfo) {
    modules.emplace_back(std::make_shared<ShaderModule>(info, vulkanShaderModule));
  }

  Result::setResult(outResult, std::move(result));
  auto shaderLibrary = std::make_unique<ShaderLibrary>(std::move(modules));

  if (hasResourceTracker()) {
    shaderLibrary->initResourceTracker(getResourceTracker(), desc.debugName);
  }

  return shaderLibrary;
}

/**
 * @brief Queries whether a specific device feature is supported
 *        by the Vulkan physical device.
 *
 * Implements the ICapabilities::hasFeature interface for the
 * Vulkan backend. Feature support is determined by querying
 * Vulkan physical device properties, limits, and available
 * extensions via VulkanContext.
 *
 * @param[in] feature The device feature to query.
 * @return True if the feature is supported, false otherwise.
 */
bool Device::hasFeatureInternal(DeviceFeatures feature) const {
  IGL_PROFILER_FUNCTION();

  const VkPhysicalDevice physicalDevice = ctx_->vkPhysicalDevice_;
  if (physicalDevice == VK_NULL_HANDLE) {
    IGL_SOFT_ERROR("VkPhysicalDevice is null");
    return false;
  }
  const VkPhysicalDeviceProperties& deviceProperties = ctx_->getVkPhysicalDeviceProperties();

  switch (feature) {
  case DeviceFeatures::MultiSample:
  case DeviceFeatures::MultiSampleResolve:
    return deviceProperties.limits.framebufferColorSampleCounts > VK_SAMPLE_COUNT_1_BIT;
  case DeviceFeatures::TextureFilterAnisotropic:
    return deviceProperties.limits.maxSamplerAnisotropy > 1;
  case DeviceFeatures::MapBufferRange:
    return true;
  case DeviceFeatures::MeshShaders:
    return ctx_->features_.has_VK_EXT_mesh_shader;
  case DeviceFeatures::MultipleRenderTargets:
    return deviceProperties.limits.maxColorAttachments > 1;
  case DeviceFeatures::StandardDerivative:
    return true;
  case DeviceFeatures::StandardDerivativeExt:
    return false;
  case DeviceFeatures::TextureFormatRG:
    return supportsFormat(ctx_->vf_, physicalDevice, VK_FORMAT_R8G8_UNORM);
  case DeviceFeatures::TextureFormatRGB:
    return supportsFormat(ctx_->vf_, physicalDevice, VK_FORMAT_R8G8B8_SRGB);
  case DeviceFeatures::ReadWriteFramebuffer:
    return true;
  case DeviceFeatures::TextureNotPot:
    return true;
  case DeviceFeatures::UniformBlocks:
    return true;
  case DeviceFeatures::TextureHalfFloat:
    return supportsFormat(ctx_->vf_, physicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT) ||
           supportsFormat(ctx_->vf_, physicalDevice, VK_FORMAT_R16_SFLOAT);
  case DeviceFeatures::TextureFloat:
    return supportsFormat(ctx_->vf_, physicalDevice, VK_FORMAT_R32G32B32A32_SFLOAT) ||
           supportsFormat(ctx_->vf_, physicalDevice, VK_FORMAT_R32_SFLOAT);
  case DeviceFeatures::Texture2DArray:
  case DeviceFeatures::Texture3D:
    return true;
  case DeviceFeatures::StorageBuffers:
    return true;
  case DeviceFeatures::ShaderTextureLod:
    return true;
  case DeviceFeatures::ShaderTextureLodExt:
    return false;
  case DeviceFeatures::DepthShaderRead:
    return true;
  case DeviceFeatures::DepthCompare:
    return true;
  case DeviceFeatures::MinMaxBlend:
    return true;
  case DeviceFeatures::TextureExternalImage:
    return false;
  case DeviceFeatures::Compute:
    return true;
  case DeviceFeatures::CopyBuffer:
  case DeviceFeatures::FillBuffer:
  case DeviceFeatures::UpdateBuffer:
    return true;
  case DeviceFeatures::ExplicitBinding:
    return true;
  case DeviceFeatures::ExplicitBindingExt:
    return false;
  case DeviceFeatures::ExternalMemoryObjects:
    return true;
  case DeviceFeatures::TextureBindless:
    return ctx_->vkPhysicalDeviceDescriptorIndexingProperties_
               .shaderSampledImageArrayNonUniformIndexingNative == VK_TRUE;
  case DeviceFeatures::PushConstants:
    return true;
  case DeviceFeatures::BufferDeviceAddress:
    return true;
  case DeviceFeatures::Multiview:
    return ctx_->features().featuresMultiview.multiview == VK_TRUE;
  case DeviceFeatures::MultiViewMultisample:
    return ctx_->features().featuresMultiview.multiview == VK_TRUE &&
           deviceProperties.limits.framebufferColorSampleCounts > VK_SAMPLE_COUNT_1_BIT;
  case DeviceFeatures::BindUniform:
    return false;
  case DeviceFeatures::TexturePartialMipChain:
    return true;
  case DeviceFeatures::BufferRing:
    return true;
  case DeviceFeatures::BufferNoCopy:
    return false;
  case DeviceFeatures::ShaderLibrary:
    return true;
  case DeviceFeatures::BindBytes:
    return false;
  case DeviceFeatures::TextureArrayExt:
    return false;
  case DeviceFeatures::SRGB:
    return true;
  case DeviceFeatures::SRGBSwapchain:
    return true;
  // on Metal and Vulkan, the framebuffer pixel format dictates sRGB control.
  case DeviceFeatures::SRGBWriteControl:
    return false;
  case DeviceFeatures::SamplerMinMaxLod:
    return true;
  case DeviceFeatures::DrawFirstIndexFirstVertex:
    return true;
  case DeviceFeatures::DrawIndexedIndirect:
    return true;
  case DeviceFeatures::DrawInstanced:
    return true;
  case DeviceFeatures::DynamicCullMode:
    return ctx_->vf_.vkCmdSetCullMode != nullptr;
  case DeviceFeatures::DynamicFrontFacingWinding:
    return ctx_->vf_.vkCmdSetFrontFace != nullptr;
  case DeviceFeatures::DynamicVertexBufferStride:
    return ctx_->vf_.vkCmdBindVertexBuffers2 != nullptr &&
           ctx_->config_.enableDynamicVertexBufferStride;
  case DeviceFeatures::Indices8Bit:
    return ctx_->features_.has_VK_EXT_index_type_uint8;
  case DeviceFeatures::ValidationLayersEnabled:
    return ctx_->areValidationLayersEnabled();
  case DeviceFeatures::TextureViews:
    return true;
  case DeviceFeatures::Timers:
    return deviceProperties.limits.timestampPeriod > 0.0f &&
           (deviceProperties.limits.timestampComputeAndGraphics == VK_TRUE) &&
           ctx_->getGraphicsQueueTimestampValidBits() > 0;
  case DeviceFeatures::TimestampQueries:
    return deviceProperties.limits.timestampPeriod > 0.0f &&
           (deviceProperties.limits.timestampComputeAndGraphics == VK_TRUE);
  case DeviceFeatures::Subpasses:
    return ctx_->supportsSubpasses();
  case DeviceFeatures::MemorylessAttachments:
    return ctx_->supportsLazilyAllocatedMemory();
  case DeviceFeatures::MultiviewSubpasses:
    return ctx_->supportsMultiviewSubpasses();
  }

  IGL_DEBUG_ABORT("DeviceFeatures value not handled: %d", static_cast<int>(feature));

  return false;
}

bool Device::hasRequirementInternal(DeviceRequirement requirement) const {
  IGL_PROFILER_FUNCTION();

  switch (requirement) {
  case DeviceRequirement::ExplicitBindingExtReq:
  case DeviceRequirement::StandardDerivativeExtReq:
  case DeviceRequirement::TextureArrayExtReq:
  case DeviceRequirement::TextureFormatRGExtReq:
  case DeviceRequirement::ShaderTextureLodExtReq:
    return false;
  };

  assert(false);

  return false;
}

/**
 * @brief Queries Vulkan physical device limits for a given
 *        feature limit type.
 *
 * Maps IGL DeviceFeatureLimits values to corresponding
 * VkPhysicalDeviceLimits properties. D3D12-specific descriptor
 * heap limits are unsupported and return false.
 *
 * @param[in] featureLimits The feature limit to query.
 * @param[out] result Populated with the limit value on
 *        success, or 0 if unsupported.
 * @return True if the limit is supported, false otherwise.
 */
bool Device::getFeatureLimitsInternal(DeviceFeatureLimits featureLimits, size_t& result) const {
  IGL_PROFILER_FUNCTION();

  const VkPhysicalDeviceLimits& limits = ctx_->getVkPhysicalDeviceProperties().limits;

  switch (featureLimits) {
  case DeviceFeatureLimits::MaxTextureDimension1D2D:
    result = std::min(limits.maxImageDimension1D, limits.maxImageDimension2D);
    return true;
  case DeviceFeatureLimits::MaxCubeMapDimension:
    result = limits.maxImageDimensionCube;
    return true;
  case DeviceFeatureLimits::MaxStorageBufferBytes:
    result = limits.maxStorageBufferRange;
    return true;
  case DeviceFeatureLimits::MaxVertexUniformVectors:
  case DeviceFeatureLimits::MaxFragmentUniformVectors:
  case DeviceFeatureLimits::MaxUniformBufferBytes:
    result = limits.maxUniformBufferRange;
    return true;
  case DeviceFeatureLimits::MaxPushConstantBytes:
    result = limits.maxPushConstantsSize;
    return true;
  case DeviceFeatureLimits::MaxMultisampleCount: {
    const VkSampleCountFlags sampleCounts = limits.framebufferColorSampleCounts;
    if ((sampleCounts & VK_SAMPLE_COUNT_64_BIT) != 0) {
      result = 64;
    } else if ((sampleCounts & VK_SAMPLE_COUNT_32_BIT) != 0) {
      result = 32;
    } else if ((sampleCounts & VK_SAMPLE_COUNT_16_BIT) != 0) {
      result = 16;
    } else if ((sampleCounts & VK_SAMPLE_COUNT_8_BIT) != 0) {
      result = 8;
    } else if ((sampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0) {
      result = 4;
    } else if ((sampleCounts & VK_SAMPLE_COUNT_2_BIT) != 0) {
      result = 2;
    } else {
      result = 1;
    }
    return true;
  }
  case DeviceFeatureLimits::PushConstantsAlignment:
    result = 4;
    return true;
  case DeviceFeatureLimits::ShaderStorageBufferOffsetAlignment:
    result = limits.minStorageBufferOffsetAlignment;
    return true;
  case DeviceFeatureLimits::BufferAlignment:
    result = limits.minUniformBufferOffsetAlignment;
    return true;
  case DeviceFeatureLimits::BufferNoCopyAlignment:
    result = 0;
    return true;
  case DeviceFeatureLimits::MaxBindBytesBytes:
    result = 0;
    return true;
  case DeviceFeatureLimits::MaxTextureDimension3D:
    result = limits.maxImageDimension3D;
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupSizeX:
    result = limits.maxComputeWorkGroupSize[0];
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupSizeY:
    result = limits.maxComputeWorkGroupSize[1];
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupSizeZ:
    result = limits.maxComputeWorkGroupSize[2];
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupInvocations:
    result = limits.maxComputeWorkGroupInvocations;
    return true;
  case DeviceFeatureLimits::MaxVertexInputAttributes:
    result = limits.maxVertexInputAttributes;
    return true;
  case DeviceFeatureLimits::MaxColorAttachments:
    result = limits.maxColorAttachments;
    return true;
  // D3D12-specific descriptor heap limits - not applicable to Vulkan
  case DeviceFeatureLimits::MaxDescriptorHeapCbvSrvUav:
  case DeviceFeatureLimits::MaxDescriptorHeapSamplers:
  case DeviceFeatureLimits::MaxDescriptorHeapRtvs:
  case DeviceFeatureLimits::MaxDescriptorHeapDsvs:
    result = 0;
    return false;
  }

  IGL_DEBUG_ABORT("DeviceFeatureLimits value not handled: %d", static_cast<int>(featureLimits));
  result = 0;
  return false;
}

ICapabilities::TextureFormatCapabilities Device::getTextureFormatCapabilitiesInternal(
    TextureFormat format) const {
  IGL_PROFILER_FUNCTION();

  const VkFormat vkFormat = igl::vulkan::textureFormatToVkFormat(format);

  if (vkFormat == VK_FORMAT_UNDEFINED) {
    return TextureFormatCapabilityBits::Unsupported;
  }

  if (vkFormat == VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG ||
      vkFormat == VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG ||
      vkFormat == VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG ||
      vkFormat == VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG ||
      vkFormat == VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG ||
      vkFormat == VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG ||
      vkFormat == VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG ||
      vkFormat == VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG) {
    // Deprecated without replacement
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_IMG_format_pvrtc.html
    return TextureFormatCapabilityBits::Unsupported;
  }

  VkFormatProperties properties;
  ctx_->vf_.vkGetPhysicalDeviceFormatProperties(ctx_->vkPhysicalDevice_, vkFormat, &properties);

  const VkFormatFeatureFlags features = properties.optimalTilingFeatures;

  TextureFormatCapabilities caps = TextureFormatCapabilityBits::Unsupported;

  if ((features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0) {
    caps |= TextureFormatCapabilityBits::Sampled;
  }
  if ((features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0) {
    caps |= TextureFormatCapabilityBits::Storage;
  }
  if ((features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0) {
    caps |= TextureFormatCapabilityBits::SampledFiltered;
  }
  if ((features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0) {
    caps |= TextureFormatCapabilityBits::Attachment;
  }
  if ((features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
    caps |= TextureFormatCapabilityBits::Attachment;
  }

  // Special handling for when a format can be sampled AND used as an attachment
  if (contains(caps, TextureFormatCapabilityBits::Sampled) &&
      contains(caps, TextureFormatCapabilityBits::Attachment)) {
    caps |= TextureFormatCapabilityBits::SampledAttachment;
  }

  return caps;
}

ShaderVersion Device::getShaderVersionInternal() const {
  return {.family = ShaderFamily::SpirV, .majorVersion = 1, .minorVersion = 5, .extra = 0};
}

BackendVersion Device::getBackendVersionInternal() const {
  const uint32_t apiVersion = ctx_->vkPhysicalDeviceProperties2_.properties.apiVersion;
  return {
      .flavor = BackendFlavor::Vulkan,
      .majorVersion = static_cast<uint8_t>(VK_API_VERSION_MAJOR(apiVersion)),
      .minorVersion = static_cast<uint8_t>(VK_API_VERSION_MINOR(apiVersion)),
  };
}

Holder<BindGroupTextureHandle> Device::createBindGroupInternal(
    const BindGroupTextureDesc& desc,
    const IRenderPipelineState* IGL_NULLABLE compatiblePipeline,
    Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(ctx_);
  IGL_DEBUG_ASSERT(!desc.debugName.empty(), "Each bind group should have a debug name");
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  return {this, ctx_->createBindGroup(desc, compatiblePipeline, outResult)};
}

Holder<BindGroupBufferHandle> Device::createBindGroupInternal(const BindGroupBufferDesc& desc,
                                                              Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(ctx_);
  IGL_DEBUG_ASSERT(!desc.debugName.empty(), "Each bind group should have a debug name");
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  return {this, ctx_->createBindGroup(desc, outResult)};
}

std::shared_ptr<ITimer> Device::createTimer(Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  if (!hasFeatureInternal(DeviceFeatures::Timers)) {
    Result::setResult(
        outResult, Result::Code::Unsupported, "Timers are not supported by this Vulkan device");
    return nullptr;
  }

  auto timer = std::make_shared<Timer>(*ctx_);
  if (!timer->isValid()) {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "Timer: failed to create Vulkan query pool");
    return nullptr;
  }

  Result::setOk(outResult);
  return timer;
}

void Device::destroyInternal(BindGroupTextureHandle handle) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  IGL_DEBUG_ASSERT(ctx_);
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  ctx_->destroy(handle);
}

void Device::destroyInternal(BindGroupBufferHandle handle) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  IGL_DEBUG_ASSERT(ctx_);
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  ctx_->destroy(handle);
}

void Device::destroyInternal(SamplerHandle handle) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  IGL_DEBUG_ASSERT(ctx_);
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(ctx_);

  ctx_->destroy(handle);
}

void Device::setCurrentThreadInternal() {
  IGL_PROFILER_FUNCTION();
  IGL_DEBUG_ASSERT(ctx_);

  ctx_->setCurrentContextThread();
}

bool Device::isDeviceLostInternal() const noexcept {
  return ctx_->isDeviceLost();
}

} // namespace igl::vulkan
