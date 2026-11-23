/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ShaderStagesCreator.h"

namespace igl::shell {

namespace {
const char* getEntryPointName(ShaderStage stage, BackendType backendType) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (backendType) {
  case BackendType::Metal:
    switch (stage) {
    case ShaderStage::Vertex:
      return "vertexMain";
    case ShaderStage::Fragment:
      return "fragmentMain";
    case ShaderStage::Compute:
      return "computeMain";
    case ShaderStage::Task:
      return "taskMain";
    case ShaderStage::Mesh:
      return "meshMain";
    }
  case BackendType::Vulkan:
    return "main";
  default:
    return "main";
  }
}

ShaderModuleInfo getShaderModuleInfo(ShaderStage stage, BackendType backendType) {
  return {.stage = stage, .entryPoint = getEntryPointName(stage, backendType)};
}
} // namespace

std::unique_ptr<IShaderStages> createRenderPipelineStages(
    const IDevice& device,
    const IShaderProvider& vertShaderProvider,
    const IShaderProvider& fragShaderProvider) {
  Result result;
  auto backend = device.getBackendType();
  if (backend == BackendType::Vulkan) {
    auto vertWords = vertShaderProvider.getShaderBinary(device);
    auto fragWords = fragShaderProvider.getShaderBinary(device);
    auto vertModule =
        igl::ShaderModuleCreator::fromBinaryInput(device,
                                                  vertWords.data(),
                                                  vertWords.size() * sizeof(uint32_t),
                                                  getShaderModuleInfo(ShaderStage::Vertex, backend),
                                                  "",
                                                  &result);
    auto fragModule = igl::ShaderModuleCreator::fromBinaryInput(
        device,
        fragWords.data(),
        fragWords.size() * sizeof(uint32_t),
        getShaderModuleInfo(ShaderStage::Fragment, backend),
        "",
        &result);
    return igl::ShaderStagesCreator::fromRenderModules(
        device, std::move(vertModule), std::move(fragModule), &result);
  } else {
    auto vertModule =
        igl::ShaderModuleCreator::fromStringInput(device,
                                                  vertShaderProvider.getShaderText(device).c_str(),
                                                  getShaderModuleInfo(ShaderStage::Vertex, backend),
                                                  "",
                                                  &result);
    auto fragModule = igl::ShaderModuleCreator::fromStringInput(
        device,
        fragShaderProvider.getShaderText(device).c_str(),
        getShaderModuleInfo(ShaderStage::Fragment, backend),
        "",
        &result);
    return igl::ShaderStagesCreator::fromRenderModules(
        device, std::move(vertModule), std::move(fragModule), &result);
  }
}

std::unique_ptr<IShaderStages> createComputePipelineStages(
    const IDevice& device,
    const IShaderProvider& compShaderProvider) {
  Result result;
  auto backend = device.getBackendType();
  if (backend == BackendType::Vulkan) {
    auto compWords = compShaderProvider.getShaderBinary(device);
    auto compModule = igl::ShaderModuleCreator::fromBinaryInput(
        device,
        compWords.data(),
        compWords.size() * sizeof(uint32_t),
        getShaderModuleInfo(ShaderStage::Compute, backend),
        "",
        &result);
    return igl::ShaderStagesCreator::fromComputeModule(device, std::move(compModule), &result);
  } else {
    auto compModule = igl::ShaderModuleCreator::fromStringInput(
        device,
        compShaderProvider.getShaderText(device).c_str(),
        getShaderModuleInfo(ShaderStage::Compute, backend),
        "",
        &result);
    return igl::ShaderStagesCreator::fromComputeModule(device, std::move(compModule), &result);
  }
}
} // namespace igl::shell
