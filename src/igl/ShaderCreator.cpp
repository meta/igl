/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/ShaderCreator.h>

#include <igl/Device.h>

namespace igl {
namespace {
std::unique_ptr<IShaderStages> fromLibraryDesc(const IDevice& device,
                                               const ShaderLibraryDesc& libraryDesc,
                                               Result* IGL_NULLABLE outResult);
} // namespace

std::shared_ptr<IShaderModule> ShaderModuleCreator::fromStringInput(const IDevice& device,
                                                                    const char* IGL_NONNULL source,
                                                                    ShaderModuleInfo info,
                                                                    std::string debugName,
                                                                    Result* IGL_NULLABLE
                                                                        outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  const auto desc =
      igl::ShaderModuleDesc::fromStringInput(source, std::move(info), std::move(debugName));

  auto sm = device.createShaderModule(desc, result);
  if (result && !result->isOk()) {
    return nullptr;
  }

  return sm;
}

std::shared_ptr<IShaderModule> ShaderModuleCreator::fromBinaryInput(const IDevice& device,
                                                                    const void* IGL_NONNULL data,
                                                                    size_t dataLength,
                                                                    ShaderModuleInfo info,
                                                                    std::string debugName,
                                                                    Result* IGL_NULLABLE
                                                                        outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  const auto desc = igl::ShaderModuleDesc::fromBinaryInput(
      data, dataLength, std::move(info), std::move(debugName));

  auto sm = device.createShaderModule(desc, result);
  if (result && !result->isOk()) {
    return nullptr;
  }

  return sm;
}

std::unique_ptr<IShaderLibrary> ShaderLibraryCreator::fromStringInput(
    const IDevice& device,
    const char* IGL_NONNULL librarySource,
    std::string vertexEntryPoint,
    std::string fragmentEntryPoint,
    std::string libraryDebugName,
    Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  const auto libraryDesc = igl::ShaderLibraryDesc::fromStringInput(
      librarySource,
      {
          {.stage = ShaderStage::Vertex, .entryPoint = std::move(vertexEntryPoint)},
          {.stage = ShaderStage::Fragment, .entryPoint = std::move(fragmentEntryPoint)},
      },
      std::move(libraryDebugName));

  auto library = device.createShaderLibrary(libraryDesc, result);
  return (result && result->isOk()) ? std::move(library) : nullptr;
}

std::unique_ptr<IShaderLibrary> ShaderLibraryCreator::fromBinaryInput(
    const IDevice& device,
    const void* IGL_NONNULL libraryData,
    size_t libraryDataLength,
    std::string vertexEntryPoint,
    std::string fragmentEntryPoint,
    std::string libraryDebugName,
    Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  const auto libraryDesc = igl::ShaderLibraryDesc::fromBinaryInput(
      libraryData,
      libraryDataLength,
      {
          {.stage = ShaderStage::Vertex, .entryPoint = std::move(vertexEntryPoint)},
          {.stage = ShaderStage::Fragment, .entryPoint = std::move(fragmentEntryPoint)},
      },
      std::move(libraryDebugName));

  auto library = device.createShaderLibrary(libraryDesc, result);
  return (result && result->isOk()) ? std::move(library) : nullptr;
}

std::unique_ptr<IShaderLibrary> ShaderLibraryCreator::fromStringInput(
    const IDevice& device,
    const char* IGL_NONNULL librarySource,
    std::vector<ShaderModuleInfo> moduleInfo,
    std::string libraryDebugName,
    Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  const auto libraryDesc = igl::ShaderLibraryDesc::fromStringInput(
      librarySource, std::move(moduleInfo), std::move(libraryDebugName));

  auto library = device.createShaderLibrary(libraryDesc, result);
  return (result && result->isOk()) ? std::move(library) : nullptr;
}

std::unique_ptr<IShaderLibrary> ShaderLibraryCreator::fromBinaryInput(
    const IDevice& device,
    const void* IGL_NONNULL libraryData,
    size_t libraryDataLength,
    std::vector<ShaderModuleInfo> moduleInfo,
    std::string libraryDebugName,
    Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  const auto libraryDesc = igl::ShaderLibraryDesc::fromBinaryInput(
      libraryData, libraryDataLength, std::move(moduleInfo), std::move(libraryDebugName));

  auto library = device.createShaderLibrary(libraryDesc, result);
  return (result && result->isOk()) ? std::move(library) : nullptr;
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromModuleStringInput(
    const IDevice& device,
    const char* IGL_NONNULL vertexSource,
    std::string vertexEntryPoint,
    std::string vertexDebugName,
    const char* IGL_NONNULL fragmentSource,
    std::string fragmentEntryPoint,
    std::string fragmentDebugName,
    Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  std::shared_ptr<IShaderModule> vertexModule, fragmentModule;
  vertexModule = ShaderModuleCreator::fromStringInput(
      device,
      vertexSource,
      {.stage = ShaderStage::Vertex, .entryPoint = std::move(vertexEntryPoint)},
      std::move(vertexDebugName),
      result);
  if (result->isOk()) {
    fragmentModule = ShaderModuleCreator::fromStringInput(
        device,
        fragmentSource,
        {.stage = ShaderStage::Fragment, .entryPoint = std::move(fragmentEntryPoint)},
        std::move(fragmentDebugName),
        result);
  }

  return (result && result->isOk())
             ? fromRenderModules(device, std::move(vertexModule), std::move(fragmentModule), result)
             : nullptr;
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromModuleBinaryInput(
    const IDevice& device,
    const void* IGL_NONNULL vertexData,
    size_t vertexDataLength,
    std::string vertexEntryPoint,
    std::string vertexDebugName,
    const void* IGL_NONNULL fragmentData,
    size_t fragmentDataLength,
    std::string fragmentEntryPoint,
    std::string fragmentDebugName,
    Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  std::shared_ptr<IShaderModule> vertexModule, fragmentModule;
  vertexModule = ShaderModuleCreator::fromBinaryInput(
      device,
      vertexData,
      vertexDataLength,
      {.stage = ShaderStage::Vertex, .entryPoint = std::move(vertexEntryPoint)},
      std::move(vertexDebugName),
      result);
  if (result->isOk()) {
    fragmentModule = ShaderModuleCreator::fromBinaryInput(
        device,
        fragmentData,
        fragmentDataLength,
        {.stage = ShaderStage::Fragment, .entryPoint = std::move(fragmentEntryPoint)},
        std::move(fragmentDebugName),
        result);
  }

  return (result && result->isOk())
             ? fromRenderModules(device, std::move(vertexModule), std::move(fragmentModule), result)
             : nullptr;
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromLibraryStringInput(
    const IDevice& device,
    const char* IGL_NONNULL librarySource,
    std::string vertexEntryPoint,
    std::string fragmentEntryPoint,
    std::string libraryDebugName,
    Result* IGL_NULLABLE outResult) {
  const auto libraryDesc = ShaderLibraryDesc::fromStringInput(
      librarySource,
      {
          {.stage = ShaderStage::Vertex, .entryPoint = std::move(vertexEntryPoint)},
          {.stage = ShaderStage::Fragment, .entryPoint = std::move(fragmentEntryPoint)},
      },
      std::move(libraryDebugName));

  return fromLibraryDesc(device, libraryDesc, outResult);
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromLibraryBinaryInput(
    const IDevice& device,
    const void* IGL_NONNULL libraryData,
    size_t libraryDataLength,
    std::string vertexEntryPoint,
    std::string fragmentEntryPoint,
    std::string libraryDebugName,
    Result* IGL_NULLABLE outResult) {
  const auto libraryDesc = ShaderLibraryDesc::fromBinaryInput(
      libraryData,
      libraryDataLength,
      {
          {.stage = ShaderStage::Vertex, .entryPoint = std::move(vertexEntryPoint)},
          {.stage = ShaderStage::Fragment, .entryPoint = std::move(fragmentEntryPoint)},
      },
      std::move(libraryDebugName));

  return fromLibraryDesc(device, libraryDesc, outResult);
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromModuleStringInput(
    const IDevice& device,
    const char* IGL_NONNULL computeSource,
    std::string computeEntryPoint,
    std::string computeDebugName,
    Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  auto computeModule = ShaderModuleCreator::fromStringInput(
      device,
      computeSource,
      {.stage = ShaderStage::Compute, .entryPoint = std::move(computeEntryPoint)},
      std::move(computeDebugName),
      result);
  return (result && result->isOk()) ? fromComputeModule(device, std::move(computeModule), result)
                                    : nullptr;
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromModuleBinaryInput(
    const IDevice& device,
    const void* IGL_NONNULL computeData,
    size_t computeDataLength,
    std::string computeEntryPoint,
    std::string computeDebugName,
    Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  auto computeModule = ShaderModuleCreator::fromBinaryInput(
      device,
      computeData,
      computeDataLength,
      {.stage = ShaderStage::Compute, .entryPoint = std::move(computeEntryPoint)},
      std::move(computeDebugName),
      result);
  return (result && result->isOk()) ? fromComputeModule(device, std::move(computeModule), result)
                                    : nullptr;
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromRenderModules(
    const IDevice& device,
    std::shared_ptr<IShaderModule> vertexModule,
    std::shared_ptr<IShaderModule> fragmentModule,
    Result* IGL_NULLABLE outResult) {
  const auto desc =
      ShaderStagesDesc::fromRenderModules(std::move(vertexModule), std::move(fragmentModule));
  return device.createShaderStages(desc, outResult);
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromComputeModule(
    const IDevice& device,
    std::shared_ptr<IShaderModule> computeModule,
    Result* IGL_NULLABLE outResult) {
  const auto desc = ShaderStagesDesc::fromComputeModule(std::move(computeModule));
  return device.createShaderStages(desc, outResult);
}

std::unique_ptr<IShaderStages> ShaderStagesCreator::fromMeshRenderModules(
    const IDevice& device,
    std::shared_ptr<IShaderModule> taskModule,
    std::shared_ptr<IShaderModule> meshModule,
    std::shared_ptr<IShaderModule> fragmentModule,
    Result* IGL_NULLABLE outResult) {
  const auto desc = ShaderStagesDesc::fromMeshRenderModules(
      std::move(taskModule), std::move(meshModule), std::move(fragmentModule));
  return device.createShaderStages(desc, outResult);
}

namespace {
std::unique_ptr<IShaderStages> fromLibraryDesc(const IDevice& device,
                                               const ShaderLibraryDesc& libraryDesc,
                                               Result* IGL_NULLABLE outResult) {
  Result localResult;
  Result* result = outResult ? outResult : &localResult;
  IGL_DEBUG_ASSERT(result);

  if (!IGL_DEBUG_VERIFY(libraryDesc.moduleInfo.size() == 2)) {
    Result::setResult(result,
                      Result::Code::ArgumentInvalid,
                      "Library descriptor must contain 2 module info entries");
    return nullptr;
  }

  auto library = device.createShaderLibrary(libraryDesc, result);
  if (result && !result->isOk()) {
    return nullptr;
  }

  auto vertexModule = library->getShaderModule(libraryDesc.moduleInfo[0].entryPoint);
  if (vertexModule == nullptr) {
    Result::setResult(
        result, Result::Code::ArgumentInvalid, "Could not retrieve vertex module from library");
    return nullptr;
  }
  auto fragmentModule = library->getShaderModule(libraryDesc.moduleInfo[1].entryPoint);
  if (fragmentModule == nullptr) {
    Result::setResult(
        result, Result::Code::ArgumentInvalid, "Could not retrieve fragment module from library");
    return nullptr;
  }

  return ShaderStagesCreator::fromRenderModules(
      device, std::move(vertexModule), std::move(fragmentModule), result);
}
} // namespace
} // namespace igl
