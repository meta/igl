/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "Common.h"

#include "../data/ShaderData.h"
#include "TestDevice.h"

#include <igl/ShaderCreator.h>

namespace igl::tests::util {

//
// Creates an IGL device and a command queue
//
void createDeviceAndQueue(std::shared_ptr<IDevice>& dev, std::shared_ptr<ICommandQueue>& cq) {
  Result ret;

  // Create Device
  dev = createTestDevice();
  ASSERT_TRUE(dev != nullptr);

  // Create Command Queue
  const CommandQueueDesc cqDesc = {};
  cq = dev->createCommandQueue(cqDesc, &ret);

  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(cq != nullptr); // Shouldn't trigger if above is okay
}

void createShaderStages(const std::shared_ptr<IDevice>& dev,
                        std::string_view vertexSource,
                        std::string_view vertexEntryPoint,
                        std::string_view fragmentSource,
                        std::string_view fragmentEntryPoint,
                        std::unique_ptr<IShaderStages>& stages) {
  Result ret;
  stages = ShaderStagesCreator::fromModuleStringInput(*dev,
                                                      std::string(vertexSource).c_str(),
                                                      std::string(vertexEntryPoint),
                                                      "",
                                                      std::string(fragmentSource).c_str(),
                                                      std::string(fragmentEntryPoint),
                                                      "",
                                                      &ret);

  ASSERT_EQ(ret.code, Result::Code::Ok) << ret.message.c_str();
  ASSERT_TRUE(stages != nullptr);
}

void createShaderStages(const std::shared_ptr<IDevice>& dev,
                        std::string_view librarySource,
                        std::string_view vertexEntryPoint,
                        std::string_view fragmentEntryPoint,
                        std::unique_ptr<IShaderStages>& stages) {
  Result ret;
  stages = ShaderStagesCreator::fromLibraryStringInput(*dev,
                                                       std::string(librarySource).c_str(),
                                                       std::string(vertexEntryPoint),
                                                       std::string(fragmentEntryPoint),
                                                       "",
                                                       &ret);

  ASSERT_EQ(ret.code, Result::Code::Ok) << ret.message;
  ASSERT_TRUE(stages != nullptr);
}

void createSimpleShaderStages(const std::shared_ptr<IDevice>& dev,
                              std::unique_ptr<IShaderStages>& stages,
                              TextureFormat outputFormat) {
  const auto backendVersion = dev->getBackendVersion();

  if (backendVersion.flavor == BackendFlavor::OpenGL ||
      backendVersion.flavor == BackendFlavor::OpenGL_ES) {
    const bool isGles3 = backendVersion.flavor == BackendFlavor::OpenGL_ES &&
                         backendVersion.majorVersion >= 3;
    std::string_view vertexShader = isGles3 ? data::shader::kOglSimpleVertShaderEs3
                                            : data::shader::kOglSimpleVertShader;
    std::string_view fragmentShader = isGles3 ? data::shader::kOglSimpleFragShaderEs3
                                              : data::shader::kOglSimpleFragShader;

    createShaderStages(dev,
                       vertexShader,
                       data::shader::kShaderFunc,
                       fragmentShader,
                       data::shader::kShaderFunc,
                       stages);
  } else if (backendVersion.flavor == BackendFlavor::Metal) {
    std::string_view shader = data::shader::kMtlSimpleShader;
    if (outputFormat == TextureFormat::RG_UInt16) {
      shader = data::shader::kMtlSimpleShaderUshort2;
    } else if (outputFormat == TextureFormat::R_UInt16) {
      shader = data::shader::kMtlSimpleShaderUshort2;
    } else if (outputFormat == TextureFormat::RGB10_A2_Uint_Rev) {
      shader = data::shader::kMtlSimpleShaderUshort4;
    } else if (outputFormat == TextureFormat::RGBA_UInt32) {
      shader = data::shader::kMtlSimpleShaderUint4;
    } else if (outputFormat == TextureFormat::R_UInt32) {
      shader = data::shader::kMtlSimpleShaderUint;
    } else if (outputFormat != TextureFormat::Invalid) {
      auto components = TextureFormatProperties::fromTextureFormat(outputFormat).componentsPerPixel;
      switch (components) {
      case 1:
        shader = data::shader::kMtlSimpleShaderFloat;
        break;
      case 2:
        shader = data::shader::kMtlSimpleShaderFloat2;
        break;
      case 3:
        shader = data::shader::kMtlSimpleShaderFloat3;
        break;
      case 4:
        shader = data::shader::kMtlSimpleShaderFloat4;
        break;
      default:
        ASSERT_TRUE(false);
        break;
      }
    }
    createShaderStages(
        dev, shader, data::shader::kSimpleVertFunc, data::shader::kSimpleFragFunc, stages);
  } else if (backendVersion.flavor == BackendFlavor::Vulkan) {
    // Output format-specific shaders needed for MoltenVK
    std::string_view fragShader = data::shader::kVulkanSimpleFragShader;
    if (outputFormat == TextureFormat::RG_UInt16) {
      fragShader = data::shader::kVulkanSimpleFragShaderUint2;
    } else if (outputFormat == TextureFormat::R_UInt16) {
      fragShader = data::shader::kVulkanSimpleFragShaderUint2;
    } else if (outputFormat == TextureFormat::RGB10_A2_Uint_Rev) {
      fragShader = data::shader::kVulkanSimpleFragShaderUint4;
    } else if (outputFormat == TextureFormat::RGBA_UInt32) {
      fragShader = data::shader::kVulkanSimpleFragShaderUint4;
    } else if (outputFormat == TextureFormat::R_UInt32) {
      fragShader = data::shader::kVulkanSimpleFragShaderUint;
    } else if (outputFormat != TextureFormat::Invalid) {
      auto components = TextureFormatProperties::fromTextureFormat(outputFormat).componentsPerPixel;
      switch (components) {
      case 1:
        fragShader = data::shader::kVulkanSimpleFragShaderFloat;
        break;
      case 2:
        fragShader = data::shader::kVulkanSimpleFragShaderFloat2;
        break;
      case 3:
        fragShader = data::shader::kVulkanSimpleFragShaderFloat3;
        break;
      case 4:
        fragShader = data::shader::kVulkanSimpleFragShaderFloat4;
        break;
      default:
        ASSERT_TRUE(false);
        break;
      }
    }
    createShaderStages(dev,
                       data::shader::kVulkanSimpleVertShader,
                       std::string(data::shader::kShaderFunc),
                       fragShader,
                       std::string(data::shader::kShaderFunc),
                       stages);
  } else if (backendVersion.flavor == BackendFlavor::WebGPU) {
    // WebGPU requires the fragment output type to match the color target's sample type; extra
    // output components are dropped.
    const bool isUint = outputFormat != TextureFormat::Invalid &&
                        TextureFormatProperties::fromTextureFormat(outputFormat).isInteger();
    const std::string_view fragShader = isUint ? data::shader::kWgslSimpleFragShaderUint4
                                               : data::shader::kWgslSimpleFragShader;
    createShaderStages(dev,
                       data::shader::kWgslSimpleVertShader,
                       data::shader::kShaderFunc,
                       fragShader,
                       data::shader::kShaderFunc,
                       stages);
  } else if (backendVersion.flavor == BackendFlavor::D3D12) {
    // Minimal HLSL equivalent used for D3D12 tests
    const char* vsHlsl = R"(
struct VSIn { float4 position_in : POSITION; float2 uv_in : TEXCOORD0; };
struct PSIn { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
PSIn main(VSIn i) { PSIn o; o.position = i.position_in; o.uv = i.uv_in; return o; }
)";
    const char* psHlsl = R"(
Texture2D inputImage : register(t0);
SamplerState samp0 : register(s0);
struct PSIn { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
float4 main(PSIn i) : SV_TARGET { return inputImage.Sample(samp0, i.uv); }
)";
    createShaderStages(dev, vsHlsl, std::string("main"), psHlsl, std::string("main"), stages);
  } else {
    ASSERT_TRUE(0);
  }
}

} // namespace igl::tests::util
