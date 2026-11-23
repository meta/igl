/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "data/ShaderData.h"
#include "util/Common.h"

#include <igl/ShaderCreator.h>

namespace igl::tests {

class ShaderLibraryTest : public ::testing::Test {
 private:
 public:
  ShaderLibraryTest() = default;

  //
  // SetUp()
  //
  void SetUp() override {
    // Turn off debug breaks, only use in debug mode
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

TEST_F(ShaderLibraryTest, CreateFromSource) {
  Result ret;
  if (!iglDev_->hasFeature(DeviceFeatures::ShaderLibrary)) {
    GTEST_SKIP() << "Shader Libraries are unsupported for this platform.";
  }

  const char* source = nullptr;
  if (iglDev_->getBackendType() == BackendType::Metal) {
    source = data::shader::kMtlSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::Vulkan) {
    source = data::shader::kVulkanSimpleVertShader.data();
  } else if (iglDev_->getBackendType() == BackendType::WebGPU) {
    source = data::shader::kWgslSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::D3D12) {
    source = data::shader::kD3D12SimpleShader.data();
  } else {
    IGL_DEBUG_ASSERT_NOT_REACHED();
  }

  // Check if source is null before passing it to fromStringInput
  if (source == nullptr) {
    GTEST_SKIP() << "No shader source available for this backend.";
  }

  auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
      *iglDev_, source, {{.stage = ShaderStage::Vertex, .entryPoint = "vertexShader"}}, "", &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(shaderLibrary != nullptr);

  auto vertShaderModule = shaderLibrary->getShaderModule("vertexShader");
  ASSERT_TRUE(vertShaderModule);
}

TEST_F(ShaderLibraryTest, CreateFromSingleModuleReturnNullWithEmptyInput) {
  Result ret;
  if (!iglDev_->hasFeature(DeviceFeatures::ShaderLibrary)) {
    GTEST_SKIP() << "Shader Libraries are unsupported for this platform.";
  }

  auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
      *iglDev_, "", {{.stage = ShaderStage::Vertex, .entryPoint = ""}}, "", &ret);
  ASSERT_TRUE(!ret.isOk());
  ASSERT_TRUE(shaderLibrary == nullptr);
}

TEST_F(ShaderLibraryTest, CreateFromSourceMultipleModules) {
  Result ret;
  if (!iglDev_->hasFeature(DeviceFeatures::ShaderLibrary)) {
    GTEST_SKIP() << "Shader Libraries are unsupported for this platform.";
  }

  const char* source = nullptr;
  if (iglDev_->getBackendType() == BackendType::Metal) {
    source = data::shader::kMtlSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::Vulkan) {
    GTEST_SKIP() << "Vulkan does not support multiple modules from the same source code.";
  } else if (iglDev_->getBackendType() == BackendType::WebGPU) {
    source = data::shader::kWgslSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::D3D12) {
    source = data::shader::kD3D12SimpleShader.data();
  }

  // Check if source is null before passing it to fromStringInput
  if (source == nullptr) {
    GTEST_SKIP() << "No shader source available for this backend.";
  }

  auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
      *iglDev_,
      source,
      {
          {.stage = ShaderStage::Vertex, .entryPoint = "vertexShader"},
          {.stage = ShaderStage::Fragment, .entryPoint = "fragmentShader"},
      },
      "",
      &ret);

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(shaderLibrary != nullptr);

  auto vertShaderModule = shaderLibrary->getShaderModule("vertexShader");
  ASSERT_TRUE(vertShaderModule);

  auto fragShaderModule = shaderLibrary->getShaderModule("fragmentShader");
  ASSERT_TRUE(fragShaderModule);
}

TEST_F(ShaderLibraryTest, CreateFromSourceNoResult) {
  if (!iglDev_->hasFeature(DeviceFeatures::ShaderLibrary)) {
    GTEST_SKIP() << "Shader Libraries are unsupported for this platform.";
  }

  const char* source = nullptr;
  if (iglDev_->getBackendType() == BackendType::Metal) {
    source = data::shader::kMtlSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::Vulkan) {
    source = data::shader::kVulkanSimpleVertShader.data();
  } else if (iglDev_->getBackendType() == BackendType::WebGPU) {
    source = data::shader::kWgslSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::D3D12) {
    source = data::shader::kD3D12SimpleShader.data();
  } else {
    IGL_DEBUG_ASSERT_NOT_REACHED();
  }

  // Check if source is null before passing it to fromStringInput
  if (source == nullptr) {
    GTEST_SKIP() << "No shader source available for this backend.";
  }

  auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
      *iglDev_,
      source,
      {{.stage = ShaderStage::Vertex, .entryPoint = "vertexShader"}},
      "",
      nullptr);
  ASSERT_TRUE(shaderLibrary != nullptr);

  auto vertShaderModule = shaderLibrary->getShaderModule("vertexShader");
  ASSERT_TRUE(vertShaderModule);
}
TEST_F(ShaderLibraryTest, GetNonExistentModuleReturnsNull) {
  Result ret;
  if (!iglDev_->hasFeature(DeviceFeatures::ShaderLibrary)) {
    GTEST_SKIP() << "Shader Libraries are unsupported for this platform.";
  }

  const char* source = nullptr;
  if (iglDev_->getBackendType() == BackendType::Metal) {
    source = data::shader::kMtlSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::Vulkan) {
    source = data::shader::kVulkanSimpleVertShader.data();
  } else if (iglDev_->getBackendType() == BackendType::WebGPU) {
    source = data::shader::kWgslSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::D3D12) {
    source = data::shader::kD3D12SimpleShader.data();
  }

  if (source == nullptr) {
    GTEST_SKIP() << "No shader source available for this backend.";
  }

  auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
      *iglDev_, source, {{.stage = ShaderStage::Vertex, .entryPoint = "vertexShader"}}, "", &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(shaderLibrary, nullptr);

  auto nonExistent = shaderLibrary->getShaderModule("doesNotExist");
  EXPECT_EQ(nonExistent, nullptr);
}

TEST_F(ShaderLibraryTest, CreateWithDebugName) {
  Result ret;
  if (!iglDev_->hasFeature(DeviceFeatures::ShaderLibrary)) {
    GTEST_SKIP() << "Shader Libraries are unsupported for this platform.";
  }

  const char* source = nullptr;
  if (iglDev_->getBackendType() == BackendType::Metal) {
    source = data::shader::kMtlSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::Vulkan) {
    source = data::shader::kVulkanSimpleVertShader.data();
  } else if (iglDev_->getBackendType() == BackendType::WebGPU) {
    source = data::shader::kWgslSimpleShader.data();
  } else if (iglDev_->getBackendType() == BackendType::D3D12) {
    source = data::shader::kD3D12SimpleShader.data();
  }

  if (source == nullptr) {
    GTEST_SKIP() << "No shader source available for this backend.";
  }

  auto shaderLibrary = ShaderLibraryCreator::fromStringInput(
      *iglDev_,
      source,
      {{.stage = ShaderStage::Vertex, .entryPoint = "vertexShader"}},
      "TestDebugName",
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(shaderLibrary, nullptr);

  auto vertModule = shaderLibrary->getShaderModule("vertexShader");
  ASSERT_NE(vertModule, nullptr);
}

} // namespace igl::tests
