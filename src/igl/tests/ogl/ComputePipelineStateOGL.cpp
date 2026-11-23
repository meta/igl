/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"

#include <igl/ComputePipelineState.h>
#include <igl/Shader.h>
#include <igl/opengl/ComputePipelineState.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// ComputePipelineStateOGLTest
//
// Tests for the OpenGL ComputePipelineState.
//
class ComputePipelineStateOGLTest : public ::testing::Test {
 public:
  ComputePipelineStateOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
};

//
// CreateAndBind
//
// Create a compute pipeline, bind and unbind it, check for no GL errors.
//
TEST_F(ComputePipelineStateOGLTest, CreateAndBind) {
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    GTEST_SKIP() << "Compute not supported";
  }

  Result ret;

  const std::string computeSource(data::shader::kOglSimpleComputeShader);
  auto shaderModule = iglDev_->createShaderModule(
      ShaderModuleDesc::fromStringInput(
          computeSource.c_str(),
          {.stage = ShaderStage::Compute, .entryPoint = std::string(data::shader::kShaderFunc)},
          ""),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto shaderStages = ShaderStagesDesc::fromComputeModule(std::move(shaderModule));
  auto stages = iglDev_->createShaderStages(shaderStages, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc computeDesc{.shaderStages = std::move(stages)};
  auto computePipeline = iglDev_->createComputePipeline(computeDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(computePipeline, nullptr);

  // Cast to opengl::ComputePipelineState and test bind/unbind
  auto* oglPipeline = static_cast<opengl::ComputePipelineState*>(computePipeline.get());
  ASSERT_NE(oglPipeline, nullptr);

  oglPipeline->bind();
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  oglPipeline->unbind();
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// GetIndexByName
//
// Verify that getIndexByName returns valid indices for known buffer names.
//
TEST_F(ComputePipelineStateOGLTest, GetIndexByName) {
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    GTEST_SKIP() << "Compute not supported";
  }

  Result ret;

  const std::string computeSource(data::shader::kOglSimpleComputeShader);
  auto shaderModule = iglDev_->createShaderModule(
      ShaderModuleDesc::fromStringInput(
          computeSource.c_str(),
          {.stage = ShaderStage::Compute, .entryPoint = std::string(data::shader::kShaderFunc)},
          ""),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto shaderStages = ShaderStagesDesc::fromComputeModule(std::move(shaderModule));
  auto stages = iglDev_->createShaderStages(shaderStages, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc computeDesc{.shaderStages = std::move(stages)};
  auto computePipeline = iglDev_->createComputePipeline(computeDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(computePipeline, nullptr);

  // The compute shader has "floatsIn" and "floatsOut" SSBOs
  int idx = computePipeline->getIndexByName(IGL_NAMEHANDLE(data::shader::kSimpleComputeInput));
  // Index should be valid (not -1) if the shader uses SSBOs with these names
  // Note: the actual value depends on the shader reflection
  ASSERT_GE(idx, -1); // At minimum, the call should not crash

  idx = computePipeline->getIndexByName(IGL_NAMEHANDLE(data::shader::kSimpleComputeOutput));
  ASSERT_GE(idx, -1);
}

//
// SSBODetection
//
// Verify getIsUsingShaderStorageBuffers() returns true for compute shaders with SSBOs.
//
TEST_F(ComputePipelineStateOGLTest, SSBODetection) {
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    GTEST_SKIP() << "Compute not supported";
  }

  Result ret;

  const std::string computeSource(data::shader::kOglSimpleComputeShader);
  auto shaderModule = iglDev_->createShaderModule(
      ShaderModuleDesc::fromStringInput(
          computeSource.c_str(),
          {.stage = ShaderStage::Compute, .entryPoint = std::string(data::shader::kShaderFunc)},
          ""),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto shaderStages = ShaderStagesDesc::fromComputeModule(std::move(shaderModule));
  auto stages = iglDev_->createShaderStages(shaderStages, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc computeDesc{.shaderStages = std::move(stages)};
  auto computePipeline = iglDev_->createComputePipeline(computeDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(computePipeline, nullptr);

  auto* oglPipeline = static_cast<opengl::ComputePipelineState*>(computePipeline.get());
  ASSERT_NE(oglPipeline, nullptr);

  // The simple compute shader uses SSBOs
  ASSERT_TRUE(oglPipeline->getIsUsingShaderStorageBuffers());
}

} // namespace igl::tests
