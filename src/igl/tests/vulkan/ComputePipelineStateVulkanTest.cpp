/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../util/TestDevice.h"

#include <cstring>
#include <igl/CommandBuffer.h>
#include <igl/ComputeCommandEncoder.h>
#include <igl/ComputePipelineState.h>
#include <igl/ShaderCreator.h>
#include <igl/vulkan/ComputePipelineState.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX

namespace igl::tests {

//
// ComputePipelineStateVulkanTest
//
// Unit tests for igl::vulkan::ComputePipelineState
//
class ComputePipelineStateVulkanTest : public ::testing::Test {
 public:
  ComputePipelineStateVulkanTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    device_ = util::createTestDevice();
    ASSERT_NE(device_, nullptr);
    ASSERT_EQ(device_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";

    auto& vulkanDevice = static_cast<vulkan::Device&>(*device_);
    context_ = &vulkanDevice.getVulkanContext();
    ASSERT_NE(context_, nullptr);

    Result ret;
    cmdQueue_ = device_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> device_;
  vulkan::VulkanContext* context_ = nullptr;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

//
// CreateComputePipeline
//
// Test creating a compute pipeline with a valid compute shader.
//
TEST_F(ComputePipelineStateVulkanTest, CreateComputePipeline) {
  Result ret;

  const std::string computeSource(data::shader::kVulkanSimpleComputeShader);
  auto computeModule =
      ShaderModuleCreator::fromStringInput(*device_,
                                           computeSource.c_str(),
                                           {.stage = ShaderStage::Compute, .entryPoint = "main"},
                                           "TestComputePipeline",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(computeModule, nullptr);

  auto stages = ShaderStagesCreator::fromComputeModule(*device_, computeModule, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc pipelineDesc{.shaderStages = std::move(stages),
                                         .debugName = "TestComputePipeline"};

  auto pipeline = device_->createComputePipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);
}

//
// CreateComputePipelineNullStages
//
// Test that creating a compute pipeline with null shader stages returns an error.
//
TEST_F(ComputePipelineStateVulkanTest, CreateComputePipelineNullStages) {
  Result ret;

  const ComputePipelineDesc pipelineDesc{.shaderStages = nullptr, .debugName = "TestNullStages"};

  auto pipeline = device_->createComputePipeline(pipelineDesc, &ret);
  EXPECT_FALSE(ret.isOk());
  EXPECT_EQ(pipeline, nullptr);
}

//
// GetComputePipelineDesc
//
// Test that getComputePipelineDesc() returns the correct descriptor.
//
TEST_F(ComputePipelineStateVulkanTest, GetComputePipelineDesc) {
  Result ret;

  const std::string computeSource(data::shader::kVulkanSimpleComputeShader);
  auto computeModule =
      ShaderModuleCreator::fromStringInput(*device_,
                                           computeSource.c_str(),
                                           {.stage = ShaderStage::Compute, .entryPoint = "main"},
                                           "TestGetDesc",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto stages = ShaderStagesCreator::fromComputeModule(*device_, computeModule, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc pipelineDesc{.shaderStages = std::move(stages),
                                         .debugName = "TestGetDesc"};

  auto pipeline = device_->createComputePipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);

  auto* vulkanPipeline = static_cast<vulkan::ComputePipelineState*>(pipeline.get());
  ASSERT_NE(vulkanPipeline, nullptr);

  const auto& desc = vulkanPipeline->getComputePipelineDesc();
  EXPECT_EQ(desc.debugName, "TestGetDesc");
  EXPECT_NE(desc.shaderStages, nullptr);
}

//
// GetVkPipeline
//
// Test that getVkPipeline() creates a valid VkPipeline on demand.
//
TEST_F(ComputePipelineStateVulkanTest, GetVkPipeline) {
  Result ret;

  const std::string computeSource(data::shader::kVulkanSimpleComputeShader);
  auto computeModule =
      ShaderModuleCreator::fromStringInput(*device_,
                                           computeSource.c_str(),
                                           {.stage = ShaderStage::Compute, .entryPoint = "main"},
                                           "TestGetVkPipeline",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto stages = ShaderStagesCreator::fromComputeModule(*device_, computeModule, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc pipelineDesc{.shaderStages = std::move(stages),
                                         .debugName = "TestGetVkPipeline"};

  auto pipeline = device_->createComputePipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);

  auto* vulkanPipeline = static_cast<vulkan::ComputePipelineState*>(pipeline.get());
  ASSERT_NE(vulkanPipeline, nullptr);

  // First call creates the pipeline
  VkPipeline vkPipeline1 = vulkanPipeline->getVkPipeline();
  EXPECT_NE(vkPipeline1, VK_NULL_HANDLE);

  // Second call should return the cached pipeline
  VkPipeline vkPipeline2 = vulkanPipeline->getVkPipeline();
  EXPECT_EQ(vkPipeline1, vkPipeline2);
}

//
// BindAndDispatch
//
// Test that a compute pipeline can be bound and dispatched.
//
TEST_F(ComputePipelineStateVulkanTest, BindAndDispatch) {
  Result ret;

  const std::string computeSource(data::shader::kVulkanSimpleComputeShader);
  auto computeModule =
      ShaderModuleCreator::fromStringInput(*device_,
                                           computeSource.c_str(),
                                           {.stage = ShaderStage::Compute, .entryPoint = "main"},
                                           "TestBindAndDispatch",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto stages = ShaderStagesCreator::fromComputeModule(*device_, computeModule, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc pipelineDesc{
      .buffersMap = {{data::shader::kSimpleComputeInputIndex,
                      IGL_NAMEHANDLE(data::shader::kSimpleComputeInput)},
                     {data::shader::kSimpleComputeOutputIndex,
                      IGL_NAMEHANDLE(data::shader::kSimpleComputeOutput)}},
      .shaderStages = std::move(stages),
      .debugName = "TestBindAndDispatch"};

  auto pipeline = device_->createComputePipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);

  // Create input/output buffers
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const BufferDesc inputBufferDesc{.type = BufferDesc::BufferTypeBits::Storage,
                                   .data = inputData.data(),
                                   .length = inputData.size() * sizeof(float),
                                   .storage = ResourceStorage::Shared};
  auto inputBuffer = device_->createBuffer(inputBufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const BufferDesc outputBufferDesc{.type = BufferDesc::BufferTypeBits::Storage,
                                    .length = inputData.size() * sizeof(float),
                                    .storage = ResourceStorage::Shared};
  auto outputBuffer = device_->createBuffer(outputBufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Create command buffer and encoder
  auto cmdBuffer = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto computeEncoder = cmdBuffer->createComputeCommandEncoder();
  ASSERT_NE(computeEncoder, nullptr);

  computeEncoder->bindComputePipelineState(pipeline);
  computeEncoder->bindBuffer(data::shader::kSimpleComputeInputIndex, inputBuffer.get());
  computeEncoder->bindBuffer(data::shader::kSimpleComputeOutputIndex, outputBuffer.get());

  computeEncoder->dispatchThreadGroups(Dimensions(1, 1, 1), Dimensions(6, 1, 1));
  computeEncoder->endEncoding();

  cmdQueue_->submit(*cmdBuffer);
  cmdBuffer->waitUntilCompleted();

  const auto range = BufferRange(inputData.size() * sizeof(float), 0);
  auto* data = outputBuffer->map(range, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(data, nullptr);

  std::vector<float> outputData(inputData.size());
  std::memcpy(outputData.data(), data, inputData.size() * sizeof(float));
  outputBuffer->unmap();

  for (size_t i = 0; i < inputData.size(); i++) {
    EXPECT_FLOAT_EQ(outputData[i], inputData[i] * 2.0f);
  }
}

//
// ComputePipelineReflection
//
// Test that computePipelineReflection returns nullptr (not implemented for Vulkan).
//
TEST_F(ComputePipelineStateVulkanTest, ComputePipelineReflection) {
  Result ret;

  const std::string computeSource(data::shader::kVulkanSimpleComputeShader);
  auto computeModule =
      ShaderModuleCreator::fromStringInput(*device_,
                                           computeSource.c_str(),
                                           {.stage = ShaderStage::Compute, .entryPoint = "main"},
                                           "TestReflection",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto stages = ShaderStagesCreator::fromComputeModule(*device_, computeModule, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc pipelineDesc{.shaderStages = std::move(stages),
                                         .debugName = "TestReflection"};

  auto pipeline = device_->createComputePipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);

  // Vulkan implementation returns nullptr for pipeline reflection
  auto reflection = pipeline->computePipelineReflection();
  EXPECT_EQ(reflection, nullptr);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX
