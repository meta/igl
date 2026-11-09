/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../util/TestDevice.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/ComputeCommandEncoder.h>
#include <igl/ComputePipelineState.h>
#include <igl/Device.h>
#include <igl/NameHandle.h>
#include <igl/SamplerState.h>
#include <igl/Shader.h>
#include <igl/ShaderCreator.h>
#include <igl/Texture.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanImage.h>
#include <igl/vulkan/VulkanTexture.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

class ComputeCommandEncoderVulkanTest : public ::testing::Test {
 public:
  ComputeCommandEncoderVulkanTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";

    Result ret;
    cmdQueue_ = iglDev_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

TEST_F(ComputeCommandEncoderVulkanTest, CreateAndEndEncoding) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto encoder = cmdBuf->createComputeCommandEncoder();
  ASSERT_NE(encoder, nullptr);

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(ComputeCommandEncoderVulkanTest, BindPushConstants) {
  // NOTE: bindPushConstants requires a compute pipeline to be bound first.
  // Since creating a compute pipeline requires a compute shader, and this is
  // just testing the encoder API, we skip this test. The push constants
  // functionality is tested through integration tests that have full pipelines.
  GTEST_SKIP() << "bindPushConstants requires a bound compute pipeline with a shader";
}

// bindTexture() on a Sampled+Storage texture must transition to
// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (matching the sampled descriptor write hardcoded in
// VulkanContext::updateBindingsTextures()), not VK_IMAGE_LAYOUT_GENERAL. Prior to the fix, the
// storage-first branch in bindTexture() picked GENERAL for Sampled+Storage textures, which
// produced VUID-vkCmdDispatch-None-08114 and VUID-VkDescriptorImageInfo-imageLayout-00344 on
// every dispatch.
//
// The test wires up a real compute pipeline that samples from the Sampled+Storage texture and
// writes the result into a storage buffer, then dispatches. This actually drives
// `binder_.updateBindings()` (which writes the descriptor at SHADER_READ_ONLY_OPTIMAL) and
// `vkCmdDispatch()` — the same code path that fires the validation errors when the layout is
// wrong. The post-bind layout assertion is the deterministic regression signal even when the
// build runs without validation layers (e.g., the fbcode lavapipe configuration), while the
// dispatch + waitUntilCompleted ensures Windows / Android Vulkan validation layers will catch
// any future regression that re-introduces the GENERAL transition.
TEST_F(ComputeCommandEncoderVulkanTest, BindTextureSampledStorageTransitionsToShaderReadOnly) {
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    GTEST_SKIP() << "Compute not supported on this device";
  }

  Result ret;

  // Sampled+Storage input texture — the configuration that previously transitioned to GENERAL.
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Storage);
  auto texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(texture, nullptr);

  const auto* vkTexture = static_cast<igl::vulkan::Texture*>(texture.get());
  const igl::vulkan::VulkanImage& vkImage = vkTexture->getVulkanTexture().image;
  ASSERT_TRUE(vkImage.isSampledImage());
  ASSERT_TRUE(vkImage.isStorageImage());

  // Initialize the texture with deterministic pixel data so the sample isn't reading garbage.
  constexpr std::array<uint32_t, 16> kPixels = {0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu,
                                                0xFF0000FFu};
  ret = texture->upload(texture->getFullRange(0), kPixels.data());
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  // Storage output buffer that the compute shader will write to.
  constexpr std::string_view kOutBufName = "outBuf";
  constexpr size_t kOutBufIndex = 0;
  const BufferDesc outBufDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = sizeof(float) * 4,
      .storage = ResourceStorage::Shared,
  };
  auto outBuffer = iglDev_->createBuffer(outBufDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(outBuffer, nullptr);

  // Sampler for the combined image sampler binding.
  auto sampler = iglDev_->createSamplerState({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(sampler, nullptr);

  // Vulkan compute shader: sample the input texture and write the texel into the SSBO.
  // (set = 0, binding = 0) → combined image sampler (matches kBindPoint_CombinedImageSamplers).
  // (set = 1, binding = 0) → storage buffer (matches kBindPoint_Buffers).
  static constexpr std::string_view kShader =
      "layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
      "layout (set = 0, binding = 0) uniform sampler2D inputTex;\n"
      "layout (std430, binding = 0, set = 1) writeonly buffer outBuf {\n"
      "  vec4 fOut[];\n"
      "};\n"
      "void main() {\n"
      "  fOut[0] = texture(inputTex, vec2(0.5, 0.5));\n"
      "}\n";

  auto computeStages = ShaderStagesCreator::fromModuleStringInput(
      *iglDev_,
      kShader.data(), // NOLINT(bugprone-suspicious-stringview-data-usage)
      "main",
      "",
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(computeStages, nullptr);

  const ComputePipelineDesc computeDesc{.buffersMap = {{kOutBufIndex, IGL_NAMEHANDLE(kOutBufName)}},
                                        .shaderStages = std::move(computeStages)};
  auto pipeline = iglDev_->createComputePipeline(computeDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);

  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdBuf, nullptr);

  auto encoder = cmdBuf->createComputeCommandEncoder();
  ASSERT_NE(encoder, nullptr);

  encoder->bindComputePipelineState(pipeline);
  encoder->bindTexture(0, texture.get());
  encoder->bindSamplerState(0, sampler.get());
  encoder->bindBuffer(kOutBufIndex, outBuffer.get());

  // Deterministic regression signal: bindTexture() must have queued a transition to
  // SHADER_READ_ONLY_OPTIMAL (matching the descriptor write), NOT GENERAL.
  EXPECT_EQ(vkImage.imageLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  EXPECT_NE(vkImage.imageLayout_, VK_IMAGE_LAYOUT_GENERAL);

  // Drive the full descriptor write + vkCmdDispatch path so validation layers (when present)
  // observe the layout/descriptor pairing. Without the fix, this dispatch fires
  // VUID-vkCmdDispatch-None-08114 and VUID-VkDescriptorImageInfo-imageLayout-00344 on
  // platforms with the Vulkan validation layer enabled.
  encoder->dispatchThreadGroups(Dimensions(1, 1, 1), Dimensions(1, 1, 1), {});
  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

TEST_F(ComputeCommandEncoderVulkanTest, DebugGroupLabels) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto encoder = cmdBuf->createComputeCommandEncoder();
  ASSERT_NE(encoder, nullptr);

  encoder->pushDebugGroupLabel("ComputeGroup", Color(0.0f, 1.0f, 0.0f, 1.0f));
  encoder->popDebugGroupLabel();

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
}

TEST_F(ComputeCommandEncoderVulkanTest, DispatchThreadGroupsIndirect) {
  Result ret;

  const std::string computeSource(data::shader::kVulkanSimpleComputeShader);
  auto computeModule =
      ShaderModuleCreator::fromStringInput(*iglDev_,
                                           computeSource.c_str(),
                                           {.stage = ShaderStage::Compute, .entryPoint = "main"},
                                           "TestIndirectDispatch",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto stages = ShaderStagesCreator::fromComputeModule(*iglDev_, computeModule, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc pipelineDesc{
      .buffersMap =
          {
              {data::shader::kSimpleComputeInputIndex,
               IGL_NAMEHANDLE(data::shader::kSimpleComputeInput)},
              {data::shader::kSimpleComputeOutputIndex,
               IGL_NAMEHANDLE(data::shader::kSimpleComputeOutput)},
          },
      .shaderStages = std::move(stages),
      .debugName = "TestIndirectDispatch",
  };

  auto pipeline = iglDev_->createComputePipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);

  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const BufferDesc inputBufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .data = inputData.data(),
      .length = inputData.size() * sizeof(float),
      .storage = ResourceStorage::Shared,
  };
  auto inputBuffer = iglDev_->createBuffer(inputBufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const BufferDesc outputBufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = inputData.size() * sizeof(float),
      .storage = ResourceStorage::Shared,
  };
  auto outputBuffer = iglDev_->createBuffer(outputBufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  struct DispatchIndirectArgs {
    uint32_t groupCountX = 0;
    uint32_t groupCountY = 0;
    uint32_t groupCountZ = 0;
  };
  const DispatchIndirectArgs indirectArgs{.groupCountX = 1, .groupCountY = 1, .groupCountZ = 1};
  const BufferDesc indirectBufferDesc{
      .type = BufferDesc::BufferTypeBits::Indirect,
      .data = &indirectArgs,
      .length = sizeof(indirectArgs),
      .storage = ResourceStorage::Shared,
  };
  auto indirectBuffer = iglDev_->createBuffer(indirectBufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto cmdBuffer = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto computeEncoder = cmdBuffer->createComputeCommandEncoder();
  ASSERT_NE(computeEncoder, nullptr);

  computeEncoder->bindComputePipelineState(pipeline);
  computeEncoder->bindBuffer(data::shader::kSimpleComputeInputIndex, inputBuffer.get());
  computeEncoder->bindBuffer(data::shader::kSimpleComputeOutputIndex, outputBuffer.get());

  Dependencies deps{};
  deps.buffers[0] = indirectBuffer.get();
  computeEncoder->dispatchThreadGroupsIndirect(*indirectBuffer, 0, Dimensions(6, 1, 1), deps);
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

TEST_F(ComputeCommandEncoderVulkanTest, DirectDispatch) {
  Result ret;

  const std::string computeSource(data::shader::kVulkanSimpleComputeShader);
  auto computeModule =
      ShaderModuleCreator::fromStringInput(*iglDev_,
                                           computeSource.c_str(),
                                           {.stage = ShaderStage::Compute, .entryPoint = "main"},
                                           "TestDirectDispatch",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto stages = ShaderStagesCreator::fromComputeModule(*iglDev_, computeModule, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const ComputePipelineDesc pipelineDesc{
      .buffersMap =
          {
              {data::shader::kSimpleComputeInputIndex,
               IGL_NAMEHANDLE(data::shader::kSimpleComputeInput)},
              {data::shader::kSimpleComputeOutputIndex,
               IGL_NAMEHANDLE(data::shader::kSimpleComputeOutput)},
          },
      .shaderStages = std::move(stages),
      .debugName = "TestDirectDispatch",
  };

  auto pipeline = iglDev_->createComputePipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(pipeline, nullptr);

  const std::vector<float> inputData = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  const BufferDesc inputBufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .data = inputData.data(),
      .length = inputData.size() * sizeof(float),
      .storage = ResourceStorage::Shared,
  };
  auto inputBuffer = iglDev_->createBuffer(inputBufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  const BufferDesc outputBufferDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = inputData.size() * sizeof(float),
      .storage = ResourceStorage::Shared,
  };
  auto outputBuffer = iglDev_->createBuffer(outputBufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  auto cmdBuffer = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

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

TEST_F(ComputeCommandEncoderVulkanTest, NestedDebugGroupLabels) {
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto encoder = cmdBuf->createComputeCommandEncoder();
  ASSERT_NE(encoder, nullptr);

  encoder->pushDebugGroupLabel("Outer", Color(1.0f, 0.0f, 0.0f, 1.0f));
  encoder->pushDebugGroupLabel("Inner", Color(0.0f, 1.0f, 0.0f, 1.0f));
  encoder->popDebugGroupLabel();
  encoder->popDebugGroupLabel();

  encoder->endEncoding();
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
