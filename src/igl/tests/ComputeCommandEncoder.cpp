/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/ComputeCommandEncoder.h>

#include "data/ShaderData.h"
#include "util/Common.h"

#include <array>
#include <cstring>
#include <memory>
#include <vector>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/ComputePipelineState.h>
#include <igl/Shader.h>
#include <igl/ShaderCreator.h>

namespace igl::tests {

namespace {
constexpr std::array<float, 6> kDataIn = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
} // namespace

/**
 * @brief ComputeCommandEncoderTest is a test fixture for all the tests in this file.
 * It takes care of common initialization and allocating of common resources.
 */
class ComputeCommandEncoderTest : public ::testing::Test {
 private:
 public:
  ComputeCommandEncoderTest() = default;

  /**
   * @brief This function sets up compute buffers and compiles the compute shader.
   */
  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
      return;
    }

    const BufferDesc vbInDesc{
        .type = BufferDesc::BufferTypeBits::Storage,
        .data = kDataIn.data(),
        .length = sizeof(float) * kDataIn.size(),
    };
    bufferIn_ = iglDev_->createBuffer(vbInDesc, nullptr);
    ASSERT_TRUE(bufferIn_ != nullptr);
    // Use ResourceStorage::Shared for output buffers so they can be mapped for reading
    const BufferDesc bufferOutDesc{
        .type = BufferDesc::BufferTypeBits::Storage,
        .length = sizeof(float) * kDataIn.size(),
        .storage = ResourceStorage::Shared,
    };
    bufferOut0_ = iglDev_->createBuffer(bufferOutDesc, nullptr);
    ASSERT_TRUE(bufferOut0_ != nullptr);
    bufferOut1_ = iglDev_->createBuffer(bufferOutDesc, nullptr);
    ASSERT_TRUE(bufferOut1_ != nullptr);
    bufferOut2_ = iglDev_->createBuffer(bufferOutDesc, nullptr);
    ASSERT_TRUE(bufferOut2_ != nullptr);

    { // Compile CS
      std::string_view entryName;
      std::string_view source;
      if (iglDev_->getBackendType() == BackendType::OpenGL) {
        source = data::shader::kOglSimpleComputeShader;
        entryName = data::shader::kSimpleComputeFunc;
      } else if (iglDev_->getBackendType() == BackendType::Vulkan) {
        source = data::shader::kVulkanSimpleComputeShader;
        entryName = "main"; // entry point is not pipelined. Hardcoding to main for now.
      } else if (iglDev_->getBackendType() == BackendType::Metal) {
        source = data::shader::kMtlSimpleComputeShader;
        entryName = data::shader::kSimpleComputeFunc;
      } else if (iglDev_->getBackendType() == BackendType::D3D12) {
        source = data::shader::kD3D12SimpleComputeShader;
        entryName = data::shader::kSimpleComputeFunc;
      } else if (iglDev_->getBackendType() == BackendType::WebGPU) {
        source = data::shader::kWgslSimpleComputeShader;
        entryName = "main";
      } else {
        IGL_DEBUG_ASSERT_NOT_REACHED();
      }

      Result ret;
      computeStages_ = ShaderStagesCreator::fromModuleStringInput(
          *iglDev_, std::string(source).c_str(), std::string(entryName), "", &ret);
      ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
      ASSERT_TRUE(computeStages_ != nullptr);
    }
  }

  /**
   * @brief This function binds bufferIn and bufferOut to a new computePipelineState and encodes the
   * computePipelineState to a new computeCommandEncoder.
   */
  void encodeCompute(const std::shared_ptr<ICommandBuffer>& cmdBuffer,
                     const std::shared_ptr<IBuffer>& bufferIn,
                     const std::shared_ptr<IBuffer>& bufferOut,
                     std::shared_ptr<IComputePipelineState>& ret) {
    ASSERT_TRUE(computeStages_ != nullptr);
    const ComputePipelineDesc computeDesc{
        .buffersMap = {{data::shader::kSimpleComputeInputIndex,
                        IGL_NAMEHANDLE(data::shader::kSimpleComputeInput)},
                       {data::shader::kSimpleComputeOutputIndex,
                        IGL_NAMEHANDLE(data::shader::kSimpleComputeOutput)}},
        .shaderStages = computeStages_,
    };
    auto computePipelineState = iglDev_->createComputePipeline(computeDesc, nullptr);
    ASSERT_TRUE(computePipelineState != nullptr);

    auto computeEncoder = cmdBuffer->createComputeCommandEncoder();
    ASSERT_TRUE(computeEncoder != nullptr);

    computeEncoder->insertDebugEventLabel("Running ComputeCommandEncoderTest...");

    computeEncoder->bindComputePipelineState(computePipelineState);
    computeEncoder->bindBuffer(data::shader::kSimpleComputeInputIndex, bufferIn.get());
    computeEncoder->bindBuffer(data::shader::kSimpleComputeOutputIndex, bufferOut.get());

    const Dimensions threadgroupSize(kDataIn.size(), 1, 1);
    const Dimensions threadgroupCount(1, 1, 1);
    computeEncoder->dispatchThreadGroups(
        threadgroupCount, threadgroupSize, {.buffers = {bufferIn.get()}});
    computeEncoder->endEncoding();
    ret = computePipelineState;
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;

  std::shared_ptr<IShaderStages> computeStages_;

  std::shared_ptr<IBuffer> bufferIn_, bufferOut0_, bufferOut1_, bufferOut2_;
  std::shared_ptr<IComputePipelineState> cps1_, cps2_, cps3_;
};

TEST_F(ComputeCommandEncoderTest, canEncodeBasicBufferOperation) {
#if IGL_PLATFORM_LINUX && !IGL_PLATFORM_LINUX_USE_EGL
  GTEST_SKIP() << "Fix this test on Linux";
#endif
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    return;
  }

  ASSERT_TRUE(cmdQueue_ != nullptr);

  const CommandBufferDesc cbDesc;
  auto cmdBuffer = cmdQueue_->createCommandBuffer(cbDesc, nullptr);
  ASSERT_TRUE(cmdBuffer != nullptr);

  encodeCompute(cmdBuffer, bufferIn_, bufferOut0_, cps1_);

  ASSERT_TRUE(cmdQueue_ != nullptr);
  cmdQueue_->submit(*cmdBuffer);

  cmdBuffer->waitUntilCompleted();

  std::vector<float> bytes(kDataIn.size());
  auto range = BufferRange(sizeof(float) * kDataIn.size(), 0);
  Result ret;
  auto* data = bufferOut0_->map(range, &ret);
  ASSERT_TRUE(data != nullptr);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  std::memcpy(bytes.data(), data, sizeof(float) * kDataIn.size());
  ASSERT_FALSE(kDataIn.empty());
  for (size_t i = 0; i < kDataIn.size(); i++) {
    ASSERT_EQ(kDataIn[i] * 2.0f, bytes[i]);
  }
  bufferOut0_->unmap();
}

TEST_F(ComputeCommandEncoderTest, bindImageTexture) {
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    return;
  }

  auto cmdBuffer = cmdQueue_->createCommandBuffer({}, nullptr);
  ASSERT_TRUE(cmdBuffer != nullptr);

  auto computeCommandEncoder = cmdBuffer->createComputeCommandEncoder();
  computeCommandEncoder->bindImageTexture(0, nullptr, TextureFormat::Invalid);
  computeCommandEncoder->endEncoding();
  cmdQueue_->submit(*cmdBuffer);
  cmdBuffer->waitUntilCompleted();
}

TEST_F(ComputeCommandEncoderTest, canUseOutputBufferFromOnePassAsInputToNext) {
#if IGL_PLATFORM_LINUX && !IGL_PLATFORM_LINUX_USE_EGL
  GTEST_SKIP() << "Fix this test on Linux";
#endif
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    return;
  }

  ASSERT_TRUE(cmdQueue_ != nullptr);

  {
    const CommandBufferDesc cbDesc;
    auto cmdBuffer = cmdQueue_->createCommandBuffer(cbDesc, nullptr);
    ASSERT_TRUE(cmdBuffer != nullptr);

    encodeCompute(cmdBuffer, bufferIn_, bufferOut0_, cps1_);

    cmdQueue_->submit(*cmdBuffer);
    cmdBuffer->waitUntilCompleted();
  }
  {
    const CommandBufferDesc cbDesc;
    auto cmdBuffer = cmdQueue_->createCommandBuffer(cbDesc, nullptr);
    ASSERT_TRUE(cmdBuffer != nullptr);

    encodeCompute(cmdBuffer, bufferOut0_, bufferOut1_, cps2_);
    cmdQueue_->submit(*cmdBuffer);
    cmdBuffer->waitUntilCompleted();
  }
  {
    const CommandBufferDesc cbDesc;
    auto cmdBuffer = cmdQueue_->createCommandBuffer(cbDesc, nullptr);
    ASSERT_TRUE(cmdBuffer != nullptr);

    encodeCompute(cmdBuffer, bufferOut1_, bufferOut2_, cps3_);
    cmdQueue_->submit(*cmdBuffer);
    cmdBuffer->waitUntilCompleted();
  }

  std::vector<float> bytes(kDataIn.size());
  auto range = BufferRange(sizeof(float) * kDataIn.size(), 0);
  Result ret;
  auto* data = bufferOut2_->map(range, &ret);
  ASSERT_TRUE(data != nullptr);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  std::memcpy(bytes.data(), data, sizeof(float) * kDataIn.size());
  ASSERT_FALSE(kDataIn.empty());
  for (size_t i = 0; i < kDataIn.size(); i++) {
    // Compute pass ran 3 times
    ASSERT_EQ(kDataIn[i] * 2.0f * 2.0f * 2.0f, bytes[i]);
  }
  bufferOut2_->unmap();
}

TEST_F(ComputeCommandEncoderTest, bindSamplerState) {
  if (!iglDev_->hasFeature(DeviceFeatures::Compute)) {
    return;
  }

  auto cmdBuffer = cmdQueue_->createCommandBuffer({}, nullptr);
  ASSERT_TRUE(cmdBuffer != nullptr);

  auto computeCommandEncoder = cmdBuffer->createComputeCommandEncoder();
  computeCommandEncoder->bindSamplerState(0, nullptr);
  computeCommandEncoder->endEncoding();
  cmdQueue_->submit(*cmdBuffer);
  cmdBuffer->waitUntilCompleted();
}

TEST_F(ComputeCommandEncoderTest, copyBuffer) {
  if (!iglDev_->hasFeature(DeviceFeatures::CopyBuffer)) {
    return;
  }

  ASSERT_TRUE(cmdQueue_ != nullptr);

  std::vector<uint8_t> dataIn2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 10};

  auto bufferSrc = iglDev_->createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Storage,
          .data = dataIn2.data(),
          .length = dataIn2.size(),
          .storage = ResourceStorage::Private,
          .debugName = "bufferSrc",
      },
      nullptr);

  auto bufferDst = iglDev_->createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Storage,
          .length = dataIn2.size(),
          .storage = ResourceStorage::Shared,
          .debugName = "bufferDst",
      },
      nullptr);

  {
    auto cmdBuffer = cmdQueue_->createCommandBuffer({}, nullptr);
    ASSERT_TRUE(cmdBuffer != nullptr);

    cmdBuffer->copyBuffer(*bufferSrc, *bufferDst, 0, 0, dataIn2.size());

    cmdQueue_->submit(*cmdBuffer);
    cmdBuffer->waitUntilCompleted();
  }

  Result ret;
  const uint8_t* dataOut =
      static_cast<const uint8_t*>(bufferDst->map(BufferRange(dataIn2.size(), 0), &ret));
  ASSERT_TRUE(dataOut != nullptr);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  for (size_t i = 0; i < dataIn2.size(); i++) {
    ASSERT_EQ(dataIn2[i], dataOut[i]);
  }
  bufferDst->unmap();
}

} // namespace igl::tests
