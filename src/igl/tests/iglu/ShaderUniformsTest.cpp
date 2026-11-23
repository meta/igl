/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IGLU/simple_renderer/ShaderUniforms.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"
#include "RecordingDevice.h"

#include <igl/CommandBuffer.h>
#include <igl/Framebuffer.h>
#include <igl/NameHandle.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/Texture.h>
#include <igl/VertexInputState.h>

namespace igl::tests {

namespace {
class TestRenderPipelineReflection final : public IRenderPipelineReflection {
 public:
  [[nodiscard]] const std::vector<BufferArgDesc>& allUniformBuffers() const override {
    return bufferArguments_;
  }
  [[nodiscard]] const std::vector<SamplerArgDesc>& allSamplers() const override {
    return samplerArguments_;
  }
  [[nodiscard]] const std::vector<TextureArgDesc>& allTextures() const override {
    return textureArguments_;
  }

  TestRenderPipelineReflection(std::vector<BufferArgDesc> bufferArguments,
                               std::vector<SamplerArgDesc> samplerArguments,
                               std::vector<TextureArgDesc> textureArguments) :
    bufferArguments_(std::move(bufferArguments)),
    samplerArguments_(std::move(samplerArguments)),
    textureArguments_(std::move(textureArguments)) {}

 private:
  std::vector<BufferArgDesc> bufferArguments_;
  std::vector<SamplerArgDesc> samplerArguments_;
  std::vector<TextureArgDesc> textureArguments_;
};
} // namespace

class ShaderUniformsTest : public ::testing::Test {
 public:
  ShaderUniformsTest() = default;

  // Set up common resources. This will create a device and a command queue
  void SetUp() override {
    // Turn off debug break so unit tests can run
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

TEST_F(ShaderUniformsTest, SettersCoverage) {
  TestRenderPipelineReflection reflection{
      std::vector<BufferArgDesc>{},
      std::vector<SamplerArgDesc>{},
      std::vector<TextureArgDesc>{},
  };
  iglu::material::ShaderUniforms shaderUniforms(*iglDev_, reflection);

  NameHandle testName;
  bool boolValue = true;
  shaderUniforms.setBool(testName, boolValue);
  shaderUniforms.setBool(testName, testName, testName, boolValue);
  shaderUniforms.setBoolArray(testName, &boolValue, 1);
  shaderUniforms.setBoolArray(testName, testName, testName, &boolValue, 1);

  iglu::simdtypes::float1 floatValue = 1.0f;
  shaderUniforms.setFloat(testName, floatValue);
  shaderUniforms.setFloat(testName, testName, testName, floatValue);
  shaderUniforms.setFloatArray(testName, &floatValue, 1);
  shaderUniforms.setFloatArray(testName, testName, testName, &floatValue, 1);

  iglu::simdtypes::float2 float2Value = {1.0f, 2.0f};
  shaderUniforms.setFloat2(testName, float2Value);
  shaderUniforms.setFloat2(testName, testName, testName, float2Value);
  shaderUniforms.setFloat2Array(testName, &float2Value, 1);
  shaderUniforms.setFloat2Array(testName, testName, testName, &float2Value, 1);

  iglu::simdtypes::float3 float3Value = {1.0f, 2.0f, 3.0f};
  shaderUniforms.setFloat3(testName, float3Value);
  shaderUniforms.setFloat3Array(testName, &float3Value, 1);

  iglu::simdtypes::float4 float4Value = {1.0f, 2.0f, 3.0f, 4.0f};
  shaderUniforms.setFloat4(testName, float4Value);
  shaderUniforms.setFloat4(testName, testName, testName, float4Value);
  shaderUniforms.setFloat4Array(testName, &float4Value, 1);
  shaderUniforms.setFloat4Array(testName, testName, testName, &float4Value, 1);

  iglu::simdtypes::int1 intValue = 1;
  shaderUniforms.setInt(testName, intValue);
  shaderUniforms.setInt(testName, testName, testName, intValue);
  shaderUniforms.setIntArray(testName, &intValue, 1);
  shaderUniforms.setIntArray(testName, testName, testName, &intValue, 1);

  iglu::simdtypes::int2 int2Value = {1, 2};
  shaderUniforms.setInt2(testName, int2Value);
  shaderUniforms.setInt2(testName, testName, testName, int2Value);

  iglu::simdtypes::float2x2 float2x2Value = {float2Value, float2Value};
  shaderUniforms.setFloat2x2(testName, float2x2Value);
  shaderUniforms.setFloat2x2(testName, testName, testName, float2x2Value);
  shaderUniforms.setFloat2x2Array(testName, &float2x2Value, 1);
  shaderUniforms.setFloat2x2Array(testName, testName, testName, &float2x2Value, 1);

  iglu::simdtypes::float3x3 float3x3Value = {float3Value, float3Value, float3Value};
  shaderUniforms.setFloat3x3(testName, float3x3Value);
  shaderUniforms.setFloat3x3(testName, testName, testName, float3x3Value);
  shaderUniforms.setFloat3x3Array(testName, &float3x3Value, 1);
  shaderUniforms.setFloat3x3Array(testName, testName, testName, &float3x3Value, 1);

  iglu::simdtypes::float4x4 float4x4Value = {float4Value, float4Value, float4Value, float4Value};
  shaderUniforms.setFloat4x4(testName, float4x4Value);
  shaderUniforms.setFloat4x4(testName, testName, testName, float4x4Value);
  shaderUniforms.setFloat4x4Array(testName, &float4x4Value, 1);
  shaderUniforms.setFloat4x4Array(testName, testName, testName, &float4x4Value, 1);

  std::shared_ptr<ITexture> texture;
  std::shared_ptr<ISamplerState> sampler;
  shaderUniforms.setTexture("test", texture, sampler);
  shaderUniforms.setTexture("test", nullptr, sampler);
  shaderUniforms.setTexture("test", nullptr, nullptr);
}

// Test construction with vertex-stage uniform buffer
TEST_F(ShaderUniformsTest, ConstructWithVertexStageBuffer) {
  BufferArgDesc vertexBuffer{
      .name = genNameHandle("vertexUniforms"),
      .bufferDataSize = 64,
      .bufferIndex = 0,
      .shaderStage = ShaderStage::Vertex,
      .isUniformBlock = true,
      .members = {{.name = genNameHandle("modelMatrix"),
                   .type = UniformType::Mat4x4,
                   .offset = 0,
                   .arrayLength = 1}},
  };

  TestRenderPipelineReflection reflection{
      std::vector<BufferArgDesc>{vertexBuffer},
      std::vector<SamplerArgDesc>{},
      std::vector<TextureArgDesc>{},
  };

  iglu::material::ShaderUniforms shaderUniforms(*iglDev_, reflection);

  // Verify uniform was registered and is accessible
  const iglu::simdtypes::float4 col = {1.0f, 0.0f, 0.0f, 0.0f};
  const iglu::simdtypes::float4x4 identity(col, col, col, col);
  shaderUniforms.setFloat4x4(genNameHandle("modelMatrix"), identity);
}

// Test construction with fragment-stage uniform buffer
TEST_F(ShaderUniformsTest, ConstructWithFragmentStageBuffer) {
  BufferArgDesc fragmentBuffer{
      .name = genNameHandle("fragmentUniforms"),
      .bufferDataSize = 16,
      .bufferIndex = 1,
      .shaderStage = ShaderStage::Fragment,
      .isUniformBlock = true,
      .members = {{.name = genNameHandle("color"),
                   .type = UniformType::Float4,
                   .offset = 0,
                   .arrayLength = 1}},
  };

  TestRenderPipelineReflection reflection{
      std::vector<BufferArgDesc>{fragmentBuffer},
      std::vector<SamplerArgDesc>{},
      std::vector<TextureArgDesc>{},
  };

  iglu::material::ShaderUniforms shaderUniforms(*iglDev_, reflection);

  // Verify uniform was registered and is accessible
  const iglu::simdtypes::float4 redColor = {1.0f, 0.0f, 0.0f, 1.0f};
  shaderUniforms.setFloat4(genNameHandle("color"), redColor);
}

// Test construction with both vertex and fragment stage buffers
TEST_F(ShaderUniformsTest, ConstructWithMixedStageBuffers) {
  BufferArgDesc vertexBuffer{
      .name = genNameHandle("vertexUniforms"),
      .bufferDataSize = 64,
      .bufferIndex = 0,
      .shaderStage = ShaderStage::Vertex,
      .isUniformBlock = true,
      .members = {{.name = genNameHandle("mvpMatrix"),
                   .type = UniformType::Mat4x4,
                   .offset = 0,
                   .arrayLength = 1}},
  };

  BufferArgDesc fragmentBuffer{
      .name = genNameHandle("fragmentUniforms"),
      .bufferDataSize = 20,
      .bufferIndex = 1,
      .shaderStage = ShaderStage::Fragment,
      .isUniformBlock = true,
      .members = {{.name = genNameHandle("baseColor"),
                   .type = UniformType::Float4,
                   .offset = 0,
                   .arrayLength = 1},
                  {.name = genNameHandle("roughness"),
                   .type = UniformType::Float,
                   .offset = 16,
                   .arrayLength = 1}},
  };

  TestRenderPipelineReflection reflection{
      std::vector<BufferArgDesc>{vertexBuffer, fragmentBuffer},
      std::vector<SamplerArgDesc>{},
      std::vector<TextureArgDesc>{},
  };

  iglu::material::ShaderUniforms shaderUniforms(*iglDev_, reflection);

  // Verify both stages' uniforms are accessible
  const iglu::simdtypes::float4 col = {1.0f, 0.0f, 0.0f, 0.0f};
  const iglu::simdtypes::float4x4 matrix(col, col, col, col);
  shaderUniforms.setFloat4x4(genNameHandle("mvpMatrix"), matrix);

  const iglu::simdtypes::float4 color = {0.5f, 0.5f, 0.5f, 1.0f};
  shaderUniforms.setFloat4(genNameHandle("baseColor"), color);

  const iglu::simdtypes::float1 roughness = 0.8f;
  shaderUniforms.setFloat(genNameHandle("roughness"), roughness);
}

// Test construction with multiple members in a single buffer
TEST_F(ShaderUniformsTest, ConstructWithMultiMemberBuffer) {
  BufferArgDesc buffer{
      .name = genNameHandle("perFrameUniforms"),
      .bufferDataSize = 128,
      .bufferIndex = 0,
      .shaderStage = ShaderStage::Vertex,
      .isUniformBlock = true,
      .members = {{.name = genNameHandle("modelMatrix"),
                   .type = UniformType::Mat4x4,
                   .offset = 0,
                   .arrayLength = 1},
                  {.name = genNameHandle("viewMatrix"),
                   .type = UniformType::Mat4x4,
                   .offset = 64,
                   .arrayLength = 1}},
  };

  TestRenderPipelineReflection reflection{
      std::vector<BufferArgDesc>{buffer},
      std::vector<SamplerArgDesc>{},
      std::vector<TextureArgDesc>{},
  };

  iglu::material::ShaderUniforms shaderUniforms(*iglDev_, reflection);

  // Both members should be settable
  const iglu::simdtypes::float4 col = {1.0f, 0.0f, 0.0f, 0.0f};
  const iglu::simdtypes::float4x4 identity(col, col, col, col);
  shaderUniforms.setFloat4x4(genNameHandle("modelMatrix"), identity);
  shaderUniforms.setFloat4x4(genNameHandle("viewMatrix"), identity);
}

// Drives a real uniform-buffer upload on Vulkan through bind(). setFloat4() copies the
// uniform bytes into the malloc'd allocation (the setUniformBytes() memcpy path) and
// bind() uploads them into the IGL uniform buffer via the
// static_cast<uint8_t*>(allocation->ptr) + suballocation-offset source pointer in
// bindBuffer(). Suballocation slot 1 is selected so a non-zero offset is exercised.
TEST_F(ShaderUniformsTest, BindUploadsSuballocatedUniformBuffer) {
  if (iglDev_->getBackendType() != BackendType::Vulkan) {
    GTEST_SKIP() << "Suballocated uniform-buffer upload path is Vulkan-only";
  }

  const NameHandle blockName = genNameHandle("FragmentUniforms");
  const NameHandle memberName = genNameHandle("color");
  const BufferArgDesc fragmentBuffer{
      .name = blockName,
      .bufferDataSize = 16,
      .bufferIndex = 0,
      .shaderStage = ShaderStage::Fragment,
      .isUniformBlock = true,
      .members = {{.name = memberName, .type = UniformType::Float4, .offset = 0, .arrayLength = 1}},
  };
  TestRenderPipelineReflection reflection{
      std::vector<BufferArgDesc>{fragmentBuffer},
      std::vector<SamplerArgDesc>{},
      std::vector<TextureArgDesc>{},
  };
  iglu::material::ShaderUniforms shaderUniforms(*iglDev_, reflection);
  ASSERT_TRUE(shaderUniforms.containsUniform(memberName))
      << "reflection member should be registered as a uniform";

  // Selecting suballocation slot 1 forces a non-zero suballocation offset in both the
  // memcpy destination and the upload source pointer.
  const Result subResult = shaderUniforms.setSuballocationIndex(memberName, 1);
  ASSERT_TRUE(subResult.isOk()) << subResult.message;

  const iglu::simdtypes::float4 color = {0.25f, 0.5f, 0.75f, 1.0f};
  shaderUniforms.setFloat4(memberName, color);

  // Minimal offscreen render target so a real render command encoder can be opened.
  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 4;
  constexpr float kClear = 0.501f;
  constexpr uint32_t kClearHex = 0x80808080;

  Result ret;
  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 kWidth,
                                                 kHeight,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Attachment);
  std::shared_ptr<ITexture> offscreen = iglDev_->createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_TRUE(offscreen != nullptr);

  const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreen}}};
  std::shared_ptr<IFramebuffer> framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_TRUE(framebuffer != nullptr);

  RenderPassDesc renderPass;
  renderPass = {
      .colorAttachments = {{.loadAction = LoadAction::Clear,
                            .storeAction = StoreAction::Store,
                            .clearColor = {kClear, kClear, kClear, kClear}}},
  };

  std::unique_ptr<IShaderStages> stages;
  util::createSimpleShaderStages(iglDev_, stages);
  std::shared_ptr<IShaderStages> shaderStages = std::move(stages);

  // numAttributes has to equal numInputBindings when using more than one buffer
  const VertexInputStateDesc inputDesc{
      .numAttributes = 2,
      .attributes = {{.bufferIndex = data::shader::kSimplePosIndex,
                      .format = VertexAttributeFormat::Float4,
                      .offset = 0,
                      .name = std::string(data::shader::kSimplePos),
                      .location = 0},
                     {.bufferIndex = data::shader::kSimpleUvIndex,
                      .format = VertexAttributeFormat::Float2,
                      .offset = 0,
                      .name = std::string(data::shader::kSimpleUv),
                      .location = 1}},
      .numInputBindings = 2,
      .inputBindings = {{.stride = sizeof(float) * 4}, {.stride = sizeof(float) * 2}},
  };
  std::shared_ptr<IVertexInputState> vertexInputState =
      iglDev_->createVertexInputState(inputDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_TRUE(vertexInputState != nullptr);

  const size_t textureUnit = 0;
  const RenderPipelineDesc pipelineDesc = {
      .vertexInputState = vertexInputState,
      .shaderStages = shaderStages,
      .targetDesc = {.colorAttachments = {{.textureFormat = offscreen->getFormat()}}},
      .cullMode = CullMode::Disabled,
      .fragmentUnitSamplerMap = {{textureUnit, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
  };
  std::shared_ptr<IRenderPipelineState> pipelineState =
      iglDev_->createRenderPipeline(pipelineDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_TRUE(pipelineState != nullptr);

  auto cmdBuffer = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_TRUE(cmdBuffer != nullptr);

  auto encoder = cmdBuffer->createRenderCommandEncoder(renderPass, framebuffer);
  ASSERT_TRUE(encoder != nullptr);

  // Exercises ShaderUniforms::bindBuffer() -> iglBuffer->upload() with the
  // static_cast<uint8_t*>(allocation->ptr) + subAllocatedOffset source pointer.
  shaderUniforms.bind(*iglDev_, *pipelineState, *encoder);
  encoder->endEncoding();

  cmdQueue_->submit(*cmdBuffer);
  cmdBuffer->waitUntilCompleted();

  // The render pass that contained the uniform-buffer upload completed on the GPU, so the
  // offscreen target holds the clear color.
  std::vector<uint32_t> pixels(kWidth * kHeight);
  framebuffer->copyBytesColorAttachment(
      *cmdQueue_, 0, pixels.data(), TextureRangeDesc::new2D(0, 0, kWidth, kHeight));
  for (const uint32_t pixel : pixels) {
    EXPECT_EQ(pixel, kClearHex) << "offscreen target should hold the clear color after bind()";
  }
}

namespace {

TestRenderPipelineReflection colorBlockReflection() {
  const BufferArgDesc fragmentBuffer{
      .name = genNameHandle("FragmentUniforms"),
      .bufferDataSize = 16,
      .bufferIndex = 0,
      .shaderStage = ShaderStage::Fragment,
      .isUniformBlock = true,
      .members = {{.name = genNameHandle("color"),
                   .type = UniformType::Float4,
                   .offset = 0,
                   .arrayLength = 1}},
  };
  return TestRenderPipelineReflection{
      std::vector<BufferArgDesc>{fragmentBuffer},
      std::vector<SamplerArgDesc>{},
      std::vector<TextureArgDesc>{},
  };
}
} // namespace

// UploadMode::CommandBuffer writes buffers in command buffers, so they need neither CPU access nor
// a Ring. Without DeviceFeatures::UpdateBuffer the mode falls back to the AtBind buffers.
TEST_F(ShaderUniformsTest, CommandBufferModeCreatesPrivateBuffersWithoutRing) {
  const TestRenderPipelineReflection reflection = colorBlockReflection();
  RecordingDevice device(*iglDev_);
  device.backendType = BackendType::Vulkan;

  device.reportsUpdateBuffer = true;
  {
    const iglu::material::ShaderUniforms shaderUniforms(
        device, reflection, false, iglu::material::ShaderUniforms::UploadMode::CommandBuffer);
    EXPECT_EQ(device.createBufferCount, 1u);
    EXPECT_EQ(device.recordedHint & BufferDesc::BufferAPIHintBits::Ring, 0);
    EXPECT_EQ(device.recordedStorage, ResourceStorage::Private);
  }

  device.reportsUpdateBuffer = false;
  {
    const iglu::material::ShaderUniforms shaderUniforms(
        device, reflection, false, iglu::material::ShaderUniforms::UploadMode::CommandBuffer);
    EXPECT_EQ(device.createBufferCount, 2u);
    EXPECT_NE(device.recordedHint & BufferDesc::BufferAPIHintBits::Ring, 0);
    EXPECT_EQ(device.recordedStorage, ResourceStorage::Shared);
  }
}

// update() records the uniform values into the command buffer, so commands recorded after it see
// them.
TEST_F(ShaderUniformsTest, CommandBufferModeUploadsInUpdate) {
  if (!iglDev_->hasFeature(DeviceFeatures::UpdateBuffer)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::UpdateBuffer";
  }
  const TestRenderPipelineReflection reflection = colorBlockReflection();
  RecordingDevice device(*iglDev_);
  device.backendType = iglDev_->getBackendType();
  device.reportsUpdateBuffer = true;
  iglu::material::ShaderUniforms shaderUniforms(
      device, reflection, false, iglu::material::ShaderUniforms::UploadMode::CommandBuffer);
  ASSERT_NE(device.createdBuffer, nullptr);
  IBuffer& uniformBuffer = *device.createdBuffer;

  const iglu::simdtypes::float4 color = {0.25f, 0.5f, 0.75f, 1.0f};
  shaderUniforms.setFloat4(genNameHandle("color"), color);

  Result ret;
  const BufferDesc readbackDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = sizeof(color),
      .storage = ResourceStorage::Shared,
  };
  auto readback = iglDev_->createBuffer(readbackDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  auto cmdBuffer = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  shaderUniforms.update(*cmdBuffer);
  cmdBuffer->copyBuffer(uniformBuffer, *readback, 0, 0, sizeof(color));
  cmdQueue_->submit(*cmdBuffer);
  cmdBuffer->waitUntilCompleted();

  const auto* values =
      static_cast<const float*>(readback->map(BufferRange(sizeof(color), 0), &ret));
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(values[0], 0.25f);
  EXPECT_EQ(values[1], 0.5f);
  EXPECT_EQ(values[2], 0.75f);
  EXPECT_EQ(values[3], 1.0f);
  readback->unmap();
}

// update() writes every buffer, not only the ones changed since the previous call, so the GPU copy
// is restored even when the command buffer of the previous update() never reached the GPU.
TEST_F(ShaderUniformsTest, CommandBufferModeUpdateRewritesUnchangedBuffers) {
  if (!iglDev_->hasFeature(DeviceFeatures::UpdateBuffer) ||
      !iglDev_->hasFeature(DeviceFeatures::FillBuffer)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::UpdateBuffer and FillBuffer";
  }
  const TestRenderPipelineReflection reflection = colorBlockReflection();
  RecordingDevice device(*iglDev_);
  device.backendType = iglDev_->getBackendType();
  device.reportsUpdateBuffer = true;
  iglu::material::ShaderUniforms shaderUniforms(
      device, reflection, false, iglu::material::ShaderUniforms::UploadMode::CommandBuffer);
  ASSERT_NE(device.createdBuffer, nullptr);
  IBuffer& uniformBuffer = *device.createdBuffer;

  const iglu::simdtypes::float4 color = {0.25f, 0.5f, 0.75f, 1.0f};
  shaderUniforms.setFloat4(genNameHandle("color"), color);

  Result ret;
  auto first = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  shaderUniforms.update(*first);
  // Stands in for a lost write: the GPU copy no longer matches the CPU copy.
  first->fillBuffer(uniformBuffer, BufferRange(sizeof(color), 0), 0);
  cmdQueue_->submit(*first);
  first->waitUntilCompleted();

  const BufferDesc readbackDesc{
      .type = BufferDesc::BufferTypeBits::Storage,
      .length = sizeof(color),
      .storage = ResourceStorage::Shared,
  };
  auto readback = iglDev_->createBuffer(readbackDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  auto second = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  shaderUniforms.update(*second);
  second->copyBuffer(uniformBuffer, *readback, 0, 0, sizeof(color));
  cmdQueue_->submit(*second);
  second->waitUntilCompleted();

  const auto* values =
      static_cast<const float*>(readback->map(BufferRange(sizeof(color), 0), &ret));
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(values[0], 0.25f);
  EXPECT_EQ(values[1], 0.5f);
  EXPECT_EQ(values[2], 0.75f);
  EXPECT_EQ(values[3], 1.0f);
  readback->unmap();
}

} // namespace igl::tests
