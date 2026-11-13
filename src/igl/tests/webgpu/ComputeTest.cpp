/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <igl/CommandBuffer.h>
#include <igl/CommandQueue.h>
#include <igl/ComputeCommandEncoder.h>
#include <igl/ComputePipelineState.h>
#include <igl/ShaderCreator.h>
#include <igl/tests/util/device/webgpu/TestDevice.h>
#include <igl/webgpu/ComputePipelineState.h>
#include <igl/webgpu/ResourcesBinder.h>
#include <igl/webgpu/Texture.h>

namespace igl::tests {

namespace {

constexpr const char* kScaleShader = R"(
@id(0) override scale : f32 = 1.0;
@id(1) override bias : i32 = 0;
@group(1) @binding(0) var<storage, read> values : array<f32>;
@group(1) @binding(1) var<storage, read_write> results : array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id : vec3u) {
  results[id.x] = values[id.x] * scale + f32(bias);
}
)";

// Copies texture unit 0 into storage texture 0 with the channels reversed.
constexpr const char* kSwizzleShader = R"(
@group(0) @binding(0) var src : texture_2d<f32>;
@group(2) @binding(0) var dst : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  textureStore(dst, vec2i(id.xy), textureLoad(src, vec2i(id.xy), 0).abgr);
}
)";

// Copies texture unit 0 into storage texture 8, and with the channels reversed into storage
// texture 15, the highest storage texture binding.
constexpr const char* kHighStorageBindingsShader = R"(
@group(0) @binding(0) var src : texture_2d<f32>;
@group(2) @binding(8) var dst8 : texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(15) var dst15 : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let texel = textureLoad(src, vec2i(id.xy), 0);
  textureStore(dst8, vec2i(id.xy), texel);
  textureStore(dst15, vec2i(id.xy), texel.abgr);
}
)";

// Push constants live in the group 3 uniform buffer.
constexpr const char* kPushConstantShader = R"(
struct PushConstants {
  scale : f32,
  offsets : array<vec4f, 2>,
};
@group(1) @binding(0) var<storage, read_write> results : array<f32>;
@group(3) @binding(0) var<uniform> pc : PushConstants;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id : vec3u) {
  results[id.x] = pc.scale * f32(id.x) + pc.offsets[1][id.x];
}
)";

constexpr uint32_t kTextureSize = 2;

} // namespace

class WebGPUComputeTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
    auto device = util::device::webgpu::createTestDevice();
    ASSERT_NE(device, nullptr);
    webgpuDevice_ = device.get();
    device_ = std::move(device);
    Result ret;
    queue_ = device_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
  }

  void TearDown() override {
    EXPECT_EQ(webgpuDevice_->getContext().getUncapturedErrorCount(), 0u);
  }

 protected:
  [[nodiscard]] std::shared_ptr<IComputePipelineState> createPipeline(
      const char* source,
      const FunctionConstantValues& constants = {}) {
    Result ret;
    auto module =
        device_->createShaderModule({.info = {.stage = ShaderStage::Compute,
                                              .entryPoint = "main",
                                              .functionConstantValues = constants},
                                     .input = {.source = source, .type = ShaderInputType::String}},
                                    &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    auto stages = device_->createShaderStages(ShaderStagesDesc::fromComputeModule(module), &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    auto pipeline = device_->createComputePipeline({.shaderStages = std::move(stages)}, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    return pipeline;
  }

  [[nodiscard]] std::shared_ptr<IBuffer> createStorage(const std::vector<float>& values) {
    Result ret;
    auto buffer = device_->createBuffer({.type = BufferDesc::BufferTypeBits::Storage,
                                         .data = values.data(),
                                         .length = values.size() * sizeof(float)},
                                        &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    return buffer;
  }

  [[nodiscard]] std::vector<float> read(IBuffer& buffer, size_t count) {
    std::vector<float> values(count);
    Result ret;
    const void* data = buffer.map({count * sizeof(float), 0}, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    if (data != nullptr) {
      std::memcpy(values.data(), data, count * sizeof(float));
    }
    buffer.unmap();
    return values;
  }

  // Encodes `encode` in one compute pass, submits it and waits.
  void run(const std::function<void(IComputeCommandEncoder&)>& encode) {
    auto cmdBuffer = queue_->createCommandBuffer({}, nullptr);
    ASSERT_NE(cmdBuffer, nullptr);
    auto encoder = cmdBuffer->createComputeCommandEncoder();
    ASSERT_NE(encoder, nullptr);
    encode(*encoder);
    encoder->endEncoding();
    queue_->submit(*cmdBuffer);
    cmdBuffer->waitUntilCompleted();
  }

  std::shared_ptr<IDevice> device_;
  webgpu::Device* webgpuDevice_ = nullptr;
  std::shared_ptr<ICommandQueue> queue_;
};

TEST_F(WebGPUComputeTest, StorageBuffersInSteadyState) {
  auto pipeline = createPipeline(kScaleShader);
  ASSERT_NE(pipeline, nullptr);
  auto values = createStorage({1, 2, 3, 4});
  auto results = createStorage({0, 0, 0, 0});
  const auto& webgpuPipeline = static_cast<const webgpu::ComputePipelineState&>(*pipeline);
  auto& cache = webgpuDevice_->getContext().getBindGroupCache();

  size_t bindGroups = 0;
  for (int frame = 0; frame < 3; ++frame) {
    run([&](IComputeCommandEncoder& encoder) {
      encoder.bindComputePipelineState(pipeline);
      encoder.bindBuffer(0, values.get());
      encoder.bindBuffer(1, results.get());
      encoder.dispatchThreadGroups({1, 1, 1}, {4, 1, 1});
    });
    if (frame == 0) {
      bindGroups = cache.getCreationCount();
    }
  }
  EXPECT_EQ(read(*results, 4), (std::vector<float>{1, 2, 3, 4}));
  EXPECT_EQ(webgpuPipeline.getPipelineCreationCount(), 1u);
  EXPECT_EQ(cache.getCreationCount(), bindGroups);
  EXPECT_EQ(pipeline->getIndexByName(IGL_NAMEHANDLE("results")), 1);
}

TEST_F(WebGPUComputeTest, PushConstantsKeepPartialUpdatesAcrossDispatches) {
  auto pipeline = createPipeline(kPushConstantShader);
  ASSERT_NE(pipeline, nullptr);
  auto first = createStorage({0, 0, 0, 0});
  auto second = createStorage({0, 0, 0, 0});
  const float scale = 2.0f;
  const std::array<float, 4> offsets = {10, 20, 30, 40};
  const float newScale = 3.0f;
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.bindPushConstants(&scale, sizeof(scale), 0);
    // offsets[1] starts at byte 32.
    encoder.bindPushConstants(offsets.data(), sizeof(offsets), 32);
    encoder.bindBuffer(0, first.get());
    encoder.dispatchThreadGroups({1, 1, 1}, {4, 1, 1});
    // Only the scale changes; the offsets stay.
    encoder.bindPushConstants(&newScale, sizeof(newScale), 0);
    encoder.bindBuffer(0, second.get());
    encoder.dispatchThreadGroups({1, 1, 1}, {4, 1, 1});
  });
  EXPECT_EQ(read(*first, 4), (std::vector<float>{10, 22, 34, 46}));
  EXPECT_EQ(read(*second, 4), (std::vector<float>{10, 23, 36, 49}));
}

TEST_F(WebGPUComputeTest, PushConstantsOutOfRangeAreIgnored) {
  auto pipeline = createPipeline(kPushConstantShader);
  ASSERT_NE(pipeline, nullptr);
  auto results = createStorage({1, 1, 1, 1});
  const std::array<float, 33> tooMany = {};
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.bindPushConstants(tooMany.data(), sizeof(tooMany), 0);
    encoder.bindBuffer(0, results.get());
    encoder.dispatchThreadGroups({1, 1, 1}, {4, 1, 1});
  });
  // Never-set push constants read zeros.
  EXPECT_EQ(read(*results, 4), (std::vector<float>{0, 0, 0, 0}));
}

TEST_F(WebGPUComputeTest, OverrideConstants) {
  FunctionConstantValues constants;
  const float scale = 3.0f;
  const int32_t bias = -1;
  const float unused = 7.0f;
  constants.setConstantValue(0, ConstantValueType::Float1, &scale);
  constants.setConstantValue(1, ConstantValueType::Int1, &bias);
  constants.setConstantValue(5, ConstantValueType::Float1, &unused);
  auto pipeline = createPipeline(kScaleShader, constants);
  ASSERT_NE(pipeline, nullptr);
  auto values = createStorage({1, 2, 3, 4});
  auto results = createStorage({0, 0, 0, 0});
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.bindBuffer(0, values.get());
    encoder.bindBuffer(1, results.get());
    encoder.dispatchThreadGroups({1, 1, 1}, {4, 1, 1});
  });
  EXPECT_EQ(read(*results, 4), (std::vector<float>{2, 5, 8, 11}));
}

TEST_F(WebGPUComputeTest, IndirectDispatch) {
  auto pipeline = createPipeline(kScaleShader);
  ASSERT_NE(pipeline, nullptr);
  auto values = createStorage({1, 2, 3, 4, 5, 6, 7, 8});
  auto results = createStorage(std::vector<float>(8, 0.0f));
  const std::array<uint32_t, 3> groups = {2, 1, 1};
  Result ret;
  auto indirect = device_->createBuffer(
      {.type = BufferDesc::BufferTypeBits::Indirect, .data = groups.data(), .length = 12}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.bindBuffer(0, values.get());
    encoder.bindBuffer(1, results.get());
    encoder.dispatchThreadGroupsIndirect(*indirect, 0, {4, 1, 1});
  });
  EXPECT_EQ(read(*results, 8), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST_F(WebGPUComputeTest, IndirectDispatchPastTheEndIsSkipped) {
  auto pipeline = createPipeline(kScaleShader);
  ASSERT_NE(pipeline, nullptr);
  auto values = createStorage({1, 2, 3, 4});
  auto results = createStorage({0, 0, 0, 0});
  const std::array<uint32_t, 3> groups = {1, 1, 1};
  Result ret;
  auto indirect = device_->createBuffer(
      {.type = BufferDesc::BufferTypeBits::Indirect, .data = groups.data(), .length = 12}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.bindBuffer(0, values.get());
    encoder.bindBuffer(1, results.get());
    encoder.dispatchThreadGroupsIndirect(*indirect, 4, {4, 1, 1});
  });
  EXPECT_EQ(read(*results, 4), (std::vector<float>{0, 0, 0, 0}));
}

TEST_F(WebGPUComputeTest, SampledAndStorageTextures) {
  auto pipeline = createPipeline(kSwizzleShader);
  ASSERT_NE(pipeline, nullptr);
  Result ret;
  auto src = device_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                       kTextureSize,
                                                       kTextureSize,
                                                       TextureDesc::TextureUsageBits::Sampled),
                                    &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto dst = device_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                       kTextureSize,
                                                       kTextureSize,
                                                       TextureDesc::TextureUsageBits::Storage),
                                    &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const std::array<uint32_t, 4> pixels = {0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d};
  ASSERT_TRUE(
      src->upload(TextureRangeDesc::new2D(0, 0, kTextureSize, kTextureSize), pixels.data()).isOk());

  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.bindTexture(0, src.get());
    encoder.bindImageTexture(0, dst.get(), TextureFormat::RGBA_UNorm8);
    encoder.dispatchThreadGroups({kTextureSize, kTextureSize, 1}, {1, 1, 1});
  });

  std::array<uint32_t, 4> out = {};
  ASSERT_TRUE(static_cast<const webgpu::Texture&>(*dst)
                  .getBytes(TextureRangeDesc::new2D(0, 0, kTextureSize, kTextureSize),
                            WGPUTextureAspect_All,
                            out.data(),
                            kTextureSize * 4,
                            /*flipVertically=*/false)
                  .isOk());
  EXPECT_EQ(out, (std::array<uint32_t, 4>{0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10}));
  EXPECT_EQ(pipeline->getIndexByName(IGL_NAMEHANDLE("src")), 0);
  EXPECT_EQ(pipeline->getIndexByName(IGL_NAMEHANDLE("dst")), 0);
  EXPECT_EQ(pipeline->computePipelineReflection()->allTextures().size(), 2u);
}

TEST_F(WebGPUComputeTest, StorageTexturesAtBindings8And15) {
  auto pipeline = createPipeline(kHighStorageBindingsShader);
  ASSERT_NE(pipeline, nullptr);
  Result ret;
  auto src = device_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                       kTextureSize,
                                                       kTextureSize,
                                                       TextureDesc::TextureUsageBits::Sampled),
                                    &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const TextureDesc storageDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                     kTextureSize,
                                                     kTextureSize,
                                                     TextureDesc::TextureUsageBits::Storage);
  auto dst8 = device_->createTexture(storageDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto dst15 = device_->createTexture(storageDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const std::array<uint32_t, 4> pixels = {0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d};
  ASSERT_TRUE(
      src->upload(TextureRangeDesc::new2D(0, 0, kTextureSize, kTextureSize), pixels.data()).isOk());

  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.bindTexture(0, src.get());
    encoder.bindImageTexture(8, dst8.get(), TextureFormat::RGBA_UNorm8);
    encoder.bindImageTexture(15, dst15.get(), TextureFormat::RGBA_UNorm8);
    encoder.dispatchThreadGroups({kTextureSize, kTextureSize, 1}, {1, 1, 1});
  });

  const auto readTexels = [](const ITexture& texture) {
    std::array<uint32_t, 4> texels = {};
    EXPECT_TRUE(static_cast<const webgpu::Texture&>(texture)
                    .getBytes(TextureRangeDesc::new2D(0, 0, kTextureSize, kTextureSize),
                              WGPUTextureAspect_All,
                              texels.data(),
                              kTextureSize * 4,
                              /*flipVertically=*/false)
                    .isOk());
    return texels;
  };
  EXPECT_EQ(readTexels(*dst8), pixels);
  EXPECT_EQ(readTexels(*dst15),
            (std::array<uint32_t, 4>{0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10}));
  EXPECT_EQ(pipeline->getIndexByName(IGL_NAMEHANDLE("dst8")), 8);
  EXPECT_EQ(pipeline->getIndexByName(IGL_NAMEHANDLE("dst15")), 15);
}

TEST_F(WebGPUComputeTest, UnboundWritableStorageTexturesDoNotAlias) {
  auto pipeline = createPipeline(R"(
@group(2) @binding(0) var a : texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(1) var b : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main() {
  textureStore(a, vec2i(0), vec4f(1.0));
  textureStore(b, vec2i(0), vec4f(1.0));
}
)");
  ASSERT_NE(pipeline, nullptr);
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.dispatchThreadGroups({1, 1, 1}, {1, 1, 1});
  });
}

TEST_F(WebGPUComputeTest, UnboundWritableStorageBuffersDoNotAlias) {
  auto pipeline = createPipeline(R"(
struct Data { values : array<u32, 4>, };
@group(1) @binding(0) var<storage, read_write> a : Data;
@group(1) @binding(1) var<storage, read_write> b : Data;

@compute @workgroup_size(1)
fn main() {
  a.values[0] = 1u;
  b.values[0] = 2u;
}
)");
  ASSERT_NE(pipeline, nullptr);
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.dispatchThreadGroups({1, 1, 1}, {1, 1, 1});
  });
}

TEST_F(WebGPUComputeTest, UnboundWritableStorageBufferDoesNotAliasAUniform) {
  auto pipeline = createPipeline(R"(
struct Data { values : array<u32, 4>, };
struct Params { scale : vec4u, };
@group(1) @binding(0) var<uniform> params : Params;
@group(1) @binding(1) var<storage, read_write> out : Data;

@compute @workgroup_size(1)
fn main() {
  out.values[0] = params.scale.x;
}
)");
  ASSERT_NE(pipeline, nullptr);
  run([&](IComputeCommandEncoder& encoder) {
    encoder.bindComputePipelineState(pipeline);
    encoder.dispatchThreadGroups({1, 1, 1}, {1, 1, 1});
  });
}

TEST_F(WebGPUComputeTest, RejectedBindingsSkipTheDispatch) {
  auto pipeline = createPipeline(kSwizzleShader);
  ASSERT_NE(pipeline, nullptr);
  Result ret;
  auto src = device_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                       kTextureSize,
                                                       kTextureSize,
                                                       TextureDesc::TextureUsageBits::Sampled),
                                    &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  // Storage texture of another format than the declaration, and one without Storage usage.
  auto wrongFormat = device_->createTexture(
      TextureDesc::new2D(
          TextureFormat::R_F32, kTextureSize, kTextureSize, TextureDesc::TextureUsageBits::Storage),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto notStorage =
      device_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                kTextureSize,
                                                kTextureSize,
                                                TextureDesc::TextureUsageBits::Sampled),
                             &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  for (ITexture* texture : {wrongFormat.get(), notStorage.get()}) {
    run([&](IComputeCommandEncoder& encoder) {
      encoder.bindComputePipelineState(pipeline);
      encoder.bindTexture(0, src.get());
      encoder.bindImageTexture(0, texture, TextureFormat::RGBA_UNorm8);
      encoder.dispatchThreadGroups({kTextureSize, kTextureSize, 1}, {1, 1, 1});
    });
  }

  // Formats without STORAGE_BINDING cannot be created with Storage usage.
  EXPECT_EQ(device_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_SRGB,
                                                      kTextureSize,
                                                      kTextureSize,
                                                      TextureDesc::TextureUsageBits::Storage),
                                   &ret),
            nullptr);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);
}

} // namespace igl::tests
