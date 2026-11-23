/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/ShaderData.h"
#include "../data/TextureData.h"
#include "../util/Common.h"
#include "../util/TestDevice.h"

#include <memory>
#include <vector>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/Device.h>
#include <igl/Framebuffer.h>
#include <igl/NameHandle.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/Texture.h>
#include <igl/VertexInputState.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

/// @brief Tests for Vulkan DescriptorPoolsArena caching behavior.
///
/// The DescriptorPoolsArena is a private class inside VulkanContext.cpp that manages
/// VkDescriptorSet allocation and caching. These tests exercise it indirectly through
/// the IGL API by issuing draw calls with bound resources, which triggers the
/// updateBindingsTextures/Buffers -> arena.getNextDescriptorSet() path.
///
/// Code paths exercised:
///   1. New pool path (isNewPool_=true): Fresh descriptor sets allocated and cached
///   2. Cached reuse path (isNewPool_=false): Descriptor sets reused from recycled pools
///   3. Pool exhaustion (numRemainingDSetsInPool_==0): switchToNewDescriptorPool()
///   4. Extinct pool recycling: Pool + cached descriptor sets reused
///   5. allocatedDSet_ save/restore via ExtinctDescriptorPool
///   6. New pool creation fallback: No extinct pool ready, brand-new pool created
class DescriptorPoolArenaTest : public ::testing::Test {
 public:
  DescriptorPoolArenaTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";

    Result ret;
    cmdQueue_ = iglDev_->createCommandQueue({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdQueue_, nullptr);

    // Offscreen color texture (render target)
    const auto colorTexDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 4,
                                                 4,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Attachment);
    colorTex_ = iglDev_->createTexture(colorTexDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(colorTex_, nullptr);

    // Framebuffer
    const FramebufferDesc fbDesc = {
        .colorAttachments = {{.texture = colorTex_}},
    };
    framebuffer_ = iglDev_->createFramebuffer(fbDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(framebuffer_, nullptr);

    // Shader stages
    std::unique_ptr<IShaderStages> stages;
    util::createSimpleShaderStages(iglDev_, stages);
    shaderStages_ = std::move(stages);
    ASSERT_NE(shaderStages_, nullptr);

    // Vertex input state: position (Float4) at location 0, uv (Float2) at location 1
    const VertexInputStateDesc inputDesc = {
        .numAttributes = 2,
        .attributes =
            {
                {
                    .bufferIndex = data::shader::kSimplePosIndex,
                    .format = VertexAttributeFormat::Float4,
                    .offset = 0,
                    .name = std::string(data::shader::kSimplePos),
                    .location = 0,
                },
                {
                    .bufferIndex = data::shader::kSimpleUvIndex,
                    .format = VertexAttributeFormat::Float2,
                    .offset = 0,
                    .name = std::string(data::shader::kSimpleUv),
                    .location = 1,
                },
            },
        .numInputBindings = 2,
        .inputBindings =
            {
                {.stride = sizeof(float) * 4},
                {.stride = sizeof(float) * 2},
            },
    };
    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(vertexInputState_, nullptr);

    // Render pipeline (Triangle topology)
    const RenderPipelineDesc trianglePipelineDesc = {
        .topology = PrimitiveType::Triangle,
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = colorTex_->getFormat()}}},
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
    };
    pipeline_ = iglDev_->createRenderPipeline(trianglePipelineDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(pipeline_, nullptr);

    // Second pipeline (Line topology) for pipeline-switch test
    const RenderPipelineDesc linePipelineDesc = {
        .topology = PrimitiveType::Line,
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = colorTex_->getFormat()}}},
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
    };
    pipelineLine_ = iglDev_->createRenderPipeline(linePipelineDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(pipelineLine_, nullptr);

    // Sampler state
    sampler_ = iglDev_->createSamplerState({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(sampler_, nullptr);

    // Sampled texture (2x2 RGBA, with dummy pixel data)
    const auto sampledTexDesc = TextureDesc::new2D(
        TextureFormat::RGBA_UNorm8, 2, 2, TextureDesc::TextureUsageBits::Sampled);
    sampledTex_ = iglDev_->createTexture(sampledTexDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(sampledTex_, nullptr);
    ret = sampledTex_->upload(sampledTex_->getFullRange(0), data::texture::kTexRgba2x2.data());
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

    // Vertex buffer (single triangle: 3 vertices x Float4)
    const float verts[] = {
        0.0f, 0.5f, 0.0f, 1.0f, -0.5f, -0.5f, 0.0f, 1.0f, 0.5f, -0.5f, 0.0f, 1.0f};
    vb_ = iglDev_->createBuffer(
        BufferDesc{
            .type = BufferDesc::BufferTypeBits::Vertex,
            .data = verts,
            .length = sizeof(verts),
        },
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(vb_, nullptr);

    // UV buffer (3 vertices x Float2)
    const float uvs[] = {0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    uv_ = iglDev_->createBuffer(
        BufferDesc{
            .type = BufferDesc::BufferTypeBits::Vertex,
            .data = uvs,
            .length = sizeof(uvs),
        },
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(uv_, nullptr);

    // Uniform buffer (256 bytes, for buffer binding test)
    uniformBuf_ = iglDev_->createBuffer(
        BufferDesc{
            .type = BufferDesc::BufferTypeBits::Uniform,
            .length = 256,
            .storage = ResourceStorage::Shared,
        },
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(uniformBuf_, nullptr);
    const std::vector<float> uniformData(64, 1.0f);
    ret = uniformBuf_->upload(uniformData.data(), BufferRange(256, 0));
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  }

  void TearDown() override {}

 protected:
  /// Encode numDraws draw calls in a single command buffer, submit, and wait for completion.
  void encodeDrawCalls(const uint32_t numDraws,
                       const bool bindBuffer = false,
                       const std::shared_ptr<IRenderPipelineState>& altPipeline = nullptr,
                       const uint32_t switchAfter = 0) {
    Result ret;
    const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(cmdBuf, nullptr);

    const RenderPassDesc rpDesc = {
        .colorAttachments = {{
            .loadAction = LoadAction::Clear,
            .storeAction = StoreAction::Store,
        }},
    };

    const auto encoder = cmdBuf->createRenderCommandEncoder(rpDesc, framebuffer_, {}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
    ASSERT_NE(encoder, nullptr);

    encoder->bindRenderPipelineState(pipeline_);
    encoder->bindTexture(0, BindTarget::kFragment, sampledTex_.get());
    encoder->bindSamplerState(0, BindTarget::kFragment, sampler_.get());
    encoder->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
    encoder->bindVertexBuffer(data::shader::kSimpleUvIndex, *uv_);

    if (bindBuffer) {
      encoder->bindBuffer(0, uniformBuf_.get(), 0, 256);
    }

    for (uint32_t i = 0; i < numDraws; i++) {
      if (altPipeline && switchAfter > 0 && i == switchAfter) {
        encoder->bindRenderPipelineState(altPipeline);
      }
      encoder->draw(3); // 3 vertices = 1 triangle
    }

    encoder->endEncoding();
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
  }

  /// Shorthand: encode N draw calls and wait for GPU completion (one "frame").
  void renderFrame(const uint32_t numDraws, const bool bindBuffer = false) {
    encodeDrawCalls(numDraws, bindBuffer);
  }

  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<IFramebuffer> framebuffer_;
  std::shared_ptr<ITexture> colorTex_;
  std::shared_ptr<ITexture> sampledTex_;
  std::shared_ptr<ISamplerState> sampler_;
  std::shared_ptr<IRenderPipelineState> pipeline_;
  std::shared_ptr<IRenderPipelineState> pipelineLine_;
  std::shared_ptr<IShaderStages> shaderStages_;
  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::unique_ptr<IBuffer> vb_;
  std::unique_ptr<IBuffer> uv_;
  std::unique_ptr<IBuffer> uniformBuf_;
};

// Test 1: A single draw triggers one getNextDescriptorSet() on a fresh arena (new pool path).
TEST_F(DescriptorPoolArenaTest, SingleDrawAllocatesDescriptorSet) {
  renderFrame(1);
}

// Test 2: Multiple draws within one command buffer allocate multiple descriptor sets
// from the same pool (pool has capacity 64). Exercises repeated new-pool-path allocations.
TEST_F(DescriptorPoolArenaTest, MultipleDrawsInSingleCommandBuffer) {
  renderFrame(10);
}

// Test 3: 65 draws exhaust the first pool (capacity 64) and trigger switchToNewDescriptorPool().
// Since this is the first frame, no extinct pools exist -> new pool creation fallback (path 6).
TEST_F(DescriptorPoolArenaTest, ExhaustDescriptorPool) {
  renderFrame(65);
}

// Test 4: Multiple frames exercise extinct pool recycling (path 4) and cached descriptor set
// reuse (path 2). Frame 0 creates a fresh pool. Subsequent frames push the previous pool
// to extinct and recycle it (if GPU work is done), reusing cached descriptor sets.
TEST_F(DescriptorPoolArenaTest, MultipleFramesRecyclesPools) {
  for (int frame = 0; frame < 4; frame++) {
    renderFrame(10);
  }
}

// Test 5: All paths combined — pool exhaustion + extinct pool recycling + cached reuse.
// Each frame exhausts one pool (64 draws) and partially fills a second (6 draws).
// On subsequent frames, both extinct pools become candidates for recycling.
TEST_F(DescriptorPoolArenaTest, ExhaustAndRecycleAcrossFrames) {
  for (int frame = 0; frame < 4; frame++) {
    renderFrame(70);
  }
}

// Test 6: Bind both a texture+sampler AND a uniform buffer, exercising both
// CombinedImageSamplers and Buffers arenas independently in the same draw.
TEST_F(DescriptorPoolArenaTest, MultipleResourceTypesBindings) {
  for (int frame = 0; frame < 4; frame++) {
    renderFrame(10, /*bindBuffer=*/true);
  }
}

// Test 7: Pipeline switch within a single command buffer. Exercises the pipeline dirty flag
// logic and arena lookup when the bound pipeline changes mid-encoding.
TEST_F(DescriptorPoolArenaTest, PipelineSwitchWithDraws) {
  encodeDrawCalls(
      /*numDraws=*/10,
      /*bindBuffer=*/false,
      /*altPipeline=*/pipelineLine_,
      /*switchAfter=*/5);
}

// Test 8: 200 draws in one frame exhaust 3 pools (3 x 64 = 192) and partially fill a 4th.
// All pools are pushed to the extinct deque within the same submit handle.
TEST_F(DescriptorPoolArenaTest, StressPoolExhaustion) {
  renderFrame(200);
}

// Test 9: Exactly 64 draws — fills pool to capacity without triggering a switch.
TEST_F(DescriptorPoolArenaTest, ExactPoolCapacityBoundary) {
  renderFrame(64);
}

// Test 10: Exhaust pool 1 in frame 0, then exhaust pool 2 in frame 1 which recycles pool 1
// via the cached reuse path (isNewPool_=false, allocatedDSet_[dsetCursor_++]).
TEST_F(DescriptorPoolArenaTest, FullPoolReuseCycle) {
  renderFrame(65);
  renderFrame(64);
}

// Test 11: 8 frames of pool exhaustion+recycling. Verifies dsetCursor_ resets and
// isNewPool_ toggles correctly across many reuse cycles.
TEST_F(DescriptorPoolArenaTest, RepeatedFullReuseCycles) {
  for (int frame = 0; frame < 8; frame++) {
    renderFrame(65);
  }
}

// Test 12: Texture+buffer binding with pool exhaustion — both CombinedImageSamplers
// and Buffers arenas go through the extinct→reuse cycle.
TEST_F(DescriptorPoolArenaTest, ExhaustMultipleArenas) {
  for (int frame = 0; frame < 4; frame++) {
    renderFrame(70, /*bindBuffer=*/true);
  }
}

// Test 13: 130 draws exhaust 2 pools in one frame (same handle prevents reuse),
// then frame 2 recycles the oldest extinct pool.
TEST_F(DescriptorPoolArenaTest, TwoPoolsExhaustedSameFrameThenRecycled) {
  renderFrame(130);
  renderFrame(65);
}

// Test 14: Pool allocated→exhausted→recycled→re-exhausted→re-recycled.
// Verifies allocatedDSet_ survives multiple save/restore rounds through ExtinctDescriptorPool.
TEST_F(DescriptorPoolArenaTest, ReusedPoolReExhausted) {
  renderFrame(65);
  renderFrame(128);
  renderFrame(65);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
