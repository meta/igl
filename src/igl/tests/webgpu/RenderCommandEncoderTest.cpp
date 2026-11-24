/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/webgpu/RenderCommandEncoder.h>

#include <IGLU/simple_renderer/ShaderUniforms.h>
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <igl/CommandBuffer.h>
#include <igl/CommandQueue.h>
#include <igl/ShaderCreator.h>
#include <igl/tests/data/ShaderData.h>
#include <igl/tests/util/device/webgpu/TestDevice.h>
#include <igl/webgpu/RenderPipelineState.h>
#include <igl/webgpu/ResourcesBinder.h>

namespace igl::tests {

namespace {

constexpr uint32_t kSize = 4;

// Solid color from a uniform buffer.
constexpr const char* kColorFragment = R"(
struct Uniforms { color : vec4f, };
@group(1) @binding(0) var<uniform> uniforms : Uniforms;

@fragment
fn main() -> @location(0) vec4f {
  return uniforms.color;
}
)";

// A full-screen quad as a triangle strip, at depth `z`.
std::vector<float> quad(float z) {
  return {-1, 1, z, 1, -1, -1, z, 1, 1, 1, z, 1, 1, -1, z, 1};
}

// Lower-right triangle, counter-clockwise.
std::vector<float> lowerRightTriangle() {
  return {-1, -1, 0, 1, 1, -1, 0, 1, 1, 1, 0, 1};
}

// Upper-left triangle, counter-clockwise.
std::vector<float> upperLeftTriangle() {
  return {-1, -1, 0, 1, 1, 1, 0, 1, -1, 1, 0, 1};
}

struct DrawIndirectArgs {
  uint32_t vertexCount = 0;
  uint32_t instanceCount = 0;
  uint32_t firstVertex = 0;
  uint32_t firstInstance = 0;
};

struct DrawIndexedIndirectArgs {
  uint32_t indexCount = 0;
  uint32_t instanceCount = 0;
  uint32_t firstIndex = 0;
  int32_t baseVertex = 0;
  uint32_t firstInstance = 0;
};

uint32_t rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  return r | (g << 8) | (b << 16) | (static_cast<uint32_t>(a) << 24);
}

} // namespace

class WebGPURenderCommandEncoderTest : public ::testing::Test {
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
    color_ = device_->createTexture(TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                       kSize,
                                                       kSize,
                                                       TextureDesc::TextureUsageBits::Attachment |
                                                           TextureDesc::TextureUsageBits::Sampled),
                                    &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    depth_ = device_->createTexture(TextureDesc::new2D(TextureFormat::S8_UInt_Z32_UNorm,
                                                       kSize,
                                                       kSize,
                                                       TextureDesc::TextureUsageBits::Attachment),
                                    &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    framebuffer_ = device_->createFramebuffer({.colorAttachments = {{.texture = color_}},
                                               .depthAttachment = {.texture = depth_},
                                               .stencilAttachment = {.texture = depth_}},
                                              &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    uniforms_ =
        device_->createBuffer({.type = BufferDesc::BufferTypeBits::Uniform, .length = 512}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    vertexInput_ = device_->createVertexInputState(
        {
            .numAttributes = 1,
            .attributes = {{.bufferIndex = 0,
                            .format = VertexAttributeFormat::Float4,
                            .location = 0}},
            .numInputBindings = 1,
            .inputBindings = {{.stride = 16}},
        },
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    stages_ = ShaderStagesCreator::fromModuleStringInput(
        *device_,
        "@vertex fn main(@location(0) p : vec4f) -> @builtin(position) vec4f { return p; }",
        "main",
        "",
        kColorFragment,
        "main",
        "",
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
  }

  void TearDown() override {
    EXPECT_EQ(webgpuDevice_->getContext().getUncapturedErrorCount(), 0u);
  }

 protected:
  [[nodiscard]] std::shared_ptr<IRenderPipelineState> createPipeline(
      PrimitiveType topology,
      const RenderPipelineDesc::TargetDesc::ColorAttachment& color = {
          .textureFormat = TextureFormat::RGBA_UNorm8}) {
    Result ret;
    auto pipeline = device_->createRenderPipeline(
        {
            .topology = topology,
            .vertexInputState = vertexInput_,
            .shaderStages = stages_,
            .targetDesc = {.colorAttachments = {color},
                           .depthAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm,
                           .stencilAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm},
        },
        &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    return pipeline;
  }

  [[nodiscard]] std::shared_ptr<IBuffer> createVertices(const std::vector<float>& vertices) {
    Result ret;
    auto buffer = device_->createBuffer({.type = BufferDesc::BufferTypeBits::Vertex,
                                         .data = vertices.data(),
                                         .length = vertices.size() * sizeof(float)},
                                        &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    return buffer;
  }

  [[nodiscard]] std::shared_ptr<IBuffer> createIndirect(const void* data, size_t length) {
    Result ret;
    auto buffer = device_->createBuffer(
        {.type = BufferDesc::BufferTypeBits::Indirect, .data = data, .length = length}, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    return buffer;
  }

  [[nodiscard]] std::shared_ptr<IBuffer> twoTriangles() {
    std::vector<float> vertices = lowerRightTriangle();
    const std::vector<float> upperLeft = upperLeftTriangle();
    vertices.insert(vertices.end(), upperLeft.begin(), upperLeft.end());
    return createVertices(vertices);
  }

  void setColor(const std::array<float, 4>& color, size_t offset = 0) {
    ASSERT_TRUE(uniforms_->upload(color.data(), {sizeof(color), offset}).isOk());
  }

  void encode(const std::function<void(IRenderCommandEncoder&)>& func,
              const RenderPassDesc& renderPass) {
    Result ret;
    auto cmdBuffer = queue_->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    auto encoder = cmdBuffer->createRenderCommandEncoder(renderPass, framebuffer_, {}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(encoder, nullptr);
    func(*encoder);
    encoder->endEncoding();
    queue_->submit(*cmdBuffer);
    cmdBuffer->waitUntilCompleted();
  }

  // Top row first.
  [[nodiscard]] std::vector<uint32_t> readColor() const {
    std::vector<uint32_t> pixels(kSize * kSize);
    framebuffer_->copyBytesColorAttachment(
        *queue_, 0, pixels.data(), TextureRangeDesc::new2D(0, 0, kSize, kSize));
    std::vector<uint32_t> topDown(pixels.size());
    for (uint32_t row = 0; row < kSize; ++row) {
      std::memcpy(topDown.data() + row * kSize,
                  pixels.data() + (kSize - 1 - row) * kSize,
                  kSize * sizeof(uint32_t));
    }
    return topDown;
  }

  static RenderPassDesc clearPass(float depth = 1.0f, uint32_t stencil = 0) {
    return {
        .colorAttachments = {{.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::Store,
                              .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
        .depthAttachment = {.loadAction = LoadAction::Clear,
                            .storeAction = StoreAction::Store,
                            .clearDepth = depth},
        .stencilAttachment = {.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::Store,
                              .clearStencil = stencil},
    };
  }

  std::shared_ptr<IDevice> device_;
  webgpu::Device* webgpuDevice_ = nullptr;
  std::shared_ptr<ICommandQueue> queue_;
  std::shared_ptr<ITexture> color_;
  std::shared_ptr<ITexture> depth_;
  std::shared_ptr<IFramebuffer> framebuffer_;
  std::shared_ptr<IBuffer> uniforms_;
  std::shared_ptr<IVertexInputState> vertexInput_;
  std::shared_ptr<IShaderStages> stages_;
};

TEST_F(WebGPURenderCommandEncoderTest, ClearsDepthAndStencil) {
  encode([](IRenderCommandEncoder& /*encoder*/) {}, clearPass(0.25f, 0x5a));
  std::vector<float> depth(kSize * kSize);
  framebuffer_->copyBytesDepthAttachment(
      *queue_, depth.data(), TextureRangeDesc::new2D(0, 0, kSize, kSize));
  EXPECT_EQ(depth[5], 0.25f);
  std::vector<uint8_t> stencil(kSize * kSize);
  framebuffer_->copyBytesStencilAttachment(
      *queue_, stencil.data(), TextureRangeDesc::new2D(0, 0, kSize, kSize));
  EXPECT_EQ(stencil[5], 0x5a);
}

TEST_F(WebGPURenderCommandEncoderTest, DepthTestPicksVariants) {
  auto pipeline = createPipeline(PrimitiveType::TriangleStrip);
  auto nearQuad = createVertices(quad(0.25f));
  auto farQuad = createVertices(quad(0.75f));
  Result ret;
  auto lessWrite = device_->createDepthStencilState(
      {.compareFunction = CompareFunction::Less, .isDepthWriteEnabled = true}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  setColor({1, 0, 0, 1}, 0);
  setColor({0, 1, 0, 1}, 256);
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindDepthStencilState(lessWrite);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *nearQuad);
        encoder.draw(4);
        // Behind the first quad: rejected by the depth test.
        encoder.bindBuffer(0, uniforms_.get(), 256, 16);
        encoder.bindVertexBuffer(0, *farQuad);
        encoder.draw(4);
      },
      clearPass());
  for (const uint32_t pixel : readColor()) {
    EXPECT_EQ(pixel, rgba(255, 0, 0, 255));
  }
  // Default state (AlwaysPass) plus Less-and-write.
  EXPECT_EQ(static_cast<webgpu::RenderPipelineState&>(*pipeline).getVariantCount(), 2u);
}

TEST_F(WebGPURenderCommandEncoderTest, CullModeAndWindingOverrides) {
  auto pipeline = createPipeline(PrimitiveType::Triangle);
  auto triangle = createVertices(lowerRightTriangle());
  setColor({1, 1, 1, 1});
  const auto drawWith = [&](CullMode cullMode, WindingMode winding) {
    encode(
        [&](IRenderCommandEncoder& encoder) {
          encoder.bindRenderPipelineState(pipeline);
          encoder.bindBuffer(0, uniforms_.get(), 0, 16);
          encoder.bindVertexBuffer(0, *triangle);
          encoder.setCullMode(cullMode);
          encoder.setFrontFacingWinding(winding);
          encoder.draw(3);
        },
        clearPass());
    return readColor()[kSize * kSize - 1];
  };
  EXPECT_EQ(drawWith(CullMode::Back, WindingMode::CounterClockwise), rgba(255, 255, 255, 255));
  EXPECT_EQ(drawWith(CullMode::Back, WindingMode::Clockwise), rgba(0, 0, 0, 255));
  EXPECT_EQ(drawWith(CullMode::Front, WindingMode::Clockwise), rgba(255, 255, 255, 255));

  // A pipeline bound after the override uses its own cull mode (none) again.
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.setCullMode(CullMode::Front);
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *triangle);
        encoder.draw(3);
      },
      clearPass());
  EXPECT_EQ(readColor()[kSize * kSize - 1], rgba(255, 255, 255, 255));
}

TEST_F(WebGPURenderCommandEncoderTest, DynamicBindGroupBuffersNeedOffsets) {
  auto pipeline = createPipeline(PrimitiveType::TriangleStrip);
  auto vertices = createVertices(quad(0.5f));
  setColor({1, 0, 0, 1}, 0);
  setColor({0, 1, 0, 1}, 256);
  Result ret;
  auto bindGroup = device_->createBindGroup(
      BindGroupBufferDesc{.buffers = {uniforms_}, .size = {16}, .isDynamicBufferMask = 1}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto drawWith = [&](const uint32_t* dynamicOffsets, uint32_t numDynamicOffsets) {
    encode(
        [&](IRenderCommandEncoder& encoder) {
          encoder.bindRenderPipelineState(pipeline);
          encoder.bindBuffer(0, uniforms_.get(), 256, 16);
          encoder.bindVertexBuffer(0, *vertices);
          encoder.bindBindGroup(bindGroup, numDynamicOffsets, dynamicOffsets);
          encoder.draw(4);
        },
        clearPass());
    return readColor()[5];
  };
  const uint32_t offset = 0;
  EXPECT_EQ(drawWith(&offset, 1), rgba(255, 0, 0, 255));
  // Without an offset the dynamic buffer is not bound, so the earlier binding stays.
  EXPECT_EQ(drawWith(nullptr, 0), rgba(0, 255, 0, 255));
}

// ShaderUniforms writes mat3x3f with 16-byte columns and puts blocks over the bindBytes() limit
// (4 KiB) in a buffer.
TEST_F(WebGPURenderCommandEncoderTest, ShaderUniformsLargeBlockAndMat3x3) {
  constexpr const char* kFragment = R"(
struct Uniforms { m : mat3x3f, pad : array<vec4f, 300>, color : vec4f, };
@group(1) @binding(0) var<uniform> uniforms : Uniforms;

@fragment
fn main() -> @location(0) vec4f {
  return vec4f(uniforms.m[2], 1.0) * uniforms.color;
}
)";
  Result ret;
  std::shared_ptr<IShaderStages> stages = ShaderStagesCreator::fromModuleStringInput(
      *device_,
      "@vertex fn main(@location(0) p : vec4f) -> @builtin(position) vec4f { return p; }",
      "main",
      "",
      kFragment,
      "main",
      "",
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto pipeline = device_->createRenderPipeline(
      {
          .topology = PrimitiveType::TriangleStrip,
          .vertexInputState = vertexInput_,
          .shaderStages = stages,
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}},
                         .depthAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm,
                         .stencilAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto reflection = pipeline->renderPipelineReflection();
  ASSERT_NE(reflection, nullptr);
  iglu::material::ShaderUniforms shaderUniforms(*device_, *reflection);
  const iglu::simdtypes::float3 zero = {0.0f, 0.0f, 0.0f};
  const iglu::simdtypes::float3 green = {0.0f, 1.0f, 0.0f};
  shaderUniforms.setFloat3x3(genNameHandle("m"), iglu::simdtypes::float3x3(zero, zero, green));
  const iglu::simdtypes::float4 white = {1.0f, 1.0f, 1.0f, 1.0f};
  shaderUniforms.setFloat4(genNameHandle("color"), white);
  auto vertices = createVertices(quad(0.5f));
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        shaderUniforms.bind(*device_, *pipeline, encoder);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.draw(4);
      },
      clearPass());
  EXPECT_EQ(readColor()[5], rgba(0, 255, 0, 255));
}

TEST_F(WebGPURenderCommandEncoderTest, PushConstantBlocksOfDifferentSizesPerStage) {
  if (!device_->hasFeature(DeviceFeatures::PushConstants)) {
    GTEST_SKIP() << "No push constants";
  }
  constexpr const char* kVertex = R"(
struct VertexConstants { scale : vec4f, };
@group(3) @binding(0) var<uniform> pc : VertexConstants;
@vertex fn main(@location(0) p : vec4f) -> @builtin(position) vec4f {
  return vec4f(p.xyz * pc.scale.x, 1.0);
}
)";
  constexpr const char* kFragment = R"(
struct FragmentConstants { scale : vec4f, color : vec4f, };
@group(3) @binding(0) var<uniform> pc : FragmentConstants;
@fragment fn main() -> @location(0) vec4f {
  return pc.color;
}
)";
  Result ret;
  std::shared_ptr<IShaderStages> stages = ShaderStagesCreator::fromModuleStringInput(
      *device_, kVertex, "main", "", kFragment, "main", "", &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto pipeline = device_->createRenderPipeline(
      {
          .topology = PrimitiveType::TriangleStrip,
          .vertexInputState = vertexInput_,
          .shaderStages = stages,
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}},
                         .depthAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm,
                         .stencilAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto vertices = createVertices(quad(0.5f));
  const std::array<float, 8> constants = {1, 1, 1, 1, 0, 1, 0, 1};
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindPushConstants(constants.data(), sizeof(constants));
        encoder.bindVertexBuffer(0, *vertices);
        encoder.draw(4);
      },
      clearPass());
  EXPECT_EQ(readColor()[5], rgba(0, 255, 0, 255));
}

TEST_F(WebGPURenderCommandEncoderTest, ScissorAndViewport) {
  auto pipeline = createPipeline(PrimitiveType::TriangleStrip);
  auto vertices = createVertices(quad(0.5f));
  setColor({0, 0, 1, 1});
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *vertices);
        // Top-left quadrant; the rectangle is clamped to the target.
        encoder.bindViewport({.x = 0, .y = 0, .width = 2, .height = 2});
        encoder.bindScissorRect({.x = 0, .y = 0, .width = 100, .height = 100});
        encoder.draw(4);
      },
      clearPass());
  const std::vector<uint32_t> pixels = readColor();
  EXPECT_EQ(pixels[0], rgba(0, 0, 255, 255));
  EXPECT_EQ(pixels[kSize + 1], rgba(0, 0, 255, 255));
  EXPECT_EQ(pixels[2], rgba(0, 0, 0, 255));
  EXPECT_EQ(pixels[2 * kSize], rgba(0, 0, 0, 255));
}

TEST_F(WebGPURenderCommandEncoderTest, IndexedStripAndBlendColor) {
  auto pipeline = createPipeline(PrimitiveType::TriangleStrip,
                                 {.textureFormat = TextureFormat::RGBA_UNorm8,
                                  .blendEnabled = true,
                                  .srcRGBBlendFactor = BlendFactor::BlendColor,
                                  .srcAlphaBlendFactor = BlendFactor::One,
                                  .dstRGBBlendFactor = BlendFactor::Zero,
                                  .dstAlphaBlendFactor = BlendFactor::Zero});
  auto vertices = createVertices(quad(0.5f));
  const std::array<uint16_t, 4> indices = {0, 1, 2, 3};
  Result ret;
  auto indexBuffer = device_->createBuffer({.type = BufferDesc::BufferTypeBits::Index,
                                            .data = indices.data(),
                                            .length = sizeof(indices)},
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  setColor({1, 1, 1, 1});
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.bindIndexBuffer(*indexBuffer, IndexFormat::UInt16);
        encoder.setBlendColor({0.0f, 1.0f, 0.0f, 1.0f});
        encoder.drawIndexed(4);
      },
      clearPass());
  for (const uint32_t pixel : readColor()) {
    EXPECT_EQ(pixel, rgba(0, 255, 0, 255));
  }
}

TEST_F(WebGPURenderCommandEncoderTest, SkipsDrawsWebGPUWouldReject) {
  auto pipeline = createPipeline(PrimitiveType::TriangleStrip);
  auto vertices = createVertices(quad(0.5f));
  setColor({1, 1, 1, 1});
  const size_t drawCount = device_->getCurrentDrawCount();
  encode(
      [&](IRenderCommandEncoder& encoder) {
        // No pipeline, then no vertex buffer, then no index buffer.
        encoder.draw(4);
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.draw(4);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.drawIndexed(4);
        // 8-bit indices do not exist in WebGPU.
        encoder.bindIndexBuffer(*vertices, IndexFormat::UInt8);
        encoder.drawIndexed(4);
        // A uniform offset that is not a multiple of 256.
        encoder.bindBuffer(0, uniforms_.get(), 16, 16);
        encoder.draw(4);
      },
      clearPass());
  for (const uint32_t pixel : readColor()) {
    EXPECT_EQ(pixel, rgba(0, 0, 0, 255));
  }
  // Every draw call counts, as on the other backends (shared DeviceTest.LastDrawStat).
  EXPECT_EQ(device_->getCurrentDrawCount(), drawCount + 5);
}

TEST_F(WebGPURenderCommandEncoderTest, SampledAndAttachedTextureIsRejected) {
  Result ret;
  std::shared_ptr<IShaderStages> stages = ShaderStagesCreator::fromModuleStringInput(
      *device_,
      std::string(data::shader::kWgslSimpleVertShader).c_str(),
      "main",
      "",
      std::string(data::shader::kWgslSimpleFragShader).c_str(),
      "main",
      "",
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto uvInput = device_->createVertexInputState(
      {
          .numAttributes = 2,
          .attributes = {{.bufferIndex = 0, .format = VertexAttributeFormat::Float4, .location = 0},
                         {.bufferIndex = 0,
                          .format = VertexAttributeFormat::Float2,
                          .location = 1}},
          .numInputBindings = 1,
          .inputBindings = {{.stride = 16}},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto pipeline = device_->createRenderPipeline(
      {
          .topology = PrimitiveType::TriangleStrip,
          .vertexInputState = uvInput,
          .shaderStages = stages,
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}},
                         .depthAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm,
                         .stencilAttachmentFormat = TextureFormat::S8_UInt_Z32_UNorm},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  auto vertices = createVertices(quad(0.5f));
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.bindTexture(0, color_.get());
        encoder.draw(4);
      },
      clearPass());
}

TEST_F(WebGPURenderCommandEncoderTest, SteadyStateCreatesNothing) {
  auto pipeline = createPipeline(PrimitiveType::TriangleStrip);
  auto vertices = createVertices(quad(0.5f));
  setColor({1, 0, 1, 1});
  const auto frame = [&] {
    encode(
        [&](IRenderCommandEncoder& encoder) {
          encoder.pushDebugGroupLabel("frame");
          encoder.bindRenderPipelineState(pipeline);
          encoder.bindBuffer(0, uniforms_.get(), 0, 16);
          encoder.bindVertexBuffer(0, *vertices);
          encoder.insertDebugEventLabel("draw");
          encoder.draw(4);
          encoder.popDebugGroupLabel();
        },
        clearPass());
  };
  frame();
  auto& webgpuPipeline = static_cast<webgpu::RenderPipelineState&>(*pipeline);
  const size_t pipelines = webgpuPipeline.getPipelineCreationCount();
  const size_t bindGroups = webgpuDevice_->getContext().getBindGroupCache().getCreationCount();
  for (int i = 0; i < 5; ++i) {
    frame();
  }
  EXPECT_EQ(webgpuPipeline.getPipelineCreationCount(), pipelines);
  EXPECT_EQ(webgpuDevice_->getContext().getBindGroupCache().getCreationCount(), bindGroups);
  EXPECT_EQ(readColor()[0], rgba(255, 0, 255, 255));
}

TEST_F(WebGPURenderCommandEncoderTest, MultiDrawIndirect) {
  ASSERT_TRUE(device_->hasFeature(DeviceFeatures::DrawIndexedIndirect));
  auto pipeline = createPipeline(PrimitiveType::Triangle);
  auto vertices = twoTriangles();
  // The middle record draws nothing; the other two cover the target.
  const std::array<DrawIndirectArgs, 3> args = {{
      {.vertexCount = 3, .instanceCount = 1},
      {.vertexCount = 3, .instanceCount = 0},
      {.vertexCount = 3, .instanceCount = 1, .firstVertex = 3},
  }};
  auto indirect = createIndirect(args.data(), sizeof(args));
  setColor({1, 1, 0, 1});
  const size_t drawCount = device_->getCurrentDrawCount();
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.multiDrawIndirect(*indirect, 0, 3);
      },
      clearPass());
  for (const uint32_t pixel : readColor()) {
    EXPECT_EQ(pixel, rgba(255, 255, 0, 255));
  }
  EXPECT_EQ(device_->getCurrentDrawCount(), drawCount + 3);
}

TEST_F(WebGPURenderCommandEncoderTest, MultiDrawIndirectOffsetAndStride) {
  auto pipeline = createPipeline(PrimitiveType::Triangle);
  auto vertices = twoTriangles();
  // Records 32 bytes apart after a 16-byte header; only the upper-left triangle is drawn.
  std::array<uint32_t, 20> words = {};
  const DrawIndirectArgs upperLeft = {.vertexCount = 3, .instanceCount = 1, .firstVertex = 3};
  const DrawIndirectArgs lowerRight = {.vertexCount = 3, .instanceCount = 1};
  std::memcpy(&words[4], &upperLeft, sizeof(upperLeft));
  std::memcpy(&words[12], &lowerRight, sizeof(lowerRight));
  auto indirect = createIndirect(words.data(), sizeof(words));
  setColor({0, 1, 1, 1});
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.multiDrawIndirect(*indirect, 16, 1, 32);
      },
      clearPass());
  const std::vector<uint32_t> pixels = readColor();
  EXPECT_EQ(pixels[0], rgba(0, 255, 255, 255));
  EXPECT_EQ(pixels[kSize * kSize - 1], rgba(0, 0, 0, 255));

  // Two records 32 bytes apart draw both triangles.
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.multiDrawIndirect(*indirect, 16, 2, 32);
      },
      clearPass());
  const std::vector<uint32_t> both = readColor();
  EXPECT_EQ(both[0], rgba(0, 255, 255, 255));
  EXPECT_EQ(both[kSize * kSize - 1], rgba(0, 255, 255, 255));
}

TEST_F(WebGPURenderCommandEncoderTest, MultiDrawIndexedIndirect) {
  auto pipeline = createPipeline(PrimitiveType::Triangle);
  auto vertices = twoTriangles();
  const std::array<uint16_t, 6> indices = {0, 1, 2, 0, 1, 2};
  Result ret;
  auto indexBuffer = device_->createBuffer({.type = BufferDesc::BufferTypeBits::Index,
                                            .data = indices.data(),
                                            .length = sizeof(indices)},
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  // The second record reaches the upper-left triangle through baseVertex.
  const std::array<DrawIndexedIndirectArgs, 2> args = {{
      {.indexCount = 3, .instanceCount = 1},
      {.indexCount = 3, .instanceCount = 1, .firstIndex = 3, .baseVertex = 3},
  }};
  auto indirect = createIndirect(args.data(), sizeof(args));
  setColor({1, 0, 1, 1});
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *vertices);
        encoder.bindIndexBuffer(*indexBuffer, IndexFormat::UInt16);
        encoder.multiDrawIndexedIndirect(*indirect, 0, 2);
      },
      clearPass());
  for (const uint32_t pixel : readColor()) {
    EXPECT_EQ(pixel, rgba(255, 0, 255, 255));
  }
}

TEST_F(WebGPURenderCommandEncoderTest, SkipsInvalidIndirectDraws) {
  auto pipeline = createPipeline(PrimitiveType::Triangle);
  auto vertices = twoTriangles();
  const DrawIndirectArgs args = {.vertexCount = 3, .instanceCount = 1};
  auto indirect = createIndirect(&args, sizeof(args));
  const DrawIndexedIndirectArgs indexedArgs = {.indexCount = 3, .instanceCount = 1};
  auto indexedIndirect = createIndirect(&indexedArgs, sizeof(indexedArgs));
  auto notIndirect = createVertices({3, 1, 0, 0});
  setColor({1, 1, 1, 1});
  const size_t drawCount = device_->getCurrentDrawCount();
  encode(
      [&](IRenderCommandEncoder& encoder) {
        encoder.bindRenderPipelineState(pipeline);
        encoder.bindBuffer(0, uniforms_.get(), 0, 16);
        encoder.bindVertexBuffer(0, *vertices);
        // Not an Indirect buffer; unaligned offset; past the end; no index buffer.
        encoder.multiDrawIndirect(*notIndirect, 0, 1);
        encoder.multiDrawIndirect(*indirect, 2, 1);
        encoder.multiDrawIndirect(*indirect, 0, 2);
        encoder.multiDrawIndexedIndirect(*indexedIndirect, 0, 1);
      },
      clearPass());
  for (const uint32_t pixel : readColor()) {
    EXPECT_EQ(pixel, rgba(0, 0, 0, 255));
  }
  // Skipped draws still count (shared DeviceTest.LastDrawStat).
  EXPECT_EQ(device_->getCurrentDrawCount(), drawCount + 5);
}

} // namespace igl::tests
