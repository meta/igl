/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <igl/Buffer.h>
#include <igl/CommandBuffer.h>
#include <igl/DepthStencilState.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/ShaderCreator.h>
#include <igl/VertexInputState.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/Framebuffer.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/Texture.h>

namespace igl::tests {

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kSamples = 4;

// Opaque colors as the pixels of an RGBA_UNorm8 texture that is read back
constexpr uint32_t kBlack = 0xFF000000;
constexpr uint32_t kRed = 0xFF0000FF;
constexpr uint32_t kGreen = 0xFF00FF00;

constexpr const char* kVertexShader = R"(#version 300 es
layout(location = 0) in vec4 position_in;
void main() {
  gl_Position = position_in;
})";

// The vertices of three triangles in the vertex buffer:
//  - a triangle with its right angle at the bottom left of the framebuffer. Its hypotenuse crosses
//    the framebuffer diagonally, through the centers of the pixels along it.
//  - a triangle that covers the framebuffer, at the depth of 0.25
//  - a triangle that covers the framebuffer, at the depth of 0.75
constexpr std::array<float, 36> kVertices = {
    -1.0f, -1.0f, 0.0f,  1.0f, 1.0f, -1.0f, 0.0f,  1.0f, -1.0f, 1.0f, 0.0f,  1.0f,
    -1.0f, -1.0f, -0.5f, 1.0f, 3.0f, -1.0f, -0.5f, 1.0f, -1.0f, 3.0f, -0.5f, 1.0f,
    -1.0f, -1.0f, 0.5f,  1.0f, 3.0f, -1.0f, 0.5f,  1.0f, -1.0f, 3.0f, 0.5f,  1.0f,
};
constexpr uint32_t kDiagonalFirstVertex = 0;
constexpr uint32_t kNearFirstVertex = 3;
constexpr uint32_t kFarFirstVertex = 6;

constexpr const char* kRedFragmentShader = R"(#version 300 es
precision highp float;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(1.0, 0.0, 0.0, 1.0);
})";

constexpr const char* kGreenFragmentShader = R"(#version 300 es
precision highp float;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(0.0, 1.0, 0.0, 1.0);
})";

} // namespace

//
// MultisampledRenderToTextureOGLTest
//
// Tests for resolving Memoryless multisampled attachments with
// GL_EXT_multisampled_render_to_texture where the device has it, and with a framebuffer blit where
// it does not.
//
class MultisampledRenderToTextureOGLTest : public ::testing::Test {
 public:
  MultisampledRenderToTextureOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
    createVertices();
  }

  void TearDown() override {}

 protected:
  [[nodiscard]] bool resolvesWithExtension() const {
    return context_->deviceFeatures().hasInternalFeature(
        opengl::InternalFeatures::MultisampledRenderToTexture);
  }

  // Returns why the device has no 4x multisampled attachments, or nullptr if it has
  [[nodiscard]] const char* IGL_NULLABLE multisampleSkipReason() const {
    if (!iglDev_->hasFeature(DeviceFeatures::MultiSample)) {
      return "MultiSample not supported";
    }
    size_t maxSamples = 0;
    if (!iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxMultisampleCount, maxSamples) ||
        maxSamples < kSamples) {
      return "4x multisampling not supported";
    }
    return nullptr;
  }

  // Returns why a pass cannot resolve a multisampled attachment, or nullptr if it can. That takes
  // the extension or a framebuffer blit.
  [[nodiscard]] const char* IGL_NULLABLE resolveSkipReason() const {
    if (const char* reason = multisampleSkipReason()) {
      return reason;
    }
    if (!resolvesWithExtension() &&
        !context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::FramebufferBlit)) {
      return "Neither multisampled render-to-texture nor framebuffer blit is supported";
    }
    return nullptr;
  }

  // Returns why the draws of the tests cannot run, or nullptr if they can
  [[nodiscard]] const char* IGL_NULLABLE drawSkipReason() const {
    if (const char* reason = resolveSkipReason()) {
      return reason;
    }
    const BackendVersion version = iglDev_->getBackendVersion();
    if (version.flavor != BackendFlavor::OpenGL_ES || version.majorVersion < 3) {
      return "The draws have OpenGL ES 3.0 shaders only";
    }
    return nullptr;
  }

  [[nodiscard]] std::shared_ptr<ITexture> createMemorylessTexture(TextureFormat format,
                                                                  uint32_t size = kSize) const {
    const TextureDesc desc{
        .width = size,
        .height = size,
        .numSamples = kSamples,
        .usage = TextureDesc::TextureUsageBits::Attachment,
        .type = TextureType::TwoD,
        .format = format,
        .storage = ResourceStorage::Memoryless,
    };
    Result ret;
    auto texture = iglDev_->createTexture(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(texture, nullptr);
    return texture;
  }

  // A TextureBuffer, which a framebuffer can resolve into with the extension
  [[nodiscard]] std::shared_ptr<ITexture> createResolveTexture(uint32_t size = kSize) const {
    const TextureDesc desc{
        .width = size,
        .height = size,
        .usage = TextureDesc::TextureUsageBits::Attachment | TextureDesc::TextureUsageBits::Sampled,
        .type = TextureType::TwoD,
        .format = TextureFormat::RGBA_UNorm8,
    };
    Result ret;
    auto texture = iglDev_->createTexture(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(texture, nullptr);
    return texture;
  }

  // A TextureTarget, which is a renderbuffer that a framebuffer cannot resolve into with the
  // extension
  [[nodiscard]] std::shared_ptr<ITexture> createRenderbufferTexture() const {
    const TextureDesc desc{
        .width = kSize,
        .height = kSize,
        .usage = TextureDesc::TextureUsageBits::Attachment,
        .type = TextureType::TwoD,
        .format = TextureFormat::RGBA_UNorm8,
    };
    Result ret;
    auto texture = iglDev_->createTexture(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(texture, nullptr);
    return texture;
  }

  [[nodiscard]] std::shared_ptr<IFramebuffer> createFramebuffer(const FramebufferDesc& desc) const {
    Result ret;
    auto framebuffer = iglDev_->createFramebuffer(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(framebuffer, nullptr);
    return framebuffer;
  }

  [[nodiscard]] static const opengl::CustomFramebuffer& custom(const IFramebuffer& framebuffer) {
    return static_cast<const opengl::CustomFramebuffer&>(framebuffer);
  }

  [[nodiscard]] static RenderPassDesc makeRenderPass(const Color& clearColor) {
    return {.colorAttachments = {{.loadAction = LoadAction::Clear,
                                  .storeAction = StoreAction::MsaaResolve,
                                  .clearColor = clearColor}}};
  }

  // Runs a pass of `framebuffer` that calls `encode` between its beginning and its end
  void render(const std::shared_ptr<IFramebuffer>& framebuffer,
              const RenderPassDesc& renderPass,
              const std::function<void(IRenderCommandEncoder&)>& encode = {}) const {
    Result ret;
    const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    const auto encoder = cmdBuf->createRenderCommandEncoder(renderPass, framebuffer, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(encoder, nullptr);
    if (encode) {
      encode(*encoder);
    }
    encoder->endEncoding();
    cmdQueue_->submit(*cmdBuf);
    cmdBuf->waitUntilCompleted();
  }

  // Creates the vertex buffer with the triangles of the tests, and the vertex input state that
  // reads their positions
  void createVertices() {
    Result ret;
    const BufferDesc bufferDesc{.type = BufferDesc::BufferTypeBits::Vertex,
                                .data = kVertices.data(),
                                .length = sizeof(kVertices)};
    vertexBuffer_ = iglDev_->createBuffer(bufferDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(vertexBuffer_, nullptr);
    const VertexInputStateDesc inputDesc{
        .numAttributes = 1,
        .attributes = {{.bufferIndex = 0,
                        .format = VertexAttributeFormat::Float4,
                        .offset = 0,
                        .name = "position_in",
                        .location = 0}},
        .numInputBindings = 1,
        .inputBindings = {{.stride = sizeof(float) * 4}},
    };
    vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(vertexInputState_, nullptr);
  }

  [[nodiscard]] std::shared_ptr<IRenderPipelineState> createPipeline(
      const char* fragmentShader,
      TextureFormat depthFormat = TextureFormat::Invalid) const {
    Result ret;
    auto stages = ShaderStagesCreator::fromModuleStringInput(
        *iglDev_, kVertexShader, "main", "", fragmentShader, "main", "", &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    RenderPipelineDesc::TargetDesc targetDesc;
    targetDesc.colorAttachments.resize(1);
    targetDesc.colorAttachments[0].textureFormat = TextureFormat::RGBA_UNorm8;
    targetDesc.depthAttachmentFormat = depthFormat;
    const RenderPipelineDesc desc{
        .vertexInputState = vertexInputState_,
        .shaderStages = std::move(stages),
        .targetDesc = std::move(targetDesc),
        .cullMode = CullMode::Disabled,
    };
    auto pipeline = iglDev_->createRenderPipeline(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    EXPECT_NE(pipeline, nullptr);
    return pipeline;
  }

  // Draws the triangle that starts at `firstVertex` of the vertex buffer with `pipeline`
  void drawTriangle(IRenderCommandEncoder& encoder,
                    const std::shared_ptr<IRenderPipelineState>& pipeline,
                    uint32_t firstVertex) const {
    encoder.bindRenderPipelineState(pipeline);
    encoder.bindVertexBuffer(0, *vertexBuffer_);
    encoder.draw(3, 1, firstVertex);
  }

  // The width of the storage of the renderbuffer of `texture`, which is 0 while it has no storage
  [[nodiscard]] GLint getRenderbufferWidth(const std::shared_ptr<ITexture>& texture) const {
    GLint width = -1;
    auto& renderbuffer = static_cast<opengl::Texture&>(*texture);
    renderbuffer.bind();
    iglGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width);
    renderbuffer.unbind();
    return width;
  }

  // Reads every pixel of `texture` through a framebuffer that has it as its only attachment
  [[nodiscard]] std::array<uint32_t, kSize * kSize> readPixels(
      const std::shared_ptr<ITexture>& texture) const {
    std::array<uint32_t, kSize * kSize> pixels{};
    const auto framebuffer = createFramebuffer({.colorAttachments = {{.texture = texture}}});
    if (framebuffer) {
      framebuffer->copyBytesColorAttachment(
          *cmdQueue_, 0, pixels.data(), TextureRangeDesc::new2D(0, 0, kSize, kSize));
    }
    return pixels;
  }

  void expectPixels(const std::shared_ptr<ITexture>& texture, uint32_t expected) const {
    for (const uint32_t pixel : readPixels(texture)) {
      EXPECT_EQ(pixel, expected) << std::hex << pixel;
    }
  }

  // Like expectPixels(), for a texture with `size` pixels on each side
  void expectPixels(const std::shared_ptr<ITexture>& texture,
                    uint32_t expected,
                    uint32_t size) const {
    std::vector<uint32_t> pixels(static_cast<size_t>(size) * size);
    const auto framebuffer = createFramebuffer({.colorAttachments = {{.texture = texture}}});
    ASSERT_NE(framebuffer, nullptr);
    framebuffer->copyBytesColorAttachment(
        *cmdQueue_, 0, pixels.data(), TextureRangeDesc::new2D(0, 0, size, size));
    for (const uint32_t pixel : pixels) {
      EXPECT_EQ(pixel, expected) << std::hex << pixel;
    }
  }

  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
  std::unique_ptr<IBuffer> vertexBuffer_;
  std::shared_ptr<IVertexInputState> vertexInputState_;
};

//
// ResolvesWithTheExtensionWhereItIsAvailable
//
// A device with the extension attaches the resolve texture with implicit multisampling and never
// attaches the multisampled texture. Otherwise the framebuffer has a resolve framebuffer for the
// blit that resolves.
//
TEST_F(MultisampledRenderToTextureOGLTest, ResolvesWithTheExtensionWhereItIsAvailable) {
  if (const char* reason = multisampleSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);

  const GLint renderbufferWidth = getRenderbufferWidth(color);

  if (resolvesWithExtension()) {
    EXPECT_TRUE(custom(*framebuffer).resolvesImplicitly());
    EXPECT_EQ(custom(*framebuffer).getResolveFramebuffer(), nullptr);
    // The multisampled texture is never attached, so it never gets storage
    EXPECT_EQ(renderbufferWidth, 0);

    const opengl::FramebufferBindingGuard guard(*context_);
    custom(*framebuffer).bindBuffer();
    GLint type = GL_NONE;
    GLint name = 0;
    context_->getFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
    context_->getFramebufferAttachmentParameteriv(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
    EXPECT_EQ(type, GL_TEXTURE);
    EXPECT_EQ(static_cast<GLuint>(name), static_cast<opengl::Texture&>(*resolve).getId());

    // GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT is only valid when MultiSampleExt is
    // supported (invalid enum on IMG-only drivers)
    if (context_->deviceFeatures().hasExtension(opengl::Extensions::MultiSampleExt)) {
      GLint samples = 0;
      context_->getFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,
                                                    GL_COLOR_ATTACHMENT0,
                                                    GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT,
                                                    &samples);
      EXPECT_GE(samples, static_cast<GLint>(kSamples));
    }
  } else {
    EXPECT_FALSE(custom(*framebuffer).resolvesImplicitly());
    EXPECT_NE(custom(*framebuffer).getResolveFramebuffer(), nullptr);
    EXPECT_EQ(renderbufferWidth, static_cast<GLint>(kSize));
  }
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// MemorylessRenderbufferGetsStorageWhenAttached
//
// A Memoryless renderbuffer has no storage until a framebuffer attaches it.
//
TEST_F(MultisampledRenderToTextureOGLTest, MemorylessRenderbufferGetsStorageWhenAttached) {
  const TextureDesc desc{
      .width = kSize,
      .height = kSize,
      .usage = TextureDesc::TextureUsageBits::Attachment,
      .type = TextureType::TwoD,
      .format = TextureFormat::RGBA_UNorm8,
      .storage = ResourceStorage::Memoryless,
  };
  Result ret;
  const auto texture = iglDev_->createTexture(desc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(texture, nullptr);
  EXPECT_EQ(getRenderbufferWidth(texture), 0);

  const auto framebuffer = createFramebuffer({.colorAttachments = {{.texture = texture}}});
  ASSERT_NE(framebuffer, nullptr);

  EXPECT_EQ(getRenderbufferWidth(texture), static_cast<GLint>(kSize));
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// ResolvesClearedColor
//
// A pass that clears a Memoryless multisampled attachment and resolves it fills the resolve
// texture, with and without the extension.
//
TEST_F(MultisampledRenderToTextureOGLTest, ResolvesClearedColor) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesWithExtension());

  render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));

  expectPixels(resolve, kRed);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// ResolvesTheSamplesOfAnEdge
//
// The pixels that a diagonal edge crosses are partially covered, which is only possible if the
// samples of the Memoryless attachment resolve.
//
TEST_F(MultisampledRenderToTextureOGLTest, ResolvesTheSamplesOfAnEdge) {
  if (const char* reason = drawSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);
  const auto pipeline = createPipeline(kRedFragmentShader);
  ASSERT_NE(pipeline, nullptr);

  render(
      framebuffer, makeRenderPass({0.0f, 0.0f, 0.0f, 1.0f}), [&](IRenderCommandEncoder& encoder) {
        drawTriangle(encoder, pipeline, kDiagonalFirstVertex);
      });

  size_t numRed = 0;
  size_t numBlack = 0;
  size_t numPartial = 0;
  for (const uint32_t pixel : readPixels(resolve)) {
    if (pixel == kRed) {
      ++numRed;
    } else if (pixel == kBlack) {
      ++numBlack;
    } else {
      ++numPartial;
      const uint32_t red = pixel & 0xFF;
      EXPECT_GT(red, 0u);
      EXPECT_LT(red, 0xFFu);
    }
  }
  EXPECT_GT(numRed, 0u);
  EXPECT_GT(numBlack, 0u);
  EXPECT_GT(numPartial, 0u);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// RenderbufferResolveTextureFallsBackToTheBlit
//
// A TextureTarget is a renderbuffer, which the extension cannot resolve into, so its framebuffer
// resolves with a blit, even on a device that has the extension.
//
TEST_F(MultisampledRenderToTextureOGLTest, RenderbufferResolveTextureFallsBackToTheBlit) {
  if (const char* reason = multisampleSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createRenderbufferTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);

  EXPECT_FALSE(custom(*framebuffer).resolvesImplicitly());
  EXPECT_NE(custom(*framebuffer).getResolveFramebuffer(), nullptr);

  if (context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::FramebufferBlit)) {
    render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));
    expectPixels(resolve, kRed);
  }
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// MemorylessDepthIsTested
//
// A Memoryless multisampled depth attachment makes a complete framebuffer, and fragments fail its
// depth test: the farther triangle that is drawn second leaves the nearer red one in the result.
//
TEST_F(MultisampledRenderToTextureOGLTest, MemorylessDepthIsTested) {
  if (const char* reason = drawSkipReason()) {
    GTEST_SKIP() << reason;
  }
  TextureFormat depthFormat = TextureFormat::Invalid;
  for (const TextureFormat format : {TextureFormat::Z_UNorm24, TextureFormat::Z_UNorm16}) {
    if ((iglDev_->getTextureFormatCapabilities(format) &
         ICapabilities::TextureFormatCapabilityBits::Attachment) != 0) {
      depthFormat = format;
      break;
    }
  }
  if (depthFormat == TextureFormat::Invalid) {
    GTEST_SKIP() << "No depth format for attachments";
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  const auto depth = createMemorylessTexture(depthFormat);
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  ASSERT_NE(depth, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}},
                         .depthAttachment = {.texture = depth}});
  ASSERT_NE(framebuffer, nullptr);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesWithExtension());

  const auto nearPipeline = createPipeline(kRedFragmentShader, depthFormat);
  const auto farPipeline = createPipeline(kGreenFragmentShader, depthFormat);
  ASSERT_NE(nearPipeline, nullptr);
  ASSERT_NE(farPipeline, nullptr);
  Result ret;
  const auto depthState = iglDev_->createDepthStencilState(
      {.compareFunction = CompareFunction::Less, .isDepthWriteEnabled = true}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  render(
      framebuffer, makeRenderPass({0.0f, 0.0f, 0.0f, 1.0f}), [&](IRenderCommandEncoder& encoder) {
        encoder.bindDepthStencilState(depthState);
        drawTriangle(encoder, nearPipeline, kNearFirstVertex);
        drawTriangle(encoder, farPipeline, kFarFirstVertex);
      });

  expectPixels(resolve, kRed);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// MemorylessDepthStencil
//
// A Memoryless multisampled texture with depth and stencil, attached as both, makes a complete
// framebuffer that resolves.
//
TEST_F(MultisampledRenderToTextureOGLTest, MemorylessDepthStencil) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }
  if ((iglDev_->getTextureFormatCapabilities(TextureFormat::S8_UInt_Z24_UNorm) &
       ICapabilities::TextureFormatCapabilityBits::Attachment) == 0) {
    GTEST_SKIP() << "No depth and stencil format for attachments";
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  const auto depthStencil = createMemorylessTexture(TextureFormat::S8_UInt_Z24_UNorm);
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  ASSERT_NE(depthStencil, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}},
                         .depthAttachment = {.texture = depthStencil},
                         .stencilAttachment = {.texture = depthStencil}});
  ASSERT_NE(framebuffer, nullptr);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesWithExtension());

  render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));

  expectPixels(resolve, kRed);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// ResolvesEveryColorAttachment
//
// Each resolve texture gets the color that its multisampled attachment was cleared to: implicitly
// with the second version of the extension, and with a blit on devices without the extension.
//
TEST_F(MultisampledRenderToTextureOGLTest, ResolvesEveryColorAttachment) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const bool supportsImplicitResolve =
      resolvesWithExtension() &&
      context_->deviceFeatures().hasExtension(opengl::Extensions::MultiSampleExt2);
  const bool supportsBlitResolve =
      !resolvesWithExtension() &&
      context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::FramebufferBlit);

  if (!supportsImplicitResolve && !supportsBlitResolve) {
    GTEST_SKIP() << "Neither GL_EXT_multisampled_render_to_texture2 nor framebuffer blit is "
                    "supported";
  }

  // SwiftShader's legacy blit always writes draw color attachment 0, so skip on blit path
  const auto* renderer = reinterpret_cast<const char*>(context_->getString(GL_RENDERER));
  if (supportsBlitResolve && (renderer != nullptr) &&
      std::string_view(renderer).find("SwiftShader") != std::string_view::npos) {
    GTEST_SKIP() << "SwiftShader blit path does not support multiple render targets";
  }

  if (!iglDev_->hasFeature(DeviceFeatures::MultipleRenderTargets)) {
    GTEST_SKIP() << "MultipleRenderTargets not supported";
  }

  const auto firstColor = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto secondColor = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto firstResolve = createResolveTexture();
  const auto secondResolve = createResolveTexture();
  ASSERT_NE(firstColor, nullptr);
  ASSERT_NE(secondColor, nullptr);
  ASSERT_NE(firstResolve, nullptr);
  ASSERT_NE(secondResolve, nullptr);
  const auto framebuffer = createFramebuffer(
      {.colorAttachments = {{.texture = firstColor, .resolveTexture = firstResolve},
                            {.texture = secondColor, .resolveTexture = secondResolve}}});
  ASSERT_NE(framebuffer, nullptr);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), supportsImplicitResolve);

  const RenderPassDesc renderPass{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                        .storeAction = StoreAction::MsaaResolve,
                                                        .clearColor = {1.0f, 0.0f, 0.0f, 1.0f}},
                                                       {.loadAction = LoadAction::Clear,
                                                        .storeAction = StoreAction::MsaaResolve,
                                                        .clearColor = {0.0f, 1.0f, 0.0f, 1.0f}}}};
  render(framebuffer, renderPass);

  expectPixels(firstResolve, kRed);
  expectPixels(secondResolve, kGreen);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// DontCareStoreOnImplicitFramebuffer
//
// A pass with StoreAction::DontCare on an implicit framebuffer renders without error. The
// framebuffer remains usable for subsequent passes, which produce the expected color.
//
TEST_F(MultisampledRenderToTextureOGLTest, DontCareStoreOnImplicitFramebuffer) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesWithExtension());

  // First pass with DontCare store: the resolve contents are undefined, so we do NOT assert pixels
  const RenderPassDesc dontCarePass{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                          .storeAction = StoreAction::DontCare,
                                                          .clearColor = {1.0f, 0.0f, 0.0f, 1.0f}}}};
  render(framebuffer, dontCarePass);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  // Second pass with Clear + MsaaResolve: should produce expected color, proving framebuffer is
  // usable
  render(framebuffer, makeRenderPass({0.0f, 1.0f, 0.0f, 1.0f}));
  expectPixels(resolve, kGreen);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// MultisampledAttachmentWithoutResolveTextureStillRenders
//
// A Memoryless multisampled attachment without a resolve texture has nothing to resolve into, so it
// gets regular storage, and its framebuffer is complete and renders without a resolve framebuffer.
//
TEST_F(MultisampledRenderToTextureOGLTest,
       MultisampledAttachmentWithoutResolveTextureStillRenders) {
  if (const char* reason = multisampleSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  ASSERT_NE(color, nullptr);
  const auto framebuffer = createFramebuffer({.colorAttachments = {{.texture = color}}});
  ASSERT_NE(framebuffer, nullptr);

  EXPECT_FALSE(custom(*framebuffer).resolvesImplicitly());
  EXPECT_EQ(custom(*framebuffer).getResolveFramebuffer(), nullptr);
  const RenderPassDesc renderPass{.colorAttachments = {{.loadAction = LoadAction::Clear,
                                                        .storeAction = StoreAction::DontCare,
                                                        .clearColor = {1.0f, 0.0f, 0.0f, 1.0f}}}};
  render(framebuffer, renderPass);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// UpdateResolveAttachmentAttachesTheNewTexture
//
// A pass resolves into the resolve texture that updateResolveAttachment() set, and the framebuffer
// reports that texture.
//
TEST_F(MultisampledRenderToTextureOGLTest, UpdateResolveAttachmentAttachesTheNewTexture) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto firstResolve = createResolveTexture();
  const auto secondResolve = createResolveTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(firstResolve, nullptr);
  ASSERT_NE(secondResolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = firstResolve}}});
  ASSERT_NE(framebuffer, nullptr);
  const bool resolvesImplicitly = custom(*framebuffer).resolvesImplicitly();
  EXPECT_EQ(framebuffer->getResolveColorAttachment(0), firstResolve);

  render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));
  framebuffer->updateResolveAttachment(secondResolve);
  EXPECT_EQ(framebuffer->getResolveColorAttachment(0), secondResolve);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);
  render(framebuffer, makeRenderPass({0.0f, 1.0f, 0.0f, 1.0f}));

  expectPixels(firstResolve, kRed);
  expectPixels(secondResolve, kGreen);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// UpdateResolveAttachmentEveryFrame
//
// A framebuffer with depth that cycles through resolve textures, as one does for the images of a
// swapchain, resolves each frame into the texture of that frame.
//
TEST_F(MultisampledRenderToTextureOGLTest, UpdateResolveAttachmentEveryFrame) {
  if (const char* reason = drawSkipReason()) {
    GTEST_SKIP() << reason;
  }
  if ((iglDev_->getTextureFormatCapabilities(TextureFormat::Z_UNorm16) &
       ICapabilities::TextureFormatCapabilityBits::Attachment) == 0) {
    GTEST_SKIP() << "No depth format for attachments";
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto depth = createMemorylessTexture(TextureFormat::Z_UNorm16);
  const std::array<std::shared_ptr<ITexture>, 3> resolves = {
      createResolveTexture(), createResolveTexture(), createResolveTexture()};
  ASSERT_NE(color, nullptr);
  ASSERT_NE(depth, nullptr);
  for (const auto& resolve : resolves) {
    ASSERT_NE(resolve, nullptr);
  }
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolves[0]}},
                         .depthAttachment = {.texture = depth}});
  ASSERT_NE(framebuffer, nullptr);
  const bool resolvesImplicitly = custom(*framebuffer).resolvesImplicitly();
  const auto redPipeline = createPipeline(kRedFragmentShader, TextureFormat::Z_UNorm16);
  const auto greenPipeline = createPipeline(kGreenFragmentShader, TextureFormat::Z_UNorm16);
  ASSERT_NE(redPipeline, nullptr);
  ASSERT_NE(greenPipeline, nullptr);
  Result ret;
  const auto depthState = iglDev_->createDepthStencilState(
      {.compareFunction = CompareFunction::Less, .isDepthWriteEnabled = true}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  // Even frames fill the framebuffer with red, odd ones with green
  constexpr uint32_t kNumFrames = 7;
  for (uint32_t frame = 0; frame != kNumFrames; ++frame) {
    framebuffer->updateResolveAttachment(resolves[frame % resolves.size()]);
    EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);
    render(
        framebuffer, makeRenderPass({0.0f, 0.0f, 0.0f, 1.0f}), [&](IRenderCommandEncoder& encoder) {
          encoder.bindDepthStencilState(depthState);
          drawTriangle(encoder, frame % 2 == 0 ? redPipeline : greenPipeline, kNearFirstVertex);
        });
  }

  // The last frame to resolve into each texture is frame 6 (red), 4 (red) and 5 (green)
  expectPixels(resolves[0], kRed);
  expectPixels(resolves[1], kRed);
  expectPixels(resolves[2], kGreen);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// FramebuffersShareMultisampledAttachments
//
// Framebuffers with the same Memoryless multisampled color and depth textures but different resolve
// textures each resolve into their own.
//
TEST_F(MultisampledRenderToTextureOGLTest, FramebuffersShareMultisampledAttachments) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }
  if ((iglDev_->getTextureFormatCapabilities(TextureFormat::Z_UNorm16) &
       ICapabilities::TextureFormatCapabilityBits::Attachment) == 0) {
    GTEST_SKIP() << "No depth format for attachments";
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto depth = createMemorylessTexture(TextureFormat::Z_UNorm16);
  const auto firstResolve = createResolveTexture();
  const auto secondResolve = createResolveTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(depth, nullptr);
  ASSERT_NE(firstResolve, nullptr);
  ASSERT_NE(secondResolve, nullptr);
  const auto firstFramebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = firstResolve}},
                         .depthAttachment = {.texture = depth}});
  const auto secondFramebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = secondResolve}},
                         .depthAttachment = {.texture = depth}});
  ASSERT_NE(firstFramebuffer, nullptr);
  ASSERT_NE(secondFramebuffer, nullptr);

  render(firstFramebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));
  render(secondFramebuffer, makeRenderPass({0.0f, 1.0f, 0.0f, 1.0f}));
  render(firstFramebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));

  expectPixels(firstResolve, kRed);
  expectPixels(secondResolve, kGreen);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// PassesThatFollowEachOtherEndUnbound
//
// Passes of one command buffer that resolve implicitly into the same texture each end with no
// framebuffer bound, and the texture holds the result of the last one.
//
TEST_F(MultisampledRenderToTextureOGLTest, PassesThatFollowEachOtherEndUnbound) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesWithExtension());

  Result ret;
  const auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const std::array<Color, 3> clearColors = {
      Color{1.0f, 0.0f, 0.0f, 1.0f}, Color{0.0f, 0.0f, 0.0f, 1.0f}, Color{0.0f, 1.0f, 0.0f, 1.0f}};
  for (const Color& clearColor : clearColors) {
    const auto encoder =
        cmdBuf->createRenderCommandEncoder(makeRenderPass(clearColor), framebuffer, &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
    ASSERT_NE(encoder, nullptr);
    encoder->endEncoding();
    if (resolvesWithExtension()) {
      GLint binding = -1;
      context_->getIntegerv(GL_FRAMEBUFFER_BINDING, &binding);
      EXPECT_EQ(binding, 0);
    }
  }
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  expectPixels(resolve, kGreen);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// UpdateResolveAttachmentToRenderbufferFallsBackToTheBlit
//
// A renderbuffer cannot be the resolve texture of the extension, so the framebuffer then resolves
// with a blit.
//
TEST_F(MultisampledRenderToTextureOGLTest,
       UpdateResolveAttachmentToRenderbufferFallsBackToTheBlit) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto firstResolve = createResolveTexture();
  const auto renderbufferResolve = createRenderbufferTexture();
  ASSERT_NE(color, nullptr);
  ASSERT_NE(firstResolve, nullptr);
  ASSERT_NE(renderbufferResolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = firstResolve}}});
  ASSERT_NE(framebuffer, nullptr);

  framebuffer->updateResolveAttachment(renderbufferResolve);

  EXPECT_EQ(framebuffer->getResolveColorAttachment(0), renderbufferResolve);
  EXPECT_FALSE(custom(*framebuffer).resolvesImplicitly());
  EXPECT_NE(custom(*framebuffer).getResolveFramebuffer(), nullptr);
  if (context_->deviceFeatures().hasInternalFeature(opengl::InternalFeatures::FramebufferBlit)) {
    render(framebuffer, makeRenderPass({0.0f, 1.0f, 0.0f, 1.0f}));
    expectPixels(renderbufferResolve, kGreen);
  }
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// UpdateDrawableReplacesTheMultisampledTexture
//
// The framebuffer reports the new multisampled texture and still resolves, with and without the
// extension, as a Memoryless texture of the same size and format keeps the framebuffer eligible.
//
TEST_F(MultisampledRenderToTextureOGLTest, UpdateDrawableReplacesTheMultisampledTexture) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }

  const auto firstColor = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto secondColor = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  ASSERT_NE(firstColor, nullptr);
  ASSERT_NE(secondColor, nullptr);
  ASSERT_NE(resolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = firstColor, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);
  const bool resolvesImplicitly = custom(*framebuffer).resolvesImplicitly();

  framebuffer->updateDrawable(secondColor);

  EXPECT_EQ(framebuffer->getColorAttachment(0), secondColor);
  EXPECT_EQ(framebuffer->getResolveColorAttachment(0), resolve);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);
  render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));
  expectPixels(resolve, kRed);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// UpdateDrawableAddsMemorylessDepth
//
// Surface textures with a Memoryless multisampled depth texture add the depth attachment to a
// framebuffer that had none, and a depth-tested draw then uses it.
//
TEST_F(MultisampledRenderToTextureOGLTest, UpdateDrawableAddsMemorylessDepth) {
  if (const char* reason = drawSkipReason()) {
    GTEST_SKIP() << reason;
  }
  if ((iglDev_->getTextureFormatCapabilities(TextureFormat::Z_UNorm16) &
       ICapabilities::TextureFormatCapabilityBits::Attachment) == 0) {
    GTEST_SKIP() << "No depth format for attachments";
  }

  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto resolve = createResolveTexture();
  const auto depth = createMemorylessTexture(TextureFormat::Z_UNorm16);
  ASSERT_NE(color, nullptr);
  ASSERT_NE(resolve, nullptr);
  ASSERT_NE(depth, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = resolve}}});
  ASSERT_NE(framebuffer, nullptr);
  const bool resolvesImplicitly = custom(*framebuffer).resolvesImplicitly();

  framebuffer->updateDrawable(SurfaceTextures{.color = color, .depth = depth});

  EXPECT_EQ(framebuffer->getDepthAttachment(), depth);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);

  const auto nearPipeline = createPipeline(kRedFragmentShader, TextureFormat::Z_UNorm16);
  const auto farPipeline = createPipeline(kGreenFragmentShader, TextureFormat::Z_UNorm16);
  ASSERT_NE(nearPipeline, nullptr);
  ASSERT_NE(farPipeline, nullptr);
  Result ret;
  const auto depthState = iglDev_->createDepthStencilState(
      {.compareFunction = CompareFunction::Less, .isDepthWriteEnabled = true}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;

  render(
      framebuffer, makeRenderPass({0.0f, 0.0f, 0.0f, 1.0f}), [&](IRenderCommandEncoder& encoder) {
        encoder.bindDepthStencilState(depthState);
        drawTriangle(encoder, nearPipeline, kNearFirstVertex);
        drawTriangle(encoder, farPipeline, kFarFirstVertex);
      });

  expectPixels(resolve, kRed);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// UpdatesRecoverFromNullTexturesAndResizes
//
// A null texture only removes that attachment, so a framebuffer that updates to null resolve or
// color textures and then to textures again resolves into the textures that it has after. That
// includes a resize that sets the multisampled and the resolve textures one after the other, in
// both orders, which ends with the framebuffer resolving the way that it began.
//
TEST_F(MultisampledRenderToTextureOGLTest, UpdatesRecoverFromNullTexturesAndResizes) {
  if (const char* reason = resolveSkipReason()) {
    GTEST_SKIP() << reason;
  }
  // The depth texture goes with the multisampled color texture, where there is a format for it
  const bool hasDepth = (iglDev_->getTextureFormatCapabilities(TextureFormat::Z_UNorm16) &
                         ICapabilities::TextureFormatCapabilityBits::Attachment) != 0;
  const auto createDepth = [&](uint32_t size) {
    return hasDepth ? createMemorylessTexture(TextureFormat::Z_UNorm16, size) : nullptr;
  };

  constexpr uint32_t kLargeSize = 2 * kSize;
  const auto color = createMemorylessTexture(TextureFormat::RGBA_UNorm8);
  const auto depth = createDepth(kSize);
  const auto firstResolve = createResolveTexture();
  const auto secondResolve = createResolveTexture();
  const auto largeColor = createMemorylessTexture(TextureFormat::RGBA_UNorm8, kLargeSize);
  const auto largeDepth = createDepth(kLargeSize);
  const auto largeResolve = createResolveTexture(kLargeSize);
  ASSERT_NE(color, nullptr);
  ASSERT_NE(firstResolve, nullptr);
  ASSERT_NE(secondResolve, nullptr);
  ASSERT_NE(largeColor, nullptr);
  ASSERT_NE(largeResolve, nullptr);
  const auto framebuffer =
      createFramebuffer({.colorAttachments = {{.texture = color, .resolveTexture = firstResolve}},
                         .depthAttachment = {.texture = depth}});
  ASSERT_NE(framebuffer, nullptr);
  const bool resolvesImplicitly = custom(*framebuffer).resolvesImplicitly();
  EXPECT_EQ(resolvesImplicitly, resolvesWithExtension());

  // Without a resolve texture there is nothing to resolve into, and a texture that follows gets it
  framebuffer->updateResolveAttachment(nullptr);
  EXPECT_EQ(framebuffer->getResolveColorAttachment(0), nullptr);
  EXPECT_FALSE(custom(*framebuffer).resolvesImplicitly());
  EXPECT_EQ(custom(*framebuffer).getResolveFramebuffer(), nullptr);
  framebuffer->updateResolveAttachment(secondResolve);
  EXPECT_EQ(framebuffer->getResolveColorAttachment(0), secondResolve);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);
  EXPECT_EQ(custom(*framebuffer).getResolveFramebuffer() != nullptr, !resolvesImplicitly);
  render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));
  expectPixels(secondResolve, kRed);

  // Without a color texture the resolve texture stays for the one that follows
  framebuffer->updateDrawable(std::shared_ptr<ITexture>());
  EXPECT_EQ(framebuffer->getColorAttachment(0), nullptr);
  EXPECT_EQ(framebuffer->getResolveColorAttachment(0), secondResolve);
  framebuffer->updateDrawable(color);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);
  render(framebuffer, makeRenderPass({0.0f, 1.0f, 0.0f, 1.0f}));
  expectPixels(secondResolve, kGreen);

  // A resize that sets the multisampled textures first resolves with a blit while the resolve
  // texture has the old size, and implicitly again once the new one follows
  framebuffer->updateDrawable(SurfaceTextures{.color = largeColor, .depth = largeDepth});
  EXPECT_FALSE(custom(*framebuffer).resolvesImplicitly());
  framebuffer->updateResolveAttachment(largeResolve);
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);
  render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));
  expectPixels(largeResolve, kRed, kLargeSize);

  // A resize back that sets the resolve texture first
  framebuffer->updateResolveAttachment(secondResolve);
  EXPECT_FALSE(custom(*framebuffer).resolvesImplicitly());
  framebuffer->updateDrawable(SurfaceTextures{.color = color, .depth = depth});
  EXPECT_EQ(custom(*framebuffer).resolvesImplicitly(), resolvesImplicitly);
  render(framebuffer, makeRenderPass({1.0f, 0.0f, 0.0f, 1.0f}));
  expectPixels(secondResolve, kRed);
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
}

//
// KeepsTheFeaturesOfTheBlitPath
//
// Callers branch on these device features, and a device that resolves implicitly keeps reporting
// what it did before, as it still creates the multisampled attachments.
//
TEST_F(MultisampledRenderToTextureOGLTest, KeepsTheFeaturesOfTheBlitPath) {
  EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MultiSampleResolve));
  EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MemorylessAttachments));

  const auto& features = context_->deviceFeatures();
  if (features.hasInternalFeature(opengl::InternalFeatures::MultisampledRenderToTexture)) {
    EXPECT_TRUE(opengl::DeviceFeatureSet::usesOpenGLES());
    EXPECT_TRUE(features.hasExtension(opengl::Extensions::MultiSampleExt) ||
                features.hasExtension(opengl::Extensions::MultiSampleImg));
  }
}

//
// ExtensionIsMatchedAsAWholeName
//
// GL_EXT_multisampled_render_to_texture is a prefix of GL_EXT_multisampled_render_to_texture2, so
// extensions have to be matched as whole names. It is also available with the IMG version.
//
TEST_F(MultisampledRenderToTextureOGLTest, ExtensionIsMatchedAsAWholeName) {
  const auto hasExtension = [this](std::string extensions,
                                   std::unordered_set<std::string> extensionSet,
                                   opengl::Extensions extension) {
    opengl::DeviceFeatureSet features(*context_);
    features.initializeExtensions(std::move(extensions), std::move(extensionSet));
    return features.hasExtension(extension);
  };
  const auto hasFeature = [this](std::string extensions) {
    opengl::DeviceFeatureSet features(*context_);
    features.initializeExtensions(std::move(extensions), {});
    return features.hasInternalFeature(opengl::InternalFeatures::MultisampledRenderToTexture);
  };
  const bool es = opengl::DeviceFeatureSet::usesOpenGLES();
  const std::string first = "GL_EXT_multisampled_render_to_texture";
  const std::string second = "GL_EXT_multisampled_render_to_texture2";
  const std::string img = "GL_IMG_multisampled_render_to_texture";

  // The string that OpenGL (ES) below 3.0 reports
  EXPECT_FALSE(hasExtension(second, {}, opengl::Extensions::MultiSampleExt));
  EXPECT_EQ(hasExtension(second, {}, opengl::Extensions::MultiSampleExt2), es);
  EXPECT_EQ(hasExtension(first, {}, opengl::Extensions::MultiSampleExt), es);
  EXPECT_EQ(hasExtension(second + " " + first, {}, opengl::Extensions::MultiSampleExt), es);
  // The set of names that it reports from 3.0
  EXPECT_FALSE(hasExtension("", {second}, opengl::Extensions::MultiSampleExt));
  EXPECT_EQ(hasExtension("", {first, second}, opengl::Extensions::MultiSampleExt), es);

  // The feature needs the first version or the IMG version, on a GPU that renders correctly with it
  const auto* renderer = reinterpret_cast<const char*>(context_->getString(GL_RENDERER));
  const auto* vendor = reinterpret_cast<const char*>(context_->getString(GL_VENDOR));
  const bool works = es && !opengl::hasBrokenMultisampledRenderToTexture(renderer, vendor);
  EXPECT_FALSE(hasFeature(second));
  EXPECT_FALSE(hasFeature("GL_OES_depth24"));
  EXPECT_EQ(hasFeature(first), works);
  EXPECT_EQ(hasFeature(img), works);
}

} // namespace igl::tests
