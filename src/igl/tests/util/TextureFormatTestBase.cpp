/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "TextureFormatTestBase.h"

#include "../data/ShaderData.h"
#include "../data/VertexIndexData.h"
#include "Common.h"

#include <igl/CommandBuffer.h>
#include <igl/Framebuffer.h>
#include <igl/SamplerState.h>
#include <igl/VertexInputState.h>

namespace igl::tests::util {

#define OFFSCREEN_TEX_WIDTH 2
#define OFFSCREEN_TEX_HEIGHT 2

void TextureFormatTestBase::SetUp() {
  setDebugBreakEnabled(false);

  util::createDeviceAndQueue(iglDev_, cmdQueue_);
  ASSERT_TRUE(iglDev_ != nullptr);
  ASSERT_TRUE(cmdQueue_ != nullptr);

  // Create a sampled and an attachment texture for use in tests
  TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                           OFFSCREEN_TEX_WIDTH,
                                           OFFSCREEN_TEX_HEIGHT,
                                           TextureDesc::TextureUsageBits::Sampled);
  texDesc.debugName = "TextureFormatTestBase rgba unorm8 sampled";

  Result ret;
  sampledTexture_ = iglDev_->createTexture(texDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok) << ret.message;
  ASSERT_TRUE(sampledTexture_ != nullptr);

  const auto attachmentFormat = iglDev_->getBackendType() == BackendType::OpenGL
                                    ? TextureFormat::ABGR_UNorm4
                                    : TextureFormat::RGBA_UNorm8;
  texDesc = TextureDesc::new2D(attachmentFormat,
                               OFFSCREEN_TEX_WIDTH,
                               OFFSCREEN_TEX_HEIGHT,
                               TextureDesc::TextureUsageBits::Attachment);
  attachmentTexture_ = iglDev_->createTexture(texDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok) << ret.message;
  ASSERT_TRUE(attachmentTexture_ != nullptr);

  // Initialize render pass descriptor
  renderPass_ = {
      .colorAttachments = {{.loadAction = LoadAction::Clear,
                            .storeAction = StoreAction::Store,
                            .clearColor = {0.0, 0.0, 0.0, 1.0}}},
  };

  // Initialize input to vertex shader
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

  vertexInputState_ = iglDev_->createVertexInputState(inputDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(vertexInputState_ != nullptr);

  // Initialize index buffer
  BufferDesc bufDesc;

  bufDesc.type = BufferDesc::BufferTypeBits::Index;
  bufDesc.data = data::vertex_index::kQuadInd.data();
  bufDesc.length = sizeof(data::vertex_index::kQuadInd);

  ib_ = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(ib_ != nullptr);

  // Initialize vertex and sampler buffers
  bufDesc.type = BufferDesc::BufferTypeBits::Vertex;
  bufDesc.data = data::vertex_index::kQuadVert.data();
  bufDesc.length = sizeof(data::vertex_index::kQuadVert);

  vb_ = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(vb_ != nullptr);

  bufDesc.type = BufferDesc::BufferTypeBits::Vertex;
  bufDesc.data = data::vertex_index::kQuadUv.data();
  bufDesc.length = sizeof(data::vertex_index::kQuadUv);

  uv_ = iglDev_->createBuffer(bufDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(uv_ != nullptr);

  // Initialize sampler state
  nearestSampler_ = iglDev_->createSamplerState(SamplerStateDesc{}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(nearestSampler_ != nullptr);

  linearSampler_ = iglDev_->createSamplerState(SamplerStateDesc::newLinear(), &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(linearSampler_ != nullptr);

  // Initialize Graphics Pipeline Descriptor, but leave the creation
  // to the individual tests in case further customization is required
  renderPipelineDesc_.vertexInputState = vertexInputState_;
  renderPipelineDesc_.targetDesc.colorAttachments.resize(1);
  renderPipelineDesc_.fragmentUnitSamplerMap[textureUnit_] =
      IGL_NAMEHANDLE(data::shader::kSimpleSampler);
  renderPipelineDesc_.cullMode = CullMode::Disabled;
}

std::shared_ptr<IFramebuffer> TextureFormatTestBase::createFramebuffer(
    std::shared_ptr<ITexture> attachmentTexture,
    Result& ret) {
  FramebufferDesc framebufferDesc;
  const auto attachmentFormat = attachmentTexture->getFormat();

  if (attachmentTexture->getProperties().isDepthOrStencil()) {
    // For depth/stencil textures:
    //  1) Attach to the appropriate part of the framebuffer
    //  2) Use ivar as color attachment.

    // Update renderPipelineDesc_
    renderPipelineDesc_.targetDesc.colorAttachments[0].textureFormat =
        attachmentTexture_->getFormat();
    renderPipelineDesc_.targetDesc.depthAttachmentFormat = TextureFormat::Invalid;
    renderPipelineDesc_.targetDesc.stencilAttachmentFormat = TextureFormat::Invalid;

    if (attachmentFormat != TextureFormat::S_UInt8) {
      framebufferDesc.depthAttachment.texture = attachmentTexture;
      renderPipelineDesc_.targetDesc.depthAttachmentFormat = attachmentFormat;
    }
    if (attachmentFormat != TextureFormat::Z_UNorm16 &&
        attachmentFormat != TextureFormat::Z_UNorm24 &&
        attachmentFormat != TextureFormat::Z_UNorm32) {
      framebufferDesc.stencilAttachment.texture = attachmentTexture;
      renderPipelineDesc_.targetDesc.stencilAttachmentFormat = attachmentFormat;
    }
    framebufferDesc.colorAttachments[0].texture = attachmentTexture_;

    std::unique_ptr<IShaderStages> stages;
    igl::tests::util::createSimpleShaderStages(iglDev_, stages, attachmentTexture_->getFormat());
    renderPipelineDesc_.shaderStages = std::move(stages);

  } else {
    framebufferDesc.colorAttachments[0].texture = attachmentTexture;

    // Update renderPipelineDesc_
    renderPipelineDesc_.targetDesc.colorAttachments[0].textureFormat = attachmentFormat;
    renderPipelineDesc_.targetDesc.depthAttachmentFormat = TextureFormat::Invalid;
    renderPipelineDesc_.targetDesc.stencilAttachmentFormat = TextureFormat::Invalid;

    std::unique_ptr<IShaderStages> stages;
    igl::tests::util::createSimpleShaderStages(iglDev_, stages, attachmentFormat);
    renderPipelineDesc_.shaderStages = std::move(stages);
  }

  auto framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
  return framebuffer;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void TextureFormatTestBase::render(std::shared_ptr<ITexture> sampledTexture,
                                   std::shared_ptr<ITexture> attachmentTexture,
                                   bool linearSampling,
                                   TextureFormatProperties testProperties) {
  Result ret;

  //-------
  // Render
  //-------
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok) << ret.message;
  ASSERT_TRUE(cmdBuf != nullptr);

  auto framebuffer = createFramebuffer(attachmentTexture, ret);
  ASSERT_TRUE(ret.isOk()) << testProperties.name << ": " << ret.message;
  ASSERT_TRUE(framebuffer != nullptr);

  // Add sampled textures as dependencies so that their layout is transitioned correctly for Vulkan
  Dependencies dep;
  dep.textures[0] = sampledTexture.get();

  Result result;
  auto cmds = cmdBuf->createRenderCommandEncoder(renderPass_, framebuffer, dep, &result);
  ASSERT_TRUE(result.isOk());
  cmds->bindVertexBuffer(data::shader::kSimplePosIndex, *vb_);
  cmds->bindVertexBuffer(data::shader::kSimpleUvIndex, *uv_);

  // Create createFramebuffer fills in proper texture formats and shader stages in
  // renderPipelineDesc_

  auto pipelineState = iglDev_->createRenderPipeline(renderPipelineDesc_, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok) << ret.message;
  ASSERT_TRUE(pipelineState != nullptr);

  cmds->bindRenderPipelineState(pipelineState);

  cmds->bindTexture(textureUnit_, BindTarget::kFragment, sampledTexture.get());
  // Choose appropriate sampler.
  cmds->bindSamplerState(textureUnit_,
                         BindTarget::kFragment,
                         (linearSampling ? linearSampler_ : nearestSampler_).get());

  cmds->bindIndexBuffer(*ib_, IndexFormat::UInt16);
  cmds->drawIndexed(6);

  cmds->endEncoding();

  cmdQueue_->submit(*cmdBuf);

  cmdBuf->waitUntilCompleted();
}

std::pair<TextureFormat, bool> TextureFormatTestBase::checkSupport(
    TextureFormat format,
    TextureDesc::TextureUsage usage) {
  const bool sampled = (usage & TextureDesc::TextureUsageBits::Sampled) != 0;
  const bool storage = (usage & TextureDesc::TextureUsageBits::Storage) != 0;
  const bool attachment = (usage & TextureDesc::TextureUsageBits::Attachment) != 0;
  const bool sampledAttachment = sampled && attachment;

  const auto capabilities = iglDev_->getTextureFormatCapabilities(format);
  bool supported = false;
  if (sampledAttachment) {
    if ((capabilities & ICapabilities::TextureFormatCapabilityBits::SampledAttachment) != 0) {
      supported = true;
    }
  } else if (attachment &&
             (capabilities & ICapabilities::TextureFormatCapabilityBits::Attachment) != 0) {
    supported = true;
  } else if (sampled && (capabilities & ICapabilities::TextureFormatCapabilityBits::Sampled) != 0) {
    supported = true;
  } else if (storage && (capabilities & ICapabilities::TextureFormatCapabilityBits::Storage) != 0) {
    supported = true;
  }

  return std::make_pair(format, supported);
}

std::vector<std::pair<TextureFormat, bool>> TextureFormatTestBase::getFormatSupport(
    TextureDesc::TextureUsage usage) {
  std::vector<std::pair<TextureFormat, bool>> formatSupport;
  formatSupport.emplace_back(checkSupport(TextureFormat::Invalid, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::A_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::L_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_F16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_UInt16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_UNorm16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::B5G5R5A1_UNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::B5G6R5_UNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::ABGR_UNorm4, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::LA_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RG_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R4G2B2_UNorm_Apple, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R4G2B2_UNorm_Rev_Apple, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R5G5B5A1_UNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBX_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::BGRA_UNorm8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::BGRA_UNorm8_Rev, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_SRGB, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::BGRA_SRGB, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RG_F16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RG_UInt16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RG_UNorm16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB10_A2_UNorm_Rev, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB10_A2_Uint_Rev, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::BGR10_A2_Unorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::B10G11R11_UFloat, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_F32, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_UInt32, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB_F16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_F16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RG_F32, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB_F32, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_UInt32, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_F32, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_4x4, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_4x4, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_5x4, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_5x4, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_5x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_5x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_6x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_6x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_6x6, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_6x6, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_8x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_8x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_8x6, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_8x6, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_8x8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_8x8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_10x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_10x5, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_10x6, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_10x6, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_10x8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_10x8, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_10x10, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_10x10, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_12x10, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_12x10, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_ASTC_12x12, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_ASTC_12x12, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_PVRTC_2BPPV1, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB_PVRTC_2BPPV1, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_PVRTC_4BPPV1, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB_PVRTC_4BPPV1, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB8_ETC1, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB8_ETC2, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_ETC2, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGB8_Punchthrough_A1_ETC2, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_Punchthrough_A1_ETC2, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA8_EAC_ETC2, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::SRGB8_A8_EAC_ETC2, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RG_EAC_UNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RG_EAC_SNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_EAC_UNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::R_EAC_SNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_BC7_UNORM_4x4, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::RGBA_BC7_SRGB_4x4, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::Z_UNorm16, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::Z_UNorm24, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::Z_UNorm32, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::S8_UInt_Z24_UNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::S8_UInt_Z32_UNorm, usage));
  formatSupport.emplace_back(checkSupport(TextureFormat::S_UInt8, usage));
  return formatSupport;
}

// Attempts to upload data to texture.
void TextureFormatTestBase::testUpload(std::shared_ptr<ITexture> texture) {
  const auto size = texture->getEstimatedSizeInBytes();
  std::vector<uint8_t> data(size);
  const auto range = texture->getFullRange();
  const Result result = texture->upload(range, data.data());
  ASSERT_TRUE(result.isOk()) << texture->getProperties().name;
  // flush upload
  Result ret;
  auto cmdBuf = cmdQueue_->createCommandBuffer({}, &ret);
  cmdQueue_->submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();
}

// Attempts to render into texture.
void TextureFormatTestBase::testAttachment(std::shared_ptr<ITexture> texture) {
  render(sampledTexture_, texture, false, texture->getProperties());
}

// Attempts to sample from texture when rendering.
void TextureFormatTestBase::testSampled(std::shared_ptr<ITexture> texture, bool linearSampling) {
  render(texture, attachmentTexture_, linearSampling, texture->getProperties());
}

void TextureFormatTestBase::testUsage(TextureDesc::TextureUsage usage, const char* usageName) {
  const auto formatSupport = getFormatSupport(usage);
  for (const auto& fs : formatSupport) {
    testUsage(fs, usage, usageName);
  }
}

void TextureFormatTestBase::testUsage(std::pair<TextureFormat, bool> formatSupport,
                                      TextureDesc::TextureUsage usage,
                                      const char* usageName) {
  std::shared_ptr<ITexture> texture;
  Result ret;
  const auto [textureFormat, supported] = formatSupport;
  const auto properties = TextureFormatProperties::fromTextureFormat(textureFormat);
  if (!supported) {
    // Comment this out to test unsupported formats.
    IGL_LOG_INFO("%s: Skipping %s: Capabilities: 0x%x\n",
                 usageName,
                 properties.name,
                 iglDev_->getTextureFormatCapabilities(textureFormat));
    return;
  }
  IGL_LOG_INFO("%s: Testing %s\n", usageName, properties.name);
  auto texDesc =
      TextureDesc::new2D(textureFormat, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT, usage);
  texDesc.debugName = std::string("TextureFormatTestBase:") + usageName + ":" + properties.name;
  if (iglDev_->getBackendType() == BackendType::Metal && properties.isDepthOrStencil()) {
    texDesc.storage = ResourceStorage::Private;
  }
  texture = iglDev_->createTexture(texDesc, &ret);
  ASSERT_EQ(ret.code, supported ? Result::Code::Ok : Result::Code::ArgumentInvalid)
      << properties.name << ": " << ret.message;

  testUsage(texture, usage, usageName);
}
void TextureFormatTestBase::testUsage(std::shared_ptr<ITexture> texture,
                                      TextureDesc::TextureUsage usage,
                                      const char* usageName) {
  const auto properties = texture->getProperties();
  // Non-normalized integer formats cannot be sampled with `float` GLSL samplers `sampler2D` on
  // Vulkan (need `usampler2D` etc)
  const bool isIntegerFormat = (properties.flags & TextureFormatProperties::Flags::Integer) != 0;
  const bool isVulkan = iglDev_->getBackendType() == BackendType::Vulkan;
  // WGSL cannot bind integer formats to `texture_2d<f32>` declarations either.
  const bool isWgsl = iglDev_->getShaderVersion().family == ShaderFamily::Wgsl;
  const bool shouldSkip = (isVulkan || isWgsl) && isIntegerFormat;
  if (!shouldSkip && (usage & TextureDesc::TextureUsageBits::Sampled) != 0) {
    const bool linearSampling =
        (iglDev_->getTextureFormatCapabilities(properties.format) &
         ICapabilities::TextureFormatCapabilityBits::SampledAttachment) != 0;
    IGL_LOG_INFO("%s: Test Sampled: %s\n", usageName, properties.name);
    testSampled(texture, linearSampling);
  }

  if ((usage & (TextureDesc::TextureUsageBits::Attachment)) != 0) {
    IGL_LOG_INFO("%s: Test Attachment: %s\n", usageName, properties.name);
    testAttachment(texture);
  }

  if (texture->supportsUpload()) {
    IGL_LOG_INFO("%s: Test Upload: %s\n", usageName, properties.name);
    testUpload(texture);
  }
}

} // namespace igl::tests::util
