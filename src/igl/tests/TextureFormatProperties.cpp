/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "data/ShaderData.h"
#include "data/VertexIndexData.h"
#include "util/Common.h"

#include <cstddef>
#include <string>
#include <igl/CommandBuffer.h>
#include <igl/Framebuffer.h>
#include <igl/NameHandle.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/SamplerState.h>
#include <igl/VertexInputState.h>

namespace igl::tests {

// Picking this just to match the texture we will use. If you use a different
// size texture, then you will have to either create a new offscreenTexture_
// and the framebuffer object in your test, so know exactly what the end result
// would be after sampling
constexpr uint32_t kOffscreenTexWidth = 2u;
constexpr uint32_t kOffscreenTexHeight = 2u;

//
// TextureTest
//
// Test fixture for all the tests in this file. Takes care of common
// initialization and allocating of common resources.
//
class TextureTest : public ::testing::Test {
 private:
 public:
  TextureTest() = default;

  //
  // SetUp()
  //
  // This function sets up a render pass and a graphics pipeline descriptor
  // so it is ready to render a simple quad with an input texture to an
  // offscreen texture.
  //
  // The actual creation of the graphics pipeline state object is left
  // to each test so that tests can replace the default settings with
  // something more appropriate.
  //
  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    // Create an offscreen texture to render to
    TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                             kOffscreenTexWidth,
                                             kOffscreenTexHeight,
                                             TextureDesc::TextureUsageBits::Sampled |
                                                 TextureDesc::TextureUsageBits::Attachment);
    texDesc.debugName = "test";

    Result ret;
    offscreenTexture_ = iglDev_->createTexture(texDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok) << ret.message;
    ASSERT_TRUE(offscreenTexture_ != nullptr);

    // Create framebuffer using the offscreen texture
    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = offscreenTexture_}}};
    framebuffer_ = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(framebuffer_ != nullptr);

    // Initialize render pass descriptor
    renderPass_ = {
        .colorAttachments = {{.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::Store,
                              .clearColor = {0.0, 0.0, 0.0, 1.0}}},
    };

    // Initialize shader stages
    std::unique_ptr<IShaderStages> stages;
    igl::tests::util::createSimpleShaderStages(iglDev_, stages);
    shaderStages_ = std::move(stages);

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
    const SamplerStateDesc samplerDesc;
    samp_ = iglDev_->createSamplerState(samplerDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(samp_ != nullptr);

    // Initialize Graphics Pipeline Descriptor, but leave the creation
    // to the individual tests in case further customization is required
    renderPipelineDesc_ = {
        .vertexInputState = vertexInputState_,
        .shaderStages = shaderStages_,
        .targetDesc = {.colorAttachments = {{.textureFormat = offscreenTexture_->getFormat()}}},
        .cullMode = CullMode::Disabled,
        .fragmentUnitSamplerMap = {{textureUnit_, IGL_NAMEHANDLE(data::shader::kSimpleSampler)}},
    };
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ICommandBuffer> cmdBuf_;
  CommandBufferDesc cbDesc_ = {};
  const std::string backend_ = IGL_BACKEND_TYPE;

  RenderPassDesc renderPass_;
  std::shared_ptr<ITexture> offscreenTexture_;
  std::shared_ptr<IFramebuffer> framebuffer_;

  // Currently it is left to individual tests to initialize this
  std::shared_ptr<ITexture> inputTexture_;

  std::shared_ptr<IShaderStages> shaderStages_;

  std::shared_ptr<IVertexInputState> vertexInputState_;
  std::shared_ptr<IBuffer> vb_, uv_, ib_;

  std::shared_ptr<ISamplerState> samp_;

  RenderPipelineDesc renderPipelineDesc_;
  size_t textureUnit_ = 0;
};

TEST(TextureFormatProperties, Construction) {
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(std::string(props.name), std::string("RGBA_UNorm8"));
    EXPECT_EQ(props.format, TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(props.componentsPerPixel, 4);
    EXPECT_EQ(props.bytesPerBlock, 4);
    EXPECT_EQ(props.blockWidth, 1);
    EXPECT_EQ(props.blockHeight, 1);
    EXPECT_EQ(props.blockDepth, 1);
    EXPECT_EQ(props.minBlocksX, 1);
    EXPECT_EQ(props.minBlocksY, 1);
    EXPECT_EQ(props.minBlocksZ, 1);
  }
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    EXPECT_EQ(std::string(props.name), std::string("RGB_PVRTC_2BPPV1"));
    EXPECT_EQ(props.format, TextureFormat::RGB_PVRTC_2BPPV1);
    EXPECT_EQ(props.componentsPerPixel, 3);
    EXPECT_EQ(props.bytesPerBlock, 8);
    EXPECT_EQ(props.blockWidth, 8);
    EXPECT_EQ(props.blockHeight, 4);
    EXPECT_EQ(props.blockDepth, 1);
    EXPECT_EQ(props.minBlocksX, 2);
    EXPECT_EQ(props.minBlocksY, 2);
    EXPECT_EQ(props.minBlocksZ, 1);
  }
}

TEST(TextureFormatProperties, PackedUnsignedFloatFormat) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::B10G11R11_UFloat);
  EXPECT_EQ(props.componentsPerPixel, 3);
  EXPECT_EQ(props.bytesPerBlock, 4);
  EXPECT_TRUE(props.isHDR());
  EXPECT_FALSE(props.isCompressed());
  EXPECT_FALSE(props.hasDepth());
  EXPECT_FALSE(props.hasStencil());
  EXPECT_TRUE(props.hasColor());
}

TEST(TextureFormatProperties, GetRows) {
  {
    const auto range = TextureRangeDesc::new2D(0, 0, 2, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(props.getRows(range), 2);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 3);
  }
  {
    const auto range = TextureRangeDesc::new2D(0, 0, 2, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    // MinBlocksY = 2
    EXPECT_EQ(props.getRows(range), 2);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 4);
  }
  {
    const auto range = TextureRangeDesc::new3D(0, 0, 0, 2, 2, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(props.getRows(range), 4);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 5);
  }
  {
    const auto range = TextureRangeDesc::new3D(0, 0, 0, 2, 2, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    // MinBlocksY = 2
    EXPECT_EQ(props.getRows(range), 4);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 6);
  }
  {
    const auto range = TextureRangeDesc::new2DArray(0, 0, 2, 2, 0, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(props.getRows(range), 4);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 6);
  }
  {
    const auto range = TextureRangeDesc::new2DArray(0, 0, 2, 2, 0, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    // MinBlocksY = 2
    EXPECT_EQ(props.getRows(range), 4);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 8);
  }
  {
    const auto range = TextureRangeDesc::newCube(0, 0, 2, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(props.getRows(range), 12);
    EXPECT_EQ(props.getRows(range.atFace(TextureCubeFace::NegX)), 2);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 18);
  }
  {
    const auto range = TextureRangeDesc::newCube(0, 0, 2, 2);
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    // MinBlocksY = 2
    EXPECT_EQ(props.getRows(range), 12);
    EXPECT_EQ(props.getRows(range.atFace(TextureCubeFace::NegX)), 2);
    EXPECT_EQ(props.getRows(range.withNumMipLevels(2)), 24);
  }
}

TEST(TextureFormatProperties, GetBytesPerRow) {
  const auto range = TextureRangeDesc::new2D(0, 0, 2, 2);
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(props.getBytesPerRow(range), 2 * 4);
  }
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    // minBlocksX = 2
    EXPECT_EQ(props.getBytesPerRow(range), 2 * 1 * 8);
  }
}

TEST(TextureFormatProperties, getBytesPerLayer) {
  const auto range = TextureRangeDesc::new2D(0, 0, 10, 10);
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    EXPECT_EQ(props.getBytesPerLayer(range), 10 * 10 * 4);
    EXPECT_EQ(props.getBytesPerLayer(range, 50), 10 * 50);

    EXPECT_EQ(props.getBytesPerLayer(10, 10, 1), 10 * 10 * 4);
    EXPECT_EQ(props.getBytesPerLayer(10, 10, 1, 50), 10 * 50);
  }
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    // 2 blocks x 3 blocks
    EXPECT_EQ(props.getBytesPerLayer(range), 2 * 3 * 8);
    EXPECT_EQ(props.getBytesPerLayer(10, 10, 1), 2 * 3 * 8);
  }
}

TEST(TextureFormatProperties, getBytesPerRange) {
  auto range = TextureRangeDesc::new2D(0, 0, 10, 10, 0, 3);
  auto cubeRange = TextureRangeDesc::newCube(0, 0, 10, 10, 0, 3);
  auto cubeFaceRange = TextureRangeDesc::newCubeFace(0, 0, 10, 10, TextureCubeFace::PosZ, 0, 3);
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    // Level 0: 10 pixels x 10 pixels
    // Level 1:  5 pixels x  5 pixels
    // Level 2:  2 pixels x  2 pixels
    const auto bytes = ((10 * 10) + (5 * 5) + (2 * 2)) * 4;
    EXPECT_EQ(props.getBytesPerRange(range), bytes);
    EXPECT_EQ(props.getBytesPerRange(cubeRange), bytes * 6);
    EXPECT_EQ(props.getBytesPerRange(cubeFaceRange), bytes);
  }
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    // Level 0: 10 pixels x 10 pixels
    const auto bytes = 10 * 50;
    EXPECT_EQ(props.getBytesPerRange(range.atMipLevel(0), 50), bytes);
    EXPECT_EQ(props.getBytesPerRange(cubeRange.atMipLevel(0), 50), bytes * 6);
    EXPECT_EQ(props.getBytesPerRange(cubeFaceRange.atMipLevel(0), 50), bytes);
  }
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    // Level 0: 2 blocks x 3 blocks
    // Level 1: 2 blocks x 2 blocks
    // Level 2: 2 blocks x 2 blocks
    const auto bytes = ((2 * 3) + (2 * 2) + (2 * 2)) * 8;
    EXPECT_EQ(props.getBytesPerRange(range), bytes);
    EXPECT_EQ(props.getBytesPerRange(cubeRange), bytes * 6);
    EXPECT_EQ(props.getBytesPerRange(cubeFaceRange), bytes);
  }
}

TEST(TextureFormatProperties, getNumMipLevels) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
  // 4x4 RGBA_UNorm8 mip chain:
  //   Level 0: 4x4 x 4 bytes = 64 bytes
  //   Level 1: 2x2 x 4 bytes = 16 bytes
  //   Level 2: 1x1 x 4 bytes =  4 bytes
  // Full chain = 84 bytes

  // Fewer bytes than the base level holds no complete mip level.
  EXPECT_EQ(props.getNumMipLevels(4, 4, 63), 0);
  // Exactly enough for the base level.
  EXPECT_EQ(props.getNumMipLevels(4, 4, 64), 1);
  // Enough for the base level plus the first mip level.
  EXPECT_EQ(props.getNumMipLevels(4, 4, 80), 2);
  // Enough for the full mip chain.
  EXPECT_EQ(props.getNumMipLevels(4, 4, 84), 3);
}

TEST(TextureFormatProperties, getSubRangeByteOffset) {
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    {
      // 2D Range
      const auto range = TextureRangeDesc::new2D(0, 0, 10, 10, 0, 3);
      // Level 0: 10 pixels x 10 pixels = 400 bytes
      // Level 1:  5 pixels x  5 pixels = 100 bytes
      // Level 2:  2 pixels x  2 pixels =  16 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 400);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(2)), 500);
    }

    {
      // 2D Aray Range
      const auto range = TextureRangeDesc::new2DArray(0, 0, 10, 10, 0, 2, 0, 3);
      // Level 0: 10 pixels x 10 pixels x 2 layers = 800 bytes
      // Level 1:  5 pixels x  5 pixels x 2 layers = 200 bytes
      // Level 2:  2 pixels x  2 pixels x 2 layers =  32 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atLayer(1)), 400);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 800);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1).atLayer(1)), 900);

      // Custom row length
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.withNumMipLevels(1).atLayer(1), 50), 500);
    }

    {
      // 3D Range
      const auto range = TextureRangeDesc::new3D(0, 0, 0, 10, 10, 10, 0, 3);
      // Level 0: 10 pixels x 10 pixels x 10 pixels = 4000 bytes
      // Level 1:  5 pixels x  5 pixels x  5 pixels =  500 bytes
      // Level 2:  2 pixels x  2 pixels x  2 pixels =   32 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 4000);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(2)), 4500);
    }

    {
      // Cube Range
      const auto range = TextureRangeDesc::newCube(0, 0, 10, 10, 0, 3);
      // Level 0: 10 pixels x 10 pixels x 6 faces = 2400 bytes
      // Level 1:  5 pixels x  5 pixels x 6 faces =  600 bytes
      // Level 2:  2 pixels x  2 pixels x 6 faces =   96 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atFace(1)), 400);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 2400);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1).atFace(1)), 2500);

      // Custom row length
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.withNumMipLevels(1).atFace(1), 50), 500);
    }
  }
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGB_PVRTC_2BPPV1);
    {
      // 2D Range
      const auto range = TextureRangeDesc::new2D(0, 0, 10, 10, 0, 3);
      // Level 0: 2 blocks x 3 blocks x 8 bytes = 48 bytes
      // Level 1: 2 blocks x 2 blocks x 8 bytes = 32 bytes
      // Level 2: 2 blocks x 2 blocks x 8 bytes = 32 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 48);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(2)), 80);
    }

    {
      // 2D Aray Range
      const auto range = TextureRangeDesc::new2DArray(0, 0, 10, 10, 0, 2, 0, 3);
      // Level 0: 2 blocks x 3 blocks x 2 layers x 8 bytes = 96 bytes
      // Level 1: 2 blocks x 2 blocks x 2 layers x 8 bytes = 64 bytes
      // Level 2: 2 blocks x 2 blocks x 2 layers x 8 bytes = 64 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atLayer(1)), 48);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 96);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1).atLayer(1)), 128);
    }

    {
      // 3D Range
      const auto range = TextureRangeDesc::new3D(0, 0, 0, 10, 10, 10, 0, 3);
      // Level 0: 2 blocks x 3 blocks x 10 pixels x 8 bytes = 480 bytes
      // Level 1: 2 blocks x 2 blocks x  5 pixels x 8 bytes = 160 bytes
      // Level 2: 2 blocks x 2 blocks x  2 pixels x 8 bytes =  64 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 480);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(2)), 640);
    }

    {
      // Cube Range
      const auto range = TextureRangeDesc::newCube(0, 0, 10, 10, 0, 3);
      // Level 0: 2 blocks x 3 blocks x 6 faces x 8 bytes = 288 bytes
      // Level 1: 2 blocks x 2 blocks x 6 faces x 8 bytes = 192 bytes
      // Level 2: 2 blocks x 2 blocks x 6 faces x 8 bytes = 192 bytes

      EXPECT_EQ(props.getSubRangeByteOffset(range, range), 0);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atFace(1)), 48);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1)), 288);
      EXPECT_EQ(props.getSubRangeByteOffset(range, range.atMipLevel(1).atFace(1)), 320);
    }
  }
}

TEST(TextureFormatProperties, GetNumMipLevels) {
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    // 4x4 RGBA_UNorm8 mip chain:
    // Level 0: 4 pixels x 4 pixels = 64 bytes (cumulative 64)
    // Level 1: 2 pixels x 2 pixels = 16 bytes (cumulative 80)
    // Level 2: 1 pixel  x 1 pixel  =  4 bytes (cumulative 84)
    EXPECT_EQ(props.getNumMipLevels(4, 4, 64), 1);
    EXPECT_EQ(props.getNumMipLevels(4, 4, 80), 2);
    EXPECT_EQ(props.getNumMipLevels(4, 4, 84), 3);
  }
  {
    const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
    // 10x10 RGBA_UNorm8 mip chain:
    // Level 0: 10 pixels x 10 pixels = 400 bytes (cumulative 400)
    // Level 1:  5 pixels x  5 pixels = 100 bytes (cumulative 500)
    // Level 2:  2 pixels x  2 pixels =  16 bytes (cumulative 516)
    // Level 3:  1 pixel  x  1 pixel  =   4 bytes (cumulative 520)
    EXPECT_EQ(props.getNumMipLevels(10, 10, 400), 1);
    EXPECT_EQ(props.getNumMipLevels(10, 10, 500), 2);
    EXPECT_EQ(props.getNumMipLevels(10, 10, 516), 3);
    EXPECT_EQ(props.getNumMipLevels(10, 10, 520), 4);

    // Not enough bytes for even the base mip level yields zero.
    EXPECT_EQ(props.getNumMipLevels(10, 10, 399), 0);
    // A partially-filled second mip level does not count.
    EXPECT_EQ(props.getNumMipLevels(10, 10, 450), 1);
  }
}

TEST(TextureFormatProperties, InvalidFormat) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::Invalid);
  EXPECT_FALSE(props.isValid());
}

TEST(TextureFormatProperties, ColorFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UNorm8);
  EXPECT_TRUE(props.isValid());
  EXPECT_FALSE(props.isCompressed());
  EXPECT_FALSE(props.isInteger());
  EXPECT_FALSE(props.isSRGB());
  EXPECT_FALSE(props.isHDR());
  EXPECT_FALSE(props.hasDepth());
  EXPECT_FALSE(props.hasStencil());
  EXPECT_TRUE(props.hasColor());
  EXPECT_FALSE(props.isDepthOnly());
  EXPECT_FALSE(props.isDepthOrDepthStencil());
  EXPECT_FALSE(props.isStencilOnly());
  EXPECT_FALSE(props.isDepthOrStencil());
  EXPECT_FALSE(props.isVariableLength());
}

TEST(TextureFormatProperties, SrgbFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_SRGB);
  EXPECT_TRUE(props.isSRGB());
  EXPECT_FALSE(props.isCompressed());
  EXPECT_TRUE(props.hasColor());
}

TEST(TextureFormatProperties, HdrFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_F16);
  EXPECT_TRUE(props.isHDR());
  EXPECT_FALSE(props.isInteger());
  EXPECT_TRUE(props.hasColor());
}

TEST(TextureFormatProperties, IntegerFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_UInt32);
  EXPECT_TRUE(props.isInteger());
  EXPECT_TRUE(props.isHDR());
  EXPECT_TRUE(props.hasColor());
}

TEST(TextureFormatProperties, CompressedFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::RGBA_ASTC_4x4);
  EXPECT_TRUE(props.isCompressed());
  EXPECT_FALSE(props.isSRGB());
  EXPECT_TRUE(props.hasColor());
}

TEST(TextureFormatProperties, DepthOnlyFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::Z_UNorm24);
  EXPECT_TRUE(props.hasDepth());
  EXPECT_FALSE(props.hasStencil());
  EXPECT_FALSE(props.hasColor());
  EXPECT_TRUE(props.isDepthOnly());
  EXPECT_TRUE(props.isDepthOrDepthStencil());
  EXPECT_FALSE(props.isStencilOnly());
  EXPECT_TRUE(props.isDepthOrStencil());
}

TEST(TextureFormatProperties, DepthStencilFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::S8_UInt_Z24_UNorm);
  EXPECT_TRUE(props.hasDepth());
  EXPECT_TRUE(props.hasStencil());
  EXPECT_FALSE(props.hasColor());
  EXPECT_FALSE(props.isDepthOnly());
  EXPECT_TRUE(props.isDepthOrDepthStencil());
  EXPECT_FALSE(props.isStencilOnly());
  EXPECT_TRUE(props.isDepthOrStencil());
}

TEST(TextureFormatProperties, StencilOnlyFormatFlags) {
  const auto props = TextureFormatProperties::fromTextureFormat(TextureFormat::S_UInt8);
  EXPECT_FALSE(props.hasDepth());
  EXPECT_TRUE(props.hasStencil());
  EXPECT_FALSE(props.hasColor());
  EXPECT_FALSE(props.isDepthOnly());
  EXPECT_FALSE(props.isDepthOrDepthStencil());
  EXPECT_TRUE(props.isStencilOnly());
  EXPECT_TRUE(props.isDepthOrStencil());
}

} // namespace igl::tests
