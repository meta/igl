/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../data/TextureData.h"
#include "../util/Common.h"

#if !defined(_WIN32)
#include <IGLU/texture_accessor/ITextureAccessor.h>
#include <IGLU/texture_accessor/TextureAccessorFactory.h>
#endif
#include <igl/Common.h>
#include <igl/Framebuffer.h>

#define OFFSCREEN_TEX_HEIGHT 2
#define OFFSCREEN_TEX_WIDTH 2

namespace igl::tests {

//
// TextureAccessorDesignatedInitTest
//
// Validates that FramebufferDesc constructed via C++20 designated initializers
// (as used by iglu::opengl::TextureAccessor) produces a valid framebuffer
// identical to field-by-field initialization.
//
class TextureAccessorDesignatedInitTest : public ::testing::Test {
 public:
  TextureAccessorDesignatedInitTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_TRUE(iglDev_ != nullptr);
    ASSERT_TRUE(cmdQueue_ != nullptr);

    Result result;
    texDesc_ = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                  OFFSCREEN_TEX_WIDTH,
                                  OFFSCREEN_TEX_HEIGHT,
                                  TextureDesc::TextureUsageBits::Sampled |
                                      TextureDesc::TextureUsageBits::Attachment);
    texture_ = iglDev_->createTexture(texDesc_, &result);
    ASSERT_TRUE(result.isOk());
    ASSERT_TRUE(texture_ != nullptr);

    textureSizeInBytes_ = texture_->getProperties().getBytesPerRange(texture_->getFullRange());
    const auto range = TextureRangeDesc::new2D(0, 0, OFFSCREEN_TEX_WIDTH, OFFSCREEN_TEX_HEIGHT);
    texture_->upload(range, data::texture::kTexRgba2x2.data());
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  std::shared_ptr<ITexture> texture_;
  TextureDesc texDesc_;
  int textureSizeInBytes_{};
};

//
// Verify that a FramebufferDesc created with C++20 designated initializers
// produces a valid framebuffer that can be used for readback.
// This is the exact pattern used by iglu::opengl::TextureAccessor:
//   const FramebufferDesc framebufferDesc{
//       .colorAttachments = {{.texture = texture_}},
//   };
//
TEST_F(TextureAccessorDesignatedInitTest, DesignatedInitProducesValidFramebuffer) {
  Result result;

  // C++20 designated initializer style
  const FramebufferDesc framebufferDesc{
      .colorAttachments = {{.texture = texture_}},
  };

  auto framebuffer = iglDev_->createFramebuffer(framebufferDesc, &result);
  ASSERT_TRUE(result.isOk()) << result.message.c_str();
  ASSERT_TRUE(framebuffer != nullptr);

  // Verify the color attachment is correctly set
  auto colorAttachment = framebuffer->getColorAttachment(0);
  ASSERT_TRUE(colorAttachment != nullptr);
  EXPECT_EQ(colorAttachment.get(), texture_.get());
}

//
// Verify that designated-initializer FramebufferDesc and field-by-field
// initialization produce framebuffers with identical behavior for readback.
//
TEST_F(TextureAccessorDesignatedInitTest, DesignatedInitMatchesFieldByField) {
  Result result1;
  Result result2;

  // Old style: field-by-field
  const FramebufferDesc descOld{.colorAttachments = {{.texture = texture_}}};
  auto fbOld = iglDev_->createFramebuffer(descOld, &result1);
  ASSERT_TRUE(result1.isOk()) << result1.message.c_str();
  ASSERT_TRUE(fbOld != nullptr);

  // New style: designated initializer
  const FramebufferDesc descNew{
      .colorAttachments = {{.texture = texture_}},
  };
  auto fbNew = iglDev_->createFramebuffer(descNew, &result2);
  ASSERT_TRUE(result2.isOk()) << result2.message.c_str();
  ASSERT_TRUE(fbNew != nullptr);

  // Both framebuffers should have the same attachment
  EXPECT_EQ(fbOld->getColorAttachment(0).get(), fbNew->getColorAttachment(0).get());

  // Both should have no depth/stencil
  EXPECT_EQ(fbOld->getDepthAttachment(), nullptr);
  EXPECT_EQ(fbNew->getDepthAttachment(), nullptr);
  EXPECT_EQ(fbOld->getStencilAttachment(), nullptr);
  EXPECT_EQ(fbNew->getStencilAttachment(), nullptr);
}

#if !defined(_WIN32)
//
// End-to-end: create TextureAccessor via factory (which internally uses the
// designated-initializer pattern), do a sync readback, and
// verify pixel data matches what was uploaded.
//
TEST_F(TextureAccessorDesignatedInitTest, TextureAccessorReadbackWithDesignatedInit) {
  const auto dimensions = texture_->getDimensions();
  ASSERT_EQ(dimensions.width, static_cast<uint32_t>(OFFSCREEN_TEX_WIDTH));
  ASSERT_EQ(dimensions.height, static_cast<uint32_t>(OFFSCREEN_TEX_HEIGHT));

  auto textureAccessor = iglu::textureaccessor::TextureAccessorFactory::createTextureAccessor(
      iglDev_->getBackendType(), texture_, *iglDev_);
  ASSERT_TRUE(textureAccessor != nullptr);

  ASSERT_EQ(textureAccessor->getRequestStatus(),
            iglu::textureaccessor::RequestStatus::NotInitialized);

  auto bytes = textureAccessor->requestAndGetBytesSync(*cmdQueue_);
  ASSERT_EQ(textureAccessor->getRequestStatus(), iglu::textureaccessor::RequestStatus::Ready);

  // 2x2 RGBA = 16 bytes
  ASSERT_EQ(bytes.size(), 16u);

  // Verify pixel data matches uploaded texture
  const auto* pixels = reinterpret_cast<const uint32_t*>(bytes.data());
  for (int i = 0; i < textureSizeInBytes_ / 4; i++) {
    ASSERT_EQ(pixels[i], data::texture::kTexRgba2x2[i]);
  }
}
#endif // !defined(_WIN32)

//
// Verify that other colorAttachments slots remain null when only slot 0
// is set via designated initializer.
//
TEST_F(TextureAccessorDesignatedInitTest, OtherSlotsRemainNullWithDesignatedInit) {
  Result result;

  const FramebufferDesc framebufferDesc{
      .colorAttachments = {{.texture = texture_}},
  };

  auto framebuffer = iglDev_->createFramebuffer(framebufferDesc, &result);
  ASSERT_TRUE(result.isOk()) << result.message.c_str();
  ASSERT_TRUE(framebuffer != nullptr);

  // Only slot 0 should have an attachment
  EXPECT_NE(framebuffer->getColorAttachment(0), nullptr);

  // Remaining slots should be empty
  const auto indices = framebuffer->getColorAttachmentIndices();
  EXPECT_EQ(indices.size(), 1u);
  EXPECT_EQ(indices[0], 0u);
}

} // namespace igl::tests
