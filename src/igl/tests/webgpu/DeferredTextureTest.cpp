/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <igl/tests/util/device/webgpu/TestDevice.h>
#include <igl/tests/webgpu/TriangleRender.h>
#include <igl/webgpu/CommandBuffer.h>
#include <igl/webgpu/Readback.h>
#include <igl/webgpu/Texture.h>

namespace igl::tests {

namespace {

constexpr uint32_t kSize = 8;

} // namespace

// Deferred textures back Surface::getCurrentTexture(); these tests stand in a plain WGPUTexture for
// the surface's.
class WebGPUDeferredTextureTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
    device_ = util::device::webgpu::createTestDevice();
    ASSERT_NE(device_, nullptr);
    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    desc.size = {.width = kSize, .height = kSize, .depthOrArrayLayers = 1};
    desc.format = WGPUTextureFormat_RGBA8Unorm;
    source_.reset(wgpuDeviceCreateTexture(device_->getContext().getDevice(), &desc));
  }

  void TearDown() override {
    EXPECT_EQ(device_->getContext().getUncapturedErrorCount(), 0u);
  }

 protected:
  [[nodiscard]] std::shared_ptr<igl::webgpu::Texture> createDeferred(bool succeeds) {
    Result ret;
    auto texture = igl::webgpu::Texture::createDeferred(
        device_->getContext(),
        device_->getDeviceFeatureSet(),
        TextureDesc::new2D(
            TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Attachment),
        {
            .acquire =
                [this, succeeds] {
                  ++acquireCount_;
                  return succeeds ? source_ : igl::webgpu::Handle<WGPUTexture>();
                },
            .present = [this] { ++presentCount_; },
        },
        &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    return texture;
  }

  std::unique_ptr<igl::webgpu::Device> device_;
  igl::webgpu::Handle<WGPUTexture> source_;
  int acquireCount_ = 0;
  int presentCount_ = 0;
};

TEST_F(WebGPUDeferredTextureTest, AcquiresAtFirstAttachmentUseAndPresentsAfterSubmit) {
  webgpu::TriangleRenderer renderer;
  ASSERT_TRUE(renderer.initialize(*device_, TextureFormat::RGBA_UNorm8).isOk());
  const auto texture = createDeferred(/*succeeds=*/true);
  ASSERT_NE(texture, nullptr);
  EXPECT_FALSE(texture->isAcquired());

  Result ret;
  const auto framebuffer =
      device_->createFramebuffer({.colorAttachments = {{.texture = texture}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_EQ(acquireCount_, 0);

  // TriangleRenderer::render() presents its target before submitting.
  ASSERT_TRUE(renderer.render(texture).isOk());
  EXPECT_TRUE(texture->isAcquired());
  EXPECT_EQ(acquireCount_, 1);
  EXPECT_EQ(presentCount_, 1);
  ASSERT_TRUE(renderer.render(texture).isOk());
  EXPECT_EQ(acquireCount_, 1);
  EXPECT_EQ(presentCount_, 1);

  // Released, not destroyed: the source texture stays readable.
  std::vector<uint8_t> rgba(size_t{kSize} * kSize * 4);
  const WGPUTexture raw = texture->getWGPUTexture();
  EXPECT_EQ(raw, source_.get());
  ret = igl::webgpu::readTexture(
      device_->getContext(),
      {.texture = source_.get(), .width = kSize, .height = kSize, .bytesPerTexel = 4},
      rgba.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_NEAR(rgba[0], 51, 1);
  EXPECT_NEAR(rgba[1], 77, 1);
  EXPECT_NEAR(rgba[2], 102, 1);
}

TEST_F(WebGPUDeferredTextureTest, IsNotDestroyedWhenReleased) {
  webgpu::TriangleRenderer renderer;
  ASSERT_TRUE(renderer.initialize(*device_, TextureFormat::RGBA_UNorm8).isOk());
  {
    const auto texture = createDeferred(/*succeeds=*/true);
    ASSERT_TRUE(renderer.render(texture).isOk());
  }
  std::vector<uint8_t> rgba(size_t{kSize} * kSize * 4);
  const Result ret = igl::webgpu::readTexture(
      device_->getContext(),
      {.texture = source_.get(), .width = kSize, .height = kSize, .bytesPerTexel = 4},
      rgba.data());
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_NEAR(rgba[0], 51, 1);
}

TEST_F(WebGPUDeferredTextureTest, FailedAcquireFailsTheRenderPass) {
  const auto texture = createDeferred(/*succeeds=*/false);
  ASSERT_NE(texture, nullptr);
  Result ret;
  const auto framebuffer =
      device_->createFramebuffer({.colorAttachments = {{.texture = texture}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto queue = device_->createCommandQueue({}, &ret);
  const auto cmdBuffer = queue->createCommandBuffer({}, &ret);
  const auto encoder = cmdBuffer->createRenderCommandEncoder(
      {.colorAttachments = {{.loadAction = LoadAction::Clear}}}, framebuffer, {}, &ret);
  EXPECT_EQ(encoder, nullptr);
  EXPECT_EQ(ret.code, Result::Code::RuntimeError);
  EXPECT_EQ(acquireCount_, 1);
  cmdBuffer->present(texture);
  queue->submit(*cmdBuffer);
  EXPECT_EQ(presentCount_, 0);
}

TEST_F(WebGPUDeferredTextureTest, FailedAcquireAsResolveTargetFailsTheRenderPass) {
  const auto resolve = createDeferred(/*succeeds=*/false);
  ASSERT_NE(resolve, nullptr);
  Result ret;
  TextureDesc msaaDesc = TextureDesc::new2D(
      TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Attachment);
  msaaDesc.numSamples = 4;
  const auto msaa = device_->createTexture(msaaDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto framebuffer = device_->createFramebuffer(
      {.colorAttachments = {{.texture = msaa, .resolveTexture = resolve}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto queue = device_->createCommandQueue({}, &ret);
  const auto cmdBuffer = queue->createCommandBuffer({}, &ret);
  const auto encoder = cmdBuffer->createRenderCommandEncoder(
      {.colorAttachments = {{.loadAction = LoadAction::Clear,
                             .storeAction = StoreAction::MsaaResolve}}},
      framebuffer,
      {},
      &ret);
  EXPECT_EQ(encoder, nullptr);
  EXPECT_EQ(ret.code, Result::Code::RuntimeError);
}

TEST_F(WebGPUDeferredTextureTest, FailedAcquireFailsCopiesAndUploads) {
  const auto texture = createDeferred(/*succeeds=*/false);
  ASSERT_NE(texture, nullptr);
  std::vector<uint32_t> pixels(kSize * kSize, 0xff00ff00);
  EXPECT_EQ(texture->upload(texture->getFullRange(0), pixels.data()).code,
            Result::Code::InvalidOperation);
  Result ret;
  const auto queue = device_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  texture->generateMipmap(*queue);
  const auto buffer = device_->createBuffer(
      {.type = BufferDesc::BufferTypeBits::Storage, .length = pixels.size() * sizeof(uint32_t)},
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const auto cmdBuffer = queue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  cmdBuffer->copyTextureToBuffer(*texture, *buffer, 0);
  queue->submit(*cmdBuffer);
  cmdBuffer->waitUntilCompleted();
  const auto framebuffer =
      device_->createFramebuffer({.colorAttachments = {{.texture = texture}}}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  framebuffer->copyBytesColorAttachment(
      *queue, 0, pixels.data(), TextureRangeDesc::new2D(0, 0, kSize, kSize));
  EXPECT_EQ(pixels[0], 0xff00ff00u);
  EXPECT_EQ(acquireCount_, 1);
}

TEST_F(WebGPUDeferredTextureTest, RejectsSampledUsageAndMissingSource) {
  Result ret;
  EXPECT_EQ(igl::webgpu::Texture::createDeferred(
                device_->getContext(),
                device_->getDeviceFeatureSet(),
                TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                   kSize,
                                   kSize,
                                   TextureDesc::TextureUsageBits::Attachment |
                                       TextureDesc::TextureUsageBits::Sampled),
                {.acquire = [this] { return source_; }},
                &ret),
            nullptr);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);
  EXPECT_EQ(
      igl::webgpu::Texture::createDeferred(
          device_->getContext(),
          device_->getDeviceFeatureSet(),
          TextureDesc::new2D(
              TextureFormat::RGBA_UNorm8, kSize, kSize, TextureDesc::TextureUsageBits::Attachment),
          {},
          &ret),
      nullptr);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

} // namespace igl::tests
