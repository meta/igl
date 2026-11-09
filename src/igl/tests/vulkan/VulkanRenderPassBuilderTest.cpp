/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanRenderPassBuilder.h>

#include "../util/TestDevice.h"

#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX

namespace igl::tests {

namespace {

constexpr VkDependencyFlags kByRegion = VK_DEPENDENCY_BY_REGION_BIT;
constexpr VkDependencyFlags kViewLocal = VK_DEPENDENCY_VIEW_LOCAL_BIT;
constexpr VkDependencyFlags kByRegionViewLocal = kByRegion | kViewLocal;

} // namespace

class VulkanRenderPassBuilderTest : public ::testing::Test {
 public:
  VulkanRenderPassBuilderTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_EQ(iglDev_->getBackendType(), BackendType::Vulkan) << "Test requires Vulkan backend";
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;

  igl::vulkan::VulkanContext& getVulkanContext() {
    auto& device = static_cast<igl::vulkan::Device&>(*iglDev_);
    return device.getVulkanContext();
  }
};

TEST_F(VulkanRenderPassBuilderTest, SingleColorAttachment) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, ColorWithDepthStencil) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addDepthStencil(
      VK_FORMAT_D24_UNORM_S8_UINT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, MultipleColorAttachments) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(VK_FORMAT_R8G8B8A8_UNORM,
                   VK_ATTACHMENT_LOAD_OP_LOAD,
                   VK_ATTACHMENT_STORE_OP_STORE,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, WithResolveAttachment) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(VK_FORMAT_R8G8B8A8_UNORM,
                   VK_ATTACHMENT_LOAD_OP_CLEAR,
                   VK_ATTACHMENT_STORE_OP_STORE,
                   VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_SAMPLE_COUNT_4_BIT);
  builder.addColorResolve(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE);

  auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, MultiviewRenderPass) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.setMultiviewMasks(0x3, 0x3);

  auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, HashEquality) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder::HashFunction hasher;
  EXPECT_EQ(hasher(builder1), hasher(builder2));
}

TEST_F(VulkanRenderPassBuilderTest, HashInequality) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_B8G8R8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder::HashFunction hasher;
  EXPECT_NE(hasher(builder1), hasher(builder2));
}

TEST_F(VulkanRenderPassBuilderTest, WithDepthStencilResolveAttachment) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(VK_FORMAT_R8G8B8A8_UNORM,
                   VK_ATTACHMENT_LOAD_OP_CLEAR,
                   VK_ATTACHMENT_STORE_OP_STORE,
                   VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_SAMPLE_COUNT_4_BIT);
  builder.addDepthStencil(VK_FORMAT_D24_UNORM_S8_UINT,
                          VK_ATTACHMENT_LOAD_OP_CLEAR,
                          VK_ATTACHMENT_STORE_OP_DONT_CARE,
                          VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                          VK_ATTACHMENT_STORE_OP_DONT_CARE,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                          VK_SAMPLE_COUNT_4_BIT);
  builder.addDepthStencilResolve(
      VK_FORMAT_D24_UNORM_S8_UINT, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE);

  auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, EqualityOperator) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  EXPECT_TRUE(builder1 == builder2);

  igl::vulkan::VulkanRenderPassBuilder builder3;
  builder3.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);

  EXPECT_FALSE(builder1 == builder3);
}

TEST_F(VulkanRenderPassBuilderTest, HashInequalityDepthStencil) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addDepthStencil(
      VK_FORMAT_D24_UNORM_S8_UINT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.addDepthStencil(
      VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder::HashFunction hasher;
  EXPECT_NE(hasher(builder1), hasher(builder2));
}

TEST_F(VulkanRenderPassBuilderTest, EqualityDepthStencil) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addDepthStencil(
      VK_FORMAT_D24_UNORM_S8_UINT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.addDepthStencil(
      VK_FORMAT_D24_UNORM_S8_UINT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  EXPECT_TRUE(builder1 == builder2);

  igl::vulkan::VulkanRenderPassBuilder builder3;
  builder3.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder3.addDepthStencil(
      VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  EXPECT_FALSE(builder1 == builder3);
}

TEST_F(VulkanRenderPassBuilderTest, DepthStencilWithExplicitStencilOps) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addDepthStencil(VK_FORMAT_D24_UNORM_S8_UINT,
                          VK_ATTACHMENT_LOAD_OP_CLEAR,
                          VK_ATTACHMENT_STORE_OP_STORE,
                          VK_ATTACHMENT_LOAD_OP_CLEAR,
                          VK_ATTACHMENT_STORE_OP_STORE);

  auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, EqualityMultiview) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.setMultiviewMasks(0x3, 0x3);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.setMultiviewMasks(0x3, 0x3);

  EXPECT_TRUE(builder1 == builder2);

  igl::vulkan::VulkanRenderPassBuilder builder3;
  builder3.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder3.setMultiviewMasks(0xF, 0xF);

  EXPECT_FALSE(builder1 == builder3);
}

TEST_F(VulkanRenderPassBuilderTest, DefaultBuilderEquality) {
  const igl::vulkan::VulkanRenderPassBuilder a;
  const igl::vulkan::VulkanRenderPassBuilder b;

  EXPECT_TRUE(a == b);

  igl::vulkan::VulkanRenderPassBuilder::HashFunction hasher;
  EXPECT_EQ(hasher(a), hasher(b));
}

TEST_F(VulkanRenderPassBuilderTest, HashInequalityMultiview) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.setMultiviewMasks(0x3, 0x3);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.setMultiviewMasks(0xF, 0xF);

  igl::vulkan::VulkanRenderPassBuilder::HashFunction hasher;
  EXPECT_NE(hasher(builder1), hasher(builder2));
}

TEST_F(VulkanRenderPassBuilderTest, RenderPassCaching) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const auto rp1 = ctx.findRenderPass(builder1);
  const auto rp2 = ctx.findRenderPass(builder2);

  EXPECT_NE(rp1.pass, VK_NULL_HANDLE);
  EXPECT_EQ(rp1.pass, rp2.pass);
}

// Multisubpass render pass tests

TEST_F(VulkanRenderPassBuilderTest, NumSubpassesDefaultsToOne) {
  igl::vulkan::VulkanRenderPassBuilder builder;
  EXPECT_EQ(builder.getNumSubpasses(), 1u);
}

TEST_F(VulkanRenderPassBuilderTest, SetSubpassesReportsCount) {
  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b1110, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b0001, .inputAttachmentMask = 0b1110},
  };
  builder.setSubpasses(subpasses, 2);
  EXPECT_EQ(builder.getNumSubpasses(), 2u);
}

TEST_F(VulkanRenderPassBuilderTest, DeferredLayoutPreserveAndDependencies) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b0001, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b1110, .inputAttachmentMask = 0b0001},
  };
  builder.setSubpasses(subpasses, 2);

  EXPECT_EQ(builder.getPreserveMask(0), 0u);
  EXPECT_EQ(builder.getPreserveMask(1), 0u);
  // 0 -> 1, 1 -> EXTERNAL, and EXTERNAL -> 0 and EXTERNAL -> 1 for the first uses of the
  // attachments
  EXPECT_EQ(builder.getNumDependencies(), 4u);

  const auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
  EXPECT_EQ(rp.numSubpasses, 2u);
}

TEST_F(VulkanRenderPassBuilderTest, ThreeSubpassChainPreserveMasks) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b0011, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b0100, .inputAttachmentMask = 0b0001},
      {.colorAttachmentMask = 0b1000, .inputAttachmentMask = 0b0110},
  };
  builder.setSubpasses(subpasses, 3);

  EXPECT_EQ(builder.getPreserveMask(0), 0u);
  EXPECT_EQ(builder.getPreserveMask(1), 0b0010u);
  EXPECT_EQ(builder.getPreserveMask(2), 0b0001u);
  // 3 between subpasses, 2 to VK_SUBPASS_EXTERNAL and 3 from it
  EXPECT_EQ(builder.getNumDependencies(), 8u);

  const auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
  EXPECT_EQ(rp.numSubpasses, 3u);
}

TEST_F(VulkanRenderPassBuilderTest, ThreeSubpassChainWithoutStorePreserve) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b0011, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b0100, .inputAttachmentMask = 0b0001},
      {.colorAttachmentMask = 0b1000, .inputAttachmentMask = 0b0110},
  };
  builder.setSubpasses(subpasses, 3);

  EXPECT_EQ(builder.getPreserveMask(0), 0u);
  EXPECT_EQ(builder.getPreserveMask(1), 0b0010u);
  EXPECT_EQ(builder.getPreserveMask(2), 0u);
  EXPECT_EQ(builder.getNumDependencies(), 8u);

  const auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
}

TEST_F(VulkanRenderPassBuilderTest, FourSubpassChainPreserveMask) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b0001, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b0010, .inputAttachmentMask = 0b0001},
      {.colorAttachmentMask = 0b0100, .inputAttachmentMask = 0b0010},
      {.colorAttachmentMask = 0b1000, .inputAttachmentMask = 0b0101},
  };
  builder.setSubpasses(subpasses, 4);

  EXPECT_EQ(builder.getPreserveMask(2), 0b0001u);
  // Every subpass is the first to use one attachment, so each has a dependency from
  // VK_SUBPASS_EXTERNAL
  EXPECT_EQ(builder.getNumDependencies(), 10u);

  const auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
  EXPECT_EQ(rp.numSubpasses, 4u);
}

// The dependency from VK_SUBPASS_EXTERNAL orders the transition away from the initial layout after
// the attachment writes of earlier render passes, and the one to it orders later reads
TEST_F(VulkanRenderPassBuilderTest, LegacyBuilderHasDependenciesFromAndToExternal) {
  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  EXPECT_EQ(builder.getNumDependencies(), 2u);
}

TEST_F(VulkanRenderPassBuilderTest, SubpassMasksAffectEqualityAndHash) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses1[] = {
      {.colorAttachmentMask = 0b0110, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b1001, .inputAttachmentMask = 0b0110},
  };
  builder1.setSubpasses(subpasses1, 2);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses2[] = {
      {.colorAttachmentMask = 0b0110, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b1001, .inputAttachmentMask = 0b0001},
  };
  builder2.setSubpasses(subpasses2, 2);

  EXPECT_FALSE(builder1 == builder2);
  igl::vulkan::VulkanRenderPassBuilder::HashFunction hasher;
  EXPECT_NE(hasher(builder1), hasher(builder2));
}

TEST_F(VulkanRenderPassBuilderTest, SubpassMasksIdenticalAffectEqualityAndHash) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b01, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b10, .inputAttachmentMask = 0b01},
  };
  builder1.setSubpasses(subpasses, 2);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.setSubpasses(subpasses, 2);

  EXPECT_TRUE(builder1 == builder2);
  igl::vulkan::VulkanRenderPassBuilder::HashFunction hasher;
  EXPECT_EQ(hasher(builder1), hasher(builder2));
}

TEST_F(VulkanRenderPassBuilderTest, CompatibilityKeyIgnoresLoadOpsAndInitialLayout) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(VK_FORMAT_R8G8B8A8_UNORM,
                    VK_ATTACHMENT_LOAD_OP_CLEAR,
                    VK_ATTACHMENT_STORE_OP_STORE,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(VK_FORMAT_R8G8B8A8_UNORM,
                    VK_ATTACHMENT_LOAD_OP_LOAD,
                    VK_ATTACHMENT_STORE_OP_STORE,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  const auto key1 = builder1.compatibilityKey();
  const auto key2 = builder2.compatibilityKey();

  EXPECT_TRUE(key1 == key2);
}

TEST_F(VulkanRenderPassBuilderTest, CompatibilityKeyIgnoresStoreOpsForSingleSubpass) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);

  const auto key1 = builder1.compatibilityKey();
  const auto key2 = builder2.compatibilityKey();

  EXPECT_TRUE(key1 == key2);
}

TEST_F(VulkanRenderPassBuilderTest, CompatibilityKeyIncludesPreserveMasks) {
  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses1[] = {
      {.colorAttachmentMask = 0b0011, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b0100, .inputAttachmentMask = 0b0001},
      {.colorAttachmentMask = 0b1000, .inputAttachmentMask = 0b0110},
  };
  builder1.setSubpasses(subpasses1, 3);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);

  const igl::RenderPassDesc::SubpassDesc subpasses2[] = {
      {.colorAttachmentMask = 0b0011, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b0100, .inputAttachmentMask = 0b0001},
      {.colorAttachmentMask = 0b1000, .inputAttachmentMask = 0b0110},
  };
  builder2.setSubpasses(subpasses2, 3);

  const auto key1 = builder1.compatibilityKey();
  const auto key2 = builder2.compatibilityKey();

  EXPECT_FALSE(key1 == key2);
}

TEST_F(VulkanRenderPassBuilderTest, FindRenderPassMultiSubpassCreatesPass) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b01, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b10, .inputAttachmentMask = 0b01},
  };
  builder.setSubpasses(subpasses, 2);

  const auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
  EXPECT_EQ(rp.numSubpasses, 2u);
}

TEST_F(VulkanRenderPassBuilderTest, FindRenderPassSharesCompatIndexAcrossLoadOps) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(VK_FORMAT_R8G8B8A8_UNORM,
                    VK_ATTACHMENT_LOAD_OP_CLEAR,
                    VK_ATTACHMENT_STORE_OP_STORE,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(VK_FORMAT_R8G8B8A8_UNORM,
                    VK_ATTACHMENT_LOAD_OP_LOAD,
                    VK_ATTACHMENT_STORE_OP_STORE,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  const auto rp1 = ctx.findRenderPass(builder1);
  const auto rp2 = ctx.findRenderPass(builder2);

  EXPECT_NE(rp1.index, rp2.index);
  EXPECT_EQ(rp1.compatIndex, rp2.compatIndex);
}

TEST_F(VulkanRenderPassBuilderTest, FindRenderPassSeparatesSingleAndMultiSubpass) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder1;
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder1.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  igl::vulkan::VulkanRenderPassBuilder builder2;
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder2.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b01, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b10, .inputAttachmentMask = 0b01},
  };
  builder2.setSubpasses(subpasses, 2);

  const auto rp1 = ctx.findRenderPass(builder1);
  const auto rp2 = ctx.findRenderPass(builder2);

  EXPECT_NE(rp1.compatIndex, rp2.compatIndex);
}

TEST_F(VulkanRenderPassBuilderTest, StereoSubpassDependencyFlags) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.setMultiviewMasks(0x3, 0x3);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b10, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b01, .inputAttachmentMask = 0b10},
  };
  builder.setSubpasses(subpasses, 2);

  const auto rp = ctx.findRenderPass(builder);
  EXPECT_NE(rp.pass, VK_NULL_HANDLE);
  EXPECT_EQ(rp.numSubpasses, 2u);

  // build() creates the dependency 0 -> 1 first, then the dependency of subpass 1, which is the
  // last one to use both attachments, on VK_SUBPASS_EXTERNAL, and then the dependencies from
  // VK_SUBPASS_EXTERNAL of subpass 0 and subpass 1, the first ones to use attachment 1 and 0
  ASSERT_EQ(builder.getNumDependencies(), 4u);
  EXPECT_EQ(builder.getDependencyFlags(0), kByRegionViewLocal);
  // A dependency on or from VK_SUBPASS_EXTERNAL cannot be view-local
  for (uint32_t i = 1; i != 4; ++i) {
    EXPECT_EQ(builder.getDependencyFlags(i), 0u) << "dependency " << i;
  }
  // There is no fifth dependency
  EXPECT_EQ(builder.getDependencyFlags(4), 0u);
}

TEST_F(VulkanRenderPassBuilderTest, MonoSubpassDependencyFlags) {
  igl::vulkan::VulkanRenderPassBuilder builder;
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builder.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b10, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b01, .inputAttachmentMask = 0b10},
  };
  builder.setSubpasses(subpasses, 2);

  // The dependencies of StereoSubpassDependencyFlags, which are not view-local without a view mask
  ASSERT_EQ(builder.getNumDependencies(), 4u);
  EXPECT_EQ(builder.getDependencyFlags(0), kByRegion);
  for (uint32_t i = 1; i != 4; ++i) {
    EXPECT_EQ(builder.getDependencyFlags(i), 0u) << "dependency " << i;
  }
}

TEST_F(VulkanRenderPassBuilderTest, SubpassDependencyFlagsOfEveryDependency) {
  // Both the Mono and the Stereo builder
  for (const uint32_t viewMask : {0u, 0x3u}) {
    SCOPED_TRACE(viewMask != 0 ? "Stereo" : "Mono");

    igl::vulkan::VulkanRenderPassBuilder builder;
    for (uint32_t i = 0; i != 4; ++i) {
      builder.addColor(
          VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
    }
    if (viewMask != 0) {
      builder.setMultiviewMasks(viewMask, viewMask);
    }

    const igl::RenderPassDesc::SubpassDesc subpasses[] = {
        {.colorAttachmentMask = 0b0011, .inputAttachmentMask = 0},
        {.colorAttachmentMask = 0b0100, .inputAttachmentMask = 0b0001},
        {.colorAttachmentMask = 0b1000, .inputAttachmentMask = 0b0110},
    };
    builder.setSubpasses(subpasses, 3);

    // build() creates the dependencies of subpass 1 and 2 on the earlier subpasses first: 0 is
    // 0 -> 1 (consecutive), 1 is 0 -> 2 (subpass 2 reads attachment 1 of subpass 0), and 2 is
    // 1 -> 2 (consecutive). 3 and 4 are the ones of subpass 1, which is the last to use
    // attachment 0, and of subpass 2, the last to use attachments 1, 2 and 3, on
    // VK_SUBPASS_EXTERNAL. 5, 6 and 7 are the ones from VK_SUBPASS_EXTERNAL of subpass 0, the
    // first to use attachments 0 and 1, of subpass 1 (attachment 2) and of subpass 2 (attachment
    // 3).
    ASSERT_EQ(builder.getNumDependencies(), 8u);
    const VkDependencyFlags subpassFlags = viewMask != 0 ? kByRegionViewLocal : kByRegion;
    for (uint32_t i = 0; i != 3; ++i) {
      EXPECT_EQ(builder.getDependencyFlags(i), subpassFlags) << "dependency " << i;
    }
    for (uint32_t i = 3; i != 8; ++i) {
      EXPECT_EQ(builder.getDependencyFlags(i), 0u) << "dependency " << i;
    }
  }
}

TEST_F(VulkanRenderPassBuilderTest, StereoSubpassVsMonoCompatibility) {
  auto& ctx = getVulkanContext();

  igl::vulkan::VulkanRenderPassBuilder builderMono;
  builderMono.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builderMono.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);

  const igl::RenderPassDesc::SubpassDesc subpasses[] = {
      {.colorAttachmentMask = 0b10, .inputAttachmentMask = 0},
      {.colorAttachmentMask = 0b01, .inputAttachmentMask = 0b10},
  };
  builderMono.setSubpasses(subpasses, 2);

  igl::vulkan::VulkanRenderPassBuilder builderStereo;
  builderStereo.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builderStereo.addColor(
      VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
  builderStereo.setMultiviewMasks(0x3, 0x3);
  builderStereo.setSubpasses(subpasses, 2);

  const auto rpMono = ctx.findRenderPass(builderMono);
  const auto rpStereo = ctx.findRenderPass(builderStereo);

  EXPECT_NE(rpMono.compatIndex, rpStereo.compatIndex);
}

} // namespace igl::tests

#endif // IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
