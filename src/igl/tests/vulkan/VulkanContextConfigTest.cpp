/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/Common.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_LINUX
namespace igl::tests {

TEST(VulkanContextConfigTest, DefaultBooleanFields) {
  const vulkan::VulkanContextConfig config;
  EXPECT_FALSE(config.terminateOnValidationError);
  EXPECT_FALSE(config.enableConcurrentVkDevicesSupport);
  EXPECT_TRUE(config.enableValidation);
  EXPECT_TRUE(config.enableGPUAssistedValidation);
  EXPECT_FALSE(config.enableValidationOnAndroid);
  EXPECT_TRUE(config.enableExtraLogs);
  EXPECT_FALSE(config.enableDescriptorIndexing);
  EXPECT_TRUE(config.enableShaderInt16);
  EXPECT_TRUE(config.enableShaderDrawParameters);
  EXPECT_TRUE(config.enableStorageBuffer16BitAccess);
  EXPECT_TRUE(config.enableDualSrcBlend);
  EXPECT_FALSE(config.enableGfxReconstruct);
  EXPECT_FALSE(config.enableMultiviewPerViewViewports);
  EXPECT_FALSE(config.exportableFences);
  EXPECT_TRUE(config.chainSubmitsWithSemaphores);
  EXPECT_FALSE(config.headless);
}

TEST(VulkanContextConfigTest, DefaultSwapchainSettings) {
  const vulkan::VulkanContextConfig config;
  EXPECT_EQ(config.swapChainColorSpace, igl::ColorSpace::SRGBNonlinear);
  EXPECT_EQ(config.requestedSwapChainTextureFormat, igl::TextureFormat::RGBA_UNorm8);
}

TEST(VulkanContextConfigTest, DefaultResourceAndMemorySettings) {
  const vulkan::VulkanContextConfig config;
  EXPECT_EQ(config.maxResourceCount, 3u);
  EXPECT_EQ(config.pipelineCacheData, nullptr);
  EXPECT_EQ(config.pipelineCacheDataSize, 0u);
  EXPECT_EQ(config.vmaPreferredLargeHeapBlockSize, 0u);
  EXPECT_EQ(config.fenceTimeoutNanoseconds, UINT64_MAX);
}

TEST(VulkanContextConfigTest, DefaultExtensionSettings) {
  const vulkan::VulkanContextConfig config;
  EXPECT_EQ(config.numExtraInstanceExtensions, 0u);
  EXPECT_EQ(config.extraInstanceExtensions, nullptr);
}

TEST(VulkanContextConfigTest, DefaultEngineNames) {
  const vulkan::VulkanContextConfig config;
  EXPECT_STREQ(config.engineName, "IGL/Vulkan");
  EXPECT_STREQ(config.applicationName, "IGL/Vulkan");
}

} // namespace igl::tests
#endif
