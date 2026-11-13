/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../iglu/RecordingDevice.h"
#include "../util/TestDevice.h"

#include <IGLU/managedUniformBuffer/ManagedUniformBuffer.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include <igl/CommandBuffer.h>
#include <igl/Shader.h>
#include <igl/ShaderCreator.h>
#include <igl/glslang/GlslCompiler.h>
#include <igl/glslang/GlslangHelpers.h>
#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
#include <igl/vulkan/Buffer.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/ResourcesBinder.h>
#include <igl/vulkan/SamplerState.h>
#include <igl/vulkan/Texture.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanFeatures.h>
#include <igl/vulkan/VulkanTexture.h>
#endif

namespace igl::tests {

class DeviceVulkanTest : public ::testing::Test {
 public:
  DeviceVulkanTest() = default;

  // Set up common resources. This will create a device
  void SetUp() override {
    // Turn off debug break so unit tests can run
    igl::setDebugBreakEnabled(false);

    iglDev_ = util::createTestDevice();
    ASSERT_NE(iglDev_, nullptr);
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
};

/// CreateCommandQueue
/// Once the backend is more mature, we will use the IGL level test. For now
/// this is just here as a proof of concept.
TEST_F(DeviceVulkanTest, CreateCommandQueue) {
  Result ret;
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdQueue, nullptr);
}

TEST_F(DeviceVulkanTest, PlatformDevice) {
  auto& platformDevice = iglDev_->getPlatformDevice();
  auto& vulkanPlatformDevice = static_cast<vulkan::PlatformDevice&>(platformDevice);
  Result ret;
  auto depthTexture = vulkanPlatformDevice.createTextureFromNativeDepth(2, 2, &ret);
  ASSERT_TRUE(ret.isOk());
  // ASSERT_TRUE(depthTexture != nullptr); // no swapchain so null
  auto texture = vulkanPlatformDevice.createTextureFromNativeDrawable(&ret);
  ASSERT_TRUE(ret.isOk());
  // ASSERT_TRUE(texture != nullptr); // no swapchain so null

  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdQueue, nullptr);
  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  auto submitHandle = cmdQueue->submit(*cmdBuf);

  // NOLINTNEXTLINE(readability-qualified-auto)
  auto fence1 = vulkanPlatformDevice.getVkFenceFromSubmitHandle(submitHandle);
  ASSERT_NE(fence1, VK_NULL_HANDLE);

  vulkanPlatformDevice.waitOnSubmitHandle(submitHandle);
}

TEST_F(DeviceVulkanTest, Semaphores) {
  auto& device = static_cast<igl::vulkan::Device&>(*iglDev_);
  auto& ctx = device.getVulkanContext();

  // binary semaphore
  {
    auto semaphore = std::make_unique<igl::vulkan::VulkanSemaphore>(
        ctx.vf_, ctx.getVkDevice(), false, "semaphore");

    ASSERT_NE(semaphore, nullptr);
    ASSERT_NE(semaphore->getVkSemaphore(), VK_NULL_HANDLE);
    ASSERT_EQ(semaphore->getFileDescriptor(), -1);
  }
  // timeline semaphore
  {
    auto semaphore = std::make_unique<igl::vulkan::VulkanSemaphore>(
        ctx.vf_, ctx.getVkDevice(), 0, false, "timelineSemaphore");

    ASSERT_NE(semaphore, nullptr);
    ASSERT_NE(semaphore->getVkSemaphore(), VK_NULL_HANDLE);
    ASSERT_EQ(semaphore->getFileDescriptor(), -1);
  }
}

TEST_F(DeviceVulkanTest, PlatformDeviceSampler) {
  Result ret;
  TextureDesc textureDesc = TextureDesc::new2D(
      igl::TextureFormat::RGBA_UNorm8, 2, 2, TextureDesc::TextureUsageBits::Sampled);
  auto texture = iglDev_->createTexture(textureDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(texture != nullptr);
  auto* vulkanTexture = static_cast<vulkan::Texture*>(texture.get());
  auto& innerVulkanTexture = vulkanTexture->getVulkanTexture();
  (void)innerVulkanTexture.imageView_;
  ASSERT_TRUE(innerVulkanTexture.textureId_ != 0);
  auto samplerState = iglDev_->createSamplerState({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  auto* vulkanSamplerState = static_cast<vulkan::SamplerState*>(samplerState.get());
  auto samplerId = vulkanSamplerState->getSamplerId();
  ASSERT_EQ(samplerId, 1);
  ASSERT_FALSE(vulkanSamplerState->isYUV());

  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(cmdQueue, nullptr);
  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  cmdQueue->submit(*cmdBuf);
}

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
TEST_F(DeviceVulkanTest, StagingDeviceLargeBufferTest) {
  Result ret;

  // create a GPU device-local storage buffer large enough to force Vulkan staging device to upload
  // it in multiple chunks
  igl::vulkan::VulkanContext& ctx =
      static_cast<igl::vulkan::Device*>(iglDev_.get())->getVulkanContext();

  const VkDeviceSize kMaxStagingBufferSize = ctx.stagingDevice_->getMaxStagingBufferSize();

  const std::array<VkDeviceSize, 2> kDesiredBufferSizes = {kMaxStagingBufferSize * 2u,
                                                           kMaxStagingBufferSize + 2u};

  size_t maxBufferLength = 0;
  iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxStorageBufferBytes, maxBufferLength);

  for (const auto kDesiredBufferSize : kDesiredBufferSizes) {
    const auto length = std::min<VkDeviceSize>(kDesiredBufferSize, maxBufferLength);

    ASSERT_TRUE(length % 2 == 0);

    const BufferDesc bufferDesc{
        .type = BufferDesc::BufferTypeBits::Storage,
        .length = static_cast<size_t>(length),
        .storage = ResourceStorage::Private,
    };
    const std::shared_ptr<IBuffer> buffer = iglDev_->createBuffer(bufferDesc, &ret);

    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(buffer != nullptr);

    // upload
    {
      std::vector<uint16_t> bufferData(length / 2);

      uint16_t* data = bufferData.data();

      for (size_t i = 0; i != length / 2; i++) {
        data[i] = uint16_t(i & 0xffff);
      }

      ret = buffer->upload(data, BufferRange(length, 0));

      ASSERT_EQ(ret.code, Result::Code::Ok);
    }
    // download
    {
      // map() will create a CPU-copy of data
      const auto* data = static_cast<uint16_t*>(buffer->map(BufferRange(length, 0), &ret));

      ASSERT_EQ(ret.code, Result::Code::Ok);

      for (size_t i = 0; i != length / 2; i++) {
        ASSERT_EQ(data[i], uint16_t(i & 0xffff));
      }

      buffer->unmap();
    }

    ASSERT_EQ(ret.code, Result::Code::Ok);
  }
}

TEST_F(DeviceVulkanTest, DestroyEmptyHandles) {
  igl::destroy(iglDev_.get(), BindGroupTextureHandle{});
  igl::destroy(iglDev_.get(), BindGroupBufferHandle{});
  igl::destroy(iglDev_.get(), TextureHandle{});
  igl::destroy(iglDev_.get(), SamplerHandle{});
  igl::destroy(iglDev_.get(), DepthStencilStateHandle{});
}

TEST_F(DeviceVulkanTest, CurrentThreadIdTest) {
  igl::vulkan::VulkanContext& ctx =
      static_cast<igl::vulkan::Device*>(iglDev_.get())->getVulkanContext();

  ctx.ensureCurrentContextThread();
}

TEST_F(DeviceVulkanTest, EnsureValidation) {
  GTEST_SKIP() << "Some tests are still running without Validation Layers enabled, so this test "
                  "has been temporarily disabled.";

#if !defined(IGL_DISABLE_VALIDATION)
  // @fb-only
  // igl::vulkan::VulkanContext& ctx =
  //   static_cast<igl::vulkan::Device*>(iglDev_.get())->getVulkanContext();

  // ASSERT_TRUE(ctx.areValidationLayersEnabled());
#endif // !defined(IGL_DISABLE_VALIDATION)
}

TEST_F(DeviceVulkanTest, UpdateGlslangResource) {
  const igl::vulkan::VulkanContext& ctx =
      static_cast<const igl::vulkan::Device*>(iglDev_.get())->getVulkanContext();

  ivkUpdateGlslangResource(nullptr, nullptr, nullptr);

  glslang_resource_t res = {};
  const VkPhysicalDeviceProperties& props = ctx.getVkPhysicalDeviceProperties();

  ivkUpdateGlslangResource(&res, &props, nullptr);

  ASSERT_EQ(res.max_vertex_attribs, static_cast<int>(props.limits.maxVertexInputAttributes));
  ASSERT_EQ(res.max_clip_distances, static_cast<int>(props.limits.maxClipDistances));
  ASSERT_EQ(res.max_compute_work_group_count_x,
            static_cast<int>(props.limits.maxComputeWorkGroupCount[0]));
  ASSERT_EQ(res.max_compute_work_group_count_y,
            static_cast<int>(props.limits.maxComputeWorkGroupCount[1]));
  ASSERT_EQ(res.max_compute_work_group_count_z,
            static_cast<int>(props.limits.maxComputeWorkGroupCount[2]));
  ASSERT_EQ(res.max_compute_work_group_size_x,
            static_cast<int>(props.limits.maxComputeWorkGroupSize[0]));
  ASSERT_EQ(res.max_compute_work_group_size_y,
            static_cast<int>(props.limits.maxComputeWorkGroupSize[1]));
  ASSERT_EQ(res.max_compute_work_group_size_z,
            static_cast<int>(props.limits.maxComputeWorkGroupSize[2]));
  ASSERT_EQ(res.max_vertex_output_components,
            static_cast<int>(props.limits.maxVertexOutputComponents));
  ASSERT_EQ(res.max_geometry_input_components,
            static_cast<int>(props.limits.maxGeometryInputComponents));
  ASSERT_EQ(res.max_geometry_output_components,
            static_cast<int>(props.limits.maxGeometryOutputComponents));
  ASSERT_EQ(res.max_fragment_input_components,
            static_cast<int>(props.limits.maxFragmentInputComponents));
  ASSERT_EQ(res.max_geometry_output_vertices,
            static_cast<int>(props.limits.maxGeometryOutputVertices));
  ASSERT_EQ(res.max_geometry_total_output_components,
            static_cast<int>(props.limits.maxGeometryTotalOutputComponents));
  ASSERT_EQ(res.max_tess_control_input_components,
            static_cast<int>(props.limits.maxTessellationControlPerVertexInputComponents));
  ASSERT_EQ(res.max_tess_control_output_components,
            static_cast<int>(props.limits.maxTessellationControlPerVertexOutputComponents));
  ASSERT_EQ(res.max_tess_evaluation_input_components,
            static_cast<int>(props.limits.maxTessellationEvaluationInputComponents));
  ASSERT_EQ(res.max_tess_evaluation_output_components,
            static_cast<int>(props.limits.maxTessellationEvaluationOutputComponents));
  ASSERT_EQ(res.max_viewports, static_cast<int>(props.limits.maxViewports));
  ASSERT_EQ(res.max_cull_distances, static_cast<int>(props.limits.maxCullDistances));
  ASSERT_EQ(res.max_combined_clip_and_cull_distances,
            static_cast<int>(props.limits.maxCombinedClipAndCullDistances));
}

namespace {
constexpr const char* kCountedFragmentShader = R"(#version 460
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(1.0, 0.5, 0.25, 1.0);
}
)";

std::vector<uint32_t> compileCountedFragmentShader() {
  glslang_resource_t resource = {};
  glslangGetDefaultResource(&resource);
  std::vector<uint32_t> spirv;
  const Result result =
      glslang::compileShader(ShaderStage::Fragment, kCountedFragmentShader, spirv, &resource);
  EXPECT_TRUE(result.isOk()) << result.message.c_str();
  return spirv;
}
} // namespace

// getShaderCompilationCount() counts shaders compiled from source. A shader module or library
// created from a SPIR-V binary is not compiled, so it leaves the count unchanged.
TEST_F(DeviceVulkanTest, ShaderCompilationCountIgnoresBinaryModule) {
  const std::vector<uint32_t> spirv = compileCountedFragmentShader();
  ASSERT_FALSE(spirv.empty());
  const size_t countBefore = iglDev_->getShaderCompilationCount();

  Result ret;
  auto module =
      ShaderModuleCreator::fromBinaryInput(*iglDev_,
                                           spirv.data(),
                                           spirv.size() * sizeof(uint32_t),
                                           {.stage = ShaderStage::Fragment, .entryPoint = "main"},
                                           "ShaderCompilationCountIgnoresBinaryModule",
                                           &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(module, nullptr);

  EXPECT_EQ(iglDev_->getShaderCompilationCount(), countBefore);
}

TEST_F(DeviceVulkanTest, ShaderCompilationCountIgnoresBinaryLibrary) {
  const std::vector<uint32_t> spirv = compileCountedFragmentShader();
  ASSERT_FALSE(spirv.empty());
  const size_t countBefore = iglDev_->getShaderCompilationCount();

  Result ret;
  auto library = ShaderLibraryCreator::fromBinaryInput(
      *iglDev_,
      spirv.data(),
      spirv.size() * sizeof(uint32_t),
      {{.stage = ShaderStage::Fragment, .entryPoint = "main"}},
      "ShaderCompilationCountIgnoresBinaryLibrary",
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(library, nullptr);

  EXPECT_EQ(iglDev_->getShaderCompilationCount(), countBefore);
}

TEST_F(DeviceVulkanTest, ShaderCompilationCountCountsSourceLibrary) {
  const size_t countBefore = iglDev_->getShaderCompilationCount();

  Result ret;
  auto library = ShaderLibraryCreator::fromStringInput(
      *iglDev_,
      kCountedFragmentShader,
      {{.stage = ShaderStage::Fragment, .entryPoint = "main"}},
      "ShaderCompilationCountCountsSourceLibrary",
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(library, nullptr);

  EXPECT_EQ(iglDev_->getShaderCompilationCount(), countBefore + 1);
}

// Diff-2 plumbing (unit): getSpvOptions() maps each ShaderOptimization onto the exact glslang
// SPIR-V flags from the spec. Default reproduces IGL's historical hardcoded options byte-for-byte
// (optimizer on, optimize-for-size) so existing clients are unaffected; the debug path disables the
// optimizer; the release path optimizes for performance (not size). generate_debug_info stays true
// in every mode, so debug info is retained for both debug and release builds. Pure mapping logic —
// no glslang process or Vulkan device required.
GTEST_TEST(GlslCompilerOptimization, SpvOptionsMapping) {
  const glslang_spv_options_t defaultOpts =
      glslang::getSpvOptions(ShaderCompilerOptions{.optimization = ShaderOptimization::Default});
  EXPECT_TRUE(defaultOpts.generate_debug_info);
  EXPECT_FALSE(defaultOpts.strip_debug_info);
  EXPECT_FALSE(defaultOpts.disable_optimizer);
  EXPECT_TRUE(defaultOpts.optimize_size);

  const glslang_spv_options_t noOpt =
      glslang::getSpvOptions(ShaderCompilerOptions{.optimization = ShaderOptimization::NoOpt});
  EXPECT_TRUE(noOpt.generate_debug_info); // debug info retained in debug builds
  EXPECT_TRUE(noOpt.disable_optimizer); // optimizer off for clean stepping + reference numerics

  const glslang_spv_options_t performance = glslang::getSpvOptions(
      ShaderCompilerOptions{.optimization = ShaderOptimization::Performance});
  EXPECT_TRUE(performance.generate_debug_info); // debug info retained in release/profiling builds
  EXPECT_FALSE(performance.disable_optimizer); // optimizer on
  EXPECT_FALSE(performance.optimize_size); // optimize for performance, not size (PRIMARY goal)
}

// Diff-2 plumbing (integration): every ShaderOptimization compiles end-to-end through
// glslang::compileShader() to valid, non-empty SPIR-V via the new options parameter. Whether the
// SPIRV-Tools optimizer alters the bytes depends on the glslang build + debug-info config; the
// release-vs-debug GPU perf delta is validated on-device per the diff test plan, not here.
GTEST_TEST(GlslCompilerOptimization, CompilesValidSpirvForEachOption) {
  igl::glslang::initializeCompiler();

  glslang_resource_t resource = {};
  glslangGetDefaultResource(&resource);

  const char* source = R"(#version 460
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(1.0, 0.5, 0.25, 1.0);
}
)";

  constexpr uint32_t kSpirvMagic = 0x07230203;
  for (const ShaderOptimization optimization :
       {ShaderOptimization::Default, ShaderOptimization::NoOpt, ShaderOptimization::Performance}) {
    ShaderCompilerOptions options;
    options.optimization = optimization;
    std::vector<uint32_t> spirv;
    const Result result =
        glslang::compileShader(ShaderStage::Fragment, source, spirv, &resource, options);
    EXPECT_TRUE(result.isOk()) << result.message.c_str();
    ASSERT_FALSE(spirv.empty());
    EXPECT_EQ(spirv.front(), kSpirvMagic);
  }

  igl::glslang::finalizeCompiler();
}

// SPIR-V 1.5 is rejected on pre-1.2 devices (VUID-VkShaderModuleCreateInfo-pCode-08737),
// so the compile target must be clampable: the default stays 1.5 (byte-for-byte stability)
// while an explicit Vulkan 1.1 / SPIR-V 1.3 target emits version word 0x10300.
GTEST_TEST(GlslCompilerTargetVersion, DefaultAndClampedSpirvVersion) {
  igl::glslang::initializeCompiler();

  glslang_resource_t resource = {};
  glslangGetDefaultResource(&resource);

  const char* source = R"(#version 460
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(1.0, 0.5, 0.25, 1.0);
}
)";

  constexpr uint32_t kSpirvMagic = 0x07230203;
  std::vector<uint32_t> spirvDefault;
  EXPECT_TRUE(
      glslang::compileShader(ShaderStage::Fragment, source, spirvDefault, &resource).isOk());
  ASSERT_GT(spirvDefault.size(), 1u);
  EXPECT_EQ(spirvDefault.front(), kSpirvMagic);
  EXPECT_EQ(spirvDefault[1], 0x00010500u);

  std::vector<uint32_t> spirvClamped;
  const glslang::GlslTargetVersion target11 = {.clientVersion = GLSLANG_TARGET_VULKAN_1_1,
                                               .spirvVersion = GLSLANG_TARGET_SPV_1_3};
  EXPECT_TRUE(
      glslang::compileShader(ShaderStage::Fragment, source, spirvClamped, &resource, {}, target11)
          .isOk());
  ASSERT_GT(spirvClamped.size(), 1u);
  EXPECT_EQ(spirvClamped.front(), kSpirvMagic);
  EXPECT_EQ(spirvClamped[1], 0x00010300u);

  igl::glslang::finalizeCompiler();
}

// glslTargetForVulkanVersion() maps each Vulkan API version to the newest SPIR-V target it
// accepts: pre-1.1 (including unknown 0.0) to Vulkan 1.0 / SPIR-V 1.0, 1.x<1.2 to Vulkan 1.1 /
// SPIR-V 1.3, and 1.2+ to the default. Pure mapping logic - no compiler or device required.
GTEST_TEST(GlslTargetVersion, MapsVulkanApiBoundaries) {
  using glslang::glslTargetForVulkanVersion;
  using glslang::GlslTargetVersion;

  const GlslTargetVersion v00 = glslTargetForVulkanVersion(0, 0);
  EXPECT_EQ(v00.clientVersion, GLSLANG_TARGET_VULKAN_1_0);
  EXPECT_EQ(v00.spirvVersion, GLSLANG_TARGET_SPV_1_0);

  const GlslTargetVersion v10 = glslTargetForVulkanVersion(1, 0);
  EXPECT_EQ(v10.clientVersion, GLSLANG_TARGET_VULKAN_1_0);
  EXPECT_EQ(v10.spirvVersion, GLSLANG_TARGET_SPV_1_0);

  const GlslTargetVersion v11 = glslTargetForVulkanVersion(1, 1);
  EXPECT_EQ(v11.clientVersion, GLSLANG_TARGET_VULKAN_1_1);
  EXPECT_EQ(v11.spirvVersion, GLSLANG_TARGET_SPV_1_3);

  const GlslTargetVersion v12 = glslTargetForVulkanVersion(1, 2);
  EXPECT_EQ(v12.clientVersion, GLSLANG_TARGET_VULKAN_1_2);
  EXPECT_EQ(v12.spirvVersion, GLSLANG_TARGET_SPV_1_5);

  const GlslTargetVersion v13 = glslTargetForVulkanVersion(1, 3);
  EXPECT_EQ(v13.clientVersion, GLSLANG_TARGET_VULKAN_1_2);
  EXPECT_EQ(v13.spirvVersion, GLSLANG_TARGET_SPV_1_5);
}

TEST_F(DeviceVulkanTest, BufferDeviceAddress) {
  const igl::vulkan::VulkanContext& ctx =
      static_cast<const igl::vulkan::Device*>(iglDev_.get())->getVulkanContext();

  if (!ctx.features().has_VK_KHR_buffer_device_address) {
    return;
  }

  Result ret;

  auto buffer = iglDev_->createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Uniform,
          .length = 256,
          .storage = ResourceStorage::Shared,
      },
      &ret);

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);

  if (!buffer) {
    return;
  }

  ASSERT_NE(buffer->gpuAddress(), 0u);
}

GTEST_TEST(VulkanContext, DescriptorIndexing) {
  const igl::vulkan::VulkanContextConfig config = {
#if IGL_PLATFORM_MACOSX
      .terminateOnValidationError = false,
#elif IGL_DEBUG
      .terminateOnValidationError = true,
#ifdef IGL_DISABLE_VALIDATION
      .enableValidation = false,
#else
      .enableValidation = true,
#endif // IGL_DISABLE_VALIDATION
#else
      .terminateOnValidationError = false,
#ifdef IGL_DISABLE_VALIDATION
      .enableValidation = false,
#else
      .enableValidation = true,
#endif // IGL_DISABLE_VALIDATION
#endif
      .enableExtraLogs = true,
      .enableDescriptorIndexing = true,
  };

  std::unique_ptr<igl::vulkan::Device> iglDev = nullptr;

  auto ctx = igl::vulkan::HWDevice::createContext(config, nullptr);
  ASSERT_NE(ctx, nullptr);

  Result ret;

  std::vector<HWDeviceDesc> devices =
      igl::vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::Unknown), &ret);

  ASSERT_TRUE(!devices.empty());

  if (ret.isOk()) {
    igl::vulkan::VulkanFeatures features(config);
    features.populateWithAvailablePhysicalDeviceFeatures(
        *ctx,
        (VkPhysicalDevice)devices[0].guid); // NOLINT(performance-no-int-to-ptr)

    const VkPhysicalDeviceDescriptorIndexingFeaturesEXT& dif = features.featuresDescriptorIndexing;
    if (!dif.shaderSampledImageArrayNonUniformIndexing ||
        !dif.descriptorBindingUniformBufferUpdateAfterBind ||
        !dif.descriptorBindingSampledImageUpdateAfterBind ||
        !dif.descriptorBindingStorageImageUpdateAfterBind ||
        !dif.descriptorBindingStorageBufferUpdateAfterBind ||
        !dif.descriptorBindingUpdateUnusedWhilePending || !dif.descriptorBindingPartiallyBound ||
        !dif.runtimeDescriptorArray) {
      return;
    }

    iglDev = igl::vulkan::HWDevice::create(std::move(ctx),
                                           devices[0],
                                           0, // width
                                           0, // height,
                                           0,
                                           nullptr,
                                           &features,
                                           "VulkanContext Test",
                                           &ret);
    if (!ret.isOk()) {
      iglDev = nullptr;
    }
  }

  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(iglDev, nullptr);

  if (!iglDev) {
    return;
  }

  const TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                 1,
                                                 1,
                                                 TextureDesc::TextureUsageBits::Sampled |
                                                     TextureDesc::TextureUsageBits::Attachment);

  auto texture = iglDev->createTexture(texDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_NE(texture, nullptr);

  if (!texture) {
    return;
  }

  ASSERT_NE(texture->getTextureId(), 0u);
}

TEST_F(DeviceVulkanTest, UniformBlockRingBufferTest) {
  // The rotation below is what the Ring hint buys, so it is only meaningful on a device that
  // implements it. DeviceFeatureSetTest is where the Vulkan answer itself is pinned; skipping here
  // keeps this a behavior test rather than a second capability assertion.
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }

  Result ret;

  // Create uniform buffer with ring buffer hint
  const size_t bufferSize = 256;
  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Uniform,
      .length = bufferSize,
      .storage = ResourceStorage::Shared,
      .hint = BufferDesc::BufferAPIHintBits::Ring | BufferDesc::BufferAPIHintBits::UniformBlock,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(buffer, nullptr);

  // Upload and verify data
  std::vector<uint32_t> testData(bufferSize / sizeof(uint32_t));
  for (unsigned int& i : testData) {
    // NOLINTNEXTLINE(cert-msc50-cpp, facebook-hte-BadCall-rand)
    i = rand();
  }

  ret = buffer->upload(testData.data(), BufferRange(bufferSize, 0));
  ASSERT_TRUE(ret.isOk());

  // Create and submit multiple command buffers
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  std::vector<VkBuffer> bufferHandles;
  // By default the VulkanContextConfig.maxResourceCount is 3, so we should create at most 3 unique
  // VkBuffers. getVkBuffer() is the read that makes each submit move to the next copy.
  for (int i = 0; i < 4; i++) {
    auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk());

    auto* vulkanBufferCast = static_cast<vulkan::Buffer*>(buffer.get());
    bufferHandles.push_back(vulkanBufferCast->getVkBuffer());
    ASSERT_EQ(vulkanBufferCast->currentVulkanBuffer()->getSize(), bufferSize);

    cmdQueue->submit(*cmdBuf);
  }

  // Verify the first 3 buffer handles are mutually distinct. Compared pairwise rather than only
  // consecutively, so a ring that got shallower fails on the distinctness itself instead of only
  // on the wrap check below.
  ASSERT_NE(bufferHandles[0], bufferHandles[1]);
  ASSERT_NE(bufferHandles[0], bufferHandles[2]);
  ASSERT_NE(bufferHandles[1], bufferHandles[2]);
  // First and last handles should be the same
  ASSERT_EQ(bufferHandles[3], bufferHandles[0]);
}

// Control for UniformBlockRingBufferTest: the rotation there is what the Ring hint buys, not
// something every uniform buffer already gets. Without the hint the same VkBuffer is handed out on
// every frame, which is the single-slot exposure a caller avoids by gating on BufferRing.
TEST_F(DeviceVulkanTest, UniformBlockWithoutRingHintKeepsOneBuffer) {
  Result ret;

  const size_t bufferSize = 256;
  const BufferDesc bufferDesc{
      .type = BufferDesc::BufferTypeBits::Uniform,
      .length = bufferSize,
      .storage = ResourceStorage::Shared,
      .hint = BufferDesc::BufferAPIHintBits::UniformBlock,
  };
  auto buffer = iglDev_->createBuffer(bufferDesc, &ret);
  ASSERT_TRUE(ret.isOk());
  ASSERT_NE(buffer, nullptr);

  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto* vulkanBuffer = static_cast<vulkan::Buffer*>(buffer.get());
  const VkBuffer firstHandle = vulkanBuffer->currentVulkanBuffer()->getVkBuffer();
  for (int i = 0; i < 4; i++) {
    auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk());

    EXPECT_EQ(vulkanBuffer->currentVulkanBuffer()->getVkBuffer(), firstHandle);

    cmdQueue->submit(*cmdBuf);
  }
}

// End-to-end version of UniformBlockRingBufferTest: the descriptor is the one ManagedUniformBuffer
// builds rather than one the test writes, so it also covers the uniform buffer carrying initial
// data, which is how every IGPrism FilterNode allocates its params block.
TEST_F(DeviceVulkanTest, ManagedUniformBufferRotatesItsRingBuffer) {
  // Same gate as UniformBlockRingBufferTest: the rotation below is only meaningful on a device
  // that implements the hint, and the Vulkan answer itself is pinned by DeviceFeatureSetTest.
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }

  // ManagedUniformBuffer keeps its buffer private, so the creation request is routed through a
  // recording device that forwards to the real one and keeps a handle on what came back.
  RecordingDevice device(*iglDev_);
  device.reportsBufferRing = true;
  const iglu::ManagedUniformBuffer uniforms(device, {.index = 0, .length = 256});
  ASSERT_TRUE(uniforms.result.isOk());
  ASSERT_EQ(device.createBufferCount, 1u);
  ASSERT_NE(device.recordedHint & BufferDesc::BufferAPIHintBits::Ring, 0);
  ASSERT_NE(device.createdBuffer, nullptr);

  Result ret;
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto* vulkanBuffer = static_cast<vulkan::Buffer*>(device.createdBuffer);
  std::vector<VkBuffer> bufferHandles;
  for (int i = 0; i < 4; i++) {
    auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk());

    bufferHandles.push_back(vulkanBuffer->getVkBuffer());

    cmdQueue->submit(*cmdBuf);
  }

  // maxResourceCount defaults to 3: the three slots are mutually distinct, then the ring wraps.
  // Compared pairwise rather than only consecutively, so a ring that got shallower fails on the
  // distinctness itself instead of only on the wrap check below.
  EXPECT_NE(bufferHandles[0], bufferHandles[1]);
  EXPECT_NE(bufferHandles[0], bufferHandles[2]);
  EXPECT_NE(bufferHandles[1], bufferHandles[2]);
  EXPECT_EQ(bufferHandles[3], bufferHandles[0]);
}

namespace {

std::shared_ptr<IBuffer> createRingUniformBuffer(IDevice& device, Result* outResult) {
  return device.createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Uniform,
          .length = 256,
          .storage = ResourceStorage::Shared,
          .hint = BufferDesc::BufferAPIHintBits::Ring | BufferDesc::BufferAPIHintBits::UniformBlock,
      },
      outResult);
}

// Observes the current copy without marking it as read.
VkBuffer currentCopy(const IBuffer& buffer) {
  return static_cast<const vulkan::Buffer&>(buffer).currentVulkanBuffer()->getVkBuffer();
}

} // namespace

// A submit that read no Ring buffer copy must not move to the next copy, even while a Ring buffer
// is alive.
TEST_F(DeviceVulkanTest, SubmitsWithoutRingReadsDoNotRotate) {
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }
  Result ret;
  auto buffer = createRingUniformBuffer(*iglDev_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const VkBuffer firstCopy = currentCopy(*buffer);
  for (int i = 0; i < 5; i++) {
    auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk());
    cmdQueue->submit(*cmdBuf);
    EXPECT_EQ(currentCopy(*buffer), firstCopy) << "submit " << i;
  }
}

// Only submits that read a Ring buffer copy advance the ring; submits in between leave it alone.
TEST_F(DeviceVulkanTest, MixedSubmitsRotateOnlyOnRingReads) {
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }
  Result ret;
  auto buffer = createRingUniformBuffer(*iglDev_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const VkBuffer firstCopy = currentCopy(*buffer);
  VkBuffer previousCopy = firstCopy;
  constexpr std::array<bool, 6> kReadsRing = {true, false, false, true, false, true};
  for (size_t i = 0; i < kReadsRing.size(); i++) {
    auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
    ASSERT_TRUE(ret.isOk());
    if (kReadsRing[i]) {
      EXPECT_EQ(static_cast<vulkan::Buffer&>(*buffer).getVkBuffer(), previousCopy);
    }
    cmdQueue->submit(*cmdBuf);

    const VkBuffer copy = currentCopy(*buffer);
    if (kReadsRing[i]) {
      EXPECT_NE(copy, previousCopy) << "submit " << i;
    } else {
      EXPECT_EQ(copy, previousCopy) << "submit " << i;
    }
    previousCopy = copy;
  }
  // Three Ring reads with maxResourceCount = 3 make a full lap.
  EXPECT_EQ(previousCopy, firstCopy);
}

// upload() writes the current copy but does not read it on the GPU, so a following submit keeps
// the copy, and the uploaded data is still what the next read sees.
TEST_F(DeviceVulkanTest, UploadOnlySubmitDoesNotRotate) {
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }
  Result ret;
  auto buffer = createRingUniformBuffer(*iglDev_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  std::array<uint32_t, 64> data{};
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = 0x12345678u + static_cast<uint32_t>(i);
  }
  const VkBuffer firstCopy = currentCopy(*buffer);
  ASSERT_TRUE(buffer->upload(data.data(), BufferRange(sizeof(data), 0)).isOk());

  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  cmdQueue->submit(*cmdBuf);

  EXPECT_EQ(currentCopy(*buffer), firstCopy);
  const auto& copy = static_cast<vulkan::Buffer&>(*buffer).currentVulkanBuffer();
  if (copy->isMapped()) {
    EXPECT_EQ(std::memcmp(copy->getMappedPtr(), data.data(), sizeof(data)), 0);
  }
}

// bindBuffer() on an encoder's binder is how uniform blocks reach a draw; it must count as a read.
TEST_F(DeviceVulkanTest, BindBufferMarksRingRead) {
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }
  Result ret;
  auto buffer = createRingUniformBuffer(*iglDev_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  auto& ctx = static_cast<vulkan::Device&>(*iglDev_).getVulkanContext();
  const VkBuffer firstCopy = currentCopy(*buffer);
  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  {
    vulkan::ResourcesBinder binder(nullptr, ctx, VK_PIPELINE_BIND_POINT_GRAPHICS);
    binder.bindBuffer(0, static_cast<vulkan::Buffer*>(buffer.get()), 0, 0);
  }
  cmdQueue->submit(*cmdBuf);

  EXPECT_NE(currentCopy(*buffer), firstCopy);
}

// gpuAddress() hands the current copy to shaders through buffer device addresses, so it counts as
// a read too.
TEST_F(DeviceVulkanTest, GpuAddressMarksRingRead) {
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }
  const auto& ctx = static_cast<const vulkan::Device&>(*iglDev_).getVulkanContext();
  if (!ctx.features().has_VK_KHR_buffer_device_address) {
    GTEST_SKIP() << "VK_KHR_buffer_device_address is not available";
  }
  Result ret;
  auto buffer = createRingUniformBuffer(*iglDev_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(buffer, nullptr);
  auto cmdQueue = iglDev_->createCommandQueue({}, &ret);
  ASSERT_TRUE(ret.isOk());

  const VkBuffer firstCopy = currentCopy(*buffer);
  auto cmdBuf = cmdQueue->createCommandBuffer({}, &ret);
  ASSERT_TRUE(ret.isOk());
  EXPECT_NE(buffer->gpuAddress(), 0u);
  cmdQueue->submit(*cmdBuf);

  EXPECT_NE(currentCopy(*buffer), firstCopy);
}

namespace {
bool sBindGroupDebugAbortFired = false;
} // namespace

// A bind group writes its descriptors once, so it would pin one Ring buffer copy. Creating one with
// a Ring buffer must fail, and trip the debug check where debug aborts are enabled.
TEST_F(DeviceVulkanTest, RingBufferInBindGroupIsRejected) {
  if (!iglDev_->hasFeature(DeviceFeatures::BufferRing)) {
    GTEST_SKIP() << "Device does not report DeviceFeatures::BufferRing";
  }
  Result ret;
  auto ringBuffer = createRingUniformBuffer(*iglDev_, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  const std::shared_ptr<IBuffer> plainBuffer = iglDev_->createBuffer(
      BufferDesc{
          .type = BufferDesc::BufferTypeBits::Uniform,
          .length = 256,
          .storage = ResourceStorage::Shared,
          .hint = BufferDesc::BufferAPIHintBits::UniformBlock,
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();

  sBindGroupDebugAbortFired = false;
  iglSetDebugAbortListener([](const char* /*category*/,
                              const char* /*reason*/,
                              const char* /*file*/,
                              const char* /*func*/,
                              int /*line*/,
                              const char* /*format*/,
                              va_list /*ap*/) { sBindGroupDebugAbortFired = true; });

  BindGroupBufferDesc plainDesc;
  plainDesc.buffers[0] = plainBuffer;
  plainDesc.debugName = "plain";
  {
    auto plainGroup = iglDev_->createBindGroup(plainDesc, &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message.c_str();
    EXPECT_FALSE(plainGroup.empty());
  }
  EXPECT_FALSE(sBindGroupDebugAbortFired) << "A bind group without a Ring buffer must be accepted";

  BindGroupBufferDesc ringDesc;
  ringDesc.buffers[0] = plainBuffer;
  ringDesc.buffers[1] = ringBuffer;
  ringDesc.debugName = "ring";
  {
    auto ringGroup = iglDev_->createBindGroup(ringDesc, &ret);
    EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
    EXPECT_TRUE(ringGroup.empty());
  }
#if IGL_DEBUG_ABORT_ENABLED
  EXPECT_TRUE(sBindGroupDebugAbortFired)
      << "A Ring buffer in a bind group must trip the debug check";
#endif

  iglSetDebugAbortListener(nullptr);
}
#endif

} // namespace igl::tests
