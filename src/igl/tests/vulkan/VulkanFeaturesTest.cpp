/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/vulkan/VulkanFeatures.h>

#include <algorithm>
#include <cstring>
#include <vector>
#include <igl/tests/util/device/vulkan/TestDevice.h>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/VulkanContext.h>

#if IGL_PLATFORM_MACOSX
#include <igl/vulkan/moltenvk/MoltenVkHelpers.h>
#endif

namespace igl::tests {

class VulkanApiVersionTest : public ::testing::Test {};

TEST_F(VulkanApiVersionTest, RejectsVulkan10) {
  EXPECT_EQ(vulkan::checkVulkanApiVersion(VK_API_VERSION_1_0).code, Result::Code::Unsupported);
  const Result result = vulkan::checkVulkanApiVersion(VK_MAKE_API_VERSION(0, 1, 0, 99));
  EXPECT_EQ(result.code, Result::Code::Unsupported);
  EXPECT_EQ(result.message, "Vulkan 1.1 or newer is required (found 1.0.99)");
  EXPECT_EQ(vulkan::checkVulkanApiVersion(VK_MAKE_API_VERSION(0, 1, 0, 4095)).code,
            Result::Code::Unsupported);
}

TEST_F(VulkanApiVersionTest, AcceptsVulkan11AndNewer) {
  EXPECT_TRUE(vulkan::checkVulkanApiVersion(VK_API_VERSION_1_1).isOk());
  EXPECT_TRUE(vulkan::checkVulkanApiVersion(VK_MAKE_API_VERSION(0, 1, 1, 4095)).isOk());
  EXPECT_TRUE(vulkan::checkVulkanApiVersion(VK_API_VERSION_1_2).isOk());
  EXPECT_TRUE(vulkan::checkVulkanApiVersion(VK_API_VERSION_1_3).isOk());
}

TEST_F(VulkanApiVersionTest, CreateRejectsMissingContext) {
  Result result;
  const auto device = vulkan::HWDevice::create(
      nullptr, HWDeviceDesc(0, HWDeviceType::Unknown), 0, 0, 0, nullptr, nullptr, nullptr, &result);
  EXPECT_EQ(device, nullptr);
  EXPECT_EQ(result.code, Result::Code::Unsupported);
}

#if IGL_PLATFORM_MACOSX && !defined(IGL_USE_STATIC_KOSMICKRISP)
TEST_F(VulkanApiVersionTest, InstanceCreationFailureKeepsContext) {
  vulkan::setupMoltenVKEnvironment();
  const vulkan::VulkanContextConfig config = {.enableValidation = false, .enableExtraLogs = false};
  {
    const auto context = vulkan::HWDevice::createContext(config, nullptr);
    ASSERT_NE(context, nullptr);
    uint32_t count = 0;
    ASSERT_EQ(context->vf_.vkEnumerateInstanceLayerProperties(&count, nullptr), VK_SUCCESS);
    std::vector<VkLayerProperties> layers(count);
    ASSERT_EQ(context->vf_.vkEnumerateInstanceLayerProperties(&count, layers.data()), VK_SUCCESS);
    if (std::any_of(layers.begin(), layers.end(), [](const auto& layer) {
          return std::strcmp(layer.layerName, "VK_LAYER_LUNARG_gfxreconstruct") == 0;
        })) {
      GTEST_SKIP() << "The gfxreconstruct layer is available";
    }
  }
  auto failedConfig = config;
  failedConfig.enableGfxReconstruct = true;
  const bool debugBreakEnabled = igl::isDebugBreakEnabled();
  igl::setDebugBreakEnabled(false);
  const auto context = vulkan::HWDevice::createContext(failedConfig, nullptr);
  igl::setDebugBreakEnabled(debugBreakEnabled);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->getVkInstance(), VK_NULL_HANDLE);
  Result result;
  const auto devices =
      vulkan::HWDevice::queryDevices(*context, HWDeviceQueryDesc(HWDeviceType::Unknown), &result);
  EXPECT_TRUE(devices.empty());
  EXPECT_EQ(result.code, Result::Code::Unsupported);
}
#endif

//
// VulkanFeaturesTest
//
// Unit tests for VulkanFeatures
//

// Constructing ***********************************************************************
class VulkanFeaturesTest : public ::testing::Test {};

TEST_F(VulkanFeaturesTest, Construct) {
  const igl::vulkan::VulkanContextConfig config;

  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_EQ(features.featuresShaderFloat16Int8.pNext, nullptr);
}

TEST_F(VulkanFeaturesTest, MissingFeatureQueryClearsCachedValues) {
#if IGL_PLATFORM_MACOSX
  vulkan::setupMoltenVKEnvironment();
#endif
  const vulkan::VulkanContextConfig config = {.enableValidation = false, .enableExtraLogs = false};
  const auto context = vulkan::HWDevice::createContext(config, nullptr);
  ASSERT_NE(context, nullptr);
  vulkan::VulkanFeatures features(config);
  features.featuresShaderFloat16Int8.shaderFloat16 = VK_TRUE;
  auto& functions = const_cast<VulkanFunctionTable&>(context->vf_);
  const PFN_vkGetPhysicalDeviceFeatures2 getFeatures = functions.vkGetPhysicalDeviceFeatures2;
  functions.vkGetPhysicalDeviceFeatures2 = nullptr;
  features.populateWithAvailablePhysicalDeviceFeatures(*context, VK_NULL_HANDLE);
  functions.vkGetPhysicalDeviceFeatures2 = getFeatures;
  const VkPhysicalDeviceFeatures empty{};
  EXPECT_EQ(std::memcmp(&features.vkPhysicalDeviceFeatures2.features, &empty, sizeof(empty)), 0);
  EXPECT_EQ(features.featuresMultiview.multiview, VK_FALSE);
  EXPECT_EQ(features.featuresSamplerYcbcrConversion.samplerYcbcrConversion, VK_FALSE);
  EXPECT_EQ(features.featuresBufferDeviceAddress.bufferDeviceAddress, VK_FALSE);
  EXPECT_EQ(features.featuresShaderFloat16Int8.shaderFloat16, VK_FALSE);
  EXPECT_EQ(features.vkPhysicalDeviceFeatures2.sType, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
  EXPECT_NE(features.vkPhysicalDeviceFeatures2.pNext, nullptr);
}

// Copying ***************************************************************************
TEST_F(VulkanFeaturesTest, CopyNotPerformed) {
  const igl::vulkan::VulkanContextConfig configSrc;
  EXPECT_FALSE(configSrc.enableDescriptorIndexing);

  igl::vulkan::VulkanContextConfig configDst;
  configDst.enableDescriptorIndexing = true;
  EXPECT_TRUE(configDst.enableDescriptorIndexing);

  const igl::vulkan::VulkanFeatures featuresSrc(configSrc);
  igl::vulkan::VulkanFeatures featuresDst(configDst);

  featuresDst = featuresSrc;

  // Unchanged
  EXPECT_TRUE(configDst.enableDescriptorIndexing);
}

// Copy with Same Config **************************************************************
TEST_F(VulkanFeaturesTest, CopyPerformed) {
  const igl::vulkan::VulkanContextConfig config;

  igl::vulkan::VulkanFeatures featuresSrc(config);
  featuresSrc.vkPhysicalDeviceFeatures2.features.dualSrcBlend = VK_FALSE;
  featuresSrc.featuresMultiview.multiview = VK_FALSE;
  featuresSrc.featuresSynchronization2.synchronization2 = VK_FALSE;

  igl::vulkan::VulkanFeatures featuresDst(config);
  EXPECT_TRUE(featuresDst.vkPhysicalDeviceFeatures2.features.dualSrcBlend);
  EXPECT_TRUE(featuresDst.featuresMultiview.multiview);
  EXPECT_TRUE(featuresDst.featuresSynchronization2.synchronization2);

  featuresDst = featuresSrc;

  EXPECT_FALSE(featuresDst.vkPhysicalDeviceFeatures2.features.dualSrcBlend);
  EXPECT_FALSE(featuresDst.featuresMultiview.multiview);
  EXPECT_FALSE(featuresDst.featuresSynchronization2.synchronization2);
}

// Self-Assignment ******************************************************************
TEST_F(VulkanFeaturesTest, SelfAssignment) {
  const igl::vulkan::VulkanContextConfig config;

  igl::vulkan::VulkanFeatures features(config);
  const VkBool32 originalDualSrcBlend = features.vkPhysicalDeviceFeatures2.features.dualSrcBlend;

  igl::vulkan::VulkanFeatures& ref = features;
  features = ref;

  EXPECT_EQ(features.vkPhysicalDeviceFeatures2.features.dualSrcBlend, originalDualSrcBlend);
}

// Check Selected Features ****************************************************
TEST_F(VulkanFeaturesTest, CheckSelectedFeaturesAllPresent) {
  const igl::vulkan::VulkanContextConfig config;

  const igl::vulkan::VulkanFeatures requested(config);
  const igl::vulkan::VulkanFeatures available(config);

  const Result result = requested.checkSelectedFeatures(available);
  EXPECT_TRUE(result.isOk());
}

// Extension Bool Defaults ****************************************************
TEST_F(VulkanFeaturesTest, ExtensionBoolsDefaultToFalse) {
  const igl::vulkan::VulkanContextConfig config;
  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_FALSE(features.has_VK_EXT_descriptor_buffer);
  EXPECT_FALSE(features.has_VK_EXT_descriptor_indexing);
  EXPECT_FALSE(features.has_VK_EXT_fragment_density_map);
  EXPECT_FALSE(features.has_VK_EXT_headless_surface);
  EXPECT_FALSE(features.has_VK_EXT_index_type_uint8);
  EXPECT_FALSE(features.has_VK_EXT_mesh_shader);
  EXPECT_FALSE(features.has_VK_EXT_queue_family_foreign);
  EXPECT_FALSE(features.has_VK_KHR_8bit_storage);
  EXPECT_FALSE(features.has_VK_KHR_buffer_device_address);
  EXPECT_FALSE(features.has_VK_KHR_get_surface_capabilities2);
  EXPECT_FALSE(features.has_VK_KHR_portability_enumeration);
  EXPECT_FALSE(features.has_VK_KHR_shader_non_semantic_info);
  EXPECT_FALSE(features.has_VK_KHR_synchronization2);
  EXPECT_FALSE(features.has_VK_KHR_timeline_semaphore);
  EXPECT_FALSE(features.has_VK_KHR_uniform_buffer_standard_layout);
  EXPECT_FALSE(features.has_VK_KHR_vulkan_memory_model);
  EXPECT_FALSE(features.has_VK_QCOM_multiview_per_view_viewports);
}

// Enable Default Features ****************************************************
TEST_F(VulkanFeaturesTest, EnableDefaultFeatures) {
  const igl::vulkan::VulkanContextConfig config;

  igl::vulkan::VulkanFeatures features(config);

  EXPECT_FALSE(features.featuresShaderFloat16Int8.shaderFloat16);
  EXPECT_FALSE(features.featuresShaderFloat16Int8.shaderInt8);

  EXPECT_TRUE(features.vkPhysicalDeviceFeatures2.features.dualSrcBlend);
  EXPECT_TRUE(features.vkPhysicalDeviceFeatures2.features.shaderInt16);
  EXPECT_TRUE(features.vkPhysicalDeviceFeatures2.features.multiDrawIndirect);
  EXPECT_TRUE(features.vkPhysicalDeviceFeatures2.features.drawIndirectFirstInstance);
  EXPECT_TRUE(features.vkPhysicalDeviceFeatures2.features.depthBiasClamp);
#if IGL_PLATFORM_ANDROID
  // fillModeNonSolid is not well supported on Android, only enable by default when it's not Android
  EXPECT_FALSE(features.vkPhysicalDeviceFeatures2.features.fillModeNonSolid);
#else
  EXPECT_TRUE(features.vkPhysicalDeviceFeatures2.features.fillModeNonSolid);
#endif

  {
    const auto& dif = features.featuresDescriptorIndexing;
    ASSERT_FALSE(config.enableDescriptorIndexing);
    EXPECT_TRUE(dif.shaderSampledImageArrayNonUniformIndexing);
    EXPECT_TRUE(dif.descriptorBindingUniformBufferUpdateAfterBind);
    EXPECT_TRUE(dif.descriptorBindingSampledImageUpdateAfterBind);
    EXPECT_TRUE(dif.descriptorBindingStorageImageUpdateAfterBind);
    EXPECT_TRUE(dif.descriptorBindingStorageBufferUpdateAfterBind);
    EXPECT_TRUE(dif.descriptorBindingUpdateUnusedWhilePending);
    EXPECT_TRUE(dif.descriptorBindingPartiallyBound);
    EXPECT_TRUE(dif.runtimeDescriptorArray);

    EXPECT_TRUE(features.features16BitStorage.storageBuffer16BitAccess);

    EXPECT_TRUE(features.featuresBufferDeviceAddress.bufferDeviceAddress);
    EXPECT_TRUE(features.featuresMultiview.multiview);
    EXPECT_TRUE(features.featuresSamplerYcbcrConversion.samplerYcbcrConversion);
    EXPECT_TRUE(features.featuresShaderDrawParameters.shaderDrawParameters);
  }
}

TEST_F(VulkanFeaturesTest, CheckSelectedFeaturesMissingCoreFeature) {
  igl::setDebugBreakEnabled(false);

  const igl::vulkan::VulkanContextConfig config;

  const igl::vulkan::VulkanFeatures requested(config);
  igl::vulkan::VulkanFeatures available(config);
  available.vkPhysicalDeviceFeatures2.features.dualSrcBlend = VK_FALSE;

  const Result result = requested.checkSelectedFeatures(available);
#if IGL_PLATFORM_APPLE
  EXPECT_TRUE(result.isOk());
#else
  EXPECT_FALSE(result.isOk());
#endif
}

TEST_F(VulkanFeaturesTest, CheckSelectedFeaturesMissingMultiview) {
  igl::setDebugBreakEnabled(false);

  const igl::vulkan::VulkanContextConfig config;

  const igl::vulkan::VulkanFeatures requested(config);
  igl::vulkan::VulkanFeatures available(config);
  available.featuresMultiview.multiview = VK_FALSE;

  const Result result = requested.checkSelectedFeatures(available);
#if IGL_PLATFORM_APPLE
  EXPECT_TRUE(result.isOk());
#else
  EXPECT_FALSE(result.isOk());
#endif
}

TEST_F(VulkanFeaturesTest, ConfigDisablesDualSrcBlend) {
  igl::vulkan::VulkanContextConfig config;
  config.enableDualSrcBlend = false;

  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_FALSE(features.vkPhysicalDeviceFeatures2.features.dualSrcBlend);
}

TEST_F(VulkanFeaturesTest, ConfigDisablesShaderInt16) {
  igl::vulkan::VulkanContextConfig config;
  config.enableShaderInt16 = false;

  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_FALSE(features.vkPhysicalDeviceFeatures2.features.shaderInt16);
}

TEST_F(VulkanFeaturesTest, CheckSelectedFeaturesDescriptorIndexingEnabledAllPresent) {
  igl::vulkan::VulkanContextConfig config;
  config.enableDescriptorIndexing = true;

  const igl::vulkan::VulkanFeatures requested(config);
  const igl::vulkan::VulkanFeatures available(config);

  const Result result = requested.checkSelectedFeatures(available);
  EXPECT_TRUE(result.isOk());
}

TEST_F(VulkanFeaturesTest, ConfigDisablesShaderDrawParameters) {
  igl::vulkan::VulkanContextConfig config;
  config.enableShaderDrawParameters = false;

  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_FALSE(features.featuresShaderDrawParameters.shaderDrawParameters);
}

TEST_F(VulkanFeaturesTest, ConfigDisables16BitStorageAccess) {
  igl::vulkan::VulkanContextConfig config;
  config.enableStorageBuffer16BitAccess = false;

  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_FALSE(features.features16BitStorage.storageBuffer16BitAccess);
}

TEST_F(VulkanFeaturesTest, CheckSelectedFeaturesMissingShaderDrawParameters) {
  igl::setDebugBreakEnabled(false);

  const igl::vulkan::VulkanContextConfig config;

  const igl::vulkan::VulkanFeatures requested(config);
  igl::vulkan::VulkanFeatures available(config);
  available.featuresShaderDrawParameters.shaderDrawParameters = VK_FALSE;

  const Result result = requested.checkSelectedFeatures(available);
#if IGL_PLATFORM_APPLE
  EXPECT_TRUE(result.isOk());
#else
  EXPECT_FALSE(result.isOk());
#endif
}

// shaderIntegerDotProduct opt-in **************************************************

namespace {
// Whether `wanted` is reachable from the feature chain's head. The chain is what reaches
// vkCreateDevice(), so membership -- not the member's value alone -- is what decides whether
// a feature is actually requested.
bool chainContains(const VkPhysicalDeviceFeatures2& features2, const void* wanted) {
  for (const auto* node = static_cast<const VkBaseOutStructure*>(features2.pNext); node != nullptr;
       node = node->pNext) {
    if (static_cast<const void*>(node) == wanted) {
      return true;
    }
  }
  return false;
}
} // namespace

// The default must be inert: off, and absent from the chain. Every other feature struct here
// is chained on the device advertising an extension, so a caller that never heard of integer
// dot product would silently start requesting it if this one followed that pattern.
TEST_F(VulkanFeaturesTest, IntegerDotProductDefaultsToDisabledAndUnchained) {
  const igl::vulkan::VulkanContextConfig config;
  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_EQ(features.featuresShaderIntegerDotProduct.sType,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES);
  EXPECT_FALSE(features.featuresShaderIntegerDotProduct.shaderIntegerDotProduct);
  EXPECT_FALSE(
      chainContains(features.vkPhysicalDeviceFeatures2, &features.featuresShaderIntegerDotProduct));
}

// Requesting it puts it in the chain, and the request survives the copy that VulkanContext
// makes of a caller's VulkanFeatures -- operator= reassembles the chain from scratch, so a
// member it forgot to copy, or a pNext it forgot to reset, would drop the request on the floor
// while leaving the member reading VK_TRUE.
TEST_F(VulkanFeaturesTest, IntegerDotProductChainedOnceRequestedAndSurvivesCopy) {
  const igl::vulkan::VulkanContextConfig config;

  igl::vulkan::VulkanFeatures featuresSrc(config);
  featuresSrc.featuresShaderIntegerDotProduct.shaderIntegerDotProduct = VK_TRUE;

  igl::vulkan::VulkanFeatures featuresDst(config);
  ASSERT_FALSE(featuresDst.featuresShaderIntegerDotProduct.shaderIntegerDotProduct);
  featuresDst = featuresSrc;

  EXPECT_TRUE(featuresDst.featuresShaderIntegerDotProduct.shaderIntegerDotProduct);
  // The copy must chain ITS OWN struct, never the source's -- a chain pointing into another
  // object dangles the moment that object goes away.
  EXPECT_TRUE(chainContains(featuresDst.vkPhysicalDeviceFeatures2,
                            &featuresDst.featuresShaderIntegerDotProduct));
  EXPECT_FALSE(chainContains(featuresDst.vkPhysicalDeviceFeatures2,
                             &featuresSrc.featuresShaderIntegerDotProduct));
}

// fragmentDensityMap vs pipelineFragmentShadingRate *******************************
// populateWithAvailablePhysicalDeviceFeatures() overwrites chained structs with reported values,
// so a GPU supporting both fragmentDensityMap and pipelineFragmentShadingRate would request
// both, violating VUID-VkDeviceCreateInfo-fragmentDensityMap-04481 (fatal when validation runs
// with terminateOnValidationError). resolveMutuallyExclusiveFeatures() drops the pipeline rate
// IGL never uses whenever the density map is enabled.
TEST_F(VulkanFeaturesTest, ResolveFragmentDensityMapPipelineShadingRateConflict) {
  const igl::vulkan::VulkanContextConfig config;
  igl::vulkan::VulkanFeatures features(config);

  // Simulate a GPU reporting both features, as the supported-features query would fill them in.
  features.featuresFragmentDensityMap.fragmentDensityMap = VK_TRUE;
  features.featuresFragmentShadingRate.pipelineFragmentShadingRate = VK_TRUE;

  features.resolveMutuallyExclusiveFeatures();

  EXPECT_TRUE(features.featuresFragmentDensityMap.fragmentDensityMap);
  EXPECT_FALSE(features.featuresFragmentShadingRate.pipelineFragmentShadingRate);
}

// Adreno 830 reports both fragmentDensityMap and primitiveFragmentShadingRate, violating
// VUID-VkDeviceCreateInfo-fragmentDensityMap-04482. The resolver must drop the primitive rate
// (and attachment rate) just like the pipeline rate.
TEST_F(VulkanFeaturesTest, ResolveFragmentDensityMapPrimitiveShadingRateConflict) {
  const igl::vulkan::VulkanContextConfig config;
  igl::vulkan::VulkanFeatures features(config);

  features.featuresFragmentDensityMap.fragmentDensityMap = VK_TRUE;
  features.featuresFragmentShadingRate.primitiveFragmentShadingRate = VK_TRUE;
  features.featuresFragmentShadingRate.attachmentFragmentShadingRate = VK_TRUE;
  features.featuresMeshShader.primitiveFragmentShadingRateMeshShader = VK_TRUE;

  features.resolveMutuallyExclusiveFeatures();

  EXPECT_TRUE(features.featuresFragmentDensityMap.fragmentDensityMap);
  EXPECT_FALSE(features.featuresFragmentShadingRate.primitiveFragmentShadingRate);
  EXPECT_FALSE(features.featuresFragmentShadingRate.attachmentFragmentShadingRate);
  EXPECT_FALSE(features.featuresMeshShader.primitiveFragmentShadingRateMeshShader);
}

// The resolution is narrowly scoped: without a density map there is no conflict, so a reported
// pipeline rate is left alone.
TEST_F(VulkanFeaturesTest, ResolveMutuallyExclusiveFeaturesKeepsPipelineWithoutDensityMap) {
  const igl::vulkan::VulkanContextConfig config;
  igl::vulkan::VulkanFeatures features(config);

  features.featuresFragmentDensityMap.fragmentDensityMap = VK_FALSE;
  features.featuresFragmentShadingRate.pipelineFragmentShadingRate = VK_TRUE;

  features.resolveMutuallyExclusiveFeatures();

  EXPECT_TRUE(features.featuresFragmentShadingRate.pipelineFragmentShadingRate);
}

// allAvailableExtensions initial state ********************************************
TEST_F(VulkanFeaturesTest, AllAvailableExtensionsInitiallyEmpty) {
  const igl::vulkan::VulkanContextConfig config;
  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_TRUE(features.allAvailableExtensions(igl::vulkan::VulkanFeatures::ExtensionType::Instance)
                  .empty());
  EXPECT_TRUE(
      features.allAvailableExtensions(igl::vulkan::VulkanFeatures::ExtensionType::Device).empty());
}

// available initial state **********************************************************
TEST_F(VulkanFeaturesTest, AvailableReturnsFalseWhenNoExtensionsEnumerated) {
  const igl::vulkan::VulkanContextConfig config;
  const igl::vulkan::VulkanFeatures features(config);

  EXPECT_FALSE(
      features.available("VK_KHR_swapchain", igl::vulkan::VulkanFeatures::ExtensionType::Device));
  EXPECT_FALSE(features.available(VK_KHR_SURFACE_EXTENSION_NAME,
                                  igl::vulkan::VulkanFeatures::ExtensionType::Instance));
}

// enableCommonInstanceExtensions without available extensions *********************
TEST_F(VulkanFeaturesTest, EnableCommonInstanceExtensionsNoAvailableExtensions) {
  const igl::vulkan::VulkanContextConfig config;
  igl::vulkan::VulkanFeatures features(config);

  // Without enumerate(), extensions_ is empty, so enable() is a no-op for all ext.
  features.enableCommonInstanceExtensions(config);

  EXPECT_FALSE(features.has_VK_KHR_portability_enumeration);
  EXPECT_FALSE(features.has_VK_KHR_get_surface_capabilities2);
  EXPECT_FALSE(features.has_VK_EXT_headless_surface);
}

// enableCommonDeviceExtensions without available extensions / descriptor buffer always off
TEST_F(VulkanFeaturesTest, EnableCommonDeviceExtensionsNoAvailableExtensions) {
  const igl::vulkan::VulkanContextConfig config;
  igl::vulkan::VulkanFeatures features(config);

  features.enableCommonDeviceExtensions(config);

  EXPECT_FALSE(features.has_VK_EXT_index_type_uint8);
  EXPECT_FALSE(features.has_VK_KHR_timeline_semaphore);
  EXPECT_FALSE(features.has_VK_KHR_synchronization2);
  EXPECT_FALSE(features.has_VK_KHR_8bit_storage);
  EXPECT_FALSE(features.has_VK_KHR_buffer_device_address);
  // Descriptor buffer is intentionally always disabled (hardcoded false in
  // enableCommonDeviceExtensions).
  EXPECT_FALSE(features.has_VK_EXT_descriptor_buffer);
}

// checkSelectedFeatures negative case with descriptor indexing enabled ***********
TEST_F(VulkanFeaturesTest, CheckSelectedFeaturesDescriptorIndexingEnabledMissingFeature) {
  igl::setDebugBreakEnabled(false);

  igl::vulkan::VulkanContextConfig config;
  config.enableDescriptorIndexing = true;

  const igl::vulkan::VulkanFeatures requested(config);
  igl::vulkan::VulkanFeatures available(config);
  // Remove one of the required descriptor-indexing features from "available"
  available.featuresDescriptorIndexing.shaderSampledImageArrayNonUniformIndexing = VK_FALSE;

  const Result result = requested.checkSelectedFeatures(available);
#if IGL_PLATFORM_APPLE
  EXPECT_TRUE(result.isOk());
#else
  EXPECT_FALSE(result.isOk());
#endif
}

TEST_F(VulkanFeaturesTest, CheckSelectedFeaturesMissingSynchronization2) {
  igl::setDebugBreakEnabled(false);

  const igl::vulkan::VulkanContextConfig config;

  const igl::vulkan::VulkanFeatures requested(config);
  igl::vulkan::VulkanFeatures available(config);
  available.featuresSynchronization2.synchronization2 = VK_FALSE;

  const Result result = requested.checkSelectedFeatures(available);
#if IGL_PLATFORM_APPLE
  EXPECT_TRUE(result.isOk());
#else
  EXPECT_FALSE(result.isOk());
#endif
}

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_LINUX
class VulkanFeatureRequestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    igl::setDebugBreakEnabled(false);
#if IGL_PLATFORM_MACOSX && !defined(IGL_USE_STATIC_KOSMICKRISP)
    vulkan::setupMoltenVKEnvironment();
#endif
    context_ = vulkan::HWDevice::createContext(config_, nullptr);
    ASSERT_NE(context_, nullptr);
    Result result;
    devices_ = vulkan::HWDevice::queryDevices(
        *context_, HWDeviceQueryDesc(HWDeviceType::Unknown), &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_FALSE(devices_.empty());
    physicalDevice_ =
        reinterpret_cast<VkPhysicalDevice>(devices_[0].guid); // NOLINT(performance-no-int-to-ptr)
    available_.enumerate(context_->vf_, physicalDevice_);
    VkPhysicalDeviceProperties properties{};
    context_->vf_.vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    apiVersion_ = properties.apiVersion;
  }

  template<typename Feature>
  Feature query(VkStructureType type) {
    Feature supported{.sType = type};
    VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                       .pNext = &supported};
    context_->vf_.vkGetPhysicalDeviceFeatures2(physicalDevice_, &features);
    return supported;
  }

  void create(const vulkan::VulkanFeatures& requested, const char* extension = nullptr) {
    Result result;
    device_ = vulkan::HWDevice::create(std::move(context_),
                                       devices_[0],
                                       0,
                                       0,
                                       extension ? 1 : 0,
                                       extension ? &extension : nullptr,
                                       &requested,
                                       "Feature request test",
                                       &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(device_, nullptr);
  }

  template<typename Feature>
  Feature read(const vulkan::VulkanFeatures& features, VkStructureType type) {
    Feature selected{.sType = type};
    if (const void* structure = features.feature(type)) {
      std::memcpy(&selected, structure, sizeof(Feature));
    }
    selected.pNext = nullptr;
    return selected;
  }

  size_t count(VkStructureType type) {
    size_t occurrences = 0;
    const auto* current = reinterpret_cast<const VkBaseOutStructure*>(
        &device_->getVulkanContext().features().vkPhysicalDeviceFeatures2);
    for (; current; current = current->pNext) {
      occurrences += current->sType == type ? 1 : 0;
    }
    return occurrences;
  }

  const vulkan::VulkanContextConfig config_ = util::device::vulkan::getContextConfig(false);
  vulkan::VulkanFeatures available_{config_};
  std::unique_ptr<vulkan::VulkanContext> context_;
  std::unique_ptr<vulkan::Device> device_;
  std::vector<HWDeviceDesc> devices_;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  uint32_t apiVersion_ = 0;
};

TEST_F(VulkanFeatureRequestTest, SubgroupSizeControlEnablesOnlyRequestedSupportedBits) {
  constexpr VkStructureType type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
  const auto supported = query<VkPhysicalDeviceSubgroupSizeControlFeatures>(type);
  const bool extensionAvailable = available_.available(
      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, vulkan::VulkanFeatures::ExtensionType::Device);
#if IGL_PLATFORM_LINUX
  ASSERT_TRUE(supported.subgroupSizeControl);
#endif
  vulkan::VulkanFeatures requested(config_);
  requested.requestFeature(
      VkPhysicalDeviceSubgroupSizeControlFeatures{.sType = type, .subgroupSizeControl = VK_TRUE},
      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
      VK_API_VERSION_1_3);
  ASSERT_NO_FATAL_FAILURE(create(requested, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME));
  const auto& enabled = device_->getVulkanContext().features();
  const auto selected = read<VkPhysicalDeviceSubgroupSizeControlFeatures>(enabled, type);
  EXPECT_EQ(selected.subgroupSizeControl,
            extensionAvailable || apiVersion_ >= VK_API_VERSION_1_3 ? supported.subgroupSizeControl
                                                                    : VK_FALSE);
  EXPECT_EQ(selected.computeFullSubgroups, VK_FALSE);
  EXPECT_EQ(enabled.enabled(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
                            vulkan::VulkanFeatures::ExtensionType::Device),
            extensionAvailable);
}

TEST_F(VulkanFeatureRequestTest, AvailableExtensionMustBeEnabled) {
  constexpr VkStructureType type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
  const auto supported = query<VkPhysicalDeviceSubgroupSizeControlFeatures>(type);
  vulkan::VulkanFeatures requested(config_);
  requested.requestFeature(
      VkPhysicalDeviceSubgroupSizeControlFeatures{
          .sType = type, .subgroupSizeControl = supported.subgroupSizeControl},
      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
  ASSERT_NO_FATAL_FAILURE(create(requested));
  const auto& enabled = device_->getVulkanContext().features();
  EXPECT_FALSE(enabled.enabled(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
                               vulkan::VulkanFeatures::ExtensionType::Device));
  EXPECT_EQ(read<VkPhysicalDeviceSubgroupSizeControlFeatures>(enabled, type).subgroupSizeControl,
            VK_FALSE);
}

TEST_F(VulkanFeatureRequestTest, InstanceExtensionsSurviveFeatureRequests) {
  constexpr auto type = vulkan::VulkanFeatures::ExtensionType::Instance;
  const vulkan::VulkanFeatures before(context_->features());
  ASSERT_FALSE(before.allAvailableExtensions(type).empty());
  bool hasEnabledExtension = false;
  for (const auto& name : before.allAvailableExtensions(type)) {
    hasEnabledExtension |= before.enabled(name.c_str(), type);
  }
  ASSERT_TRUE(hasEnabledExtension);
  vulkan::VulkanFeatures requested(config_);
  ASSERT_TRUE(requested.allAvailableExtensions(type).empty());
  requested.requestFeature(VkPhysicalDeviceFeatures{.shaderInt64 = VK_TRUE});
  ASSERT_NO_FATAL_FAILURE(create(requested));
  const auto& enabled = device_->getVulkanContext().features();
  EXPECT_EQ(enabled.allAvailableExtensions(type), before.allAvailableExtensions(type));
  for (const auto& name : before.allAvailableExtensions(type)) {
    EXPECT_EQ(enabled.enabled(name.c_str(), type), before.enabled(name.c_str(), type)) << name;
  }
}

TEST_F(VulkanFeatureRequestTest, CoreVersionEnablesIntegerDotProductWithoutExtension) {
  constexpr VkStructureType type =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
  const auto supported = query<VkPhysicalDeviceShaderIntegerDotProductFeatures>(type);
  vulkan::VulkanFeatures requested(config_);
  requested.requestFeature(
      VkPhysicalDeviceShaderIntegerDotProductFeatures{.sType = type,
                                                      .shaderIntegerDotProduct = VK_TRUE},
      VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
      VK_API_VERSION_1_3);
  ASSERT_NO_FATAL_FAILURE(create(requested));
  const auto& enabled = device_->getVulkanContext().features();
  EXPECT_FALSE(enabled.enabled(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
                               vulkan::VulkanFeatures::ExtensionType::Device));
  EXPECT_EQ(enabled.featuresShaderIntegerDotProduct.shaderIntegerDotProduct,
            apiVersion_ >= VK_API_VERSION_1_3 ? supported.shaderIntegerDotProduct : VK_FALSE);
  EXPECT_EQ(
      read<VkPhysicalDeviceShaderIntegerDotProductFeatures>(enabled, type).shaderIntegerDotProduct,
      enabled.featuresShaderIntegerDotProduct.shaderIntegerDotProduct);
}

TEST_F(VulkanFeatureRequestTest, MissingExtensionLeavesDeviceCreatedAndBitsOff) {
  constexpr VkStructureType type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV;
  if (available_.available(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME,
                           vulkan::VulkanFeatures::ExtensionType::Device)) {
    GTEST_SKIP() << "This device supports the extension used for the absent-extension case";
  }
  const auto supported = query<VkPhysicalDeviceCooperativeMatrixFeaturesNV>(type);
  vulkan::VulkanFeatures requested(config_);
  requested.requestFeature(
      VkPhysicalDeviceCooperativeMatrixFeaturesNV{.sType = type, .cooperativeMatrix = VK_TRUE},
      VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
  ASSERT_NO_FATAL_FAILURE(create(requested, VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME));
  const auto& enabled = device_->getVulkanContext().features();
  EXPECT_FALSE(enabled.enabled(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME,
                               vulkan::VulkanFeatures::ExtensionType::Device));
  EXPECT_EQ(read<VkPhysicalDeviceCooperativeMatrixFeaturesNV>(enabled, type).cooperativeMatrix,
            supported.cooperativeMatrix);
  EXPECT_EQ(supported.cooperativeMatrix, VK_FALSE);
}

TEST_F(VulkanFeatureRequestTest, ExistingStorageAndCoreFeaturesRemainUnchanged) {
  constexpr VkStructureType type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  const auto supported = query<VkPhysicalDevice16BitStorageFeatures>(type);
  VkPhysicalDeviceFeatures2 core{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  context_->vf_.vkGetPhysicalDeviceFeatures2(physicalDevice_, &core);
#if IGL_PLATFORM_LINUX
  ASSERT_TRUE(core.features.shaderInt64);
#endif
  vulkan::VulkanFeatures requested(config_);
  requested.requestFeature(
      VkPhysicalDevice16BitStorageFeatures{.sType = type, .storageBuffer16BitAccess = VK_TRUE},
      VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
      VK_API_VERSION_1_1);
  requested.requestFeature(
      VkPhysicalDeviceFeatures{.shaderFloat64 = VK_TRUE, .shaderInt64 = VK_TRUE});
  ASSERT_NO_FATAL_FAILURE(create(requested));
  const auto& enabled = device_->getVulkanContext().features();
  const auto selected = read<VkPhysicalDevice16BitStorageFeatures>(enabled, type);
  EXPECT_EQ(selected.storageBuffer16BitAccess, supported.storageBuffer16BitAccess);
  EXPECT_EQ(selected.uniformAndStorageBuffer16BitAccess,
            supported.uniformAndStorageBuffer16BitAccess);
  EXPECT_EQ(selected.storagePushConstant16, supported.storagePushConstant16);
  EXPECT_EQ(selected.storageInputOutput16, supported.storageInputOutput16);
  EXPECT_EQ(count(type), 1);
  EXPECT_EQ(enabled.vkPhysicalDeviceFeatures2.features.independentBlend,
            core.features.independentBlend);
  EXPECT_EQ(enabled.vkPhysicalDeviceFeatures2.features.multiDrawIndirect,
            core.features.multiDrawIndirect);
  EXPECT_EQ(enabled.vkPhysicalDeviceFeatures2.features.shaderInt64, core.features.shaderInt64);
  EXPECT_EQ(enabled.vkPhysicalDeviceFeatures2.features.shaderFloat64, core.features.shaderFloat64);
  EXPECT_EQ(enabled.vkPhysicalDeviceFeatures2.features.robustBufferAccess, VK_FALSE);
}

TEST_F(VulkanFeatureRequestTest, CopiesOwnRequestsAndEnabledChain) {
  constexpr VkStructureType type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
  const auto supported = query<VkPhysicalDeviceSubgroupSizeControlFeatures>(type);
  vulkan::VulkanFeatures assigned(config_);
  {
    VkPhysicalDevice16BitStorageFeatures ignored{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    vulkan::VulkanFeatures source(config_);
    source.requestFeature(
        VkPhysicalDeviceSubgroupSizeControlFeatures{
            .sType = type, .pNext = &ignored, .subgroupSizeControl = VK_TRUE},
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_API_VERSION_1_3);
    assigned = source;
  }
  vulkan::VulkanFeatures copied(assigned);
  assigned = vulkan::VulkanFeatures(config_);
  ASSERT_NO_FATAL_FAILURE(create(copied));
  vulkan::VulkanFeatures snapshot(device_->getVulkanContext().features());
  device_.reset();
  EXPECT_EQ(read<VkPhysicalDeviceSubgroupSizeControlFeatures>(snapshot, type).subgroupSizeControl,
            apiVersion_ >= VK_API_VERSION_1_3 ? supported.subgroupSizeControl : VK_FALSE);
  ASSERT_NO_FATAL_FAILURE(SetUp());
  ASSERT_NO_FATAL_FAILURE(create(snapshot));
  const auto selected = read<VkPhysicalDeviceSubgroupSizeControlFeatures>(
      device_->getVulkanContext().features(), type);
  EXPECT_EQ(selected.subgroupSizeControl,
            apiVersion_ >= VK_API_VERSION_1_3 ? supported.subgroupSizeControl : VK_FALSE);
  EXPECT_EQ(selected.computeFullSubgroups, VK_FALSE);
}

TEST_F(VulkanFeatureRequestTest, CopiesKeepEnabledAstcHdrWithoutGenericRequest) {
  constexpr VkStructureType type =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT;
  const auto supported = query<VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT>(type);
  const bool extensionAvailable =
      available_.available(VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME,
                           vulkan::VulkanFeatures::ExtensionType::Device);
  vulkan::VulkanFeatures requested(config_);
  ASSERT_NO_FATAL_FAILURE(create(requested, VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME));
  const auto& enabled = device_->getVulkanContext().features();
  const bool expected = extensionAvailable && (supported.textureCompressionASTC_HDR != 0u);
  ASSERT_EQ(enabled.feature(type) != nullptr, expected);
  vulkan::VulkanFeatures snapshot(enabled);
  device_.reset();
  EXPECT_EQ(snapshot.feature(type) != nullptr, expected);
  EXPECT_EQ(read<VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT>(snapshot, type)
                .textureCompressionASTC_HDR,
            expected ? VK_TRUE : VK_FALSE);
}

TEST_F(VulkanFeatureRequestTest, OverlappingRequestsShareOneStructureAndOrSupportedBits) {
  constexpr VkStructureType type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
  const auto supported = query<VkPhysicalDeviceSubgroupSizeControlFeatures>(type);
  vulkan::VulkanFeatures requested(config_);
  requested.requestFeature(
      VkPhysicalDeviceSubgroupSizeControlFeatures{.sType = type, .subgroupSizeControl = VK_TRUE},
      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
      VK_API_VERSION_1_3);
  requested.requestFeature(
      VkPhysicalDeviceSubgroupSizeControlFeatures{.sType = type, .computeFullSubgroups = VK_TRUE},
      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
      VK_API_VERSION_1_3);
  ASSERT_NO_FATAL_FAILURE(create(requested, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME));
  const auto& enabled = device_->getVulkanContext().features();
  const auto selected = read<VkPhysicalDeviceSubgroupSizeControlFeatures>(enabled, type);
  const bool provided = enabled.enabled(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
                                        vulkan::VulkanFeatures::ExtensionType::Device) ||
                        apiVersion_ >= VK_API_VERSION_1_3;
  EXPECT_EQ(selected.subgroupSizeControl, provided ? supported.subgroupSizeControl : VK_FALSE);
  EXPECT_EQ(selected.computeFullSubgroups, provided ? supported.computeFullSubgroups : VK_FALSE);
  EXPECT_EQ(count(type), provided ? 1 : 0);
}

TEST_F(VulkanFeatureRequestTest, AggregateFeatureStructuresAreDropped) {
  vulkan::VulkanFeatures requested(config_);
  requested.requestFeature(
      VkPhysicalDeviceVulkan13Features{.sType =
                                           VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                       .subgroupSizeControl = VK_TRUE},
      "",
      VK_API_VERSION_1_3);
  ASSERT_NO_FATAL_FAILURE(create(requested));
  EXPECT_EQ(count(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES), 0);
  EXPECT_EQ(device_->getVulkanContext()
                .features()
                .featuresShaderIntegerDotProduct.shaderIntegerDotProduct,
            VK_FALSE);
}
#endif

} // namespace igl::tests
