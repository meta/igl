/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/tests/util/device/vulkan/TestDevice.h>

#include <igl/Common.h>

#if IGL_PLATFORM_WINDOWS || IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || IGL_PLATFORM_IOS || \
    IGL_PLATFORM_LINUX
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanFeatures.h>
#endif

#if IGL_PLATFORM_MACOSX
#include <igl/vulkan/moltenvk/MoltenVkHelpers.h>
#endif

namespace igl::tests::util::device::vulkan {

//
// createTestDevice
//
// Used by clients to get an IGL device.
//

igl::vulkan::VulkanContextConfig getContextConfig(bool enableValidation) {
  igl::vulkan::VulkanContextConfig config;
  config.enableValidation = enableValidation;
  config.enableGPUAssistedValidation = enableValidation;

#if IGL_PLATFORM_MACOSX
  config.terminateOnValidationError = false;
#elif IGL_DEBUG
  config.terminateOnValidationError = enableValidation;
#else
  config.enableValidation = false;
  config.terminateOnValidationError = false;
#endif
#ifdef IGL_DISABLE_VALIDATION
  config.enableValidation = false;
  config.terminateOnValidationError = false;
#endif
  config.swapChainColorSpace = igl::ColorSpace::SRGBNonlinear;
  config.enableExtraLogs = enableValidation;

  return config;
}

std::unique_ptr<igl::vulkan::Device> createTestDevice(
    const igl::vulkan::VulkanContextConfig& config) {
  return createTestDevice(config, {});
}

std::unique_ptr<igl::vulkan::Device> createTestDevice(
    const igl::vulkan::VulkanContextConfig& config,
    const std::function<void(igl::vulkan::VulkanFeatures&)>& configureFeatures) {
#if IGL_PLATFORM_MACOSX && !defined(IGL_USE_STATIC_KOSMICKRISP)
  // KosmicKrisp is statically linked; there is no Vulkan loader to point at a MoltenVK ICD.
  ::igl::vulkan::setupMoltenVKEnvironment();
#endif

  std::unique_ptr<igl::vulkan::Device> iglDev = nullptr;
  Result ret;

  auto ctx = igl::vulkan::HWDevice::createContext(config, nullptr);
  if (!ctx) {
    return nullptr;
  }

  std::vector<HWDeviceDesc> devices =
      igl::vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::Unknown), &ret);

  if (ret.isOk()) {
    std::vector<const char*> extraDeviceExtensions;
    extraDeviceExtensions.emplace_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);

    igl::vulkan::VulkanFeatures features(config);
    features.populateWithAvailablePhysicalDeviceFeatures(
        *ctx,
        reinterpret_cast<VkPhysicalDevice>(devices[0].guid)); // NOLINT(performance-no-int-to-ptr)
    if (configureFeatures) {
      configureFeatures(features);
    }

    iglDev = igl::vulkan::HWDevice::create(std::move(ctx),
                                           devices[0],
                                           0, // width
                                           0, // height,
                                           extraDeviceExtensions.size(),
                                           extraDeviceExtensions.data(),
                                           &features,
                                           "Test Device",
                                           &ret);

    if (!ret.isOk()) {
      iglDev = nullptr;
    }
  }

  return iglDev;
}

std::unique_ptr<igl::vulkan::Device> createTestDevice(bool enableValidation) {
  return createTestDevice(getContextConfig(enableValidation));
}

} // namespace igl::tests::util::device::vulkan
