/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "HWDevice.h"

#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>

namespace igl::vulkan {

std::unique_ptr<VulkanContext> HWDevice::createContext(const VulkanContextConfig& config,
                                                       void* IGL_NULLABLE window,
                                                       void* IGL_NULLABLE display) {
  // @fb-only
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  auto context = std::make_unique<VulkanContext>(config, window, display);
  if (!context->loaderVersionSupported_) {
    return nullptr;
  }
  return context;
}

std::vector<HWDeviceDesc> HWDevice::queryDevices(VulkanContext& ctx,
                                                 const HWDeviceQueryDesc& desc,
                                                 Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  std::vector<HWDeviceDesc> outDevices;

  Result::setResult(outResult, ctx.queryDevices(desc, outDevices));

  return outDevices;
}

std::unique_ptr<Device> HWDevice::create(std::unique_ptr<VulkanContext> ctx,
                                         const HWDeviceDesc& desc,
                                         uint32_t width,
                                         uint32_t height,
                                         size_t numExtraDeviceExtensions,
                                         const char* IGL_NULLABLE* IGL_NULLABLE
                                             extraDeviceExtensions,
                                         const VulkanFeatures* IGL_NULLABLE features,
                                         const char* IGL_NULLABLE debugName,
                                         Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  if (!ctx) {
    Result::setResult(outResult,
                      Result(Result::Code::Unsupported,
                             "Vulkan context is unavailable; Vulkan 1.1 or newer is required"));
    return nullptr;
  }

  auto result =
      ctx->initContext(desc, numExtraDeviceExtensions, extraDeviceExtensions, features, debugName);

  Result::setResult(outResult, result);

  if (!result.isOk()) {
    return nullptr;
  }

  if (width > 0 && height > 0) {
    result = ctx->initSwapchain(width, height);

    Result::setResult(outResult, result);
  }

  return result.isOk() ? std::make_unique<Device>(std::move(ctx)) : nullptr;
}

} // namespace igl::vulkan
