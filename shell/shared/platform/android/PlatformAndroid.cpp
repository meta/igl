/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <shell/shared/platform/android/PlatformAndroid.h>

#include <shell/shared/fileLoader/android/FileLoaderAndroid.h>
#include <shell/shared/imageLoader/ImageLoader.h>
#include <shell/shared/imageWriter/ImageWriter.h>
#include <shell/shared/imageWriter/android/ImageWriterAndroid.h>

#if IGL_BACKEND_VULKAN
#include <glm/ext.hpp>
#include <shell/shared/platform/DisplayContext.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanContext.h>
#endif

namespace igl::shell {

PlatformAndroid::PlatformAndroid(std::shared_ptr<IDevice> device, bool useFakeLoader) :
  device_(std::move(device)),
  fileLoader_(std::make_unique<FileLoaderAndroid>()),
  imageLoader_(std::make_unique<ImageLoader>(*fileLoader_)) {
  if (!useFakeLoader) {
    imageWriter_ = std::make_unique<ImageWriterAndroid>();
  }
}

void PlatformAndroid::updatePreRotationMatrix() {
#if IGL_BACKEND_VULKAN
  if (device_->getBackendType() != BackendType::Vulkan) {
    return;
  }
  // Get the surface transform matrix
  getDisplayContext().preRotationMatrix = [&device = *device_]() -> glm::mat4 {
    auto& vulkanDevice = static_cast<vulkan::Device&>(device);

    float angle = 0.0f;
    switch (vulkanDevice.getVulkanContext().getSurfaceCapabilities().currentTransform) {
    case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
      angle = -90.0f;
      break;
    case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
      angle = -180.0f;
      break;
    case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
      angle = -270.0f;
      break;
    case VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR:
    default:
      return glm::mat4(1.0f);
    }
    return glm::rotate(glm::mat4(1.0f), glm::radians(angle), glm::vec3(0.0f, 0.0f, 1.0f));
  }();
#endif
}

IDevice& PlatformAndroid::getDevice() noexcept {
  return *device_;
}

std::shared_ptr<IDevice> PlatformAndroid::getDevicePtr() const noexcept {
  return device_;
}

ImageLoader& PlatformAndroid::getImageLoader() noexcept {
  return *imageLoader_;
}

const ImageWriter& PlatformAndroid::getImageWriter() const noexcept {
  return *imageWriter_;
}

FileLoader& PlatformAndroid::getFileLoader() const noexcept {
  return *fileLoader_;
}

} // namespace igl::shell
