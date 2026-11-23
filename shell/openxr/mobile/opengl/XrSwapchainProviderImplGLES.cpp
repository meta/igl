/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/openxr/mobile/opengl/XrSwapchainProviderImplGLES.h>

#include <algorithm>
#include <iterator>
#include <shell/openxr/XrLog.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/PlatformDevice.h>
#include <igl/opengl/TextureBufferExternal.h>

namespace igl::shell::openxr::mobile {
namespace {
void enumerateSwapchainImages(XrSwapchain swapchain, std::vector<uint32_t>& outImages) {
  uint32_t numImages = 0;
  XR_CHECK(xrEnumerateSwapchainImages(swapchain, 0, &numImages, nullptr));

  IGL_LOG_INFO("XRSwapchain numImages: %d\n", numImages);

#if IGL_WGL
  std::vector<XrSwapchainImageOpenGLKHR> xrImages(numImages,
                                                  {
                                                      .type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR,
                                                      .next = nullptr,
                                                  });
#else
  std::vector<XrSwapchainImageOpenGLESKHR> xrImages(
      numImages,
      {
          .type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR,
          .next = nullptr,
      });
#endif // IGL_WGL
  XR_CHECK(
      xrEnumerateSwapchainImages(swapchain,
                                 numImages,
                                 &numImages,
                                 reinterpret_cast<XrSwapchainImageBaseHeader*>(xrImages.data())));

  outImages.resize(0);
  std::transform(xrImages.cbegin(),
                 xrImages.cend(),
                 std::back_inserter(outImages),
                 [](const auto& xrImage) { return xrImage.image; });
}

std::shared_ptr<ITexture> getSurfaceTexture(IDevice& device,
                                            const XrSwapchain& swapchain,
                                            const impl::SwapchainImageInfo& swapchainImageInfo,
                                            uint8_t numViews,
                                            const std::vector<uint32_t>& images,
                                            TextureFormat externalTextureFormat,
                                            std::vector<std::shared_ptr<ITexture>>& inOutTextures) {
  uint32_t imageIndex = 0;
  const XrSwapchainImageAcquireInfo acquireInfo{
      .type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO,
      .next = nullptr,
  };
  XR_CHECK(xrAcquireSwapchainImage(swapchain, &acquireInfo, &imageIndex));

  const XrSwapchainImageWaitInfo waitInfo{
      .type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
      .next = nullptr,
      .timeout = XR_INFINITE_DURATION,
  };
  XR_CHECK(xrWaitSwapchainImage(swapchain, &waitInfo));

  const auto glTexture = images[imageIndex];

  if (imageIndex >= inOutTextures.size()) {
    inOutTextures.resize(static_cast<size_t>(imageIndex) + 1, nullptr);
  }

  auto texture = inOutTextures[imageIndex];
  if (!texture || swapchainImageInfo.imageWidth != texture->getSize().width ||
      swapchainImageInfo.imageHeight != texture->getSize().height) {
    auto* const platformDevice = device.getPlatformDevice<opengl::PlatformDevice>();
    texture = platformDevice->createTextureBufferExternal(
        glTexture,
        numViews > 1 ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D,
        igl::TextureDesc::TextureUsageBits::Attachment,
        swapchainImageInfo.imageWidth,
        swapchainImageInfo.imageHeight,
        externalTextureFormat,
        numViews);
    if (!texture) {
      IGL_LOG_ERROR("Failed to create ITexture from swapchain image.\n");
      return {};
    }
    inOutTextures[imageIndex] = std::move(texture);
  }

  return inOutTextures[imageIndex];
}
} // namespace

XrSwapchainProviderImplGLES::XrSwapchainProviderImplGLES(const IDevice& device,
                                                         TextureFormat preferredColorFormat) {
  const auto& openglDevice = static_cast<const opengl::Device&>(device);
  opengl::Texture::FormatDescGL formatDescGL;
  igl::opengl::Texture::toFormatDescGL(openglDevice.getContext(),
                                       preferredColorFormat,
                                       TextureDesc::TextureUsageBits::Attachment |
                                           TextureDesc::TextureUsageBits::Sampled,
                                       formatDescGL);

  preferredColorFormat_ = formatDescGL.internalFormat;
}

void XrSwapchainProviderImplGLES::enumerateImages(
    IDevice& /*device*/,
    XrSwapchain colorSwapchain,
    XrSwapchain depthSwapchain,
    const impl::SwapchainImageInfo& /* swapchainImageInfo */,
    uint8_t /* numViews */) noexcept { // NOLINT(bugprone-exception-escape)
  enumerateSwapchainImages(colorSwapchain, colorImages_);
  enumerateSwapchainImages(depthSwapchain, depthImages_);
}

SurfaceTextures XrSwapchainProviderImplGLES::getSurfaceTextures(
    IDevice& device,
    XrSwapchain colorSwapchain,
    XrSwapchain depthSwapchain,
    const impl::SwapchainImageInfo& swapchainImageInfo,
    uint8_t numViews) noexcept { // NOLINT(bugprone-exception-escape)
  // Assume sized format so format / type are not needed.
  auto iglColorFormat = opengl::Texture::glInternalFormatToTextureFormat(
      static_cast<GLuint>(swapchainImageInfo.colorFormat), 0, 0);
  auto colorTexture = getSurfaceTexture(device,
                                        colorSwapchain,
                                        swapchainImageInfo,
                                        numViews,
                                        colorImages_,
                                        iglColorFormat,
                                        colorTextures_);

  auto iglDepthFormat = opengl::Texture::glInternalFormatToTextureFormat(
      static_cast<GLuint>(swapchainImageInfo.depthFormat), 0, 0);
  auto depthTexture = getSurfaceTexture(device,
                                        depthSwapchain,
                                        swapchainImageInfo,
                                        numViews,
                                        depthImages_,
                                        iglDepthFormat,
                                        depthTextures_);

  return {.color = colorTexture, .depth = depthTexture};
}
} // namespace igl::shell::openxr::mobile
