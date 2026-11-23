/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/openxr/mobile/opengl/XrAppImplGLES.h>

#include <igl/Common.h>
#include <igl/opengl/Device.h>
#if IGL_WGL
#include <igl/opengl/wgl/Context.h>
#include <igl/opengl/wgl/HWDevice.h>
#else
#include <igl/opengl/egl/Context.h>
#include <igl/opengl/egl/HWDevice.h>
#endif // IGL_WGL

#include <shell/openxr/XrLog.h>
#include <shell/openxr/mobile/opengl/XrSwapchainProviderImplGLES.h>

namespace igl::shell::openxr::mobile {
RenderSessionConfig XrAppImplGLES::suggestedSessionConfig() const {
  return {
      .displayName = "OpenGL ES 3.2",
      .backendVersion = {.flavor = BackendFlavor::OpenGL_ES, .majorVersion = 3, .minorVersion = 2},
      .swapchainColorTextureFormat = TextureFormat::RGBA_SRGB};
}

std::vector<const char*> XrAppImplGLES::getXrRequiredExtensions() const {
  return {
#if IGL_WGL
      XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
#else
      XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME,
#endif // IGL_WGL
      XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME,
  };
}

std::vector<const char*> XrAppImplGLES::getXrOptionalExtensions() const {
  return {};
}

std::unique_ptr<IDevice> XrAppImplGLES::initIGL(XrInstance instance, XrSystemId systemId) {
  // Get the graphics requirements.
  // XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING is returned on calls to xrCreateSession()
  // if this function has not been called for the instance and systemId before
  // xrCreateSession().
#if IGL_WGL
  PFN_xrGetOpenGLGraphicsRequirementsKHR pfnGetOpenGLGraphicsRequirementsKHR = nullptr;
  XR_CHECK(xrGetInstanceProcAddr(
      instance,
      "xrGetOpenGLGraphicsRequirementsKHR",
      reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetOpenGLGraphicsRequirementsKHR)));
  XR_CHECK(pfnGetOpenGLGraphicsRequirementsKHR(instance, systemId, &graphicsRequirements_));

  auto hwDevice = opengl::wgl::HWDevice();
#else
  PFN_xrGetOpenGLESGraphicsRequirementsKHR pfnGetOpenGLESGraphicsRequirementsKHR = nullptr;
  XR_CHECK(xrGetInstanceProcAddr(
      instance,
      "xrGetOpenGLESGraphicsRequirementsKHR",
      reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetOpenGLESGraphicsRequirementsKHR)));
  XR_CHECK(pfnGetOpenGLESGraphicsRequirementsKHR(instance, systemId, &graphicsRequirements_));

  auto hwDevice = opengl::egl::HWDevice();
#endif // IGL_WGL

  Result result;
  return hwDevice.create(&result);
}

XrSession XrAppImplGLES::initXrSession(XrInstance instance,
                                       XrSystemId systemId,
                                       IDevice& device,
                                       const RenderSessionConfig& sessionConfig) {
  IGL_DEBUG_ASSERT(sessionConfig.backendVersion.flavor == igl::BackendFlavor::OpenGL_ES);
  sessionConfig_ = sessionConfig;
  device_ = &device;
  const auto& glDevice = static_cast<opengl::Device&>(device); // Downcast is safe here

#if IGL_WGL
  const auto& context =
      static_cast<igl::opengl::wgl::Context&>(glDevice.getContext()); // Downcast is safe here
  XrGraphicsBindingOpenGLWin32KHR graphicsBindingGL = {
      .type = XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR,
      .next = nullptr,
      .hDC = context.getDeviceContext(),
      .hGLRC = context.getRenderContext()};
#else
  const auto& context =
      static_cast<opengl::egl::Context&>(glDevice.getContext()); // Downcast is safe here
  XrGraphicsBindingOpenGLESAndroidKHR graphicsBindingGL = {
      .type = XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR,
      .next = nullptr,
      .display = context.getDisplay(),
      .config = context.getConfig(),
      .context = context.get(),
  };
#endif // IGL_WGL

  const XrSessionCreateInfo sessionCreateInfo = {
      .type = XR_TYPE_SESSION_CREATE_INFO,
      .next = &graphicsBindingGL,
      .createFlags = 0,
      .systemId = systemId,
  };

  XrResult xrResult(XR_SUCCESS);
  XrSession session = nullptr;
  XR_CHECK(xrResult = xrCreateSession(instance, &sessionCreateInfo, &session));
  if (xrResult != XR_SUCCESS) {
    IGL_LOG_ERROR("Failed to create XR session: %d.\n", xrResult);
    return XR_NULL_HANDLE;
  }
  IGL_LOG_INFO("XR session created.\n");

  return session;
}

std::unique_ptr<impl::XrSwapchainProviderImpl> XrAppImplGLES::createSwapchainProviderImpl() const {
  if (IGL_DEBUG_VERIFY(device_)) {
    return std::make_unique<XrSwapchainProviderImplGLES>(
        *device_, sessionConfig_.swapchainColorTextureFormat);
  }

  return nullptr;
}
} // namespace igl::shell::openxr::mobile
