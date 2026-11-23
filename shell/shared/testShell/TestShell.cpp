/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <string_view>
#include <utility>
#include <shell/shared/platform/Platform.h>
#if IGL_PLATFORM_ANDROID
#include <shell/shared/platform/android/PlatformAndroid.h>
#endif
#if IGL_PLATFORM_IOS
#include <shell/shared/platform/ios/PlatformIos.h>
#endif
#if IGL_PLATFORM_LINUX
#include <shell/shared/platform/linux/PlatformLinux.h>
#endif
#if IGL_PLATFORM_MACOSX
#include <shell/shared/platform/mac/PlatformMac.h>
#endif
#if IGL_PLATFORM_WINDOWS
#include <shell/shared/platform/win/PlatformWin.h>
#endif
#include <shell/shared/renderSession/ShellParams.h>
#include <shell/shared/testShell/TestShell.h>
#include <igl/tests/util/device/TestDevice.h>

namespace igl::shell {

namespace {

std::shared_ptr<IDevice> createTestDevice() {
  static constexpr std::pair<std::string_view, BackendType> kBackends[] = {
      {"ogl", BackendType::OpenGL},
      {"metal", BackendType::Metal},
      {"vulkan", BackendType::Vulkan},
      {"webgpu", BackendType::WebGPU},
      // @fb-only
  };
  const std::string_view backend(IGL_BACKEND_TYPE);

  for (const auto& [name, type] : kBackends) {
    if (backend != name) {
      continue;
    }
#ifdef IGL_UNIT_TESTS_GLES_VERSION
    if (type == BackendType::OpenGL) {
      return tests::util::device::createTestDevice(type,
                                                   {.flavor = BackendFlavor::OpenGL_ES,
                                                    .majorVersion = IGL_UNIT_TESTS_GLES_VERSION,
                                                    .minorVersion = 0});
    }
#endif
    return tests::util::device::createTestDevice(type);
  }
  return nullptr;
}

void ensureCommandLineArgsInitialized() {
  // Fake initialization of command line args so sessions don't assert when accessing them.
  // Only do it once, otherwise it triggers an internal assert.

#if IGL_PLATFORM_ANDROID
  static bool sInitialized = true; // Android prohibits initialization of command line args
#else
  static bool sInitialized = false;
#endif
  if (!sInitialized) {
    sInitialized = true;
    Platform::initializeCommandLineArgs(0, nullptr);
  }
}

} // namespace

void TestShellBase::setUpInternal(ScreenSize screenSize, bool prefersRGB) {
  ensureCommandLineArgsInitialized();

  // Create igl device for requested backend
  std::shared_ptr<IDevice> iglDevice = createTestDevice();
  ASSERT_TRUE(iglDevice != nullptr);
  // Create platform shell to run the tests with
#if defined(IGL_PLATFORM_MACOSX) && IGL_PLATFORM_MACOSX
  platform_ = std::make_shared<igl::shell::PlatformMac>(std::move(iglDevice));
#elif defined(IGL_PLATFORM_IOS) && IGL_PLATFORM_IOS
  platform_ = std::make_shared<PlatformIos>(std::move(iglDevice));
#elif defined(IGL_PLATFORM_WINDOWS) && IGL_PLATFORM_WINDOWS
  platform_ = std::make_shared<igl::shell::PlatformWin>(std::move(iglDevice));
#elif defined(IGL_PLATFORM_ANDROID) && IGL_PLATFORM_ANDROID
  platform_ = std::make_shared<igl::shell::PlatformAndroid>(std::move(iglDevice));
#elif defined(IGL_PLATFORM_LINUX) && IGL_PLATFORM_LINUX
  platform_ = std::make_shared<igl::shell::PlatformLinux>(std::move(iglDevice));
#endif

  IGL_DEBUG_ASSERT(platform_);

  if (platform_->getDevice().getBackendType() == BackendType::OpenGL) {
    auto version = platform_->getDevice().getBackendVersion();
    if (version.majorVersion < 2) {
      GTEST_SKIP() << "OpenGL version " << static_cast<int>(version.majorVersion) << "."
                   << static_cast<int>(version.minorVersion) << " is too low";
    }
  }
  // Create an offscreen texture to render to
  Result ret;
  auto hasNativesRGBSupport = platform_->getDevice().hasFeature(DeviceFeatures::SRGB);
  auto colorFormat = prefersRGB && hasNativesRGBSupport ? TextureFormat::RGBA_SRGB
                                                        : TextureFormat::RGBA_UNorm8;

  TextureDesc texDesc = igl::TextureDesc::new2D(colorFormat,
                                                screenSize.width,
                                                screenSize.height,
                                                igl::TextureDesc::TextureUsageBits::Sampled |
                                                    igl::TextureDesc::TextureUsageBits::Attachment);
  offscreenTexture_ = platform_->getDevice().createTexture(texDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(offscreenTexture_ != nullptr);
  TextureDesc depthDextureDesc = igl::TextureDesc::new2D(
      TextureFormat::Z_UNorm24,
      screenSize.width,
      screenSize.height,
      igl::TextureDesc::TextureUsageBits::Sampled | igl::TextureDesc::TextureUsageBits::Attachment);
  depthDextureDesc.storage = ResourceStorage::Private;
  offscreenDepthTexture_ = platform_->getDevice().createTexture(depthDextureDesc, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_TRUE(offscreenDepthTexture_ != nullptr);
}

void TestShell::run(RenderSession& session, size_t numFrames) {
  ShellParams shellParams;
  session.setShellParams(shellParams);
  session.initialize();
  for (size_t i = 0; i < numFrames; ++i) {
    const DeviceScope scope(platform_->getDevice());
    session.update({.color = offscreenTexture_, .depth = offscreenDepthTexture_});
  }
  session.teardown();
}

} // namespace igl::shell
