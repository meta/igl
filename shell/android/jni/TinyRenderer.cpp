/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include "TinyRenderer.h"

#ifdef IGL_WITH_PERFETTO
#include <shell/shared/profiling/IglPerfetto.h>
#endif

#include <EGL/egl.h>
#include <android/log.h>
#include <android/native_window.h>
#if IGL_BACKEND_OPENGL
#include <igl/opengl/egl/HWDevice.h>
#include <igl/opengl/egl/PlatformDevice.h>
#endif
#include <shell/shared/fileLoader/android/FileLoaderAndroid.h>
#include <shell/shared/input/InputDispatcher.h>
#include <shell/shared/platform/DisplayContext.h>
#include <shell/shared/platform/PresentationRateController.h>
#include <shell/shared/platform/android/PresentationRateAndroid.h>
#include <shell/shared/renderSession/AppParams.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <shell/shared/renderSession/ShellParams.h>
#if IGL_BACKEND_VULKAN
#include <igl/ShaderCreator.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/PlatformDevice.h>
#include <igl/vulkan/VulkanContext.h>
#endif
#include <algorithm>
#include <array>
#include <memory>
#include <sys/system_properties.h>
#include <unordered_set>

namespace {

// Helper functions to read Android system properties.
std::optional<std::string> getAndroidSystemProperty(const char* keyName) noexcept {
  const prop_info* pi = __system_property_find(keyName);
  if (!pi) {
    return std::nullopt;
  }
  std::string result;
#if __ANDROID_API__ >= 26
  // Reads values longer than PROP_VALUE_MAX, which __system_property_read() truncates.
  __system_property_read_callback(
      pi,
      [](void* cookie, const char* /*name*/, const char* value, uint32_t /*serial*/) {
        *static_cast<std::string*>(cookie) = value;
      },
      &result);
#else
  std::array<char, PROP_VALUE_MAX> value{};
  if (__system_property_read(pi, nullptr, value.data()) > 0) {
    result = value.data();
  }
#endif
  return result.empty() ? std::nullopt : std::make_optional(std::move(result));
}

// NOLINTNEXTLINE(bugprone-exception-escape)
std::optional<bool> getAndroidSystemPropertyBool(const char* keyName) noexcept {
  auto prop = getAndroidSystemProperty(keyName);
  if (!prop.has_value()) {
    return std::nullopt;
  }
  const auto& value = prop.value();
  if (value == "true" || value == "1") {
    return true;
  }
  if (value == "false" || value == "0") {
    return false;
  }
  return std::nullopt;
}

std::optional<int> getAndroidSystemPropertyInt(const char* keyName) noexcept {
  auto prop = getAndroidSystemProperty(keyName);
  if (!prop.has_value()) {
    return std::nullopt;
  }
  try {
    return std::stoi(prop.value());
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<size_t> getAndroidSystemPropertySizeT(const char* keyName) noexcept {
  auto prop = getAndroidSystemProperty(keyName);
  if (!prop.has_value()) {
    return std::nullopt;
  }
  try {
    return std::stoul(prop.value());
  } catch (...) {
    return std::nullopt;
  }
}

// Read basic shell parameters (headless, screenshot, viewport, fps, multiview) from Android
// system properties.
// NOLINTNEXTLINE(bugprone-exception-escape)
void readBasicShellParams(igl::shell::ShellParams& shellParams,
                          const std::string& prefixStr) noexcept {
  auto headless = getAndroidSystemPropertyBool((prefixStr + "headless").c_str());
  if (headless.has_value()) {
    shellParams.isHeadless = headless.value();
    if (shellParams.isHeadless) {
      shellParams.shouldPresent = false;
      if (shellParams.screenshotNumber == ~0u) {
        shellParams.screenshotNumber = 0;
      }
    }
  }

  auto disableVulkanValidation =
      getAndroidSystemPropertyBool((prefixStr + "disable-vulkan-validation-layers").c_str());
  if (disableVulkanValidation.has_value()) {
    shellParams.enableVulkanValidationLayers = !disableVulkanValidation.value();
  }

  auto screenshotFile = getAndroidSystemProperty((prefixStr + "screenshot-file").c_str());
  if (screenshotFile.has_value()) {
    shellParams.screenshotFileName = screenshotFile.value();
  }

  auto screenshotNumber = getAndroidSystemPropertyInt((prefixStr + "screenshot-number").c_str());
  if (screenshotNumber.has_value()) {
    shellParams.screenshotNumber = static_cast<uint32_t>(screenshotNumber.value());
  }

  auto viewportSize = getAndroidSystemProperty((prefixStr + "viewport-size").c_str());
  if (viewportSize.has_value()) {
    unsigned int w = 0;
    unsigned int h = 0;
    if (sscanf(viewportSize.value().c_str(), "%ux%u", &w, &h) == 2) {
      if (w && h) {
        shellParams.viewportSize = glm::vec2(w, h);
      }
    }
  }

  auto fpsThrottle = getAndroidSystemPropertyInt((prefixStr + "fps-throttle").c_str());
  if (fpsThrottle.has_value()) {
    shellParams.fpsThrottleMs = static_cast<uint32_t>(fpsThrottle.value());
  }

  auto fpsThrottleRandom =
      getAndroidSystemPropertyBool((prefixStr + "fps-throttle-random").c_str());
  if (fpsThrottleRandom.has_value()) {
    shellParams.fpsThrottleRandom = fpsThrottleRandom.value();
  }

  auto freezeAtFrame = getAndroidSystemPropertyInt((prefixStr + "freeze-at-frame").c_str());
  if (freezeAtFrame.has_value()) {
    shellParams.freezeAtFrame = static_cast<uint32_t>(freezeAtFrame.value());
  }

  auto forceMultiview = getAndroidSystemPropertyBool((prefixStr + "force-multiview").c_str());
  if (forceMultiview.has_value() && forceMultiview.value()) {
    shellParams.forceMultiview = true;
    shellParams.renderMode = igl::shell::RenderMode::SinglePassStereo;
    shellParams.viewParams.resize(2);
    shellParams.viewParams[0].viewIndex = 0;
    shellParams.viewParams[1].viewIndex = 1;
  }
}

// Apply benchmark parameters to shellParams. Uses an early return if no benchmark
// parameter is present, avoiding a long compound condition.
// NOLINTNEXTLINE(bugprone-exception-escape)
void applyBenchmarkParamsToShellParams(
    igl::shell::ShellParams& shellParams,
    std::optional<size_t> timeout,
    std::optional<size_t> sessions,
    std::optional<bool> logReporter,
    std::optional<bool> offscreenOnly,
    std::optional<bool> benchmark,
    std::optional<size_t> benchmarkDuration,
    std::optional<size_t> reportInterval,
    const std::optional<std::string>& hiccupMultiplier,
    std::optional<size_t> renderBufferSize,
    const std::vector<std::pair<std::string, std::string>>& customParams) noexcept {
  // Check if any benchmark parameter is set using std::any_of to avoid a long || chain
  const std::array<bool, 9> benchmarkParamFlags = {timeout.has_value(),
                                                   sessions.has_value(),
                                                   logReporter.has_value(),
                                                   offscreenOnly.has_value(),
                                                   benchmark.has_value(),
                                                   benchmarkDuration.has_value(),
                                                   reportInterval.has_value(),
                                                   hiccupMultiplier.has_value(),
                                                   renderBufferSize.has_value()};
  const bool hasAnyBenchmarkParam =
      std::any_of(benchmarkParamFlags.begin(), benchmarkParamFlags.end(), [](bool v) { return v; });
  if (!hasAnyBenchmarkParam && customParams.empty()) {
    return;
  }

  if (!shellParams.benchmarkParams.has_value()) {
    shellParams.benchmarkParams = igl::shell::BenchmarkRenderSessionParams();
  }

  if (timeout.has_value()) {
    shellParams.benchmarkParams->renderSessionTimeoutMs = timeout.value();
  }
  if (sessions.has_value()) {
    shellParams.benchmarkParams->numSessionsToRun = sessions.value();
  }
  if (logReporter.has_value()) {
    shellParams.benchmarkParams->logReporter = logReporter.value();
  }
  if (offscreenOnly.has_value()) {
    shellParams.benchmarkParams->offscreenRenderingOnly = offscreenOnly.value();
  }

  // Apply new benchmark parameters
  if (benchmarkDuration.has_value()) {
    shellParams.benchmarkParams->benchmarkDurationMs = benchmarkDuration.value();
  }
  if (reportInterval.has_value()) {
    shellParams.benchmarkParams->reportIntervalMs = reportInterval.value();
  }
  if (hiccupMultiplier.has_value()) {
    try {
      shellParams.benchmarkParams->hiccupMultiplier = std::stod(hiccupMultiplier.value());
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
  }
  if (renderBufferSize.has_value()) {
    shellParams.benchmarkParams->renderTimeBufferSize = renderBufferSize.value();
  }

  // Add custom parameters
  for (const auto& [key, value] : customParams) {
    shellParams.benchmarkParams->customParams.emplace_back(key, value);
  }
}

// Read shell parameters from Android system properties.
// perfettoEnabled is set via the __system_property_foreach iteration which handles
// property names longer than 31 characters (the __system_property_get limit).
// NOLINTNEXTLINE(bugprone-exception-escape)
void readShellParamsFromAndroidProps(igl::shell::ShellParams& shellParams,
                                     const char* prefix,
                                     bool& perfettoEnabled) noexcept {
  perfettoEnabled = false;
  std::string prefixStr(prefix);
  prefixStr += ".";

  // Read basic ShellParams (headless, screenshot, viewport, fps, multiview)
  readBasicShellParams(shellParams, prefixStr);

  // Read BenchmarkRenderSessionParams - always try to read them
  auto timeout = getAndroidSystemPropertySizeT((prefixStr + "timeout").c_str());
  auto sessions = getAndroidSystemPropertySizeT((prefixStr + "sessions").c_str());
  auto logReporter = getAndroidSystemPropertyBool((prefixStr + "log-reporter").c_str());
  auto offscreenOnly = getAndroidSystemPropertyBool((prefixStr + "offscreen-only").c_str());
  auto benchmark = getAndroidSystemPropertyBool((prefixStr + "benchmark").c_str());

  // Read new benchmark parameters
  auto benchmarkDuration =
      getAndroidSystemPropertySizeT((prefixStr + "benchmark-duration").c_str());
  auto reportInterval = getAndroidSystemPropertySizeT((prefixStr + "report-interval").c_str());
  auto hiccupMultiplier = getAndroidSystemProperty((prefixStr + "hiccup-multiplier").c_str());
  auto renderBufferSize = getAndroidSystemPropertySizeT((prefixStr + "render-buffer-size").c_str());

  // Debug: Log what benchmark properties were found
  if (benchmark.has_value() || benchmarkDuration.has_value() || reportInterval.has_value()) {
    __android_log_print(
        ANDROID_LOG_INFO, "igl", "[IGL Benchmark] System props prefix: %s\n", prefixStr.c_str());
    __android_log_print(
        ANDROID_LOG_INFO,
        "igl",
        "[IGL Benchmark] benchmark=%s, duration=%s, interval=%s\n",
        benchmark.has_value() ? (benchmark.value() ? "true" : "false") : "not set",
        benchmarkDuration.has_value() ? std::to_string(benchmarkDuration.value()).c_str()
                                      : "not set",
        reportInterval.has_value() ? std::to_string(reportInterval.value()).c_str() : "not set");
  }

  // Read custom parameters using __system_property_foreach (API 26+)
  // Custom parameters are any properties under the prefix that are not standard params
  std::vector<std::pair<std::string, std::string>> customParams;

#if __ANDROID_API__ >= 26
  // Known standard parameter names to exclude from custom params
  // NOLINTNEXTLINE(facebook-static-object-destructor-check)
  static const std::unordered_set<std::string> kStandardParams = {
      "headless",
      "disable-vulkan-validation-layers",
      "screenshot-file",
      "screenshot-number",
      "viewport-size",
      "fps-throttle",
      "fps-throttle-random",
      "freeze-at-frame",
      "timeout",
      "sessions",
      "log-reporter",
      "offscreen-only",
      "benchmark",
      "benchmark-duration",
      "run-time",
      "report-interval",
      "hiccup-multiplier",
      "render-buffer-size",
      "force-multiview",
      "perfetto"};

  struct CallbackData {
    const std::string& prefix;
    const std::unordered_set<std::string>& standardParams;
    std::vector<std::pair<std::string, std::string>>* customParams;
    bool* perfettoEnabled;
  };

  CallbackData callbackData{.prefix = prefixStr,
                            .standardParams = kStandardParams,
                            .customParams = &customParams,
                            .perfettoEnabled = &perfettoEnabled};

  __system_property_foreach(
      [](const prop_info* pi, void* cookie) {
        auto* data = reinterpret_cast<CallbackData*>(cookie);

        // Use __system_property_read_callback to handle property names > 31 chars
        // (__system_property_read truncates names at PROP_NAME_MAX-1 = 31 chars)
        struct ReadResult {
          std::string name;
          std::string value;
        } readResult;
        __system_property_read_callback(
            pi,
            [](void* ctx, const char* n, const char* v, uint32_t /*serial*/) {
              auto* r = static_cast<ReadResult*>(ctx);
              r->name = n;
              r->value = v;
            },
            &readResult);

        // Check if property starts with our prefix
        if (readResult.name.starts_with(data->prefix)) {
          const std::string key = readResult.name.substr(data->prefix.length());
          if (key == "perfetto") {
            *data->perfettoEnabled = (readResult.value == "true" || readResult.value == "1");
          } else if (!key.empty() && data->standardParams.find(key) == data->standardParams.end()) {
            data->customParams->emplace_back(key, readResult.value);
          }
        }
      },
      reinterpret_cast<void*>(&callbackData));
#endif

  // If any benchmark parameter is set (including custom params), create the benchmark params
  applyBenchmarkParamsToShellParams(shellParams,
                                    timeout,
                                    sessions,
                                    logReporter,
                                    offscreenOnly,
                                    benchmark,
                                    benchmarkDuration,
                                    reportInterval,
                                    hiccupMultiplier,
                                    renderBufferSize,
                                    customParams);
}

#if IGL_BACKEND_OPENGL
// EGL_MAX_SWAP_INTERVAL for the config the current context was created with, or 1 when EGL
// will not say. One is the safe floor: every EGL implementation waits at least one refresh,
// so a driver we cannot interrogate is treated as one that only does the uncapped rung.
EGLint currentMaxSwapInterval(EGLDisplay display) {
  EGLContext context = eglGetCurrentContext();
  if (display == EGL_NO_DISPLAY || context == EGL_NO_CONTEXT) {
    return 1;
  }
  EGLint configId = 0;
  if (eglQueryContext(display, context, EGL_CONFIG_ID, &configId) == EGL_FALSE) {
    return 1;
  }
  const std::array<EGLint, 3> attribs = {EGL_CONFIG_ID, configId, EGL_NONE};
  EGLConfig config = nullptr;
  EGLint configCount = 0;
  if (eglChooseConfig(display, attribs.data(), &config, 1, &configCount) == EGL_FALSE ||
      configCount == 0) {
    return 1;
  }
  EGLint maxInterval = 1;
  if (eglGetConfigAttrib(display, config, EGL_MAX_SWAP_INTERVAL, &maxInterval) == EGL_FALSE) {
    return 1;
  }
  return maxInterval < 1 ? 1 : maxInterval;
}

// The GLES frame divider, or an empty one when no context is current. Built on the thread
// that owns the context, and eglSwapInterval() acts on that thread's current draw surface,
// so it must also be applied there — which is the render thread, where rate requests land.
igl::shell::FrameDivider eglSwapIntervalDivider() {
  EGLDisplay display = eglGetCurrentDisplay();
  if (display == EGL_NO_DISPLAY) {
    return {};
  }
  return igl::shell::FrameDivider{
      .setDivisor = [display](int refreshesPerFrame) -> igl::Result {
        if (eglSwapInterval(display, refreshesPerFrame) == EGL_FALSE) {
          return igl::Result{igl::Result::Code::RuntimeError,
                             "EGL would not take the swap interval this rate needs."};
        }
        return igl::Result{};
      },
      .maxDivisor = static_cast<int>(currentMaxSwapInterval(display)),
  };
}
#endif // IGL_BACKEND_OPENGL

// Stores the current EGL context when created, and restores it when destroyed.
struct ContextGuard {
  ContextGuard(const igl::IDevice& device) {
#if IGL_BACKEND_OPENGL
    backend_ = device.getBackendType();
    if (backend_ == igl::BackendType::OpenGL) {
      display_ = eglGetCurrentDisplay();
      context_ = eglGetCurrentContext();
      readSurface_ = eglGetCurrentSurface(EGL_READ);
      drawSurface_ = eglGetCurrentSurface(EGL_DRAW);
    }
#endif
  }

  ~ContextGuard() {
#if IGL_BACKEND_OPENGL
    if (backend_ == igl::BackendType::OpenGL) {
      eglMakeCurrent(display_, readSurface_, drawSurface_, context_);
    }
#endif
  }
  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;
  ContextGuard(ContextGuard&&) = delete;
  ContextGuard& operator=(ContextGuard&&) = delete;

 private:
#if IGL_BACKEND_OPENGL
  igl::BackendType backend_;
  EGLDisplay display_;
  EGLContext context_;
  EGLSurface readSurface_;
  EGLSurface drawSurface_;
#endif
};

} // namespace

namespace igl::samples {

using namespace igl;

TinyRenderer::~TinyRenderer() {
  adoptNativeWindow(nullptr);
}

void TinyRenderer::adoptNativeWindow(ANativeWindow* nativeWindow) {
  if (nativeWindow_ == nativeWindow) {
    // Same window, second reference. ANativeWindow_fromSurface() acquires on every call and
    // returns the same pointer for the same Surface, which is exactly what a resize does, so
    // the caller's reference is redundant rather than new. Releasing it here is what keeps a
    // resize from leaking one reference per call; simply returning would.
    if (nativeWindow != nullptr) {
      ANativeWindow_release(nativeWindow);
    }
    return;
  }
  if (nativeWindow_ != nullptr) {
    ANativeWindow_release(nativeWindow_);
  }
  // Ownership of the reference the caller acquired now sits here.
  nativeWindow_ = nativeWindow;
}

void TinyRenderer::releaseNativeWindowIfHeld(ANativeWindow* nativeWindow) {
  if (nativeWindow_ != nullptr && nativeWindow_ == nativeWindow) {
    adoptNativeWindow(nullptr);
  }
}

void TinyRenderer::init(AAssetManager* mgr,
                        ANativeWindow* nativeWindow,
                        shell::IRenderSessionFactory& factory,
                        BackendVersion backendVersion,
                        TextureFormat swapchainColorTextureFormat,
                        const std::vector<std::string>& args,
                        float displayCurrentRefreshRateHz,
                        float displayMaxRefreshRateHz) {
  backendVersion_ = backendVersion;
  adoptNativeWindow(nativeWindow);
  displayRates_->currentHz = displayCurrentRefreshRateHz;
  displayRates_->maxHz = displayMaxRefreshRateHz;
  Result result;
  const HWDeviceQueryDesc queryDesc(HWDeviceType::IntegratedGpu);
  std::unique_ptr<IDevice> d;

  // Read shell params from Android system properties.
  // perfettoEnabled is read via __system_property_foreach which handles
  // property names > 31 chars (the __system_property_get limit).
  bool perfettoEnabled = false;
  readShellParamsFromAndroidProps(
      shellParams_, factory.getAndroidSystemPropsPrefix(), perfettoEnabled);

#ifdef IGL_WITH_PERFETTO
  if (perfettoEnabled) {
    igl::shell::profiling::initPerfetto();
  }
#else
  (void)perfettoEnabled;
#endif // IGL_WITH_PERFETTO

  // Debug: Log if benchmark params were set
  if (shellParams_.benchmarkParams.has_value()) {
    __android_log_print(ANDROID_LOG_INFO,
                        "igl",
                        "[IGL Benchmark] benchmarkParams SET after reading props: duration=%zu, "
                        "interval=%zu\n",
                        shellParams_.benchmarkParams->benchmarkDurationMs,
                        shellParams_.benchmarkParams->reportIntervalMs);
  } else {
    __android_log_print(
        ANDROID_LOG_INFO, "igl", "[IGL Benchmark] benchmarkParams NOT SET after reading props\n");
  }

  // Parse shell params from command line (overrides properties)
  shell::parseShellParams(args, shellParams_);

  // Debug: Log after command line parsing
  if (shellParams_.benchmarkParams.has_value()) {
    __android_log_print(ANDROID_LOG_INFO,
                        "igl",
                        "[IGL Benchmark] benchmarkParams SET after parseShellParams: duration=%zu, "
                        "interval=%zu\n",
                        shellParams_.benchmarkParams->benchmarkDurationMs,
                        shellParams_.benchmarkParams->reportIntervalMs);
  }

  switch (backendVersion_.flavor) {
#if IGL_BACKEND_OPENGL
  case igl::BackendFlavor::OpenGL_ES: {
    auto hwDevice = opengl::egl::HWDevice();
    // Decide which backend api to use, default as GLES3
    d = hwDevice.create(backendVersion_, &result);
    shellParams_.shouldPresent = false;

    if (shellParams_.isHeadless) {
      width_ = static_cast<uint32_t>(shellParams_.viewportSize.x);
      height_ = static_cast<uint32_t>(shellParams_.viewportSize.y);
    }

    if (swapchainColorTextureFormat == TextureFormat::Invalid) {
      swapchainColorTextureFormat_ = TextureFormat::RGBA_SRGB;
    }

    if (!d->hasFeature(DeviceFeatures::SRGB) && !d->hasFeature(DeviceFeatures::SRGBSwapchain)) {
      swapchainColorTextureFormat_ = TextureFormat::RGBA_UNorm8;
    }

    break;
  }
#endif

#if IGL_BACKEND_VULKAN
  case igl::BackendFlavor::Vulkan: {
    IGL_DEBUG_ASSERT(nativeWindow != nullptr);
    vulkan::VulkanContextConfig config;
    config.terminateOnValidationError = true;
    config.requestedSwapChainTextureFormat = swapchainColorTextureFormat;
    // Don't use headless mode on Android - instead we'll render to offscreen surface
    config.headless = false;
    factory.configureVulkanContext(shellParams_, config);

    auto ctx = vulkan::HWDevice::createContext(config, nativeWindow);
    if (!ctx) {
      __android_log_print(ANDROID_LOG_ERROR, "igl", "Vulkan 1.1 or newer is required.\n");
      return;
    }

    auto devices =
        vulkan::HWDevice::queryDevices(*ctx, HWDeviceQueryDesc(HWDeviceType::Unknown), &result);

    if (!result.isOk() || devices.empty()) {
      __android_log_print(ANDROID_LOG_ERROR, "igl", "Error: %s\n", result.message.c_str());
      return;
    }

    if (shellParams_.isHeadless) {
      // Use viewport size from shell params for headless mode
      width_ = static_cast<uint32_t>(shellParams_.viewportSize.x);
      height_ = static_cast<uint32_t>(shellParams_.viewportSize.y);
    } else {
      width_ = static_cast<uint32_t>(ANativeWindow_getWidth(nativeWindow));
      height_ = static_cast<uint32_t>(ANativeWindow_getHeight(nativeWindow));
    }

    // https://github.com/gpuweb/gpuweb/issues/4283
    // Only 49.5% of Android devices support dualSrcBlend.
    // Android devices that do not support dualSrcBlend primarily use ARM, ImgTec, and Qualcomm
    // GPUs.
    // https://vulkan.gpuinfo.org/listdevicescoverage.php?feature=dualSrcBlend&platform=android&option=not
    igl::vulkan::VulkanFeatures vulkanFeatures(config);
    vulkanFeatures.vkPhysicalDeviceFeatures2.features.dualSrcBlend = VK_FALSE;
    factory.addVulkanFeatureRequests(vulkanFeatures);

    // Opt-in: sessions may declare extra Vulkan device extensions via the factory.
    // Default is empty -> behavior unchanged for sessions that declare nothing.
    const std::vector<std::string> extraDeviceExtensions =
        factory.requestedVulkanDeviceExtensions();
    std::vector<const char*> extraDeviceExtensionPtrs;
    extraDeviceExtensionPtrs.reserve(extraDeviceExtensions.size());
    for (const auto& ext : extraDeviceExtensions) {
      extraDeviceExtensionPtrs.push_back(ext.c_str());
    }

    d = vulkan::HWDevice::create(std::move(ctx),
                                 devices[0],
                                 width_, // width
                                 height_, // height,,
                                 extraDeviceExtensionPtrs.size(),
                                 extraDeviceExtensionPtrs.empty() ? nullptr
                                                                  : extraDeviceExtensionPtrs.data(),
                                 &vulkanFeatures,
                                 "TinyRenderer",
                                 &result);
    break;
  }
#endif

  default: {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    return;
  }
  }

  IGL_DEBUG_ASSERT(d != nullptr);
  // We want to catch failed device creation instead of letting implicitly fail
  IGL_SOFT_ASSERT(result.isOk());
  if (d) {
    // Verify multiview support — if unsupported, report and run as usual
    if (shellParams_.forceMultiview && !d->hasFeature(DeviceFeatures::Multiview)) {
      __android_log_print(
          ANDROID_LOG_WARN,
          "igl",
          "[IGL Shell] --force-multiview: multiview is not supported by this device. "
          "Running in normal mono mode.\n");
      shellParams_.forceMultiview = false;
      shellParams_.renderMode = igl::shell::RenderMode::Mono;
      shellParams_.viewParams.clear();
      shellParams_.shouldPresent = true;
    }

    platform_ = std::make_shared<igl::shell::PlatformAndroid>(std::move(d));
    IGL_DEBUG_ASSERT(platform_ != nullptr);
    static_cast<igl::shell::FileLoaderAndroid&>(platform_->getFileLoader()).setAssetManager(mgr);

    // Sync surface dimensions to ShellParams for multiview
    if (shellParams_.forceMultiview) {
      shellParams_.viewportSize = glm::vec2(width_, height_);
      shellParams_.nativeSurfaceDimensions = glm::ivec2(width_, height_);
    }

    const ContextGuard guard(platform_->getDevice()); // wrap 'session_' operations

    session_ = factory.createRenderSession(platform_);

    session_->setShellParams(shellParams_);
    IGL_DEBUG_ASSERT(session_ != nullptr);
    session_->initialize();
  }

  // Outside the ContextGuard above, so the EGL lever is read against the context the shell
  // presents on rather than whichever one IGL left current while building the device.
  installPresentationRateBackend();
}

void TinyRenderer::setDisplayRates(float currentHz, float maxHz, bool reinstallBackend) {
  displayRates_->currentHz = currentHz;
  displayRates_->maxHz = maxHz;
  if (reinstallBackend) {
    installPresentationRateBackend();
  }
}

void TinyRenderer::installPresentationRateBackend() {
  if (!platform_) {
    return;
  }

  // Ask the compositor for the panel's best mode once, here, rather than once per request.
  // ANativeWindow_setFrameRate() is a preference with no confirmation, so it can never be
  // what a grant rests on; asking for the maximum up front instead means the rungs below
  // are whole divisions of a panel that is already running as fast as it will. Advisory, so
  // a refusal is not fatal — the rungs are then divisions of whatever mode it stayed in.
  if (nativeWindow_ != nullptr && displayRates_->maxHz > 0.0f) {
    (void)igl::shell::setNativeWindowFrameRate(nativeWindow_, displayRates_->maxHz);
  }

  igl::shell::FrameDivider divider;
#if IGL_BACKEND_OPENGL
  if (backendVersion_.flavor == igl::BackendFlavor::OpenGL_ES) {
    divider = eglSwapIntervalDivider();
  }
#endif
  // Vulkan gets no divider: Android's loader enumerates no present mode that paces below
  // the refresh rate, and this diff adds no frame-skipping pacer. The backend still grants
  // DisplayMaximum, which needs no division, and reports every capped rung as unsupported
  // rather than pretending to hold it.

  // The rates are shared rather than copied, so setDisplayRates() reaches a backend that is
  // already installed and every request reads the display as it is now.
  platform_->getPresentationRateController().setBackend(igl::shell::createFrameDividerRateBackend(
      [rates = displayRates_]() { return *rates; }, std::move(divider)));
}

void TinyRenderer::recreateSwapchain(ANativeWindow* nativeWindow, bool createSurface) {
  // Outside the backend guard on purpose: the caller handed over a reference whether or not
  // Vulkan is compiled in, so dropping it here would leak on a GLES-only build.
  adoptNativeWindow(nativeWindow);
  rebuildSwapchain(createSurface);
}

void TinyRenderer::rebuildSwapchain(bool createSurface) {
#if IGL_BACKEND_VULKAN
  if (!platform_ || !session_) {
    return;
  }
  if (nativeWindow_ == nullptr && !shellParams_.isHeadless) {
    // No window to build against: it was released and no replacement arrived (a null
    // Surface reaches onSurfacesChanged() as "nothing new", not as "window gone"). There is
    // no surface to size from or create, so there is nothing to rebuild.
    __android_log_print(ANDROID_LOG_WARN,
                        "igl",
                        "[IGL Shell] rebuildSwapchain: no native window held; skipping rebuild\n");
    return;
  }
  if (!shellParams_.isHeadless) {
    width_ = static_cast<uint32_t>(ANativeWindow_getWidth(nativeWindow_));
    height_ = static_cast<uint32_t>(ANativeWindow_getHeight(nativeWindow_));
  }

  auto* platformDevice = platform_->getDevice().getPlatformDevice<igl::vulkan::PlatformDevice>();
  // need clear the cached textures before recreate swap chain.
  platformDevice->clear();

  auto& vulkanDevice = static_cast<igl::vulkan::Device&>(platform_->getDevice());
  auto& vkContext = vulkanDevice.getVulkanContext();

  if (createSurface) {
    vkContext.createSurface(nativeWindow_, nullptr);
  }
  vkContext.initSwapchain(width_, height_);

  // need release frame buffer when recreate swap chain
  session_->releaseFramebuffer();

  // The rate the old backend was holding belonged to the window it was built around, which
  // may not be this one. Reinstalling re-applies the last request against the new window.
  installPresentationRateBackend();
#endif
}

#if IGL_BACKEND_VULKAN
namespace {
// Fullscreen triangle vertex shader for stereo side-by-side present
const char* kStereoPresentVS = R"(
#version 460
layout(location = 0) out vec2 uv;
out gl_PerVertex { vec4 gl_Position; };
void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
)";

// Fragment shader: left half = layer 0, right half = layer 1
const char* kStereoPresentFS = R"(
#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2DArray stereoTex;
void main() {
    float layer = uv.x < 0.5 ? 0.0 : 1.0;
    vec2 texCoord = vec2(uv.x < 0.5 ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
    outColor = texture(stereoTex, vec3(texCoord, layer));
}
)";
} // namespace
#endif

void TinyRenderer::initStereoPresent(IDevice& device) {
#if IGL_BACKEND_VULKAN
  if (stereoPresentInitialized_) {
    return;
  }

  presentQueue_ = device.createCommandQueue({}, nullptr);

  const SamplerStateDesc samplerDesc = {
      .minFilter = SamplerMinMagFilter::Linear,
      .magFilter = SamplerMinMagFilter::Linear,
      .addressModeU = SamplerAddressMode::Clamp,
      .addressModeV = SamplerAddressMode::Clamp,
  };
  presentSampler_ = device.createSamplerState(samplerDesc, nullptr);

  auto shaderStages = igl::ShaderStagesCreator::fromModuleStringInput(
      device, kStereoPresentVS, "main", "", kStereoPresentFS, "main", "", nullptr);

  const RenderPipelineDesc pipelineDesc = {
      .shaderStages = std::move(shaderStages),
      .targetDesc = {.colorAttachments = {{.textureFormat = swapchainColor_->getFormat()}}},
  };
  presentPipeline_ = device.createRenderPipeline(pipelineDesc, nullptr);

  stereoPresentInitialized_ = true;
  __android_log_print(
      ANDROID_LOG_INFO, "igl", "[IGL Shell] Stereo present initialized for --force-multiview\n");
#endif
}

void TinyRenderer::stereoPresent(IDevice& device) {
#if IGL_BACKEND_VULKAN
  if (!swapchainColor_ || !multiviewColor_) {
    return;
  }

  if (!stereoPresentInitialized_) {
    initStereoPresent(device);
  }

  const FramebufferDesc fbDesc = {
      .colorAttachments = {{.texture = swapchainColor_}},
  };
  Result ret;
  const auto framebuffer = device.createFramebuffer(fbDesc, &ret);
  if (!ret.isOk()) {
    return;
  }

  auto cmdBuf = presentQueue_->createCommandBuffer({}, nullptr);

  const RenderPassDesc renderPass = {
      .colorAttachments = {{
          .loadAction = LoadAction::Clear,
          .storeAction = StoreAction::Store,
          .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
      }},
  };

  auto encoder = cmdBuf->createRenderCommandEncoder(renderPass, framebuffer);

  // Set viewport to match swapchain dimensions exactly
  const auto swapDims = swapchainColor_->getDimensions();
  const Viewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(swapDims.width),
      .height = static_cast<float>(swapDims.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  encoder->bindViewport(viewport);
  const ScissorRect scissor = {
      .x = 0,
      .y = 0,
      .width = swapDims.width,
      .height = swapDims.height,
  };
  encoder->bindScissorRect(scissor);

  encoder->bindRenderPipelineState(presentPipeline_);
  encoder->bindTexture(0, BindTarget::kFragment, multiviewColor_.get());
  encoder->bindSamplerState(0, BindTarget::kFragment, presentSampler_.get());
  encoder->draw(3);
  encoder->endEncoding();

  cmdBuf->present(swapchainColor_);
  presentQueue_->submit(*cmdBuf);
#endif
}

bool TinyRenderer::render(float displayScale) {
  if (!platform_ || !session_) {
    return false;
  }
  // process user input
  IGL_DEBUG_ASSERT(platform_ != nullptr);
  platform_->getInputDispatcher().processEvents();

  // draw
  Result result;
  SurfaceTextures surfaceTextures;

  if (shellParams_.isHeadless) {
    // In headless mode, create offscreen textures instead of native drawable textures
    auto& device = platform_->getDevice();

    // Create or reuse offscreen color texture
    if (!offscreenColorTexture_ || offscreenColorTexture_->getSize().width != width_ ||
        offscreenColorTexture_->getSize().height != height_) {
      TextureDesc colorTexDesc = TextureDesc::new2D(swapchainColorTextureFormat_,
                                                    width_,
                                                    height_,
                                                    TextureDesc::TextureUsageBits::Attachment |
                                                        TextureDesc::TextureUsageBits::Sampled);
      colorTexDesc.storage = ResourceStorage::Private;
      offscreenColorTexture_ = device.createTexture(colorTexDesc, &result);
      IGL_DEBUG_ASSERT(result.isOk());
      IGL_SOFT_ASSERT(result.isOk());
    }

    // Create or reuse offscreen depth texture
    if (!offscreenDepthTexture_ || offscreenDepthTexture_->getSize().width != width_ ||
        offscreenDepthTexture_->getSize().height != height_) {
      TextureDesc depthTexDesc = TextureDesc::new2D(
          TextureFormat::Z_UNorm24, width_, height_, TextureDesc::TextureUsageBits::Attachment);
      depthTexDesc.storage = ResourceStorage::Private;
      offscreenDepthTexture_ = device.createTexture(depthTexDesc, &result);
      IGL_DEBUG_ASSERT(result.isOk());
      IGL_SOFT_ASSERT(result.isOk());
    }

    surfaceTextures.color = offscreenColorTexture_;
    surfaceTextures.depth = offscreenDepthTexture_;
  } else {
    // Normal mode: create surface textures from native drawable
    switch (backendVersion_.flavor) {
#if IGL_BACKEND_OPENGL
    case igl::BackendFlavor::OpenGL_ES: {
      auto* platformDevice =
          platform_->getDevice().getPlatformDevice<opengl::egl::PlatformDevice>();
      surfaceTextures.color =
          platformDevice->createTextureFromNativeDrawable(swapchainColorTextureFormat_, &result);
      surfaceTextures.depth =
          platformDevice->createTextureFromNativeDepth(igl::TextureFormat::Z_UNorm24, &result);
      break;
    }
#endif

#if IGL_BACKEND_VULKAN
    case igl::BackendFlavor::Vulkan: {
      auto* platformDevice = platform_->getDevice().getPlatformDevice<vulkan::PlatformDevice>();
      surfaceTextures.color = platformDevice->createTextureFromNativeDrawable(&result);
      surfaceTextures.depth =
          platformDevice->createTextureFromNativeDepth(width_, height_, &result);

      // Force multiview: create 2-layer array textures for multiview rendering.
      // stereoPresent() blits both layers side-by-side to the swapchain after update.
      if (shellParams_.forceMultiview && !shellParams_.isHeadless &&
          platform_->getDevice().hasFeature(DeviceFeatures::Multiview)) {
        shellParams_.shouldPresent = false;
        swapchainColor_ = std::move(surfaceTextures.color);
        const auto dimensions = swapchainColor_->getDimensions();
        const auto colorFormat = swapchainColor_->getFormat();
        const auto depthFormat = surfaceTextures.depth->getFormat();

        if (!multiviewColor_ || multiviewColor_->getDimensions() != dimensions) {
          const TextureDesc colorDesc = {
              .width = dimensions.width,
              .height = dimensions.height,
              .depth = 1,
              .numLayers = 2,
              .numSamples = 1,
              .usage = TextureDesc::TextureUsageBits::Attachment |
                       TextureDesc::TextureUsageBits::Sampled,
              .numMipLevels = 1,
              .type = TextureType::TwoDArray,
              .format = colorFormat,
              .storage = ResourceStorage::Private,
          };
          multiviewColor_ = platform_->getDevice().createTexture(colorDesc, &result);

          const TextureDesc depthDesc = {
              .width = dimensions.width,
              .height = dimensions.height,
              .depth = 1,
              .numLayers = 2,
              .numSamples = 1,
              .usage = TextureDesc::TextureUsageBits::Attachment,
              .numMipLevels = 1,
              .type = TextureType::TwoDArray,
              .format = depthFormat,
              .storage = ResourceStorage::Private,
          };
          multiviewDepth_ = platform_->getDevice().createTexture(depthDesc, &result);

          __android_log_print(ANDROID_LOG_INFO,
                              "igl",
                              "[IGL Shell] Created multiview textures: %ux%u, 2 layers\n",
                              dimensions.width,
                              dimensions.height);
        }

        surfaceTextures.color = multiviewColor_;
        surfaceTextures.depth = multiviewDepth_;
      }
      break;
    }
#endif

    default:
      Result::setResult(&result, Result::Code::Unsupported, "Invalid backend");
      break;
    }
    IGL_DEBUG_ASSERT(result.isOk());
    IGL_SOFT_ASSERT(result.isOk());
  }

  const ContextGuard guard(platform_->getDevice()); // wrap 'session_' operations

  platform_->getDevice().setCurrentThread();
  session_->setPixelsPerPoint(displayScale);
  session_->runUpdate(std::move(surfaceTextures));

#if IGL_BACKEND_VULKAN
  if (shellParams_.forceMultiview && swapchainColor_) {
    stereoPresent(platform_->getDevice());
  }
#endif

  // Return true if the application should exit (e.g., benchmark timeout)
  return session_->appParams().exitRequested;
}

void TinyRenderer::onSurfacesChanged(ANativeWindow* nativeWindow, int width, int height) {
  // A null window means the caller had no Surface to hand over on this call, not that the
  // window went away — that arrives as releaseNativeWindowIfHeld(). Keeping the one already held is
  // what lets the Vulkan branch below rebuild its swapchain against a window it still owns.
  if (nativeWindow != nullptr) {
    adoptNativeWindow(nativeWindow);
  }
  if (!platform_ || !session_ || shellParams_.isHeadless) {
    return;
  }
  width_ = static_cast<uint32_t>(width);
  height_ = static_cast<uint32_t>(height);
#if IGL_BACKEND_OPENGL
  if (backendVersion_.flavor == igl::BackendFlavor::OpenGL_ES) {
    auto* readSurface = eglGetCurrentSurface(EGL_READ);
    auto* drawSurface = eglGetCurrentSurface(EGL_DRAW);

    IGL_DEBUG_ASSERT(platform_ != nullptr);
    Result result;
    platform_->getDevice().getPlatformDevice<opengl::egl::PlatformDevice>()->updateSurfaces(
        readSurface, drawSurface, &result);
    IGL_DEBUG_ASSERT(result.isOk());
    IGL_SOFT_ASSERT(result.isOk());

    // A swap interval belongs to the EGLSurface it was set on, so the replacement surface
    // starts back at one while the controller still holds the old rung's grant. Reinstall
    // here, with the replacement context and surface current, so the divider is rebuilt
    // against them and setBackend() re-applies the last request through it.
    installPresentationRateBackend();
  }
#endif

#if IGL_BACKEND_VULKAN
  if (backendVersion_.flavor == igl::BackendFlavor::Vulkan) {
    // rebuildSwapchain() rather than recreateSwapchain(): the window was already adopted at
    // the top of this function, so going back through the adopt would release a reference
    // this path never acquired.
    rebuildSwapchain(false);
    platform_->updatePreRotationMatrix();
  }
#endif
}

void TinyRenderer::touchEvent(bool isDown, float x, float y, float dx, float dy) {
  if (!platform_) {
    return;
  }
  const float scale = platform_->getDisplayContext().pixelsPerPoint;
  IGL_DEBUG_ASSERT(scale > 0.0f);
  platform_->getInputDispatcher().queueEvent(
      igl::shell::TouchEvent(isDown, x / scale, y / scale, dx / scale, dy / scale));
}

void TinyRenderer::setClearColorValue(float r, float g, float b, float a) {
  shellParams_.clearColorValue = {r, g, b, a};
}

} // namespace igl::samples
