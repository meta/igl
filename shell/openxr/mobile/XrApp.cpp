/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/openxr/XrApp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>

#if IGL_PLATFORM_ANDROID
// NOLINTNEXTLINE(facebook-unused-include-check)
#include <android_native_app_glue.h>
#endif

#include <glm/gtc/type_ptr.hpp>
#include <xr_linear.h>

#if IGL_PLATFORM_ANDROID
#include <shell/shared/fileLoader/android/FileLoaderAndroid.h>
#include <shell/shared/platform/android/PlatformAndroid.h>
#endif
#if IGL_PLATFORM_WINDOWS
#include <shell/shared/platform/win/PlatformWin.h>
#endif

#include <shell/openxr/XrCompositionProjection.h>
#include <shell/openxr/XrCompositionQuad.h>
#include <shell/openxr/XrHands.h>
#include <shell/openxr/XrLog.h>
#include <shell/openxr/XrPassthrough.h>
#include <shell/openxr/impl/XrAppImpl.h>
#include <shell/openxr/impl/XrSwapchainProviderImpl.h>
#include <shell/shared/input/InputDispatcher.h>
#include <shell/shared/input/IntentListener.h>
#include <shell/shared/renderSession/AppParams.h>
#include <shell/shared/renderSession/DefaultRenderSessionFactory.h>
#include <shell/shared/renderSession/ShellParams.h>

#if !IGL_PLATFORM_ANDROID
struct android_app {};
struct AAssetManager {};
#endif

namespace igl::shell::openxr {
constexpr auto kAppName = "IGL Shell OpenXR";
constexpr auto kEngineName = "IGL";
constexpr auto kSupportedViewConfigType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

XrApp::XrApp(std::unique_ptr<impl::XrAppImpl>&& impl, bool shouldPresent) :
  impl_(std::move(impl)), shellParams_(std::make_unique<ShellParams>()) {
  shellParams_->shouldPresent = shouldPresent;
  viewports_.fill({XR_TYPE_VIEW_CONFIGURATION_VIEW});
  views_.fill({XR_TYPE_VIEW});
#ifdef USE_COMPOSITION_LAYER_QUAD
  useQuadLayerComposition_ = true;
#endif
}

XrApp::~XrApp() {
  if (!initialized_) {
    return;
  }

  renderSession_.reset();
  compositionLayers_.clear();
  passthrough_.reset();
  hands_.reset();

  if (currentSpace_ != XR_NULL_HANDLE) {
    xrDestroySpace(currentSpace_);
  }
  if (headSpace_ != XR_NULL_HANDLE) {
    xrDestroySpace(headSpace_);
  }
  if (session_ != XR_NULL_HANDLE) {
    xrDestroySession(session_);
  }
  if (instance_ != XR_NULL_HANDLE) {
    xrDestroyInstance(instance_);
  }

  platform_.reset();
}

XrInstance XrApp::instance() const {
  return instance_;
}

XrSession XrApp::session() const {
  return session_;
}

bool XrApp::checkExtensions() {
  PFN_xrEnumerateInstanceExtensionProperties xrEnumerateInstanceExtensionProperties = nullptr;
  const XrResult result = xrGetInstanceProcAddr(
      XR_NULL_HANDLE,
      "xrEnumerateInstanceExtensionProperties",
      reinterpret_cast<PFN_xrVoidFunction*>(&xrEnumerateInstanceExtensionProperties));
  XR_CHECK(result);
  if (result != XR_SUCCESS) {
    IGL_LOG_ERROR("Failed to get xrEnumerateInstanceExtensionProperties function pointer.\n");
    return false;
  }

  uint32_t numExtensions = 0;
  XR_CHECK(xrEnumerateInstanceExtensionProperties(nullptr, 0, &numExtensions, nullptr));
  IGL_LOG_INFO("xrEnumerateInstanceExtensionProperties found %u extension(s).\n", numExtensions);

  extensions_.resize(numExtensions, {XR_TYPE_EXTENSION_PROPERTIES});

  XR_CHECK(xrEnumerateInstanceExtensionProperties(
      nullptr, numExtensions, &numExtensions, extensions_.data()));
  for (uint32_t i = 0; i < numExtensions; i++) {
    IGL_LOG_INFO("Extension #%d = '%s'.\n", i, extensions_[i].extensionName);
  }

  auto checkExtensionSupported = [this](const char* name) {
    return std::any_of(std::begin(extensions_),
                       std::end(extensions_),
                       [&](const XrExtensionProperties& extension) {
                         return strcmp(extension.extensionName, name) == 0;
                       });
  };

  // Check all required extensions are supported.
  auto requiredExtensionsImpl = impl_->getXrRequiredExtensions();
  for (const char* requiredExtension : requiredExtensionsImpl) {
    if (!checkExtensionSupported(requiredExtension)) {
      IGL_LOG_ERROR("Extension %s is required, but not supported.\n", requiredExtension);
      return false;
    }
  }

  auto checkNeedEnableExtension = [this](const char* name) {
    return std::find_if(std::begin(enabledExtensions_),
                        std::end(enabledExtensions_),
                        [&](const char* extensionName) {
                          return strcmp(extensionName, name) == 0;
                        }) == std::end(enabledExtensions_);
  };

  // Add required extensions to enabledExtensions_.
  for (const char* requiredExtension : requiredExtensionsImpl) {
    if (checkNeedEnableExtension(requiredExtension)) {
      IGL_LOG_INFO("Extension %s is enabled.\n", requiredExtension);
      enabledExtensions_.push_back(requiredExtension);
    }
  }

  // Get list of all optional extensions.
  auto optionalExtensionsImpl = impl_->getXrOptionalExtensions();
  std::vector<const char*> additionalOptionalExtensions = {
#if IGL_PLATFORM_ANDROID
      XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
#endif // IGL_PLATFORM_ANDROID
#ifdef XR_FB_composition_layer_alpha_blend
      XR_FB_COMPOSITION_LAYER_ALPHA_BLEND_EXTENSION_NAME,
#endif // XR_FB_composition_layer_alpha_blend
  };

  optionalExtensionsImpl.insert(optionalExtensionsImpl.end(),
                                std::begin(XrPassthrough::getExtensions()),
                                std::end(XrPassthrough::getExtensions()));

  optionalExtensionsImpl.insert(optionalExtensionsImpl.end(),
                                std::begin(XrHands::getExtensions()),
                                std::end(XrHands::getExtensions()));

  optionalExtensionsImpl.insert(optionalExtensionsImpl.end(),
                                std::begin(XrRefreshRate::getExtensions()),
                                std::end(XrRefreshRate::getExtensions()));

  optionalExtensionsImpl.insert(optionalExtensionsImpl.end(),
                                std::begin(additionalOptionalExtensions),
                                std::end(additionalOptionalExtensions));

  // Add optional extensions to enabledExtensions_.
  for (const char* optionalExtension : optionalExtensionsImpl) {
    if (checkExtensionSupported(optionalExtension)) {
      supportedOptionalXrExtensions_.insert(optionalExtension);
      if (checkNeedEnableExtension(optionalExtension)) {
        IGL_LOG_INFO("Extension %s is enabled.\n", optionalExtension);
        enabledExtensions_.push_back(optionalExtension);
      }
    } else {
      IGL_LOG_INFO("Warning: Extension %s is not supported.\n", optionalExtension);
    }
  }

  return true;
}

bool XrApp::createInstance() {
  XrApplicationInfo appInfo = {};
  std::snprintf(appInfo.applicationName, sizeof(appInfo.applicationName), "%s", kAppName);
  appInfo.applicationVersion = 0;
  std::snprintf(appInfo.engineName, sizeof(appInfo.engineName), "%s", kEngineName);
  appInfo.engineVersion = 0;
  appInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34);

  const XrInstanceCreateInfo instanceCreateInfo = {
      .type = XR_TYPE_INSTANCE_CREATE_INFO,
#if IGL_PLATFORM_ANDROID
      .next = instanceCreateInfoAndroidSupported() ? &instanceCreateInfoAndroid_ : nullptr,
#else
      .next = nullptr,
#endif // IGL_PLATFORM_ANDROID
      .createFlags = 0,
      .applicationInfo = appInfo,
      .enabledApiLayerCount = 0,
      .enabledApiLayerNames = nullptr,
      .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions_.size()),
      .enabledExtensionNames = enabledExtensions_.data(),
  };

  const XrResult initResult = xrCreateInstance(&instanceCreateInfo, &instance_);
  XR_CHECK(initResult);
  if (initResult != XR_SUCCESS) {
    IGL_LOG_ERROR("Failed to create XR instance: %d.\n", initResult);
    return false;
  }

  XR_CHECK(xrGetInstanceProperties(instance_, &instanceProps_));
  IGL_LOG_INFO("Runtime %s: Version : %u.%u.%u\n",
               instanceProps_.runtimeName,
               XR_VERSION_MAJOR(instanceProps_.runtimeVersion),
               XR_VERSION_MINOR(instanceProps_.runtimeVersion),
               XR_VERSION_PATCH(instanceProps_.runtimeVersion));

  return true;
} // namespace igl::shell::openxr

bool XrApp::createSystem() {
  const XrSystemGetInfo systemGetInfo = {
      .type = XR_TYPE_SYSTEM_GET_INFO,
      .formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY,
  };

  const XrResult result = xrGetSystem(instance_, &systemGetInfo, &systemId_);
  XR_CHECK(result);
  if (result != XR_SUCCESS) {
    IGL_LOG_ERROR("Failed to get system.\n");
    return false;
  }

  XR_CHECK(xrGetSystemProperties(instance_, systemId_, &systemProps_));

  IGL_LOG_INFO(
      "System Properties: Name=%s VendorId=%x\n", systemProps_.systemName, systemProps_.vendorId);
  IGL_LOG_INFO("System Graphics Properties: MaxWidth=%d MaxHeight=%d MaxLayers=%d\n",
               systemProps_.graphicsProperties.maxSwapchainImageWidth,
               systemProps_.graphicsProperties.maxSwapchainImageHeight,
               systemProps_.graphicsProperties.maxLayerCount);
  IGL_LOG_INFO("System Tracking Properties: OrientationTracking=%s PositionTracking=%s\n",
               systemProps_.trackingProperties.orientationTracking ? "True" : "False",
               systemProps_.trackingProperties.positionTracking ? "True" : "False");
  IGL_LOG_INFO("System Hand Tracking Properties: Supported=%s\n",
               handTrackingSystemProps_.supportsHandTracking ? "True" : "False");
  return true;
}

bool XrApp::enumerateViewConfigurations() {
  uint32_t numViewConfigs = 0;
  XR_CHECK(xrEnumerateViewConfigurations(instance_, systemId_, 0, &numViewConfigs, nullptr));

  std::vector<XrViewConfigurationType> viewConfigTypes(numViewConfigs);
  XR_CHECK(xrEnumerateViewConfigurations(
      instance_, systemId_, numViewConfigs, &numViewConfigs, viewConfigTypes.data()));

  IGL_LOG_INFO("Available Viewport Configuration Types: %d\n", numViewConfigs);
  auto foundViewConfig = false;
  for (auto& viewConfigType : viewConfigTypes) {
    IGL_LOG_INFO("View configuration type %d : %s\n",
                 viewConfigType,
                 viewConfigType == kSupportedViewConfigType ? "Selected" : "");

    if (viewConfigType != kSupportedViewConfigType) {
      continue;
    }

    // Check properties
    XrViewConfigurationProperties viewConfigProps = {XR_TYPE_VIEW_CONFIGURATION_PROPERTIES};
    XR_CHECK(
        xrGetViewConfigurationProperties(instance_, systemId_, viewConfigType, &viewConfigProps));
    IGL_LOG_INFO("FovMutable=%s ConfigurationType %d\n",
                 viewConfigProps.fovMutable ? "true" : "false",
                 viewConfigProps.viewConfigurationType);

    // Check views
    uint32_t numViewports = 0;
    XR_CHECK(xrEnumerateViewConfigurationViews(
        instance_, systemId_, viewConfigType, 0, &numViewports, nullptr));

    if (!IGL_DEBUG_VERIFY(numViewports == XrComposition::kNumViews)) {
      IGL_LOG_ERROR(
          "numViewports must be %d. Make sure XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO is used.\n",
          XrComposition::kNumViews);
      return false;
    }

    XR_CHECK(xrEnumerateViewConfigurationViews(
        instance_, systemId_, viewConfigType, numViewports, &numViewports, viewports_.data()));

    for (auto& view : viewports_) {
      (void)view; // doesn't compile in release for unused variable
      IGL_LOG_INFO("Viewport [%d]: Recommended Width=%d Height=%d SampleCount=%d\n",
                   view,
                   view.recommendedImageRectWidth,
                   view.recommendedImageRectHeight,
                   view.recommendedSwapchainSampleCount);

      IGL_LOG_INFO("Viewport [%d]: Max Width=%d Height=%d SampleCount=%d\n",
                   view,
                   view.maxImageRectWidth,
                   view.maxImageRectHeight,
                   view.maxSwapchainSampleCount);
    }

    viewConfigProps_ = viewConfigProps;

    foundViewConfig = true;

    break;
  }

  IGL_DEBUG_ASSERT(
      foundViewConfig, "XrViewConfigurationType %d not found.", kSupportedViewConfigType);

  return true;
}

void XrApp::enumerateReferenceSpaces() {
  uint32_t numRefSpaceTypes = 0;
  XR_CHECK(xrEnumerateReferenceSpaces(session_, 0, &numRefSpaceTypes, nullptr));

  std::vector<XrReferenceSpaceType> refSpaceTypes(numRefSpaceTypes);

  XR_CHECK(xrEnumerateReferenceSpaces(
      session_, numRefSpaceTypes, &numRefSpaceTypes, refSpaceTypes.data()));

  stageSpaceSupported_ =
      std::any_of(std::begin(refSpaceTypes), std::end(refSpaceTypes), [](const auto& type) {
        return type == XR_REFERENCE_SPACE_TYPE_STAGE;
      });
  IGL_LOG_INFO("OpenXR stage reference space is %s\n",
               stageSpaceSupported_ ? "supported" : "not supported");
}

void XrApp::enumerateBlendModes() {
  uint32_t numBlendModes = 0;
  XR_CHECK(xrEnumerateEnvironmentBlendModes(
      instance_, systemId_, kSupportedViewConfigType, 0, &numBlendModes, nullptr));

  std::vector<XrEnvironmentBlendMode> blendModes(numBlendModes);
  XR_CHECK(xrEnumerateEnvironmentBlendModes(instance_,
                                            systemId_,
                                            kSupportedViewConfigType,
                                            numBlendModes,
                                            &numBlendModes,
                                            blendModes.data()));

  additiveBlendingSupported_ =
      std::any_of(std::begin(blendModes), std::end(blendModes), [](const auto& type) {
        return type == XR_ENVIRONMENT_BLEND_MODE_ADDITIVE;
      });
  IGL_LOG_INFO("OpenXR additive blending %s\n",
               additiveBlendingSupported_ ? "supported" : "not supported");
}

bool XrApp::initialize(const struct android_app* app, const InitParams& params) {
  if (initialized_) {
    return false;
  }

#if IGL_PLATFORM_ANDROID
  PFN_xrInitializeLoaderKHR xrInitializeLoaderKHR = nullptr;
  XR_CHECK(xrGetInstanceProcAddr(XR_NULL_HANDLE,
                                 "xrInitializeLoaderKHR",
                                 reinterpret_cast<PFN_xrVoidFunction*>(&xrInitializeLoaderKHR)));
  if (xrInitializeLoaderKHR) {
    XrLoaderInitInfoAndroidKHR loaderInitializeInfoAndroid = {
        .type = XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR,
        .next = nullptr,
        .applicationVM = app->activity->vm,
        .applicationContext = app->activity->clazz,
    };

    XR_CHECK(xrInitializeLoaderKHR(
        reinterpret_cast<XrLoaderInitInfoBaseHeaderKHR*>(&loaderInitializeInfoAndroid)));
  }

  instanceCreateInfoAndroid_.applicationVM = app->activity->vm;
  instanceCreateInfoAndroid_.applicationActivity = app->activity->clazz;
#endif

  if (!checkExtensions()) {
    return false;
  }

  if (!createInstance()) {
    return false;
  }

  if (!createSystem()) {
    return false;
  }

  if (!enumerateViewConfigurations()) {
    return false;
  }

  std::unique_ptr<IDevice> device;
  device = impl_->initIGL(instance_, systemId_);
  if (!device) {
    IGL_LOG_ERROR("Failed to initialize IGL\n");
    return false;
  }

#if IGL_WGL
  // Single stereo render pass is not supported for OpenGL on Windows.
  useSinglePassStereo_ = false;
#else
  useSinglePassStereo_ = useSinglePassStereo_ && device->hasFeature(DeviceFeatures::Multiview);
#endif

#if IGL_PLATFORM_ANDROID
  createShellSession(std::move(device), app->activity->assetManager);
#else
  createShellSession(std::move(device), nullptr);
#endif

  session_ = impl_->initXrSession(instance_, systemId_, platform_->getDevice(), sessionConfig_);
  if (session_ == XR_NULL_HANDLE) {
    IGL_LOG_ERROR("Failed to initialize graphics system\n");
    return false;
  }

  // The following are initialization steps that happen after XrSession is created.
  enumerateReferenceSpaces();
  enumerateBlendModes();
  createSpaces();
  if (!initializeOptionalFeatures(params)) {
    return false;
  }

  createActions();

  if (hands_) {
    hands_->updateMeshes(shellParams_->handMeshes);
  }

  IGL_DEBUG_ASSERT(renderSession_ != nullptr);
  renderSession_->initialize();

  if (useQuadLayerComposition_) {
    updateQuadComposition();
  } else {
    compositionLayers_.emplace_back(std::make_unique<XrCompositionProjection>(
        *impl_, platform_, session_, useSinglePassStereo_));

    // Resolution controlled by the render session via AppParams.
    // When useMaxRenderResolution is true, use maximum hardware resolution.
    // Otherwise, use the runtime's recommended resolution.
    const bool useMax = renderSession_ && renderSession_->appParams().useMaxRenderResolution;

    auto getWidth = [useMax](const XrViewConfigurationView& v) -> uint32_t {
      return useMax ? v.maxImageRectWidth : v.recommendedImageRectWidth;
    };
    auto getHeight = [useMax](const XrViewConfigurationView& v) -> uint32_t {
      return useMax ? v.maxImageRectHeight : v.recommendedImageRectHeight;
    };

    compositionLayers_.back()->updateSwapchainImageInfo(
        {impl::SwapchainImageInfo{
             .imageWidth = getWidth(viewports_[0]),
             .imageHeight = getHeight(viewports_[0]),
         },
         impl::SwapchainImageInfo{
             .imageWidth = getWidth(viewports_[1]),
             .imageHeight = getHeight(viewports_[1]),
         }});
  }

  initialized_ = true;
  return initialized_;
}

bool XrApp::initializeOptionalFeatures(const InitParams& params) {
  if (passthroughSupported()) {
    passthrough_ = std::make_unique<XrPassthrough>(instance_, session_);
    if (!passthrough_->initialize()) {
      return false;
    }
  }
  if (handTrackingSupported()) {
    hands_ = std::make_unique<XrHands>(instance_, session_, handTrackingMeshSupported());
    if (!hands_->initialize()) {
      return false;
    }
  }
  if (refreshRateExtensionSupported()) {
    refreshRate_ = std::make_unique<XrRefreshRate>(instance_, session_);
    if (!refreshRate_->initialize(params.refreshRateParams)) {
      return false;
    }
  }
  return true;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void XrApp::updateQuadComposition() noexcept {
  const auto& appParams = renderSession_->appParams();

  constexpr uint32_t kQuadLayerDefaultImageSize = 1024;

  const auto aspect = appParams.sizeY / appParams.sizeX;
  QuadLayerParams quadLayersParams = {
      .layerInfo = {{
#if USE_LOCAL_AR_SPACE
          .position = {0.0f, 0.0f, -1.0f},
#else
          .position = {0.0f, 0.0f, 0.0f},
#endif
          .size = {appParams.sizeX, appParams.sizeY},
          .blendMode = LayerBlendMode::AlphaBlend,
          .imageWidth = kQuadLayerDefaultImageSize,
          .imageHeight = static_cast<uint32_t>(kQuadLayerDefaultImageSize * aspect),
      }}};

  if (appParams.quadLayerParamsGetter) {
    auto params = appParams.quadLayerParamsGetter();
    if (params.numQuads() > 0) {
      quadLayersParams = std::move(params);
    }
  }

  std::array<impl::SwapchainImageInfo, XrComposition::kNumViews> swapchainImageInfo{};
  for (size_t i = 0; i < quadLayersParams.numQuads(); ++i) {
    swapchainImageInfo.fill({
        .imageWidth = quadLayersParams.layerInfo[i].imageWidth,
        .imageHeight = quadLayersParams.layerInfo[i].imageHeight,
    });

    if (i < compositionLayers_.size()) {
      auto* quadLayer = static_cast<XrCompositionQuad*>(compositionLayers_[i].get());
      quadLayer->updateQuadLayerInfo(quadLayersParams.layerInfo[i]);
      quadLayer->updateSwapchainImageInfo(swapchainImageInfo);
    } else {
      compositionLayers_.emplace_back(
          std::make_unique<XrCompositionQuad>(*impl_,
                                              platform_,
                                              session_,
                                              useSinglePassStereo_,
                                              alphaBlendCompositionSupported(),
                                              quadLayersParams.layerInfo[i]));
      compositionLayers_.back()->updateSwapchainImageInfo(swapchainImageInfo);
    }
  }

  // Remove any layers that are no longer needed.
  compositionLayers_.resize(quadLayersParams.numQuads());
}

void XrApp::createShellSession(std::unique_ptr<IDevice> device, AAssetManager* assetMgr) {
#if IGL_PLATFORM_ANDROID
  platform_ = std::make_shared<PlatformAndroid>(std::move(device));
  IGL_DEBUG_ASSERT(platform_ != nullptr);
  static_cast<FileLoaderAndroid&>(platform_->getFileLoader()).setAssetManager(assetMgr);
#elif IGL_PLATFORM_APPLE
  platform_ = std::make_shared<igl::shell::PlatformMac>(std::move(device));
#elif IGL_PLATFORM_WINDOWS
  platform_ = std::make_shared<igl::shell::PlatformWin>(std::move(device));
#endif

  auto factory = createDefaultRenderSessionFactory();
  const auto requestedSessionConfigs =
      factory->requestedSessionConfigs(ShellType::OpenXR, {impl_->suggestedSessionConfig()});
  if (IGL_DEBUG_VERIFY_NOT(requestedSessionConfigs.size() != 1)) {
    return;
  }
  sessionConfig_ = requestedSessionConfigs[0];

  renderSession_ = factory->createRenderSession(platform_);
  shellParams_->shellControlsViewParams = true;
  shellParams_->rightHandedCoordinateSystem = true;
  shellParams_->renderMode = useSinglePassStereo_ ? RenderMode::SinglePassStereo
                                                  : RenderMode::DualPassStereo;
  shellParams_->viewParams.resize(useSinglePassStereo_ ? 2 : 1);
  renderSession_->setShellParams(*shellParams_);
}

void XrApp::createSpaces() {
  XrReferenceSpaceCreateInfo spaceCreateInfo = {
      .type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
      .next = nullptr,
      .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW,
      .poseInReferenceSpace = {.orientation = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f}},
  };
  XR_CHECK(xrCreateReferenceSpace(session_, &spaceCreateInfo, &headSpace_));

#if USE_LOCAL_AR_SPACE
  spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
#else
  spaceCreateInfo.referenceSpaceType = stageSpaceSupported_ ? XR_REFERENCE_SPACE_TYPE_STAGE
                                                            : XR_REFERENCE_SPACE_TYPE_LOCAL;
#endif
  XR_CHECK(xrCreateReferenceSpace(session_, &spaceCreateInfo, &currentSpace_));
}

void XrApp::createActions() {
  // Create action set
  XrActionSetCreateInfo actionSetInfo = {XR_TYPE_ACTION_SET_CREATE_INFO};
  std::snprintf(actionSetInfo.actionSetName, sizeof(actionSetInfo.actionSetName), "%s", "gameplay");
  std::snprintf(actionSetInfo.localizedActionSetName,
                sizeof(actionSetInfo.localizedActionSetName),
                "%s",
                "Gameplay");
  XrResult res = xrCreateActionSet(instance_, &actionSetInfo, &actionSet_);
  if (res != XR_SUCCESS) {
    IGL_LOG_ERROR("Failed to create action set: %d\n", res);
    return;
  }

  XrPath leftHandPath = 0, rightHandPath = 0;
  xrStringToPath(instance_, "/user/hand/left", &leftHandPath);
  xrStringToPath(instance_, "/user/hand/right", &rightHandPath);

  // Button definitions
  struct ButtonDef {
    const char* name;
    const char* localizedName;
    XrPath* subactionPaths;
    Button id;
    XrActionType type;
    uint32_t subactionCount;
  };

  using igl::shell::Button;

  ButtonDef buttonDefs[] = {
      {.name = "left_trigger",
       .localizedName = "Left Trigger",
       .subactionPaths = &leftHandPath,
       .id = Button::LeftTrigger,
       .type = XR_ACTION_TYPE_FLOAT_INPUT,
       .subactionCount = 1},
      {.name = "right_trigger",
       .localizedName = "Right Trigger",
       .subactionPaths = &rightHandPath,
       .id = Button::RightTrigger,
       .type = XR_ACTION_TYPE_FLOAT_INPUT,
       .subactionCount = 1},
      {.name = "a_button",
       .localizedName = "A Button",
       .subactionPaths = &rightHandPath,
       .id = Button::A,
       .type = XR_ACTION_TYPE_BOOLEAN_INPUT,
       .subactionCount = 1},
      {.name = "b_button",
       .localizedName = "B Button",
       .subactionPaths = &rightHandPath,
       .id = Button::B,
       .type = XR_ACTION_TYPE_BOOLEAN_INPUT,
       .subactionCount = 1},
      {.name = "x_button",
       .localizedName = "X Button",
       .subactionPaths = &leftHandPath,
       .id = Button::X,
       .type = XR_ACTION_TYPE_BOOLEAN_INPUT,
       .subactionCount = 1},
      {.name = "y_button",
       .localizedName = "Y Button",
       .subactionPaths = &leftHandPath,
       .id = Button::Y,
       .type = XR_ACTION_TYPE_BOOLEAN_INPUT,
       .subactionCount = 1},
      {.name = "right_grip",
       .localizedName = "Right Grip",
       .subactionPaths = &rightHandPath,
       .id = Button::RightGrip,
       .type = XR_ACTION_TYPE_FLOAT_INPUT,
       .subactionCount = 1},
  };

  for (const auto& def : buttonDefs) {
    XrActionCreateInfo actionInfo = {XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = def.type;
    std::snprintf(actionInfo.actionName, sizeof(actionInfo.actionName), "%s", def.name);
    std::snprintf(actionInfo.localizedActionName,
                  sizeof(actionInfo.localizedActionName),
                  "%s",
                  def.localizedName);
    actionInfo.countSubactionPaths = def.subactionCount;
    actionInfo.subactionPaths = def.subactionPaths;
    res = xrCreateAction(actionSet_, &actionInfo, &buttonActions_[static_cast<int>(def.id)]);
    if (res != XR_SUCCESS) {
      IGL_LOG_ERROR("Failed to create action '%s': %d\n", def.name, res);
      return;
    }
  }

  // Right thumbstick (vector2f — separate from button actions)
  {
    XrActionCreateInfo actionInfo = {XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
    std::snprintf(actionInfo.actionName, sizeof(actionInfo.actionName), "%s", "right_thumbstick");
    std::snprintf(actionInfo.localizedActionName,
                  sizeof(actionInfo.localizedActionName),
                  "%s",
                  "Right Thumbstick");
    actionInfo.countSubactionPaths = 1;
    actionInfo.subactionPaths = &rightHandPath;
    res = xrCreateAction(actionSet_, &actionInfo, &rightThumbstickAction_);
    if (res != XR_SUCCESS) {
      IGL_LOG_ERROR("Failed to create right thumbstick action: %d\n", res);
      return;
    }
  }

  // Binding paths for Oculus Touch controllers
  struct BindingDef {
    Button id;
    const char* path;
  };

  BindingDef bindingDefs[] = {
      {.id = Button::LeftTrigger, .path = "/user/hand/left/input/trigger/value"},
      {.id = Button::RightTrigger, .path = "/user/hand/right/input/trigger/value"},
      {.id = Button::A, .path = "/user/hand/right/input/a/click"},
      {.id = Button::B, .path = "/user/hand/right/input/b/click"},
      {.id = Button::X, .path = "/user/hand/left/input/x/click"},
      {.id = Button::Y, .path = "/user/hand/left/input/y/click"},
      {.id = Button::RightGrip, .path = "/user/hand/right/input/squeeze/value"},
  };

  constexpr size_t kNumBindings = sizeof(bindingDefs) / sizeof(bindingDefs[0]);
  constexpr size_t kTotalBindings = kNumBindings + 1; // +1 for right thumbstick
  XrActionSuggestedBinding bindings[kTotalBindings];
  for (size_t i = 0; i < kNumBindings; ++i) {
    XrPath bindingPath = 0;
    xrStringToPath(instance_, bindingDefs[i].path, &bindingPath);
    bindings[i] = {.action = buttonActions_[static_cast<int>(bindingDefs[i].id)],
                   .binding = bindingPath};
  }

  // Right thumbstick binding
  {
    XrPath thumbstickPath = 0;
    xrStringToPath(instance_, "/user/hand/right/input/thumbstick", &thumbstickPath);
    bindings[kNumBindings] = {.action = rightThumbstickAction_, .binding = thumbstickPath};
  }

  XrPath profilePath = 0;
  xrStringToPath(instance_, "/interaction_profiles/oculus/touch_controller", &profilePath);

  XrInteractionProfileSuggestedBinding suggestedBindings = {
      .type = XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING,
      .interactionProfile = profilePath,
      .countSuggestedBindings = kTotalBindings,
      .suggestedBindings = bindings,
  };
  res = xrSuggestInteractionProfileBindings(instance_, &suggestedBindings);
  if (res != XR_SUCCESS) {
    IGL_LOG_ERROR("Failed to suggest interaction profile bindings: %d\n", res);
    return;
  }

  // Attach action set to session
  XrSessionActionSetsAttachInfo attachInfo = {
      .type = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO,
      .countActionSets = 1,
      .actionSets = &actionSet_,
  };
  res = xrAttachSessionActionSets(session_, &attachInfo);
  if (res != XR_SUCCESS) {
    IGL_LOG_ERROR("Failed to attach action sets: %d\n", res);
  }
}

void XrApp::handleXrEvents() {
  XrEventDataBuffer eventDataBuffer = {};

  // Poll for events
  for (;;) {
    auto* baseEventHeader = reinterpret_cast<XrEventDataBaseHeader*>(&eventDataBuffer);
    baseEventHeader->type = XR_TYPE_EVENT_DATA_BUFFER;
    baseEventHeader->next = nullptr;
    const XrResult res = xrPollEvent(instance_, &eventDataBuffer);
    XR_CHECK(res);
    if (res != XR_SUCCESS) {
      break;
    }

    switch (baseEventHeader->type) {
    case XR_TYPE_EVENT_DATA_EVENTS_LOST:
      IGL_LOG_INFO("xrPollEvent: received XR_TYPE_EVENT_DATA_EVENTS_LOST event\n");
      break;
    case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
      IGL_LOG_INFO("xrPollEvent: received XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING event\n");
      break;
    case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
      IGL_LOG_INFO("xrPollEvent: received XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED event\n");
      break;
    case XR_TYPE_EVENT_DATA_PERF_SETTINGS_EXT: {
      const XrEventDataPerfSettingsEXT* perfSettingsEvent =
          reinterpret_cast<XrEventDataPerfSettingsEXT*>(baseEventHeader);
      (void)perfSettingsEvent; // suppress unused warning
      IGL_LOG_INFO(
          "xrPollEvent: received XR_TYPE_EVENT_DATA_PERF_SETTINGS_EXT event: type %d subdomain %d "
          ": level %d -> level %d\n",
          perfSettingsEvent->type,
          perfSettingsEvent->subDomain,
          perfSettingsEvent->fromLevel,
          perfSettingsEvent->toLevel);
    } break;
    case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
      IGL_LOG_INFO(
          "xrPollEvent: received XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING event\n");
      break;
    case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
      const XrEventDataSessionStateChanged* sessionStateChangedEvent =
          reinterpret_cast<XrEventDataSessionStateChanged*>(baseEventHeader);
      IGL_LOG_INFO(
          "xrPollEvent: received XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: %d for session %p at "
          "time %lld\n",
          sessionStateChangedEvent->state,
          reinterpret_cast<const void*>(sessionStateChangedEvent->session),
          sessionStateChangedEvent->time);

      switch (sessionStateChangedEvent->state) {
      case XR_SESSION_STATE_READY:
      case XR_SESSION_STATE_STOPPING:
        handleSessionStateChanges(sessionStateChangedEvent->state);
        break;
      default:
        break;
      }
    } break;
    default:
      IGL_LOG_INFO("xrPollEvent: Unknown event\n");
      break;
    }
  }
}

void XrApp::handleActionView(const std::string& data) {
  if (platform_ != nullptr) {
    IntentEvent event;
    event.type = IntentType::ActionView;
    event.data = data;
    platform_->getInputDispatcher().queueEvent(event);
  }
}

void XrApp::handleSessionStateChanges(XrSessionState state) {
  if (state == XR_SESSION_STATE_READY) {
#if !defined(IGL_CMAKE_BUILD)
    assert(resumed_);
#endif // IGL_CMAKE_BUILD
    assert(!sessionActive_);

    const XrSessionBeginInfo sessionBeginInfo{
        .type = XR_TYPE_SESSION_BEGIN_INFO,
        .next = nullptr,
        .primaryViewConfigurationType = viewConfigProps_.viewConfigurationType,
    };

    const XrResult result = xrBeginSession(session_, &sessionBeginInfo);
    XR_CHECK(result);

    sessionActive_ = (result == XR_SUCCESS);
    IGL_LOG_INFO("XR session active\n");
  } else if (state == XR_SESSION_STATE_STOPPING) {
    assert(sessionActive_);
    XR_CHECK(xrEndSession(session_));
    sessionActive_ = false;
    IGL_LOG_INFO("XR session inactive\n");
  }
}

XrFrameState XrApp::beginFrame() {
  if (passthrough_) {
    const bool ptEnabled = passthroughEnabled();
    passthrough_->setEnabled(ptEnabled);
    if (ptEnabled) {
      passthrough_->setOpacity(renderSession_->appParams().passthroughOpacity);
    }
  }

  if (useQuadLayerComposition_) {
    updateQuadComposition();
  }

  const XrFrameWaitInfo waitFrameInfo = {XR_TYPE_FRAME_WAIT_INFO};

  XrFrameState frameState = {XR_TYPE_FRAME_STATE};

  XR_CHECK(xrWaitFrame(session_, &waitFrameInfo, &frameState));

  const XrFrameBeginInfo beginFrameInfo = {XR_TYPE_FRAME_BEGIN_INFO};

  XR_CHECK(xrBeginFrame(session_, &beginFrameInfo));

  XrSpaceLocation loc = {
      .type = XR_TYPE_SPACE_LOCATION,
  };
  XR_CHECK(xrLocateSpace(headSpace_, currentSpace_, frameState.predictedDisplayTime, &loc));
  const XrPosef headPose = loc.pose;

  XrViewState viewState = {XR_TYPE_VIEW_STATE};

  const XrViewLocateInfo projectionInfo = {
      .type = XR_TYPE_VIEW_LOCATE_INFO,
      .next = nullptr,
      .viewConfigurationType = viewConfigProps_.viewConfigurationType,
      .displayTime = frameState.predictedDisplayTime,
      .space = headSpace_,
  };

  uint32_t numViews = views_.size();

  XR_CHECK(xrLocateViews(
      session_, &projectionInfo, &viewState, views_.size(), &numViews, views_.data()));

  for (size_t i = 0; i < XrComposition::kNumViews; i++) {
    const XrPosef eyePose = views_[i].pose;
    XrPosef_Multiply(&viewStagePoses_[i], &headPose, &eyePose);
    XrPosef viewTransformXrPosef{};
    XrPosef_Invert(&viewTransformXrPosef, &viewStagePoses_[i]);
    XrMatrix4x4f xrMat4{};
    XrMatrix4x4f_CreateFromRigidTransform(&xrMat4, &viewTransformXrPosef);
    viewTransforms_[i] = glm::make_mat4(xrMat4.m);
    cameraPositions_[i] = glm::vec3(eyePose.position.x, eyePose.position.y, eyePose.position.z);
  }

  if (hands_) {
    hands_->updateTracking(currentSpace_, shellParams_->handTracking);
  }

  return frameState;
}

void XrApp::render() {
  if (passthrough_) {
    if (passthroughEnabled()) {
      shellParams_->clearColorValue = Color{0.0f, 0.0f, 0.0f, 0.0f};
    } else {
      shellParams_->clearColorValue.reset();
    }
  }
#if USE_FORCE_ZERO_CLEAR
  else {
    shellParams_->clearColorValue = Color{0.0f, 0.0f, 0.0f, 0.0f};
  }
#endif

  for (size_t layerIndex = 0; layerIndex < compositionLayers_.size(); ++layerIndex) {
    if (!compositionLayers_[layerIndex]->isValid()) {
      continue;
    }

    for (uint32_t i = 0; i < compositionLayers_[layerIndex]->renderPassesCount(); ++i) {
      auto surfaceTextures = compositionLayers_[layerIndex]->beginRendering(
          i, views_, viewTransforms_, cameraPositions_, shellParams_->viewParams);

      renderSession_->setCurrentQuadLayer(useQuadLayerComposition_ ? layerIndex : 0);
      renderSession_->update(std::move(surfaceTextures));

      compositionLayers_[layerIndex]->endRendering(i);
    }
  }
}

void XrApp::endFrame(XrFrameState frameState) {
  XrCompositionLayerFlags compositionFlags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT;
  if (passthroughEnabled()) {
    compositionFlags |= XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  }

  std::vector<const XrCompositionLayerBaseHeader*> layers;
  layers.reserve(1 + compositionLayers_.size() * (useQuadLayerComposition_ ? 2 : 1));

  if (passthroughEnabled()) {
    passthrough_->injectLayer(layers);
  }

  const auto& appParams = renderSession_->appParams();
  for (const auto& layer : compositionLayers_) {
    if (layer->isValid()) {
      layer->doComposition(
          appParams.depthParams, views_, viewStagePoses_, currentSpace_, compositionFlags, layers);
    }
  }

  const XrFrameEndInfo endFrameInfo{
      .type = XR_TYPE_FRAME_END_INFO,
      .next = nullptr,
      .displayTime = frameState.predictedDisplayTime,
      .environmentBlendMode = additiveBlendingSupported_ ? XR_ENVIRONMENT_BLEND_MODE_ADDITIVE
                                                         : XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
      .layerCount = static_cast<uint32_t>(layers.size()),
      .layers = layers.data(),
  };
  XR_CHECK(xrEndFrame(session_, &endFrameInfo));
}

void XrApp::update() {
  if (!initialized_ || !resumed_ || !sessionActive_) {
    return;
  }

  // Sync and poll controller input
  if (actionSet_ != XR_NULL_HANDLE && platform_ != nullptr) {
    const XrActiveActionSet activeActionSet = {
        .actionSet = actionSet_,
        .subactionPath = XR_NULL_PATH,
    };
    XrActionsSyncInfo syncInfo = {
        .type = XR_TYPE_ACTIONS_SYNC_INFO,
        .countActiveActionSets = 1,
        .activeActionSets = &activeActionSet,
    };
    if (xrSyncActions(session_, &syncInfo) == XR_SUCCESS) {
      pollActions();
    }
  }

  if (platform_ != nullptr) {
    platform_->getInputDispatcher().processEvents();
  }

  auto frameState = beginFrame();
  render();
  endFrame(frameState);
}

void XrApp::pollActions() {
  constexpr int numButtons = static_cast<int>(Button::Count);
  for (int i = 0; i < numButtons; ++i) {
    if (buttonActions_[i] == XR_NULL_HANDLE) {
      continue;
    }
    auto buttonId = static_cast<Button>(i);
    RayEvent rayEvent;
    rayEvent.button = buttonId;
    XrActionStateGetInfo getInfo = {
        .type = XR_TYPE_ACTION_STATE_GET_INFO,
        .action = buttonActions_[i],
    };

    if (buttonId == Button::LeftTrigger || buttonId == Button::RightTrigger ||
        buttonId == Button::RightGrip) {
      XrActionStateFloat state = {.type = XR_TYPE_ACTION_STATE_FLOAT};
      if (xrGetActionStateFloat(session_, &getInfo, &state) == XR_SUCCESS && state.isActive) {
        rayEvent.buttonPressed = state.currentState > 0.5f;
        platform_->getInputDispatcher().queueEvent(rayEvent);
      }
    } else {
      XrActionStateBoolean state = {.type = XR_TYPE_ACTION_STATE_BOOLEAN};
      if (xrGetActionStateBoolean(session_, &getInfo, &state) == XR_SUCCESS && state.isActive) {
        rayEvent.buttonPressed = (state.currentState != 0u);
        platform_->getInputDispatcher().queueEvent(rayEvent);
      }
    }
  }

  // Right thumbstick → virtual left/right buttons
  if (rightThumbstickAction_ != XR_NULL_HANDLE) {
    XrActionStateGetInfo getInfo = {
        .type = XR_TYPE_ACTION_STATE_GET_INFO,
        .action = rightThumbstickAction_,
    };
    XrActionStateVector2f state = {.type = XR_TYPE_ACTION_STATE_VECTOR2F};
    if (xrGetActionStateVector2f(session_, &getInfo, &state) == XR_SUCCESS && state.isActive) {
      RayEvent leftEvent;
      leftEvent.button = Button::RightThumbstickLeft;
      leftEvent.buttonPressed = state.currentState.x < -0.5f;
      platform_->getInputDispatcher().queueEvent(leftEvent);

      RayEvent rightEvent;
      rightEvent.button = Button::RightThumbstickRight;
      rightEvent.buttonPressed = state.currentState.x > 0.5f;
      platform_->getInputDispatcher().queueEvent(rightEvent);

      RayEvent upEvent;
      upEvent.button = Button::RightThumbstickUp;
      upEvent.buttonPressed = state.currentState.y > 0.5f;
      platform_->getInputDispatcher().queueEvent(upEvent);

      RayEvent downEvent;
      downEvent.button = Button::RightThumbstickDown;
      downEvent.buttonPressed = state.currentState.y < -0.5f;
      platform_->getInputDispatcher().queueEvent(downEvent);
    }
  }
}

bool XrApp::passthroughSupported() const noexcept { // NOLINT(bugprone-exception-escape)
  return supportedOptionalXrExtensions_.count(XR_FB_PASSTHROUGH_EXTENSION_NAME) != 0;
}

bool XrApp::passthroughEnabled() const noexcept { // NOLINT(bugprone-exception-escape)
  if (!renderSession_ || !passthrough_) {
    return false;
  }
  const auto& appParams = renderSession_->appParams();
  return appParams.passthroughGetter ? appParams.passthroughGetter() : useQuadLayerComposition_;
}

bool XrApp::handTrackingSupported() const noexcept { // NOLINT(bugprone-exception-escape)
#if IGL_PLATFORM_ANDROID
  return supportedOptionalXrExtensions_.count(XR_EXT_HAND_TRACKING_EXTENSION_NAME) != 0 &&
         handTrackingSystemProps_.supportsHandTracking != 0u;
#else
  return false;
#endif // IGL_PLATFORM_ANDROID
}

bool XrApp::handTrackingMeshSupported() const noexcept { // NOLINT(bugprone-exception-escape)
#if IGL_PLATFORM_ANDROID
  return supportedOptionalXrExtensions_.count(XR_FB_HAND_TRACKING_MESH_EXTENSION_NAME) != 0;
#else
  return false;
#endif // IGL_PLATFORM_ANDROID
}

bool XrApp::refreshRateExtensionSupported() const noexcept { // NOLINT(bugprone-exception-escape)
  return supportedOptionalXrExtensions_.count(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME) != 0;
}

bool XrApp::instanceCreateInfoAndroidSupported()
    const noexcept { // NOLINT(bugprone-exception-escape)
#if IGL_PLATFORM_ANDROID
  return supportedOptionalXrExtensions_.count(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME) != 0;
#else
  return false;
#endif // IGL_PLATFORM_ANDROID
}

bool XrApp::alphaBlendCompositionSupported() const noexcept { // NOLINT(bugprone-exception-escape)
#ifdef XR_FB_composition_layer_alpha_blend
  return supportedOptionalXrExtensions_.count(XR_FB_COMPOSITION_LAYER_ALPHA_BLEND_EXTENSION_NAME) !=
         0;
#else
  return false;
#endif // XR_FB_composition_layer_alpha_blend
}

} // namespace igl::shell::openxr
