/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#import "ViewController.h"

#import "GLView.h" // IWYU pragma: keep
#import "HeadlessView.h"
#import "MetalView.h"
#import "WebGPUView.h"
// @fb-only

#import <AppKit/NSApplication.h>
#import <AppKit/NSEvent.h>
#import <AppKit/NSOpenGL.h>
#import <AppKit/NSOpenGLView.h>
#import <AppKit/NSView.h>
#import <shell/shared/input/InputDispatcher.h>
#include <shell/shared/platform/Platform.h>
#import <igl/Common.h>
#import <igl/IGL.h>
#if IGL_BACKEND_METAL
#include <igl/metal/ColorSpace.h>
#include <igl/metal/HWDevice.h>
#include <igl/metal/Texture.h>
#include <igl/metal/macos/Device.h>
#endif
#if IGL_BACKEND_OPENGL
#include <igl/opengl/macos/Context.h>
#include <igl/opengl/macos/HWDevice.h>
#include <igl/opengl/macos/PlatformDevice.h>
#endif
#import "DisplayLinkRatePacing.h"

#include <shell/shared/platform/PresentationRateController.h>
#include <shell/shared/platform/apple/PresentationRateApple.h>
#include <shell/shared/platform/mac/PlatformMac.h>
#include <shell/shared/renderSession/AppParams.h>
#include <shell/shared/renderSession/RenderSession.h>
#include <shell/shared/renderSession/ShellParams.h>
// @fb-only
// @fb-only
// @fb-only
// @fb-only
// @fb-only
#if IGL_BACKEND_WEBGPU
#include <igl/webgpu/HWDevice.h>
#include <igl/webgpu/Surface.h>
#endif
#if IGL_BACKEND_VULKAN
#import "VulkanView.h"

#include <igl/vulkan/Device.h>
#include <igl/vulkan/HWDevice.h>
#include <igl/vulkan/VulkanContext.h>
#endif
#import <cmath>
#import <simd/simd.h>
#include <thread>

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace igl;

@interface ViewController () {
  igl::shell::IRenderSessionFactory* _factory;
  igl::shell::RenderSessionConfig _config;
  igl::shell::ShellParams _shellParams;
  CGRect _frame;
  CVDisplayLinkRef _displayLink; // For OpenGL (via GLView) and opt-in Metal timer rendering
  std::unique_ptr<igl::shell::DisplayLinkRatePacing> _displayLinkRatePacing;
  id<CAMetalDrawable> _currentDrawable;
  id<MTLTexture> _depthStencilTexture;
  std::shared_ptr<igl::shell::Platform> _shellPlatform;
  std::unique_ptr<igl::shell::RenderSession> _session;
  glm::ivec2 _lastSurfaceDimensions;
  float _kMouseSpeed;
  // Offscreen textures for headless rendering (bypass drawable/vsync)
  std::shared_ptr<igl::ITexture> _offscreenColor;
  std::shared_ptr<igl::ITexture> _offscreenDepth;
  // Headless render thread
  std::thread _headlessThread;
  BOOL _headlessRunning;
#if IGL_BACKEND_WEBGPU
  std::unique_ptr<igl::webgpu::Surface> _webgpuSurface;
  std::shared_ptr<igl::ITexture> _webgpuDepth;
#endif
}
@end

@implementation ViewController

@synthesize iglView = _iglView;

///--------------------------------------
/// MARK: - Init
///--------------------------------------

- (instancetype)initWithFrame:(CGRect)frame
                      factory:(igl::shell::IRenderSessionFactory&)factory
                       config:(igl::shell::RenderSessionConfig)config {
  self = [super initWithNibName:nil bundle:nil];
  if (!self) {
    return self;
  }

  self->_config = std::move(config);
  self->_factory = &factory;
  self->_shellParams = igl::shell::ShellParams();
  self->_shellParams.viewportSize.x = frame.size.width;
  self->_shellParams.viewportSize.y = frame.size.height;
  self->_frame = frame;
  self->_kMouseSpeed = 0.05f;
  self->_currentDrawable = nil;
  self->_depthStencilTexture = nil;

  return self;
}

- (void)dealloc {
  // dealloc may run on the headless thread itself (its lambda retains self), where join() throws.
  if (_headlessThread.joinable()) {
    _headlessThread.detach();
  }
}

- (void)initModule {
}

- (void)teardown {
  [self stopHeadlessRendering];
  if (_session) {
    _session->teardown();
  }
  _session = nullptr;
#if IGL_BACKEND_WEBGPU
  _webgpuDepth = nullptr;
  _webgpuSurface = nullptr;
  if ([self.view isKindOfClass:[WebGPUView class]]) {
    // The view's rate pacing points at the platform released below.
    [static_cast<WebGPUView*>(self.view) detachRatePacing];
  }
#endif
  _shellPlatform = nullptr;
}

- (void)render {
  // In headless mode only the render thread renders; AppKit still calls this on the main thread.
  if (_shellParams.isHeadless && [NSThread isMainThread]) {
    return;
  }
  if (_session == nullptr) {
    return;
  }

  if (!_shellParams.isHeadless) {
    // Only access AppKit view properties from the main thread
    const NSRect contentRect = self.view.frame;
    _shellParams.viewportSize = glm::vec2(contentRect.size.width, contentRect.size.height);
    _shellParams.viewportScale = self.view.window.backingScaleFactor;
  }
  _session->setShellParams(_shellParams);
  // process user input
  _shellPlatform->getInputDispatcher().processEvents();

  igl::SurfaceTextures surfaceTextures;
  if (_config.backendVersion.flavor != igl::BackendFlavor::Invalid &&
      _shellPlatform->getDevicePtr() != nullptr) {
    if (_shellParams.isHeadless) {
      // Headless mode: create offscreen textures to bypass drawable/vsync blocking.
      // This allows the render loop to run at unrestricted FPS.
      auto& device = _shellPlatform->getDevice();
      const uint32_t w = static_cast<uint32_t>(_shellParams.viewportSize.x);
      const uint32_t h = static_cast<uint32_t>(_shellParams.viewportSize.y);

      if (!_offscreenColor || _offscreenColor->getSize().width != w ||
          _offscreenColor->getSize().height != h) {
        _offscreenColor = device.createTexture(
            igl::TextureDesc::new2D(igl::TextureFormat::BGRA_SRGB,
                                    w,
                                    h,
                                    igl::TextureDesc::TextureUsageBits::Attachment |
                                        igl::TextureDesc::TextureUsageBits::Sampled),
            nullptr);
        _offscreenDepth = device.createTexture(
            igl::TextureDesc::new2D(igl::TextureFormat::Z_UNorm24,
                                    w,
                                    h,
                                    igl::TextureDesc::TextureUsageBits::Attachment),
            nullptr);
      }

      surfaceTextures.color = _offscreenColor;
      surfaceTextures.depth = _offscreenDepth;
    } else {
// @fb-only
      // @fb-only
      // @fb-only
        // @fb-only
            // @fb-only
        // @fb-only
                                 // @fb-only
      // @fb-only
// @fb-only

      // surface textures
      surfaceTextures = igl::SurfaceTextures{.color = [self createTextureFromNativeDrawable],
                                             .depth = [self createTextureFromNativeDepth]};
    }
    if (surfaceTextures.color == nullptr) {
      // No drawable this frame (e.g. a zero-sized or unconfigured surface); skip it.
      return;
    }
    IGL_DEBUG_ASSERT(surfaceTextures.depth != nullptr);
    const auto& dims = surfaceTextures.color->getDimensions();
    _shellParams.nativeSurfaceDimensions = glm::ivec2{dims.width, dims.height};
    // After a resize, sessions that built their framebuffer once would keep attachments of the old
    // size, so the framebuffer is released as on iOS (drawableSizeWillChange).
    if (_lastSurfaceDimensions != _shellParams.nativeSurfaceDimensions) {
      if (_lastSurfaceDimensions != glm::ivec2(0)) {
        _session->releaseFramebuffer();
      }
      _lastSurfaceDimensions = _shellParams.nativeSurfaceDimensions;
    }

    // update retina scale
    float pixelsPerPoint = _shellParams.nativeSurfaceDimensions.x / _shellParams.viewportSize.x;
    _session->setPixelsPerPoint(pixelsPerPoint);
// @fb-only
    // @fb-only
    // @fb-only
      // @fb-only
          // @fb-only
      // @fb-only
    // @fb-only
// @fb-only
  }

  // Use runUpdate() which automatically handles benchmark timing, reporting, and expiration
  _session->runUpdate(std::move(surfaceTextures));

  if (_session->appParams().exitRequested) {
    dispatch_async(dispatch_get_main_queue(), ^{
      [[NSApplication sharedApplication] terminate:nil];
    });
  }
}

- (void)loadView {
  // We don't care about the device type, just
  // return something that works
  HWDeviceQueryDesc queryDesc(HWDeviceType::Unknown);

  switch (_config.backendVersion.flavor) {
  case igl::BackendFlavor::Invalid: {
    auto headlessView = [[HeadlessView alloc] initWithFrame:_frame];
    self.view = headlessView;

    // @fb-only
        // @fb-only

    // Headless platform does not run on a real device
    _shellPlatform = std::make_shared<igl::shell::PlatformMac>(nullptr);

    [headlessView prepareHeadless];
    break;
  }

#if IGL_BACKEND_METAL
  case igl::BackendFlavor::Metal: {
    auto hwDevices = metal::HWDevice().queryDevices(queryDesc, nullptr);
    auto device = metal::HWDevice().create(hwDevices[0], nullptr);

    auto d = static_cast<igl::metal::macos::Device*>(device.get())->get();

    auto metalView = [[MetalView alloc] initWithFrame:_frame device:d];
    metalView.depthStencilPixelFormat = MTLPixelFormatDepth32Float_Stencil8;

    metalView.delegate = self;

    metalView.colorPixelFormat =
        metal::Texture::textureFormatToMTLPixelFormat(_config.swapchainColorTextureFormat);
    metalView.colorspace = metal::colorSpaceToCGColorSpace(_config.swapchainColorSpace);

    metalView.framebufferOnly = NO;
    [metalView setViewController:self];
    self.view = metalView;
    _shellPlatform = std::make_shared<igl::shell::PlatformMac>(std::move(device));
    break;
  }
#endif

#if IGL_BACKEND_OPENGL
  // NOLINTBEGIN(clang-diagnostic-deprecated-declarations)
  case igl::BackendFlavor::OpenGL: {
    const bool enableStencilBuffer =
        _config.depthTextureFormat == igl::TextureFormat::S8_UInt_Z24_UNorm ||
        _config.depthTextureFormat == igl::TextureFormat::S_UInt8;
    const NSOpenGLPixelFormatAttribute stencilSize = enableStencilBuffer ? 8 : 0;

    NSOpenGLPixelFormat* pixelFormat;
    if (_config.backendVersion.majorVersion == 4 && _config.backendVersion.minorVersion == 1) {
      static NSOpenGLPixelFormatAttribute attributes[] = {
          NSOpenGLPFADoubleBuffer,
          NSOpenGLPFAAllowOfflineRenderers,
          NSOpenGLPFAMultisample,
          1,
          NSOpenGLPFASampleBuffers,
          1,
          NSOpenGLPFASamples,
          4,
          NSOpenGLPFAColorSize,
          32,
          NSOpenGLPFADepthSize,
          24,
          NSOpenGLPFAStencilSize,
          stencilSize,
          NSOpenGLPFAOpenGLProfile,
          NSOpenGLProfileVersion4_1Core,
          0,
      };
      pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
      IGL_DEBUG_ASSERT(pixelFormat, "Requested attributes not supported");
    } else if (_config.backendVersion.majorVersion == 3 &&
               _config.backendVersion.minorVersion == 2) {
      static NSOpenGLPixelFormatAttribute attributes[] = {
          NSOpenGLPFADoubleBuffer,
          NSOpenGLPFAAllowOfflineRenderers,
          NSOpenGLPFAMultisample,
          1,
          NSOpenGLPFASampleBuffers,
          1,
          NSOpenGLPFASamples,
          4,
          NSOpenGLPFAColorSize,
          32,
          NSOpenGLPFADepthSize,
          24,
          NSOpenGLPFAStencilSize,
          stencilSize,
          NSOpenGLPFAOpenGLProfile,
          NSOpenGLProfileVersion3_2Core,
          0,
      };
      pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
    } else if (_config.backendVersion.majorVersion == 2 &&
               _config.backendVersion.minorVersion == 1) {
      static NSOpenGLPixelFormatAttribute attributes[] = {
          NSOpenGLPFADoubleBuffer,
          NSOpenGLPFAAllowOfflineRenderers,
          NSOpenGLPFAMultisample,
          1,
          NSOpenGLPFASampleBuffers,
          1,
          NSOpenGLPFASamples,
          4,
          NSOpenGLPFAColorSize,
          32,
          NSOpenGLPFADepthSize,
          24,
          NSOpenGLPFAStencilSize,
          stencilSize,
          NSOpenGLPFAOpenGLProfile,
          NSOpenGLProfileVersionLegacy,
          0,
      };
      pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
    } else {
      IGL_DEBUG_ABORT("Unsupported OpenGL version: %u.%u\n",
                      _config.backendVersion.majorVersion,
                      _config.backendVersion.minorVersion);
    }
    auto openGLView = [[GLView alloc] initWithFrame:_frame pixelFormat:pixelFormat];
    igl::Result result;
    auto context = igl::opengl::macos::Context::createContext(openGLView.openGLContext, &result);
    IGL_DEBUG_ASSERT(result.isOk());
    _shellPlatform = std::make_shared<igl::shell::PlatformMac>(
        opengl::macos::HWDevice().createWithContext(std::move(context), nullptr));

    auto& device = _shellPlatform->getDevice();
    auto* platformDevice = device.getPlatformDevice<igl::opengl::macos::PlatformDevice>();
    platformDevice->setNativeDrawableTextureFormat(_config.swapchainColorTextureFormat, nullptr);
    self.view = openGLView;
    break;
  }
  // NOLINTEND(clang-diagnostic-deprecated-declarations)
#endif

#if IGL_BACKEND_VULKAN
  case igl::BackendFlavor::Vulkan: {
    auto vulkanView = [[VulkanView alloc] initWithFrame:_frame];

    self.view = vulkanView;

    // vulkanView.wantsLayer =
    // YES; // Back the view with a layer created by the makeBackingLayer method.

    igl::vulkan::VulkanContextConfig vulkanContextConfig;
    vulkanContextConfig.terminateOnValidationError = true;

    vulkanContextConfig.swapChainColorSpace = _config.swapchainColorSpace;
    vulkanContextConfig.requestedSwapChainTextureFormat = _config.swapchainColorTextureFormat;

    auto context =
        igl::vulkan::HWDevice::createContext(vulkanContextConfig, (__bridge void*)vulkanView);
    if (!context) {
      IGLLog(IGLLogError, "Vulkan 1.1 or newer is required.\n");
      return;
    }
    auto devices = igl::vulkan::HWDevice::queryDevices(
        *context, igl::HWDeviceQueryDesc(igl::HWDeviceType::DiscreteGpu), nullptr);
    if (devices.empty()) {
      devices = igl::vulkan::HWDevice::queryDevices(
          *context, igl::HWDeviceQueryDesc(igl::HWDeviceType::IntegratedGpu), nullptr);
    }
    if (devices.empty()) {
      devices = igl::vulkan::HWDevice::queryDevices(
          *context, igl::HWDeviceQueryDesc(igl::HWDeviceType::SoftwareGpu), nullptr);
    }
    if (devices.empty()) {
      IGLLog(IGLLogError, "No suitable Vulkan device.\n");
      return;
    }
    auto device = igl::vulkan::HWDevice::create(
        std::move(context), devices[0], 0, 0, 0, nullptr, nullptr, "IGL Shell", nullptr);
    if (!device) {
      IGLLog(IGLLogError, "Failed to create a Vulkan device.\n");
      return;
    }

    _shellPlatform = std::make_shared<igl::shell::PlatformMac>(std::move(device));
    [vulkanView prepareVulkan:_shellPlatform.get()];
    break;
  }
#endif

#if IGL_BACKEND_WEBGPU
  case igl::BackendFlavor::WebGPU: {
    igl::Result result;
    auto context = igl::webgpu::HWDevice::createContext({}, &result);
    IGL_DEBUG_ASSERT(result.isOk(), "%s", result.message.c_str());
    auto devices = igl::webgpu::HWDevice::queryDevices(*context, queryDesc, &result);
    IGL_DEBUG_ASSERT(!devices.empty(), "%s", result.message.c_str());
    auto device = igl::webgpu::HWDevice::create(std::move(context), devices[0], &result);
    IGL_DEBUG_ASSERT(result.isOk(), "%s", result.message.c_str());

    auto webgpuView = [[WebGPUView alloc] initWithFrame:_frame];
    self.view = webgpuView;
    webgpuView.wantsLayer = YES;
    _webgpuSurface = igl::webgpu::Surface::createFromMetalLayer(
        *device, (__bridge void*)webgpuView.layer, &result);
    IGL_DEBUG_ASSERT(result.isOk(), "%s", result.message.c_str());
    _shellPlatform = std::make_shared<igl::shell::PlatformMac>(std::move(device));
    [webgpuView prepareWebGPU:_shellPlatform.get() controller:self];
    break;
  }
#endif

// @fb-only
  // @fb-only
    // @fb-only

    // @fb-only
        // @fb-only

    // Set panel apps to full screen in the reprojection buffer. @fb-only
    // @fb-only
    // @fb-only
    // @fb-only

    // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only

    // @fb-only
    // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
            // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
    // @fb-only

    // @fb-only

    // @fb-only

    // @fb-only
    // @fb-only
  // @fb-only
// @fb-only

  default: {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    break;
  }
  }

  // Get initial native surface dimensions
  _shellParams.nativeSurfaceDimensions = glm::ivec2(2048, 1536);
  auto args =
      shell::convertArgvToParams(igl::shell::Platform::argc(), igl::shell::Platform::argv());
  shell::parseShellParams(args, _shellParams);

  // When timer rendering is opted in, configure the Metal view for external
  // CVDisplayLink-driven rendering instead of MTKView's internal display link
  // (which may not fire for non-frontmost apps on macOS).
  if (_shellParams.useTimerRendering && [self.view isKindOfClass:[MetalView class]]) {
    MetalView* metalView = (MetalView*)self.view;
    metalView.enableSetNeedsDisplay = YES;
    metalView.paused = YES;
  }

  if (_shellParams.isHeadless) {
    // The headless render thread creates, runs and destroys the session.
    return;
  }
  _session = _factory->createRenderSession(_shellPlatform);
  IGL_DEBUG_ASSERT(_session, "createDefaultRenderSession() must return a valid session");
  _session->setShellParams(_shellParams);
  _session->initialize();
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static CVReturn metalDisplayLinkCallback(CVDisplayLinkRef /*displayLink*/,
                                         const CVTimeStamp* /*now*/,
                                         const CVTimeStamp* /*outputTime*/,
                                         CVOptionFlags /*flagsIn*/,
                                         CVOptionFlags* /*flagsOut*/,
                                         void* userdata) {
  auto controller = (__bridge ViewController*)userdata;
  // A CVDisplayLink cannot be slowed down, so a capped rate is reached by returning early
  // from the callbacks in between. Called exactly once per callback: it advances the counter.
  if (controller->_displayLinkRatePacing && !controller->_displayLinkRatePacing->shouldRender()) {
    return kCVReturnSuccess;
  }
  [controller performSelectorOnMainThread:@selector(triggerMetalRender)
                               withObject:nil
                            waitUntilDone:NO];
  return kCVReturnSuccess;
}

- (void)triggerMetalRender {
  [self.view setNeedsDisplay:YES];
}

- (void)viewDidAppear {
  if (!_shellPlatform) {
    return;
  }
  if (_shellParams.isHeadless) {
    // Headless mode: pause the MTKView display link and use a dedicated
    // render thread with offscreen textures for unrestricted FPS.
    if ([self.view isKindOfClass:[MetalView class]]) {
      MetalView* v = (MetalView*)self.view;
      v.paused = YES;
    }
    [self stopHeadlessRendering];
    if (_shellPlatform != nullptr) {
      _headlessRunning = YES;
      _headlessThread = std::thread([self]() {
        pthread_setname_np("IGL Headless Render");
        [self makeDeviceCurrentOnThisThread];
        @autoreleasepool {
          self->_session = self->_factory->createRenderSession(self->_shellPlatform);
          IGL_DEBUG_ASSERT(self->_session,
                           "createDefaultRenderSession() must return a valid session");
          if (self->_session == nullptr) {
            IGLLog(IGLLogError, "Headless render thread could not create a render session\n");
            return;
          }
          self->_session->setShellParams(self->_shellParams);
          self->_session->initialize();
        }
        while (self->_headlessRunning) {
          @autoreleasepool {
            [self render];
          }
        }
        @autoreleasepool {
          self->_session->teardown();
          self->_session = nullptr;
        }
      });
    }
  } else if ([self.view isKindOfClass:[MetalView class]]) {
    if (_shellParams.useTimerRendering) {
      // Use a CVDisplayLink to drive Metal rendering, synced to the display refresh
      // rate. MTKView's internal CVDisplayLink may not fire for non-frontmost apps
      // (e.g., when launched from automated tools for screenshot capture).
      CVDisplayLinkCreateWithActiveCGDisplays(&_displayLink);
      CVDisplayLinkSetOutputCallback(_displayLink, &metalDisplayLinkCallback, (__bridge void*)self);
      if (_shellPlatform) {
        // Publish before CVDisplayLinkStart: the callback reads this handle on the link thread,
        // so assigning it after the link is firing races the write (the null check does not
        // cure a non-atomic read/write). Teardown keeps the mirror order via detach()-then-reset.
        _displayLinkRatePacing =
            std::make_unique<igl::shell::DisplayLinkRatePacing>(*_shellPlatform, _displayLink);
      }
      CVDisplayLinkStart(_displayLink);
    } else {
      MetalView* v = (MetalView*)self.view;
      v.paused = NO;
      if (_shellPlatform) {
        _shellPlatform->getPresentationRateController().setBackend(
            igl::shell::createMTKViewPresentationRateBackend(v));
      }
    }
  } else if ([self.view isKindOfClass:[GLView class]]) {
    GLView* v = (GLView*)self.view;
    [v startTimer];
  } else if ([self.view isKindOfClass:[WebGPUView class]]) {
    [static_cast<WebGPUView*>(self.view) startTimer];
  }
  [self.view.window makeFirstResponder:self];
}

- (void)stopHeadlessRendering {
  _headlessRunning = NO;
  if (_headlessThread.joinable()) {
    _headlessThread.join();
    [self makeDeviceCurrentOnThisThread];
  }
}

/// GL contexts are current per thread, and Vulkan checks the owning thread.
- (void)makeDeviceCurrentOnThisThread {
  if (_shellPlatform == nullptr) {
    return;
  }
  if (const auto device = _shellPlatform->getDevicePtr()) {
    device->setCurrentThread();
  }
}

- (void)viewWillDisappear {
  if (_shellParams.isHeadless) {
    [self stopHeadlessRendering];
    return;
  }
  if ([self.view isKindOfClass:[MetalView class]]) {
    if (_shellParams.useTimerRendering) {
      if (_displayLinkRatePacing) {
        // Detach before the reset, not through it. reset() stores null before it runs the
        // destructor, so the callback above would be reading _displayLinkRatePacing while
        // this thread writes it, with the link still firing. detach() leaves the handle
        // alone and stops the link, and CVDisplayLinkStop() does not return while a
        // callback is running — so once it has, nothing else reads the handle or the link.
        _displayLinkRatePacing->detach();
      }
      _displayLinkRatePacing.reset();
      CVDisplayLinkStop(_displayLink);
      CVDisplayLinkRelease(_displayLink);
      _displayLink = nullptr;
    } else {
      MetalView* v = (MetalView*)self.view;
      v.paused = YES;
      if (_shellPlatform) {
        // Whatever rate was in effect belonged to the link this view just stopped.
        _shellPlatform->getPresentationRateController().setBackend(nullptr);
      }
    }
  } else if ([self.view isKindOfClass:[GLView class]]) {
    GLView* v = (GLView*)self.view;
    [v stopTimer];
  } else if ([self.view isKindOfClass:[WebGPUView class]]) {
    [static_cast<WebGPUView*>(self.view) stopTimer];
  }
}

- (void)drawInMTKView:(nonnull MTKView*)view {
  @autoreleasepool {
    _currentDrawable = view.currentDrawable;
    _depthStencilTexture = view.depthStencilTexture;
    [self render];
    _currentDrawable = nil;
  }
}

- (void)mtkView:(nonnull MTKView*)view drawableSizeWillChange:(CGSize)size {
  _frame.size = size;
}

- (void)mtkView:(nonnull MTKView*)view drawableSizeDidChange:(CGSize)size {
}

- (std::shared_ptr<igl::ITexture>)createTextureFromNativeDrawable {
  switch (_config.backendVersion.flavor) {
#if IGL_BACKEND_METAL
  case igl::BackendFlavor::Metal: {
    auto& device = _shellPlatform->getDevice();
    auto* platformDevice = device.getPlatformDevice<igl::metal::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice);
    IGL_DEBUG_ASSERT(_currentDrawable != nil);
    auto texture = platformDevice->createTextureFromNativeDrawable(_currentDrawable, nullptr);
    return texture;
  }
#endif

#if IGL_BACKEND_OPENGL
  case igl::BackendFlavor::OpenGL: {
    auto& device = _shellPlatform->getDevice();
    auto* platformDevice = device.getPlatformDevice<igl::opengl::macos::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice);
    auto texture = platformDevice->createTextureFromNativeDrawable(nullptr);
    return texture;
  }
#endif

#if IGL_BACKEND_VULKAN
  case igl::BackendFlavor::Vulkan: {
    auto& device = _shellPlatform->getDevice();
    auto* platformDevice = device.getPlatformDevice<igl::vulkan::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice);
    auto texture = platformDevice->createTextureFromNativeDrawable(nullptr);
    return texture;
  }
#endif

#if IGL_BACKEND_WEBGPU
  case igl::BackendFlavor::WebGPU: {
    const CGSize size = [static_cast<WebGPUView*>(self.view) drawableSize];
    const auto width = static_cast<uint32_t>(size.width);
    const auto height = static_cast<uint32_t>(size.height);
    if (width == 0 || height == 0) {
      return nullptr;
    }
    if (_webgpuSurface->getWidth() != width || _webgpuSurface->getHeight() != height) {
      const igl::Result result =
          _webgpuSurface->configure(width, height, _config.swapchainColorTextureFormat);
      if (!result.isOk()) {
        IGL_LOG_ERROR("WebGPU surface configuration failed: %s\n", result.message.c_str());
        return nullptr;
      }
    }
    return _webgpuSurface->getCurrentTexture(nullptr);
  }
#endif

// @fb-only
  // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
        // @fb-only
    // @fb-only
                             // @fb-only
    // @fb-only
    // @fb-only
  // @fb-only
// @fb-only

  default: {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    return nullptr;
  }
  }
}

- (std::shared_ptr<igl::ITexture>)createTextureFromNativeDepth {
  switch (_config.backendVersion.flavor) {
#if IGL_BACKEND_METAL
  case igl::BackendFlavor::Metal: {
    auto& device = _shellPlatform->getDevice();
    auto* platformDevice = device.getPlatformDevice<igl::metal::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice);
    auto texture = platformDevice->createTextureFromNativeDepth(_depthStencilTexture, nullptr);
    return texture;
  }
#endif

#if IGL_BACKEND_OPENGL
  case igl::BackendFlavor::OpenGL: {
    auto& device = _shellPlatform->getDevice();
    auto* platformDevice = device.getPlatformDevice<igl::opengl::macos::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice);
    auto texture = platformDevice->createTextureFromNativeDepth(nullptr);
    return texture;
  }
#endif

#if IGL_BACKEND_VULKAN
  case igl::BackendFlavor::Vulkan: {
    auto& device = static_cast<igl::vulkan::Device&>(_shellPlatform->getDevice());
    auto extents = device.getVulkanContext().getSwapchainExtent();
    auto* platformDevice =
        _shellPlatform->getDevice().getPlatformDevice<igl::vulkan::PlatformDevice>();

    IGL_DEBUG_ASSERT(platformDevice);
    auto texture =
        platformDevice->createTextureFromNativeDepth(extents.width, extents.height, nullptr);
    return texture;
  }
#endif

#if IGL_BACKEND_WEBGPU
  case igl::BackendFlavor::WebGPU: {
    const uint32_t width = _webgpuSurface->getWidth();
    const uint32_t height = _webgpuSurface->getHeight();
    if (!_webgpuDepth || _webgpuDepth->getDimensions().width != width ||
        _webgpuDepth->getDimensions().height != height) {
      auto& device = _shellPlatform->getDevice();
      const igl::TextureFormat format =
          device.getTextureFormatCapabilities(igl::TextureFormat::S8_UInt_Z32_UNorm) != 0
              ? igl::TextureFormat::S8_UInt_Z32_UNorm
              : igl::TextureFormat::S8_UInt_Z24_UNorm;
      _webgpuDepth = device.createTexture(
          igl::TextureDesc::new2D(
              format, width, height, igl::TextureDesc::TextureUsageBits::Attachment),
          nullptr);
    }
    return _webgpuDepth;
  }
#endif

// @fb-only
  // @fb-only
    // @fb-only
        // @fb-only
        // @fb-only
        // @fb-only
  // @fb-only
// @fb-only

  default: {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    return nullptr;
  }
  }
}

- (BOOL)acceptsFirstResponder {
  return _shellPlatform != nullptr;
}

- (NSPoint)locationForEvent:(NSEvent*)event {
  NSRect contentRect = self.view.frame;
  NSPoint pos = [event locationInWindow];
  return NSMakePoint(pos.x, contentRect.size.height - pos.y);
}

// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static uint32_t getModifiers(NSEvent* event) {
  uint32_t modifiers = igl::shell::kKeyEventModifierNone;
  const NSUInteger flags = [event modifierFlags] & NSEventModifierFlagDeviceIndependentFlagsMask;

  if (flags & NSEventModifierFlagShift) {
    modifiers |= igl::shell::kKeyEventModifierShift;
  }
  if (flags & NSEventModifierFlagCapsLock) {
    modifiers |= igl::shell::kKeyEventModifierCapsLock;
  }
  if (flags & NSEventModifierFlagControl) {
    modifiers |= igl::shell::kKeyEventModifierControl;
  }
  if (flags & NSEventModifierFlagOption) {
    modifiers |= igl::shell::kKeyEventModifierOption;
  }
  if (flags & NSEventModifierFlagCommand) {
    modifiers |= igl::shell::kKeyEventModifierCommand;
  }
  if (flags & NSEventModifierFlagNumericPad) {
    modifiers |= igl::shell::kKeyEventModifierNumLock;
  }
  return modifiers;
}

- (void)keyUp:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::KeyEvent(false, event.keyCode, getModifiers(event)));
}

- (void)keyDown:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::KeyEvent(true, event.keyCode, getModifiers(event)));
  // NOLINTNEXTLINE(clang-analyzer-nullability.NullablePassedToNonnull)
  std::string characters([event.characters UTF8String]);
  for (const auto& c : characters) {
    _shellPlatform->getInputDispatcher().queueEvent(igl::shell::CharEvent{.character = c});
  }
}

- (void)mouseDown:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseButtonEvent(igl::shell::MouseButton::Left, true, curPoint.x, curPoint.y));
}

- (void)rightMouseDown:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseButtonEvent(igl::shell::MouseButton::Right, true, curPoint.x, curPoint.y));
}

- (void)otherMouseDown:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseButtonEvent(igl::shell::MouseButton::Middle, true, curPoint.x, curPoint.y));
}

- (void)mouseUp:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseButtonEvent(igl::shell::MouseButton::Left, false, curPoint.x, curPoint.y));
}

- (void)rightMouseUp:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseButtonEvent(igl::shell::MouseButton::Right, false, curPoint.x, curPoint.y));
}

- (void)otherMouseUp:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseButtonEvent(igl::shell::MouseButton::Middle, false, curPoint.x, curPoint.y));
}

- (void)mouseMoved:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseMotionEvent(curPoint.x, curPoint.y, event.deltaX, event.deltaY));
}

- (void)mouseDragged:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseMotionEvent(curPoint.x, curPoint.y, event.deltaX, event.deltaY));
}

- (void)rightMouseDragged:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseMotionEvent(curPoint.x, curPoint.y, event.deltaX, event.deltaY));
}

- (void)otherMouseDragged:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  NSPoint curPoint = [self locationForEvent:event];
  _shellPlatform->getInputDispatcher().queueEvent(
      igl::shell::MouseMotionEvent(curPoint.x, curPoint.y, event.deltaX, event.deltaY));
}

- (void)scrollWheel:(NSEvent*)event {
  if (!_shellPlatform) {
    return;
  }
  _shellPlatform->getInputDispatcher().queueEvent(igl::shell::MouseWheelEvent(
      event.scrollingDeltaX * _kMouseSpeed, event.scrollingDeltaY * _kMouseSpeed));
}

- (CGRect)frame {
  return _frame;
}

- (igl::ColorSpace)colorSpace {
  return _config.swapchainColorSpace;
}

@end
