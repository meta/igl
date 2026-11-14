/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#import "ViewController.h"

#import "BackendVersion.h"
#import "IglShellPlatformAdapter.h"
#import "IglShellPlatformAdapterInternal.hpp" // IWYU pragma: keep
#import "IglSurfaceTexturesAdapter.h"
#import "IglSurfaceTexturesAdapterInternal.hpp" // IWYU pragma: keep
#import "RenderSessionController.h" // IWYU pragma: keep
#import "RenderSessionFactoryProvider.h"
#import "View.h"

#import <shell/shared/input/InputDispatcher.h>
#include <shell/shared/platform/Platform.h>
#import <igl/IGL.h> // IWYU pragma: keep
#include <igl/Texture.h>

#if IGL_BACKEND_METAL
#import <Metal/Metal.h>
#include <igl/metal/Device.h>
#include <igl/metal/Texture.h>
#endif

#if IGL_BACKEND_OPENGL
#include <igl/opengl/ios/Context.h>
#include <igl/opengl/ios/PlatformDevice.h>
#endif

#if IGL_BACKEND_WEBGPU
#include <igl/webgpu/Device.h>
#include <igl/webgpu/Surface.h>
#endif

// @fb-only
// @fb-only
// @fb-only
// @fb-only
// @fb-only

#include <memory>
#include <shell/shared/input/TouchListener.h>
#include <shell/shared/platform/PresentationRateController.h>
#include <shell/shared/platform/apple/PresentationRateApple.h>
#include <shell/shared/platform/ios/PlatformIos.h>
#include <shell/shared/renderSession/RenderSessionConfig.h>
#include <igl/DeviceFeatures.h>

@interface ViewController () <TouchDelegate, ViewSizeChangeDelegate, IglSurfaceTexturesProvider> {
  igl::shell::RenderSessionConfig _config;
  CALayer* _layer;
  CGRect _frame;
  id<CAMetalDrawable> _currentDrawable;
  id<MTLTexture> _depthStencilTexture;

  RenderSessionController* _renderSessionController;
  IglSurfaceTexturesAdapter _surfaceTexturesAdapter;
#if IGL_BACKEND_WEBGPU
  std::unique_ptr<igl::webgpu::Surface> _webgpuSurface;
  std::shared_ptr<igl::ITexture> _webgpuDepth;
#endif
}
- (BackendVersion*)toBackendVersion:(igl::BackendVersion)iglBackendVersion;
@end

@implementation ViewController

- (void)drawInMTKView:(nonnull MTKView*)view {
  _currentDrawable = view.currentDrawable;
  _depthStencilTexture = view.depthStencilTexture;

  IGL_DEBUG_ASSERT(_renderSessionController);
  [_renderSessionController tick];
}

- (void)mtkView:(nonnull MTKView*)view drawableSizeWillChange:(CGSize)size {
  [_renderSessionController releaseSessionFrameBuffer];
}

- (void)onViewSizeChange {
  [_renderSessionController releaseSessionFrameBuffer];
}

- (instancetype)init:(igl::shell::RenderSessionConfig)config
     factoryProvider:(RenderSessionFactoryProvider*)factoryProvider
               frame:(CGRect)frame {
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  if (self = [super initWithNibName:nil bundle:nil]) {
    self->_config = config;
    self->_frame = frame;
    _renderSessionController = [[RenderSessionController alloc]
        initWithBackendVersion:[self toBackendVersion:config.backendVersion]
               factoryProvider:factoryProvider
               surfaceProvider:self];
  }
  return self;
}

- (void)initRenderSessionController {
  IGL_DEBUG_ASSERT(_renderSessionController);

// @fb-only
  // @fb-only
    // @fb-only
    // @fb-only
  // @fb-only
// @fb-only

  [_renderSessionController initializeDevice];
}

- (igl::shell::Platform*)platform {
  IglShellPlatformAdapter* adapter = [_renderSessionController adapter];
  IGL_DEBUG_ASSERT(adapter);
  return adapter->platform;
}

// clang-format off
- (igl::SurfaceTextures)createSurfaceTexturesInternal {
  [[maybe_unused]] auto& device = [self platform]->getDevice();
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (_config.backendVersion.flavor) {
#if IGL_BACKEND_METAL
  case igl::BackendFlavor::Metal: {
    auto *platformDevice = device.getPlatformDevice<igl::metal::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice);
    return igl::SurfaceTextures{
        .color = platformDevice->createTextureFromNativeDrawable(_currentDrawable, nullptr),
        .depth = platformDevice->createTextureFromNativeDepth(_depthStencilTexture, nullptr),
    };
  }
#endif

#if IGL_BACKEND_OPENGL
  case igl::BackendFlavor::OpenGL_ES: {
    auto *platformDevice = device.getPlatformDevice<igl::opengl::ios::PlatformDevice>();
    IGL_DEBUG_ASSERT(platformDevice);
    return igl::SurfaceTextures{
        .color = platformDevice->createTextureFromNativeDrawable((CAEAGLLayer*)_layer, nullptr),
        .depth = platformDevice->createTextureFromNativeDepth((CAEAGLLayer*)_layer, _config.depthTextureFormat, nullptr),
    };
  }
#endif

#if IGL_BACKEND_WEBGPU
  case igl::BackendFlavor::WebGPU:
    return [self createWebGPUSurfaceTextures:static_cast<igl::webgpu::Device&>(device)];
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

  default: {
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return igl::SurfaceTextures{};
  }
  }
}
// clang-format on

#if IGL_BACKEND_WEBGPU
- (igl::SurfaceTextures)createWebGPUSurfaceTextures:(igl::webgpu::Device&)device {
  igl::Result result;
  if (!_webgpuSurface) {
    _webgpuSurface =
        igl::webgpu::Surface::createFromMetalLayer(device, (__bridge void*)_layer, &result);
    if (!_webgpuSurface) {
      IGL_LOG_ERROR("WebGPU surface creation failed: %s\n", result.message.c_str());
      return {};
    }
  }
  const CGFloat scale = self.view.contentScaleFactor;
  const auto width = static_cast<uint32_t>(self.view.bounds.size.width * scale);
  const auto height = static_cast<uint32_t>(self.view.bounds.size.height * scale);
  if (width == 0 || height == 0) {
    return {};
  }
  if (_webgpuSurface->getWidth() != width || _webgpuSurface->getHeight() != height) {
    result = _webgpuSurface->configure(width, height, _config.swapchainColorTextureFormat);
    if (!result.isOk()) {
      IGL_LOG_ERROR("WebGPU surface configuration failed: %s\n", result.message.c_str());
      return {};
    }
  }
  if (!_webgpuDepth || _webgpuDepth->getDimensions().width != width ||
      _webgpuDepth->getDimensions().height != height) {
    const igl::TextureFormat format =
        device.getTextureFormatCapabilities(_config.depthTextureFormat) != 0
            ? _config.depthTextureFormat
            : igl::TextureFormat::S8_UInt_Z24_UNorm;
    _webgpuDepth = device.createTexture(
        igl::TextureDesc::new2D(
            format, width, height, igl::TextureDesc::TextureUsageBits::Attachment),
        &result);
    if (!_webgpuDepth) {
      IGL_LOG_ERROR("WebGPU depth texture creation failed: %s\n", result.message.c_str());
      return {};
    }
  }
  auto color = _webgpuSurface->getCurrentTexture(&result);
  if (!color) {
    IGL_LOG_ERROR("WebGPU surface texture acquisition failed: %s\n", result.message.c_str());
    return {};
  }
  return igl::SurfaceTextures{
      .color = std::move(color),
      .depth = _webgpuDepth,
  };
}
#endif

// Protocol IglSurfaceTexturesProvider
- (IglSurfacesTextureAdapterPtr)createSurfaceTextures {
  _surfaceTexturesAdapter.surfaceTextures = [self createSurfaceTexturesInternal];
  return &_surfaceTexturesAdapter;
}

- (void)loadView {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (_config.backendVersion.flavor) {
  case igl::BackendFlavor::Invalid:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    break;
  case igl::BackendFlavor::Metal: {
#if IGL_BACKEND_METAL
    [self initRenderSessionController];
    auto d = static_cast<igl::metal::Device&>([self platform]->getDevice()).get();

    auto metalView = [[MetalView alloc] initWithFrame:_frame device:d];
    metalView.colorPixelFormat =
        igl::metal::Texture::textureFormatToMTLPixelFormat(_config.swapchainColorTextureFormat);
    metalView.depthStencilPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    if (_config.swapchainColorSpace == igl::ColorSpace::ExtendedSRGBLinear) {
      if (@available(iOS 16.0, *)) {
        CAMetalLayer* metalLayer = (CAMetalLayer*)metalView.layer;
        CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
        metalLayer.colorspace = colorSpace;
        CGColorSpaceRelease(colorSpace);
        metalLayer.wantsExtendedDynamicRangeContent = YES;
        [[maybe_unused]] const char* colorSpaceName =
            [(__bridge NSString*)CGColorSpaceGetName(metalLayer.colorspace) UTF8String];
        IGL_LOG_INFO("Metal layer: wantsExtendedDynamicRangeContent=%s colorspace=%s\n",
                     metalLayer.wantsExtendedDynamicRangeContent ? "YES" : "NO",
                     colorSpaceName ? colorSpaceName : "none");
      } else {
        IGL_LOG_INFO("Metal layer: EDR unsupported before iOS 16, layer unchanged\n");
      }
    }

    metalView.delegate = self;
    [metalView setTouchDelegate:self];
    self.view = metalView;
    _layer = metalView.layer;

    // The Metal path has no CADisplayLink of its own — drawInMTKView: is called from the
    // one MTKView owns — so the view itself is the tick source the seam has to reach.
    [self platform]->getPresentationRateController().setBackend(
        igl::shell::createMTKViewPresentationRateBackend(metalView));
#endif
    break;
  }
  case igl::BackendFlavor::OpenGL_ES: {
#if IGL_BACKEND_OPENGL
    auto openGLView = [[OpenGLView alloc] initWithTouchDelegate:self];
    openGLView.viewSizeChangeDelegate = self;

    NSString* drawablePropertyColorFormat = kEAGLColorFormatRGBA8;

    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (_config.swapchainColorTextureFormat) {
    case igl::TextureFormat::BGRA_UNorm8:
      drawablePropertyColorFormat = kEAGLColorFormatRGBA8;
      break;

    case igl::TextureFormat::BGRA_SRGB:
      drawablePropertyColorFormat = kEAGLColorFormatSRGBA8;
      break;

    default:
      break;
    }

    ((CAEAGLLayer*)openGLView.layer).drawableProperties =
        [NSDictionary dictionaryWithObjectsAndKeys:drawablePropertyColorFormat,
                                                   kEAGLDrawablePropertyColorFormat,
                                                   nil];
    self.view = openGLView;
#endif
    break;
  }
  case igl::BackendFlavor::OpenGL:
    IGL_DEBUG_ABORT("IGL Samples not set up for Desktop OpenGL backend");
    break;
  case igl::BackendFlavor::Vulkan:
    IGL_DEBUG_ABORT("IGL Samples not set up for Vulkan backend");
    break;
  case igl::BackendFlavor::D3D12:
    IGL_DEBUG_ABORT("IGL Samples not set up for D3D12 backend");
    break;
  case igl::BackendFlavor::WebGPU: {
#if IGL_BACKEND_WEBGPU
    auto webgpuView = [[WebGPUView alloc] initWithTouchDelegate:self];
    webgpuView.viewSizeChangeDelegate = self;
    self.view = webgpuView;
#else
    IGL_DEBUG_ABORT("IGL Samples not set up for WebGPU backend");
#endif
    break;
  }
  // @fb-only
    // @fb-only
    // @fb-only
  default:
    break;
  }
}

- (void)viewDidLoad {
  [super viewDidLoad];

  if (_config.backendVersion.flavor != igl::BackendFlavor::Metal) {
    _layer = self.view.layer;
    [self initRenderSessionController];
  }
}

- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
  if (_config.backendVersion.flavor != igl::BackendFlavor::Metal) {
    IGL_DEBUG_ASSERT(_renderSessionController);
    [_renderSessionController start];
  }
}

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:animated];

  if (_config.backendVersion.flavor != igl::BackendFlavor::Metal) {
    IGL_DEBUG_ASSERT(_renderSessionController);
    [_renderSessionController stop];
  }
}

- (void)touchBegan:(UITouch*)touch {
  CGPoint curPoint = [touch locationInView:self.view];
  CGPoint lastPoint = [touch previousLocationInView:self.view];
  [self platform]->getInputDispatcher().queueEvent(igl::shell::TouchEvent(
      true, curPoint.x, curPoint.y, curPoint.x - lastPoint.x, curPoint.y - lastPoint.y));
}

- (void)touchEnded:(UITouch*)touch {
  CGPoint curPoint = [touch locationInView:self.view];
  CGPoint lastPoint = [touch previousLocationInView:self.view];
  [self platform]->getInputDispatcher().queueEvent(igl::shell::TouchEvent(
      false, curPoint.x, curPoint.y, curPoint.x - lastPoint.x, curPoint.y - lastPoint.y));
}

- (void)touchMoved:(UITouch*)touch {
  CGPoint curPoint = [touch locationInView:self.view];
  CGPoint lastPoint = [touch previousLocationInView:self.view];
  [self platform]->getInputDispatcher().queueEvent(igl::shell::TouchEvent(
      true, curPoint.x, curPoint.y, curPoint.x - lastPoint.x, curPoint.y - lastPoint.y));
}

- (BackendVersion*)toBackendVersion:(igl::BackendVersion)iglBackendVersion {
  return [[BackendVersion alloc] init:static_cast<BackendFlavor>(iglBackendVersion.flavor)
                         majorVersion:iglBackendVersion.majorVersion
                         minorVersion:iglBackendVersion.minorVersion];
}

@end
