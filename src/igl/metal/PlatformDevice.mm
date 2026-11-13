/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/metal/PlatformDevice.h>

#import <CoreGraphics/CGGeometry.h>
#import <CoreVideo/CVPixelBuffer.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/CALayer.h>
#import <QuartzCore/CAMetalLayer.h>
#include <igl/Macros.h>
#include <igl/metal/DepthStencilState.h>
#include <igl/metal/Device.h>
#include <igl/metal/Framebuffer.h>
#include <igl/metal/Result.h>
#include <igl/metal/SamplerState.h>
#include <igl/metal/Shader.h>
#include <igl/metal/Texture.h>

namespace igl::metal {

namespace {
constexpr const char* kDynamicLibrariesUnsupported =
    "This device cannot link Metal dynamic libraries";

// Infer the IGL TextureFormat from a CVPixelBuffer's pixel format type and plane index.
// This mirrors the automatic format inference that the OpenGL iOS backend performs
// in its TextureBuffer (opengl/ios/TextureBuffer.mm), making the Metal path support
// the same createTextureFromNativePixelBuffer overload without an explicit format.
// The conversion from CVPixelFormatType to igl::TextureFormat is inferred from CVPixelBuffer.h
TextureFormat convertToTextureFormat(OSType pixelFormat, size_t planeIndex) {
  switch (pixelFormat) {
  case kCVPixelFormatType_32BGRA:
    return TextureFormat::BGRA_UNorm8;

  case kCVPixelFormatType_32RGBA:
    return TextureFormat::RGBA_UNorm8;

  case kCVPixelFormatType_64RGBAHalf:
    return TextureFormat::RGBA_F16;

  case kCVPixelFormatType_OneComponent8:
    return TextureFormat::R_UNorm8;

  // TODO: There are explicit YUV formats in TextureFormat that are not supported by IGL Metal
  // backend. Once those are properly supported, we can add the conversion here as well.
  case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
  case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
    if (planeIndex == 0) {
      return TextureFormat::R_UNorm8;
    } else if (planeIndex == 1) {
      return TextureFormat::RG_UNorm8;
    } else {
      return TextureFormat::Invalid;
    }

  case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
  case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
    if (planeIndex == 0) {
      return TextureFormat::R_UInt16;
    } else if (planeIndex == 1) {
      return TextureFormat::RG_UInt16;
    } else {
      return TextureFormat::Invalid;
    }

  default:
    return TextureFormat::Invalid;
  }
}
} // namespace

PlatformDevice::PlatformDevice(Device& device) : device_(device) {}

PlatformDevice::~PlatformDevice() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  if (textureCache_) {
    CFRelease(textureCache_);
    textureCache_ = nullptr;
  }
}

std::shared_ptr<SamplerState> PlatformDevice::createSamplerState(const SamplerStateDesc& desc,
                                                                 Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  MTLSamplerDescriptor* metalDesc = [MTLSamplerDescriptor new];
  metalDesc.label = [NSString stringWithUTF8String:desc.debugName.c_str()];
  metalDesc.minFilter = SamplerState::convertMinMagFilter(desc.minFilter);
  metalDesc.magFilter = SamplerState::convertMinMagFilter(desc.magFilter);
  metalDesc.mipFilter = SamplerState::convertMipFilter(desc.mipFilter);
  metalDesc.lodMinClamp = desc.mipLodMin;
  metalDesc.lodMaxClamp = desc.mipLodMax;
  metalDesc.sAddressMode = SamplerState::convertAddressMode(desc.addressModeU);
  metalDesc.tAddressMode = SamplerState::convertAddressMode(desc.addressModeV);
  metalDesc.rAddressMode = SamplerState::convertAddressMode(desc.addressModeW);
  metalDesc.maxAnisotropy = desc.maxAnisotropic;
  if (desc.depthCompareEnabled && device_.hasFeature(DeviceFeatures::DepthCompare)) {
    metalDesc.compareFunction =
        DepthStencilState::convertCompareFunction(desc.depthCompareFunction);
  }

  id<MTLSamplerState> metalObject = [device_.get() newSamplerStateWithDescriptor:metalDesc];
  auto resource = std::make_shared<SamplerState>(metalObject);
  if (device_.hasResourceTracker()) {
    resource->initResourceTracker(device_.getResourceTracker(), desc.debugName);
  }
  Result::setOk(outResult);
  return resource;
}

std::shared_ptr<Framebuffer> PlatformDevice::createFramebuffer(const FramebufferDesc& desc,
                                                               Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  return std::static_pointer_cast<Framebuffer>(device_.createFramebuffer(desc, outResult));
}

std::unique_ptr<ITexture> PlatformDevice::createTextureFromNativeDrawable(
    id<CAMetalDrawable> nativeDrawable,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  auto iglObject = std::make_unique<Texture>(nativeDrawable, device_);
  if (auto resourceTracker = device_.getResourceTracker()) {
    iglObject->initResourceTracker(resourceTracker);
  }
  Result::setOk(outResult);
  return iglObject;
}

std::unique_ptr<ITexture> PlatformDevice::createTextureFromNativeDrawable(
    id<MTLTexture> nativeDrawable,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  auto iglObject = std::make_unique<Texture>(nativeDrawable, device_);
  if (auto resourceTracker = device_.getResourceTracker()) {
    iglObject->initResourceTracker(resourceTracker);
  }
  Result::setOk(outResult);
  return iglObject;
}

std::unique_ptr<ITexture> PlatformDevice::createTextureFromNativeDrawable(CALayer* nativeDrawable,
                                                                          Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (!nativeDrawable) {
    Result::setResult(outResult, Result::Code::ArgumentNull, "Invalid native drawable");
    return nullptr;
  }
#if (!TARGET_OS_SIMULATOR || __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000)
  if ([nativeDrawable isKindOfClass:[CAMetalLayer class]]) {
    id<CAMetalDrawable> drawableObject = [(CAMetalLayer*)nativeDrawable nextDrawable];
    if (!drawableObject) {
      Result::setResult(outResult, Result::Code::RuntimeError, "Could not retrieve a drawable.");
      return nullptr;
    }

    return createTextureFromNativeDrawable(drawableObject, outResult);
  } else {
    // Layer is not CAMetalLayer
    // This should never hit, unless there's a new layer type that supports Metal
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    Result::setResult(outResult, Result::Code::Unsupported);
    return nullptr;
  }
#else
  Result::setResult(outResult, Result::Code::Unsupported);
  return nullptr;
#endif
}

std::unique_ptr<ITexture> PlatformDevice::createTextureFromNativeDepth(
    id<MTLTexture> depthStencilTexture,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  auto iglObject = std::make_unique<Texture>(depthStencilTexture, device_);
  if (auto resourceTracker = device_.getResourceTracker()) {
    iglObject->initResourceTracker(resourceTracker);
  }
  Result::setOk(outResult);
  return iglObject;
}

std::unique_ptr<ITexture> PlatformDevice::createTextureFromNativePixelBuffer(
    CVImageBufferRef sourceImage,
    TextureFormat format,
    size_t planeIndex,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const bool isPlanar = CVPixelBufferIsPlanar(sourceImage) != 0u;
  const size_t width = (isPlanar ? CVPixelBufferGetWidthOfPlane(sourceImage, planeIndex)
                                 : CVPixelBufferGetWidth(sourceImage));
  const size_t height = (isPlanar ? CVPixelBufferGetHeightOfPlane(sourceImage, planeIndex)
                                  : CVPixelBufferGetHeight(sourceImage));
  return PlatformDevice::createTextureFromNativePixelBufferWithSize(
      sourceImage, format, width, height, planeIndex, outResult);
}

std::unique_ptr<ITexture> PlatformDevice::createTextureFromNativePixelBuffer(
    CVImageBufferRef sourceImage,
    size_t planeIndex,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const TextureFormat format =
      convertToTextureFormat(CVPixelBufferGetPixelFormatType(sourceImage), planeIndex);
  return PlatformDevice::createTextureFromNativePixelBuffer(
      sourceImage, format, planeIndex, outResult);
}

std::unique_ptr<ITexture> PlatformDevice::createTextureFromNativePixelBufferWithSize(
    CVImageBufferRef sourceImage,
    TextureFormat format,
    size_t width,
    size_t height,
    size_t planeIndex,
    Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  std::unique_ptr<Texture> resultTexture = nullptr;

#if (!TARGET_OS_SIMULATOR || __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000)
  CVMetalTextureCacheRef textureCache = getTextureCache();
  if (textureCache) {
    // Use the user provided texture instead
    const MTLPixelFormat metalFormat = Texture::textureFormatToMTLPixelFormat(format);
    if (metalFormat == MTLPixelFormatInvalid) {
      Result::setResult(outResult,
                        Result::Code::Unsupported,
                        "Invalid Texture Format : " +
                            std::string(TextureFormatProperties::fromTextureFormat(format).name));
      IGL_DEBUG_ABORT(outResult->message.c_str());
      return nullptr;
    }

    CVMetalTextureRef cvMetalTexture = nullptr;
    const CVReturn result = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                                      textureCache,
                                                                      sourceImage,
                                                                      nil,
                                                                      metalFormat,
                                                                      width,
                                                                      height,
                                                                      planeIndex,
                                                                      &cvMetalTexture);
    IGL_DEBUG_ASSERT(result == kCVReturnSuccess,
                     "Failed to create Metal texture from pixel buffer");

    if (result != kCVReturnSuccess) {
      Result::setResult(
          outResult,
          Result::Code::RuntimeError,
          "Failed to create Metal texture from pixel buffer: " + std::to_string(result));
      return nullptr;
    }

    id<MTLTexture> metalTexture = CVMetalTextureGetTexture(cvMetalTexture);
    CVBufferRelease(cvMetalTexture);
    cvMetalTexture = nullptr;

    if (metalTexture == nil) {
      Result::setResult(
          outResult, Result::Code::RuntimeError, "CoreVideo returned an empty Metal texture");
      return nullptr;
    }

    resultTexture = std::make_unique<Texture>(metalTexture, device_);
    if (auto resourceTracker = device_.getResourceTracker()) {
      resultTexture->initResourceTracker(resourceTracker);
    }
  } else {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "Failed to get the Metal texture cache");
    return nullptr;
  }
#else
  Result::setResult(outResult,
                    Result::Code::Unsupported,
                    "Metal pixel-buffer textures are not supported on this target");
  return nullptr;
#endif

  Result::setOk(outResult);
  return resultTexture;
}

Size PlatformDevice::getNativeDrawableSize(CALayer* nativeDrawable, Result* outResult) {
  IGL_PROFILER_FUNCTION();
#if (!TARGET_OS_SIMULATOR || __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000)
  Result::setOk(outResult);
  return {static_cast<float>(CGRectGetWidth(nativeDrawable.bounds)),
          static_cast<float>(CGRectGetHeight(nativeDrawable.bounds))};
#else
  Result::setResult(outResult, Result::Code::Unsupported, "Metal not supported on iOS simulator.");
  return {};
#endif
}

TextureFormat PlatformDevice::getNativeDrawableTextureFormat(CALayer* nativeDrawable,
                                                             Result* outResult) {
  IGL_PROFILER_FUNCTION();
  TextureFormat formatResult = TextureFormat::Invalid;

#if (!TARGET_OS_SIMULATOR || __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000)
  if ([nativeDrawable isKindOfClass:[CAMetalLayer class]]) {
    auto metalLayer = (CAMetalLayer*)nativeDrawable;
    formatResult = Texture::mtlPixelFormatToTextureFormat(metalLayer.pixelFormat);
    Result::setOk(outResult);
  } else {
    // Layer is not CAMetalLayer
    // This should never hit, unless there's a new layer type that supports Metal
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    Result::setResult(outResult, Result::Code::Unsupported);
  }
#else
  Result::setResult(outResult, Result::Code::Unsupported, "Metal not supported on iOS simulator.");
#endif

  return formatResult;
}

CVMetalTextureCacheRef PlatformDevice::getTextureCache() {
  IGL_PROFILER_FUNCTION();
#if (!TARGET_OS_SIMULATOR || __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000)
  if (textureCache_ == nullptr && device_.get() != nullptr) {
    const CVReturn result =
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device_.get(), nil, &textureCache_);
    IGL_DEBUG_ASSERT(result == kCVReturnSuccess, "Failed to created texture cache");

    if (result != kCVReturnSuccess) {
      NSLog(@"Failed to created texture cache");
      if (textureCache_) {
        CFRelease(textureCache_);
      }
      textureCache_ = nullptr;
    }
  }
#endif
  return textureCache_;
}

void PlatformDevice::flushNativeTextureCache() const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);
#if (!TARGET_OS_SIMULATOR || __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000)
  if (textureCache_) {
    CVMetalTextureCacheFlush(textureCache_, 0);
  }
#endif
}

void PlatformDevice::resetNativeTextureCache() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
#if (!TARGET_OS_SIMULATOR || __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000)
  const CVMetalTextureCacheRef cache = textureCache_;
  textureCache_ = nullptr;
  if (cache) {
    CVMetalTextureCacheFlush(cache, 0);
    CFRelease(cache);
  }
#endif
}

bool PlatformDevice::supportsDynamicLibraries() const noexcept {
  // The pipeline descriptors' preloaded-library properties start at macOS 12 / iOS 15.
  if (@available(macOS 12.0, iOS 15.0, *)) {
    return device_.get().supportsDynamicLibraries && device_.get().supportsRenderDynamicLibraries;
  }
  return false;
}

std::shared_ptr<DynamicLibrary> PlatformDevice::createDynamicLibrary(const void* data,
                                                                     size_t length,
                                                                     const std::string& debugName,
                                                                     Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (data == nullptr || length == 0) {
    Result::setResult(outResult, Result::Code::ArgumentNull, "No dynamic library data");
    return nullptr;
  }
  if (@available(macOS 12.0, iOS 15.0, *)) {
    if (supportsDynamicLibraries()) {
      // A null destructor is DISPATCH_DATA_DESTRUCTOR_DEFAULT, which copies the bytes.
      dispatch_data_t dispatchData = dispatch_data_create(data, length, nullptr, nullptr);
      NSError* error = nil;
      id<MTLLibrary> library = [device_.get() newLibraryWithData:dispatchData error:&error];
      if (library == nil) {
        const char* message = error.localizedDescription.UTF8String;
        Result::setResult(
            outResult, Result::Code::ArgumentInvalid, message ? message : "Not a Metal library");
        return nullptr;
      }
      if (library.type != MTLLibraryTypeDynamic) {
        Result::setResult(outResult,
                          Result::Code::ArgumentInvalid,
                          "Not a dynamic library; build it with `metal -dynamiclib`");
        return nullptr;
      }
      id<MTLDynamicLibrary> dynamicLibrary = [device_.get() newDynamicLibrary:library error:&error];
      if (dynamicLibrary == nil) {
        setResultFrom(outResult, error);
        return nullptr;
      }
      dynamicLibrary.label = [NSString stringWithUTF8String:debugName.c_str()];
      Result::setOk(outResult);
      return std::make_shared<DynamicLibrary>(dynamicLibrary);
    }
  }
  Result::setResult(outResult, Result::Code::Unsupported, kDynamicLibrariesUnsupported);
  return nullptr;
}

std::unique_ptr<IShaderStages> PlatformDevice::createShaderStages(
    const ShaderStagesDesc& desc,
    std::vector<std::shared_ptr<DynamicLibrary>> linkedLibraries,
    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (!linkedLibraries.empty() && !supportsDynamicLibraries()) {
    Result::setResult(outResult, Result::Code::Unsupported, kDynamicLibrariesUnsupported);
    return nullptr;
  }
  for (const auto& library : linkedLibraries) {
    if (library == nullptr) {
      Result::setResult(outResult, Result::Code::ArgumentNull, "Null dynamic library");
      return nullptr;
    }
  }
  auto stages = std::make_unique<ShaderStages>(desc, std::move(linkedLibraries));
  if (auto resourceTracker = device_.getResourceTracker()) {
    stages->initResourceTracker(resourceTracker, desc.debugName);
  }
  Result::setOk(outResult);
  return stages;
}

} // namespace igl::metal
