/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/metal/Device.h>

#import <Foundation/Foundation.h>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <igl/FramebufferWrapper.h>
#include <igl/Macros.h>
#include <igl/metal/Buffer.h>
#include <igl/metal/BufferSynchronizationManager.h>
#include <igl/metal/CommandQueue.h>
#include <igl/metal/ComputePipelineState.h>
#include <igl/metal/DepthStencilState.h>
#include <igl/metal/Framebuffer.h>
#include <igl/metal/PlatformDevice.h>
#include <igl/metal/RenderPipelineState.h>
#include <igl/metal/Result.h>
#include <igl/metal/SamplerState.h>
#include <igl/metal/Shader.h>
#include <igl/metal/Texture.h>
#include <igl/metal/Timer.h>
#include <igl/metal/TimestampQueries.h>
#include <igl/metal/VertexInputState.h>

namespace igl::metal {

// https://developer.apple.com/documentation/quartzcore/cametallayer/2938720-maximumdrawablecount?language=objc
// Max number of Metal drawables in the resource pool managed by Core Animation is 3.
#define IGL_METAL_MAX_IN_FLIGHT_BUFFERS 3

Device::Device(id<MTLDevice> device) :
  device_(device), platformDevice_(*this), deviceFeatureSet_(device) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  bufferSyncManager_ =
      std::make_shared<BufferSynchronizationManager>(IGL_METAL_MAX_IN_FLIGHT_BUFFERS);
}

Device::~Device() = default;

std::shared_ptr<ICommandQueue> Device::createCommandQueue( // NOLINT(bugprone-exception-escape)
    const CommandQueueDesc& /*desc*/,
    Result* outResult) noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  id<MTLCommandQueue> metalObject = [device_ newCommandQueue];
  auto resource =
      std::make_shared<CommandQueue>(*this, metalObject, bufferSyncManager_, deviceStatistics_);

  {
    const std::lock_guard<std::mutex> lock(mostRecentCommandQueueMutex_);
    mostRecentCommandQueue_ = resource;
  }

  Result::setOk(outResult);
  return resource;
}

namespace {
id<MTLBuffer> createMetalBuffer(id<MTLDevice> device,
                                const BufferDesc& desc,
                                MTLResourceOptions options) {
  id<MTLBuffer> metalObject = nil;
  if (desc.data != nullptr) {
    metalObject = [device newBufferWithBytes:desc.data length:desc.length options:options];
  } else {
    metalObject = [device newBufferWithLength:desc.length options:options];
  }
  metalObject.label = [NSString stringWithUTF8String:desc.debugName.c_str()];
  return metalObject;
}

// A buffer can't be memoryless, and Buffer::upload() writes through [buffer contents], which is nil
// for private storage. A Memoryless buffer gets Managed storage, which is Shared on iOS and tvOS.
MTLResourceOptions toBufferResourceStorageMode(ResourceStorage storage) {
  const ResourceStorage bufferStorage =
      storage == ResourceStorage::Memoryless ? ResourceStorage::Managed : storage;
  return Device::toMTLResourceStorageMode(bufferStorage, /*supportsMemoryless=*/false);
}

// The dynamic libraries that pipelines created from `stages` link, or nil when there are none.
NSArray<id<MTLDynamicLibrary>>* preloadedLibraries(const IShaderStages& stages)
    API_AVAILABLE(macos(12.0), ios(15.0)) {
  const auto& libraries = static_cast<const ShaderStages&>(stages).getLinkedLibraries();
  if (libraries.empty()) {
    return nil;
  }
  NSMutableArray<id<MTLDynamicLibrary>>* metalLibraries =
      [NSMutableArray arrayWithCapacity:libraries.size()];
  for (const auto& library : libraries) {
    [metalLibraries addObject:library->get()];
  }
  return metalLibraries;
}
} // namespace

std::unique_ptr<IBuffer> Device::createBuffer( // NOLINT(bugprone-exception-escape)
    const BufferDesc& desc,
    Result* outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (desc.hint & BufferDesc::BufferAPIHintBits::Ring) {
    return createRingBuffer(desc, outResult);
  }
  if (desc.hint & BufferDesc::BufferAPIHintBits::NoCopy) {
    return createBufferNoCopy(desc, outResult);
  }
  const MTLResourceOptions storage = toBufferResourceStorageMode(desc.storage);
  const MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | storage;

  id<MTLBuffer> metalObject = createMetalBuffer(device_, desc, options);
  std::unique_ptr<IBuffer> resource = std::make_unique<Buffer>(
      std::move(metalObject), options, desc.hint, 0 /* No accepted hints */, desc.type);
  if (hasResourceTracker()) {
    resource->initResourceTracker(getResourceTracker(), desc.debugName);
  }
  Result::setOk(outResult);
  return resource;
}

std::unique_ptr<IBuffer> Device::createRingBuffer( // NOLINT(bugprone-exception-escape)
    const BufferDesc& desc,
    Result* outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const MTLResourceOptions storage = toBufferResourceStorageMode(desc.storage);
  const MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | storage;

  // Create a ring of buffers
  std::vector<id<MTLBuffer>> bufferRing;
  for (size_t i = 0; i < bufferSyncManager_->getMaxInflightBuffers(); i++) {
    id<MTLBuffer> metalObject = createMetalBuffer(device_, desc, options);
    bufferRing.push_back(metalObject);
  }
  std::unique_ptr<IBuffer> resource = std::make_unique<RingBuffer>(
      std::move(bufferRing), options, bufferSyncManager_, desc.hint, desc.type);

  if (hasResourceTracker()) {
    resource->initResourceTracker(getResourceTracker(), desc.debugName);
  }
  Result::setOk(outResult);
  return resource;
}

std::unique_ptr<IBuffer> Device::createBufferNoCopy(const BufferDesc& desc,
                                                    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const MTLResourceOptions storage = toBufferResourceStorageMode(desc.storage);

  using Deallocator = void (^)(void*, NSUInteger);
  const Deallocator deallocator = nil;
  const MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | storage;
  id<MTLBuffer> metalObject = [device_ newBufferWithBytesNoCopy:const_cast<void*>(desc.data)
                                                         length:desc.length
                                                        options:options
                                                    deallocator:deallocator];

  std::unique_ptr<IBuffer> resource = std::make_unique<Buffer>(
      metalObject, options, desc.hint, BufferDesc::BufferAPIHintBits::NoCopy, desc.type);
  if (hasResourceTracker()) {
    resource->initResourceTracker(getResourceTracker(), desc.debugName);
  }
  Result::setOk(outResult);
  return resource;
}

std::shared_ptr<ISamplerState> Device::createSamplerState(const SamplerStateDesc& desc,
                                                          Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  return platformDevice_.createSamplerState(desc, outResult);
}

/**
 * @brief Creates a Metal texture from the given descriptor.
 *
 * Sanitizes the descriptor to ensure valid minimum values
 * for dimensions, layers, samples, and mip levels. Rejects
 * array textures unless type is TwoDArray, and rejects
 * exportable textures. For multisample textures, mip level
 * count is forced to 1. Multisample textures use Private
 * storage: Metal rejects multisample textures with Shared
 * or Managed storage in `validateWithDevice:`, which aborts
 * the process instead of returning an error, so a descriptor
 * that does not already request Private/Memoryless storage
 * is upgraded rather than handed to Metal unchanged. An unsupported
 * sample count is rejected with `Unsupported` for the same
 * reason.
 *
 * @param[in] desc Texture descriptor specifying format,
 *        dimensions, usage, storage, and other properties.
 * @param[out] outResult Optional error reporting. Set to
 *        ArgumentInvalid for usage and storage combinations
 *        rejected by validateTextureDesc(), Unsupported for
 *        invalid format or array type mismatch, Unimplemented
 *        for exportable textures, or RuntimeError if Metal
 *        texture allocation fails.
 * @return The created texture, or nullptr on failure.
 */
std::shared_ptr<ITexture> Device::createTexture( // NOLINT(bugprone-exception-escape)
    const TextureDesc& desc,
    Result* outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const Result validation = validateTextureDesc(desc);
  if (!validation.isOk()) {
    Result::setResult(outResult, validation);
    return nullptr;
  }
  const auto sanitized = sanitize(desc);
  if (desc.numLayers > 1 && desc.type != TextureType::TwoDArray) {
    Result::setResult(outResult,
                      Result::Code::Unsupported,
                      "Array textures are only supported when type is TwoDArray.");
    return nullptr;
  }
  if (desc.exportability != TextureDesc::TextureExportability::NoExport) {
    Result::setResult(outResult,
                      Result::Code::Unimplemented,
                      "Exportable textures are not supported on this platform.");
    return nullptr;
  };

  MTLTextureDescriptor* metalDesc = [MTLTextureDescriptor new];
  metalDesc.textureType = Texture::convertType(sanitized.type, sanitized.numSamples);
  metalDesc.pixelFormat = Texture::textureFormatToMTLPixelFormat(sanitized.format);
  if (metalDesc.pixelFormat == MTLPixelFormatInvalid) {
    Result::setResult(
        outResult,
        Result::Code::Unsupported,
        "Invalid Texture Format : " +
            std::string(TextureFormatProperties::fromTextureFormat(sanitized.format).name));
    IGL_DEBUG_ABORT(outResult->message.c_str());
    return nullptr;
  }
  metalDesc.width = sanitized.width;
  metalDesc.height = sanitized.height;
  metalDesc.depth = sanitized.depth;
  metalDesc.arrayLength = sanitized.numLayers;
  metalDesc.mipmapLevelCount = sanitized.numMipLevels;

  // The default sampleCount value is 1. If textureType is not MTLTextureType2DMultisample, this
  // value must be 1. Support for different sample count values varies by device. Call the
  // supportsTextureSampleCount: method to determine if your desired sample count value is
  // supported.
  IGL_DEBUG_ASSERT(sanitized.numSamples > 0, "Texture : Samples cannot be zero");
  metalDesc.sampleCount = 1;
  if (metalDesc.textureType == MTLTextureType2DMultisample) {
    metalDesc.mipmapLevelCount = 1;
    if (![device_ supportsTextureSampleCount:sanitized.numSamples]) {
      Result::setResult(outResult,
                        Result::Code::Unsupported,
                        "Device does not support the requested texture sample count");
      return nullptr;
    }
    metalDesc.sampleCount = sanitized.numSamples;
  }
#if TARGET_OS_OSX
  // MTLTextureType2DMultisampleArray not available in iOS
  if (@available(macOS 10.14, *)) {
    if (metalDesc.textureType == MTLTextureType2DMultisampleArray) {
      metalDesc.mipmapLevelCount = 1;
      if (![device_ supportsTextureSampleCount:sanitized.numSamples]) {
        Result::setResult(outResult,
                          Result::Code::Unsupported,
                          "Device does not support the requested texture sample count");
        return nullptr;
      }
      metalDesc.sampleCount = sanitized.numSamples;
    }
  }
#endif
  const bool supportsMemoryless = deviceFeatureSet_.supportsMemorylessTextures();
  metalDesc.usage = Texture::toMTLTextureUsage(sanitized.usage);
  // Metal rejects multisample textures with Shared or Managed storage. Upgrade any other
  // requested storage (including the default Invalid, which maps to Shared/Managed) to Private,
  // matching how `toMTLStorageMode()` falls back to Private for Memoryless textures on devices
  // without tile memory.
  const ResourceStorage textureStorage = sanitized.numSamples > 1 &&
                                                 sanitized.storage != ResourceStorage::Private &&
                                                 sanitized.storage != ResourceStorage::Memoryless
                                             ? ResourceStorage::Private
                                             : sanitized.storage;
  metalDesc.storageMode = toMTLStorageMode(textureStorage, supportsMemoryless);

  metalDesc.resourceOptions = MTLResourceCPUCacheModeDefaultCache |
                              toMTLResourceStorageMode(textureStorage, supportsMemoryless);

  id<MTLTexture> metalObject = [device_ newTextureWithDescriptor:metalDesc];
  if (!metalObject) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Failed to create Metal texture");
    IGL_DEBUG_ABORT(outResult->message.c_str());
    return nullptr;
  }
  metalObject.label = [NSString stringWithUTF8String:desc.debugName.c_str()];
  auto iglObject = std::make_shared<Texture>(
      metalObject, *this, desc.mipmapGeneration, sanitized.usage, textureStorage);
  if (hasResourceTracker()) {
    iglObject->initResourceTracker(getResourceTracker(), desc.debugName);
  }
  Result::setOk(outResult);

  // sanity check to ensure that the Result value and the returned object are in sync
  // i.e. we never have a valid Result with a nullptr return value, or vice versa
  IGL_DEBUG_ASSERT(outResult == nullptr || (outResult->isOk() == (iglObject != nullptr)));

  return iglObject;
}

std::shared_ptr<ITexture> Device::createTextureView( // NOLINT(bugprone-exception-escape)
    std::shared_ptr<ITexture> texture,
    const TextureViewDesc& desc,
    Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();

  Result::setResult(
      outResult, Result::Code::Unimplemented, "Texture views are not (yet) implemented");

  return nullptr;
}

std::shared_ptr<ITimer> Device::createTimer(Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (outResult) {
    Result::setOk(outResult);
  }
  return std::make_shared<Timer>();
}

// NOLINTNEXTLINE(bugprone-exception-escape)
std::shared_ptr<ITimestampQueries> Device::createTimestampQueries(uint32_t maxTimestamps,
                                                                  Result* IGL_NULLABLE
                                                                      outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (@available(macOS 10.15, iOS 14.0, *)) {
    // Find the timestamp counter set
    id<MTLCounterSet> timestampCounterSet = nil;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    for (id<MTLCounterSet> counterSet in device_.counterSets) {
      if ([counterSet.name isEqualToString:MTLCommonCounterSetTimestamp]) {
        timestampCounterSet = counterSet;
        break;
      }
    }
    if (!timestampCounterSet) {
      Result::setResult(
          outResult, Result::Code::Unsupported, "Device does not support timestamp counters");
      return nullptr;
    }

    MTLCounterSampleBufferDescriptor* descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
    descriptor.counterSet = timestampCounterSet;
    descriptor.sampleCount = maxTimestamps * TimestampQueries::kSamplesPerTimingSlot;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.label = @"IGL TimestampQueries";

    NSError* error = nil;
    id<MTLCounterSampleBuffer> csb = [device_ newCounterSampleBufferWithDescriptor:descriptor
                                                                             error:&error];
    if (!csb || error) {
      Result::setResult(outResult,
                        Result::Code::RuntimeError,
                        // NOLINTNEXTLINE(clang-analyzer-nullability.NullablePassedToNonnull)
                        error ? [error.localizedDescription UTF8String]
                              : "Failed to create counter sample buffer");
      return nullptr;
    }

    Result::setOk(outResult);
    return std::make_shared<TimestampQueries>(csb, maxTimestamps);
  } else {
    Result::setResult(
        outResult, Result::Code::Unsupported, "TimestampQueries require macOS 10.15+ or iOS 14+");
    return nullptr;
  }
}

/**
 * @brief Creates a Metal vertex input state from a descriptor.
 *
 * Validates the descriptor's attributes and input bindings,
 * ensuring counts are within limits, buffer indices and attribute
 * locations are in range, buffer indices match the binding count,
 * and attribute locations are unique. On success, populates a
 * MTLVertexDescriptor with attribute formats, buffer layouts,
 * and step functions, and wraps it in a VertexInputState.
 *
 * @param[in] desc Vertex input state descriptor specifying
 *        attributes, bindings, strides, and sample functions.
 * @param[out] outResult Receives the result status. Set to an
 *        error code (ArgumentOutOfRange, ArgumentInvalid, or
 *        RuntimeError) on failure, or OK on success.
 * @return Shared pointer to the created VertexInputState, or
 *         nullptr on validation or creation failure.
 */
std::shared_ptr<IVertexInputState> Device::createVertexInputState(const VertexInputStateDesc& desc,
                                                                  Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // Avoid buffer overrun in numAttributes.
  if (desc.numAttributes > IGL_VERTEX_ATTRIBUTES_MAX) {
    Result::setResult(outResult,
                      Result::Code::ArgumentOutOfRange,
                      "numAttributes is too large in VertexInputStateDesc");
    IGL_DEBUG_ABORT(outResult->message.c_str());
    return nullptr;
  }

  // Avoid buffer overrun in numInputBindings.
  if (desc.numInputBindings > IGL_BUFFER_BINDINGS_MAX) {
    Result::setResult(outResult,
                      Result::Code::ArgumentOutOfRange,
                      "numInputBindings is too large in VertexInputStateDesc");
    IGL_DEBUG_ABORT(outResult->message.c_str());
    return nullptr;
  }

  // Verify that bufferIndex and location are in their respective ranges.
  std::unordered_set<int> bufferIndexSet;
  std::unordered_set<int> attributeLocationSet;
  for (size_t i = 0; i < desc.numAttributes; ++i) {
    size_t bufferIndex = desc.attributes[i].bufferIndex;
    if (bufferIndex >= IGL_BUFFER_BINDINGS_MAX) {
      Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "bufferIndex out of range");
      IGL_DEBUG_ABORT(outResult->message.c_str());
      return nullptr;
    }

    const int attribLocation = desc.attributes[i].location;
    if (attribLocation < 0 || attribLocation >= IGL_VERTEX_ATTRIBUTES_MAX) {
      Result::setResult(
          outResult, Result::Code::ArgumentOutOfRange, "attribute location out of range");
      IGL_DEBUG_ABORT(outResult->message.c_str());
      return nullptr;
    }

    bufferIndexSet.insert(static_cast<int>(bufferIndex));
    attributeLocationSet.insert(attribLocation);
  }

  // Number of unique buffer indices should match the number of bindings
  if (bufferIndexSet.size() != desc.numInputBindings) {
    std::ostringstream msg;
    msg << "desc.numInputBindings : expected value is " << bufferIndexSet.size()
        << ", but actual value is " << desc.numInputBindings;
    Result::setResult(outResult, Result::Code::ArgumentInvalid, msg.str());
    IGL_DEBUG_ABORT(outResult->message.c_str());
    return nullptr;
  }

  // Attribute locations must be unique.
  // So number of attribute locations should match the desc.numAttributes
  if (attributeLocationSet.size() != desc.numAttributes) {
    std::ostringstream msg;
    msg << "desc.numAttributes : expected value is " << attributeLocationSet.size()
        << ", but actual value is " << desc.numAttributes;
    Result::setResult(
        outResult, Result::Code::ArgumentInvalid, "attribute locations are not unique");
    IGL_DEBUG_ABORT(outResult->message.c_str());
    return nullptr;
  }

  MTLVertexDescriptor* metalDesc = [MTLVertexDescriptor vertexDescriptor];
  if (metalDesc == nil) {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "failed to create MTLVertexDescriptor");
    IGL_DEBUG_ABORT(outResult->message.c_str());
    return nullptr;
  }

  // Validation completed. Populate the metal vertex descriptor.
  for (size_t i = 0; i < desc.numAttributes; ++i) {
    const size_t bufferIndex = desc.attributes[i].bufferIndex;
    const size_t dstAttribIndex = desc.attributes[i].location;

    metalDesc.attributes[dstAttribIndex].format =
        VertexInputState::convertAttributeFormat(desc.attributes[i].format);
    metalDesc.attributes[dstAttribIndex].bufferIndex = bufferIndex;
    metalDesc.attributes[dstAttribIndex].offset = desc.attributes[i].offset;

    metalDesc.layouts[bufferIndex].stride = desc.inputBindings[bufferIndex].stride;
    metalDesc.layouts[bufferIndex].stepFunction =
        VertexInputState::convertSampleFunction(desc.inputBindings[bufferIndex].sampleFunction);
    metalDesc.layouts[bufferIndex].stepRate = desc.inputBindings[bufferIndex].sampleRate;
  }

  auto iglObject = std::make_shared<VertexInputState>(metalDesc);

  Result::setOk(outResult);
  return iglObject;
}

std::shared_ptr<IDepthStencilState> Device::createDepthStencilState(
    const DepthStencilStateDesc& desc,
    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  MTLDepthStencilDescriptor* metalDesc = [MTLDepthStencilDescriptor new];
  metalDesc.label = [NSString stringWithUTF8String:desc.debugName.c_str()];
  metalDesc.depthCompareFunction = DepthStencilState::convertCompareFunction(desc.compareFunction);
  metalDesc.depthWriteEnabled = desc.isDepthWriteEnabled;
  metalDesc.frontFaceStencil = DepthStencilState::convertStencilDescriptor(desc.frontFaceStencil);
  metalDesc.backFaceStencil = DepthStencilState::convertStencilDescriptor(desc.backFaceStencil);
  id<MTLDepthStencilState> metalObject = [device_ newDepthStencilStateWithDescriptor:metalDesc];

  std::shared_ptr<DepthStencilState> iglObject = std::make_shared<DepthStencilState>(metalObject);

  Result::setOk(outResult);
  return iglObject;
}

std::shared_ptr<IComputePipelineState> Device::createComputePipeline(
    const ComputePipelineDesc& desc,
    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  NSError* error = nil;

  if (IGL_DEBUG_VERIFY_NOT(desc.shaderStages == nullptr)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing shader stages");
    return nullptr;
  }
  if (!IGL_DEBUG_VERIFY(desc.shaderStages->getType() == ShaderStagesType::Compute)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Shader stages not for compute");
    return nullptr;
  }
  if (!IGL_DEBUG_VERIFY(desc.shaderStages->getComputeModule())) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Missing compute shader");
    return nullptr;
  }

  MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
  descriptor.computeFunction =
      static_cast<ShaderModule*>(desc.shaderStages->getComputeModule().get())->get();
  if (@available(macOS 12.0, iOS 15.0, *)) {
    if (NSArray<id<MTLDynamicLibrary>>* libraries = preloadedLibraries(*desc.shaderStages)) {
      descriptor.preloadedLibraries = libraries;
    }
  }
  MTLComputePipelineReflection* reflection = nil;
  id<MTLComputePipelineState> metalObject =
      [device_ newComputePipelineStateWithDescriptor:descriptor
                                             options:MTLPipelineOptionNone
                                          reflection:&reflection
                                               error:&error];
  setResultFrom(outResult, error);
  if (error != nil) {
    return nullptr;
  }
  std::shared_ptr<ComputePipelineState> computePipelineState =
      std::make_shared<ComputePipelineState>(metalObject, reflection);

  return computePipelineState;
}

std::shared_ptr<IRenderPipelineState> Device::createRenderPipeline(const RenderPipelineDesc& desc,
                                                                   Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (!IGL_DEBUG_VERIFY(desc.shaderStages)) {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "RenderPipeline requires shader stages");
    return nullptr;
  }

  if (desc.shaderStages->getType() == ShaderStagesType::Render) {
    return createTraditionalRenderPipeline(desc, outResult);
  } else if (desc.shaderStages->getType() == ShaderStagesType::RenderMeshShader) {
    return createMeshRenderPipeline(desc, outResult);
  } else {
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  }
}

/**
 * @brief Creates a Metal render pipeline state for
 *        traditional vertex/fragment shaders.
 *
 * Translates the IGL render pipeline descriptor into an
 * MTLRenderPipelineDescriptor, configures vertex input,
 * shader functions, color attachment blending, and
 * depth/stencil formats, then creates the
 * MTLRenderPipelineState with reflection enabled.
 *
 * @param[in] desc Render pipeline descriptor providing
 *        shader stages, vertex input state, sample count,
 *        color/depth/stencil attachment configuration,
 *        and blending parameters.
 * @param[out] outResult Optional receiver for the
 *        creation result status.
 * @return The created pipeline state, or nullptr on
 *         failure.
 */
std::shared_ptr<IRenderPipelineState> Device::createTraditionalRenderPipeline(
    const RenderPipelineDesc& desc,
    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // TODO
  //  Size drawableSize = IGLNativeDrawableSize(layer_);
  //  graphicsDesc.viewportState.viewportCount = 1;
  //  graphicsDesc.viewportState.viewports[0] = (Viewport){0.0, 0.0, drawableSize.width,
  //  drawableSize.height, 0.0, 1.0};

  NSError* error = nil;
  MTLRenderPipelineDescriptor* metalDesc = [MTLRenderPipelineDescriptor new];

  metalDesc.label = [NSString stringWithUTF8String:desc.debugName.c_str()];

  metalDesc.sampleCount = desc.sampleCount;
  metalDesc.alphaToCoverageEnabled = desc.alphaToCoverageEnabled ? YES : NO;

  // (optional, can be null) Vertex input
  auto vertexInput = desc.vertexInputState;
  auto metalVertexInput = vertexInput ? static_cast<VertexInputState*>(vertexInput.get())->get()
                                      : nullptr;

  metalDesc.vertexDescriptor = metalVertexInput;

  if (!IGL_DEBUG_VERIFY(desc.shaderStages)) {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "RenderPipeline requires shader stages");
    return nullptr;
  }
  if (!IGL_DEBUG_VERIFY(desc.shaderStages->getType() == ShaderStagesType::Render)) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid, "Shader stages not for render");
    return nullptr;
  }

  // Vertex shader is required
  auto vertexModule = desc.shaderStages->getVertexModule();
  if (!IGL_DEBUG_VERIFY(vertexModule)) {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "RenderPipeline requires vertex module");
    return nullptr;
  }

  auto* vertexFunc = static_cast<ShaderModule*>(vertexModule.get());
  metalDesc.vertexFunction = vertexFunc->get();

  if (!IGL_DEBUG_VERIFY(metalDesc.vertexFunction)) {
    Result::setResult(
        outResult, Result::Code::RuntimeError, "RenderPipeline requires non-null vertex function");
    return nullptr;
  }

  // Fragment shader is optional
  auto fragmentModule = desc.shaderStages->getFragmentModule();
  if (fragmentModule) {
    auto* fragmentFunc = static_cast<ShaderModule*>(fragmentModule.get());
    metalDesc.fragmentFunction = fragmentFunc->get();
  }

  if (@available(macOS 12.0, iOS 15.0, *)) {
    if (NSArray<id<MTLDynamicLibrary>>* libraries = preloadedLibraries(*desc.shaderStages)) {
      metalDesc.vertexPreloadedLibraries = libraries;
      metalDesc.fragmentPreloadedLibraries = libraries;
    }
  }

  // Framebuffer
  for (uint32_t i = 0; i < desc.targetDesc.colorAttachments.size(); ++i) {
    const auto& src = desc.targetDesc.colorAttachments[i];
    MTLRenderPipelineColorAttachmentDescriptor* dst = metalDesc.colorAttachments[i];
    dst.pixelFormat = Texture::textureFormatToMTLPixelFormat(src.textureFormat);
    dst.writeMask = RenderPipelineState::convertColorWriteMask(src.colorWriteMask);
    dst.blendingEnabled = src.blendEnabled;
    dst.rgbBlendOperation = static_cast<MTLBlendOperation>(src.rgbBlendOp);
    dst.alphaBlendOperation = static_cast<MTLBlendOperation>(src.alphaBlendOp);
    dst.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(src.srcRGBBlendFactor);
    dst.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>(src.srcAlphaBlendFactor);
    dst.destinationRGBBlendFactor = static_cast<MTLBlendFactor>(src.dstRGBBlendFactor);
    dst.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>(src.dstAlphaBlendFactor);
  }

  // Depth and Stencil
  metalDesc.depthAttachmentPixelFormat =
      Texture::textureFormatToMTLPixelFormat(desc.targetDesc.depthAttachmentFormat);
  metalDesc.stencilAttachmentPixelFormat =
      Texture::textureFormatToMTLPixelFormat(desc.targetDesc.stencilAttachmentFormat);

  if (!device_) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Metal device is null");
    return nullptr;
  }

  MTLRenderPipelineReflection* reflection = nil;

  // Create reflection for use later in binding, etc.
  id<MTLRenderPipelineState> metalObject =
      [device_ newRenderPipelineStateWithDescriptor:metalDesc
                                            options:MTLPipelineOptionArgumentInfo |
                                                    MTLPipelineOptionBufferTypeInfo
                                         reflection:&reflection
                                              error:&error];
  setResultFrom(outResult, error);
  if (error != nil) {
    IGL_LOG_ERROR("%s\n", [error.localizedDescription UTF8String]);
    return nullptr;
  }
  if (!metalObject) {
    Result::setResult(outResult,
                      Result::Code::RuntimeError,
                      "Metal pipeline state creation returned nil without error");
    return nullptr;
  }

  return std::make_shared<RenderPipelineState>(metalObject, reflection, desc);
}

/**
 * @brief Creates a Metal render pipeline state using mesh shaders.
 *
 * Builds an MTLMeshRenderPipelineDescriptor from the provided
 * descriptor. Requires iOS 16+/macOS 13+ and Apple GPU Family 7
 * (A14/M1 or later). The mesh shader module is required; task
 * (object) and fragment shader modules are optional. Configures
 * color attachments, blending, and depth/stencil formats from
 * the descriptor's target description.
 *
 * @param[in] desc Pipeline configuration including shader
 *        stages, attachment formats, sample count, and blending.
 * @param[out] outResult Receives the error code on failure, or
 *        OK on success. May be null.
 * @return The created pipeline state, or nullptr on failure.
 */
std::shared_ptr<IRenderPipelineState> Device::createMeshRenderPipeline(
    const RenderPipelineDesc& desc,
    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (@available(iOS 16, macOS 13, *)) {
    // Check if the device supports mesh shaders
    // Mesh shaders require Apple GPU Family 7 or higher (A14/M1 and later)
    if (@available(iOS 16, macOS 13, *)) {
      if (![device_ supportsFamily:MTLGPUFamilyApple7]) {
        Result::setResult(outResult,
                          Result::Code::Unsupported,
                          "Mesh shaders require Apple GPU Family 7 or higher (A14/M1 and later)");
        return nullptr;
      }
    }

    NSError* error = nil;
    MTLMeshRenderPipelineDescriptor* metalDesc = [MTLMeshRenderPipelineDescriptor new];

    metalDesc.label = [NSString stringWithUTF8String:desc.debugName.c_str()];

    metalDesc.rasterSampleCount = desc.sampleCount;
    metalDesc.alphaToCoverageEnabled = desc.alphaToCoverageEnabled ? YES : NO;

    if (!IGL_DEBUG_VERIFY(desc.shaderStages)) {
      Result::setResult(
          outResult, Result::Code::RuntimeError, "RenderPipeline requires shader stages");
      return nullptr;
    }
    if (!IGL_DEBUG_VERIFY(desc.shaderStages->getType() == ShaderStagesType::RenderMeshShader)) {
      Result::setResult(
          outResult, Result::Code::ArgumentInvalid, "Shader stages not for mesh render");
      return nullptr;
    }
    // MTLMeshRenderPipelineDescriptor has no preloaded libraries.
    if (!static_cast<const ShaderStages&>(*desc.shaderStages).getLinkedLibraries().empty()) {
      Result::setResult(
          outResult, Result::Code::Unsupported, "Mesh pipelines cannot link dynamic libraries");
      return nullptr;
    }

    // Mesh shader is required
    auto meshModule = desc.shaderStages->getMeshModule();
    if (!IGL_DEBUG_VERIFY(meshModule)) {
      Result::setResult(
          outResult, Result::Code::RuntimeError, "MeshRenderPipeline requires mesh module");
      return nullptr;
    }

    auto* meshFunc = static_cast<ShaderModule*>(meshModule.get());
    metalDesc.meshFunction = meshFunc->get();

    if (!IGL_DEBUG_VERIFY(metalDesc.meshFunction)) {
      Result::setResult(
          outResult, Result::Code::RuntimeError, "RenderPipeline requires non-null mesh function");
      return nullptr;
    }

    // Task shader is optional
    auto taskModule = desc.shaderStages->getTaskModule();
    if (taskModule) {
      auto* taskFunc = static_cast<ShaderModule*>(taskModule.get());
      metalDesc.objectFunction = taskFunc->get();
    }

    // Fragment shader is optional
    auto fragmentModule = desc.shaderStages->getFragmentModule();
    if (fragmentModule) {
      auto* fragmentFunc = static_cast<ShaderModule*>(fragmentModule.get());
      metalDesc.fragmentFunction = fragmentFunc->get();
    }

    // Framebuffer
    for (uint32_t i = 0; i < desc.targetDesc.colorAttachments.size(); ++i) {
      const auto& src = desc.targetDesc.colorAttachments[i];
      if (igl::TextureFormat::Invalid == src.textureFormat) {
        continue;
      }

      MTLRenderPipelineColorAttachmentDescriptor* dst = metalDesc.colorAttachments[i];
      dst.pixelFormat = Texture::textureFormatToMTLPixelFormat(src.textureFormat);
      dst.writeMask = RenderPipelineState::convertColorWriteMask(src.colorWriteMask);
      dst.blendingEnabled = src.blendEnabled;
      dst.rgbBlendOperation = static_cast<MTLBlendOperation>(src.rgbBlendOp);
      dst.alphaBlendOperation = static_cast<MTLBlendOperation>(src.alphaBlendOp);
      dst.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(src.srcRGBBlendFactor);
      dst.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>(src.srcAlphaBlendFactor);
      dst.destinationRGBBlendFactor = static_cast<MTLBlendFactor>(src.dstRGBBlendFactor);
      dst.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>(src.dstAlphaBlendFactor);
    }

    // Depth and Stencil
    metalDesc.depthAttachmentPixelFormat =
        Texture::textureFormatToMTLPixelFormat(desc.targetDesc.depthAttachmentFormat);
    metalDesc.stencilAttachmentPixelFormat =
        Texture::textureFormatToMTLPixelFormat(desc.targetDesc.stencilAttachmentFormat);

    MTLRenderPipelineReflection* reflection = nil;

    // Create reflection for use later in binding, etc.
    id<MTLRenderPipelineState> metalObject =
        [device_ newRenderPipelineStateWithMeshDescriptor:metalDesc
                                                  options:MTLPipelineOptionArgumentInfo |
                                                          MTLPipelineOptionBufferTypeInfo
                                               reflection:&reflection
                                                    error:&error];
    setResultFrom(outResult, error);
    if (error != nil) {
      IGL_LOG_ERROR("%s\n", [error.localizedDescription UTF8String]);
      return nullptr;
    }

    return std::make_shared<RenderPipelineState>(metalObject, reflection, desc);
  } else {
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return nullptr;
  }
}

namespace {
MTLDataType convertConstantValueType(ConstantValueType type) {
  switch (type) {
  case ConstantValueType::Invalid:
    return MTLDataTypeNone;
  case ConstantValueType::Float1:
    return MTLDataTypeFloat;
  case ConstantValueType::Float2:
    return MTLDataTypeFloat2;
  case ConstantValueType::Float3:
    return MTLDataTypeFloat3;
  case ConstantValueType::Float4:
    return MTLDataTypeFloat4;
  case ConstantValueType::Boolean1:
    return MTLDataTypeBool;
  case ConstantValueType::Boolean2:
    return MTLDataTypeBool2;
  case ConstantValueType::Boolean3:
    return MTLDataTypeBool3;
  case ConstantValueType::Boolean4:
    return MTLDataTypeBool4;
  case ConstantValueType::Int1:
    return MTLDataTypeInt;
  case ConstantValueType::Int2:
    return MTLDataTypeInt2;
  case ConstantValueType::Int3:
    return MTLDataTypeInt3;
  case ConstantValueType::Int4:
    return MTLDataTypeInt4;
  case ConstantValueType::Mat2x2:
    return MTLDataTypeFloat2x2;
  case ConstantValueType::Mat3x3:
    return MTLDataTypeFloat3x3;
  case ConstantValueType::Mat4x4:
    return MTLDataTypeFloat4x4;
  default:
    return MTLDataTypeNone;
  }
}
} // namespace

bool shouldEnableFastMath(const ShaderCompilerOptions& options) noexcept {
  // Map IGL's backend-agnostic optimization strategy onto Metal's fast-math lever. There is no true
  // -O0 at runtime (newLibraryWithSource: exposes only the .default / .size optimization levels),
  // so on Metal the debug/release lever is fast math, not the optimization level. The historical
  // default honors the caller's explicit flag, keeping existing clients byte-for-byte unchanged.
  switch (options.optimization) {
  case ShaderOptimization::Default:
    return options.fastMathEnabled; // historical: honor the caller's explicit flag
  case ShaderOptimization::NoOpt:
    return false; // debug: reference numerics + clean source-level stepping
  case ShaderOptimization::Performance:
    return true; // release/profiling: real optimized GPU numbers
  default:
    return options.fastMathEnabled; // safe fallback for future strategies
  }
}

std::unique_ptr<IShaderLibrary> Device::createShaderLibrary(const ShaderLibraryDesc& desc,
                                                            Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (IGL_DEBUG_VERIFY_NOT(desc.moduleInfo.empty())) {
    Result::setResult(outResult, Result::Code::ArgumentInvalid);
    return nullptr;
  }
  id<MTLLibrary> metalLibrary = nil;
  NSError* error = nil;
  if (desc.input.type == ShaderInputType::Binary) {
    if (desc.input.length == 0 || desc.input.data == nullptr) {
      Result::setResult(outResult, Result::Code::ArgumentNull);
      return nullptr;
    }
    // With null queue and destructor, dispatch_data_create() function stores a pointer to the data
    // buffer and leaves the responsibility of releasing the buffer to the client
    auto data = dispatch_data_create(desc.input.data,
                                     desc.input.length,
                                     nullptr /* dispatch_queue_t queue */,
                                     nullptr /* dispatch_block_t
                      destructor */
    );

    metalLibrary = [device_ newLibraryWithData:data error:&error];
  } else {
    if (!desc.input.source || !std::strlen(desc.input.source)) {
      Result::setResult(outResult, Result::Code::ArgumentNull);
      return nullptr;
    }
    MTLCompileOptions* compileOpts = [MTLCompileOptions new];
    // Fast math is the debug/release lever on Metal: runtime newLibraryWithSource: has no true -O0,
    // and MTLCompileOptions.optimizationLevel already defaults to
    // MTLLibraryOptimizationLevelDefault (the optimizing level; optimize-for-size is out of scope),
    // so we leave it untouched — setting it would be a no-op and its setter is only available on
    // macOS 13 / iOS 16+.
    compileOpts.fastMathEnabled = shouldEnableFastMath(desc.input.options);

    NSString* shaderSource = [NSString stringWithUTF8String:desc.input.source];
    metalLibrary = [device_ newLibraryWithSource:shaderSource options:compileOpts error:&error];
    deviceStatistics_.incrementShaderCompilationCount();
  }

  if (!metalLibrary) {
    setResultFrom(outResult, error);
    return nullptr;
  }
  if (error) {
    // Compilation successful but with warnings
    IGL_LOG_INFO("%s\n", [error.localizedDescription UTF8String]);
  }
  std::vector<std::shared_ptr<IShaderModule>> modules;
  modules.reserve(desc.moduleInfo.size());
  for (const auto& info : desc.moduleInfo) {
    NSString* shaderEntrypoint = [NSString stringWithUTF8String:info.entryPoint.c_str()];
    if (!shaderEntrypoint) {
      Result::setResult(outResult, Result::Code::RuntimeError);
      return nullptr;
    }

    const auto& constantValues = info.functionConstantValues.getConstantValues();
    id<MTLFunction> metalFunction = nil;
    if (constantValues.empty()) {
      metalFunction = [metalLibrary newFunctionWithName:shaderEntrypoint];
    }
    // A function that declares function constants can only build a pipeline state through its
    // specialized variant, so fall through to newFunctionWithName:constantValues:error: both when
    // the plain lookup returned nil (a declared constant has no default) and when it returned a
    // function that still reports constants (every declared constant has a default).
    // A function that declares no constants must not go through that call at all: on iOS 16 it
    // does not return a usable function for a constant-free entry point, and the resulting nil
    // trips the IGL_DEBUG_ABORT below.
    // @fb-only
    if (metalFunction == nil || metalFunction.functionConstantsDictionary.count > 0) {
      MTLFunctionConstantValues* metalConstantValues = [MTLFunctionConstantValues new];
      const uint8_t* constantsBase = info.functionConstantValues.getData().data();
      for (size_t i = 0; i < constantValues.size(); ++i) {
        const auto& entry = constantValues[i];
        if (entry.type == ConstantValueType::Invalid) {
          continue;
        }
        [metalConstantValues setConstantValue:(constantsBase + entry.offset)
                                         type:convertConstantValueType(entry.type)
                                      atIndex:i];
      }
      metalFunction = [metalLibrary newFunctionWithName:shaderEntrypoint
                                         constantValues:metalConstantValues
                                                  error:&error];
    }
    if (!metalFunction) {
      IGL_DEBUG_ABORT("Could not find function '%s' in library\n", info.entryPoint.c_str());
      Result::setResult(
          outResult, Result::Code::RuntimeError, "Could not find function in library");
      return nullptr;
    }
    modules.emplace_back(std::make_shared<ShaderModule>(info, metalFunction));

    if (auto resourceTracker = getResourceTracker(); resourceTracker && !modules.empty()) {
      modules.back()->initResourceTracker(resourceTracker, desc.debugName);
    }
  }

  auto shaderLibrary = std::make_unique<ShaderLibrary>(std::move(modules));
  if (auto resourceTracker = getResourceTracker()) {
    shaderLibrary->initResourceTracker(resourceTracker, desc.debugName);
  }
  Result::setOk(outResult);

  return shaderLibrary;
}

std::shared_ptr<IShaderModule> Device::createShaderModule(const ShaderModuleDesc& desc,
                                                          Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  auto libraryDesc =
      desc.input.type == ShaderInputType::String
          ? ShaderLibraryDesc::fromStringInput(desc.input.source, {desc.info}, desc.debugName)
          : ShaderLibraryDesc::fromBinaryInput(
                desc.input.data, desc.input.length, {desc.info}, desc.debugName);
  auto library = createShaderLibrary(libraryDesc, outResult);
  if (library != nullptr) {
    return library->getShaderModule(desc.info.entryPoint);
  }
  return nullptr;
}

std::unique_ptr<IShaderStages> Device::createShaderStages(const ShaderStagesDesc& desc,
                                                          Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  const Result result;
  auto stages = std::make_unique<ShaderStages>(desc);
  if (auto resourceTracker = getResourceTracker()) {
    stages->initResourceTracker(resourceTracker, desc.debugName);
  }
  Result::setResult(outResult, result.code, result.message);
  if (!result.isOk()) {
    return nullptr;
  }

  return std::move(stages);
}

const PlatformDevice& Device::getPlatformDevice() const noexcept {
  return platformDevice_;
}

bool Device::isAppleGpu() const {
  IGL_PROFILER_FUNCTION();
#if IGL_PLATFORM_IOS
  return true;
#else
  if (@available(macOS 10.15, *)) {
    std::string gpuName = [device_.name UTF8String];
    return gpuName.find("Apple") != std::string::npos;
  } else {
    // Apple Macs didn't exist yet.
    return false;
  }
#endif
}

bool Device::hasFeature(DeviceFeatures feature) const {
  return deviceFeatureSet_.hasFeature(feature);
}

bool Device::hasRequirement(DeviceRequirement requirement) const {
  return deviceFeatureSet_.hasRequirement(requirement);
}

bool Device::getFeatureLimits(DeviceFeatureLimits featureLimits, size_t& result) const {
  return deviceFeatureSet_.getFeatureLimits(featureLimits, result);
}

ICapabilities::TextureFormatCapabilities Device::getTextureFormatCapabilities(
    TextureFormat format) const {
  return deviceFeatureSet_.getTextureFormatCapabilities(format);
}

ShaderVersion Device::getShaderVersion() const {
  // From https://developer.apple.com/documentation/metal/mtllanguageversion
  ShaderVersion version{.family = ShaderFamily::Metal};
#if IGL_PLATFORM_IOS
  if (@available(iOS 15, *)) {
    version.majorVersion = 2;
    version.minorVersion = 4;
  } else if (@available(iOS 14, *)) {
    version.majorVersion = 2;
    version.minorVersion = 3;
  } else if (@available(iOS 13, *)) {
    version.majorVersion = 2;
    version.minorVersion = 2;
  } else if (@available(iOS 12, *)) {
    version.majorVersion = 2;
    version.minorVersion = 1;
  } else if (@available(iOS 11, *)) {
    version.majorVersion = 2;
    version.minorVersion = 0;
  } else if (@available(iOS 10, *)) {
    version.majorVersion = 1;
    version.minorVersion = 2;
  } else if (@available(iOS 9, *)) {
    version.majorVersion = 1;
    version.minorVersion = 1;
  }
#elif IGL_PLATFORM_MACOSX || IGL_PLATFORM_MACCATALYST
  if (@available(macOS 12.0, *)) {
    version.majorVersion = 2;
    version.minorVersion = 4;
  } else if (@available(macOS 11.0, *)) {
    version.majorVersion = 2;
    version.minorVersion = 3;
  } else if (@available(macOS 10.15, *)) {
    version.majorVersion = 2;
    version.minorVersion = 2;
  } else if (@available(macOS 10.14, *)) {
    version.majorVersion = 2;
    version.minorVersion = 1;
  } else if (@available(macOS 10.13, *)) {
    version.majorVersion = 2;
    version.minorVersion = 0;
  } else if (@available(macOS 10.12, *)) {
    version.majorVersion = 1;
    version.minorVersion = 2;
  } else if (@available(macOS 10.11, *)) {
    version.majorVersion = 1;
    version.minorVersion = 1;
  }
#endif
  return version;
}

BackendVersion Device::getBackendVersion() const {
  if (@available(macOS 13.0, iOS 16.0, *)) {
    return {.flavor = BackendFlavor::Metal, .majorVersion = 3, .minorVersion = 0};
  }
#if TARGET_OS_OSX
#if TARGET_CPU_ARM64
  if (@available(macOS 10.13, iOS 11.0, *)) {
    return {.flavor = BackendFlavor::Metal, .majorVersion = 2, .minorVersion = 0};
  }
#else
  if (@available(macOS 11.0, iOS 11.0, *)) {
    return {.flavor = BackendFlavor::Metal, .majorVersion = 2, .minorVersion = 0};
  }
#endif
#endif

  return {.flavor = BackendFlavor::Metal, .majorVersion = 1, .minorVersion = 0};
}

size_t Device::getCurrentDrawCount() const {
  return deviceStatistics_.getDrawCount();
}

size_t Device::getShaderCompilationCount() const {
  return deviceStatistics_.getShaderCompilationCount();
}

size_t Device::getGPUMemoryUsage() const {
  return [device_ currentAllocatedSize];
}

MTLStorageMode Device::toMTLStorageMode(ResourceStorage storage, bool supportsMemoryless) {
  switch (storage) {
  case ResourceStorage::Private:
    return MTLStorageModePrivate;
  case ResourceStorage::Shared:
    return MTLStorageModeShared;
  case ResourceStorage::Memoryless:
    return supportsMemoryless ? MTLStorageModeMemoryless : MTLStorageModePrivate;
#if IGL_PLATFORM_MACOSX || IGL_PLATFORM_MACCATALYST
  case ResourceStorage::Managed:
  case ResourceStorage::Invalid:
  default:
    return MTLStorageModeManaged;
#else
  case ResourceStorage::Managed:
  case ResourceStorage::Invalid:
  default:
    return MTLStorageModeShared;
#endif
  }
}

MTLResourceOptions Device::toMTLResourceStorageMode(ResourceStorage storage,
                                                    bool supportsMemoryless) {
  switch (storage) {
  case ResourceStorage::Private:
    return MTLResourceStorageModePrivate;
  case ResourceStorage::Shared:
    return MTLResourceStorageModeShared;
  case ResourceStorage::Memoryless:
    return supportsMemoryless ? MTLResourceStorageModeMemoryless : MTLResourceStorageModePrivate;
#if IGL_PLATFORM_MACOSX || IGL_PLATFORM_MACCATALYST
  case ResourceStorage::Managed:
  case ResourceStorage::Invalid:
  default:
    return MTLResourceStorageModeManaged;
#else
  case ResourceStorage::Managed:
  case ResourceStorage::Invalid:
  default:
    return MTLResourceStorageModeShared;
#endif
  }
}

std::shared_ptr<ICommandQueue> Device::getMostRecentCommandQueue() const noexcept {
  const std::lock_guard<std::mutex> lock(mostRecentCommandQueueMutex_);
  return mostRecentCommandQueue_;
}

Holder<BindGroupTextureHandle> Device::createBindGroup(
    const BindGroupTextureDesc& desc,
    const IRenderPipelineState* IGL_NULLABLE /*compatiblePipeline*/,
    Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(!desc.debugName.empty(), "Each bind group should have a debug name");

  BindGroupTextureDesc description(desc);

  const auto handle = bindGroupTexturesPool.create(std::move(description));

  Result::setResult(outResult,
                    handle.empty() ? Result(Result::Code::RuntimeError, "Cannot create bind group")
                                   : Result());

  return {this, handle};
}

Holder<BindGroupBufferHandle> Device::createBindGroup(const BindGroupBufferDesc& desc,
                                                      Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(!desc.debugName.empty(), "Each bind group should have a debug name");

  BindGroupBufferDesc description(desc);

  const auto handle = bindGroupBuffersPool.create(std::move(description));

  Result::setResult(outResult,
                    handle.empty() ? Result(Result::Code::RuntimeError, "Cannot create bind group")
                                   : Result());

  return {this, handle};
}

void Device::destroy(BindGroupTextureHandle handle) {
  IGL_PROFILER_FUNCTION();
  if (handle.empty()) {
    return;
  }

  bindGroupTexturesPool.destroy(handle);
}

void Device::destroy(BindGroupBufferHandle handle) {
  IGL_PROFILER_FUNCTION();
  if (handle.empty()) {
    return;
  }

  bindGroupBuffersPool.destroy(handle);
}

void Device::destroy(SamplerHandle handle) {
  (void)handle;
  // IGL/Metal is not using sampler handles
}

base::IFramebufferInterop* IGL_NULLABLE
Device::createFramebufferInterop(const base::FramebufferInteropDesc& desc) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  auto framebuffer = createFramebufferFromBaseDesc(desc);
  if (!framebuffer) {
    return nullptr;
  }
  return new (std::nothrow) FramebufferWrapper(std::move(framebuffer));
}

} // namespace igl::metal
