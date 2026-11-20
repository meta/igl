/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/Device.h>

#include <algorithm>
#include <cstring>
#include <igl/Macros.h>
#include <igl/opengl/Buffer.h>
#include <igl/opengl/CommandQueue.h>
#include <igl/opengl/ComputePipelineState.h>
#include <igl/opengl/DepthStencilState.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/Framebuffer.h>
#include <igl/opengl/FramebufferWrapper.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/RenderPipelineState.h>
#include <igl/opengl/SamplerState.h>
#include <igl/opengl/Shader.h>
#include <igl/opengl/TextureBuffer.h>
#include <igl/opengl/TextureTarget.h>
#include <igl/opengl/Timer.h>
#include <igl/opengl/TimestampQueries.h>
#include <igl/opengl/UniformBuffer.h>
#include <igl/opengl/VertexInputState.h>

namespace igl::opengl {

namespace {
std::unique_ptr<Buffer> allocateBuffer(BufferDesc::BufferType bufferType,
                                       BufferDesc::BufferAPIHint requestedApiHints,
                                       IContext& context) {
  std::unique_ptr<Buffer> resource;

  if ((bufferType & BufferDesc::BufferTypeBits::Index) != 0 ||
      (bufferType & BufferDesc::BufferTypeBits::Vertex) != 0 ||
      (bufferType & BufferDesc::BufferTypeBits::Indirect) != 0 ||
      (bufferType & BufferDesc::BufferTypeBits::Storage) != 0) {
    resource = std::make_unique<ArrayBuffer>(context, requestedApiHints, bufferType);
  } else if ((bufferType & BufferDesc::BufferTypeBits::Uniform) != 0) {
    if ((requestedApiHints & BufferDesc::BufferAPIHintBits::UniformBlock) != 0) {
      resource = std::make_unique<UniformBlockBuffer>(context, requestedApiHints, bufferType);
    } else {
      resource = std::make_unique<UniformBuffer>(context, requestedApiHints, bufferType);
    }

  } else {
    IGL_DEBUG_ASSERT_NOT_REACHED(); // desc.type is corrupt or new enum type was introduced
  }

  return resource;
}

template<class Ptr>
Ptr verifyResult(Ptr resource, Result inResult, Result* outResult) noexcept {
  if (inResult.isOk()) {
    Result::setOk(outResult);
  } else {
    IGL_DEBUG_ABORT(inResult.message.c_str());
    resource = {};
    Result::setResult(outResult, std::move(inResult));
  }

  return resource;
}

// common signature to the creation of all resources
template<class Ptr, class Desc, class... Params>
Ptr createResource(const Desc& desc, Result* outResult, Params&&... constructorParams) {
  // Create the object type and a smart pointer to hold it.
  using T = typename Ptr::element_type;
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  Ptr resource(new T(std::forward<Params>(constructorParams)...));

  // Create the resource using desc.
  Result result = resource->create(desc);
  return verifyResult<Ptr>(std::move(resource), std::move(result), outResult);
}

template<class T, class Desc, class... Params>
std::shared_ptr<T> createSharedResource(const Desc& desc,
                                        Result* outResult,
                                        Params&&... constructorParams) {
  return createResource<std::shared_ptr<T>>(
      desc, outResult, std::forward<Params>(constructorParams)...);
}

template<class T, class Desc, class... Params>
std::unique_ptr<T> createUniqueResource(const Desc& desc,
                                        Result* outResult,
                                        Params&&... constructorParams) {
  return createResource<std::unique_ptr<T>>(
      desc, outResult, std::forward<Params>(constructorParams)...);
}

} // namespace

Device::Device(std::unique_ptr<IContext> context) : // NOLINT(bugprone-exception-escape)
  context_(std::move(context)), deviceFeatureSet_(getContext().deviceFeatures()) {}
Device::~Device() = default;

// debug markers useful in GPU captures
void Device::pushMarker(int len, const char* name) {
  IGL_PROFILER_FUNCTION();
  if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessage)) {
    context_->pushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, len, name);
  } else {
    IGL_LOG_ERROR_ONCE("Device::pushMarker not supported in this context!");
  }
}

void Device::popMarker() {
  IGL_PROFILER_FUNCTION();
  if (deviceFeatureSet_.hasInternalFeature(InternalFeatures::DebugMessage)) {
    context_->popDebugGroup();
  } else {
    IGL_LOG_ERROR_ONCE("Device::popMarker not supported in this context!");
  }
}

// Command Queue
std::shared_ptr<ICommandQueue> Device::createCommandQueue(const CommandQueueDesc& /*desc*/,
                                                          Result* outResult) noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // we only use a single command queue on OpenGL
  if (!commandQueue_) {
    commandQueue_ = std::make_shared<CommandQueue>();
    commandQueue_->setInitialContext(context_);
  }
  Result::setOk(outResult);
  return commandQueue_;
}

// Resources
std::unique_ptr<IBuffer> Device::createBuffer( // NOLINT(bugprone-exception-escape)
    const BufferDesc& desc,
    Result* outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  std::unique_ptr<Buffer> resource = allocateBuffer(desc.type, desc.hint, getContext());

  if (resource) {
    resource->initialize(desc, outResult);
    if (hasResourceTracker()) {
      resource->initResourceTracker(getResourceTracker(), desc.debugName);
    }
  } else {
    Result::setResult(outResult, Result::Code::RuntimeError, "Could not instantiate buffer.");
  }

  return resource;
}

std::shared_ptr<IDepthStencilState> Device::createDepthStencilState(
    const DepthStencilStateDesc& desc,
    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // `outResult` is a null-safe Result* out-param: it is only ever touched via
  // Result::setResult()/setOk(), which null-check it. clang-tidy cannot see through the
  // resource-creation helper templates, so the nullable-dereference findings below are
  // false positives.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  return createSharedResource<DepthStencilState>(desc, outResult, getContext());
}

std::shared_ptr<ISamplerState> Device::createSamplerState(const SamplerStateDesc& desc,
                                                          Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  auto resource = std::make_shared<SamplerState>(getContext(), desc);
  if (hasResourceTracker()) {
    resource->initResourceTracker(getResourceTracker(), desc.debugName);
  }
  Result::setOk(outResult);
  return resource;
}

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

  std::unique_ptr<Texture> texture;
#if IGL_DEBUG_ABORT_ENABLED
  if (sanitized.type == TextureType::TwoD || sanitized.type == TextureType::TwoDArray) {
    size_t textureSizeLimit = 0;
    getFeatureLimits(DeviceFeatureLimits::MaxTextureDimension1D2D, textureSizeLimit);
    IGL_DEBUG_ASSERT(sanitized.width <= textureSizeLimit && sanitized.height <= textureSizeLimit,
                     "Texture limit size %zu is smaller than texture size %zux%zu",
                     textureSizeLimit,
                     sanitized.width,
                     sanitized.height);
  }
#endif

  if ((sanitized.usage & TextureDesc::TextureUsageBits::Sampled) != 0 ||
      (sanitized.usage & TextureDesc::TextureUsageBits::Storage) != 0) {
    texture = std::make_unique<TextureBuffer>(getContext(), desc.format);
  } else if ((sanitized.usage & TextureDesc::TextureUsageBits::Attachment) != 0) {
    if (sanitized.type == TextureType::TwoD && sanitized.numMipLevels == 1 &&
        sanitized.numLayers == 1) {
      texture = std::make_unique<TextureTarget>(getContext(), desc.format);
    } else {
      // Fall back to texture. e.g. TextureType::TwoDArray
      texture = std::make_unique<TextureBuffer>(getContext(), desc.format);
    }
  }

  if (texture != nullptr) {
    Result result = texture->create(sanitized, false);

    if (!result.isOk()) {
      texture = nullptr;
    } else if (hasResourceTracker()) {
      texture->initResourceTracker(getResourceTracker(), desc.debugName);
    }

    Result::setResult(outResult, std::move(result));
  } else {
    Result::setResult(
        outResult, Result::Code::Unsupported, "Unknown/unsupported texture usage bits.");
  }

  // sanity check to ensure that the Result value and the returned object are in sync
  // i.e. we never have a valid Result with a nullptr return value, or vice versa
  IGL_DEBUG_ASSERT(outResult == nullptr || (outResult->isOk() == (texture != nullptr)));

  return texture;
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

std::shared_ptr<IVertexInputState> Device::createVertexInputState(const VertexInputStateDesc& desc,
                                                                  Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  return createSharedResource<VertexInputState>(desc, outResult);
}

// Pipelines
std::shared_ptr<IRenderPipelineState> Device::createRenderPipeline(const RenderPipelineDesc& desc,
                                                                   Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  Result res;
  auto resource = std::make_shared<RenderPipelineState>(getContext(), desc, &res);
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  return verifyResult(std::move(resource), res, outResult);
}

std::shared_ptr<IComputePipelineState> Device::createComputePipeline(
    const ComputePipelineDesc& desc,
    Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  return createSharedResource<ComputePipelineState>(desc, outResult, getContext());
}

// Shaders

std::unique_ptr<IShaderLibrary> Device::createShaderLibrary(const ShaderLibraryDesc& /*desc*/,
                                                            Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  Result::setResult(outResult, Result::Code::Unsupported);
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  return nullptr;
}

std::shared_ptr<IShaderModule> Device::createShaderModule(const ShaderModuleDesc& desc,
                                                          Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  auto sm = createSharedResource<ShaderModule>(desc, outResult, getContext(), desc.info);
  if (auto resourceTracker = getResourceTracker(); sm && resourceTracker) {
    sm->initResourceTracker(resourceTracker, desc.debugName);
  }
  return sm;
}

std::unique_ptr<IShaderStages> Device::createShaderStages(const ShaderStagesDesc& desc,
                                                          Result* outResult) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // Need to pass desc twice.
  // The first instance is for the createUniqueResource pattern.
  // The second instance is so it also gets passed to the ShaderStages constructor.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  auto stages = createUniqueResource<ShaderStages>(desc, outResult, desc, getContext());
  if (auto resourceTracker = getResourceTracker(); stages && resourceTracker) {
    stages->initResourceTracker(std::move(resourceTracker), desc.debugName);
  }
  return stages;
}

std::shared_ptr<IFramebuffer> Device::createFramebuffer(const FramebufferDesc& desc,
                                                        Result* outResult) noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(deviceFeatureSet_.hasInternalFeature(InternalFeatures::FramebufferObject));
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  return getPlatformDevice().createFramebuffer(desc, outResult);
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

bool Device::hasFeature(DeviceFeatures capability) const {
  return deviceFeatureSet_.hasFeature(capability);
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
  IGL_DEBUG_ASSERT(context_);
  return deviceFeatureSet_.getShaderVersion();
}

BackendVersion Device::getBackendVersion() const {
  IGL_DEBUG_ASSERT(context_);
  return deviceFeatureSet_.getBackendVersion();
}

void Device::beginScope() {
  IGL_PROFILER_FUNCTION();
  IDevice::beginScope();

  IGL_DEBUG_ASSERT(context_);
  context_->setCurrent();

  // UnbindPolicy is fixed for duration of this scope
  cachedUnbindPolicy_ = getContext().getUnbindPolicy();
}

void Device::endScope() {
  IGL_PROFILER_FUNCTION();
  if (cachedUnbindPolicy_ == UnbindPolicy::EndScope) {
    // Ensure state on exit is consistent, for any external rendering that happens later.
    context_->colorMask(1u, 1u, 1u, 1u);
    context_->blendFunc(GL_ONE, GL_ZERO);
    context_->bindBuffer(GL_ARRAY_BUFFER, 0);
    context_->bindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    context_->pixelStorei(GL_PACK_ALIGNMENT, 4);
    context_->pixelStorei(GL_UNPACK_ALIGNMENT, 4);
  }

  IDevice::endScope();

  // Clear current context, if we are supposed to.
  if (cachedUnbindPolicy_ == UnbindPolicy::ClearContext && !verifyScope()) {
    context_->clearCurrentContext();
  }
}

void Device::updateSurface(void* nativeWindowType) {}

bool Device::verifyScope() {
  IGL_DEBUG_ASSERT(context_);
  return IDevice::verifyScope() && context_->isCurrentContext();
}

size_t Device::getCurrentDrawCount() const {
  return context_->getCurrentDrawCount();
}

size_t Device::getShaderCompilationCount() const {
  return context_->getShaderCompilationCount();
}

Holder<BindGroupTextureHandle> Device::createBindGroup(
    const BindGroupTextureDesc& desc,
    const IRenderPipelineState* IGL_NULLABLE /*compatiblePipeline*/,
    Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(context_);
  IGL_DEBUG_ASSERT(!desc.debugName.empty(), "Each bind group should have a debug name");

  BindGroupTextureDesc description(desc);

  const auto handle = context_->bindGroupTexturesPool.create(std::move(description));

  Result::setResult(outResult,
                    handle.empty() ? Result(Result::Code::RuntimeError, "Cannot create bind group")
                                   : Result());

  return {this, handle};
}

Holder<BindGroupBufferHandle> Device::createBindGroup(const BindGroupBufferDesc& desc,
                                                      Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(context_);
  IGL_DEBUG_ASSERT(!desc.debugName.empty(), "Each bind group should have a debug name");

  BindGroupBufferDesc description(desc);

  const auto handle = context_->bindGroupBuffersPool.create(std::move(description));

  Result::setResult(outResult,
                    handle.empty() ? Result(Result::Code::RuntimeError, "Cannot create bind group")
                                   : Result());

  return {this, handle};
}

// NOLINTNEXTLINE(bugprone-exception-escape)
std::shared_ptr<ITimer> Device::createTimer(Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (deviceFeatureSet_.hasFeature(DeviceFeatures::Timers)) {
    Result::setOk(outResult);
    return std::make_shared<Timer>(*context_);
  }
  Result::setResult(
      outResult, Result::Code::Unsupported, "Timers are not supported on this device");
  return nullptr;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
std::shared_ptr<ITimestampQueries> Device::createTimestampQueries(uint32_t maxTimestamps,
                                                                  Result* IGL_NULLABLE
                                                                      outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // Tier-based limit from GL_RENDERER / GL_VENDOR classification.
  return createTimestampQueriesForTier(
      maxTimestamps, deviceFeatureSet_.getGpuTimerTier(), outResult);
}

std::shared_ptr<ITimestampQueries> Device::createTimestampQueriesWithLabOnlyTierOverride(
    uint32_t maxTimestamps,
    GpuTimerTier tierOverride,
    Result* IGL_NULLABLE outResult) const noexcept {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  return createTimestampQueriesForTier(
      maxTimestamps, deviceFeatureSet_.getGpuTimerTierWithOverride(tierOverride), outResult);
}

std::shared_ptr<ITimestampQueries> Device::createTimestampQueriesForTier(
    uint32_t maxTimestamps,
    GpuTimerTier tier,
    Result* IGL_NULLABLE outResult) const noexcept {
  if (tier == GpuTimerTier::Disabled) {
    Result::setResult(outResult,
                      Result::Code::Unsupported,
                      "TimestampQueries disabled (no extension or blocked GPU tier)");
    return nullptr;
  }
  const uint32_t tierMax = static_cast<uint32_t>(tier);

  const uint32_t cappedMax = std::min(maxTimestamps, tierMax);
  auto queries = std::make_shared<TimestampQueries>(getContext(), cappedMax);
  if (!queries->isValid()) {
    // Driver reports timer query support but iglGenQueries silently failed
    // (e.g. broken Mali budget GPU drivers). Return nullptr so
    // GPUTimingCollector::isEnabled() returns false.
    Result::setResult(outResult,
                      Result::Code::RuntimeError,
                      "TimestampQueries: iglGenQueries failed (query IDs are zero)");
    return nullptr;
  }
  Result::setOk(outResult);
  return queries;
}

void Device::destroy(BindGroupTextureHandle handle) {
  IGL_PROFILER_FUNCTION();
  if (handle.empty()) {
    return;
  }

  IGL_DEBUG_ASSERT(context_);

  context_->bindGroupTexturesPool.destroy(handle);
}

void Device::destroy(BindGroupBufferHandle handle) {
  IGL_PROFILER_FUNCTION();
  if (handle.empty()) {
    return;
  }

  IGL_DEBUG_ASSERT(context_);

  context_->bindGroupBuffersPool.destroy(handle);
}

void Device::destroy(SamplerHandle handle) {
  (void)handle;
  // IGL/OpenGL is not using sampler handles
}

void Device::setCurrentThread() {
  getContext().setCurrent();
}

} // namespace igl::opengl
