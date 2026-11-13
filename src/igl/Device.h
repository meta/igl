/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <utility>
#include <igl/Common.h>
#include <igl/DeviceFeatures.h>
#include <igl/IResourceTracker.h>
#include <igl/PlatformDevice.h>
#include <igl/Texture.h>
#include <igl/base/IDeviceBase.h>

namespace igl {

struct BindGroupBufferDesc;
struct BindGroupTextureDesc;
struct BufferDesc;
struct ComputePipelineDesc;
struct DepthStencilStateDesc;
struct FramebufferDesc;
struct RenderPipelineDesc;
struct SamplerStateDesc;
struct ShaderLibraryDesc;
struct ShaderModuleDesc;
struct ShaderStagesDesc;
struct VertexInputStateDesc;
class IBuffer;
class ICommandBuffer;
class IComputePipelineState;
class IDepthStencilState;
class IFramebuffer;
class IRenderPipelineState;
class ISamplerState;
class IShaderLibrary;
class IShaderModule;
class IShaderStages;
class ITimer;
class ITimestampQueries;
class IVertexInputState;

/**
 * @brief InDevelopmentFeature is where you'd add in your own enum for testing out
 * an IGL feature you are working on. Once you declare it, you'd set it with
 * setDevelopmentFlags() from outside of IGL and then check for it with
 * testDevelopmentFlags() inside of IGL.
 *
 * For the IGL data types without access to IDevice, you'd need to do some
 * additional plumbing to pass the flag through.
 *
 * Note that none of this in-development code will be upstreamed. These flags
 * are only here so you have a way to revert to a safe path while testing in
 * production.
 */

// @fb-only
 // @fb-only
 // @fb-only
enum class InDevelopementFeatures : uint8_t {
  // Define your in-development feature enums here
  DummyFeatureExample,
};

/**
 * @brief Interface to a GPU that is used to draw graphics or do parallel computation.
 */
class IDevice : public ICapabilities, public base::IDeviceBase {
 public:
  ~IDevice() override = default;

  /*
   * Create a new BindGroup for textures.
   *
   * Vulkan: If `compatiblePipeline` is provided, the resulting BindGroup will be populated with
   * additional (dummy) textures and samplers in the binding slots where none were specified but are
   * expected by GLSL shaders. This ensures that the BindGroup is compatible with GLSL shaders from
   * the specified pipeline. If there's no pipeline specified, users must ensure all
   * textures/samplers expected by shaders are provided in the BindGroup description.
   */
  virtual Holder<BindGroupTextureHandle> createBindGroup(
      const BindGroupTextureDesc& desc,
      const IRenderPipelineState* IGL_NULLABLE compatiblePipeline = nullptr,
      Result* IGL_NULLABLE outResult = nullptr) = 0;
  virtual Holder<BindGroupBufferHandle> createBindGroup(
      const BindGroupBufferDesc& desc,
      Result* IGL_NULLABLE outResult = nullptr) = 0;

  virtual void destroy(BindGroupTextureHandle handle) = 0;
  virtual void destroy(BindGroupBufferHandle handle) = 0;
  virtual void destroy(SamplerHandle handle) = 0;

  /**
   * @brief Creates a command queue.
   * @see igl::CommandQueueDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created queue.
   */
  virtual std::shared_ptr<ICommandQueue> createCommandQueue(const CommandQueueDesc& desc,
                                                            Result* IGL_NULLABLE
                                                                outResult) noexcept = 0;

  /**
   * @brief Creates a buffer resource.
   * @see igl::BufferDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Unique pointer to the created buffer.
   */
  virtual std::unique_ptr<IBuffer> createBuffer(const BufferDesc& desc,
                                                Result* IGL_NULLABLE outResult) const noexcept = 0;

  /**
   * @brief Creates a depth stencil state.
   * @see igl::DepthStencilStateDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created depth stencil state.
   */
  virtual std::shared_ptr<IDepthStencilState> createDepthStencilState(
      const DepthStencilStateDesc& desc,
      Result* IGL_NULLABLE outResult) const = 0;

  /**
   * @brief Creates a sampler state.
   * @see igl::SamplerStateDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created sampler state.
   */
  virtual std::shared_ptr<ISamplerState> createSamplerState(const SamplerStateDesc& desc,
                                                            Result* IGL_NULLABLE
                                                                outResult) const = 0;

  /**
   * @brief Creates a texture resource.
   * @see igl::TextureDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created texture.
   */
  virtual std::shared_ptr<ITexture> createTexture(const TextureDesc& desc,
                                                  Result* IGL_NULLABLE
                                                      outResult) const noexcept = 0;

  /**
   * @brief Creates a texture view resource.
   * @see igl::TextureViewDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created texture.
   */
  virtual std::shared_ptr<ITexture> createTextureView(std::shared_ptr<ITexture> texture,
                                                      const TextureViewDesc& desc,
                                                      Result* IGL_NULLABLE
                                                          outResult) const noexcept = 0;

  /**
   * @brief Creates a timer for measuring the time taken by command buffers.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created timer.
   */
  virtual std::shared_ptr<ITimer> createTimer(Result* IGL_NULLABLE outResult) const noexcept = 0;

  /**
   * @brief Create a timestamp queries object that can hold up to maxTimestamps entries.
   * Returns nullptr if not supported on this backend/device.
   */
  // NOLINTNEXTLINE(bugprone-exception-escape)
  virtual std::shared_ptr<ITimestampQueries> createTimestampQueries(uint32_t maxTimestamps,
                                                                    Result* IGL_NULLABLE
                                                                        outResult) const noexcept {
    Result::setResult(
        outResult, Result::Code::Unsupported, "TimestampQueries not supported on this backend");
    (void)maxTimestamps;
    return nullptr;
  }

  /**
   * @brief Creates a vertex input state.
   * @see igl::VertexInputStateDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created vertex input state.
   */
  virtual std::shared_ptr<IVertexInputState> createVertexInputState(
      const VertexInputStateDesc& desc,
      Result* IGL_NULLABLE outResult) const = 0;

  /**
   * @brief Creates a compute pipeline state.
   * @see igl::ComputePipelineDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created compute pipeline state.
   */
  virtual std::shared_ptr<IComputePipelineState> createComputePipeline(
      const ComputePipelineDesc& desc,
      Result* IGL_NULLABLE outResult) const = 0;

  /**
   * @brief Creates a render pipeline state.
   * @see igl::RenderPipelineDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created render pipeline state.
   */
  virtual std::shared_ptr<IRenderPipelineState> createRenderPipeline(const RenderPipelineDesc& desc,
                                                                     Result* IGL_NULLABLE
                                                                         outResult) const = 0;

  /**
   * @brief Creates a shader module from either source code or pre-compiled data.
   * @see igl::ShaderModuleDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created shader module.
   */
  virtual std::shared_ptr<IShaderModule> createShaderModule(const ShaderModuleDesc& desc,
                                                            Result* IGL_NULLABLE
                                                                outResult) const = 0;

  /**
   * @brief Creates a frame buffer object.
   * @see igl::FramebufferDesc
   * @param desc Description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Shared pointer to the created frame buffer.
   */
  virtual std::shared_ptr<IFramebuffer> createFramebuffer(const FramebufferDesc& desc,
                                                          Result* IGL_NULLABLE outResult) = 0;

  /**
   * @brief Returns a platform-specific device. If the requested device type does not match that of
   * the actual underlying device, then null is returned.
   * @return Pointer to the underlying platform-specific device.
   */
  template<typename T>
  T* IGL_NULLABLE getPlatformDevice() noexcept {
    static_assert(std::is_base_of<IPlatformDevice, T>::value,
                  "getPlatformDevice() requires T to be derived from IPlatformDevice");
    return const_cast<T*>(static_cast<const IDevice*>(this)->getPlatformDevice<T>());
  }

  /**
   * @brief Returns a platform-specific device. If the requested device type does not match that of
   * the actual underlying device, then null is returned.
   * @return Pointer to the underlying platform-specific device.
   */
  template<typename T>
  const T* IGL_NULLABLE getPlatformDevice() const noexcept {
    static_assert(std::is_base_of<IPlatformDevice, T>::value,
                  "getPlatformDevice() requires T to be derived from IPlatformDevice");
    const IPlatformDevice& platformDevice = getPlatformDevice();
    if (platformDevice.isType(T::kType)) {
      return static_cast<const T*>(&platformDevice);
    }
    return nullptr;
  }

  /**
   * @brief Returns a platform-specific device. Returned value should not be held longer than the
   * original `IDevice`.
   * @return Pointer to the underlying platform-specific device.
   */
  IPlatformDevice& getPlatformDevice() noexcept {
    return const_cast<IPlatformDevice&>(static_cast<const IDevice*>(this)->getPlatformDevice());
  }

  /**
   * @brief Returns a platform-specific device. Returned value should not be held longer than the
   * original `IDevice`.
   * @return Pointer to the underlying platform-specific device.
   */
  [[nodiscard]] virtual const IPlatformDevice& getPlatformDevice() const noexcept = 0;

  /**
   * @brief Allow clients to verify that the scope that they are making IGL calls is current and
   * valid.
   * @return Whether or not the current scope is valid.
   */
  virtual bool verifyScope() {
    return defaultVerifyScope();
  }

  /**
   * @brief Get access to the staging buffer.
   * @return Pointer to the staging buffer interface, or nullptr if not available.
   */
  [[nodiscard]] base::IStagingBufferInterop* IGL_NULLABLE getStagingBufferInterop() override {
    return nullptr;
  }

  /**
   * @brief Returns the backend type.
   * @return The backend type enum value.
   */
  [[nodiscard]] BackendType getBackendType() const override = 0;

  /**
   * @brief Returns whether the GPU device has been lost (e.g. due to a hardware disconnect or
   * driver error). After device lost, all GPU operations should be skipped until the device is
   * recreated. Default implementation returns false (most backends are always valid).
   * @return true if the device is lost and cannot accept GPU commands.
   */
  [[nodiscard]] virtual bool isDeviceLost() const noexcept {
    return false;
  }

  /// Marks the device lost after its transport disconnects.
  /// Default is a no-op for backends that do not track device loss.
  virtual void markDeviceLost() noexcept {}

  /**
   * @brief Transitions device state from Lost to RecoveryArmed.
   * Called when the transport layer (e.g. Intralink) signals reconnect.
   * Default no-op for backends that do not support device loss recovery.
   */
  virtual void markRecoveryArmed() noexcept {}

  /// Mark the device healthy after successful recovery from device-lost state.
  /// Default is no-op; backends that track device-lost state override this to
  /// transition their internal state machine back to the healthy state.
  virtual void markHealthy() noexcept {}

  /**
   * @brief Returns raw pointer to native device handle.
   * @return Platform-specific device handle.
   */
  [[nodiscard]] void* IGL_NULLABLE getNativeDevice() const override = 0;

  /**
   * @brief Create a framebuffer from base descriptor.
   * @param desc The framebuffer descriptor.
   * @return Pointer to created framebuffer, or nullptr if not supported.
   */
  [[nodiscard]] base::IFramebufferInterop* IGL_NULLABLE
  createFramebufferInterop(const base::FramebufferInteropDesc& desc) override {
    (void)desc;
    return nullptr;
  }

  /**
   * @brief Returns the range of Z values in normalized device coordinates considered to be within
   * the viewing volume ie. [-1, 1], [0, 1]. Can be used when a client needs a generic way to adapt
   * how different backends handle NDC.
   * @return The Z value range within the viewing volume.
   */
  [[nodiscard]] virtual NormalizedZRange getNormalizedZRange() const {
    return NormalizedZRange::NegOneToOne;
  }

  /**
   * @brief Returns the number of draw calls made using this device.
   * @return The number of draw calls made so far.
   */
  [[nodiscard]] virtual size_t getCurrentDrawCount() const = 0;

  /**
   * @brief Returns the number of shaders compiled from source using this device. Creating shaders
   * from precompiled binaries does not count.
   * @return The number of shaders compiled so far.
   */
  [[nodiscard]] virtual size_t getShaderCompilationCount() const = 0;

  /**
   * @brief Returns the number of bytes of GPU memory currently in use, or 0 if the device does not
   * support memory tracking.
   * @return Used GPU memory
   */
  [[nodiscard]] virtual size_t getGPUMemoryUsage() const {
    return 0;
  }

  /**
   * @brief Creates a shader library with one or more shader modules.
   * @see igl::ShaderCompileDesc
   * @param desc The description for the shader library to be created.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Unique pointer to the created shader library.
   */
  virtual std::unique_ptr<IShaderLibrary> createShaderLibrary(const ShaderLibraryDesc& desc,
                                                              Result* IGL_NULLABLE
                                                                  outResult) const = 0;

  /**
   * @brief This is only used by EGL-based clients, e.g. Android, to set the default framebuffer to
   * render to. For all other clients, this is a no-op.
   */
  virtual void updateSurface(void* IGL_NONNULL /*nativeWindowType*/) {}

  /**
   * @brief Creates a shader stages object.
   * @see igl::ShaderStagesDesc
   * @param desc The description for the desired resource.
   * @param outResult Pointer to where the result (success, failure, etc) is written. Can be null if
   * no reporting is desired.
   * @return Unique pointer to the created shader stages object.
   */
  virtual std::unique_ptr<IShaderStages> createShaderStages(const ShaderStagesDesc& desc,
                                                            Result* IGL_NULLABLE
                                                                outResult) const = 0;

  /**
   * @brief Sets the resource tracker used by this device.
   * @see igl::IResourceTracker
   * @param tracker Shared pointer to the tracker.
   */
  void setResourceTracker(std::shared_ptr<IResourceTracker> tracker) noexcept {
    resourceTracker_ = std::move(tracker);
  }

  /**
   * @brief Returns the resource tracker used by this device.
   * @see igl::IResourceTracker
   * @return Shared pointer to the tracker.
   */
  [[nodiscard]] std::shared_ptr<IResourceTracker> getResourceTracker() const noexcept {
    return resourceTracker_;
  }

  /**
   * @brief Returns if a resource tracker is used by this device.
   */
  [[nodiscard]] bool hasResourceTracker() const noexcept {
    return resourceTracker_ != nullptr;
  }

  /**
   * @brief Returns a backend-specific color for debugging purposes
   *  - OpenGL: Yellow
   *  - Metal: Magenta
   *  - Vulkan: Cyan
   // @fb-only
   *  - Custom: Blue
   */
  [[nodiscard]] Color backendDebugColor() const noexcept;

  /**
   * @brief Controls an opaque internal bit field that enables/disables certain
   * in-development paths at run time.
   *
   * It is strongly recommended to set the fields during device creation or very near it.
   *
   * Some examples of the usage:
   *     * Preserving the original path while making a particularly dangerous change
   *     * Enabling a new IGL feature to a subset of users as an experiment.
   */
  bool testDevelopmentFlags(InDevelopementFeatures featureEnum) {
    const uint8_t pos = static_cast<uint8_t>(featureEnum);
    IGL_DEBUG_ASSERT(pos < 64);

    return (inDevelopmentFlags_ & (1ULL << pos)) != 0u;
  }

  /**
   * @brief Set/Unset an In-Development Flag
   * The pos/value is only meaningful to the client of IGL and whatever
   * in-development IGL feature you have in your IGL code base. No
   * in-development paths will be upstreamed or accepted into public IGL
   */
  void setDevelopmentFlags(InDevelopementFeatures featureEnum, bool val) {
    const uint8_t pos = static_cast<uint8_t>(featureEnum);
    IGL_DEBUG_ASSERT(pos < 64);

    if (val) {
      inDevelopmentFlags_ |= 1ULL << pos;
    } else {
      inDevelopmentFlags_ &= ~(1ULL << pos);
    }
  }

  /**
   * IGL can only be accessed by 1 thread at a time. Call this function to mark the current thread
   * as the "owning" thread.
   */
  virtual void setCurrentThread() {
  } // NOTE: for now, this is implemented only in IGL/Vulkan and IGL/OpenGL

 protected:
  virtual void beginScope() {
    ++scopeDepth_;
  }
  virtual void endScope() {
    --scopeDepth_;
  }
  [[nodiscard]] TextureDesc sanitize(const TextureDesc& desc) const;
  /// @brief Checks the usage and storage combinations that every backend rejects: InputAttachment
  /// requires Attachment, and ResourceStorage::Memoryless textures must be single-mip, optimally
  /// tiled, non-exportable 2D or 2D array attachments with no other usage.
  [[nodiscard]] Result validateTextureDesc(const TextureDesc& desc) const;
  IDevice() = default;

  /// @brief Helper to create framebuffer from base descriptor
  /// @param desc The base framebuffer descriptor
  /// @return Shared pointer to the created framebuffer, or nullptr on failure
  [[nodiscard]] std::shared_ptr<IFramebuffer> createFramebufferFromBaseDesc(
      const base::FramebufferInteropDesc& desc);

  uint64_t inDevelopmentFlags_ = 0;

 private:
  bool defaultVerifyScope();

  int scopeDepth_ = 0;
  std::shared_ptr<IResourceTracker> resourceTracker_;

  friend struct DeviceScope;
};

/**
 * @brief Delineates a scope for making API calls into IGL. Useful for marking diagnostics
 * boundaries.
 * @details To use this, instantiate a DeviceScope at the beginning of a code block that contains
 * a sequence of IGL calls. Typically, you do this at a top-level call such as initialization or a
 * per-frame render function. For clarity, it's best to place this sequence of calls inside braces.
 * For methods that call IGL, it's best to add an assert verifying the scope to ensure that the call
 * is occurring inside a valid DeviceScope.
 */
struct DeviceScope final {
  /**
   * @brief Creates a device scope associated with a given device.
   * @param device The device to be associated with the created scope.
   */
  explicit DeviceScope(IDevice& device);
  ~DeviceScope();

 private:
  IDevice& device_;

  // Prevent heap allocation
  static void* IGL_NONNULL operator new(std::size_t);
  static void* IGL_NONNULL operator new[](std::size_t);
};

} // namespace igl
