/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/PlatformDevice.h>
#include <igl/opengl/UnbindPolicy.h>

namespace igl::opengl {
class CommandQueue;

class Device : public IDevice {
  friend class HWDevice;
  friend class PlatformDevice;

 public:
  explicit Device(std::unique_ptr<IContext> context);
  ~Device() override;

  [[nodiscard]] Holder<BindGroupTextureHandle> createBindGroup(
      const BindGroupTextureDesc& desc,
      const IRenderPipelineState* IGL_NULLABLE compatiblePipeline,
      Result* IGL_NULLABLE outResult) override;
  [[nodiscard]] Holder<BindGroupBufferHandle> createBindGroup(const BindGroupBufferDesc& desc,
                                                              Result* IGL_NULLABLE
                                                                  outResult) override;
  void destroy(BindGroupTextureHandle handle) override;
  void destroy(BindGroupBufferHandle handle) override;
  void destroy(SamplerHandle handle) override;

  // Command Queue
  std::shared_ptr<ICommandQueue> createCommandQueue(const CommandQueueDesc& desc,
                                                    Result* IGL_NULLABLE
                                                        outResult) noexcept override;

  // Backend type query
  [[nodiscard]] BackendType getBackendType() const override {
    return BackendType::OpenGL;
  }

  // OpenGL doesn't have a single native device handle
  [[nodiscard]] void* IGL_NULLABLE getNativeDevice() const override {
    return nullptr;
  }

  // Resources
  std::unique_ptr<IBuffer> createBuffer(const BufferDesc& desc,
                                        Result* IGL_NULLABLE outResult) const noexcept override;
  std::shared_ptr<IDepthStencilState> createDepthStencilState(const DepthStencilStateDesc& desc,
                                                              Result* IGL_NULLABLE
                                                                  outResult) const override;
  std::shared_ptr<ISamplerState> createSamplerState(const SamplerStateDesc& desc,
                                                    Result* IGL_NULLABLE outResult) const override;
  std::shared_ptr<ITexture> createTexture(const TextureDesc& desc,
                                          Result* IGL_NULLABLE outResult) const noexcept override;
  std::shared_ptr<ITexture> createTextureView(std::shared_ptr<ITexture> texture,
                                              const TextureViewDesc& desc,
                                              Result* IGL_NULLABLE
                                                  outResult) const noexcept override;

  std::shared_ptr<IVertexInputState> createVertexInputState(const VertexInputStateDesc& desc,
                                                            Result* IGL_NULLABLE
                                                                outResult) const override;

  // Pipelines
  std::shared_ptr<IComputePipelineState> createComputePipeline(const ComputePipelineDesc& desc,
                                                               Result* IGL_NULLABLE
                                                                   outResult) const override;
  std::shared_ptr<IRenderPipelineState> createRenderPipeline(const RenderPipelineDesc& desc,
                                                             Result* IGL_NULLABLE
                                                                 outResult) const override;

  // Shaders
  std::unique_ptr<IShaderLibrary> createShaderLibrary(const ShaderLibraryDesc& desc,
                                                      Result* IGL_NULLABLE
                                                          outResult) const override;

  std::shared_ptr<IShaderModule> createShaderModule(const ShaderModuleDesc& desc,
                                                    Result* IGL_NULLABLE outResult) const override;

  std::unique_ptr<IShaderStages> createShaderStages(const ShaderStagesDesc& desc,
                                                    Result* IGL_NULLABLE outResult) const override;

  std::shared_ptr<IFramebuffer> createFramebuffer(const FramebufferDesc& desc,
                                                  Result* IGL_NULLABLE outResult) noexcept override;

  [[nodiscard]] base::IFramebufferInterop* IGL_NULLABLE
  createFramebufferInterop(const base::FramebufferInteropDesc& desc) override;

  // Timers
  std::shared_ptr<ITimer> createTimer(Result* IGL_NULLABLE outResult) const noexcept override;

  // Timestamp Queries
  std::shared_ptr<ITimestampQueries> createTimestampQueries(uint32_t maxTimestamps,
                                                            Result* IGL_NULLABLE
                                                                outResult) const noexcept override;

  /// createTimestampQueries() at the timer tier from
  /// DeviceFeatureSet::getGpuTimerTierWithOverride(), for a lab performance run that opts in to
  /// measuring GPU time on a GPU whose classified tier is lower.
  ///
  /// Lab only. It is off unless the caller opts in, and a release build must never call it. IGL
  /// cannot tell a lab build from a release build, so the caller must gate the call. It reaches
  /// GPUs whose timer queries crashed in production: Mali-G31 to Mali-G68 and Adreno below 620 (SEV
  /// S638750, with about 128 timer queries allocated, which two 64-slot pools at the Full tier
  /// allocate again), so a run on them can crash. The GPUs whose timer queries kept crashing after
  /// the classification (SEV S647462 and later), and a GPU with a null GL_RENDERER or GL_VENDOR,
  /// stay Disabled in every build; see isExcludedFromGpuTimerTierOverride().
  [[nodiscard]] std::shared_ptr<ITimestampQueries> createTimestampQueriesWithLabOnlyTierOverride(
      uint32_t maxTimestamps,
      GpuTimerTier tierOverride,
      Result* IGL_NULLABLE outResult) const noexcept;

  // debug markers useful in GPU captures
  void pushMarker(int len, const char* IGL_NULLABLE name);
  void popMarker();

  [[nodiscard]] const PlatformDevice& getPlatformDevice() const noexcept override = 0;

  void setCurrentThread() override;

  [[nodiscard]] IContext& getContext() const {
    return *context_;
  }

  // ICapabilities
  [[nodiscard]] bool hasFeature(DeviceFeatures capability) const override;
  [[nodiscard]] bool hasRequirement(DeviceRequirement requirement) const override;
  bool getFeatureLimits(DeviceFeatureLimits featureLimits, size_t& result) const override;
  [[nodiscard]] TextureFormatCapabilities getTextureFormatCapabilities(
      TextureFormat format) const override;
  [[nodiscard]] ShaderVersion getShaderVersion() const override;
  [[nodiscard]] BackendVersion getBackendVersion() const override;

  // Device Statistics
  [[nodiscard]] size_t getCurrentDrawCount() const override;
  [[nodiscard]] size_t getShaderCompilationCount() const override;

  bool verifyScope() override;

  void updateSurface(void* IGL_NULLABLE nativeWindowType) override;

 protected:
  void beginScope() override;
  void endScope() override;

  [[nodiscard]] const std::shared_ptr<IContext>& getSharedContext() const {
    return context_;
  }

 private:
  [[nodiscard]] std::shared_ptr<ITimestampQueries> createTimestampQueriesForTier(
      uint32_t maxTimestamps,
      GpuTimerTier tier,
      Result* IGL_NULLABLE outResult) const noexcept;

  GLint defaultFrameBufferID_{};
  GLint defaultFrameBufferWidth_{};
  GLint defaultFrameBufferHeight_{};
  const std::shared_ptr<IContext> context_;
  // on OpenGL we only need one command queue
  std::shared_ptr<CommandQueue> commandQueue_;
  const DeviceFeatureSet& deviceFeatureSet_;
  UnbindPolicy cachedUnbindPolicy_{};
};

} // namespace igl::opengl
