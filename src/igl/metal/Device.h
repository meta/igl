/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

@protocol MTLCommandQueue;
@protocol MTLDevice;
#import <Metal/MTLTypes.h>
#include <ldrutils/lutils/Pool.h>
#include <mutex>
#include <igl/CommandEncoder.h>
#include <igl/Device.h>
#include <igl/Shader.h>
#include <igl/metal/DeviceFeatureSet.h>
#include <igl/metal/DeviceStatistics.h>
#include <igl/metal/PlatformDevice.h>

namespace igl::metal {

class BufferSynchronizationManager;

class Device : public IDevice {
  friend class HWDevice;

 public:
  explicit Device(id<MTLDevice> IGL_NONNULL device);
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

  std::shared_ptr<ITimer> createTimer(Result* IGL_NULLABLE outResult) const noexcept override;

  std::shared_ptr<ITimestampQueries> createTimestampQueries(uint32_t maxTimestamps,
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

  // Platform-specific extensions
  [[nodiscard]] const PlatformDevice& getPlatformDevice() const noexcept override;

  IGL_INLINE id<MTLDevice> IGL_NONNULL get() const {
    return device_;
  }

  [[nodiscard]] void* IGL_NULLABLE getNativeDevice() const override {
    return (__bridge void*)device_;
  }

  // ICapabilities
  [[nodiscard]] bool isAppleGpu() const;
  [[nodiscard]] bool hasFeature(DeviceFeatures feature) const override;
  [[nodiscard]] bool hasRequirement(DeviceRequirement requirement) const override;
  bool getFeatureLimits(DeviceFeatureLimits featureLimits, size_t& result) const override;
  [[nodiscard]] TextureFormatCapabilities getTextureFormatCapabilities(
      TextureFormat format) const override;
  [[nodiscard]] ShaderVersion getShaderVersion() const override;
  [[nodiscard]] BackendVersion getBackendVersion() const override;

  // Device Statistics
  [[nodiscard]] size_t getCurrentDrawCount() const override;

  [[nodiscard]] size_t getShaderCompilationCount() const override;

  [[nodiscard]] size_t getGPUMemoryUsage() const override;

  [[nodiscard]] BackendType getBackendType() const override {
    return BackendType::Metal;
  }

  [[nodiscard]] base::IFramebufferInterop* IGL_NULLABLE
  createFramebufferInterop(const base::FramebufferInteropDesc& desc) override;

  [[nodiscard]] NormalizedZRange getNormalizedZRange() const override {
    return NormalizedZRange::ZeroToOne;
  }

  /// ResourceStorage::Memoryless maps to memoryless storage if `supportsMemoryless` and to private
  /// storage otherwise. Only textures can be memoryless.
  static MTLStorageMode toMTLStorageMode(ResourceStorage storage, bool supportsMemoryless);
  static MTLResourceOptions toMTLResourceStorageMode(ResourceStorage storage,
                                                     bool supportsMemoryless);

  /// Safe to call from any thread, alongside createCommandQueue().
  std::shared_ptr<ICommandQueue> getMostRecentCommandQueue() const noexcept;

 public:
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> bindGroupBuffersPool;
  ldr::Pool<BindGroupTextureTag, BindGroupTextureDesc> bindGroupTexturesPool;

 private:
  std::unique_ptr<IBuffer> createRingBuffer(const BufferDesc& desc,
                                            Result* IGL_NULLABLE outResult) const noexcept;

  std::unique_ptr<IBuffer> createBufferNoCopy(const BufferDesc& desc,
                                              Result* IGL_NULLABLE outResult) const;

  std::shared_ptr<IRenderPipelineState> createTraditionalRenderPipeline(
      const RenderPipelineDesc& desc,
      Result* IGL_NULLABLE outResult) const;

  std::shared_ptr<IRenderPipelineState> createMeshRenderPipeline(const RenderPipelineDesc& desc,
                                                                 Result* IGL_NULLABLE
                                                                     outResult) const;

  id<MTLDevice> IGL_NONNULL device_;
  PlatformDevice platformDevice_;

  DeviceFeatureSet deviceFeatureSet_;
  std::shared_ptr<BufferSynchronizationManager> bufferSyncManager_;
  mutable DeviceStatistics deviceStatistics_;
  mutable std::mutex mostRecentCommandQueueMutex_;
  std::shared_ptr<ICommandQueue> mostRecentCommandQueue_;

  id<MTLCommandQueue> IGL_NULLABLE getBufferInitQueue() const;
  mutable std::once_flag bufferInitQueueFlag_;
  mutable id<MTLCommandQueue> IGL_NULLABLE bufferInitQueue_ = nil;
};

/// Resolves whether Metal fast math should be enabled for an MSL compile from IGL's
/// backend-agnostic `ShaderCompilerOptions`. On Metal the build-mode lever is fast math rather than
/// the optimization level: runtime `newLibraryWithSource:` exposes no true `-O0`, so debug builds
/// disable fast math (reference numerics + clean source-level stepping) and release/profiling
/// builds enable it. `ShaderOptimization::Default` preserves the caller's explicit
/// `fastMathEnabled` flag so existing IGL clients are byte-for-byte unchanged.
[[nodiscard]] bool shouldEnableFastMath(const ShaderCompilerOptions& options) noexcept;

} // namespace igl::metal
