/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <igl/DeviceFeatures.h>
#include <igl/Texture.h>
#include <igl/opengl/Version.h>

namespace igl::opengl {

class IContext;

// clang-format off
enum class Extensions {
  AppleRgb422,                // GL_APPLE_rgb_422 is supported
  BindlessTextureArb,         // GL_ARB_bindless_texture is supported
  BindlessTextureNv,          // GL_NV_bindless_texture is supported
  Debug,                      // GL_KHR_debug is supported
  DebugLabel,                 // GL_EXT_debug_label is supported
  DebugMarker,                // GL_EXT_debug_marker is supported
  Depth24,                    // GL_OES_depth24 is supported
  Depth32,                    // GL_OES_depth32 is supported
  DepthTexture,               // GL_OES_depth_texture is supported
  DiscardFramebuffer,         // GL_EXT_discard_framebuffer is supported
  Es2Compatibility,           // GL_ARB_ES2_compatibility is supported
  DrawBuffers,                // GL_EXT_draw_buffers is supported
  DrawBuffersIndexedExt,      // GL_EXT_draw_buffers_indexed is supported
  DrawBuffersIndexedOes,      // GL_OES_draw_buffers_indexed is supported
  FramebufferBlit,            // GL_EXT_framebuffer_blit is supported
  FramebufferObject,          // GL_ARB_framebuffer_object is supported
  InvalidateSubdata,          // GL_ARB_invalidate_subdata is supported
  MapBuffer,                  // GL_OES_mapbuffer is supported
  MapBufferRange,             // GL_EXT_map_buffer_range is supported
  MultiSampleApple,           // GL_APPLE_framebuffer_multisample is supported
  MultiSampleExt,             // GL_EXT_multisampled_render_to_texture is supported
  MultiSampleExt2,            // GL_EXT_multisampled_render_to_texture2 is supported
  MultiSampleImg,             // GL_IMG_multisampled_render_to_texture is supported
  MultiViewMultiSample,       // GL_OVR_multiview_multisampled_render_to_texture is supported
  PolygonOffsetClamp,         // GL_ARB_polygon_offset_clamp is supported
  RequiredInternalFormat,     // GL_OES_required_internalformat is supported
  ShaderFramebufferFetch,     // GL_EXT_shader_framebuffer_fetch is supported (not _non_coherent)
  ShaderImageLoadStore,       // GL_EXT_shader_image_load_store is supported
  Srgb,                       // GL_EXT_sRGB is supported
  SrgbWriteControl,           // GL_EXT_sRGB_write_control is supported
  Sync,                       // GL_APPLE_sync is supported
  TexStorage,                 // GL_EXT_texture_storage is supported
  Texture3D,                  // GL_OES_texture_3D is supported
  TextureFormatBgra8888Apple, // GL_EXT_texture_format_BGRA8888 is supported
  TextureFormatBgra8888Ext,   // GL_APPLE_texture_format_BGRA8888 is supported
  TextureFloat,               // GL_ARB_texture_float is supported
  TextureHalfFloat,           // GL_OES_texture_half_float is supported
  TextureRgArb,               // GL_ARB_texture_rg is supported
  TextureRgExt,               // GL_EXT_texture_rg is supported
  TextureSrgb,                // GL_EXT_texture_sRGB is supported
  TextureType2101010Rev,  // GL_EXT_texture_type_2_10_10_10_REV is supporteds
  TimerQuery,                 // GL_EXT_timer_query is supported
  VertexArrayObject,          // GL_OES_vertex_array_object is supported
  VertexAttribDivisor,        // GL_NV_instanced_arrays is supported
};
// clang-format on

// clang-format off
enum class InternalFeatures {
  ClearBufferfv,             // glClearBufferfv is supported
  ClearDepthf,               // glClearDepthf is supported
  DebugLabel,                // Debug labels on objects are supported
  DebugMessage,              // Debug messages and group markers are supported
  DebugMessageCallback,      // Debug message callbacks are supported
  DrawArraysIndirect,        // glDrawArraysIndirect is supported
  DrawBuffersIndexed,        // glColorMaski, glEnablei and glBlend*Separatei are supported
  MultiDrawIndirect,         // glMultiDrawArraysIndirect is supported
  // GL_EXT_multisampled_render_to_texture or the IMG version works
  MultisampledRenderToTexture,
  FramebufferBlit,           // BlitFramebuffer is supported
  FramebufferObject,         // Framebuffer objects are supported
  GetStringi,                // GetStringi is supported
  InternalFormatQuery,       // glGetInternalformativ is supported
  InvalidateFramebuffer,     // glInvalidateFramebuffer is supported
  MapBuffer,                 // glMapBuffer is supported
  PackRowLength,             // GL_PACK_ROW_LENGTH is supported with glPixelStorei
  PixelBufferObject,         // PBOs are available
  PolygonFillMode,           // glPolygonFillMode is supported
  ProgramInterfaceQuery,     // Querying info about shader program interfaces is supported
  SeamlessCubeMap,           // GL_TEXTURE_CUBE_MAP_SEAMLESS is supported
  ShaderImageLoadStore,      // Shader image load/store is supported
  Sync,                      // Sync objects are supported
  TexStorage,                // glTexStorage* is available
  TextureClampToBorder,      // GL_CLAMP_TO_BORDER is supported (GL 1.3+, ES 3.2+, or extension)
  TextureCompare,            // GL_TEXTURE_COMPARE_MODE and GL_TEXTURE_COMPARE_FUNC are supported
  UnmapBuffer,               // glUnmapBuffer is supported
  UnpackRowLength,           // GL_UNPACK_ROW_LENGTH is supported with glPixelStorei
  VertexArrayObject,         // VAOS are available
  VertexAttribDivisor,       // glVertexAttribDivisor is supported
};
// clang-format on

// clang-format off
enum class TextureFeatures {
  ColorFilterable16f,           // XXX16F textures can use GL_LINEAR filtering
  ColorFilterable32f,           // XXX32F textures can use GL_LINEAR filtering
  ColorFormatRgb10A2UI,         // RGB10_A2UI is supported as an internal format
  ColorFormatRgInt,             // Integer R and RG textures are supported
  ColorFormatRgUNorm16,         // UNorm 16 R and RG textures are supported
  ColorFormatRgbaUNorm16,       // UNorm 16 RGBA textures are supported
  ColorRenderbuffer16f,         // RenderbufferStorage supports XXX16F for color targets
  ColorRenderbuffer32f,         // RenderbufferStorage supports XXX32F for color targets
  ColorRenderbufferRg16f,       // RenderbufferStorage supports Rg16F for color targets
  ColorRenderbufferRg32f,       // RenderbufferStorage supports Rg32F for color targets
  ColorRenderbufferRg8,         // RenderbufferStorage supports R8 and RG8
  ColorRenderbufferRgb10A2,     // RenderbufferStorage supports RGB10_A2
  ColorRenderbufferRgb16f,      // RenderbufferStorage supports RGB16F for color targets
  ColorRenderbufferRgba8,       // RenderbufferStorage supports RGB8 and RGBA8
  ColorRenderbufferSrgba8,      // RenderbufferStorage supports SRGBA
  ColorTexImage16f,             // TexImage supports XXX16F for color targets
  ColorTexImage32f,             // TexImage supports XXX32F for color targets
  ColorTexImageBgr10A2,         // TexImage supports RGB10_A2 with format BGRA
  ColorTexImageBgr5A1,          // TexImage supports RGB5_A1 with format BGRA
  ColorTexImageBgra,            // TexImage supports BGRA
  ColorTexImageBgraRgba8,       // TexImage supports BGRA as a format with RGBA8 as an internalformat
  ColorTexImageBgraSrgba,       // TexImage supports BGRA as a format with SRGB_ALPHA or SRGB8_ALPHA8 as an internalformat
  ColorTexImageA8,              // TexImage supports ALPHA8
  ColorTexImageLa,              // TexImage supports LUMINANCE and LUMINANCE_ALPHA
  ColorTexImageLa8,             // TexImage supports LUMINANCE8 and LUMINANCE8_ALPHA8
  ColorTexImageRg8,             // TexImage supports R8 and RG8
  ColorTexImageRgb10A2,         // TexImage supports RGB10_A2 or RGBA + UNSIGNED_INT_2_10_10_10_REV
  ColorTexImageRgba8,           // TexImage supports RGB8 and RGBA8
  ColorTexImageSrgba8,          // TexImage supports SRGBA
  ColorTexStorage16f,           // TexStorage supports XXX16F for color targets
  ColorTexStorage32f,           // TexStorage supports XXX32F for color targets
  ColorTexStorageA8,            // TexStorage supports ALPHA8 (or R8 w/ swizzle)
  ColorTexStorageBgra8,         // TexStorage supports BGRA8_EXT
  ColorTexStorageLa8,           // TexStorage supports LUMINANCE8 and LUMINANCE8_ALPHA8
  ColorTexStorageRg8,           // TexStorage supports R8 and RG8
  ColorTexStorageRgb10A2,       // TexStorage supports RGB10_A2
  ColorTexStorageRgba8,         // TexStorage supports RGB8 and RGBA8
  ColorTexStorageSrgba8,        // TexStorage supports SRGB8_ALPHA8
  Depth24Stencil8,              // TextureBuffer and TextureTarget can use DEPTH24_STENCIL8
  Depth32FStencil8,             // TextureBuffer and TextureTarget can use DEPTH32F_STENCIL8
  DepthFilterable,              // Depth textures are filterable
  DepthRenderbuffer16,          // RenderbufferStorage supports DEPTH_COMPONENT16
  DepthRenderbuffer24,          // RenderbufferStorage supports DEPTH_COMPONENT24
  DepthRenderbuffer32,          // RenderbufferStorage supports DEPTH_COMPONENT32
  DepthTexImage,                // TexImage supports depth formats
  DepthTexImage16,              // TextureBuffer can use DEPTH_COMPONENT16
  DepthTexImage24,              // TextureBuffer can use DEPTH_COMPONENT24
  DepthTexImage32,              // TextureBuffer can use DEPTH_COMPONENT32
  DepthTexStorage16,            // TextureBuffer can use DEPTH_COMPONENT16
  DepthTexStorage24,            // TextureBuffer can use DEPTH_COMPONENT24
  DepthTexStorage32,            // TextureBuffer can use DEPTH_COMPONENT32
  StencilTexture8,              // TextureBuffer can use STENCIL_TEXTURE8
  TextureCompressionAstc,       // Adaptive scalable texture compression is supported
  TextureCompressionBptc,       // DirectX BPTC texture compression is supp
  TextureCompressionEtc1,       // Ericsson texture compression is supported
  TextureCompressionEtc2Eac,    // ETC2/EAC texture compression is supportedorted
  TextureCompressionPvrtc,      // PowerVR Texture compression is supported
  TextureCompressionTexImage,   // TexImage can be used to initialize compressed textures
  TextureCompressionTexStorage, // TexStorage can be used to initialize compressed textures
  TextureInteger,               // Integer textures are supported
  TextureTypeUInt8888Rev,       // GL_UNSIGNED_INT_8_8_8_8_REV is supported
};
// clang-format on

enum class InternalRequirement {
  ColorTexImageRgb10A2Unsized,
  ColorTexImageRgb5A1Unsized,
  ColorTexImageRgba4Unsized,
  ColorTexImageRgbApple422Unsized,
  DebugMessageExtReq,
  DebugMessageCallbackExtReq,
  DebugLabelExtEnumsReq,
  DebugLabelExtReq,
  Depth24Stencil8Unsized,
  Depth32Unsized,
  DrawBuffersExtReq,
  DrawBuffersIndexedExtReq,
  FramebufferBlitExtReq,
  InvalidateFramebufferExtReq,
  MapBufferExtReq,
  MapBufferRangeExtReq,
  MultiSampleExtReq,
  ShaderImageLoadStoreExtReq,
  SyncExtReq,
  SwizzleAlphaTexturesReq,
  TexStorageExtReq,
  Texture3DExtReq,
  TextureHalfFloatExtReq,
  UnmapBufferExtReq,
  VertexArrayObjectExtReq,
  VertexAttribDivisorExtReq,
};

/// GPU timer query tier. Underlying values are max query-slot counts,
/// usable directly via static_cast<uint32_t>(tier).
enum class GpuTimerTier : uint32_t {
  Disabled = 0,
  Conservative = 32,
  Full = 64,
};

/// Classify GPU timer tier from raw GL_RENDERER / GL_VENDOR strings.
/// Exposed as a free function for unit testing without a GL context.
[[nodiscard]] GpuTimerTier classifyGpuTimerTier(const char* IGL_NULLABLE renderer,
                                                const char* IGL_NULLABLE vendor);

/// Environment variable read by getGpuTimerTierDebugOverride().
inline constexpr const char* IGL_NONNULL kGpuTimerTierOverrideEnvVar =
    "IGL_GPU_TIMER_TIER_OVERRIDE";

/// Parses a timer-tier override: "full", "conservative" or "disabled". Anything else, including
/// null, is no override.
[[nodiscard]] std::optional<GpuTimerTier> parseGpuTimerTierOverride(const char* IGL_NULLABLE value);

/// The timer tier set in kGpuTimerTierOverrideEnvVar, for on-device GPU performance measurement in
/// debug builds. When set, DeviceFeatureSet::getGpuTimerTier() returns it in place of the
/// GL_RENDERER / GL_VENDOR classification, so timer queries can be re-enabled on a GPU that
/// classifyGpuTimerTier() disables. Always std::nullopt when IGL_DEBUG is 0, so release builds use
/// the classification. An environment variable is used because each shared library that links IGL
/// statically has its own copy of any static state; the process environment is shared by all of
/// them.
[[nodiscard]] std::optional<GpuTimerTier> getGpuTimerTierDebugOverride();

/// Returns true if the GPU with these raw GL_RENDERER / GL_VENDOR strings advertises
/// GL_EXT_shader_framebuffer_fetch but its driver breaks it, as the Adreno 500 series does.
/// Exposed as a free function for unit testing without a GL context.
[[nodiscard]] bool hasBrokenFramebufferFetch(const char* IGL_NULLABLE renderer,
                                             const char* IGL_NULLABLE vendor);

/// Returns true if the GPU with these raw GL_RENDERER / GL_VENDOR strings advertises
/// GL_EXT_multisampled_render_to_texture but its driver renders incorrectly with it, as the
/// Adreno 4xx, 5xx, 615 and 620 do.
/// Exposed as a free function for unit testing without a GL context.
[[nodiscard]] bool hasBrokenMultisampledRenderToTexture(const char* IGL_NULLABLE renderer,
                                                        const char* IGL_NULLABLE vendor);

/// Returns true if `extensions`, a space-separated GL_EXTENSIONS string, lists `extension` as a
/// whole token. Unlike a substring search, GL_EXT_foo does not match GL_EXT_foo_bar.
/// Exposed as a free function for unit testing without a GL context.
[[nodiscard]] bool containsExtensionToken(std::string_view extensions, std::string_view extension);

class DeviceFeatureSet final {
 public:
  explicit DeviceFeatureSet(IContext& glContext);

  [[nodiscard]] static bool usesOpenGLES() noexcept;

  void initializeVersion(GLVersion version);
  void initializeExtensions(std::string extensions,
                            std::unordered_set<std::string> supportedExtensions);

  // @fb-only
  [[nodiscard]] GLVersion getGLVersion() const noexcept;
  [[nodiscard]] ShaderVersion getShaderVersion() const;
  [[nodiscard]] BackendVersion getBackendVersion() const;

  bool isSupported(const std::string& extensionName) const;

  bool hasExtension(Extensions extension) const;
  bool hasFeature(DeviceFeatures feature) const;
  bool hasInternalFeature(InternalFeatures feature) const;
  bool hasTextureFeature(TextureFeatures feature) const;

  bool hasRequirement(DeviceRequirement requirement) const;
  bool hasInternalRequirement(InternalRequirement requirement) const;

  bool getFeatureLimits(DeviceFeatureLimits featureLimits, size_t& result) const;

  /// Returns the GPU timer tier based on GL_RENDERER / GL_VENDOR, or the debug override from
  /// getGpuTimerTierDebugOverride() when one is set. Disabled without a timer query extension.
  [[nodiscard]] GpuTimerTier getGpuTimerTier() const;

  /// Convenience: static_cast<uint32_t>(getGpuTimerTier()).
  [[nodiscard]] uint32_t getTimerQueryMaxSlots() const;

  ICapabilities::TextureFormatCapabilities getTextureFormatCapabilities(TextureFormat format) const;

  uint32_t getMaxVertexUniforms() const;
  uint32_t getMaxFragmentUniforms() const;
  uint32_t getMaxComputeUniforms() const;

 private:
  ICapabilities::TextureFormatCapabilities getCompressedTextureCapabilities() const;
  bool isExtensionSupported(Extensions extension) const;
  // Unlike isSupported(), never matches an extension whose name only starts with `extensionName`.
  bool isSupportedExact(const std::string& extensionName) const;
  bool isFeatureSupported(DeviceFeatures feature) const;
  bool isFeatureSupportedTextureGroup(DeviceFeatures feature) const;
  bool isFeatureSupportedMiscGroup(DeviceFeatures feature) const;
  bool isInternalFeatureSupported(InternalFeatures feature) const;
  bool isInternalFeatureSupportedBufferGroup(InternalFeatures feature) const;
  bool isInternalFeatureSupportedTextureGroup(InternalFeatures feature) const;
  bool isTextureFeatureSupported(TextureFeatures feature) const;
  bool isColorFilterableFeatureSupported(TextureFeatures feature) const;
  bool isColorRenderbufferFeatureSupported(TextureFeatures feature) const;
  bool isColorTexImageFeatureSupported(TextureFeatures feature) const;
  bool isColorTexStorageFeatureSupported(TextureFeatures feature) const;
  bool isColorTexStorageFloatFeatureSupported(TextureFeatures feature) const;
  bool isColorTexStorageOtherFeatureSupported(TextureFeatures feature) const;
  bool isDepthFeatureSupported(TextureFeatures feature) const;
  bool isDepthRenderbufferFeatureSupported(TextureFeatures feature) const;
  bool isDepthTexImageFeatureSupported(TextureFeatures feature) const;
  bool isCompressionFeatureSupported(TextureFeatures feature) const;
  bool hasInternalRequirementColorGroup(InternalRequirement requirement) const;
  bool hasInternalRequirementMiscGroup(InternalRequirement requirement) const;
  ICapabilities::TextureFormatCapabilities getColorUNorm8BasicTextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getColorUNorm8BgraSrgbTextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getColorUNormWideTextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getColorF16TextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getColorF32TextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getSpecialColorTextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getDepthUNorm16UNorm32TextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getDepthUNorm24TextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getStencilTextureFormatCapabilities(
      TextureFormat format) const;
  ICapabilities::TextureFormatCapabilities getCompressedTextureFormatCapabilities(
      TextureFormat format) const;

  std::unordered_set<std::string> supportedExtensions_;
  std::string extensions_;
  // The caches fill lazily on first query, and objects created on a shared context query its
  // feature set from every thread that uses them, so each cache is safe to fill concurrently. A
  // value bit is published before its initialized bit.
  mutable std::mutex textureCapabilityCacheMutex_;
  mutable std::unordered_map<TextureFormat, ICapabilities::TextureFormatCapabilities>
      textureCapabilityCache_;
  mutable std::atomic<uint64_t> extensionCache_{0};
  mutable std::atomic<uint64_t> extensionCacheInitialized_{0};
  mutable std::atomic<uint64_t> featureCache_{0};
  mutable std::atomic<uint64_t> featureCacheInitialized_{0};
  mutable std::atomic<uint32_t> internalFeatureCache_{0};
  mutable std::atomic<uint32_t> internalFeatureCacheInitialized_{0};
  mutable std::atomic<uint64_t> textureFeatureCache_{0};
  mutable std::atomic<uint64_t> textureFeatureCacheInitialized_{0};
  IContext& glContext_;
  GLVersion version_ = GLVersion::NotAvailable;
};

} // namespace igl::opengl
