/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/DeviceFeatureSet.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <igl/Common.h>
#include <igl/Macros.h>
#include <igl/opengl/GLIncludes.h>
#include <igl/opengl/IContext.h>

namespace igl::opengl {
namespace {
bool hasVersion(const DeviceFeatureSet& dfs, bool usesOpenGLES, GLVersion minSupportedVersion) {
  return DeviceFeatureSet::usesOpenGLES() == usesOpenGLES &&
         dfs.getGLVersion() >= minSupportedVersion;
}

bool hasVersionOrExtension(const DeviceFeatureSet& dfs,
                           bool usesOpenGLES,
                           GLVersion minSupportedVersion,
                           const char* extension) {
  return hasVersion(dfs, usesOpenGLES, minSupportedVersion) || dfs.isSupported(extension);
}

bool hasDesktopVersion(const DeviceFeatureSet& dfs, GLVersion minSupportedVersion) {
  return hasVersion(dfs, false, minSupportedVersion);
}

bool hasESVersion(const DeviceFeatureSet& dfs, GLVersion minSupportedVersion) {
  return hasVersion(dfs, true, minSupportedVersion);
}

bool hasESExtension(const DeviceFeatureSet& dfs, const char* extension) {
  return DeviceFeatureSet::usesOpenGLES() && dfs.isSupported(extension);
}

bool hasDesktopExtension(const DeviceFeatureSet& dfs, const char* extension) {
  return !DeviceFeatureSet::usesOpenGLES() && dfs.isSupported(extension);
}

bool hasDesktopOrESVersion(const DeviceFeatureSet& dfs,
                           GLVersion minDesktopSupportedVersion,
                           GLVersion minESSupportedVersion) {
  return hasDesktopVersion(dfs, minDesktopSupportedVersion) ||
         hasESVersion(dfs, minESSupportedVersion);
}

bool hasDesktopVersionOrExtension(const DeviceFeatureSet& dfs,
                                  GLVersion minSupportedVersion,
                                  const char* extension) {
  return hasVersionOrExtension(dfs, false, minSupportedVersion, extension);
}

bool hasESVersionOrExtension(const DeviceFeatureSet& dfs,
                             GLVersion minSupportedVersion,
                             const char* extension) {
  return hasVersionOrExtension(dfs, true, minSupportedVersion, extension);
}

bool hasDesktopOrESVersionOrExtension(const DeviceFeatureSet& dfs,
                                      GLVersion minDesktopSupportedVersion,
                                      GLVersion minESSupportedVersion,
                                      const char* desktopExtension,
                                      const char* esExtension) {
  return hasDesktopVersionOrExtension(dfs, minDesktopSupportedVersion, desktopExtension) ||
         hasESVersionOrExtension(dfs, minESSupportedVersion, esExtension);
}

bool hasDesktopOrESVersionOrExtension(const DeviceFeatureSet& dfs,
                                      GLVersion minDesktopSupportedVersion,
                                      GLVersion minESSupportedVersion,
                                      const char* extension) {
  return hasDesktopOrESVersionOrExtension(
      dfs, minDesktopSupportedVersion, minESSupportedVersion, extension, extension);
}

bool hasDesktopOrESExtension(const DeviceFeatureSet& dfs,
                             const char* desktopExtension,
                             const char* esExtension) {
  return (!DeviceFeatureSet::usesOpenGLES() && dfs.isSupported(desktopExtension)) ||
         (DeviceFeatureSet::usesOpenGLES() && dfs.isSupported(esExtension));
}

bool hasDesktopOrESExtension(const DeviceFeatureSet& dfs, const char* extension) {
  return hasDesktopOrESExtension(dfs, extension, extension);
}

// Returns true if vendor matches any of the Qualcomm vendor string variants.
bool isQualcommVendor(const char* vendor) {
  return std::strcmp(vendor, "Qualcomm") == 0 ||
         std::strcmp(vendor, "Qualcomm Technologies, Inc.") == 0 ||
         std::strcmp(vendor, "QUALCOMM") == 0;
}

// Returns true if vendor matches any of the ARM vendor string variants.
bool isArmVendor(const char* vendor) {
  return std::strcmp(vendor, "ARM") == 0 || std::strcmp(vendor, "ARM Limited") == 0 ||
         std::strcmp(vendor, "Arm") == 0;
}

// Returns true if vendor matches any of the Imagination Technologies vendor string variants.
bool isImaginationVendor(const char* vendor) {
  return std::strcmp(vendor, "Imagination Technologies") == 0 ||
         std::strcmp(vendor, "Imagination") == 0;
}
// Returns true if renderer string matches a known broken PowerVR variant.
bool isBrokenPowerVrRenderer(const char* renderer) {
  return std::strncmp(renderer, "PowerVR Rogue GE8", 17) == 0 ||
         std::strncmp(renderer, "PowerVR Rogue GX", 16) == 0 ||
         std::strncmp(renderer, "PowerVR SGX", 11) == 0;
}

// Returns the model number of an Adreno (Qualcomm) GPU, such as 530, from the renderer string.
// Returns nullopt if renderer is not a recognized Adreno string.
// Handles both "Adreno (TM) NNN" and newer "Adreno NNN" formats. Qualcomm
// ships at least three vendor string variants in Android GL drivers —
// "Qualcomm", "Qualcomm Technologies, Inc." (newer drivers), "QUALCOMM"
// (older / some OEM-tuned builds) — accept all three. Exact `strcmp`
// here previously fell any non-"Qualcomm" variant through to the
// default `Full` tier.
std::optional<int> parseAdrenoNumber(const char* renderer, const char* IGL_NULLABLE vendor) {
  if (vendor == nullptr || !isQualcommVendor(vendor)) {
    return std::nullopt;
  }
  int adrenoNumber = 0;
  if (std::sscanf(renderer, "Adreno (TM) %d", &adrenoNumber) != 1 &&
      std::sscanf(renderer, "Adreno %d", &adrenoNumber) != 1) {
    return std::nullopt;
  }
  return adrenoNumber;
}

// Classify Adreno (Qualcomm) timer tier from renderer string.
// Returns nullopt if renderer is not a recognized Adreno string.
std::optional<GpuTimerTier> classifyAdrenoTimerTier(const char* renderer, const char* vendor) {
  const std::optional<int> adrenoNumber = parseAdrenoNumber(renderer, vendor);
  if (!adrenoNumber) {
    return std::nullopt;
  }
  if (*adrenoNumber < 620) {
    return GpuTimerTier::Disabled; // 3xx–619: 0 slots
  }
  if (*adrenoNumber < 640) {
    return GpuTimerTier::Conservative; // 620, 630: 32 slots
  }
  return GpuTimerTier::Full; // 640+: 64 slots
}

// Classify Mali (ARM) timer tier from renderer string.
// Returns nullopt if renderer is not a recognized Mali string.
// Accepts the same three ARM vendor string variants ("ARM", "ARM Limited", "Arm") —
// Mali-G70+ devices on some Samsung builds report vendor as "ARM Limited" and
// would otherwise fall through to the default `Full` tier, the same failure mode
// that has crashed Mali-T drivers in production.
std::optional<GpuTimerTier> classifyMaliTimerTier(const char* renderer, const char* vendor) {
  if (renderer == nullptr || vendor == nullptr || !isArmVendor(vendor)) {
    return std::nullopt;
  }
  if (std::strncmp(renderer, "Immortalis-", 11) == 0 ||
      std::strncmp(renderer, "Mali-G1-", 8) == 0) {
    return GpuTimerTier::Disabled;
  }
  // Mali-T series: all are budget GPUs with broken timer query implementations.
  // Common models: T720, T760, T820, T830, T860, T880.
  // SEV S647462: these were not covered by the Mali-G pattern and fell through
  // to the Full tier default, causing SIGSEGV on driver-internal resource overflow.
  int maliTNumber = 0;
  if (std::sscanf(renderer, "Mali-T%d", &maliTNumber) == 1) {
    return GpuTimerTier::Disabled;
  }
  // Mali-G series: numeric range determines tier.
  int maliGNumber = 0;
  if (std::sscanf(renderer, "Mali-G%d", &maliGNumber) != 1) {
    return std::nullopt;
  }
  if (maliGNumber <= 68 || maliGNumber >= 600) {
    return GpuTimerTier::Disabled; // G31–G68, G600+: 0 slots
  }
  if (maliGNumber <= 76) {
    return GpuTimerTier::Conservative; // G72, G76: 32 slots
  }
  return GpuTimerTier::Full;
}

// Classify PowerVR (Imagination Technologies) timer tier from renderer string.
// Returns nullopt if renderer is not a recognized PowerVR string.
// Covers GE8 (Rogue, broken), Rogue GX (older, worse), and SGX (legacy, no
// hardware timer support). Newer Android drivers report vendor as
// "Imagination" (without "Technologies") — accept both spellings.
std::optional<GpuTimerTier> classifyPowerVrTimerTier(const char* renderer, const char* vendor) {
  if (vendor == nullptr || !isImaginationVendor(vendor)) {
    return std::nullopt;
  }
  if (!isBrokenPowerVrRenderer(renderer)) {
    return std::nullopt;
  }
  return GpuTimerTier::Disabled;
}
} // namespace

std::optional<GpuTimerTier> parseGpuTimerTierOverride(const char* IGL_NULLABLE value) {
  if (value == nullptr) {
    return std::nullopt;
  }
  if (std::strcmp(value, "full") == 0) {
    return GpuTimerTier::Full;
  }
  if (std::strcmp(value, "conservative") == 0) {
    return GpuTimerTier::Conservative;
  }
  if (std::strcmp(value, "disabled") == 0) {
    return GpuTimerTier::Disabled;
  }
  return std::nullopt;
}

std::optional<GpuTimerTier> getGpuTimerTierDebugOverride() {
#if IGL_DEBUG
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  return parseGpuTimerTierOverride(std::getenv(kGpuTimerTierOverrideEnvVar));
#else
  return std::nullopt;
#endif
}

GpuTimerTier classifyGpuTimerTier(const char* renderer, const char* vendor) {
  if (renderer == nullptr) {
    return GpuTimerTier::Disabled;
  }

  // Each classify*TimerTier() helper begins with `if (vendor == nullptr || ...) return nullopt;`,
  // so a null `vendor` is handled before any dereference; these call-site findings are false
  // positives.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  if (const auto tier = classifyAdrenoTimerTier(renderer, vendor)) {
    return *tier;
  }
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  if (const auto tier = classifyMaliTimerTier(renderer, vendor)) {
    return *tier;
  }
  // NOLINTNEXTLINE(facebook-hte-NullableDereference)
  if (const auto tier = classifyPowerVrTimerTier(renderer, vendor)) {
    return *tier;
  }

  // Samsung Xclipse (RDNA): SEV S647462 — Conservative (32 slots) was
  // insufficient; must be Disabled. Vendor substring covers Samsung's
  // multiple vendor string variants ("Samsung", "Samsung Electronics Co.,
  // Ltd.", "Samsung Mobile").
  if (vendor != nullptr && std::strstr(vendor, "Samsung") != nullptr &&
      std::strstr(renderer, "Xclipse") != nullptr) {
    return GpuTimerTier::Disabled;
  }

  return GpuTimerTier::Full;
}

bool hasBrokenFramebufferFetch(const char* renderer, const char* vendor) {
  if (renderer == nullptr) {
    return false;
  }
  // The Adreno 5xx drivers advertise the extension, but fetching from the framebuffer is broken
  const std::optional<int> adrenoNumber = parseAdrenoNumber(renderer, vendor);
  return adrenoNumber.has_value() && *adrenoNumber >= 500 && *adrenoNumber < 600;
}

bool hasBrokenMultisampledRenderToTexture(const char* renderer, const char* vendor) {
  if (renderer == nullptr) {
    return false;
  }
  // These Adreno drivers advertise the extension but render incorrectly with it. ANGLE disables it
  // on the Adreno 4xx and 5xx for several Android versions, and Skia disables MSAA on the Adreno
  // 615 and 620. The Android version is unknown here, so every Adreno 4xx and 5xx counts as broken.
  const std::optional<int> adrenoNumber = parseAdrenoNumber(renderer, vendor);
  return adrenoNumber.has_value() && ((*adrenoNumber >= 400 && *adrenoNumber < 600) ||
                                      *adrenoNumber == 615 || *adrenoNumber == 620);
}

bool containsExtensionToken(std::string_view extensions, std::string_view extension) {
  if (extension.empty()) {
    return false;
  }
  std::string_view remaining = extensions;
  while (!remaining.empty()) {
    const size_t tokenEnd = remaining.find(' ');
    if (remaining.substr(0, tokenEnd) == extension) {
      return true;
    }
    if (tokenEnd == std::string_view::npos) {
      break;
    }
    remaining.remove_prefix(tokenEnd + 1);
  }
  return false;
}

bool DeviceFeatureSet::usesOpenGLES() noexcept {
#if IGL_OPENGL_ES
  return true;
#else
  return false;
#endif
}

DeviceFeatureSet::DeviceFeatureSet(IContext& glContext) : glContext_(glContext) {}

void DeviceFeatureSet::initializeVersion(GLVersion version) {
  version_ = version;
}

void DeviceFeatureSet::initializeExtensions(std::string extensions,
                                            std::unordered_set<std::string> supportedExtensions) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  extensions_ = std::move(extensions);
  supportedExtensions_ = std::move(supportedExtensions);
}

GLVersion DeviceFeatureSet::getGLVersion() const noexcept {
  return version_;
}

ShaderVersion DeviceFeatureSet::getShaderVersion() const {
  return ::igl::opengl::getShaderVersion(version_);
}

BackendVersion DeviceFeatureSet::getBackendVersion() const {
  switch (version_) {
  case GLVersion::v1_1:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 1, .minorVersion = 1};
  case GLVersion::v2_0:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 2, .minorVersion = 0};
  case GLVersion::v2_1:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 2, .minorVersion = 1};
  case GLVersion::v3_0:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 3, .minorVersion = 0};
  case GLVersion::v3_1:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 3, .minorVersion = 1};
  case GLVersion::v3_2:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 3, .minorVersion = 2};
  case GLVersion::v3_3:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 3, .minorVersion = 3};
  case GLVersion::v4_0:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 4, .minorVersion = 0};
  case GLVersion::v4_1:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 4, .minorVersion = 1};
  case GLVersion::v4_2:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 4, .minorVersion = 2};
  case GLVersion::v4_3:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 4, .minorVersion = 3};
  case GLVersion::v4_4:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 4, .minorVersion = 4};
  case GLVersion::v4_5:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 4, .minorVersion = 5};
  case GLVersion::v4_6:
    return {.flavor = BackendFlavor::OpenGL, .majorVersion = 4, .minorVersion = 6};
  case GLVersion::v2_0_ES:
    return {.flavor = BackendFlavor::OpenGL_ES, .majorVersion = 2, .minorVersion = 0};
  case GLVersion::v3_0_ES:
    return {.flavor = BackendFlavor::OpenGL_ES, .majorVersion = 3, .minorVersion = 0};
  case GLVersion::v3_1_ES:
    return {.flavor = BackendFlavor::OpenGL_ES, .majorVersion = 3, .minorVersion = 1};
  case GLVersion::v3_2_ES:
    return {.flavor = BackendFlavor::OpenGL_ES, .majorVersion = 3, .minorVersion = 2};
  case GLVersion::NotAvailable:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return {.flavor = usesOpenGLES() ? BackendFlavor::OpenGL_ES : BackendFlavor::OpenGL,
            .majorVersion = 2,
            .minorVersion = 0};
  }
  IGL_UNREACHABLE_RETURN({});
}

bool DeviceFeatureSet::isSupported(const std::string& extensionName) const {
  if (!extensions_.empty()) {
    return extensions_.find(extensionName) != std::string::npos;
  } else {
    return supportedExtensions_.find(extensionName) != supportedExtensions_.end();
  }
}

bool DeviceFeatureSet::isSupportedExact(const std::string& extensionName) const {
  if (!extensions_.empty()) {
    return containsExtensionToken(extensions_, extensionName);
  }
  return supportedExtensions_.find(extensionName) != supportedExtensions_.end();
}

bool DeviceFeatureSet::isExtensionSupported(Extensions extension) const {
  switch (extension) {
  case Extensions::AppleRgb422:
    return hasDesktopOrESExtension(*this, "GL_APPLE_rgb_422");
  case Extensions::BindlessTextureArb:
    return hasDesktopExtension(*this, "GL_ARB_bindless_texture");
  case Extensions::BindlessTextureNv:
    return hasDesktopOrESExtension(*this, "GL_NV_bindless_texture");
  case Extensions::Debug:
    return hasDesktopOrESExtension(*this, "GL_KHR_debug");
  case Extensions::DebugLabel:
    return hasDesktopOrESExtension(*this, "GL_EXT_debug_label");
  case Extensions::DebugMarker:
    return hasDesktopOrESExtension(*this, "GL_EXT_debug_marker");
  case Extensions::Depth24:
    return hasESExtension(*this, "GL_OES_depth24");
  case Extensions::Depth32:
    return hasESExtension(*this, "GL_OES_depth32");
  case Extensions::DepthTexture:
    return hasESExtension(*this, "GL_OES_depth_texture");
  case Extensions::DiscardFramebuffer:
    return hasESExtension(*this, "GL_EXT_discard_framebuffer");
  case Extensions::DrawBuffers:
    return hasESExtension(*this, "GL_EXT_draw_buffers");
  case Extensions::DrawBuffersIndexedExt:
    return hasESExtension(*this, "GL_EXT_draw_buffers_indexed");
  case Extensions::DrawBuffersIndexedOes:
    return hasESExtension(*this, "GL_OES_draw_buffers_indexed");
  case Extensions::Es2Compatibility:
    return hasDesktopExtension(*this, "GL_ARB_ES2_compatibility");
  case Extensions::FramebufferBlit:
    return hasDesktopExtension(*this, "GL_EXT_framebuffer_blit");
  case Extensions::FramebufferObject:
    return hasDesktopExtension(*this, "GL_ARB_framebuffer_object");
  case Extensions::InvalidateSubdata:
    return isSupported("GL_ARB_invalidate_subdata");
  case Extensions::MapBuffer:
    return hasESExtension(*this, "GL_OES_mapbuffer");
  case Extensions::MapBufferRange:
    return hasESExtension(*this, "GL_EXT_map_buffer_range");
  case Extensions::MultiSampleApple:
    return hasESExtension(*this, "GL_APPLE_framebuffer_multisample");
  case Extensions::MultiSampleExt:
    // This name is a prefix of GL_EXT_multisampled_render_to_texture2, so match it as a whole
    return usesOpenGLES() && isSupportedExact("GL_EXT_multisampled_render_to_texture");
  case Extensions::MultiSampleExt2:
    return hasESExtension(*this, "GL_EXT_multisampled_render_to_texture2");
  case Extensions::MultiSampleImg:
    return hasESExtension(*this, "GL_IMG_multisampled_render_to_texture");
  case Extensions::MultiViewMultiSample:
    return hasESExtension(*this, "GL_OVR_multiview_multisampled_render_to_texture");
  case Extensions::PolygonOffsetClamp:
    return hasDesktopOrESExtension(*this, "GL_ARB_polygon_offset_clamp");
  case Extensions::RequiredInternalFormat:
    return hasESExtension(*this, "GL_OES_required_internalformat");
  case Extensions::ShaderFramebufferFetch:
    // This name is a prefix of GL_EXT_shader_framebuffer_fetch_non_coherent, whose reads are not
    // coherent with earlier writes
    return isSupportedExact("GL_EXT_shader_framebuffer_fetch");
  case Extensions::ShaderImageLoadStore:
    return hasESExtension(*this, "GL_EXT_shader_image_load_store");
  case Extensions::Srgb:
    return hasESExtension(*this, "GL_EXT_sRGB");
  case Extensions::SrgbWriteControl:
    return hasESExtension(*this, "GL_EXT_sRGB_write_control");
  case Extensions::Sync:
    return hasESExtension(*this, "GL_APPLE_sync");
  case Extensions::TexStorage:
    return isSupported("GL_EXT_texture_storage");
  case Extensions::Texture3D:
    return hasESExtension(*this, "GL_OES_texture_3D");
  case Extensions::TextureFormatBgra8888Ext:
    return hasESExtension(*this, "GL_EXT_texture_format_BGRA8888");
  case Extensions::TextureFormatBgra8888Apple:
    return hasESExtension(*this, "GL_APPLE_texture_format_BGRA8888");
  case Extensions::TextureFloat:
    return hasDesktopExtension(*this, "GL_ARB_texture_float");
  case Extensions::TextureHalfFloat:
    // Necessary for GL_HALF_FLOAT_OES, which is different than GL_HALF_FLOAT
    return hasESExtension(*this, "GL_OES_texture_half_float");
  case Extensions::TextureRgArb:
    return hasDesktopExtension(*this, "GL_ARB_texture_rg");
  case Extensions::TextureRgExt:
    return hasESExtension(*this, "GL_EXT_texture_rg");
  case Extensions::TextureSrgb:
    return hasDesktopExtension(*this, "GL_EXT_texture_sRGB");
  case Extensions::TextureType2101010Rev:
    return hasESExtension(*this, "GL_EXT_texture_type_2_10_10_10_REV");
  case Extensions::TimerQuery:
    return hasDesktopOrESExtension(*this, "GL_ARB_timer_query", "GL_EXT_disjoint_timer_query");
  case Extensions::VertexArrayObject:
    return hasESExtension(*this, "GL_OES_vertex_array_object");
  case Extensions::VertexAttribDivisor:
    return hasESExtension(*this, "GL_NV_instanced_arrays");
  }
  IGL_UNREACHABLE_RETURN(false)
}

GpuTimerTier DeviceFeatureSet::getGpuTimerTier() const {
  IGL_PROFILER_FUNCTION();
  if (!hasExtension(Extensions::TimerQuery)) {
    return GpuTimerTier::Disabled;
  }
  if (const auto tierOverride = getGpuTimerTierDebugOverride()) {
    return *tierOverride;
  }
  const auto* renderer = reinterpret_cast<const char*>(glContext_.getString(GL_RENDERER));
  const auto* vendor = reinterpret_cast<const char*>(glContext_.getString(GL_VENDOR));
  return classifyGpuTimerTier(renderer, vendor);
}

uint32_t DeviceFeatureSet::getTimerQueryMaxSlots() const {
  return static_cast<uint32_t>(getGpuTimerTier());
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isFeatureSupportedTextureGroup(DeviceFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case DeviceFeatures::TextureFilterAnisotropic:
    return hasDesktopVersion(*this, GLVersion::v4_6) ||
           hasDesktopOrESExtension(*this, "GL_EXT_texture_filter_anisotropic") ||
           hasDesktopExtension(*this, "GL_ARB_texture_filter_anisotropic");

  case DeviceFeatures::TextureFormatRG:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureRgArb) || hasExtension(Extensions::TextureRgExt);

  case DeviceFeatures::TextureFormatRGB:
    return true;

  case DeviceFeatures::TextureNotPot:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v2_0, GLVersion::v3_0_ES, "GL_OES_texture_npot");

  case DeviceFeatures::TextureHalfFloat:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureFloat) || hasExtension(Extensions::TextureHalfFloat);

  case DeviceFeatures::TextureFloat:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureFloat) || hasESExtension(*this, "GL_OES_texture_float");

  case DeviceFeatures::Texture2DArray:
    return (hasDesktopOrESVersionOrExtension(
                *this, GLVersion::v3_0, GLVersion::v3_0_ES, "GL_EXT_texture_array") ||
            hasDesktopExtension(*this, "GL_EXT_gpu_shader4"));

  case DeviceFeatures::Texture3D:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v2_0, GLVersion::v3_0_ES, "GL_OES_texture_3D");

  case DeviceFeatures::TextureArrayExt:
    return hasDesktopExtension(*this, "GL_EXT_texture_array") ||
           hasDesktopExtension(*this, "GL_EXT_gpu_shader4");

  case DeviceFeatures::ShaderTextureLod:
    return hasDesktopOrESVersionOrExtension(*this,
                                            GLVersion::v3_0,
                                            GLVersion::v3_0_ES,
                                            "GL_ARB_shader_texture_lod",
                                            "GL_EXT_shader_texture_lod");

  case DeviceFeatures::ShaderTextureLodExt:
    return hasDesktopOrESExtension(*this, "GL_ARB_shader_texture_lod", "GL_EXT_shader_texture_lod");

  case DeviceFeatures::TextureExternalImage:
    return hasESVersionOrExtension(*this, GLVersion::v3_0_ES, "GL_OES_EGL_image_external_essl3") ||
           hasESExtension(*this, "GL_OES_EGL_image_external");

  case DeviceFeatures::TextureBindless:
    return hasDesktopExtension(*this, "GL_ARB_bindless_texture");

  case DeviceFeatures::TextureViews:
    return false;

  case DeviceFeatures::TexturePartialMipChain:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasESExtension(*this, "GL_APPLE_texture_max_level");

  case DeviceFeatures::SRGB:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v2_1, GLVersion::v3_0_ES, "GL_EXT_texture_sRGB", "GL_EXT_sRGB");
  case DeviceFeatures::SRGBSwapchain:
    return glContext_.eglSupportssRGB() && hasFeature(DeviceFeatures::SRGB);
  case DeviceFeatures::SRGBWriteControl:
    return hasDesktopVersion(*this, GLVersion::v3_0) ||
           hasDesktopExtension(*this, "GL_ARB_framebuffer_sRGB") ||
           hasDesktopExtension(*this, "GL_EXT_framebuffer_sRGB") ||
           hasESExtension(*this, "GL_EXT_sRGB_write_control");

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isFeatureSupportedMiscGroup(DeviceFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case DeviceFeatures::MultiSample:
    return hasDesktopVersion(*this, GLVersion::v3_0) ||
           hasExtension(Extensions::FramebufferObject) || hasESVersion(*this, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::MultiSampleApple) || hasExtension(Extensions::MultiSampleExt) ||
           hasExtension(Extensions::MultiSampleImg);

  case DeviceFeatures::MapBufferRange:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasDesktopExtension(*this, "GL_ARB_map_buffer_range") ||
           hasExtension(Extensions::MapBufferRange);

  case DeviceFeatures::MultipleRenderTargets:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v2_0, GLVersion::v3_0_ES, "GL_EXT_draw_buffers");

  case DeviceFeatures::StandardDerivative:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v2_0, GLVersion::v3_0_ES, "GL_OES_standard_derivatives");

  case DeviceFeatures::StandardDerivativeExt:
    return hasESExtension(*this, "GL_OES_standard_derivatives");

  case DeviceFeatures::ReadWriteFramebuffer:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::FramebufferObject) ||
           hasESExtension(*this, "GL_APPLE_framebuffer_multisample");

  case DeviceFeatures::UniformBlocks:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v3_1, GLVersion::v3_0_ES, "GL_ARB_uniform_buffer_object");

  case DeviceFeatures::DepthShaderRead:
    // Currently it is unclear if Depth Shader Read is the same as ARB_depth_texture extension so
    // we are using v2.1 because we know it works on the Mac.
    return hasDesktopOrESVersion(*this, GLVersion::v2_1, GLVersion::v3_0_ES);

  case DeviceFeatures::DepthCompare:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES);

  case DeviceFeatures::MinMaxBlend:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v2_0, GLVersion::v3_0_ES, "GL_EXT_blend_minmax");

  case DeviceFeatures::Compute:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_1_ES) ||
           (hasDesktopExtension(*this, "GL_ARB_compute_shader") &&
            hasInternalFeature(InternalFeatures::ProgramInterfaceQuery) &&
            hasInternalFeature(InternalFeatures::ShaderImageLoadStore));

  case DeviceFeatures::ExplicitBinding:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v4_2, GLVersion::v3_1_ES, "GL_ARB_shading_language_420pack");

  case DeviceFeatures::ExplicitBindingExt:
    return hasDesktopExtension(*this, "GL_ARB_shading_language_420pack");

  case DeviceFeatures::ExternalMemoryObjects:
    return hasDesktopOrESExtension(*this, "GL_EXT_memory_object") &&
           hasDesktopOrESExtension(*this, "GL_EXT_memory_object_fd");

  case DeviceFeatures::Multiview:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) &&
           isSupported("GL_OVR_multiview2");

  case DeviceFeatures::MultiViewMultisample:
    return hasExtension(Extensions::MultiViewMultiSample);

  case DeviceFeatures::StorageBuffers:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_1_ES) ||
           hasDesktopExtension(*this, "GL_ARB_shader_storage_buffer_object");

  case DeviceFeatures::SamplerMinMaxLod:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES);

  case DeviceFeatures::DrawFirstIndexFirstVertex:
    // https://registry.khronos.org/OpenGL-Refpages/es3/html/glDrawElementsInstancedBaseVertex.xhtml
    return hasDesktopOrESVersion(*this, GLVersion::v4_0, GLVersion::v3_2_ES);

  case DeviceFeatures::DrawIndexedIndirect:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v4_0, GLVersion::v3_1_ES, "GL_ARB_draw_indirect");

  case DeviceFeatures::DrawInstanced:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v3_1, GLVersion::v3_0_ES, "GL_ARB_draw_indirect");

  case DeviceFeatures::TimestampQueries:
    return getGpuTimerTier() != GpuTimerTier::Disabled;

  case DeviceFeatures::Timers:
    return hasExtension(Extensions::TimerQuery);

  case DeviceFeatures::Subpasses: {
    // Framebuffer fetch lets a subpass read the attachments that earlier subpasses wrote, and
    // indexed color masks and blending give each attachment its own state. Desktop OpenGL needs
    // 4.0 for the explicit fragment output locations of GLSL 4.00 and for the per-draw-buffer
    // blend functions.
    if (!hasDesktopOrESVersion(*this, GLVersion::v4_0, GLVersion::v3_0_ES) ||
        !hasExtension(Extensions::ShaderFramebufferFetch) ||
        !hasInternalFeature(InternalFeatures::DrawBuffersIndexed)) {
      return false;
    }
    const auto* renderer = reinterpret_cast<const char*>(glContext_.getString(GL_RENDERER));
    const auto* vendor = reinterpret_cast<const char*>(glContext_.getString(GL_VENDOR));
    return !hasBrokenFramebufferFetch(renderer, vendor);
  }

  case DeviceFeatures::DynamicCullMode:
  case DeviceFeatures::DynamicFrontFacingWinding:
  case DeviceFeatures::DynamicVertexBufferStride:
    return true;

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isFeatureSupported(DeviceFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case DeviceFeatures::CopyBuffer:
  case DeviceFeatures::FillBuffer:
  case DeviceFeatures::UpdateBuffer:
  case DeviceFeatures::MultiSampleResolve:
  case DeviceFeatures::MeshShaders:
  case DeviceFeatures::TextureViews:
  case DeviceFeatures::PushConstants:
  case DeviceFeatures::BufferDeviceAddress:
  case DeviceFeatures::BufferRing:
  case DeviceFeatures::BufferNoCopy:
  case DeviceFeatures::ShaderLibrary:
  case DeviceFeatures::BindBytes:
  case DeviceFeatures::ValidationLayersEnabled:
  case DeviceFeatures::MemorylessAttachments:
  case DeviceFeatures::MultiviewSubpasses:
    return false;

  case DeviceFeatures::BindUniform:
  case DeviceFeatures::Indices8Bit:
    return true;

  case DeviceFeatures::TextureFilterAnisotropic:
  case DeviceFeatures::TextureFormatRG:
  case DeviceFeatures::TextureFormatRGB:
  case DeviceFeatures::TextureNotPot:
  case DeviceFeatures::TextureHalfFloat:
  case DeviceFeatures::TextureFloat:
  case DeviceFeatures::Texture2DArray:
  case DeviceFeatures::Texture3D:
  case DeviceFeatures::TextureArrayExt:
  case DeviceFeatures::ShaderTextureLod:
  case DeviceFeatures::ShaderTextureLodExt:
  case DeviceFeatures::TextureExternalImage:
  case DeviceFeatures::TextureBindless:
  case DeviceFeatures::TexturePartialMipChain:
  case DeviceFeatures::SRGB:
  case DeviceFeatures::SRGBSwapchain:
  case DeviceFeatures::SRGBWriteControl:
    return isFeatureSupportedTextureGroup(feature);

  default:
    return isFeatureSupportedMiscGroup(feature);
  }
}

bool DeviceFeatureSet::isInternalFeatureSupportedBufferGroup(InternalFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case InternalFeatures::DrawArraysIndirect:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v4_0, GLVersion::v3_1_ES, "GL_ARB_draw_indirect");

  case InternalFeatures::MultiDrawIndirect:
    return hasDesktopVersionOrExtension(*this, GLVersion::v4_3, "GL_ARB_multi_draw_indirect") ||
           hasESExtension(*this, "GL_EXT_multi_draw_indirect");

  case InternalFeatures::MapBuffer:
    return hasDesktopVersion(*this, GLVersion::v2_0) || hasExtension(Extensions::MapBuffer);

  case InternalFeatures::PackRowLength:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES);

  case InternalFeatures::PixelBufferObject:
    return hasDesktopOrESVersionOrExtension(*this,
                                            GLVersion::v2_1,
                                            GLVersion::v3_0_ES,
                                            "GL_ARB_pixel_buffer_object",
                                            "GL_NV_pixel_buffer_object");

  case InternalFeatures::UnmapBuffer:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::MapBuffer) || hasExtension(Extensions::MapBufferRange);

  case InternalFeatures::UnpackRowLength:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasESExtension(*this, "GL_EXT_unpack_subimage");

  case InternalFeatures::VertexArrayObject:
    // We've had issues with VertexArrayObject support on mobile so this is disabled for OpenGL ES.
    // Previously it was enabled specifically for Quest 2 on OpenGLES by checking if
    // GL_VENDOR == "Qualcomm" and GL_RENDERER == "Adreno (TM) 650".
    // However, Galaxy S20 also matched that and VAO support caused issues.
    // @fb-only
    // @fb-only
    return hasDesktopVersionOrExtension(*this, GLVersion::v3_0, "GL_ARB_vertex_array_object");

  case InternalFeatures::VertexAttribDivisor:
    return hasDesktopOrESVersion(*this, GLVersion::v3_3, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::VertexAttribDivisor);

  default:
    return false;
  }
}

bool DeviceFeatureSet::isInternalFeatureSupportedTextureGroup(InternalFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case InternalFeatures::TexStorage:
    return hasDesktopOrESVersionOrExtension(
               *this, GLVersion::v4_2, GLVersion::v3_0_ES, "GL_ARB_texture_storage") ||
           hasExtension(Extensions::TexStorage);

  case InternalFeatures::ShaderImageLoadStore:
    return hasDesktopOrESVersion(*this, GLVersion::v4_2, GLVersion::v3_1_ES) ||
           hasDesktopExtension(*this, "GL_ARB_shader_image_load_store") ||
           hasExtension(Extensions::ShaderImageLoadStore);

  case InternalFeatures::TextureClampToBorder:
    return hasDesktopOrESVersionOrExtension(*this,
                                            GLVersion::v2_0,
                                            GLVersion::v3_2_ES,
                                            "GL_ARB_texture_border_clamp",
                                            "GL_EXT_texture_border_clamp");

  case InternalFeatures::TextureCompare:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasESExtension(*this, "GL_EXT_shadow_samplers");

  default:
    return false;
  }
}

bool DeviceFeatureSet::isInternalFeatureSupported(InternalFeatures feature) const {
  switch (feature) {
  case InternalFeatures::ClearBufferfv:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES);

  case InternalFeatures::ClearDepthf:
    return hasDesktopOrESVersion(*this, GLVersion::v4_1, GLVersion::v2_0_ES);

  case InternalFeatures::DebugLabel:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_2_ES) ||
           hasExtension(Extensions::Debug) || hasExtension(Extensions::DebugLabel);

  case InternalFeatures::DebugMessage:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_2_ES) ||
           hasExtension(Extensions::Debug) || hasExtension(Extensions::DebugMarker);

  case InternalFeatures::DebugMessageCallback:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_2_ES) ||
           hasExtension(Extensions::Debug);

  case InternalFeatures::DrawBuffersIndexed:
    // The functions for the draw buffer state are core from OpenGL 4.0 and OpenGL ES 3.2
    return hasDesktopOrESVersion(*this, GLVersion::v4_0, GLVersion::v3_2_ES) ||
           hasExtension(Extensions::DrawBuffersIndexedOes) ||
           hasExtension(Extensions::DrawBuffersIndexedExt);

  case InternalFeatures::FramebufferBlit:
    // TODO: Add support for GL_ANGLE_framebuffer_blit
    return hasDesktopOrESVersionOrExtension(
               *this, GLVersion::v3_0, GLVersion::v3_0_ES, "GL_EXT_framebuffer_blit") ||
           hasExtension(Extensions::FramebufferObject);

  case InternalFeatures::FramebufferObject:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v2_0_ES) ||
           hasExtension(Extensions::FramebufferObject);

  case InternalFeatures::GetStringi:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES);

  case InternalFeatures::InternalFormatQuery:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v4_2, GLVersion::v3_0_ES, "GL_ARB_internalformat_query");

  case InternalFeatures::InvalidateFramebuffer:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::InvalidateSubdata) ||
           hasExtension(Extensions::DiscardFramebuffer);

  case InternalFeatures::MultisampledRenderToTexture: {
    if (!usesOpenGLES() ||
        (!hasExtension(Extensions::MultiSampleExt) && !hasExtension(Extensions::MultiSampleImg))) {
      return false;
    }
    const auto* renderer = reinterpret_cast<const char*>(glContext_.getString(GL_RENDERER));
    const auto* vendor = reinterpret_cast<const char*>(glContext_.getString(GL_VENDOR));
    return !hasBrokenMultisampledRenderToTexture(renderer, vendor);
  }

  case InternalFeatures::PolygonFillMode:
    return hasDesktopVersion(*this, GLVersion::v2_0);

  case InternalFeatures::ProgramInterfaceQuery:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_1_ES) ||
           hasDesktopExtension(*this, "GL_ARB_program_interface_query");

  case InternalFeatures::SeamlessCubeMap:
    return hasDesktopVersionOrExtension(*this, GLVersion::v3_2, "GL_ARB_seamless_cube_map");

  case InternalFeatures::Sync:
    return hasDesktopOrESVersion(*this, GLVersion::v3_2, GLVersion::v3_0_ES) ||
           hasDesktopExtension(*this, "GL_ARB_sync") || hasExtension(Extensions::Sync);

  case InternalFeatures::DrawArraysIndirect:
  case InternalFeatures::MultiDrawIndirect:
  case InternalFeatures::MapBuffer:
  case InternalFeatures::PackRowLength:
  case InternalFeatures::PixelBufferObject:
  case InternalFeatures::UnmapBuffer:
  case InternalFeatures::UnpackRowLength:
  case InternalFeatures::VertexArrayObject:
  case InternalFeatures::VertexAttribDivisor:
    return isInternalFeatureSupportedBufferGroup(feature);

  case InternalFeatures::TexStorage:
  case InternalFeatures::ShaderImageLoadStore:
  case InternalFeatures::TextureClampToBorder:
  case InternalFeatures::TextureCompare:
    return isInternalFeatureSupportedTextureGroup(feature);
  }

  return false;
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isColorFilterableFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::ColorFilterable16f:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureFloat) ||
           hasESExtension(*this, "GL_OES_texture_half_float_linear");

  case TextureFeatures::ColorFilterable32f:
    return hasDesktopVersion(*this, GLVersion::v3_0) || hasExtension(Extensions::TextureFloat) ||
           hasESExtension(*this, "GL_OES_texture_float_linear");

  case TextureFeatures::ColorFormatRgb10A2UI:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v4_0, GLVersion::v3_0_ES, "GL_ARB_texture_rgb10_a2ui");

  case TextureFeatures::ColorFormatRgInt:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasDesktopExtension(*this, "GL_ARB_texture_rg");

  case TextureFeatures::ColorFormatRgUNorm16:
    return hasDesktopVersionOrExtension(*this, GLVersion::v3_0, "GL_ARB_texture_rg") ||
           hasESExtension(*this, "GL_EXT_texture_norm16");

  case TextureFeatures::ColorFormatRgbaUNorm16:
    return hasDesktopVersion(*this, GLVersion::v3_0) ||
           hasESExtension(*this, "GL_EXT_texture_norm16");

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isColorRenderbufferFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::ColorRenderbuffer16f:
    return hasDesktopOrESVersionOrExtension(
               *this, GLVersion::v3_0, GLVersion::v3_2_ES, "GL_EXT_color_buffer_half_float") ||
           (hasExtension(Extensions::FramebufferObject) && hasExtension(Extensions::TextureFloat));

  case TextureFeatures::ColorRenderbuffer32f:
    return hasDesktopOrESVersionOrExtension(
               *this, GLVersion::v3_0, GLVersion::v3_2_ES, "GL_EXT_color_buffer_float") ||
           (hasExtension(Extensions::FramebufferObject) && hasExtension(Extensions::TextureFloat));

  case TextureFeatures::ColorRenderbufferRg16f:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_2_ES) ||
           hasESExtension(*this, "GL_EXT_color_buffer_float") ||
           hasESExtension(*this, "GL_EXT_color_buffer_half_float");

  case TextureFeatures::ColorRenderbufferRg32f:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_2_ES) ||
           hasESExtension(*this, "GL_EXT_color_buffer_float");

  case TextureFeatures::ColorRenderbufferRg8:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           (hasExtension(Extensions::FramebufferObject) &&
            hasExtension(Extensions::TextureRgArb)) ||
           hasExtension(Extensions::TextureRgExt);

  case TextureFeatures::ColorRenderbufferRgb10A2:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::RequiredInternalFormat);

  case TextureFeatures::ColorRenderbufferRgb16f:
    return hasESExtension(*this, "GL_EXT_color_buffer_half_float");

  case TextureFeatures::ColorRenderbufferRgba8:
    return hasDesktopOrESVersionOrExtension(
               *this, GLVersion::v3_0, GLVersion::v3_0_ES, "GL_OES_rgb8_rgba8") ||
           hasExtension(Extensions::FramebufferObject) ||
           hasExtension(Extensions::RequiredInternalFormat);

  case TextureFeatures::ColorRenderbufferSrgba8:
    return hasDesktopOrESVersion(*this, GLVersion::v2_1, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::Srgb);

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isColorTexImageFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::ColorTexImage16f:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureFloat);

  case TextureFeatures::ColorTexImage32f:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureFloat);

  case TextureFeatures::ColorTexImageA8:
    // Sized alpha texture were available on Desktop OpenGL prior to deprecation in
    // Version 3.0. For later versions of OpenGL, we create GL_R8 textures and use texture
    // swizzling. Sized alpha textures are only available on OpenGL ES through extensions.
    return (hasDesktopVersion(*this, GLVersion::v2_0) &&
            !hasDesktopVersion(*this, GLVersion::v3_0)) ||
           (hasDesktopVersion(*this, GLVersion::v3_0) &&
            hasTextureFeature(TextureFeatures::ColorTexImageRg8)) ||
           hasExtension(Extensions::RequiredInternalFormat);

  case TextureFeatures::ColorTexImageBgr10A2:
    return hasDesktopVersion(*this, GLVersion::v2_0);

  case TextureFeatures::ColorTexImageBgr5A1:
    // There's no OpenGL ES extension that specifically enables support for this, but Apple
    // platforms support it.
    return !usesOpenGLES() || hasExtension(Extensions::TextureFormatBgra8888Apple);

  case TextureFeatures::ColorTexImageBgra:
    return !usesOpenGLES() || hasExtension(Extensions::TextureFormatBgra8888Ext) ||
           hasExtension(Extensions::TextureFormatBgra8888Apple);

  case TextureFeatures::ColorTexImageBgraRgba8:
    return hasDesktopVersion(*this, GLVersion::v2_0);

  case TextureFeatures::ColorTexImageBgraSrgba:
    // There's no OpenGL ES extension that specifically enables support for this, but Apple
    // platforms support it.
    return !usesOpenGLES() || (hasESVersion(*this, GLVersion::v3_0_ES) &&
                               hasExtension(Extensions::TextureFormatBgra8888Apple));

  case TextureFeatures::ColorTexImageLa:
    // LUMINANCE and LUMINANCE_ALPHA were deprecated in Desktop OpenGL 3.0, and we don't use any
    // work arounds for support after that.
    return !hasDesktopVersion(*this, GLVersion::v3_0);

  case TextureFeatures::ColorTexImageLa8:
    // Sized luminance and luminance alpha texture were available on Desktop OpenGL prior to
    // deprecation in Version 3.0. Sized luminance alpha textures are only available on OpenGL ES
    // through extensions.
    return (hasDesktopVersion(*this, GLVersion::v2_0) &&
            !hasDesktopVersion(*this, GLVersion::v3_0)) ||
           hasExtension(Extensions::RequiredInternalFormat);

  case TextureFeatures::ColorTexImageRg8:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureRgArb);

  case TextureFeatures::ColorTexImageRgb10A2:
    return hasTextureFeature(TextureFeatures::ColorRenderbufferRgb10A2) ||
           hasExtension(Extensions::TextureType2101010Rev);

  case TextureFeatures::ColorTexImageRgba8:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::RequiredInternalFormat);

  case TextureFeatures::ColorTexImageSrgba8:
    return hasDesktopOrESVersion(*this, GLVersion::v2_1, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::TextureSrgb);

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isColorTexStorageFloatFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::ColorTexStorage16f:
    return hasDesktopOrESVersion(*this, GLVersion::v4_2, GLVersion::v3_0_ES) ||
           ((hasFeature(DeviceFeatures::TextureHalfFloat) ||
             hasTextureFeature(TextureFeatures::ColorRenderbuffer16f)) &&
            hasInternalFeature(InternalFeatures::TexStorage));

  case TextureFeatures::ColorTexStorage32f:
    return hasDesktopOrESVersion(*this, GLVersion::v4_2, GLVersion::v3_0_ES) ||
           ((hasFeature(DeviceFeatures::TextureFloat) ||
             hasTextureFeature(TextureFeatures::ColorRenderbuffer32f)) &&
            hasInternalFeature(InternalFeatures::TexStorage));

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isColorTexStorageOtherFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::ColorTexStorageA8:
    // Sized alpha texture were available on Desktop OpenGL prior to deprecation in
    // Version 3.0. For later versions of OpenGL, we create GL_R8 textures and use texture
    // swizzling. Sized alpha textures are only available on OpenGL ES through extensions.
    return (hasDesktopVersion(*this, GLVersion::v3_0) &&
            hasTextureFeature(TextureFeatures::ColorTexStorageRg8)) ||
           hasExtension(Extensions::TexStorage);

  case TextureFeatures::ColorTexStorageBgra8:
    // TexStorage is explicitly supported when available by GL_APPLE_texture_format_BGRA8888
    // TexStorage for GL_EXT_texture_format_BGRA8888 is added by GL_EXT_texture_storage
    return (hasExtension(Extensions::TextureFormatBgra8888Apple) &&
            (hasESVersion(*this, GLVersion::v3_0_ES) ||
             hasInternalFeature(InternalFeatures::TexStorage))) ||
           (hasExtension(Extensions::TextureFormatBgra8888Ext) &&
            hasExtension(Extensions::TexStorage));

  case TextureFeatures::ColorTexStorageLa8:
    // TexStorage with sized luminance alpha formats is only supported with GL_EXT_texture_storage
    return hasExtension(Extensions::TexStorage);

  case TextureFeatures::ColorTexStorageRg8:
    return hasDesktopOrESVersion(*this, GLVersion::v4_2, GLVersion::v3_0_ES) ||
           (hasExtension(Extensions::TexStorage) && hasExtension(Extensions::TextureRgExt));

  case TextureFeatures::ColorTexStorageRgb10A2:
    return hasDesktopOrESVersion(*this, GLVersion::v4_2, GLVersion::v3_0_ES) ||
           (hasExtension(Extensions::TexStorage) &&
            hasExtension(Extensions::TextureType2101010Rev));

  case TextureFeatures::ColorTexStorageRgba8:
    return hasTextureFeature(TextureFeatures::ColorRenderbufferRgba8) &&
           hasInternalFeature(InternalFeatures::TexStorage);

  case TextureFeatures::ColorTexStorageSrgba8:
    // NOTE: GL_EXT_texture_storage does NOT support GL_SRGB8_ALPHA8.
    return hasFeature(DeviceFeatures::SRGB) && hasInternalFeature(InternalFeatures::TexStorage) &&
           !(hasInternalRequirement(InternalRequirement::TexStorageExtReq) &&
             hasExtension(Extensions::TexStorage));

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isColorTexStorageFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::ColorTexStorage16f:
  case TextureFeatures::ColorTexStorage32f:
    return isColorTexStorageFloatFeatureSupported(feature);

  case TextureFeatures::ColorTexStorageA8:
  case TextureFeatures::ColorTexStorageBgra8:
  case TextureFeatures::ColorTexStorageLa8:
  case TextureFeatures::ColorTexStorageRg8:
  case TextureFeatures::ColorTexStorageRgb10A2:
  case TextureFeatures::ColorTexStorageRgba8:
  case TextureFeatures::ColorTexStorageSrgba8:
    return isColorTexStorageOtherFeatureSupported(feature);

  default:
    return false;
  }
}

bool DeviceFeatureSet::isDepthRenderbufferFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::DepthFilterable:
    return hasDesktopVersion(*this, GLVersion::v2_0);

  case TextureFeatures::DepthRenderbuffer16:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v2_0_ES) ||
           hasExtension(Extensions::FramebufferObject);

  case TextureFeatures::DepthRenderbuffer24:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::FramebufferObject) || hasExtension(Extensions::Depth24);

  case TextureFeatures::DepthRenderbuffer32:
    // 32-bit integer depth textures are only supported on ES through specific extensions.
    return hasDesktopVersion(*this, GLVersion::v3_0) ||
           hasExtension(Extensions::FramebufferObject) || hasExtension(Extensions::Depth32);

  case TextureFeatures::Depth24Stencil8:
    return hasDesktopOrESVersionOrExtension(*this,
                                            GLVersion::v3_0,
                                            GLVersion::v3_0_ES,
                                            "GL_EXT_packed_depth_stencil",
                                            "GL_OES_packed_depth_stencil") ||
           hasExtension(Extensions::FramebufferObject);

  case TextureFeatures::Depth32FStencil8:
    return hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v3_0, GLVersion::v3_0_ES, "GL_ARB_depth_buffer_float");

  case TextureFeatures::StencilTexture8:
    return hasDesktopOrESVersionOrExtension(*this,
                                            GLVersion::v4_4,
                                            GLVersion::v3_2_ES,
                                            "GL_ARB_texture_stencil8",
                                            "GL_OES_texture_stencil8");

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isDepthTexImageFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::DepthTexImage:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           hasExtension(Extensions::DepthTexture);

  case TextureFeatures::DepthTexImage16:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           (hasTextureFeature(TextureFeatures::DepthTexImage) &&
            hasExtension(Extensions::RequiredInternalFormat));

  case TextureFeatures::DepthTexImage24:
    return hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES) ||
           (hasExtension(Extensions::Depth24) && hasExtension(Extensions::RequiredInternalFormat));

  case TextureFeatures::DepthTexImage32:
    // 32-bit integer depth textures are only supported on ES through specific extensions.
    return hasDesktopVersion(*this, GLVersion::v2_0) || hasExtension(Extensions::DepthTexture) ||
           (hasExtension(Extensions::Depth32) && hasExtension(Extensions::RequiredInternalFormat));

  case TextureFeatures::DepthTexStorage16:
    return hasDesktopOrESVersion(*this, GLVersion::v4_2, GLVersion::v3_0_ES) ||
           (hasTextureFeature(TextureFeatures::DepthRenderbuffer16) &&
            hasInternalFeature(InternalFeatures::TexStorage));

  case TextureFeatures::DepthTexStorage24:
    return hasDesktopOrESVersion(*this, GLVersion::v4_2, GLVersion::v3_0_ES) ||
           (hasTextureFeature(TextureFeatures::DepthRenderbuffer24) &&
            hasInternalFeature(InternalFeatures::TexStorage));

  case TextureFeatures::DepthTexStorage32:
    // 32-bit integer depth textures are only supported on ES through specific extensions.
    return hasDesktopVersion(*this, GLVersion::v4_2) ||
           (hasExtension(Extensions::DepthTexture) && hasExtension(Extensions::TexStorage));

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isDepthFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::DepthFilterable:
  case TextureFeatures::DepthRenderbuffer16:
  case TextureFeatures::DepthRenderbuffer24:
  case TextureFeatures::DepthRenderbuffer32:
  case TextureFeatures::Depth24Stencil8:
  case TextureFeatures::Depth32FStencil8:
  case TextureFeatures::StencilTexture8:
    return isDepthRenderbufferFeatureSupported(feature);

  case TextureFeatures::DepthTexImage:
  case TextureFeatures::DepthTexImage16:
  case TextureFeatures::DepthTexImage24:
  case TextureFeatures::DepthTexImage32:
  case TextureFeatures::DepthTexStorage16:
  case TextureFeatures::DepthTexStorage24:
  case TextureFeatures::DepthTexStorage32:
    return isDepthTexImageFeatureSupported(feature);

  default:
    return false;
  }
}

bool DeviceFeatureSet::isCompressionFeatureSupported(TextureFeatures feature) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (feature) {
  case TextureFeatures::TextureCompressionAstc:
    return hasESVersion(*this, GLVersion::v3_2_ES) ||
           hasDesktopOrESExtension(*this, "GL_KHR_texture_compression_astc_hdr") ||
           hasDesktopOrESExtension(*this, "GL_KHR_texture_compression_astc_ldr") ||
           hasDesktopOrESExtension(*this, "GL_OES_texture_compression_astc");

  case TextureFeatures::TextureCompressionBptc:
    return hasDesktopExtension(*this, "GL_ARB_texture_compression_bptc") ||
           hasESExtension(*this, "GL_EXT_texture_compression_bptc") ||
           hasDesktopVersion(*this, GLVersion::v4_2);

  case TextureFeatures::TextureCompressionEtc1:
    return hasESExtension(*this, "GL_EXT_compressed_ETC1_RGB8_sub_texture") ||
           hasESExtension(*this, "GL_OES_compressed_ETC1_RGB8_texture");

  case TextureFeatures::TextureCompressionEtc2Eac:
    return hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_0_ES) ||
           hasDesktopExtension(*this, "GL_ARB_ES3_compatibility");

  case TextureFeatures::TextureCompressionPvrtc:
    return hasESExtension(*this, "GL_IMG_texture_compression_pvrtc");

  case TextureFeatures::TextureCompressionTexImage:
    // On Desktop GL, TexImage can be used to initialize a compressed texture.
    // On OpenGL ES, TexImage CANNOT be used.
    return !usesOpenGLES();

  case TextureFeatures::TextureCompressionTexStorage:
    // On Desktop GL, TexStorage CANNOT be used to initialize a compressed texture.
    // On OpenGL ES, TexStorage can be used if it is available.
    return usesOpenGLES() && hasInternalFeature(InternalFeatures::TexStorage);

  case TextureFeatures::TextureInteger:
    return hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) ||
           hasDesktopExtension(*this, "GL_EXT_texture_integer");

  case TextureFeatures::TextureTypeUInt8888Rev:
    return hasDesktopVersion(*this, GLVersion::v2_0);

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::isTextureFeatureSupported(TextureFeatures feature) const {
  switch (feature) {
  case TextureFeatures::ColorFilterable16f:
  case TextureFeatures::ColorFilterable32f:
  case TextureFeatures::ColorFormatRgb10A2UI:
  case TextureFeatures::ColorFormatRgInt:
  case TextureFeatures::ColorFormatRgUNorm16:
  case TextureFeatures::ColorFormatRgbaUNorm16:
    return isColorFilterableFeatureSupported(feature);

  case TextureFeatures::ColorRenderbuffer16f:
  case TextureFeatures::ColorRenderbuffer32f:
  case TextureFeatures::ColorRenderbufferRg16f:
  case TextureFeatures::ColorRenderbufferRg32f:
  case TextureFeatures::ColorRenderbufferRg8:
  case TextureFeatures::ColorRenderbufferRgb10A2:
  case TextureFeatures::ColorRenderbufferRgb16f:
  case TextureFeatures::ColorRenderbufferRgba8:
  case TextureFeatures::ColorRenderbufferSrgba8:
    return isColorRenderbufferFeatureSupported(feature);

  case TextureFeatures::ColorTexImage16f:
  case TextureFeatures::ColorTexImage32f:
  case TextureFeatures::ColorTexImageA8:
  case TextureFeatures::ColorTexImageBgr10A2:
  case TextureFeatures::ColorTexImageBgr5A1:
  case TextureFeatures::ColorTexImageBgra:
  case TextureFeatures::ColorTexImageBgraRgba8:
  case TextureFeatures::ColorTexImageBgraSrgba:
  case TextureFeatures::ColorTexImageLa:
  case TextureFeatures::ColorTexImageLa8:
  case TextureFeatures::ColorTexImageRg8:
  case TextureFeatures::ColorTexImageRgb10A2:
  case TextureFeatures::ColorTexImageRgba8:
  case TextureFeatures::ColorTexImageSrgba8:
    return isColorTexImageFeatureSupported(feature);

  case TextureFeatures::ColorTexStorage16f:
  case TextureFeatures::ColorTexStorage32f:
  case TextureFeatures::ColorTexStorageA8:
  case TextureFeatures::ColorTexStorageBgra8:
  case TextureFeatures::ColorTexStorageLa8:
  case TextureFeatures::ColorTexStorageRg8:
  case TextureFeatures::ColorTexStorageRgb10A2:
  case TextureFeatures::ColorTexStorageRgba8:
  case TextureFeatures::ColorTexStorageSrgba8:
    return isColorTexStorageFeatureSupported(feature);

  case TextureFeatures::DepthFilterable:
  case TextureFeatures::DepthRenderbuffer16:
  case TextureFeatures::DepthRenderbuffer24:
  case TextureFeatures::DepthRenderbuffer32:
  case TextureFeatures::Depth24Stencil8:
  case TextureFeatures::Depth32FStencil8:
  case TextureFeatures::DepthTexImage:
  case TextureFeatures::DepthTexImage16:
  case TextureFeatures::DepthTexImage24:
  case TextureFeatures::DepthTexImage32:
  case TextureFeatures::DepthTexStorage16:
  case TextureFeatures::DepthTexStorage24:
  case TextureFeatures::DepthTexStorage32:
  case TextureFeatures::StencilTexture8:
    return isDepthFeatureSupported(feature);

  case TextureFeatures::TextureCompressionAstc:
  case TextureFeatures::TextureCompressionBptc:
  case TextureFeatures::TextureCompressionEtc1:
  case TextureFeatures::TextureCompressionEtc2Eac:
  case TextureFeatures::TextureCompressionPvrtc:
  case TextureFeatures::TextureCompressionTexImage:
  case TextureFeatures::TextureCompressionTexStorage:
  case TextureFeatures::TextureInteger:
  case TextureFeatures::TextureTypeUInt8888Rev:
    return isCompressionFeatureSupported(feature);
  }

  return false;
}

bool DeviceFeatureSet::hasExtension(Extensions extension) const {
  const uint64_t extensionIndex = static_cast<uint64_t>(extension);
  IGL_DEBUG_ASSERT(extensionIndex < 64);
  const uint64_t extensionBit = 1ULL << extensionIndex;
  if ((extensionCacheInitialized_ & extensionBit) == 0) {
    if (isExtensionSupported(extension)) {
      extensionCache_ |= extensionBit;
    }
    extensionCacheInitialized_ |= extensionBit;
  }

  return (extensionCache_ & extensionBit) != 0;
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::hasFeature(DeviceFeatures feature) const {
  const uint64_t featureIndex = static_cast<uint64_t>(feature);
  IGL_DEBUG_ASSERT(featureIndex < 64);
  const uint64_t featureBit = 1ULL << featureIndex;
  if ((featureCacheInitialized_ & featureBit) == 0) {
    if (isFeatureSupported(feature)) {
      featureCache_ |= featureBit;
    }
    featureCacheInitialized_ |= featureBit;
  }

  return (featureCache_ & featureBit) != 0;
}

bool DeviceFeatureSet::hasInternalFeature(InternalFeatures feature) const {
  const uint32_t featureIndex = static_cast<uint32_t>(feature);
  IGL_DEBUG_ASSERT(featureIndex < 32);
  const uint32_t featureBit = 1u << featureIndex;
  if ((internalFeatureCacheInitialized_ & featureBit) == 0) {
    if (isInternalFeatureSupported(feature)) {
      internalFeatureCache_ |= featureBit;
    }
    internalFeatureCacheInitialized_ |= featureBit;
  }

  return (internalFeatureCache_ & featureBit) != 0;
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::hasTextureFeature(TextureFeatures feature) const {
  const uint64_t featureIndex = static_cast<uint64_t>(feature);
  IGL_DEBUG_ASSERT(featureIndex < 64);
  const uint64_t featureBit = 1ULL << featureIndex;
  if ((textureFeatureCacheInitialized_ & featureBit) == 0) {
    if (isTextureFeatureSupported(feature)) {
      textureFeatureCache_ |= featureBit;
    }
    textureFeatureCacheInitialized_ |= featureBit;
  }

  return (textureFeatureCache_ & featureBit) != 0;
}

bool DeviceFeatureSet::hasRequirement(DeviceRequirement requirement) const {
  switch (requirement) {
  case DeviceRequirement::ExplicitBindingExtReq:
    return !usesOpenGLES() && !hasDesktopVersion(*this, GLVersion::v4_2);

  case DeviceRequirement::StandardDerivativeExtReq:

    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/fwidth.xhtml
    // On desktop GL derivatives were supported from 2.0
    // no need for extension

    // GL_OES_standard_derivatives extension required only for versions prior to ES 3.0
    // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case DeviceRequirement::TextureArrayExtReq:
    // Array textures were introduced in OpenGL 3.0. Before OpenGL 3.0, they can be supported via
    // `GL_EXT_texture_array`
    return !usesOpenGLES() && !hasDesktopVersion(*this, GLVersion::v3_0);

  case DeviceRequirement::TextureFormatRGExtReq:
    // If we are running in on a platform that supports OpenGL ES 3.0 (which has GL_RED/GL_RG)
    // we can check if our context is using ES 3.0, otherwise fall back to `GL_EXT_texture_rg`
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case DeviceRequirement::ShaderTextureLodExtReq:
    // Desktop GL
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureLod.xhtml
    // textureLod() was introduced in GLSL 1.3 (which corresponds to OpenGL 3.0)
    // So if we are running on anything lower than OpenGL 3.0, we will need the extension

    // GL_EXT_shader_texture_lod extension required only for versions prior to ES 3.0
    // @fb-only
    // @fb-only
    // @fb-only
    // @fb-only
    return !hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES);
  }
  return false;
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::hasInternalRequirementColorGroup(InternalRequirement requirement) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (requirement) {
  case InternalRequirement::ColorTexImageRgb5A1Unsized:
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::ColorTexImageRgb10A2Unsized:
    return !hasTextureFeature(TextureFeatures::ColorRenderbufferRgb10A2) &&
           hasExtension(Extensions::TextureType2101010Rev);

  case InternalRequirement::ColorTexImageRgba4Unsized:
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES) &&
           !hasExtension(Extensions::RequiredInternalFormat);

  case InternalRequirement::ColorTexImageRgbApple422Unsized:
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  default:
    return false;
  }
}

bool DeviceFeatureSet::hasInternalRequirementMiscGroup(InternalRequirement requirement) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (requirement) {
  case InternalRequirement::DrawBuffersExtReq:
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::DrawBuffersIndexedExtReq:
    // The functions for the draw buffer state are core from OpenGL ES 3.2; before that they need
    // one of the extensions
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_2_ES);

  case InternalRequirement::Depth24Stencil8Unsized:
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::Depth32Unsized:
    return hasExtension(Extensions::DepthTexture);

  case InternalRequirement::FramebufferBlitExtReq:
    // GL_ARB_framebuffer_object also includes glBlitFramebuffer so no need to use
    // BlitFramebufferEXT if it is present.
    return !hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES) &&
           !hasExtension(Extensions::FramebufferObject);

  case InternalRequirement::InvalidateFramebufferExtReq:
    return !hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_0_ES) &&
           !hasExtension(Extensions::InvalidateSubdata);

  case InternalRequirement::MapBufferExtReq:
    // OpenGL ES does not include MapBuffer
    return usesOpenGLES();

  case InternalRequirement::MapBufferRangeExtReq:
    // OpenGL ES 2 does not include MapBufferRange
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::MultiSampleExtReq:
    // OpenGL ES has various extensions before 3.0 that are required, and
    // GL_IMG_multisampled_render_to_texture uses different enum values than later standard
    // versions.
    return !(hasDesktopVersion(*this, GLVersion::v3_0) ||
             hasExtension(Extensions::FramebufferObject) ||
             hasESVersion(*this, GLVersion::v3_0_ES));

  case InternalRequirement::ShaderImageLoadStoreExtReq:
    return !usesOpenGLES() && !hasDesktopVersion(*this, GLVersion::v4_2);

  case InternalRequirement::SyncExtReq:
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::SwizzleAlphaTexturesReq:

    return hasDesktopVersion(*this, GLVersion::v3_0);

  case InternalRequirement::TexStorageExtReq:
    return !hasDesktopOrESVersionOrExtension(
        *this, GLVersion::v4_2, GLVersion::v3_0_ES, "GL_ARB_texture_storage");

  case InternalRequirement::Texture3DExtReq:
    return !hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES);

  case InternalRequirement::TextureHalfFloatExtReq:
    // GL_OES_texture_half_float extension uses different enum values for GL_HALF_FLOAT_OES than
    // GL_HALF_FLOAT.
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::UnmapBufferExtReq:
    // OpenGL ES 2 does not include UnmapBuffer
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::VertexArrayObjectExtReq:
    return usesOpenGLES() && !hasESVersion(*this, GLVersion::v3_0_ES);

  case InternalRequirement::VertexAttribDivisorExtReq:
    return !hasDesktopOrESVersion(*this, GLVersion::v3_3, GLVersion::v3_0_ES);

  default:
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
bool DeviceFeatureSet::hasInternalRequirement(InternalRequirement requirement) const {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (requirement) {
  case InternalRequirement::DebugMessageExtReq:
    return !hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_2_ES);

  case InternalRequirement::DebugMessageCallbackExtReq:
    return !hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_2_ES);

  case InternalRequirement::DebugLabelExtEnumsReq:
    // GL_EXT_debug_label requires extension-specific enums for some object types
    return hasInternalRequirement(InternalRequirement::DebugLabelExtReq) &&
           !hasExtension(Extensions::Debug);

  case InternalRequirement::DebugLabelExtReq:
    return !hasDesktopOrESVersion(*this, GLVersion::v4_3, GLVersion::v3_2_ES);

  case InternalRequirement::ColorTexImageRgb5A1Unsized:
  case InternalRequirement::ColorTexImageRgb10A2Unsized:
  case InternalRequirement::ColorTexImageRgba4Unsized:
  case InternalRequirement::ColorTexImageRgbApple422Unsized:
    return hasInternalRequirementColorGroup(requirement);

  default:
    return hasInternalRequirementMiscGroup(requirement);
  }
}

bool DeviceFeatureSet::getFeatureLimits(DeviceFeatureLimits featureLimits, size_t& result) const {
  IGL_PROFILER_FUNCTION();
  GLint tsize = 0;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (featureLimits) {
  case DeviceFeatureLimits::MaxTextureDimension1D2D:
    glContext_.getIntegerv(GL_MAX_TEXTURE_SIZE, &tsize);
    result = static_cast<size_t>(tsize);
    return true;

  case DeviceFeatureLimits::MaxCubeMapDimension:
    glContext_.getIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &tsize);
    result = static_cast<size_t>(tsize);
    return true;

  case DeviceFeatureLimits::MaxVertexUniformVectors:
    result = getMaxVertexUniforms();
    return true;

  case DeviceFeatureLimits::MaxFragmentUniformVectors:
    result = getMaxFragmentUniforms();
    return true;

  case DeviceFeatureLimits::MaxMultisampleCount:
    if (hasFeature(DeviceFeatures::MultiSample)) {
      if (hasInternalRequirement(InternalRequirement::MultiSampleExtReq) &&
          hasExtension(Extensions::MultiSampleImg)) {
        glContext_.getIntegerv(GL_MAX_SAMPLES_IMG, &tsize);
      } else {
        // Official standards and all other extensions use the same value for GL_MAX_SAMPLES
        glContext_.getIntegerv(GL_MAX_SAMPLES, &tsize);
      }
    }
    result = static_cast<size_t>(tsize);
    return true;
  case DeviceFeatureLimits::MaxPushConstantBytes:
    result = 0;
    return true;
  case DeviceFeatureLimits::MaxStorageBufferBytes:
    if (hasFeature(DeviceFeatures::StorageBuffers)) {
      glContext_.getIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &tsize);
    }
    result = tsize;
    return true;
  case DeviceFeatureLimits::MaxUniformBufferBytes:
    if (hasFeature(DeviceFeatures::UniformBlocks)) {
      glContext_.getIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &tsize);
    }
    result = static_cast<size_t>(tsize);
    return true;
  case DeviceFeatureLimits::PushConstantsAlignment:
    result = 0;
    return true;
  case DeviceFeatureLimits::ShaderStorageBufferOffsetAlignment:
    tsize = 256;
    if (hasFeature(DeviceFeatures::StorageBuffers)) {
      glContext_.getIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &tsize);
    }
    result = static_cast<size_t>(tsize);
    return true;
  case DeviceFeatureLimits::BufferAlignment:
    result = 16;
    if (hasFeature(DeviceFeatures::UniformBlocks)) {
      if (glContext_.isCurrentContext()) {
        glContext_.getIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &tsize);
        result = std::max(static_cast<size_t>(tsize), result);
      }
    }
    return true;
  case DeviceFeatureLimits::BufferNoCopyAlignment:
    result = 0;
    return true;
  case DeviceFeatureLimits::MaxBindBytesBytes:
    result = 0;
    return true;
  case DeviceFeatureLimits::MaxTextureDimension3D:
    glContext_.getIntegerv(GL_MAX_3D_TEXTURE_SIZE, &tsize);
    result = static_cast<size_t>(tsize);
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupSizeX:
    if (hasFeature(DeviceFeatures::Compute)) {
      // OpenGL ES 3.1+ and OpenGL 4.3+: use conservative value
      result = 64;
    } else {
      result = 0;
    }
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupSizeY:
    if (hasFeature(DeviceFeatures::Compute)) {
      // OpenGL ES 3.1+ and OpenGL 4.3+: use conservative value
      result = 64;
    } else {
      result = 0;
    }
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupSizeZ:
    if (hasFeature(DeviceFeatures::Compute)) {
      // OpenGL ES 3.1+ and OpenGL 4.3+: use conservative value
      result = 64;
    } else {
      result = 0;
    }
    return true;
  case DeviceFeatureLimits::MaxComputeWorkGroupInvocations:
    if (hasFeature(DeviceFeatures::Compute)) {
#if defined(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
      glContext_.getIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &tsize);
      result = static_cast<size_t>(tsize);
#endif
    } else {
      result = 0;
    }
    return true;
  // D3D12-specific descriptor heap limits - not applicable to OpenGL
  case DeviceFeatureLimits::MaxDescriptorHeapCbvSrvUav:
  case DeviceFeatureLimits::MaxDescriptorHeapSamplers:
  case DeviceFeatureLimits::MaxDescriptorHeapRtvs:
  case DeviceFeatureLimits::MaxDescriptorHeapDsvs:
    result = 0;
    return false;
  case DeviceFeatureLimits::MaxVertexInputAttributes:
    glContext_.getIntegerv(GL_MAX_VERTEX_ATTRIBS, &tsize);
    result = static_cast<size_t>(tsize);
    return true;
  case DeviceFeatureLimits::MaxColorAttachments:
    glContext_.getIntegerv(GL_MAX_COLOR_ATTACHMENTS, &tsize);
    result = static_cast<size_t>(tsize);
    return true;
  default:
    IGL_DEBUG_ABORT(
        "invalid feature limit query: feature limit query is not implemented or does "
        "not exist\n");
    return false;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
// NOLINTNEXTLINE(misc-no-recursion)
ICapabilities::TextureFormatCapabilities
DeviceFeatureSet::getColorUNorm8BasicTextureFormatCapabilities(TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::LA_UNorm8:
  case TextureFormat::L_UNorm8:
    if (hasTextureFeature(TextureFeatures::ColorTexImageLa)) {
      capabilities |= sampled | sampledFiltered;
      if (hasTextureFeature(TextureFeatures::ColorTexStorageLa8)) {
        capabilities |= storage;
      }
    }
    break;
  case TextureFormat::A_UNorm8:
    capabilities |= sampled | sampledFiltered;
    if (hasTextureFeature(TextureFeatures::ColorTexStorageA8)) {
      capabilities |= storage;
    }
    break;
  case TextureFormat::RGBA_UNorm8:
  case TextureFormat::RGBX_UNorm8:
    capabilities |= sampled | sampledFiltered;
    if (hasTextureFeature(TextureFeatures::ColorTexStorageRgba8)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexImageRgba8)) {
      capabilities |= sampledAttachment;
    }
    if (hasTextureFeature(TextureFeatures::ColorRenderbufferRgba8)) {
      capabilities |= attachment;
    }
    break;
  case TextureFormat::RG_UNorm8:
  case TextureFormat::R_UNorm8:
    if (hasFeature(DeviceFeatures::TextureFormatRG)) {
      capabilities |= sampled | sampledFiltered;
      if (hasTextureFeature(TextureFeatures::ColorTexStorageRg8)) {
        capabilities |= storage;
      }
      if (hasTextureFeature(TextureFeatures::ColorTexImageRg8)) {
        capabilities |= sampledAttachment;
      }
      if (hasTextureFeature(TextureFeatures::ColorRenderbufferRg8)) {
        capabilities |= attachment;
      }
    }
    break;
  default:
    break;
  }
  return capabilities;
}

// NOLINTNEXTLINE(misc-no-recursion)
ICapabilities::TextureFormatCapabilities
DeviceFeatureSet::getColorUNorm8BgraSrgbTextureFormatCapabilities(TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::BGRA_UNorm8:
    // EXT_texture_format_BGRA8888 adds support for GL_BGRA as a Renderbuffer format, but this was
    // in a later revision of the extension. It is not supported on our test devices.
    if (hasTextureFeature(TextureFeatures::ColorTexImageBgra)) {
      capabilities |= sampled | sampledFiltered;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexImageBgraRgba8)) {
      capabilities |= sampledAttachment;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexStorageBgra8)) {
      capabilities |= storage;
    }
    break;
  case TextureFormat::RGBA_SRGB:
    if (hasFeature(DeviceFeatures::SRGB)) {
      capabilities |= sampled | sampledFiltered;
      if (hasTextureFeature(TextureFeatures::ColorTexStorageSrgba8)) {
        capabilities |= storage;
      }
      if (hasTextureFeature(TextureFeatures::ColorTexImageSrgba8)) {
        capabilities |= sampledAttachment;
      }
      if (hasTextureFeature(TextureFeatures::ColorRenderbufferSrgba8)) {
        capabilities |= attachment;
      }
    }
    break;
  case TextureFormat::BGRA_SRGB:
    if (hasFeature(DeviceFeatures::SRGB) &&
        hasTextureFeature(TextureFeatures::ColorTexImageBgraSrgba)) {
      capabilities |= sampled | sampledFiltered;
    }
    break;
  default:
    break;
  }
  return capabilities;
}

ICapabilities::TextureFormatCapabilities
DeviceFeatureSet::getColorUNormWideTextureFormatCapabilities(TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;
  const auto all = sampled | sampledFiltered | storage | attachment | sampledAttachment;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::R_UNorm16:
  case TextureFormat::RG_UNorm16:
    if (hasTextureFeature(TextureFeatures::ColorFormatRgUNorm16)) {
      capabilities |= all;
    }
    break;
  case TextureFormat::RGBA_UNorm16:
    if (hasTextureFeature(TextureFeatures::ColorFormatRgbaUNorm16)) {
      capabilities |= all;
    }
    break;
  default:
    break;
  }
  return capabilities;
}

// NOLINTNEXTLINE(misc-no-recursion)
ICapabilities::TextureFormatCapabilities DeviceFeatureSet::getColorF16TextureFormatCapabilities(
    TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::RGBA_F16:
    if (hasFeature(DeviceFeatures::TextureHalfFloat)) {
      capabilities |= sampled;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexImage16f)) {
      capabilities |= sampledAttachment;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexStorage16f)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::ColorRenderbuffer16f)) {
      capabilities |= attachment;
    }
    if (hasTextureFeature(TextureFeatures::ColorFilterable16f)) {
      capabilities |= sampledFiltered;
    }
    break;
  case TextureFormat::RGB_F16:
    // RGB floating point textures are NOT renderable
    if (hasFeature(DeviceFeatures::TextureHalfFloat)) {
      capabilities |= sampled;
    }
    if (hasTextureFeature(TextureFeatures::ColorRenderbufferRgb16f)) {
      capabilities |= attachment | sampledAttachment;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexStorage16f)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::ColorFilterable16f)) {
      capabilities |= sampledFiltered;
    }
    break;
  case TextureFormat::RG_F16:
  case TextureFormat::R_F16:
    if (hasFeature(DeviceFeatures::TextureFormatRG)) {
      if (hasFeature(DeviceFeatures::TextureHalfFloat)) {
        capabilities |= sampled;
      }
      if (hasTextureFeature(TextureFeatures::ColorRenderbufferRg16f)) {
        capabilities |= attachment | sampledAttachment;
      }
      if (hasTextureFeature(TextureFeatures::ColorTexStorage16f)) {
        capabilities |= storage;
      }
      if (hasTextureFeature(TextureFeatures::ColorFilterable16f)) {
        capabilities |= sampledFiltered;
      }
    }
    break;
  default:
    break;
  }
  return capabilities;
}

// NOLINTNEXTLINE(misc-no-recursion)
ICapabilities::TextureFormatCapabilities DeviceFeatureSet::getColorF32TextureFormatCapabilities(
    TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::RGBA_F32:
    if (hasFeature(DeviceFeatures::TextureFloat)) {
      capabilities |= sampled;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexStorage32f)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::ColorRenderbuffer32f)) {
      capabilities |= attachment | sampledAttachment;
    }
    if (hasTextureFeature(TextureFeatures::ColorFilterable32f)) {
      capabilities |= sampledFiltered;
    }
    break;
  case TextureFormat::RGB_F32:
    // RGB floating point textures are NOT renderable
    if (hasFeature(DeviceFeatures::TextureFloat)) {
      capabilities |= sampled;
    }
    if (hasTextureFeature(TextureFeatures::ColorTexStorage32f)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::ColorFilterable32f)) {
      capabilities |= sampledFiltered;
    }
    break;
  case TextureFormat::RG_F32:
  case TextureFormat::R_F32:
    if (hasFeature(DeviceFeatures::TextureFormatRG)) {
      if (hasFeature(DeviceFeatures::TextureFloat)) {
        capabilities |= sampled;
      }
      if (hasTextureFeature(TextureFeatures::ColorTexStorage32f)) {
        capabilities |= storage;
      }
      if (hasTextureFeature(TextureFeatures::ColorRenderbufferRg32f)) {
        capabilities |= sampledAttachment | attachment;
      }
      if (hasTextureFeature(TextureFeatures::ColorFilterable32f)) {
        capabilities |= sampledFiltered;
      }
    }
    break;
  default:
    break;
  }
  return capabilities;
}

// NOLINTNEXTLINE(misc-no-recursion)
ICapabilities::TextureFormatCapabilities DeviceFeatureSet::getSpecialColorTextureFormatCapabilities(
    TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;
  const auto all = sampled | sampledFiltered | storage | attachment | sampledAttachment;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::R_UInt16:
  case TextureFormat::RG_UInt16:
    if (hasTextureFeature(TextureFeatures::ColorFormatRgInt)) {
      capabilities |= sampled | storage | attachment;
    }
    break;
  case TextureFormat::R_UInt32:
    if (hasTextureFeature(TextureFeatures::TextureInteger)) {
      capabilities |= sampled | storage | attachment | sampledAttachment;
    }
    break;
  case TextureFormat::RGBA_UInt32:
    if (hasTextureFeature(TextureFeatures::TextureInteger)) {
      capabilities |= sampled | storage | attachment | sampledAttachment;
    }
    break;
  case TextureFormat::B5G5R5A1_UNorm:
    if (hasTextureFeature(TextureFeatures::ColorTexImageBgr5A1)) {
      capabilities |= sampled | sampledFiltered;
    }
    break;
  case TextureFormat::ABGR_UNorm4:
    capabilities |= all;

    break;
  case TextureFormat::R4G2B2_UNorm_Apple:
    if (hasExtension(Extensions::AppleRgb422)) {
      // GL_APPLE_rgb_422 formats are not color-renderable formats
      capabilities |= sampled | sampledFiltered;
      if (hasInternalFeature(InternalFeatures::TexStorage)) {
        capabilities |= storage;
      }
    }
    break;
  case TextureFormat::R4G2B2_UNorm_Rev_Apple:
    if (hasExtension(Extensions::AppleRgb422)) {
      // GL_APPLE_rgb_422 formats are not color-renderable formats
      // TexStorage does not support UNSIGNED_SHORT_8_8_REV_APPLE
      capabilities |= sampled | sampledFiltered;
    }
    break;
  case TextureFormat::R5G5B5A1_UNorm:
    capabilities |= sampled | sampledFiltered | storage;
    if (hasInternalFeature(InternalFeatures::FramebufferObject)) {
      capabilities |= attachment | sampledAttachment;
    }
    break;
  case TextureFormat::BGR10_A2_Unorm:
    if (hasTextureFeature(TextureFeatures::ColorTexImageBgr10A2)) {
      capabilities |= sampled | sampledFiltered;
    }
    break;
  case TextureFormat::B10G11R11_UFloat:
    if (hasDesktopOrESVersion(*this, GLVersion::v3_0, GLVersion::v3_0_ES)) {
      capabilities |= sampled | sampledFiltered;
      if (hasInternalFeature(InternalFeatures::TexStorage)) {
        capabilities |= ICapabilities::TextureFormatCapabilityBits::Storage;
      }
      if (hasDesktopOrESVersionOrExtension(
              *this, GLVersion::v3_0, GLVersion::v3_2_ES, "GL_EXT_color_buffer_float")) {
        capabilities |= attachment | sampledAttachment;
      }
    }
    break;
  case TextureFormat::RGB10_A2_UNorm_Rev:
    if (hasTextureFeature(TextureFeatures::ColorTexImageRgb10A2)) {
      capabilities |= sampled | sampledFiltered;
      if (!hasInternalRequirement(InternalRequirement::ColorTexImageRgb10A2Unsized)) {
        capabilities |= sampledAttachment;
      }
    }
    if (hasTextureFeature(TextureFeatures::ColorTexStorageRgb10A2)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::ColorRenderbufferRgb10A2)) {
      capabilities |= attachment;
    }
    break;
  case TextureFormat::RGB10_A2_Uint_Rev:
    if (hasTextureFeature(TextureFeatures::ColorFormatRgb10A2UI)) {
      capabilities |= sampled | storage | attachment | sampledAttachment;
    }
    break;
  case TextureFormat::BGRA_UNorm8_Rev:
    if (hasTextureFeature(TextureFeatures::TextureTypeUInt8888Rev)) {
      capabilities |= sampled | sampledFiltered;
    }
    break;
  case TextureFormat::R5G6B5_UNorm:
    capabilities |= all;
    break;
  case TextureFormat::B5G6R5_UNorm:
    // Unsupported
    break;
  default:
    break;
  }
  return capabilities;
}

ICapabilities::TextureFormatCapabilities
DeviceFeatureSet::getDepthUNorm16UNorm32TextureFormatCapabilities(TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::Z_UNorm16:
    if (hasTextureFeature(TextureFeatures::DepthTexImage)) {
      capabilities |= sampled;
    }
    if (hasTextureFeature(TextureFeatures::DepthTexImage16)) {
      capabilities |= sampledAttachment;
    }
    if (hasTextureFeature(TextureFeatures::DepthTexStorage16)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::DepthRenderbuffer16)) {
      capabilities |= attachment;
    }
    if (hasTextureFeature(TextureFeatures::DepthFilterable)) {
      capabilities |= sampledFiltered;
    }
    break;
  case TextureFormat::Z_UNorm32:
    if (hasTextureFeature(TextureFeatures::DepthTexImage32)) {
      capabilities |= sampled | sampledAttachment;
    }
    if (hasTextureFeature(TextureFeatures::DepthTexStorage32)) {
      capabilities |= storage;
    }
    if (hasTextureFeature(TextureFeatures::DepthRenderbuffer32)) {
      capabilities |= attachment;
    }
    if (hasTextureFeature(TextureFeatures::DepthFilterable)) {
      capabilities |= sampledFiltered;
    }
    break;
  default:
    break;
  }
  return capabilities;
}

ICapabilities::TextureFormatCapabilities DeviceFeatureSet::getDepthUNorm24TextureFormatCapabilities(
    TextureFormat /*format*/) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledFiltered = ICapabilities::TextureFormatCapabilityBits::SampledFiltered;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  if (hasTextureFeature(TextureFeatures::DepthTexImage24)) {
    capabilities |= sampled | sampledAttachment;
  }
  if (hasTextureFeature(TextureFeatures::DepthTexStorage24)) {
    capabilities |= storage;
  }
  if (hasTextureFeature(TextureFeatures::DepthRenderbuffer24)) {
    capabilities |= attachment;
  }
  if (hasTextureFeature(TextureFeatures::DepthFilterable)) {
    capabilities |= sampledFiltered;
  }

  // TODO: Remove these fallback once devices can properly provide a supported format
  if (hasTextureFeature(TextureFeatures::DepthTexImage32)) {
    capabilities |= sampled | sampledAttachment;
  }
  if (hasTextureFeature(TextureFeatures::DepthTexStorage32)) {
    capabilities |= storage;
  }
  if (hasTextureFeature(TextureFeatures::DepthRenderbuffer32)) {
    capabilities |= attachment;
  }
  if (hasTextureFeature(TextureFeatures::DepthFilterable)) {
    capabilities |= sampledFiltered;
  }
  return capabilities;
}

ICapabilities::TextureFormatCapabilities DeviceFeatureSet::getStencilTextureFormatCapabilities(
    TextureFormat format) const {
  const auto sampled = ICapabilities::TextureFormatCapabilityBits::Sampled;
  const auto attachment = ICapabilities::TextureFormatCapabilityBits::Attachment;
  const auto storage = hasInternalFeature(InternalFeatures::TexStorage)
                           ? ICapabilities::TextureFormatCapabilityBits::Storage
                           : 0;
  const auto sampledAttachment = ICapabilities::TextureFormatCapabilityBits::SampledAttachment;
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::S8_UInt_Z24_UNorm:
    if (hasTextureFeature(TextureFeatures::Depth24Stencil8)) {
      capabilities |= sampled | attachment | sampledAttachment;
      if (hasInternalFeature(InternalFeatures::TexStorage)) {
        capabilities |= storage;
      }
    }
    break;
  case TextureFormat::S8_UInt_Z32_UNorm:
    if (hasTextureFeature(TextureFeatures::Depth32FStencil8)) {
      capabilities |= sampled | attachment | sampledAttachment;
      if (hasInternalFeature(InternalFeatures::TexStorage)) {
        capabilities |= storage;
      }
    }
    break;
  case TextureFormat::S_UInt8:
    if (hasTextureFeature(TextureFeatures::StencilTexture8)) {
      capabilities |= sampled | storage;
    }
    capabilities |= attachment;
    break;
  default:
    break;
  }
  return capabilities;
}

ICapabilities::TextureFormatCapabilities DeviceFeatureSet::getCompressedTextureFormatCapabilities(
    TextureFormat format) const {
  const auto compressed = ICapabilities::TextureFormatCapabilityBits::Sampled |
                          (hasTextureFeature(TextureFeatures::TextureCompressionTexStorage)
                               ? ICapabilities::TextureFormatCapabilityBits::Storage
                               : 0);
  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;

  ICapabilities::TextureFormatCapabilities capabilities = unsupported;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::RGBA_ASTC_4x4:
  case TextureFormat::SRGB8_A8_ASTC_4x4:
  case TextureFormat::RGBA_ASTC_5x4:
  case TextureFormat::SRGB8_A8_ASTC_5x4:
  case TextureFormat::RGBA_ASTC_5x5:
  case TextureFormat::SRGB8_A8_ASTC_5x5:
  case TextureFormat::RGBA_ASTC_6x5:
  case TextureFormat::SRGB8_A8_ASTC_6x5:
  case TextureFormat::RGBA_ASTC_6x6:
  case TextureFormat::SRGB8_A8_ASTC_6x6:
  case TextureFormat::RGBA_ASTC_8x5:
  case TextureFormat::SRGB8_A8_ASTC_8x5:
  case TextureFormat::RGBA_ASTC_8x6:
  case TextureFormat::SRGB8_A8_ASTC_8x6:
  case TextureFormat::RGBA_ASTC_8x8:
  case TextureFormat::SRGB8_A8_ASTC_8x8:
  case TextureFormat::RGBA_ASTC_10x5:
  case TextureFormat::SRGB8_A8_ASTC_10x5:
  case TextureFormat::RGBA_ASTC_10x6:
  case TextureFormat::SRGB8_A8_ASTC_10x6:
  case TextureFormat::RGBA_ASTC_10x8:
  case TextureFormat::SRGB8_A8_ASTC_10x8:
  case TextureFormat::RGBA_ASTC_10x10:
  case TextureFormat::SRGB8_A8_ASTC_10x10:
  case TextureFormat::RGBA_ASTC_12x10:
  case TextureFormat::SRGB8_A8_ASTC_12x10:
  case TextureFormat::RGBA_ASTC_12x12:
  case TextureFormat::SRGB8_A8_ASTC_12x12:
    if (hasTextureFeature(TextureFeatures::TextureCompressionAstc)) {
      capabilities |= compressed;
    }
    break;
  case TextureFormat::RGBA_BC7_UNORM_4x4:
  case TextureFormat::RGBA_BC7_SRGB_4x4:
    if (hasTextureFeature(TextureFeatures::TextureCompressionBptc)) {
      capabilities |= compressed;
    }
    break;
  case TextureFormat::RGBA_PVRTC_2BPPV1:
  case TextureFormat::RGB_PVRTC_2BPPV1:
  case TextureFormat::RGBA_PVRTC_4BPPV1:
  case TextureFormat::RGB_PVRTC_4BPPV1:
    if (hasTextureFeature(TextureFeatures::TextureCompressionPvrtc)) {
      capabilities |= compressed;
    }
    break;
  case TextureFormat::RGB8_ETC1:
    if (hasTextureFeature(TextureFeatures::TextureCompressionEtc1)) {
      capabilities |= compressed;
    }
    break;
  case TextureFormat::RGB8_ETC2:
  case TextureFormat::SRGB8_ETC2:
  case TextureFormat::RGB8_Punchthrough_A1_ETC2:
  case TextureFormat::SRGB8_Punchthrough_A1_ETC2:
  case TextureFormat::RGBA8_EAC_ETC2:
  case TextureFormat::SRGB8_A8_EAC_ETC2:
  case TextureFormat::RG_EAC_UNorm:
  case TextureFormat::RG_EAC_SNorm:
  case TextureFormat::R_EAC_UNorm:
  case TextureFormat::R_EAC_SNorm:
    if (hasTextureFeature(TextureFeatures::TextureCompressionEtc2Eac)) {
      capabilities |= compressed;
    }
    break;
  default:
    break;
  }
  return capabilities;
}

///
/// Returns the supported capabilities of the selected format.
/// @param format: The texture format to query
/// @return a combination of TextureFormatCapabilities flags
ICapabilities::TextureFormatCapabilities DeviceFeatureSet::getTextureFormatCapabilities(
    TextureFormat format) const {
  IGL_PROFILER_FUNCTION();
  // TODO: Remove this fallback once devices can properly provide a supported format
  if (format == TextureFormat::S8_UInt_Z32_UNorm &&
      !hasTextureFeature(TextureFeatures::Depth32FStencil8)) {
    format = TextureFormat::S8_UInt_Z24_UNorm;
  }
  const auto it = textureCapabilityCache_.find(format);
  if (it != textureCapabilityCache_.end()) {
    return it->second;
  }

  const auto unsupported = ICapabilities::TextureFormatCapabilityBits::Unsupported;
  ICapabilities::TextureFormatCapabilities capabilities = unsupported;

  // First check common formats
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (format) {
  case TextureFormat::LA_UNorm8:
  case TextureFormat::L_UNorm8:
  case TextureFormat::A_UNorm8:
  case TextureFormat::RGBA_UNorm8:
  case TextureFormat::RGBX_UNorm8:
  case TextureFormat::RG_UNorm8:
  case TextureFormat::R_UNorm8:
    capabilities = getColorUNorm8BasicTextureFormatCapabilities(format);
    break;

  case TextureFormat::BGRA_UNorm8:
  case TextureFormat::RGBA_SRGB:
  case TextureFormat::BGRA_SRGB:
    capabilities = getColorUNorm8BgraSrgbTextureFormatCapabilities(format);
    break;

  case TextureFormat::R_UNorm16:
  case TextureFormat::RG_UNorm16:
  case TextureFormat::RGBA_UNorm16:
    capabilities = getColorUNormWideTextureFormatCapabilities(format);
    break;

  case TextureFormat::RGBA_F16:
  case TextureFormat::RGB_F16:
  case TextureFormat::RG_F16:
  case TextureFormat::R_F16:
    capabilities = getColorF16TextureFormatCapabilities(format);
    break;

  case TextureFormat::RGBA_F32:
  case TextureFormat::RGB_F32:
  case TextureFormat::RG_F32:
  case TextureFormat::R_F32:
    capabilities = getColorF32TextureFormatCapabilities(format);
    break;

  case TextureFormat::R_UInt16:
  case TextureFormat::RG_UInt16:
  case TextureFormat::R_UInt32:
  case TextureFormat::RGBA_UInt32:
  case TextureFormat::B5G5R5A1_UNorm:
  case TextureFormat::ABGR_UNorm4:
  case TextureFormat::R4G2B2_UNorm_Apple:
  case TextureFormat::R4G2B2_UNorm_Rev_Apple:
  case TextureFormat::R5G5B5A1_UNorm:
  case TextureFormat::BGR10_A2_Unorm:
  case TextureFormat::B10G11R11_UFloat:
  case TextureFormat::RGB10_A2_UNorm_Rev:
  case TextureFormat::RGB10_A2_Uint_Rev:
  case TextureFormat::BGRA_UNorm8_Rev:
  case TextureFormat::R5G6B5_UNorm:
  case TextureFormat::B5G6R5_UNorm:
    capabilities = getSpecialColorTextureFormatCapabilities(format);
    break;

  case TextureFormat::Z_UNorm16:
  case TextureFormat::Z_UNorm32:
    capabilities = getDepthUNorm16UNorm32TextureFormatCapabilities(format);
    break;

  case TextureFormat::Z_UNorm24:
    capabilities = getDepthUNorm24TextureFormatCapabilities(format);
    break;

  case TextureFormat::S8_UInt_Z24_UNorm:
  case TextureFormat::S8_UInt_Z32_UNorm:
  case TextureFormat::S_UInt8:
    capabilities = getStencilTextureFormatCapabilities(format);
    break;

  case TextureFormat::RGBA_ASTC_4x4:
  case TextureFormat::SRGB8_A8_ASTC_4x4:
  case TextureFormat::RGBA_ASTC_5x4:
  case TextureFormat::SRGB8_A8_ASTC_5x4:
  case TextureFormat::RGBA_ASTC_5x5:
  case TextureFormat::SRGB8_A8_ASTC_5x5:
  case TextureFormat::RGBA_ASTC_6x5:
  case TextureFormat::SRGB8_A8_ASTC_6x5:
  case TextureFormat::RGBA_ASTC_6x6:
  case TextureFormat::SRGB8_A8_ASTC_6x6:
  case TextureFormat::RGBA_ASTC_8x5:
  case TextureFormat::SRGB8_A8_ASTC_8x5:
  case TextureFormat::RGBA_ASTC_8x6:
  case TextureFormat::SRGB8_A8_ASTC_8x6:
  case TextureFormat::RGBA_ASTC_8x8:
  case TextureFormat::SRGB8_A8_ASTC_8x8:
  case TextureFormat::RGBA_ASTC_10x5:
  case TextureFormat::SRGB8_A8_ASTC_10x5:
  case TextureFormat::RGBA_ASTC_10x6:
  case TextureFormat::SRGB8_A8_ASTC_10x6:
  case TextureFormat::RGBA_ASTC_10x8:
  case TextureFormat::SRGB8_A8_ASTC_10x8:
  case TextureFormat::RGBA_ASTC_10x10:
  case TextureFormat::SRGB8_A8_ASTC_10x10:
  case TextureFormat::RGBA_ASTC_12x10:
  case TextureFormat::SRGB8_A8_ASTC_12x10:
  case TextureFormat::RGBA_ASTC_12x12:
  case TextureFormat::SRGB8_A8_ASTC_12x12:
  case TextureFormat::RGBA_BC7_UNORM_4x4:
  case TextureFormat::RGBA_BC7_SRGB_4x4:
  case TextureFormat::RGBA_PVRTC_2BPPV1:
  case TextureFormat::RGB_PVRTC_2BPPV1:
  case TextureFormat::RGBA_PVRTC_4BPPV1:
  case TextureFormat::RGB_PVRTC_4BPPV1:
  case TextureFormat::RGB8_ETC1:
  case TextureFormat::RGB8_ETC2:
  case TextureFormat::SRGB8_ETC2:
  case TextureFormat::RGB8_Punchthrough_A1_ETC2:
  case TextureFormat::SRGB8_Punchthrough_A1_ETC2:
  case TextureFormat::RGBA8_EAC_ETC2:
  case TextureFormat::SRGB8_A8_EAC_ETC2:
  case TextureFormat::RG_EAC_UNorm:
  case TextureFormat::RG_EAC_SNorm:
  case TextureFormat::R_EAC_UNorm:
  case TextureFormat::R_EAC_SNorm:
    capabilities = getCompressedTextureFormatCapabilities(format);
    break;

  case TextureFormat::Invalid:
  case TextureFormat::YUV_NV12:
  case TextureFormat::YUV_420p:
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  // @fb-only
  default:
    // We are relying on the fact that TextureFormatCapabilities::Unsupported is 0
    return textureCapabilityCache_[format];
  };

  textureCapabilityCache_[format] = capabilities;
  return capabilities;
}

uint32_t DeviceFeatureSet::getMaxVertexUniforms() const {
  IGL_PROFILER_FUNCTION();
  GLint tsize = 0;
  // MaxVertexUniformVectors is the maximum number of 4-element vectors that can be passed as
  // uniform to a vertex shader. All uniforms are 4-element aligned, a single uniform counts at
  // least as one 4-element vector.
  // GL_MAX_VERTEX_UNIFORM_COMPONENTS available on Desktop OpenGL 2.0+ and on OpenGL ES 3.0+.
  // GL_MAX_VERTEX_UNIFORM_VECTORS is available on Desktop OpenGL 3.0+ and on OpenGL ES 2.0+.
  // GL_MAX_VERTEX_UNIFORM_VECTORS is equal to GL_MAX_VERTEX_UNIFORM_COMPONENTS / 4.
  if (hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES)) {
    glContext_.getIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &tsize);
    tsize /= 4;
  } else {
    glContext_.getIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS, &tsize);
  }
  return static_cast<uint32_t>(tsize);
}

uint32_t DeviceFeatureSet::getMaxFragmentUniforms() const {
  IGL_PROFILER_FUNCTION();
  GLint tsize = 0;
  // PLease see comments above in getMaxVertexUniforms
  if (hasDesktopOrESVersion(*this, GLVersion::v2_0, GLVersion::v3_0_ES)) {
    glContext_.getIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &tsize);
    tsize /= 4;
  } else {
    glContext_.getIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS, &tsize);
  }
  return static_cast<uint32_t>(tsize);
}

uint32_t DeviceFeatureSet::getMaxComputeUniforms() const {
  IGL_PROFILER_FUNCTION();
  if (hasFeature(DeviceFeatures::Compute)) {
    GLint tsize = 0;
    glContext_.getIntegerv(GL_MAX_COMPUTE_UNIFORM_COMPONENTS, &tsize);
    return static_cast<uint32_t>(tsize);
  }
  return 0;
}

} // namespace igl::opengl
