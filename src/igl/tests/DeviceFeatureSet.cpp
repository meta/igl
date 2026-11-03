/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "util/Common.h"
#if IGL_BACKEND_OPENGL
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/PlatformDevice.h>
#endif // IGL_BACKEND_OPENGL
#include <cstdlib>
#include <string>

namespace igl::tests {

// DeviceFeatureSetTest
// This test exercises the igl::ICapabilities API.

class DeviceFeatureSetTest : public ::testing::Test {
 public:
  DeviceFeatureSetTest() = default;

  // Set up common resources. This will create a device and a command queue
  void SetUp() override {
    setDebugBreakEnabled(false);

    util::createDeviceAndQueue(iglDev_, cmdQueue_);
  }

  void TearDown() override {}

  // Member variables
 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
};

//
// hasFeature
//
// Check DeviceFeatureSet hasFeature list
//
TEST_F(DeviceFeatureSetTest, hasFeatureForMacOSOrWinOrAndroidTest) {
  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::StandardDerivative));

  const bool backendOpenGL = iglDev_->getBackendType() == igl::BackendType::OpenGL;

  if (backendOpenGL) {
#if IGL_BACKEND_OPENGL
    auto& context = iglDev_->getPlatformDevice<igl::opengl::PlatformDevice>()->getContext();
    const bool usesOpenGLES = igl::opengl::DeviceFeatureSet::usesOpenGLES();
    const auto& deviceFeatures = context.deviceFeatures();
    auto glVersion = deviceFeatures.getGLVersion();

    const bool readWriteFramebuffer =
        glVersion >= igl::opengl::GLVersion::v3_0_ES ||
        deviceFeatures.isSupported("GL_ARB_framebuffer_object") ||
        deviceFeatures.isSupported("GL_APPLE_framebuffer_multisample");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::ReadWriteFramebuffer), readWriteFramebuffer);

    const bool texture2DArray =
        (usesOpenGLES ? glVersion >= igl::opengl::GLVersion::v3_0_ES
                      : (glVersion >= igl::opengl::GLVersion::v3_0 ||
                         deviceFeatures.isSupported("GL_EXT_texture_array")));
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::Texture2DArray), texture2DArray);

    const bool texture3D = (usesOpenGLES ? glVersion >= igl::opengl::GLVersion::v3_0_ES
                                         : glVersion >= igl::opengl::GLVersion::v2_0) ||
                           deviceFeatures.isSupported("GL_OES_texture_3D");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::Texture3D), texture3D);

    const bool textureArrayExt = !usesOpenGLES &&
                                 (deviceFeatures.isSupported("GL_EXT_texture_array") ||
                                  deviceFeatures.isSupported("GL_EXT_gpu_shader4"));
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::TextureArrayExt), textureArrayExt);

    const bool textureExternalImage =
        usesOpenGLES && (glVersion >= igl::opengl::GLVersion::v3_0_ES ||
                         deviceFeatures.isSupported("GL_OES_EGL_image_external_essl3") ||
                         deviceFeatures.isSupported("GL_OES_EGL_image_external"));
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::TextureExternalImage), textureExternalImage);

    const bool textureNotPot = (!usesOpenGLES || glVersion >= igl::opengl::GLVersion::v3_0_ES) ||
                               deviceFeatures.isSupported("GL_OES_texture_npot");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::TextureNotPot), textureNotPot);

    const bool multiView = deviceFeatures.isSupported("GL_OVR_multiview2") &&
                           (usesOpenGLES ? glVersion >= igl::opengl::GLVersion::v3_0_ES
                                         : glVersion >= igl::opengl::GLVersion::v3_0);
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::Multiview), multiView);

    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::BindUniform));

    const bool texturePartialMipChain = !usesOpenGLES ||
                                        glVersion >= igl::opengl::GLVersion::v3_0_ES ||
                                        deviceFeatures.isSupported("GL_APPLE_texture_max_level");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::TexturePartialMipChain), texturePartialMipChain);

    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferRing));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferNoCopy));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ShaderLibrary));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindBytes));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferDeviceAddress));

    const bool shaderTextureLod = iglDev_->hasFeature(DeviceFeatures::ShaderTextureLodExt) ||
                                  (usesOpenGLES ? glVersion >= igl::opengl::GLVersion::v3_0_ES
                                                : glVersion >= igl::opengl::GLVersion::v3_0);
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLod), shaderTextureLod);

    const bool shaderTextureLodExt = usesOpenGLES
                                         ? deviceFeatures.isSupported("GL_EXT_shader_texture_lod")
                                         : deviceFeatures.isSupported("GL_ARB_shader_texture_lod");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLodExt), shaderTextureLodExt);

    const bool standardDerivativeExt = usesOpenGLES &&
                                       deviceFeatures.isSupported("GL_OES_standard_derivatives");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::StandardDerivativeExt), standardDerivativeExt);

    const bool supportsSRGB =
        (usesOpenGLES ? (glVersion >= igl::opengl::GLVersion::v3_0_ES ||
                         deviceFeatures.isSupported("GL_EXT_sRGB"))
                      : glVersion >= igl::opengl::GLVersion::v2_1 ||
                            deviceFeatures.isSupported("GL_EXT_texture_sRGB"));
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::SRGB), supportsSRGB);

    const bool hasEGLsRGBSupport = context.eglSupportssRGB();
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::SRGBSwapchain), hasEGLsRGBSupport);

    const bool supportsSRGBWriteControl =
        usesOpenGLES ? deviceFeatures.isSupported("GL_EXT_sRGB_write_control")
                     : glVersion >= igl::opengl::GLVersion::v3_0 ||
                           deviceFeatures.isSupported("GL_ARB_framebuffer_sRGB") ||
                           deviceFeatures.isSupported("GL_EXT_framebuffer_sRGB");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::SRGBWriteControl), supportsSRGBWriteControl);

    const bool samplerMinMaxLod = !usesOpenGLES || glVersion >= igl::opengl::GLVersion::v3_0_ES;
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::SamplerMinMaxLod), samplerMinMaxLod);

    const bool drawIndexedIndirect =
        (usesOpenGLES && glVersion >= igl::opengl::GLVersion::v3_1_ES) ||
        (!usesOpenGLES && glVersion >= igl::opengl::GLVersion::v4_0) ||
        deviceFeatures.isSupported("GL_ARB_draw_indirect");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::DrawIndexedIndirect), drawIndexedIndirect);

    const bool multipleRenderTargets = !usesOpenGLES ||
                                       glVersion >= igl::opengl::GLVersion::v3_0_ES ||
                                       deviceFeatures.isSupported("GL_EXT_draw_buffers");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::MultipleRenderTargets), multipleRenderTargets);

    const bool explicitBinding =
        (usesOpenGLES && glVersion >= igl::opengl::GLVersion::v3_1_ES) ||
        (!usesOpenGLES && (glVersion >= igl::opengl::GLVersion::v4_2 ||
                           deviceFeatures.isSupported("GL_ARB_shading_language_420pack")));
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::ExplicitBinding), explicitBinding);

    const bool explicitBindingExt = deviceFeatures.isSupported("GL_ARB_shading_language_420pack");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::ExplicitBindingExt), explicitBindingExt);

    const bool textureFormatRG =
        (usesOpenGLES && (glVersion >= igl::opengl::GLVersion::v3_0_ES ||
                          deviceFeatures.isSupported("GL_EXT_texture_rg"))) ||
        (!usesOpenGLES && (glVersion >= igl::opengl::GLVersion::v3_0 ||
                           deviceFeatures.isSupported("GL_ARB_texture_rg")));
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::TextureFormatRG), textureFormatRG);
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ValidationLayersEnabled));
    const bool externalMemoryObjects = deviceFeatures.isSupported("GL_EXT_memory_object") &&
                                       deviceFeatures.isSupported("GL_EXT_memory_object_fd");
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::ExternalMemoryObjects), externalMemoryObjects);

    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::PushConstants));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::FillBuffer));

    // From OpenGL (ES) 3.0, which Subpasses needs, isSupported() matches whole extension names, so
    // GL_EXT_shader_framebuffer_fetch_non_coherent alone does not satisfy the fetch check
    const bool drawBuffersIndexed =
        usesOpenGLES ? (glVersion >= igl::opengl::GLVersion::v3_2_ES ||
                        deviceFeatures.isSupported("GL_OES_draw_buffers_indexed") ||
                        deviceFeatures.isSupported("GL_EXT_draw_buffers_indexed"))
                     : glVersion >= igl::opengl::GLVersion::v4_0;
    const auto* renderer = reinterpret_cast<const char*>(context.getString(GL_RENDERER));
    const auto* vendor = reinterpret_cast<const char*>(context.getString(GL_VENDOR));
    const bool subpasses = (usesOpenGLES ? glVersion >= igl::opengl::GLVersion::v3_0_ES
                                         : glVersion >= igl::opengl::GLVersion::v4_0) &&
                           deviceFeatures.isSupported("GL_EXT_shader_framebuffer_fetch") &&
                           drawBuffersIndexed &&
                           !igl::opengl::hasBrokenFramebufferFetch(renderer, vendor);
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::Subpasses), subpasses);
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MemorylessAttachments));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MultiviewSubpasses));

    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DynamicCullMode));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DynamicFrontFacingWinding));
#endif // IGL_BACKEND_OPENGL
  } else if (iglDev_->getBackendType() == igl::BackendType::WebGPU) {
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TextureNotPot));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::SRGB));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderLibrary));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLod));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::SamplerMinMaxLod));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DrawIndexedIndirect));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::MultipleRenderTargets));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ExplicitBinding));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TextureFormatRG));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ValidationLayersEnabled));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ReadWriteFramebuffer));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::SRGBWriteControl));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureArrayExt));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureExternalImage));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Multiview));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindUniform));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferRing));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferNoCopy));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferDeviceAddress));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLodExt));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::StandardDerivativeExt));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExplicitBindingExt));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExternalMemoryObjects));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Subpasses));
    EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MultiviewSubpasses));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DynamicCullMode));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DynamicFrontFacingWinding));
  } else {
    // non OpenGL backends
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ReadWriteFramebuffer));
    EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TextureNotPot));
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::SRGB), true);
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::SRGBSwapchain), true);
    EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::SRGBWriteControl), false);

    if (iglDev_->getBackendType() == igl::BackendType::Vulkan) {
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture2DArray));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture3D));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureArrayExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureExternalImage));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Multiview));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindUniform));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TexturePartialMipChain));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::BufferRing));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferNoCopy));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderLibrary));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindBytes));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::BufferDeviceAddress));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLod));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLodExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::StandardDerivativeExt));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::SamplerMinMaxLod));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DrawIndexedIndirect));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::MultipleRenderTargets));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ExplicitBinding));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExplicitBindingExt));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TextureFormatRG));
      // createInstance() requests the validation layer on neither Android nor macOS, so the
      // feature reports false there in every build mode.
#if IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX || !IGL_DEBUG || defined(IGL_DISABLE_VALIDATION)
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ValidationLayersEnabled));
#else
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ValidationLayersEnabled));
#endif // IGL_PLATFORM_ANDROID || IGL_PLATFORM_MACOSX
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ExternalMemoryObjects));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::PushConstants));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::FillBuffer));
      EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::MultiviewSubpasses),
                iglDev_->hasFeature(DeviceFeatures::Subpasses) &&
                    iglDev_->hasFeature(DeviceFeatures::Multiview));
      // vkCmdSetCullMode and vkCmdSetFrontFace are provided together (core in
      // Vulkan 1.3 / VK_EXT_extended_dynamic_state), so both dynamic-state
      // features must report the same availability.
      EXPECT_EQ(iglDev_->hasFeature(DeviceFeatures::DynamicCullMode),
                iglDev_->hasFeature(DeviceFeatures::DynamicFrontFacingWinding));
    } else if (iglDev_->getBackendType() == igl::BackendType::Metal) {
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture2DArray));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture3D));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureArrayExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureExternalImage));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Multiview));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindUniform));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TexturePartialMipChain));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::BufferRing));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::BufferNoCopy));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderLibrary));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::BindBytes));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferDeviceAddress));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLod));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLodExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::StandardDerivativeExt));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::SamplerMinMaxLod));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DrawIndexedIndirect));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::MultipleRenderTargets));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ExplicitBinding));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExplicitBindingExt));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TextureFormatRG));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ValidationLayersEnabled));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExternalMemoryObjects));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::PushConstants));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::FillBuffer));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MultiviewSubpasses));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DynamicCullMode));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::DynamicFrontFacingWinding));
    } else if (iglDev_->getBackendType() == igl::BackendType::D3D12) {
      // D3D12 backend
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture2DArray));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::Texture3D));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureArrayExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureExternalImage));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Multiview));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindUniform));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TexturePartialMipChain));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferRing));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferNoCopy));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderLibrary));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindBytes));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferDeviceAddress));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLod));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLodExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::StandardDerivativeExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::SamplerMinMaxLod));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::DrawIndexedIndirect));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::MultipleRenderTargets));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ExplicitBinding));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExplicitBindingExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureFormatRG));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ValidationLayersEnabled));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExternalMemoryObjects));
      EXPECT_TRUE(iglDev_->hasFeature(
          DeviceFeatures::PushConstants)); // D3D12 supports push constants via root constants
                                           // (shader register b2)
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::FillBuffer));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Subpasses));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MemorylessAttachments));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MultiviewSubpasses));
    } else {
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Texture2DArray));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Texture3D));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureArrayExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureExternalImage));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Multiview));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BindUniform));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TexturePartialMipChain));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferRing));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferNoCopy));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ShaderLibrary));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::BindBytes));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::BufferDeviceAddress));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLod));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ShaderTextureLodExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::StandardDerivativeExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::SamplerMinMaxLod));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::DrawIndexedIndirect));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MultipleRenderTargets));
      EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::ExplicitBinding));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExplicitBindingExt));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::TextureFormatRG));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ValidationLayersEnabled));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::ExternalMemoryObjects));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::PushConstants));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::FillBuffer));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::Subpasses));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MemorylessAttachments));
      EXPECT_FALSE(iglDev_->hasFeature(DeviceFeatures::MultiviewSubpasses));
    }
  }

  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TextureHalfFloat));
  EXPECT_TRUE(iglDev_->hasFeature(DeviceFeatures::TextureFloat));
}

//
// getTextureFormatCapabilities
//
// Since some of the formats may be platform-dependent, we will only spot
// check a few to make sure the API is working. We should continue to add formats
// here as we see fit.
//
TEST_F(DeviceFeatureSetTest, getTextureFormatCapabilities) {
  ICapabilities::TextureFormatCapabilities capability = 0;

  // RGBA_UNorm8 should be able to do everything except SampledAttachment on all the platforms
  capability = iglDev_->getTextureFormatCapabilities(TextureFormat::RGBA_UNorm8);
  EXPECT_TRUE(contains(capability, ICapabilities::TextureFormatCapabilityBits::Sampled));
  EXPECT_TRUE(contains(capability, ICapabilities::TextureFormatCapabilityBits::Attachment));
  EXPECT_TRUE(contains(capability, ICapabilities::TextureFormatCapabilityBits::SampledFiltered));
  if (iglDev_->hasFeature(DeviceFeatures::Compute)) {
    EXPECT_TRUE(contains(capability, ICapabilities::TextureFormatCapabilityBits::Storage));
  }

  // Z_UNorm16 should always be readable by a shader
  capability = iglDev_->getTextureFormatCapabilities(TextureFormat::Z_UNorm16);
  EXPECT_TRUE(contains(capability, ICapabilities::TextureFormatCapabilityBits::Sampled));
}

//
// ShaderVersion / BackendVersion value semantics
//
// ShaderVersion and BackendVersion are plain value types with defaulted fields
// and hand-written equality operators. They require no device, so the tests use
// a bare fixture and verify the default-constructed state plus that operator== /
// operator!= compare every field independently.
//

TEST(ShaderVersionTest, DefaultValues) {
  const ShaderVersion version;
  EXPECT_EQ(version.family, ShaderFamily::Unknown);
  EXPECT_EQ(version.majorVersion, 0);
  EXPECT_EQ(version.minorVersion, 0);
  EXPECT_EQ(version.extra, 0);
}

TEST(ShaderVersionTest, Equality) {
  const ShaderVersion versionA{
      .family = ShaderFamily::Glsl, .majorVersion = 4, .minorVersion = 6, .extra = 0};
  const ShaderVersion versionB{
      .family = ShaderFamily::Glsl, .majorVersion = 4, .minorVersion = 6, .extra = 0};
  EXPECT_TRUE(versionA == versionB);
  EXPECT_FALSE(versionA != versionB);
}

TEST(ShaderVersionTest, InequalityPerField) {
  const ShaderVersion base{
      .family = ShaderFamily::Metal, .majorVersion = 2, .minorVersion = 4, .extra = 1};
  {
    ShaderVersion other = base;
    other.family = ShaderFamily::SpirV;
    EXPECT_FALSE(base == other);
    EXPECT_TRUE(base != other);
  }
  {
    ShaderVersion other = base;
    other.majorVersion = 3;
    EXPECT_FALSE(base == other);
    EXPECT_TRUE(base != other);
  }
  {
    ShaderVersion other = base;
    other.minorVersion = 5;
    EXPECT_FALSE(base == other);
    EXPECT_TRUE(base != other);
  }
  {
    ShaderVersion other = base;
    other.extra = 2;
    EXPECT_FALSE(base == other);
    EXPECT_TRUE(base != other);
  }
}

TEST(BackendVersionTest, DefaultValues) {
  const BackendVersion version;
  EXPECT_EQ(version.flavor, BackendFlavor::Invalid);
  EXPECT_EQ(version.majorVersion, 0);
  EXPECT_EQ(version.minorVersion, 0);
}

TEST(BackendVersionTest, Equality) {
  const BackendVersion versionA{
      .flavor = BackendFlavor::Vulkan, .majorVersion = 1, .minorVersion = 3};
  const BackendVersion versionB{
      .flavor = BackendFlavor::Vulkan, .majorVersion = 1, .minorVersion = 3};
  EXPECT_TRUE(versionA == versionB);
  EXPECT_FALSE(versionA != versionB);
}

TEST(BackendVersionTest, InequalityPerField) {
  const BackendVersion base{.flavor = BackendFlavor::Metal, .majorVersion = 3, .minorVersion = 0};
  {
    BackendVersion other = base;
    other.flavor = BackendFlavor::OpenGL;
    EXPECT_FALSE(base == other);
    EXPECT_TRUE(base != other);
  }
  {
    BackendVersion other = base;
    other.majorVersion = 2;
    EXPECT_FALSE(base == other);
    EXPECT_TRUE(base != other);
  }
  {
    BackendVersion other = base;
    other.minorVersion = 1;
    EXPECT_FALSE(base == other);
    EXPECT_TRUE(base != other);
  }
}

#if IGL_BACKEND_OPENGL
// classifyGpuTimerTier — pure function tests (no GL context needed)
class GpuTimerTierTest : public ::testing::Test {};

TEST_F(GpuTimerTierTest, NullRenderer) {
  EXPECT_EQ(opengl::classifyGpuTimerTier(nullptr, nullptr), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier(nullptr, "ARM"), opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, AdrenoBudget) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 505", "Qualcomm"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 506", "Qualcomm"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 610", "Qualcomm"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 619", "Qualcomm"),
            opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, AdrenoConservative) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 620", "Qualcomm"),
            opengl::GpuTimerTier::Conservative);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 630", "Qualcomm"),
            opengl::GpuTimerTier::Conservative);
}

TEST_F(GpuTimerTierTest, AdrenoFull) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 640", "Qualcomm"),
            opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 730", "Qualcomm"),
            opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, AdrenoNoTM) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno 610", "Qualcomm"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno 730", "Qualcomm"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, AdrenoVendorCrossCheck) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 505", "ARM"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, AdrenoVendorVariants) {
  // Qualcomm GL drivers report at least three vendor string variants — all
  // must match so budget Adreno devices land in Disabled (not Full default).
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 505", "Qualcomm Technologies, Inc."),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 505", "QUALCOMM"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno (TM) 630", "Qualcomm Technologies, Inc."),
            opengl::GpuTimerTier::Conservative);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Adreno 730", "QUALCOMM"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, MaliBudget) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G31", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G52", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G68", "ARM"), opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, MaliConservative) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G72", "ARM"), opengl::GpuTimerTier::Conservative);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G76", "ARM"), opengl::GpuTimerTier::Conservative);
}

TEST_F(GpuTimerTierTest, MaliFull) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G77", "ARM"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G78", "ARM"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G310", "ARM"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G510", "ARM"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G599", "ARM"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, MaliModernDisabled) {
  for (const char* renderer : {"Mali-G600",
                               "Mali-G610",
                               "Mali-G615",
                               "Mali-G620",
                               "Mali-G625",
                               "Mali-G710",
                               "Mali-G715",
                               "Mali-G720",
                               "Mali-G725",
                               "Mali-G925",
                               "Mali-G610 MC6",
                               "Mali-G715MC6",
                               "Mali-G925 MC6"}) {
    for (const char* vendor : {"ARM", "ARM Limited", "Arm"}) {
      EXPECT_EQ(opengl::classifyGpuTimerTier(renderer, vendor), opengl::GpuTimerTier::Disabled)
          << renderer << ", " << vendor;
    }
  }
}

TEST_F(GpuTimerTierTest, ImmortalisDisabled) {
  for (const char* renderer : {"Immortalis-G715", "Immortalis-G720 MC6", "Immortalis-G925"}) {
    for (const char* vendor : {"ARM", "ARM Limited", "Arm"}) {
      EXPECT_EQ(opengl::classifyGpuTimerTier(renderer, vendor), opengl::GpuTimerTier::Disabled)
          << renderer << ", " << vendor;
    }
  }
}

TEST_F(GpuTimerTierTest, MaliG1Disabled) {
  for (const char* renderer : {"Mali-G1-Ultra", "Mali-G1-Premium", "Mali-G1-Pro"}) {
    for (const char* vendor : {"ARM", "ARM Limited", "Arm"}) {
      EXPECT_EQ(opengl::classifyGpuTimerTier(renderer, vendor), opengl::GpuTimerTier::Disabled)
          << renderer << ", " << vendor;
    }
  }
}

TEST_F(GpuTimerTierTest, ModernMaliVendorCrossCheck) {
  for (const char* renderer : {"Mali-G710", "Immortalis-G715", "Mali-G1-Ultra"}) {
    EXPECT_EQ(opengl::classifyGpuTimerTier(renderer, "Qualcomm"), opengl::GpuTimerTier::Full)
        << renderer;
    EXPECT_EQ(opengl::classifyGpuTimerTier(renderer, "ARM Spoof"), opengl::GpuTimerTier::Full)
        << renderer;
    EXPECT_EQ(opengl::classifyGpuTimerTier(renderer, nullptr), opengl::GpuTimerTier::Full)
        << renderer;
  }
}

TEST_F(GpuTimerTierTest, MaliVendorCrossCheck) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G52", "Qualcomm"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, MaliVendorVariants) {
  // ARM ships at least three vendor string variants in Android GL drivers —
  // same set as the Mali-T branch. Mali-G70+ on some Samsung Exynos builds
  // reports "ARM Limited"; verify all variants land in their intended tier.
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G52", "ARM Limited"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G76", "Arm"), opengl::GpuTimerTier::Conservative);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-G510", "ARM Limited"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, MaliTBudget) {
  // SEV S647462: Mali-T series must be Disabled (broken timer query drivers)
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T720", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T760", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T820", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T860", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T880", "ARM"), opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, MaliTVendorCrossCheck) {
  // Wrong vendor → falls through to Full (default)
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T880", "Qualcomm"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, MaliTWithSuffix) {
  // Real-world Mali-T renderer strings include MP core count and revision
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T880 MP12", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T860MP2", "ARM"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T760 r1p0", "ARM"), opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, MaliTMalformed) {
  // Malformed "Mali-T" without number — sscanf fails, falls through to Full
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T", "ARM"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, MaliTArmVendorVariants) {
  // SEV S647462: ARM ships at least three vendor string variants — accept all
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T880", "ARM Limited"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T760", "Arm"), opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, PowerVRBudget) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR Rogue GE8320", "Imagination Technologies"),
            opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, PowerVRVendorCrossCheck) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR Rogue GE8320", "ARM"),
            opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, PowerVRSgxAndRogueGx) {
  // SGX (legacy MediaTek/TI OMAP) has no hardware timer support; Rogue GX (older
  // MediaTek/Intel Atom) has worse driver quality than GE8. Both are Disabled.
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR SGX 540", "Imagination Technologies"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR Rogue GX6250", "Imagination Technologies"),
            opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, PowerVRImaginationVendor) {
  // Newer Android drivers report vendor as "Imagination" (no "Technologies").
  // Cover all three renderer families (GE8, GX, SGX) against the short vendor.
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR Rogue GE8320", "Imagination"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR Rogue GX6250", "Imagination"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR SGX 540", "Imagination"),
            opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, MaliTNonArmVendorFallsThrough) {
  // Defensive: a Mali-T renderer with a non-ARM vendor string must NOT be
  // Disabled by the Mali-T block — it should fall through to the Full default.
  // Catches a future bug where someone OR's vendor checks across blocks.
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T880", "Imagination"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Mali-T880", "Samsung"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, PowerVRNonImaginationVendorFallsThrough) {
  // Defensive: a PowerVR renderer with a non-Imagination vendor must NOT be
  // Disabled by the PowerVR block — it should fall through to the Full default.
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR Rogue GE8320", "ARM"),
            opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("PowerVR SGX 540", "ARM"), opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, XclipseDisabled) {
  // SEV S647462 follow-up: Conservative (32 slots) was insufficient for
  // Xclipse drivers — all variants treated as Disabled, matching Mali-T.
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 530", "Samsung"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 540", "Samsung"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 920", "Samsung"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 940", "Samsung"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 960", "Samsung"),
            opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, XclipseVendorVariants) {
  // Samsung ships at least three vendor string variants; all must match via
  // the substring check on "Samsung". A bare-renderer "Xclipse N" without the
  // "Samsung" prefix is also accepted in case future driver versions drop it.
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 940", "Samsung Electronics Co., Ltd."),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 940", "Samsung Mobile"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 940", "Samsung Electronics"),
            opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Xclipse 940", "Samsung"), opengl::GpuTimerTier::Disabled);
}

TEST_F(GpuTimerTierTest, XclipseVendorCrossCheck) {
  // Non-Samsung vendor must not match Xclipse — defense against a future GPU
  // sharing the renderer substring.
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 920", "ARM"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Samsung Xclipse 920", "Qualcomm"),
            opengl::GpuTimerTier::Full);
}

TEST_F(GpuTimerTierTest, DesktopAndUnknown) {
  EXPECT_EQ(opengl::classifyGpuTimerTier("NVIDIA GeForce RTX 3090", "NVIDIA Corporation"),
            opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("Apple M1", "Apple"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::classifyGpuTimerTier("", nullptr), opengl::GpuTimerTier::Full);
}
#endif // IGL_BACKEND_OPENGL

#if IGL_BACKEND_OPENGL
TEST_F(GpuTimerTierTest, ParseOverride) {
  EXPECT_EQ(opengl::parseGpuTimerTierOverride("full"), opengl::GpuTimerTier::Full);
  EXPECT_EQ(opengl::parseGpuTimerTierOverride("conservative"), opengl::GpuTimerTier::Conservative);
  EXPECT_EQ(opengl::parseGpuTimerTierOverride("disabled"), opengl::GpuTimerTier::Disabled);
  EXPECT_EQ(opengl::parseGpuTimerTierOverride(nullptr), std::nullopt);
  EXPECT_EQ(opengl::parseGpuTimerTierOverride(""), std::nullopt);
  EXPECT_EQ(opengl::parseGpuTimerTierOverride("Full"), std::nullopt);
  EXPECT_EQ(opengl::parseGpuTimerTierOverride("64"), std::nullopt);
}

#if !IGL_PLATFORM_WINDOWS
namespace {
// Sets or clears kGpuTimerTierOverrideEnvVar for one test and restores its previous value.
class ScopedTimerTierOverride {
 public:
  explicit ScopedTimerTierOverride(const char* IGL_NULLABLE value) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* previous = std::getenv(opengl::kGpuTimerTierOverrideEnvVar);
    if (previous != nullptr) {
      previous_ = std::string(previous);
    }
    set(value);
  }
  ~ScopedTimerTierOverride() {
    set(previous_ ? previous_->c_str() : nullptr);
  }
  ScopedTimerTierOverride(const ScopedTimerTierOverride&) = delete;
  ScopedTimerTierOverride& operator=(const ScopedTimerTierOverride&) = delete;

 private:
  static void set(const char* IGL_NULLABLE value) {
    if (value == nullptr) {
      // NOLINTNEXTLINE(concurrency-mt-unsafe)
      unsetenv(opengl::kGpuTimerTierOverrideEnvVar);
    } else {
      // NOLINTNEXTLINE(concurrency-mt-unsafe)
      setenv(opengl::kGpuTimerTierOverrideEnvVar, value, 1);
    }
  }
  std::optional<std::string> previous_;
};
} // namespace

TEST_F(GpuTimerTierTest, DebugOverrideIsReadFromTheEnvironmentOnlyInDebugBuilds) {
  {
    const ScopedTimerTierOverride scoped("full");
#if IGL_DEBUG
    EXPECT_EQ(opengl::getGpuTimerTierDebugOverride(), opengl::GpuTimerTier::Full);
#else
    EXPECT_EQ(opengl::getGpuTimerTierDebugOverride(), std::nullopt);
#endif
  }
  {
    const ScopedTimerTierOverride scoped("not-a-tier");
    EXPECT_EQ(opengl::getGpuTimerTierDebugOverride(), std::nullopt);
  }
  {
    const ScopedTimerTierOverride scoped(nullptr);
    EXPECT_EQ(opengl::getGpuTimerTierDebugOverride(), std::nullopt);
  }
}

TEST_F(DeviceFeatureSetTest, GpuTimerTierHonorsTheDebugOverride) {
  const auto* platformDevice = iglDev_->getPlatformDevice<igl::opengl::PlatformDevice>();
  if (platformDevice == nullptr) {
    GTEST_SKIP() << "OpenGL only";
  }
  const auto& featureSet = platformDevice->getContext().deviceFeatures();
  const bool hasTimerQuery = featureSet.hasExtension(opengl::Extensions::TimerQuery);
  for (const char* value : {"disabled", "conservative", "full"}) {
    const ScopedTimerTierOverride scoped(value);
    const auto expected = !hasTimerQuery ? opengl::GpuTimerTier::Disabled
#if IGL_DEBUG
                                         : *opengl::parseGpuTimerTierOverride(value);
#else
                                         : featureSet.getGpuTimerTier();
#endif
    EXPECT_EQ(featureSet.getGpuTimerTier(), expected) << value;
    EXPECT_EQ(featureSet.getTimerQueryMaxSlots(), static_cast<uint32_t>(expected)) << value;
  }
}
#endif // !IGL_PLATFORM_WINDOWS
#endif // IGL_BACKEND_OPENGL

} // namespace igl::tests
