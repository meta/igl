/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "../util/Common.h"

#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <igl/opengl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>

namespace igl::tests {

//
// DeviceFeatureSetFullOGLTest
//
// Tests for the full DeviceFeatureSet API in OpenGL.
//
class DeviceFeatureSetFullOGLTest : public ::testing::Test {
 public:
  DeviceFeatureSetFullOGLTest() = default;

  void SetUp() override {
    setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
};

//
// HasFeature
//
// Verify that hasFeature returns a boolean without crashing for various features.
//
TEST_F(DeviceFeatureSetFullOGLTest, HasFeature) {
  const auto& features = context_->deviceFeatures();

  // These should all return a boolean without crashing
  bool compute = features.hasFeature(DeviceFeatures::Compute);
  bool multiview = features.hasFeature(DeviceFeatures::Multiview);
  bool texture3d = features.hasFeature(DeviceFeatures::Texture3D);
  bool msaa = features.hasFeature(DeviceFeatures::MultiSample);
  bool uniformBlocks = features.hasFeature(DeviceFeatures::UniformBlocks);
  bool storageBuffers = features.hasFeature(DeviceFeatures::StorageBuffers);
  bool timers = features.hasFeature(DeviceFeatures::Timers);
  bool drawInstanced = features.hasFeature(DeviceFeatures::DrawInstanced);
  bool bindUniform = features.hasFeature(DeviceFeatures::BindUniform);
  bool mrt = features.hasFeature(DeviceFeatures::MultipleRenderTargets);
  bool dynamicVertexBufferStride = features.hasFeature(DeviceFeatures::DynamicVertexBufferStride);

  // Suppress unused variable warnings
  (void)compute;
  (void)multiview;
  (void)texture3d;
  (void)msaa;
  (void)uniformBlocks;
  (void)storageBuffers;
  (void)timers;
  (void)drawInstanced;
  (void)bindUniform;
  (void)mrt;
  (void)dynamicVertexBufferStride;

  // The test passes if no crash occurred
  SUCCEED();
}

//
// HasExtension
//
// Verify that hasExtension returns a boolean without crashing.
//
TEST_F(DeviceFeatureSetFullOGLTest, HasExtension) {
  const auto& features = context_->deviceFeatures();

  bool timerQuery = features.hasExtension(opengl::Extensions::TimerQuery);
  bool vao = features.hasExtension(opengl::Extensions::VertexArrayObject);
  bool fbBlit = features.hasExtension(opengl::Extensions::FramebufferBlit);
  bool mapBuf = features.hasExtension(opengl::Extensions::MapBuffer);
  bool depth24 = features.hasExtension(opengl::Extensions::Depth24);
  bool sync = features.hasExtension(opengl::Extensions::Sync);
  bool srgb = features.hasExtension(opengl::Extensions::Srgb);
  bool debug = features.hasExtension(opengl::Extensions::Debug);

  (void)timerQuery;
  (void)vao;
  (void)fbBlit;
  (void)mapBuf;
  (void)depth24;
  (void)sync;
  (void)srgb;
  (void)debug;

  SUCCEED();
}

//
// HasInternalFeature
//
// Verify that hasInternalFeature returns a boolean without crashing.
//
TEST_F(DeviceFeatureSetFullOGLTest, HasInternalFeature) {
  const auto& features = context_->deviceFeatures();

  bool fbBlit = features.hasInternalFeature(opengl::InternalFeatures::FramebufferBlit);
  bool fbObject = features.hasInternalFeature(opengl::InternalFeatures::FramebufferObject);
  bool vao = features.hasInternalFeature(opengl::InternalFeatures::VertexArrayObject);
  bool mapBuf = features.hasInternalFeature(opengl::InternalFeatures::MapBuffer);
  bool unmapBuf = features.hasInternalFeature(opengl::InternalFeatures::UnmapBuffer);
  bool texStorage = features.hasInternalFeature(opengl::InternalFeatures::TexStorage);
  bool sync = features.hasInternalFeature(opengl::InternalFeatures::Sync);
  bool debugMsg = features.hasInternalFeature(opengl::InternalFeatures::DebugMessage);
  bool invalidateFb = features.hasInternalFeature(opengl::InternalFeatures::InvalidateFramebuffer);
  bool polyFill = features.hasInternalFeature(opengl::InternalFeatures::PolygonFillMode);

  (void)fbBlit;
  (void)fbObject;
  (void)vao;
  (void)mapBuf;
  (void)unmapBuf;
  (void)texStorage;
  (void)sync;
  (void)debugMsg;
  (void)invalidateFb;
  (void)polyFill;

  SUCCEED();
}

//
// ContainsExtensionToken
//
// GL_EXT_shader_framebuffer_fetch is a prefix of GL_EXT_shader_framebuffer_fetch_non_coherent, so
// extensions have to be matched as whole tokens.
//
TEST(ExtensionTokenOGLTest, ContainsExtensionToken) {
  const std::string_view fetch = "GL_EXT_shader_framebuffer_fetch";

  EXPECT_FALSE(
      opengl::containsExtensionToken("GL_EXT_shader_framebuffer_fetch_non_coherent", fetch));
  EXPECT_FALSE(opengl::containsExtensionToken("XGL_EXT_shader_framebuffer_fetch", fetch));
  EXPECT_FALSE(opengl::containsExtensionToken("", fetch));
  EXPECT_FALSE(opengl::containsExtensionToken("GL_OES_depth24", ""));

  EXPECT_TRUE(opengl::containsExtensionToken("GL_EXT_shader_framebuffer_fetch", fetch));
  EXPECT_TRUE(opengl::containsExtensionToken(
      "GL_EXT_shader_framebuffer_fetch_non_coherent GL_EXT_shader_framebuffer_fetch", fetch));
  EXPECT_TRUE(opengl::containsExtensionToken(
      "GL_OES_depth24 GL_EXT_shader_framebuffer_fetch GL_OES_depth32 ", fetch));
}

//
// HasBrokenFramebufferFetch
//
// Adreno 5xx series has broken framebuffer fetch despite advertising the extension. Ensure the
// detection function correctly identifies broken GPUs and rejects others.
//
TEST(ExtensionTokenOGLTest, HasBrokenFramebufferFetch) {
  // Adreno 5xx series: true
  EXPECT_TRUE(opengl::hasBrokenFramebufferFetch("Adreno (TM) 505", "Qualcomm"));
  EXPECT_TRUE(opengl::hasBrokenFramebufferFetch("Adreno (TM) 530", "Qualcomm"));
  EXPECT_TRUE(opengl::hasBrokenFramebufferFetch("Adreno (TM) 540", "Qualcomm"));
  EXPECT_TRUE(opengl::hasBrokenFramebufferFetch("Adreno 530", "Qualcomm"));

  // Adreno outside 5xx range or non-Qualcomm vendor: false
  EXPECT_FALSE(opengl::hasBrokenFramebufferFetch("Adreno (TM) 418", "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenFramebufferFetch("Adreno (TM) 630", "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenFramebufferFetch("Adreno (TM) 740", "Qualcomm"));

  // Non-Adreno GPUs: false
  EXPECT_FALSE(opengl::hasBrokenFramebufferFetch("Mali-G78", "ARM"));
  EXPECT_FALSE(opengl::hasBrokenFramebufferFetch("llvmpipe (LLVM 22.1.3, 256 bits)", "Mesa"));

  // Null renderer or vendor: false
  EXPECT_FALSE(opengl::hasBrokenFramebufferFetch(nullptr, "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenFramebufferFetch("Adreno (TM) 530", nullptr));
}

//
// HasBrokenMultisampledRenderToTexture
//
// Adreno 4xx, 5xx, 615 and 620 drivers advertise the extension but render incorrectly with it.
// Ensure the detection function correctly identifies broken GPUs and rejects others.
//
TEST(ExtensionTokenOGLTest, HasBrokenMultisampledRenderToTexture) {
  // Adreno 4xx and 5xx: true
  EXPECT_TRUE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 400", "Qualcomm"));
  EXPECT_TRUE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 530", "Qualcomm"));
  EXPECT_TRUE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 599", "Qualcomm"));

  // Adreno 615 and 620: true
  EXPECT_TRUE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 615", "Qualcomm"));
  EXPECT_TRUE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 620", "Qualcomm"));
  EXPECT_TRUE(opengl::hasBrokenMultisampledRenderToTexture("Adreno 620", "Qualcomm"));
  EXPECT_TRUE(
      opengl::hasBrokenMultisampledRenderToTexture("Adreno 615", "Qualcomm Technologies, Inc."));

  // Boundary cases: 399 (before 4xx), 600 and 610 (between 5xx and 615), 618 (between 615 and
  // 620), 630 (after 620): false
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 399", "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 600", "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 610", "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 618", "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 630", "Qualcomm"));

  // Only Qualcomm drivers match, so an Adreno renderer string from another vendor (e.g. ANGLE) does
  // not
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 620", "ARM"));

  // Non-Adreno GPUs: false
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Mali-G78", "ARM"));
  EXPECT_FALSE(
      opengl::hasBrokenMultisampledRenderToTexture("llvmpipe (LLVM 22.1.3, 256 bits)", "Mesa"));

  // Null renderer or vendor: false
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture(nullptr, "Qualcomm"));
  EXPECT_FALSE(opengl::hasBrokenMultisampledRenderToTexture("Adreno (TM) 620", nullptr));
}

//
// ShaderFramebufferFetchNeedsCoherentExtension
//
// Extensions::ShaderFramebufferFetch looks for the extension in the GL_EXTENSIONS string that
// OpenGL (ES) below 3.0 reports and in the set of names that it reports from 3.0.
//
TEST_F(DeviceFeatureSetFullOGLTest, ShaderFramebufferFetchNeedsCoherentExtension) {
  const auto hasFetch = [this](std::string extensions,
                               std::unordered_set<std::string> extensionSet) {
    opengl::DeviceFeatureSet features(*context_);
    features.initializeExtensions(std::move(extensions), std::move(extensionSet));
    return features.hasExtension(opengl::Extensions::ShaderFramebufferFetch);
  };
  const std::string nonCoherent = "GL_EXT_shader_framebuffer_fetch_non_coherent";
  const std::string coherent = "GL_EXT_shader_framebuffer_fetch";

  EXPECT_FALSE(hasFetch(nonCoherent, {}));
  EXPECT_TRUE(hasFetch(nonCoherent + " " + coherent, {}));
  EXPECT_FALSE(hasFetch("", {nonCoherent}));
  EXPECT_TRUE(hasFetch("", {nonCoherent, coherent}));
}

//
// GetFeatureLimitsMaxTextureSize
//
// Verify getFeatureLimits returns a sensible value for MaxTextureSize.
//
TEST_F(DeviceFeatureSetFullOGLTest, GetFeatureLimitsMaxTextureSize) {
  size_t maxTextureSize = 0;
  bool hasLimit =
      iglDev_->getFeatureLimits(DeviceFeatureLimits::MaxTextureDimension1D2D, maxTextureSize);
  ASSERT_TRUE(hasLimit);
  // Max texture size should be at least 64 (even the smallest GL implementations support this)
  ASSERT_GE(maxTextureSize, 64u);
}

} // namespace igl::tests
