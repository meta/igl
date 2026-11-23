/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/Device.h>

#include "../data/ShaderData.h"
#include "../util/Common.h"
#include "../util/TestDevice.h"
#include "../util/TestErrorGuard.h"

#include <algorithm>
#include <igl/CommandBuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/RenderPipelineState.h>
#include <igl/ShaderCreator.h>
#include <igl/VertexInputState.h>

namespace igl::tests {

namespace {
/// Helper to create just any render pipeline in a valid way
std::shared_ptr<IRenderPipelineState> createRenderPipeline(const std::shared_ptr<IDevice>& device,
                                                           Result* result);

/// Just creates any shader
std::shared_ptr<IShaderModule> createShaderModule(const std::shared_ptr<IDevice>& device,
                                                  Result* result);
} // namespace

class DeviceOGLTest : public ::testing::Test {
 public:
  DeviceOGLTest() = default;

  // Set up common resources. This will create a device
  void SetUp() override {
    // Turn off debug break so unit tests can run
    setDebugBreakEnabled(false);

    iglDev_ = util::createTestDevice();
    ASSERT_TRUE(iglDev_ != nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
  }

  void TearDown() override {}

  // Member variables
 protected:
  opengl::IContext* context_{};
  std::shared_ptr<IDevice> iglDev_;
};

/// IGL EndScope() optionally restores several OGL states. This test makes sure those states are
/// restored correctly.
TEST_F(DeviceOGLTest, EndScope) {
#if IGL_PLATFORM_LINUX && !IGL_PLATFORM_LINUX_USE_EGL
  GTEST_SKIP() << "Fix this test on Linux";
#endif

  context_->setUnbindPolicy(opengl::UnbindPolicy::EndScope);

  std::array<GLuint, 2> buffers{};

  // Create a DeviceScope in a new scope to trigger iglDev_->beginScope and iglDev_->endScope when
  // the scope exits and the DeviceScope is destroyed
  {
    const DeviceScope deviceScope(*iglDev_);
    ASSERT_TRUE(iglDev_->verifyScope());

    context_->genBuffers(2, buffers.data());

    // Artificially set values that will be restored when endScope is called
    context_->colorMask(0u, 0u, 0u, 0u);
    context_->blendFunc(GL_SRC_COLOR, GL_DST_COLOR);

    context_->bindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    context_->bindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1]);

    context_->pixelStorei(GL_PACK_ALIGNMENT, 1);
    context_->pixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLboolean mask[4];
    context_->getBooleanv(GL_COLOR_WRITEMASK, mask);

    ASSERT_TRUE(std::all_of(mask, mask + 4, [](GLboolean value) { return !value; }));

    GLint value = 0;
    context_->getIntegerv(GL_BLEND_SRC_RGB, &value);
    ASSERT_EQ(value, GL_SRC_COLOR);

    context_->getIntegerv(GL_BLEND_DST_RGB, &value);
    ASSERT_EQ(value, GL_DST_COLOR);

    context_->getIntegerv(GL_ARRAY_BUFFER_BINDING, &value);
    ASSERT_EQ(value, buffers[0]);

    context_->getIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &value);
    ASSERT_EQ(value, buffers[1]);

    context_->getIntegerv(GL_PACK_ALIGNMENT, &value);
    ASSERT_EQ(value, 1);

    context_->getIntegerv(GL_UNPACK_ALIGNMENT, &value);
    ASSERT_EQ(value, 1);
  }

  // Check whether the correct values are restored from endScope. These are the color mask, blend
  // function, buffer bindings, and pixel storage modes.

  GLboolean mask[4];
  context_->getBooleanv(GL_COLOR_WRITEMASK, mask);

  ASSERT_TRUE(std::all_of(mask, mask + 4, [](GLboolean value) { return value; }));

  GLint value = 0;
  context_->getIntegerv(GL_BLEND_SRC_RGB, &value);
  ASSERT_EQ(value, GL_ONE);

  context_->getIntegerv(GL_BLEND_DST_RGB, &value);
  ASSERT_EQ(value, GL_ZERO);

  context_->getIntegerv(GL_ARRAY_BUFFER_BINDING, &value);
  ASSERT_EQ(value, 0);

  context_->getIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &value);
  ASSERT_EQ(value, 0);

  context_->getIntegerv(GL_PACK_ALIGNMENT, &value);
  ASSERT_EQ(value, 4);

  context_->getIntegerv(GL_UNPACK_ALIGNMENT, &value);
  ASSERT_EQ(value, 4);

  context_->deleteBuffers(2, buffers.data());

  // Check that gl version and shader version works
  auto glVersion = context_->deviceFeatures().getGLVersion();
  ASSERT_NE(glVersion, opengl::GLVersion::NotAvailable);

  auto shaderVersion = iglDev_->getShaderVersion();
  ASSERT_NE(shaderVersion.majorVersion, 0);
#if IGL_OPENGL_ES
  ASSERT_EQ(shaderVersion.family, ShaderFamily::GlslEs);
#else
  ASSERT_EQ(shaderVersion.family, ShaderFamily::Glsl);
#endif
  ASSERT_GT(shaderVersion.majorVersion + shaderVersion.minorVersion, 0);

#if IGL_BACKEND_OPENGL
  const std::string expectedVersion =
      "#version " + std::to_string(shaderVersion.majorVersion) +
      (shaderVersion.minorVersion == 0 ? "00" : std::to_string(shaderVersion.minorVersion)) +
      ((shaderVersion.family == ShaderFamily::GlslEs && shaderVersion.majorVersion > 1) ? " es"
                                                                                        : "");

  ASSERT_EQ(opengl::getStringFromShaderVersion(shaderVersion), expectedVersion);
#endif // IGL_BACKEND_OPENGL
}

TEST_F(DeviceOGLTest, EndScopeClearContext) {
  context_->setUnbindPolicy(opengl::UnbindPolicy::ClearContext);

  { // Clear current context, one level deep
    context_->clearCurrentContext();

    const DeviceScope deviceScope(*iglDev_);
    ASSERT_TRUE(iglDev_->verifyScope());
    ASSERT_TRUE(context_->isCurrentContext());
  }
  ASSERT_FALSE(context_->isCurrentContext());

  { // Clear current context, one level deep
    const DeviceScope scope1(*iglDev_);
    ASSERT_TRUE(iglDev_->verifyScope());
    ASSERT_TRUE(context_->isCurrentContext());
    {
      const DeviceScope scope2(*iglDev_);
      ASSERT_TRUE(iglDev_->verifyScope());
      ASSERT_TRUE(context_->isCurrentContext());
    }
    // Scope destroyed - validate that it's no longer current
    ASSERT_TRUE(iglDev_->verifyScope());
    ASSERT_TRUE(context_->isCurrentContext());
  }
  ASSERT_FALSE(context_->isCurrentContext());
}

TEST_F(DeviceOGLTest, DeletionTest) {
  const util::TestErrorGuard testErrorGuard;

  auto iglDev2 = util::createTestDevice();

  std::unique_ptr<IBuffer> buffer;
  std::shared_ptr<IFramebuffer> framebuffer;
  std::shared_ptr<ITexture> texture;
  std::shared_ptr<ITexture> renderbufferTexture;
  std::unique_ptr<IRenderCommandEncoder> renderCommandEncoder; // Used to hold onto a VAO if they're
                                                               // enabled
  std::shared_ptr<IRenderPipelineState> renderPipelineState; // Holds onto ShaderStages which will
  // call deleteProgram

  std::shared_ptr<IShaderModule> shaderModule; // Used to trigger deleteShader

  {
    const DeviceScope scope1(*iglDev_);

    // deleteBuffers
    const BufferDesc desc{
        .type = BufferDesc::BufferTypeBits::Vertex,
        .length = 0,
        .storage = ResourceStorage::Shared,
    };
    Result res;
    buffer = iglDev_->createBuffer(desc, &res);
    ASSERT_EQ(res.code, Result::Code::Ok);
    ASSERT_EQ(buffer->getSizeInBytes(), 0);

    // Create an offscreen texture to render to
    TextureDesc texDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                             5,
                                             5,
                                             TextureDesc::TextureUsageBits::Sampled |
                                                 TextureDesc::TextureUsageBits::Attachment);

    Result ret;
    texture = iglDev_->createTexture(texDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(texture != nullptr);

    const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = texture}}};
    framebuffer = iglDev_->createFramebuffer(framebufferDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(framebuffer != nullptr);

    texDesc.usage = TextureDesc::TextureUsageBits::Attachment;
    renderbufferTexture = iglDev_->createTexture(texDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(renderbufferTexture != nullptr);

    // To get it to create a VAO and then delete it
    const CommandQueueDesc cqDesc = {};
    auto cq = iglDev_->createCommandQueue(cqDesc, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(cq != nullptr); // Shouldn't trigger if above is okay
    auto cmd = cq->createCommandBuffer({}, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    RenderPassDesc renderPassDesc;
    renderPassDesc = {
        .colorAttachments = {{.loadAction = LoadAction::Clear,
                              .storeAction = StoreAction::Store,
                              .clearColor = {0.0, 0.0, 0.0, 1.0}}},
    };

    renderCommandEncoder = cmd->createRenderCommandEncoder(renderPassDesc, framebuffer);
    ASSERT_TRUE(renderCommandEncoder != nullptr);

    renderPipelineState = createRenderPipeline(iglDev_, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(renderPipelineState != nullptr);

    shaderModule = createShaderModule(iglDev_, &ret);
    ASSERT_EQ(ret.code, Result::Code::Ok);
    ASSERT_TRUE(shaderModule != nullptr);
  }

  // Force scope to change (workaround for leaving device scope not clearing current eagl scope)
  {
    const DeviceScope scope2(*iglDev2);
    ASSERT_TRUE(iglDev2->verifyScope());
    ASSERT_FALSE(iglDev_->verifyScope());
  }

  // This should make all the resources deallocate while not in scope.
  buffer = nullptr;
  framebuffer = nullptr;
  texture = nullptr;
  renderbufferTexture = nullptr;
  renderCommandEncoder = nullptr;
  renderPipelineState = nullptr;
  shaderModule = nullptr;

  // Entering main scope again to flush deletion queue
  {
    const DeviceScope scope3(*iglDev_);
  }
}

namespace {
std::shared_ptr<IRenderPipelineState> createRenderPipeline(const std::shared_ptr<IDevice>& device,
                                                           Result* outResult) {
  Result ret;

  RenderPipelineDesc renderPipelineDesc;

  // Initialize shader stages
  std::unique_ptr<IShaderStages> stages;
  igl::tests::util::createSimpleShaderStages(device, stages);
  renderPipelineDesc.shaderStages = std::move(stages);

  // Initialize input to vertex shader
  // numAttributes has to equal numInputBindings when using more than one buffer
  const VertexInputStateDesc inputDesc{
      .numAttributes = 2,
      .attributes = {{.bufferIndex = data::shader::kSimplePosIndex,
                      .format = VertexAttributeFormat::Float4,
                      .offset = 0,
                      .name = std::string(data::shader::kSimplePos),
                      .location = 0},
                     {.bufferIndex = data::shader::kSimpleUvIndex,
                      .format = VertexAttributeFormat::Float2,
                      .offset = 0,
                      .name = std::string(data::shader::kSimpleUv),
                      .location = 1}},
      .numInputBindings = 2,
      .inputBindings = {{.stride = sizeof(float) * 4}, {.stride = sizeof(float) * 2}},
  };

  auto vertexInputState = device->createVertexInputState(inputDesc, &ret);
  if (!ret.isOk()) {
    Result::setResult(outResult, ret.code, ret.message);
    return nullptr;
  }
  renderPipelineDesc.vertexInputState = vertexInputState;

  auto renderPipelineState = device->createRenderPipeline(renderPipelineDesc, &ret);
  if (!ret.isOk()) {
    Result::setResult(outResult, ret.code, ret.message);
    return nullptr;
  }

  return renderPipelineState;
}

std::shared_ptr<IShaderModule> createShaderModule(const std::shared_ptr<IDevice>& device,
                                                  Result* outResult) {
  Result ret;
  auto vertShader = ShaderModuleCreator::fromStringInput(
      *device,
      data::shader::kOglSimpleVertShader
          .data(), // NOLINT(bugprone-suspicious-stringview-data-usage)
      {.stage = ShaderStage::Vertex, .entryPoint = "main"},
      "",
      &ret);
  if (!ret.isOk()) {
    Result::setResult(outResult, ret.code, ret.message);
    return nullptr;
  }

  return vertShader;
}
} // namespace

TEST_F(DeviceOGLTest, CreateShaderModuleUnknownTypeFails) {
  Result ret;
  auto vertShader = ShaderModuleCreator::fromStringInput(
      *iglDev_,
      data::shader::kOglSimpleVertShader
          .data(), // NOLINT(bugprone-suspicious-stringview-data-usage)
      // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
      {.stage = static_cast<ShaderStage>(99), .entryPoint = "main"},
      "",
      &ret);
  EXPECT_FALSE(ret.isOk()) << "invalid stage to compile should result in failure";
  EXPECT_TRUE(vertShader == nullptr) << "invalid stage to compile should result in null result";
}

} // namespace igl::tests
