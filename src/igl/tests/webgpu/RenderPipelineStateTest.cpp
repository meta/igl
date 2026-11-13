/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/webgpu/RenderPipelineState.h>

#include <memory>
#include <string>
#include <vector>
#include <igl/ShaderCreator.h>
#include <igl/tests/data/ShaderData.h>
#include <igl/tests/util/Common.h>
#include <igl/tests/util/device/webgpu/TestDevice.h>
#include <igl/webgpu/BindLayouts.h>

namespace igl::tests {

namespace {

webgpu::WgslBinding declaration(
    webgpu::WgslBindingKind kind,
    webgpu::WgslSampledType sampledType = webgpu::WgslSampledType::Float) {
  return {.kind = kind, .viewDimension = WGPUTextureViewDimension_2D, .sampledType = sampledType};
}

constexpr const char* kDepthCompareFragment = R"(
@group(0) @binding(0) var shadowMap : texture_depth_2d;
@group(0) @binding(1) var shadowSampler : sampler_comparison;

@fragment
fn main(@location(0) uv : vec2f) -> @location(0) vec4f {
  return vec4f(textureSampleCompare(shadowMap, shadowSampler, uv, 0.5));
}
)";

std::string makeUniformsFragment(int numBuffers) {
  std::string source = "struct U { v : vec4f, };\n";
  std::string sum = "vec4f(0.0)";
  for (int i = 0; i < numBuffers; ++i) {
    source += "@group(1) @binding(" + std::to_string(i) + ") var<uniform> u" + std::to_string(i) +
              " : U;\n";
    sum += " + u" + std::to_string(i) + ".v";
  }
  return source + "@fragment fn main(@location(0) uv : vec2f) -> @location(0) vec4f {\n  return " +
         sum + ";\n}\n";
}

} // namespace

TEST(WebGPUBindLayoutsTest, SampleClassTable) {
  using webgpu::SampleClass;
  using webgpu::WgslBindingKind;
  using webgpu::WgslSampledType;
  const auto f32 = declaration(WgslBindingKind::Texture);
  const auto u32 = declaration(WgslBindingKind::Texture, WgslSampledType::Uint);
  const auto depth = declaration(WgslBindingKind::DepthTexture);
  const auto multisampled = declaration(WgslBindingKind::MultisampledTexture);

  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::RGBA_UNorm8, false), SampleClass::Float);
  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::RGBA_F16, false), SampleClass::Float);
  // r32float is filterable only with float32-filterable.
  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::R_F32, false),
            SampleClass::UnfilterableFloat);
  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::R_F32, true), SampleClass::Float);
  // Depth formats bound to f32 declarations are sampled through an unfilterable depth aspect.
  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::Z_UNorm32, true),
            SampleClass::UnfilterableFloat);
  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::S8_UInt_Z24_UNorm, true),
            SampleClass::UnfilterableFloat);
  EXPECT_EQ(webgpu::getSampleClass(depth, TextureFormat::Z_UNorm16, true), SampleClass::Depth);
  EXPECT_EQ(webgpu::getSampleClass(u32, TextureFormat::R_UInt32, true), SampleClass::Uint);
  EXPECT_EQ(webgpu::getSampleClass(u32, TextureFormat::S_UInt8, true), SampleClass::Uint);
  EXPECT_EQ(webgpu::getSampleClass(multisampled, TextureFormat::RGBA_UNorm8, true),
            SampleClass::UnfilterableFloat);
  // Mismatches cannot be bound.
  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::R_UInt32, true), std::nullopt);
  EXPECT_EQ(webgpu::getSampleClass(f32, TextureFormat::S_UInt8, true), std::nullopt);
  EXPECT_EQ(webgpu::getSampleClass(u32, TextureFormat::RGBA_UNorm8, true), std::nullopt);
  EXPECT_EQ(webgpu::getSampleClass(depth, TextureFormat::RGBA_UNorm8, true), std::nullopt);
  EXPECT_EQ(webgpu::getSampleClass(declaration(WgslBindingKind::Texture, WgslSampledType::Sint),
                                   TextureFormat::R_UInt32,
                                   true),
            std::nullopt);
}

TEST(WebGPUBindLayoutsTest, SampleClassPacking) {
  webgpu::SampleClasses classes = 0;
  classes = webgpu::setSampleClass(classes, 3, webgpu::SampleClass::Depth);
  classes = webgpu::setSampleClass(classes, 15, webgpu::SampleClass::Uint);
  EXPECT_EQ(webgpu::getSampleClass(classes, 3), webgpu::SampleClass::Depth);
  EXPECT_EQ(webgpu::getSampleClass(classes, 15), webgpu::SampleClass::Uint);
  EXPECT_EQ(webgpu::getSampleClass(classes, 0), webgpu::SampleClass::Float);
  classes = webgpu::setSampleClass(classes, 3, webgpu::SampleClass::Float);
  EXPECT_EQ(webgpu::getSampleClass(classes, 3), webgpu::SampleClass::Float);
}

TEST(WebGPUBindLayoutsTest, LayoutEntriesFollowSampleClasses) {
  webgpu::WgslReflection reflection;
  ASSERT_TRUE(webgpu::parseWgslReflection(data::shader::kWgslSimpleFragShader, reflection).isOk());
  webgpu::PipelineBindings bindings;
  ASSERT_TRUE(bindings.add(reflection, WGPUShaderStage_Fragment).isOk());
  EXPECT_EQ(bindings.getTextureUnitMask(), 1u);

  auto entries = webgpu::makeBindGroupLayoutEntries(bindings, 0, 0);
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].texture.sampleType, WGPUTextureSampleType_Float);
  EXPECT_EQ(entries[0].visibility, WGPUShaderStage_Fragment);
  EXPECT_EQ(entries[1].sampler.type, WGPUSamplerBindingType_Filtering);

  entries = webgpu::makeBindGroupLayoutEntries(
      bindings, 0, webgpu::setSampleClass(0, 0, webgpu::SampleClass::UnfilterableFloat));
  EXPECT_EQ(entries[0].texture.sampleType, WGPUTextureSampleType_UnfilterableFloat);
  EXPECT_EQ(entries[1].sampler.type, WGPUSamplerBindingType_NonFiltering);

  webgpu::WgslReflection depthReflection;
  ASSERT_TRUE(webgpu::parseWgslReflection(kDepthCompareFragment, depthReflection).isOk());
  webgpu::PipelineBindings depthBindings;
  ASSERT_TRUE(depthBindings.add(depthReflection, WGPUShaderStage_Fragment).isOk());
  entries =
      webgpu::makeBindGroupLayoutEntries(depthBindings, 0, depthBindings.getDefaultSampleClasses());
  EXPECT_EQ(entries[0].texture.sampleType, WGPUTextureSampleType_Depth);
  EXPECT_EQ(entries[1].sampler.type, WGPUSamplerBindingType_Comparison);
}

TEST(WebGPUBindLayoutsTest, ConventionViolations) {
  webgpu::WgslReflection reflection;
  webgpu::PipelineBindings bindings;
  ASSERT_TRUE(
      webgpu::parseWgslReflection("@group(0) @binding(0) var s : sampler;", reflection).isOk());
  EXPECT_EQ(bindings.add(reflection, WGPUShaderStage_Fragment).code, Result::Code::ArgumentInvalid);

  ASSERT_TRUE(
      webgpu::parseWgslReflection("@group(1) @binding(0) var t : texture_2d<f32>;", reflection)
          .isOk());
  EXPECT_EQ(webgpu::PipelineBindings().add(reflection, WGPUShaderStage_Fragment).code,
            Result::Code::ArgumentInvalid);

  ASSERT_TRUE(
      webgpu::parseWgslReflection("@group(2) @binding(0) var t : texture_2d<f32>;", reflection)
          .isOk());
  EXPECT_EQ(webgpu::PipelineBindings().add(reflection, WGPUShaderStage_Fragment).code,
            Result::Code::ArgumentInvalid);

  // Storage textures are group 2; writable ones are not visible to vertex shaders.
  ASSERT_TRUE(
      webgpu::parseWgslReflection(
          "@group(2) @binding(0) var t : texture_storage_2d<rgba8unorm, write>;", reflection)
          .isOk());
  webgpu::PipelineBindings storage;
  ASSERT_TRUE(storage.add(reflection, WGPUShaderStage_Vertex | WGPUShaderStage_Fragment).isOk());
  const auto storageEntries = webgpu::makeBindGroupLayoutEntries(storage, 2, 0);
  ASSERT_EQ(storageEntries.size(), 1u);
  EXPECT_EQ(storageEntries[0].visibility, WGPUShaderStage_Fragment);
  EXPECT_EQ(storageEntries[0].storageTexture.format, WGPUTextureFormat_RGBA8Unorm);
  EXPECT_EQ(storageEntries[0].storageTexture.access, WGPUStorageTextureAccess_WriteOnly);
  // Storage texture bindings end at 15.
  ASSERT_TRUE(
      webgpu::parseWgslReflection(
          "@group(2) @binding(15) var t : texture_storage_2d<rgba8unorm, write>;", reflection)
          .isOk());
  EXPECT_TRUE(webgpu::PipelineBindings().add(reflection, WGPUShaderStage_Compute).isOk());
  ASSERT_TRUE(
      webgpu::parseWgslReflection(
          "@group(2) @binding(16) var t : texture_storage_2d<rgba8unorm, write>;", reflection)
          .isOk());
  const Result pastTheEnd = webgpu::PipelineBindings().add(reflection, WGPUShaderStage_Compute);
  EXPECT_EQ(pastTheEnd.code, Result::Code::ArgumentInvalid);
  EXPECT_NE(pastTheEnd.message.find("storage textures are bindings 0-15"), std::string::npos)
      << pastTheEnd.message;

  // Group 3 holds only the push-constant uniform buffer at binding 0, of at most 128 bytes.
  ASSERT_TRUE(
      webgpu::parseWgslReflection("@group(3) @binding(1) var<uniform> pc : vec4f;", reflection)
          .isOk());
  EXPECT_EQ(webgpu::PipelineBindings().add(reflection, WGPUShaderStage_Fragment).code,
            Result::Code::ArgumentInvalid);
  ASSERT_TRUE(webgpu::parseWgslReflection(
                  "@group(3) @binding(0) var<storage, read> pc : array<f32>;", reflection)
                  .isOk());
  EXPECT_EQ(webgpu::PipelineBindings().add(reflection, WGPUShaderStage_Fragment).code,
            Result::Code::ArgumentInvalid);
  ASSERT_TRUE(webgpu::parseWgslReflection(
                  "@group(3) @binding(0) var<uniform> pc : array<vec4f, 9>;", reflection)
                  .isOk());
  EXPECT_EQ(webgpu::PipelineBindings().add(reflection, WGPUShaderStage_Fragment).code,
            Result::Code::ArgumentInvalid);

  // Two stages that disagree about a binding.
  webgpu::WgslReflection a;
  webgpu::WgslReflection b;
  ASSERT_TRUE(
      webgpu::parseWgslReflection("@group(0) @binding(0) var t : texture_2d<f32>;", a).isOk());
  ASSERT_TRUE(
      webgpu::parseWgslReflection("@group(0) @binding(0) var t : texture_cube<f32>;", b).isOk());
  webgpu::PipelineBindings merged;
  ASSERT_TRUE(merged.add(a, WGPUShaderStage_Vertex).isOk());
  EXPECT_EQ(merged.add(b, WGPUShaderStage_Fragment).code, Result::Code::ArgumentInvalid);
}

TEST(WebGPUBindLayoutsTest, DynamicUniformBudget) {
  webgpu::WgslReflection reflection;
  ASSERT_TRUE(webgpu::parseWgslReflection(makeUniformsFragment(9), reflection).isOk());
  webgpu::PipelineBindings bindings;
  ASSERT_TRUE(bindings.add(reflection, WGPUShaderStage_Fragment).isOk());
  bindings.assignDynamicOffsets(8);
  ASSERT_EQ(bindings.groups[webgpu::kBufferGroup].size(), 9u);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(bindings.groups[webgpu::kBufferGroup][i].hasDynamicOffset, i < 8) << i;
  }

  // The push-constant buffer takes one dynamic slot first.
  webgpu::WgslReflection pushConstants;
  ASSERT_TRUE(
      webgpu::parseWgslReflection("@group(3) @binding(0) var<uniform> pc : vec4f;", pushConstants)
          .isOk());
  ASSERT_TRUE(bindings.add(pushConstants, WGPUShaderStage_Fragment).isOk());
  bindings.assignDynamicOffsets(8);
  EXPECT_TRUE(bindings.groups[webgpu::kPushConstantGroup][0].hasDynamicOffset);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(bindings.groups[webgpu::kBufferGroup][i].hasDynamicOffset, i < 7) << i;
  }
}

class WebGPURenderPipelineStateTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
    auto device = util::device::webgpu::createTestDevice();
    ASSERT_NE(device, nullptr);
    webgpuDevice_ = device.get();
    device_ = std::move(device);
    Result ret;
    vertexInput_ = device_->createVertexInputState(
        {
            .numAttributes = 2,
            .attributes = {{.bufferIndex = 0,
                            .format = VertexAttributeFormat::Float4,
                            .offset = 0,
                            .location = 0},
                           {.bufferIndex = 1,
                            .format = VertexAttributeFormat::Float2,
                            .offset = 0,
                            .location = 1}},
            .numInputBindings = 2,
            .inputBindings = {{.stride = 16}, {.stride = 8}},
        },
        &ret);
    ASSERT_TRUE(ret.isOk()) << ret.message;
  }

  void TearDown() override {
    EXPECT_EQ(webgpuDevice_->getContext().getUncapturedErrorCount(), 0u);
  }

 protected:
  [[nodiscard]] std::shared_ptr<webgpu::RenderPipelineState> create(RenderPipelineDesc desc,
                                                                    Result* outResult) {
    if (!desc.vertexInputState) {
      desc.vertexInputState = vertexInput_;
    }
    return std::static_pointer_cast<webgpu::RenderPipelineState>(
        device_->createRenderPipeline(desc, outResult));
  }

  [[nodiscard]] std::shared_ptr<IShaderStages> stages(const char* fragmentSource) const {
    Result ret;
    auto result = ShaderStagesCreator::fromModuleStringInput(
        *device_,
        std::string(data::shader::kWgslSimpleVertShader).c_str(),
        "main",
        "",
        fragmentSource,
        "main",
        "",
        &ret);
    EXPECT_TRUE(ret.isOk()) << ret.message;
    return result;
  }

  std::shared_ptr<IDevice> device_;
  webgpu::Device* webgpuDevice_ = nullptr;
  std::shared_ptr<IVertexInputState> vertexInput_;
};

// Every simple WGSL shader pair against explicit layouts, for every renderable color format and
// every depth/stencil format.
TEST_F(WebGPURenderPipelineStateTest, SimpleShadersForEveryRenderableFormat) {
  size_t numColorFormats = 0;
  for (auto i = static_cast<uint8_t>(TextureFormat::Invalid);
       i <= static_cast<uint8_t>(TextureFormat::B10G11R11_UFloat);
       ++i) {
    const auto format = static_cast<TextureFormat>(i);
    const auto caps = device_->getTextureFormatCapabilities(format);
    if ((caps & ICapabilities::TextureFormatCapabilityBits::Attachment) == 0) {
      continue;
    }
    const auto props = TextureFormatProperties::fromTextureFormat(format);
    std::unique_ptr<IShaderStages> shaderStages;
    util::createSimpleShaderStages(
        device_, shaderStages, props.isDepthOrStencil() ? TextureFormat::RGBA_UNorm8 : format);
    ASSERT_NE(shaderStages, nullptr) << props.name;
    RenderPipelineDesc desc{.shaderStages = std::move(shaderStages)};
    if (props.isDepthOrStencil()) {
      desc.targetDesc.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}};
      desc.targetDesc.depthAttachmentFormat = props.hasDepth() ? format : TextureFormat::Invalid;
      desc.targetDesc.stencilAttachmentFormat = props.hasStencil() ? format
                                                                   : TextureFormat::Invalid;
    } else {
      desc.targetDesc.colorAttachments = {{.textureFormat = format}};
      ++numColorFormats;
    }
    Result ret;
    auto pipeline = create(desc, &ret);
    EXPECT_TRUE(ret.isOk()) << props.name << ": " << ret.message;
    EXPECT_NE(pipeline, nullptr) << props.name;
  }
  EXPECT_GE(numColorFormats, 20u);
}

TEST_F(WebGPURenderPipelineStateTest, VariantsAreCreatedOncePerState) {
  Result ret;
  auto pipeline = create(
      {
          .shaderStages = stages(std::string(data::shader::kWgslSimpleFragShader).c_str()),
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}},
                         .depthAttachmentFormat = TextureFormat::Z_UNorm32},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(pipeline, nullptr);
  EXPECT_TRUE(pipeline->hasDepthStencilAttachment());
  // The default variant is created up front.
  EXPECT_EQ(pipeline->getPipelineCreationCount(), 1u);

  webgpu::RenderPipelineDynamicState state = pipeline->getDefaultDynamicState();
  EXPECT_NE(pipeline->getPipeline(state, &ret), nullptr);
  EXPECT_EQ(pipeline->getPipelineCreationCount(), 1u);

  state.setDepthStencilState(
      {.compareFunction = CompareFunction::Less, .isDepthWriteEnabled = true});
  WGPURenderPipeline depthVariant = pipeline->getPipeline(state, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_NE(depthVariant, nullptr);
  EXPECT_EQ(pipeline->getPipeline(state, &ret), depthVariant);
  EXPECT_EQ(pipeline->getPipelineCreationCount(), 2u);
  EXPECT_EQ(state.getDepthStencilState().compareFunction, CompareFunction::Less);

  // Past the soft cap, variants still work.
  for (int i = 0; i < 70; ++i) {
    state.depthBias = static_cast<float>(i + 1);
    EXPECT_NE(pipeline->getPipeline(state, &ret), nullptr) << ret.message;
  }
  EXPECT_EQ(pipeline->getVariantCount(), 72u);
}

TEST_F(WebGPURenderPipelineStateTest, UnfilterableAndStripVariants) {
  Result ret;
  auto pipeline = create(
      {
          .topology = PrimitiveType::TriangleStrip,
          .shaderStages = stages(std::string(data::shader::kWgslSimpleFragShader).c_str()),
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  webgpu::RenderPipelineDynamicState state = pipeline->getDefaultDynamicState();
  state.sampleClasses =
      webgpu::setSampleClass(state.sampleClasses, 0, webgpu::SampleClass::UnfilterableFloat);
  EXPECT_NE(pipeline->getPipeline(state, &ret), nullptr) << ret.message;
  state.stripIndexFormat = static_cast<uint8_t>(IndexFormat::UInt16) + 1;
  EXPECT_NE(pipeline->getPipeline(state, &ret), nullptr) << ret.message;
  EXPECT_EQ(pipeline->getPipelineCreationCount(), 3u);
}

TEST_F(WebGPURenderPipelineStateTest, DepthComparisonSampler) {
  Result ret;
  auto pipeline = create(
      {
          .shaderStages = stages(kDepthCompareFragment),
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
      },
      &ret);
  EXPECT_TRUE(ret.isOk()) << ret.message;
  EXPECT_NE(pipeline, nullptr);
}

TEST_F(WebGPURenderPipelineStateTest, ReflectionAndIndexByName) {
  Result ret;
  auto pipeline = create(
      {
          .shaderStages = stages(makeUniformsFragment(2).c_str()),
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(pipeline->renderPipelineReflection(), nullptr);
  EXPECT_EQ(pipeline->renderPipelineReflection()->allUniformBuffers().size(), 2u);
  EXPECT_EQ(pipeline->getIndexByName("u1", ShaderStage::Fragment), 1);
  EXPECT_EQ(pipeline->getIndexByName(IGL_NAMEHANDLE("u0"), ShaderStage::Fragment), 0);
  EXPECT_EQ(pipeline->getIndexByName("u1", ShaderStage::Vertex), -1);
}

TEST_F(WebGPURenderPipelineStateTest, BindingsAndBindGroupLayouts) {
  Result ret;
  auto pipeline = create(
      {
          .shaderStages = stages(makeUniformsFragment(2).c_str()),
          .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
      },
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const webgpu::PipelineBindings& bindings = pipeline->getBindings();
  EXPECT_TRUE(bindings.groups[0].empty());
  EXPECT_EQ(bindings.groups[1].size(), 2u);
  EXPECT_NE(bindings.find(1, 1), nullptr);
  EXPECT_EQ(bindings.find(1, 2), nullptr);

  const webgpu::SampleClasses classes = bindings.getDefaultSampleClasses();
  WGPUBindGroupLayout layout = pipeline->getBindGroupLayout(1, classes, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_NE(layout, nullptr);
  // Layouts are cached.
  EXPECT_EQ(pipeline->getBindGroupLayout(1, classes, &ret), layout);
}

TEST_F(WebGPURenderPipelineStateTest, InvalidDescriptors) {
  Result ret;
  EXPECT_EQ(create({}, &ret), nullptr);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);

  const std::string fragment(data::shader::kWgslSimpleFragShader);
  EXPECT_EQ(create(
                {
                    .shaderStages = stages(fragment.c_str()),
                    .targetDesc = {.colorAttachments = {{.textureFormat =
                                                             TextureFormat::RGBA_BC7_UNORM_4x4}}},
                },
                &ret),
            nullptr);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);

  EXPECT_EQ(
      create(
          {
              .shaderStages = stages(fragment.c_str()),
              .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}},
                             .depthAttachmentFormat = TextureFormat::Z_UNorm32,
                             .stencilAttachmentFormat = TextureFormat::S_UInt8},
          },
          &ret),
      nullptr);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);

  EXPECT_EQ(
      create(
          {
              .shaderStages = stages(fragment.c_str()),
              .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::RGBA_UNorm8}}},
              .sampleCount = 2,
          },
          &ret),
      nullptr);
  EXPECT_EQ(ret.code, Result::Code::Unsupported);

  // The fragment shader writes vec4f, which a uint target rejects.
  EXPECT_EQ(
      create(
          {
              .shaderStages = stages(fragment.c_str()),
              .targetDesc = {.colorAttachments = {{.textureFormat = TextureFormat::R_UInt32}}},
          },
          &ret),
      nullptr);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
}

} // namespace igl::tests
