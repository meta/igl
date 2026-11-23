/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <shell/renderSessions/ComputeSession.h>

#include <IGLU/simdtypes/SimdTypes.h>
#include <IGLU/uniform/Descriptor.h>
#include <IGLU/uniform/Encoder.h>
#include <shell/renderSessions/shaderCode/generated/ComputeSessionFragShaderProvider.h>
#include <shell/renderSessions/shaderCode/generated/ComputeSessionGrayscaleCompShaderProvider.h>
#include <shell/renderSessions/shaderCode/generated/ComputeSessionVertShaderProvider.h>
#include <shell/renderSessions/shaderCode/generated/ComputeSessionVertexTransformCompShaderProvider.h>
#include <shell/renderSessions/wgsl/ComputeSessionWgsl.h>
#include <shell/shared/imageLoader/ImageLoader.h>
#include <shell/shared/renderSession/ShaderStagesCreator.h>
#include <igl/Buffer.h>
#include <igl/NameHandle.h>

namespace igl::shell {

namespace {

// ComputeSessionGrayscale.comp.sparksl. Its loose `grayscale` uniform is the bindBytes() block at
// buffer 0, which SPIR-V from SparkSL leaves out, so this one is written by hand.
constexpr const char* kGrayscaleCompWgsl = R"(
struct Uniforms {
  grayscale : vec3f,
}

@group(1) @binding(0) var<uniform> uniforms : Uniforms;
@group(2) @binding(0) var inTexture : texture_storage_2d<rgba8unorm, read>;
@group(2) @binding(1) var outTexture : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let gid = vec2i(id.xy);
  let inColor = textureLoad(inTexture, gid);
  let gray = dot(inColor.rgb, uniforms.grayscale);
  textureStore(outTexture, gid, vec4f(gray, gray, gray, 1.0));
}
)";

struct VertexPosUv {
  iglu::simdtypes::float3 position; // SIMD 128b aligned
  iglu::simdtypes::float2 uv; // SIMD 128b aligned
};

const VertexPosUv kVertexData[] = {
    {.position = {-1.0f, 1.0f, 0.0}, .uv = {0.0, 1.0}},
    {.position = {1.0f, 1.0f, 0.0}, .uv = {1.0, 1.0}},
    {.position = {-1.0f, -1.0f, 0.0}, .uv = {0.0, 0.0}},
    {.position = {1.0f, -1.0f, 0.0}, .uv = {1.0, 0.0}},
};

const uint16_t kIndexData[] = {
    0,
    1,
    2,
    1,
    3,
    2,
};

bool isDeviceCompatible(IDevice& device) noexcept {
  return device.hasFeature(DeviceFeatures::Compute);
}
} // namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
void ComputeSession::initialize() noexcept {
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  // Vertex buffer, Index buffer and Vertex Input
  vbIn_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Storage,
                                         .data = kVertexData,
                                         .length = sizeof(kVertexData)},
                              nullptr);
  IGL_DEBUG_ASSERT(vbIn_ != nullptr);
  // Written by the compute pass, then drawn from.
  vbOut_ = device.createBuffer(
      BufferDesc{.type = BufferDesc::BufferTypeBits::Storage | BufferDesc::BufferTypeBits::Vertex,
                 .length = sizeof(kVertexData)},
      nullptr);
  IGL_DEBUG_ASSERT(vbOut_ != nullptr);

  ib_ = device.createBuffer(BufferDesc{.type = BufferDesc::BufferTypeBits::Index,
                                       .data = kIndexData,
                                       .length = sizeof(kIndexData)},
                            nullptr);
  IGL_DEBUG_ASSERT(ib_ != nullptr);

  const VertexInputStateDesc inputDesc = {
      .numAttributes = 2,
      .attributes =
          {
              {
                  .bufferIndex = 0,
                  .format = VertexAttributeFormat::Float3,
                  .offset = offsetof(VertexPosUv, position),
                  .name = "position",
                  .location = 0,
              },
              {
                  .bufferIndex = 0,
                  .format = VertexAttributeFormat::Float2,
                  .offset = offsetof(VertexPosUv, uv),
                  .name = "uv_in",
                  .location = 1,
              },
          },
      .numInputBindings = 1,
      .inputBindings =
          {
              {
                  .stride = sizeof(VertexPosUv),
              },
          },
  };
  vertexInput_ = device.createVertexInputState(inputDesc, nullptr);
  IGL_DEBUG_ASSERT(vertexInput_ != nullptr);

  // Sampler & Texture
  samp_ = device.createSamplerState(
      SamplerStateDesc{
          .minFilter = SamplerMinMagFilter::Linear,
          .magFilter = SamplerMinMagFilter::Linear,
      },
      nullptr);
  IGL_DEBUG_ASSERT(samp_ != nullptr);
  const auto imageData = getPlatform().getImageLoader().loadImageData("igl.png");
  const TextureDesc texDesc = {
      .width = static_cast<uint32_t>(imageData.desc.width),
      .height = static_cast<uint32_t>(imageData.desc.height),
      .usage = igl::TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Storage,
      .type = TextureType::TwoD,
      .format = TextureFormat::RGBA_UNorm8,
  };

  Result res;
  tex_ = device.createTexture(texDesc, &res);
  isSupported_ = res.isOk();
  if (!isSupported_) {
    return;
  }

  IGL_DEBUG_ASSERT(tex_ != nullptr);

  tex_->upload(tex_->getFullRange(), imageData.data->data());

  const auto dimensions = tex_->getDimensions();
  const TextureDesc desc = {
      .width = dimensions.width,
      .height = dimensions.height,
      .usage = TextureDesc::TextureUsageBits::Sampled | TextureDesc::TextureUsageBits::Storage,
      .type = TextureType::TwoD,
      .format = TextureFormat::RGBA_UNorm8,
  };
  outTex_ = device.createTexture(desc, nullptr);
  IGL_DEBUG_ASSERT(outTex_ != nullptr);

  const Result result;

  if (device.getBackendType() == BackendType::WebGPU) {
    shaderStages_ = ShaderStagesCreator::fromModuleStringInput(device,
                                                               wgsl::kComputeSessionVert,
                                                               "main",
                                                               "",
                                                               wgsl::kComputeSessionFrag,
                                                               "main",
                                                               "",
                                                               nullptr);
    computeStages0_ =
        ShaderStagesCreator::fromModuleStringInput(device, kGrayscaleCompWgsl, "main", "", nullptr);
    computeStages1_ = ShaderStagesCreator::fromModuleStringInput(
        device, wgsl::kComputeSessionVertexTransformComp, "main", "", nullptr);
  } else {
    const auto vertProvider = ComputeSessionVertShaderProvider();
    const auto fragProvider = ComputeSessionFragShaderProvider();
    shaderStages_ =
        createRenderPipelineStages(getPlatform().getDevice(), vertProvider, fragProvider);
    // CS0
    computeStages0_ = createComputePipelineStages(getPlatform().getDevice(),
                                                  ComputeSessionGrayscaleCompShaderProvider());
    // CS1
    computeStages1_ = createComputePipelineStages(
        getPlatform().getDevice(), ComputeSessionVertexTransformCompShaderProvider());
  }
  IGL_DEBUG_ASSERT(shaderStages_ != nullptr);
  IGL_DEBUG_ASSERT(computeStages0_ != nullptr);
  IGL_DEBUG_ASSERT(computeStages1_ != nullptr);

  // Command queue: backed by different types of GPU HW queues
  commandQueue_ = device.createCommandQueue({}, nullptr);
  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  renderPass_ = {
      .colorAttachments =
          {
              {
                  .loadAction = LoadAction::Clear,
                  .storeAction = StoreAction::Store,
                  .clearColor = getPreferredClearColor(),
              },
          },
  };
}

void ComputeSession::createOrUpdateDefaultFramebuffer(const SurfaceTextures& surfaceTextures) {
  if (framebuffer_) {
    framebuffer_->updateDrawable(surfaceTextures.color);
    return;
  }

  // Framebuffer & Texture
  const FramebufferDesc framebufferDesc = {
      .colorAttachments = {{.texture = surfaceTextures.color}},
  };
  framebuffer_ = getPlatform().getDevice().createFramebuffer(framebufferDesc, nullptr);
  IGL_DEBUG_ASSERT(framebuffer_ != nullptr);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
void ComputeSession::update(SurfaceTextures surfaceTextures) noexcept {
  // Per IGL guidelines, surfaceTextures.color may be null on some platforms
  // before the surface is ready (e.g., during window resize on Android/iOS).
  if (!surfaceTextures.color) {
    return;
  }
  if (!isSupported_) {
    return;
  }
  auto& device = getPlatform().getDevice();
  if (!isDeviceCompatible(device)) {
    return;
  }

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);

  const auto buffer = commandQueue_->createCommandBuffer({}, nullptr);
  IGL_DEBUG_ASSERT(buffer != nullptr);
  if (!buffer) {
    return;
  }

  // compute pass
  if (computePipelineState0_ == nullptr) {
    const ComputePipelineDesc computeDesc = {
        .imagesMap =
            {
                {0, genNameHandle("inTexture")},
                {1, genNameHandle("outTexture")},
            },
        .shaderStages = computeStages0_,
    };
    computePipelineState0_ = device.createComputePipeline(computeDesc, nullptr);
    IGL_DEBUG_ASSERT(computePipelineState0_ != nullptr);
  }

  if (computePipelineState1_ == nullptr) {
    const ComputePipelineDesc computeDesc = {
        .buffersMap =
            {
                {0, genNameHandle("verticesIn")},
                {1, genNameHandle("verticesOut")},
            },
        .shaderStages = computeStages1_,
    };
    computePipelineState1_ = device.createComputePipeline(computeDesc, nullptr);
    IGL_DEBUG_ASSERT(computePipelineState1_ != nullptr);
  }

  auto computeEncoder0 = buffer->createComputeCommandEncoder();
  IGL_DEBUG_ASSERT(computeEncoder0 != nullptr);

  if (computeEncoder0) {
    computeEncoder0->bindComputePipelineState(computePipelineState0_);
    computeEncoder0->bindTexture(0, tex_.get());
    computeEncoder0->bindTexture(1, outTex_.get());

    const iglu::uniform::Encoder enc(device.getBackendType());
    // Grayscale values as uniforms
    iglu::uniform::DescriptorValue<glm::vec3> tmpFloat3(glm::vec3(0.2126f, 0.7152f, 0.0722f));
    const int index = device.getBackendType() == BackendType::OpenGL
                          ? computePipelineState0_->getIndexByName(genNameHandle("grayscale"))
                          : 0;
    tmpFloat3.setIndex(ShaderStage::Compute, index);
    enc(*computeEncoder0, tmpFloat3);

    // dispatch
    const Dimensions threadgroupSize(16, 16, 1);
    const Dimensions textureDimensions = tex_->getDimensions();
    const Dimensions threadgroupCount(
        (textureDimensions.width + threadgroupSize.width - 1) / threadgroupSize.width,
        (textureDimensions.height + threadgroupSize.height - 1) / threadgroupSize.height,
        1);
    computeEncoder0->dispatchThreadGroups(threadgroupCount, threadgroupSize);
    computeEncoder0->endEncoding();
  }

  const auto computeEncoder1 = buffer->createComputeCommandEncoder();
  IGL_DEBUG_ASSERT(computeEncoder1 != nullptr);

  if (computeEncoder1) {
    computeEncoder1->bindComputePipelineState(computePipelineState1_);
    computeEncoder1->bindBuffer(0, vbIn_.get());
    computeEncoder1->bindBuffer(1, vbOut_.get());

    // dispatch
    const Dimensions threadgroupSize(4, 1, 1);
    const Dimensions threadgroupCount(1, 1, 1);
    computeEncoder1->dispatchThreadGroups(threadgroupCount, threadgroupSize);
    computeEncoder1->endEncoding();
  }

  // display result of compute pass
  createOrUpdateDefaultFramebuffer(surfaceTextures);

  if (pipelineState_ == nullptr) {
    const RenderPipelineDesc graphicsDesc = {
        .vertexInputState = vertexInput_,
        .shaderStages = shaderStages_,
        .targetDesc =
            {
                .colorAttachments =
                    {
                        {
                            .textureFormat =
                                framebuffer_->getColorAttachment(0)->getProperties().format,
                        },
                    },
            },
        .cullMode = CullMode::Back,
        .frontFaceWinding = WindingMode::Clockwise,
        .fragmentUnitSamplerMap = {{0, IGL_NAMEHANDLE("diffuseTex")}},
    };

    pipelineState_ = device.createRenderPipeline(graphicsDesc, nullptr);
    IGL_DEBUG_ASSERT(pipelineState_ != nullptr);
  }

  const auto renderEncoder = buffer->createRenderCommandEncoder(renderPass_, framebuffer_);
  IGL_DEBUG_ASSERT(renderEncoder != nullptr);
  if (renderEncoder) {
    renderEncoder->bindVertexBuffer(0, *vbOut_);
    renderEncoder->bindRenderPipelineState(pipelineState_);
    renderEncoder->bindTexture(0, BindTarget::kFragment, outTex_.get());
    renderEncoder->bindSamplerState(0, BindTarget::kFragment, samp_.get());
    renderEncoder->bindIndexBuffer(*ib_, IndexFormat::UInt16);
    renderEncoder->drawIndexed(6);
    renderEncoder->endEncoding();
  }
  buffer->present(surfaceTextures.color);

  IGL_DEBUG_ASSERT(commandQueue_ != nullptr);
  commandQueue_->submit(*buffer); // Guarantees ordering between command buffers
}

} // namespace igl::shell
