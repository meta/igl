/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/vulkan/RenderPipelineState.h>

#include <algorithm>
#include <igl/vulkan/Common.h>
#include <igl/vulkan/Device.h>
#include <igl/vulkan/ShaderModule.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanDescriptorSetLayout.h>
#include <igl/vulkan/VulkanPipelineBuilder.h>

namespace igl::vulkan {

VkPrimitiveTopology primitiveTypeToVkPrimitiveTopology(PrimitiveType t) {
  switch (t) {
  case PrimitiveType::Point:
    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  case PrimitiveType::Line:
    return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  case PrimitiveType::LineStrip:
    return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  case PrimitiveType::Triangle:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  case PrimitiveType::TriangleStrip:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  }
  IGL_DEBUG_ABORT("Implement PrimitiveType = %u", static_cast<uint32_t>(t));
  return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
}

VkPolygonMode polygonFillModeToVkPolygonMode(PolygonFillMode mode) {
  switch (mode) {
  case PolygonFillMode::Fill:
    return VK_POLYGON_MODE_FILL;
  case PolygonFillMode::Line:
    return VK_POLYGON_MODE_LINE;
  }
  IGL_DEBUG_ABORT("Implement a missing polygon fill mode");
  return VK_POLYGON_MODE_FILL;
}

VkCullModeFlags cullModeToVkCullMode(CullMode mode) {
  switch (mode) {
  case CullMode::Disabled:
    return VK_CULL_MODE_NONE;
  case CullMode::Front:
    return VK_CULL_MODE_FRONT_BIT;
  case CullMode::Back:
    return VK_CULL_MODE_BACK_BIT;
  }
  IGL_DEBUG_ABORT("Implement a missing cull mode");
  return VK_CULL_MODE_NONE;
}

VkFrontFace windingModeToVkFrontFace(WindingMode mode) {
  switch (mode) {
  case WindingMode::Clockwise:
    return VK_FRONT_FACE_CLOCKWISE;
  case WindingMode::CounterClockwise:
    return VK_FRONT_FACE_COUNTER_CLOCKWISE;
  }
  IGL_DEBUG_ABORT("Wrong winding order (cannot be more than 2)");
  return VK_FRONT_FACE_CLOCKWISE;
}

VkFormat vertexAttributeFormatToVkFormat(VertexAttributeFormat fmt) {
  using igl::VertexAttributeFormat;
  switch (fmt) {
  case VertexAttributeFormat::Float1:
    return VK_FORMAT_R32_SFLOAT;
  case VertexAttributeFormat::Float2:
    return VK_FORMAT_R32G32_SFLOAT;
  case VertexAttributeFormat::Float3:
    return VK_FORMAT_R32G32B32_SFLOAT;
  case VertexAttributeFormat::Float4:
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  case VertexAttributeFormat::Byte1:
    return VK_FORMAT_R8_SINT;
  case VertexAttributeFormat::Byte2:
    return VK_FORMAT_R8G8_SINT;
  case VertexAttributeFormat::Byte3:
    return VK_FORMAT_R8G8B8_SINT;
  case VertexAttributeFormat::Byte4:
    return VK_FORMAT_R8G8B8A8_SINT;
  case VertexAttributeFormat::UByte1:
    return VK_FORMAT_R8_UINT;
  case VertexAttributeFormat::UByte2:
    return VK_FORMAT_R8G8_UINT;
  case VertexAttributeFormat::UByte3:
    return VK_FORMAT_R8G8B8_UINT;
  case VertexAttributeFormat::UByte4:
    return VK_FORMAT_R8G8B8A8_UINT;
  case VertexAttributeFormat::Short1:
    return VK_FORMAT_R16_SINT;
  case VertexAttributeFormat::Short2:
    return VK_FORMAT_R16G16_SINT;
  case VertexAttributeFormat::Short3:
    return VK_FORMAT_R16G16B16_SINT;
  case VertexAttributeFormat::Short4:
    return VK_FORMAT_R16G16B16A16_SINT;
  case VertexAttributeFormat::UShort1:
    return VK_FORMAT_R16_UINT;
  case VertexAttributeFormat::UShort2:
    return VK_FORMAT_R16G16_UINT;
  case VertexAttributeFormat::UShort3:
    return VK_FORMAT_R16G16B16_UINT;
  case VertexAttributeFormat::UShort4:
    return VK_FORMAT_R16G16B16A16_UINT;
    // Normalized variants
  case VertexAttributeFormat::Byte1Norm:
    return VK_FORMAT_R8_SNORM;
  case VertexAttributeFormat::Byte2Norm:
    return VK_FORMAT_R8G8_SNORM;
  case VertexAttributeFormat::Byte3Norm:
    return VK_FORMAT_R8G8B8_SNORM;
  case VertexAttributeFormat::Byte4Norm:
    return VK_FORMAT_R8G8B8A8_SNORM;
  case VertexAttributeFormat::UByte1Norm:
    return VK_FORMAT_R8_UNORM;
  case VertexAttributeFormat::UByte2Norm:
    return VK_FORMAT_R8G8_UNORM;
  case VertexAttributeFormat::UByte3Norm:
    return VK_FORMAT_R8G8B8_UNORM;
  case VertexAttributeFormat::UByte4Norm:
    return VK_FORMAT_R8G8B8A8_UNORM;
  case VertexAttributeFormat::Short1Norm:
    return VK_FORMAT_R16_SNORM;
  case VertexAttributeFormat::Short2Norm:
    return VK_FORMAT_R16G16_SNORM;
  case VertexAttributeFormat::Short3Norm:
    return VK_FORMAT_R16G16B16_SNORM;
  case VertexAttributeFormat::Short4Norm:
    return VK_FORMAT_R16G16B16A16_SNORM;
  case VertexAttributeFormat::UShort1Norm:
    return VK_FORMAT_R16_UNORM;
  case VertexAttributeFormat::UShort2Norm:
    return VK_FORMAT_R16G16_UNORM;
  case VertexAttributeFormat::UShort3Norm:
    return VK_FORMAT_R16G16B16_UNORM;
  case VertexAttributeFormat::UShort4Norm:
    return VK_FORMAT_R16G16B16A16_UNORM;
  // Integer formats
  case VertexAttributeFormat::Int1:
    return VK_FORMAT_R32_SINT;
  case VertexAttributeFormat::Int2:
    return VK_FORMAT_R32G32_SINT;
  case VertexAttributeFormat::Int3:
    return VK_FORMAT_R32G32B32_SINT;
  case VertexAttributeFormat::Int4:
    return VK_FORMAT_R32G32B32A32_SINT;
  case VertexAttributeFormat::UInt1:
    return VK_FORMAT_R32_UINT;
  case VertexAttributeFormat::UInt2:
    return VK_FORMAT_R32G32_UINT;
  case VertexAttributeFormat::UInt3:
    return VK_FORMAT_R32G32B32_UINT;
  case VertexAttributeFormat::UInt4:
    return VK_FORMAT_R32G32B32A32_UINT;
  // Half-float
  case VertexAttributeFormat::HalfFloat1:
    return VK_FORMAT_R16_SFLOAT;
  case VertexAttributeFormat::HalfFloat2:
    return VK_FORMAT_R16G16_SFLOAT;
  case VertexAttributeFormat::HalfFloat3:
    return VK_FORMAT_R16G16B16_SFLOAT;
  case VertexAttributeFormat::HalfFloat4:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  case VertexAttributeFormat::Int_2_10_10_10_REV:
    return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
  }
  IGL_DEBUG_ASSERT_NOT_REACHED();
  return VK_FORMAT_UNDEFINED;
}

VkBlendOp blendOpToVkBlendOp(BlendOp value) {
  using igl::BlendOp;
  switch (value) {
  case BlendOp::Add:
    return VK_BLEND_OP_ADD;
  case BlendOp::Subtract:
    return VK_BLEND_OP_SUBTRACT;
  case BlendOp::ReverseSubtract:
    return VK_BLEND_OP_REVERSE_SUBTRACT;
  case BlendOp::Min:
    return VK_BLEND_OP_MIN;
  case BlendOp::Max:
    return VK_BLEND_OP_MAX;
  }

  IGL_DEBUG_ASSERT_NOT_REACHED();
  return VK_BLEND_OP_ADD;
}

VkBool32 checkDualSrcBlendFactor(BlendFactor value, VkBool32 dualSrcBlendSupported) {
  if (!dualSrcBlendSupported) {
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (value) {
    case BlendFactor::Src1Color:
    case BlendFactor::OneMinusSrc1Color:
    case BlendFactor::Src1Alpha:
    case BlendFactor::OneMinusSrc1Alpha:
      IGL_DEBUG_ASSERT_NOT_REACHED();
      return VK_FALSE;
    default:
      return VK_TRUE;
    }
  }
  return VK_TRUE;
}

VkBlendFactor blendFactorToVkBlendFactor(BlendFactor value) {
  using igl::BlendFactor;
  switch (value) {
  case BlendFactor::Zero:
    return VK_BLEND_FACTOR_ZERO;
  case BlendFactor::One:
    return VK_BLEND_FACTOR_ONE;
  case BlendFactor::SrcColor:
    return VK_BLEND_FACTOR_SRC_COLOR;
  case BlendFactor::OneMinusSrcColor:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
  case BlendFactor::DstColor:
    return VK_BLEND_FACTOR_DST_COLOR;
  case BlendFactor::OneMinusDstColor:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case BlendFactor::SrcAlpha:
    return VK_BLEND_FACTOR_SRC_ALPHA;
  case BlendFactor::OneMinusSrcAlpha:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case BlendFactor::DstAlpha:
    return VK_BLEND_FACTOR_DST_ALPHA;
  case BlendFactor::OneMinusDstAlpha:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  case BlendFactor::BlendColor:
    return VK_BLEND_FACTOR_CONSTANT_COLOR;
  case BlendFactor::OneMinusBlendColor:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
  case BlendFactor::BlendAlpha:
    return VK_BLEND_FACTOR_CONSTANT_ALPHA;
  case BlendFactor::OneMinusBlendAlpha:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
  case BlendFactor::SrcAlphaSaturated:
    return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
  case BlendFactor::Src1Color:
    return VK_BLEND_FACTOR_SRC1_COLOR;
  case BlendFactor::OneMinusSrc1Color:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
  case BlendFactor::Src1Alpha:
    return VK_BLEND_FACTOR_SRC1_ALPHA;
  case BlendFactor::OneMinusSrc1Alpha:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
  default:
    IGL_DEBUG_ASSERT_NOT_REACHED();
    return VK_BLEND_FACTOR_ONE; // default for unsupported values
  }
}

VkColorComponentFlags colorWriteMaskToVkColorComponentFlags(ColorWriteMask value) {
  VkColorComponentFlags result = 0;
  if ((value & kColorWriteBitsRed) != 0) {
    result |= VK_COLOR_COMPONENT_R_BIT;
  }
  if ((value & kColorWriteBitsGreen) != 0) {
    result |= VK_COLOR_COMPONENT_G_BIT;
  }
  if ((value & kColorWriteBitsBlue) != 0) {
    result |= VK_COLOR_COMPONENT_B_BIT;
  }
  if ((value & kColorWriteBitsAlpha) != 0) {
    result |= VK_COLOR_COMPONENT_A_BIT;
  }
  return result;
}

RenderPipelineState::RenderPipelineState(const Device& device, RenderPipelineDesc desc) :
  IRenderPipelineState(desc),
  PipelineState(device.getVulkanContext(),
                desc.shaderStages.get(),
                desc.immutableSamplers,
                desc.isDynamicBufferMask,
                desc.debugName.c_str()),
  device_(device),
  reflection_(std::make_shared<RenderPipelineReflection>()) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  // Iterate and cache vertex input bindings and attributes
  const VertexInputState* vstate = static_cast<VertexInputState*>(desc_.vertexInputState.get());

  vertexInputStateCreateInfo_ = {.sType =
                                     VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

  if (vstate) {
    std::array<bool, IGL_BUFFER_BINDINGS_MAX> bufferAlreadyBound{};
    vkBindings_.reserve(vstate->desc.numInputBindings);

    for (size_t i = 0; i != vstate->desc.numAttributes; i++) {
      const VertexAttribute& attr = vstate->desc.attributes[i];
      const VkFormat format = vertexAttributeFormatToVkFormat(attr.format);
      const size_t bufferIndex = attr.bufferIndex;

      vkAttributes_[i] = VkVertexInputAttributeDescription{
          .location = static_cast<uint32_t>(attr.location),
          .binding = static_cast<uint32_t>(bufferIndex),
          .format = format,
          .offset = static_cast<uint32_t>(attr.offset),
      };

      if (!bufferAlreadyBound[bufferIndex]) {
        bufferAlreadyBound[bufferIndex] = true;

        const VertexInputBinding& binding = vstate->desc.inputBindings[bufferIndex];
        const VkVertexInputRate rate = (binding.sampleFunction == VertexSampleFunction::PerVertex)
                                           ? VK_VERTEX_INPUT_RATE_VERTEX
                                           : VK_VERTEX_INPUT_RATE_INSTANCE;
        vkBindings_.emplace_back(VkVertexInputBindingDescription{
            .binding = static_cast<uint32_t>(bufferIndex),
            .stride = static_cast<uint32_t>(binding.stride),
            .inputRate = rate,
        });
      }
    }

    vertexInputStateCreateInfo_.vertexBindingDescriptionCount =
        static_cast<uint32_t>(vstate->desc.numInputBindings);
    vertexInputStateCreateInfo_.pVertexBindingDescriptions = vkBindings_.data();
    vertexInputStateCreateInfo_.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(vstate->desc.numAttributes);
    vertexInputStateCreateInfo_.pVertexAttributeDescriptions = vkAttributes_.data();
  }
}

void RenderPipelineState::deferDestroyPipelinesAndLayout(const VulkanContext& ctx) const {
  const VkDevice device = ctx.getVkDevice();
  for (const auto& p : pipelines_) {
    if (p.second != VK_NULL_HANDLE) {
      ctx.deferredTask(std::packaged_task<void()>([vf = &ctx.vf_, device, pipeline = p.second]() {
        vf->vkDestroyPipeline(device, pipeline, nullptr);
      }));
    }
  }
  if (pipelineLayout) {
    ctx.deferredTask(std::packaged_task<void()>([vf = &ctx.vf_, device, layout = pipelineLayout]() {
      vf->vkDestroyPipelineLayout(device, layout, nullptr);
    }));
  }
}

RenderPipelineState::~RenderPipelineState() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);

  deferDestroyPipelinesAndLayout(device_.getVulkanContext());
}

// NOLINTNEXTLINE(facebook-hte-NullableReturn)
VkPipeline RenderPipelineState::getVkPipeline(
    const RenderPipelineDynamicState& dynamicState) const {
  const VulkanContext& ctx = device_.getVulkanContext();
  IGL_ENSURE_VULKAN_CONTEXT_THREAD(&ctx);

  if (ctx.config_.enableDescriptorIndexing) {
    // the bindless descriptor set layout can be changed in VulkanContext when the number of
    // existing textures increases
    if (lastBindlessVkDescriptorSetLayout != ctx.getBindlessVkDescriptorSetLayout()) {
      // there's a new descriptor set layout - drop the previous Vulkan pipeline
      deferDestroyPipelinesAndLayout(ctx);
      pipelines_.clear();
      pipelineLayout = VK_NULL_HANDLE;
      lastBindlessVkDescriptorSetLayout = ctx.getBindlessVkDescriptorSetLayout();
    }
  }

  const auto& vkFeatures = ctx.features();
  const bool isVulkan13 = ctx.getVkPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_3;
  const bool useEDS = (isVulkan13 || vkFeatures.has_VK_EXT_extended_dynamic_state) &&
                      ctx.vf_.vkCmdSetDepthTestEnable != nullptr;
  const bool useEDS2 = (isVulkan13 || vkFeatures.has_VK_EXT_extended_dynamic_state2) &&
                       ctx.vf_.vkCmdSetDepthBiasEnable != nullptr;

  RenderPipelineDynamicState cacheKey = dynamicState;
  if (useEDS) {
    // Normalize all EDS-controlled fields to their default so different depth/stencil
    // configurations collapse to the same cache entry.
    cacheKey.setDepthCompareOp(VK_COMPARE_OP_ALWAYS);
    cacheKey.setStencilStateOps(
        true, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    cacheKey.setStencilStateOps(
        false, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    cacheKey.stencilTestEnable = 0;
    cacheKey.depthWriteEnable = 0;
  }
  if (useEDS2) {
    cacheKey.depthBiasEnable = 0;
  }

  // Normalize renderPassIndex so that format-compatible render passes
  // (same formats/samples, different load/store/layout) share one PSO.
  // Per Vulkan
  // spec:https://docs.vulkan.org/spec/latest/chapters/renderpass.html#renderpass-compatibility
  // render pass compatibility only depends on format, samples,
  // and flags -- NOT on loadOp/storeOp/initialLayout/finalLayout.
  // This is safe because desc_.targetDesc already pins the format expectations
  // for this RenderPipelineState, so all render passes it encounters are compatible.
  cacheKey.renderPassIndex = 0;
  // Render passes with several subpasses are compatible with each other only if they have the same
  // subpasses, which is what `renderPassCompatIndex` identifies. A pipeline also belongs to one
  // subpass. A render pass with a single subpass leaves both out of the key.
  if (!cacheKey.multiSubpass) {
    cacheKey.renderPassCompatIndex = 0;
    cacheKey.subpassIndex = 0;
  }

  const auto it = pipelines_.find(cacheKey);

  if (it != pipelines_.end()) {
    return it->second;
  }

  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);

  if (!pipelineLayout) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    const VkDescriptorSetLayout dsls[] = {
        dslCombinedImageSamplers->getVkDescriptorSetLayout(),
        dslBuffers->getVkDescriptorSetLayout(),
        dslStorageImages->getVkDescriptorSetLayout(),
        ctx.getBindlessVkDescriptorSetLayout(),
    };

    const VkPipelineLayoutCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(ctx.config_.enableDescriptorIndexing
                                                    ? IGL_ARRAY_NUM_ELEMENTS(dsls)
                                                    : IGL_ARRAY_NUM_ELEMENTS(dsls) - 1u),
        .pSetLayouts = dsls,
        .pushConstantRangeCount = info.hasPushConstants ? 1u : 0u,
        .pPushConstantRanges = info.hasPushConstants ? &pushConstantRange : nullptr,
    };

    const VkDevice device = ctx.getVkDevice();
    VK_ASSERT(ctx.vf_.vkCreatePipelineLayout(device, &ci, nullptr, &pipelineLayout));
    VK_ASSERT(
        ivkSetDebugObjectName(&ctx.vf_,
                              device,
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              (uint64_t)pipelineLayout,
                              IGL_FORMAT("Pipeline Layout: {}", desc_.debugName.c_str()).c_str()));
  }

  const auto& deviceFeatures = ctx.features();
  const VkBool32 dualSrcBlendSupported =
      deviceFeatures.vkPhysicalDeviceFeatures2.features.dualSrcBlend;

  // build a new Vulkan pipeline
  const VkRenderPass renderPass = ctx.getRenderPass(dynamicState.renderPassIndex).pass;

  VkPipeline pipeline = VK_NULL_HANDLE;

  const VkPipelineCreateFlags flags = ctx.features().has_VK_EXT_descriptor_buffer
                                          ? VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
                                          : VkPipelineCreateFlags{};

  // Not all attachments are valid. We need to create color blend attachments only for active
  // attachments
  std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates;
  colorBlendAttachmentStates.reserve(desc_.targetDesc.colorAttachments.size());
  std::for_each(
      desc_.targetDesc.colorAttachments.begin(),
      desc_.targetDesc.colorAttachments.end(),
      [&colorBlendAttachmentStates,
       dualSrcBlendSupported = dualSrcBlendSupported](auto attachment) mutable {
        if (attachment.textureFormat != TextureFormat::Invalid) {
          // In Vulkan color write bits are part of blending.
          if (!attachment.blendEnabled && attachment.colorWriteMask == kColorWriteBitsAll) {
            colorBlendAttachmentStates.push_back(VkPipelineColorBlendAttachmentState{
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            });
          } else {
            checkDualSrcBlendFactor(attachment.srcRGBBlendFactor, dualSrcBlendSupported);
            checkDualSrcBlendFactor(attachment.dstRGBBlendFactor, dualSrcBlendSupported);
            checkDualSrcBlendFactor(attachment.srcAlphaBlendFactor, dualSrcBlendSupported);
            checkDualSrcBlendFactor(attachment.dstAlphaBlendFactor, dualSrcBlendSupported);

            colorBlendAttachmentStates.push_back(VkPipelineColorBlendAttachmentState{
                .blendEnable = VK_TRUE,
                .srcColorBlendFactor = blendFactorToVkBlendFactor(attachment.srcRGBBlendFactor),
                .dstColorBlendFactor = blendFactorToVkBlendFactor(attachment.dstRGBBlendFactor),
                .colorBlendOp = blendOpToVkBlendOp(attachment.rgbBlendOp),
                .srcAlphaBlendFactor = blendFactorToVkBlendFactor(attachment.srcAlphaBlendFactor),
                .dstAlphaBlendFactor = blendFactorToVkBlendFactor(attachment.dstAlphaBlendFactor),
                .alphaBlendOp = blendOpToVkBlendOp(attachment.alphaBlendOp),
                .colorWriteMask = colorWriteMaskToVkColorComponentFlags(attachment.colorWriteMask),
            });
          }
        }
      });

  std::vector<VkPipelineShaderStageCreateInfo> stages;

  std::vector<VkSpecializationMapEntry> vertEntries;
  std::vector<VkSpecializationMapEntry> taskEntries;
  std::vector<VkSpecializationMapEntry> meshEntries;
  std::vector<VkSpecializationMapEntry> fragEntries;
  VkSpecializationInfo vertSpecInfo{};
  VkSpecializationInfo taskSpecInfo{};
  VkSpecializationInfo meshSpecInfo{};
  VkSpecializationInfo fragSpecInfo{};

  if (desc_.shaderStages->getType() == ShaderStagesType::Render) {
    const auto& vertModule = desc_.shaderStages->getVertexModule();
    vertSpecInfo = buildSpecializationInfo(vertModule->info().functionConstantValues, vertEntries);
    stages.emplace_back(VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = igl::vulkan::ShaderModule::getVkShaderModule(vertModule),
        .pName = vertModule->info().entryPoint.c_str(),
        .pSpecializationInfo = vertSpecInfo.mapEntryCount ? &vertSpecInfo : nullptr,
    });
  } else {
    const auto& taskModule = desc_.shaderStages->getTaskModule();
    if (taskModule) {
      taskSpecInfo =
          buildSpecializationInfo(taskModule->info().functionConstantValues, taskEntries);
      stages.emplace_back(VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
          .module = igl::vulkan::ShaderModule::getVkShaderModule(taskModule),
          .pName = taskModule->info().entryPoint.c_str(),
          .pSpecializationInfo = taskSpecInfo.mapEntryCount ? &taskSpecInfo : nullptr,
      });
    }

    const auto& meshModule = desc_.shaderStages->getMeshModule();
    meshSpecInfo = buildSpecializationInfo(meshModule->info().functionConstantValues, meshEntries);
    stages.emplace_back(VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
        .module = igl::vulkan::ShaderModule::getVkShaderModule(meshModule),
        .pName = meshModule->info().entryPoint.c_str(),
        .pSpecializationInfo = meshSpecInfo.mapEntryCount ? &meshSpecInfo : nullptr,
    });
  }

  const auto& fragModule = desc_.shaderStages->getFragmentModule();
  fragSpecInfo = buildSpecializationInfo(fragModule->info().functionConstantValues, fragEntries);
  stages.emplace_back(VkPipelineShaderStageCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = igl::vulkan::ShaderModule::getVkShaderModule(fragModule),
      .pName = fragModule->info().entryPoint.c_str(),
      .pSpecializationInfo = fragSpecInfo.mapEntryCount ? &fragSpecInfo : nullptr,
  });

  // Dynamic states, passed to the builder which requires a std::vector. Reserve the maximum
  // possible size up front (7 base + 7 EDS + 1 EDS2) to avoid reallocations while appending.
  std::vector<VkDynamicState> dynamicStates;
  dynamicStates.reserve(15);
  // Base set, always available since Vulkan 1.0.
  dynamicStates.insert(dynamicStates.end(),
                       {
                           VK_DYNAMIC_STATE_VIEWPORT,
                           VK_DYNAMIC_STATE_SCISSOR,
                           VK_DYNAMIC_STATE_DEPTH_BIAS,
                           VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                           VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                           VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                           VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                       });
  if (useEDS) {
    // VK_EXT_extended_dynamic_state (promoted to Vulkan 1.3)
    dynamicStates.push_back(VK_DYNAMIC_STATE_CULL_MODE);
    dynamicStates.push_back(VK_DYNAMIC_STATE_FRONT_FACE);
    dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
    dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);
    dynamicStates.push_back(VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE);
    dynamicStates.push_back(VK_DYNAMIC_STATE_STENCIL_OP);
    if (ctx.config_.enableDynamicVertexBufferStride) {
      dynamicStates.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);
    }
  }
  if (useEDS2) {
    // VK_EXT_extended_dynamic_state2 (promoted to Vulkan 1.3)
    dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE);
  }

  // Cull mode and front face are not part of RenderPipelineDynamicState, so default them here when
  // they will be set dynamically at draw time. All depth/stencil/bias state is read directly from
  // cacheKey, which already holds these defaults when EDS is active (see the normalization above)
  // -- keeping a single source of truth for the baked static state.
  //
  // Primitive topology stays static and is always baked from desc_. Under
  // VK_EXT_extended_dynamic_state (without EDS3's dynamicPrimitiveTopologyUnrestricted),
  // vkCmdSetPrimitiveTopology() may only change topology within the pipeline's baked topology
  // class, so baking a fixed default (e.g. TRIANGLE_LIST) would break line/point pipelines. It is
  // also per-pipeline constant, so making it dynamic buys nothing.
  const VkCullModeFlags cullMode = useEDS ? VK_CULL_MODE_NONE
                                          : cullModeToVkCullMode(desc_.cullMode);
  const VkFrontFace frontFace = useEDS ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                       : windingModeToVkFrontFace(desc_.frontFaceWinding);
  const VkPrimitiveTopology primitiveTopology = primitiveTypeToVkPrimitiveTopology(desc_.topology);

  VK_ASSERT_RETURN_NULL_HANDLE(
      igl::vulkan::VulkanPipelineBuilder()
          .dynamicStates(dynamicStates)
          .primitiveTopology(primitiveTopology)
          .depthBiasEnable(cacheKey.depthBiasEnable)
          .depthCompareOp(cacheKey.getDepthCompareOp(),
                          static_cast<bool>(cacheKey.depthWriteEnable))
          .depthWriteEnable(static_cast<bool>(cacheKey.depthWriteEnable))
          .rasterizationSamples(getVulkanSampleCountFlags(desc_.sampleCount))
          .alphaToCoverageEnable(desc_.alphaToCoverageEnabled)
          .polygonMode(polygonFillModeToVkPolygonMode(desc_.polygonFillMode))
          .stencilStateOps(VK_STENCIL_FACE_FRONT_BIT,
                           cacheKey.getStencilStateFailOp(true),
                           cacheKey.getStencilStatePassOp(true),
                           cacheKey.getStencilStateDepthFailOp(true),
                           cacheKey.getStencilStateCompareOp(true))
          .stencilStateOps(VK_STENCIL_FACE_BACK_BIT,
                           cacheKey.getStencilStateFailOp(false),
                           cacheKey.getStencilStatePassOp(false),
                           cacheKey.getStencilStateDepthFailOp(false),
                           cacheKey.getStencilStateCompareOp(false))
          .shaderStages(stages)
          .cullMode(cullMode)
          .frontFace(frontFace)
          .vertexInputState(vertexInputStateCreateInfo_)
          .colorBlendAttachmentStates(colorBlendAttachmentStates)
          .subpass(cacheKey.subpassIndex)
          .build(ctx.vf_,
                 ctx.getVkDevice(),
                 flags,
                 ctx.pipelineCache_,
                 pipelineLayout,
                 renderPass,
                 &pipeline,
                 desc_.debugName.c_str()));

  IGL_DEBUG_ASSERT(pipeline != VK_NULL_HANDLE);

  pipelines_[cacheKey] = pipeline;

  // @fb-only
  // @lint-ignore CLANGTIDY
  return pipeline;
}

int RenderPipelineState::getIndexByName(const NameHandle& name, ShaderStage stage) const {
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  (void)name;
  (void)stage;
  return 0;
}

int RenderPipelineState::getIndexByName(const std::string& name, ShaderStage stage) const {
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  (void)name;
  (void)stage;
  return 0;
}

std::shared_ptr<IRenderPipelineReflection> RenderPipelineState::renderPipelineReflection() {
  return reflection_;
}

void RenderPipelineState::setRenderPipelineReflection(
    const IRenderPipelineReflection& renderPipelineReflection) {
  const auto& vulkanReflection =
      static_cast<const RenderPipelineReflection&>(renderPipelineReflection);
  auto copy = RenderPipelineReflection(vulkanReflection);

  reflection_ = std::make_shared<RenderPipelineReflection>(std::move(copy));
}

} // namespace igl::vulkan
