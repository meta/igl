/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/RenderPipelineState.h>

#include <algorithm>
#include <igl/Macros.h>
#include <igl/RenderCommandEncoder.h> // for igl::BindTarget
#include <igl/opengl/VertexInputState.h>

namespace igl::opengl {

namespace {

void logBlendFactorError([[maybe_unused]] const char* value) {
  IGL_DEBUG_ABORT("OpenGL does not support blend mode:  %s, setting to GL_ONE instead\n", value);
}

} // namespace

RenderPipelineState::RenderPipelineState(IContext& context,
                                         const RenderPipelineDesc& desc,
                                         Result* outResult) :
  WithContext(context), IRenderPipelineState(desc) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  activeAttributesLocations_.reserve(64);
  unitSamplerLocationMap_.fill(-1);
  auto ret = create();
  if (outResult) {
    *outResult = ret;
  }
}

RenderPipelineState::~RenderPipelineState() = default;

GLenum RenderPipelineState::convertBlendOp(BlendOp value) {
  // sets blending equation for both RGA and Alpha
  switch (value) {
  case BlendOp::Add:
    return GL_FUNC_ADD;
  case BlendOp::Subtract:
    return GL_FUNC_SUBTRACT;
  case BlendOp::ReverseSubtract:
    return GL_FUNC_REVERSE_SUBTRACT;
  case BlendOp::Min:
    return GL_MIN;
  case BlendOp::Max:
    return GL_MAX;
  }
  IGL_UNREACHABLE_RETURN(GL_FUNC_ADD)
}

GLenum RenderPipelineState::convertBlendFactor(BlendFactor value) {
  switch (value) {
  case BlendFactor::Zero:
    return GL_ZERO;
  case BlendFactor::One:
    return GL_ONE;
  case BlendFactor::SrcColor:
    return GL_SRC_COLOR;
  case BlendFactor::OneMinusSrcColor:
    return GL_ONE_MINUS_SRC_COLOR;
  case BlendFactor::DstColor:
    return GL_DST_COLOR;
  case BlendFactor::OneMinusDstColor:
    return GL_ONE_MINUS_DST_COLOR;
  case BlendFactor::SrcAlpha:
    return GL_SRC_ALPHA;
  case BlendFactor::OneMinusSrcAlpha:
    return GL_ONE_MINUS_SRC_ALPHA;
  case BlendFactor::DstAlpha:
    return GL_DST_ALPHA;
  case BlendFactor::OneMinusDstAlpha:
    return GL_ONE_MINUS_DST_ALPHA;
  case BlendFactor::BlendColor:
    return GL_CONSTANT_COLOR;
  case BlendFactor::OneMinusBlendColor:
    return GL_ONE_MINUS_CONSTANT_COLOR;
  case BlendFactor::BlendAlpha:
    return GL_CONSTANT_ALPHA;
  case BlendFactor::OneMinusBlendAlpha:
    return GL_ONE_MINUS_CONSTANT_ALPHA;
  case BlendFactor::SrcAlphaSaturated:
    return GL_SRC_ALPHA_SATURATE;
  case BlendFactor::Src1Color:
    logBlendFactorError(_IGL_TO_STRING_WRAPPER(GL_SRC1_COLOR));
    return GL_ONE; // default for unsupported values
  case BlendFactor::OneMinusSrc1Color:
    logBlendFactorError(_IGL_TO_STRING_WRAPPER(GL_ONE_MINUS_SRC1_COLOR));
    return GL_ONE; // default for unsupported values
  case BlendFactor::Src1Alpha:
    logBlendFactorError(_IGL_TO_STRING_WRAPPER(GL_SRC1_ALPHA));
    return GL_ONE; // default for unsupported values
  case BlendFactor::OneMinusSrc1Alpha:
    logBlendFactorError(_IGL_TO_STRING_WRAPPER(GL_ONE_MINUS_SRC1_ALPHA));
    return GL_ONE; // default for unsupported values
  }
  IGL_UNREACHABLE_RETURN(GL_ONE)
}

Result RenderPipelineState::create() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (IGL_DEBUG_VERIFY_NOT(desc_.shaderStages == nullptr)) {
    return Result(Result::Code::ArgumentInvalid, "Missing shader stages");
  }
  if (!IGL_DEBUG_VERIFY(desc_.shaderStages->getType() == ShaderStagesType::Render)) {
    return Result(Result::Code::ArgumentInvalid, "Shader stages not for render");
  }
  const auto* shaderStages = static_cast<ShaderStages*>(desc_.shaderStages.get());
  if (!shaderStages) {
    return Result(Result::Code::ArgumentInvalid,
                  "Shader stages required to create pipeline state.");
  }
  if (shaderStages->getType() != ShaderStagesType::Render) {
    return Result(Result::Code::ArgumentInvalid, "Expected render shader stages.");
  } else if (!shaderStages->isValid()) {
    return Result(Result::Code::ArgumentInvalid, "Missing required shader module(s).");
  }

  reflection_ = std::make_shared<RenderPipelineReflection>(getContext(), *shaderStages);

  const auto& mFramebufferDesc = desc_.targetDesc;
  // Get and cache all attribute locations, since this won't change throughout
  // the lifetime of this RenderPipelineState
  const auto* vertexInputState = static_cast<VertexInputState*>(desc_.vertexInputState.get());
  if (desc_.vertexInputState != nullptr) {
    auto bufferAttribMap = vertexInputState->getBufferAttribMap();

    // For each bufferIndex, storage the list of associated attribute locations
    for (const auto& [index, attribList] : bufferAttribMap) {
      for (const auto& attrib : attribList) {
        const int loc = getIndexByName(igl::genNameHandle(attrib.name), ShaderStage::Vertex);
        if (loc < 0) {
          IGL_LOG_ERROR("Vertex attribute (%s) not found in shader.", attrib.name.c_str());
        }
        IGL_DEBUG_ASSERT(index < IGL_BUFFER_BINDINGS_MAX);
        if (index < IGL_BUFFER_BINDINGS_MAX) {
          bufferAttribLocations_[index].push_back(loc);
        }
      }
    }
  }

  // Note this work is only done once. Beyond this point, there is no more query by name
  for (const auto& [textureUnit, samplerName] : desc_.fragmentUnitSamplerMap) {
    if (textureUnit >= unitSamplerLocationMap_.size()) {
      IGL_LOG_ERROR("Fragment sampler unit %zu exceeds max %zu\n",
                    static_cast<size_t>(textureUnit),
                    unitSamplerLocationMap_.size());
      continue;
    }
    const int loc = reflection_->getIndexByName(samplerName);
    if (loc >= 0) {
      unitSamplerLocationMap_[textureUnit] = loc;
    } else {
      IGL_LOG_ERROR("Sampler uniform (%s) not found in shader.\n", samplerName.c_str());
    }
  }

  for (const auto& [bindingIndex, names] : desc_.uniformBlockBindingMap) {
    for (const auto& [blockName, instanceName] : names) {
      auto& uniformBlockDict = reflection_->getUniformBlocksDictionary();
      auto blockDescIt = uniformBlockDict.find(blockName);
      if (blockDescIt != uniformBlockDict.end()) {
        auto blockIndex = blockDescIt->second.blockIndex;
        if (blockDescIt->second.bindingIndex > 0) {
          //  avoid overriding explicit binding points from shaders because we observed
          //  crashes when doing so on some Adreno devices.
          uniformBlockBindingMap_[blockIndex] = blockDescIt->second.bindingIndex;
        } else {
          uniformBlockBindingMap_[blockIndex] = bindingIndex;
          blockDescIt->second.bindingIndex = static_cast<GLint>(bindingIndex);
        }
      }
    }
  }

  for (const auto& [textureUnit, samplerName] : desc_.vertexUnitSamplerMap) {
    const int loc = reflection_->getIndexByName(samplerName);
    if (loc < 0) {
      IGL_LOG_ERROR("Sampler uniform (%s) not found in shader.\n", samplerName.c_str());
      continue;
    }

    // find the first empty slot in unitSamplerLocationMap;
    size_t realTextureUnit = 0;
    for (; realTextureUnit < unitSamplerLocationMap_.size(); realTextureUnit++) {
      if (unitSamplerLocationMap_[realTextureUnit] == -1) {
        break;
      }
    }
    if (realTextureUnit >= unitSamplerLocationMap_.size()) {
      return Result{Result::Code::RuntimeError, "Too many samplers"};
    }

    vertexTextureUnitRemap_[textureUnit] = realTextureUnit;
    vertexUnitSamplerLocationMap_[textureUnit] = loc;
  }

  if (!mFramebufferDesc.colorAttachments.empty()) {
    const ColorWriteMask colorWriteMask = mFramebufferDesc.colorAttachments[0].colorWriteMask;
    colorMask_[0] = static_cast<GLboolean>((colorWriteMask & kColorWriteBitsRed) != 0);
    colorMask_[1] = static_cast<GLboolean>((colorWriteMask & kColorWriteBitsGreen) != 0);
    colorMask_[2] = static_cast<GLboolean>((colorWriteMask & kColorWriteBitsBlue) != 0);
    colorMask_[3] = static_cast<GLboolean>((colorWriteMask & kColorWriteBitsAlpha) != 0);
  }

  if (!mFramebufferDesc.colorAttachments.empty() &&
      mFramebufferDesc.colorAttachments[0].blendEnabled) {
    blendEnabled_ = true;
    // GL equation sets blending equation for both RGB and alpha
    blendMode_ = {
        .blendOpColor = convertBlendOp(mFramebufferDesc.colorAttachments[0].rgbBlendOp),
        .blendOpAlpha = convertBlendOp(mFramebufferDesc.colorAttachments[0].alphaBlendOp),
        .srcColor = convertBlendFactor(mFramebufferDesc.colorAttachments[0].srcRGBBlendFactor),
        .dstColor = convertBlendFactor(mFramebufferDesc.colorAttachments[0].dstRGBBlendFactor),
        .srcAlpha = convertBlendFactor(mFramebufferDesc.colorAttachments[0].srcAlphaBlendFactor),
        .dstAlpha = convertBlendFactor(mFramebufferDesc.colorAttachments[0].dstAlphaBlendFactor)};
  } else {
    blendEnabled_ = false;
  }

  return Result();
}

void RenderPipelineState::bind() {
  IGL_PROFILER_FUNCTION();
  if (desc_.shaderStages) {
    const auto* shaderStages = static_cast<ShaderStages*>(desc_.shaderStages.get());
    shaderStages->bind();
    if (!uniformBlockBindingPointSet_) {
      for (const auto& binding : uniformBlockBindingMap_) {
        const auto& blockIndex = binding.first;
        const auto& bindingIndex = binding.second;
        getContext().uniformBlockBinding(shaderStages->getProgramID(),
                                         static_cast<GLuint>(blockIndex),
                                         static_cast<GLuint>(bindingIndex));
      }
      uniformBlockBindingPointSet_ = true;
    }
  }

  getContext().colorMask(colorMask_[0], colorMask_[1], colorMask_[2], colorMask_[3]);
  if (!desc_.targetDesc.colorAttachments.empty() && blendEnabled_) {
    getContext().enable(GL_BLEND);
    getContext().blendEquationSeparate(blendMode_.blendOpColor, blendMode_.blendOpAlpha);
    getContext().blendFuncSeparate(
        blendMode_.srcColor, blendMode_.dstColor, blendMode_.srcAlpha, blendMode_.dstAlpha);
  } else {
    getContext().disable(GL_BLEND);
  }

  // polygon rasterization mode
  if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::PolygonFillMode)) {
    getContext().polygonFillMode((desc_.polygonFillMode == PolygonFillMode::Fill) ? GL_FILL
                                                                                  : GL_LINE);
  }

  // alpha-to-coverage (only meaningful when MSAA is active)
  if (desc_.alphaToCoverageEnabled) {
    getContext().enable(GL_SAMPLE_ALPHA_TO_COVERAGE);
  } else {
    getContext().disable(GL_SAMPLE_ALPHA_TO_COVERAGE);
  }
}

void RenderPipelineState::unbind() {
  IGL_PROFILER_FUNCTION();
  if (desc_.shaderStages) {
    static_cast<ShaderStages*>(desc_.shaderStages.get())->unbind();
  }
}

// A buffer can be shared by multiple attributes. So bind all the attributes
// associated with the associated buffer.
// bufferOffset is an offset in bytes to the start of the vertex attributes in the buffer.
// When `stride` is non-zero, it overrides the stride declared in the pipeline's vertex
// input state (dynamic vertex buffer stride). When `stride` is zero, the pipeline-declared
// stride is used (legacy behavior).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void RenderPipelineState::bindVertexAttributes(size_t bufferIndex,
                                               size_t bufferOffset,
                                               size_t stride) {
  IGL_PROFILER_FUNCTION();
#if IGL_DEBUG_ABORT_ENABLED
  static GLint sMaxNumVertexAttribs = 0;
  if (0 == sMaxNumVertexAttribs) {
    getContext().getIntegerv(GL_MAX_VERTEX_ATTRIBS, &sMaxNumVertexAttribs);
  }
#endif

  const auto& attribList = static_cast<VertexInputState*>(desc_.vertexInputState.get())
                               ->getAssociatedAttributes(bufferIndex);
  auto& locations = bufferAttribLocations_[bufferIndex];

  // attributeList and locations should have an 1-to-1 correspondence
  IGL_DEBUG_ASSERT(attribList.size() == locations.size());

  for (size_t i = 0, iLen = attribList.size(); i < iLen; i++) {
    auto location = locations[i];
    if (location < 0) {
      // Location not found
      continue;
    }
    IGL_DEBUG_ASSERT(location < sMaxNumVertexAttribs);
    activeAttributesLocations_.push_back(location);

    prevPipelineStateAttributesLocations_.erase(
        std::remove(prevPipelineStateAttributesLocations_.begin(),
                    prevPipelineStateAttributesLocations_.end(),
                    location),
        prevPipelineStateAttributesLocations_.end());

    getContext().enableVertexAttribArray(location);
    const auto& attribute = attribList[i];
    const GLsizei effectiveStride = stride != 0 ? static_cast<GLsizei>(stride) : attribute.stride;
    getContext().vertexAttribPointer(
        location,
        attribute.numComponents,
        attribute.componentType,
        attribute.normalized,
        effectiveStride,
        reinterpret_cast<const char*>(attribute.bufferOffset) + // NOLINT(performance-no-int-to-ptr)
            bufferOffset);

    if (getContext().deviceFeatures().hasInternalFeature(InternalFeatures::VertexAttribDivisor)) {
      if (attribute.sampleFunction == VertexSampleFunction::PerVertex) {
        getContext().vertexAttribDivisor(location, 0);
      } else if (attribute.sampleFunction == VertexSampleFunction::Instance) {
        getContext().vertexAttribDivisor(location, static_cast<GLuint>(attribute.sampleRate));
      } else {
        getContext().vertexAttribDivisor(location, 0);
      }
    }
  }
}

void RenderPipelineState::unbindVertexAttributes() {
  IGL_PROFILER_FUNCTION();
  for (const auto& l : activeAttributesLocations_) {
    getContext().disableVertexAttribArray(l);
  }
  activeAttributesLocations_.clear();
}

void RenderPipelineState::unbindPrevPipelineVertexAttributes() {
  IGL_PROFILER_FUNCTION();
  for (const auto& l : prevPipelineStateAttributesLocations_) {
    getContext().disableVertexAttribArray(l);
  }
  prevPipelineStateAttributesLocations_.clear();
}

// Looks up the location the of the specified texture unit via its name,
// bind the unit to the location, then activate the unit.
//
// Prerequisite: The shader program has to be loaded
Result RenderPipelineState::bindTextureUnit(size_t unit, uint8_t bindTarget, Texture& texture) {
  IGL_PROFILER_FUNCTION();
  if (!desc_.shaderStages) {
    return Result{Result::Code::InvalidOperation, "No shader set\n"};
  }

  if (unit >= IGL_TEXTURE_SAMPLERS_MAX) {
    return Result{Result::Code::ArgumentInvalid, "Unit specified greater than maximum\n"};
  }

  GLint samplerLocation = -1;
  if (bindTarget == BindTarget::kVertex) {
    auto it = vertexTextureUnitRemap_.find(unit);
    if (it == vertexTextureUnitRemap_.end()) {
      return Result{Result::Code::RuntimeError, "Unable to find sampler location\n"};
    }
    samplerLocation = vertexUnitSamplerLocationMap_[unit];
    unit = it->second;
  } else {
    samplerLocation = unitSamplerLocationMap_[unit];
  }

  if (samplerLocation < 0) {
    return Result{Result::Code::RuntimeError, "Unable to find sampler location\n"};
  }

  getContext().activeTexture(static_cast<GLenum>(GL_TEXTURE0 + unit));
  texture.bind();
  getContext().uniform1i(samplerLocation, static_cast<GLint>(unit));

  return Result();
}

bool RenderPipelineState::matchesShaderProgram(const RenderPipelineState& rhs) const {
  return static_cast<ShaderStages*>(desc_.shaderStages.get())->getProgramID() ==
         static_cast<ShaderStages*>(rhs.desc_.shaderStages.get())->getProgramID();
}

bool RenderPipelineState::matchesVertexInputState(const RenderPipelineState& rhs) const {
  return desc_.vertexInputState == rhs.desc_.vertexInputState;
}

int RenderPipelineState::getIndexByName(const NameHandle& name, ShaderStage /*stage*/) const {
  if (reflection_ == nullptr) {
    return -1;
  }
  return reflection_->getIndexByName(name);
}

int RenderPipelineState::getIndexByName(const std::string& name, ShaderStage /*stage*/) const {
  if (reflection_ == nullptr) {
    return -1;
  }
  return reflection_->getIndexByName(igl::genNameHandle(name));
}

int RenderPipelineState::getUniformBlockBindingPoint(const NameHandle& uniformBlockName) const {
  return getIndexByName(uniformBlockName, ShaderStage::Fragment);
}

std::shared_ptr<IRenderPipelineReflection> RenderPipelineState::renderPipelineReflection() {
  return reflection_;
}

void RenderPipelineState::setRenderPipelineReflection(
    const IRenderPipelineReflection& renderPipelineReflection) {
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  (void)renderPipelineReflection;
}

} // namespace igl::opengl
