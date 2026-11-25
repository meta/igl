/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/Buffer.h>

#include <igl/Buffer.h>
#include <igl/DeviceFeatures.h>
#include <igl/Macros.h>

namespace igl::opengl {

// ********************************
// ****  ArrayBuffer
// ********************************
// the base buffer object
ArrayBuffer::ArrayBuffer(IContext& context,
                         BufferDesc::BufferAPIHint requestedApiHints,
                         BufferDesc::BufferType bufferType) :
  Buffer(context, requestedApiHints, bufferType) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  iD_ = 0;
  size_ = 0;
  isDynamic_ = false;
}

ArrayBuffer::~ArrayBuffer() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  if (iD_ != 0) {
    getContext().deleteBuffers(1, &iD_);
    getContext().unbindBuffer(target_);
    iD_ = 0;
  }
  size_ = 0;
  isDynamic_ = false;
}

// initialize a buffer with the given size
// if data is not null, copy the data into the buffer
// if the buffer is to be updated frequently, isDynamic should be set to true
void ArrayBuffer::initialize(const BufferDesc& desc, Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  // static buffers must provide their data during creation, as they can't upload data later on
  GLenum usage = GL_DYNAMIC_DRAW;
  switch (desc.storage) {
  case ResourceStorage::Shared:
    usage = GL_DYNAMIC_DRAW;
    isDynamic_ = true;
    break;
  case ResourceStorage::Managed:
    usage = GL_STATIC_DRAW;
    isDynamic_ = false;
    break;
  case ResourceStorage::Private:
    usage = GL_STATIC_DRAW;
    isDynamic_ = false;
    break;
  case ResourceStorage::Memoryless:
  case ResourceStorage::Invalid:
  default:
    break;
  }

  if (!isDynamic_ && desc.data == nullptr) {
    Result::setResult(outResult, Result::Code::ArgumentNull, "data is null");
    return;
  }

  getContext().genBuffers(1, &iD_);
  if (iD_ == 0) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Failed to create buffer");
  }

  if ((desc.type & BufferDesc::BufferTypeBits::Storage) != 0) {
    if (getContext().deviceFeatures().hasFeature(DeviceFeatures::Compute)) {
      target_ = GL_SHADER_STORAGE_BUFFER;
    } else {
      IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
    }
  } else if ((desc.type & BufferDesc::BufferTypeBits::Uniform) != 0) {
    target_ = GL_UNIFORM_BUFFER;
  } else if ((desc.type & BufferDesc::BufferTypeBits::Vertex) != 0) {
    target_ = GL_ARRAY_BUFFER;
  } else if ((desc.type & BufferDesc::BufferTypeBits::Index) != 0) {
    target_ = GL_ELEMENT_ARRAY_BUFFER;
  } else if ((desc.type & BufferDesc::BufferTypeBits::Indirect) != 0) {
    target_ = GL_DRAW_INDIRECT_BUFFER;
  } else {
    IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  }

  size_ = desc.length;

  getContext().bindBuffer(target_, iD_);
  getContext().bufferData(target_, size_, desc.data, usage);

  // make sure the buffer was fully allocated
  GLint bufferSize = 0;
  getContext().getBufferParameteriv(target_, GL_BUFFER_SIZE, &bufferSize);

  if (!desc.debugName.empty() &&
      getContext().deviceFeatures().hasInternalFeature(InternalFeatures::DebugLabel)) {
    const GLenum identifier = getContext().deviceFeatures().hasInternalRequirement(
                                  InternalRequirement::DebugLabelExtEnumsReq)
                                  ? GL_BUFFER_OBJECT_EXT
                                  : GL_BUFFER;
    getContext().objectLabel(
        identifier, iD_, static_cast<GLsizei>(desc.debugName.size()), desc.debugName.c_str());
  }

  getContext().bindBuffer(target_, 0);

  if (bufferSize != size_) {
    getContext().deleteBuffers(1, &iD_);
    iD_ = 0;
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "bufferSize != dataSize");
    return;
  }

  Result::setOk(outResult);
}

// upload data to the buffer at the given offset with the given size
Result ArrayBuffer::upload(const void* data, const BufferRange& range) {
  IGL_PROFILER_FUNCTION();
  // static buffers can only upload data once during creation
  if (!isDynamic_) {
    return Result(Result::Code::InvalidOperation, "Can't upload to static buffers");
  }

  getContext().bindBuffer(target_, iD_);

  getContext().bufferSubData(target_, range.offset, range.size, data);

  getContext().bindBuffer(target_, 0);

  return Result();
}

void* FOLLY_NULLABLE ArrayBuffer::map(const BufferRange& range, Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if ((range.size + range.offset) > getSizeInBytes()) {
    Result::setResult(
        outResult, Result::Code::ArgumentOutOfRange, "map() size + offset must be <= buffer size");
    return nullptr;
  }

  bind();

  void* srcData = nullptr;
  srcData = getContext().mapBufferRange(target_, range.offset, range.size, GL_MAP_READ_BIT);
  IGL_DEBUG_ASSERT(srcData != nullptr);
  if (srcData == nullptr) {
    Result::setResult(outResult, Result::Code::InvalidOperation);
    return nullptr;
  }

  Result::setOk(outResult);
  return srcData;
}

void ArrayBuffer::unmap() {
  IGL_PROFILER_FUNCTION();
  bind();
  getContext().unmapBuffer(target_);
}

// bind the buffer for access by the GPU
void ArrayBuffer::bind() {
  IGL_PROFILER_FUNCTION();
  getContext().bindBuffer(target_, iD_);
}

void ArrayBuffer::unbind() {
  IGL_PROFILER_FUNCTION();
  getContext().bindBuffer(target_, 0);
}

void ArrayBuffer::bindBase([[maybe_unused]] size_t index, Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if (target_ != GL_SHADER_STORAGE_BUFFER) {
    static constexpr const char* kErrorMsg = "Buffer should be GL_SHADER_STORAGE_BUFFER";
    IGL_SOFT_ERROR(kErrorMsg);
    Result::setResult(outResult, Result::Code::InvalidOperation, kErrorMsg);
    return;
  }
  getContext().bindBuffer(target_, iD_);
  getContext().bindBufferBase(target_, static_cast<GLuint>(index), iD_);
  Result::setOk(outResult);
}

void ArrayBuffer::bindForTarget(GLenum target) {
  IGL_PROFILER_FUNCTION();
  getContext().bindBuffer(target, iD_);
}

void UniformBlockBuffer::setBlockBinding(GLuint pid, GLuint blockIndex, GLuint bindingPoint) {
  IGL_PROFILER_FUNCTION();
  getContext().uniformBlockBinding(pid, blockIndex, bindingPoint);
}

void UniformBlockBuffer::bindBase(size_t index, Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if (getContext().deviceFeatures().hasFeature(DeviceFeatures::UniformBlocks)) {
    if (target_ != GL_UNIFORM_BUFFER) {
      static constexpr const char* kErrorMsg = "Buffer should be GL_UNIFORM_BUFFER";
      IGL_SOFT_ERROR(kErrorMsg);
      Result::setResult(outResult, Result::Code::InvalidOperation, kErrorMsg);
      return;
    }
    getContext().bindBufferBase(target_, static_cast<GLuint>(index), iD_);
    Result::setOk(outResult);
  } else {
    static constexpr const char* kErrorMsg = "Uniform Blocks are not supported";
    IGL_SOFT_ERROR(kErrorMsg);
    Result::setResult(outResult, Result::Code::Unimplemented, kErrorMsg);
  }
}

void UniformBlockBuffer::bindRange(size_t index,
                                   size_t offset,
                                   size_t size,
                                   Result* IGL_NULLABLE outResult) {
  IGL_PROFILER_FUNCTION();
  if (getContext().deviceFeatures().hasFeature(DeviceFeatures::UniformBlocks)) {
    if (target_ != GL_UNIFORM_BUFFER) {
      static constexpr const char* kErrorMsg = "Buffer should be GL_UNIFORM_BUFFER";
      IGL_SOFT_ERROR(kErrorMsg);
      Result::setResult(outResult, Result::Code::InvalidOperation, kErrorMsg);
      return;
    }
    getContext().bindBuffer(target_, iD_);
    IGL_DEBUG_ASSERT((offset + size) <= getSizeInBytes(),
                     "Offset or Size is invalid! (%d %d %d)",
                     offset,
                     size,
                     getSizeInBytes());
    getContext().bindBufferRange(target_,
                                 static_cast<GLuint>(index),
                                 iD_,
                                 static_cast<GLintptr>(offset),
                                 size ? size : getSizeInBytes() - offset);
    Result::setOk(outResult);
  } else {
    static constexpr const char* kErrorMsg = "Uniform Blocks are not supported";
    IGL_SOFT_ERROR(kErrorMsg);
    Result::setResult(outResult, Result::Code::Unimplemented, kErrorMsg);
  }
}

} // namespace igl::opengl
