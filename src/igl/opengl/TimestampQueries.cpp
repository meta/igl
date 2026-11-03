/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <igl/opengl/TimestampQueries.h>

#include <algorithm>
#include <utility>
#include <igl/Macros.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/GLFunc.h>

#ifndef GL_TIME_ELAPSED
#define GL_TIME_ELAPSED 0x88BF
#endif

#ifndef GL_QUERY_RESULT_AVAILABLE
#define GL_QUERY_RESULT_AVAILABLE 0x8867
#endif

#ifndef GL_QUERY_RESULT
#define GL_QUERY_RESULT 0x8866
#endif

#ifndef GL_GPU_DISJOINT_EXT
#define GL_GPU_DISJOINT_EXT 0x8FBB
#endif

namespace igl::opengl {

TimestampQueries::TimestampQueries(IContext& context, uint32_t maxSlots) :
  TimestampQueries(context, maxSlots, QueryFunctions{}) {}

TimestampQueries::TimestampQueries(IContext& context,
                                   uint32_t maxSlots,
                                   QueryFunctions queryFunctions) :
  WithContext(context), queryFunctions_(queryFunctions), maxSlots_(maxSlots) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  if (!queryFunctions_.beginQuery || !queryFunctions_.getQueryObjectiv ||
      !queryFunctions_.getQueryObjectui64v) {
    valid_ = false;
    return;
  }
  queryIds_.resize(maxSlots);
  begun_.resize(maxSlots);
  retainedFramebuffers_.resize(maxSlots);
  iglGenQueries(maxSlots, queryIds_.data());

  // Validate: broken Mali budget GPU drivers silently fail iglGenQueries,
  // leaving query IDs as 0. Any zero ID means the driver is broken - disable all.
  bool anyZero = false;
  for (uint32_t i = 0; i < maxSlots; ++i) {
    if (queryIds_[i] == 0) {
      anyZero = true;
      break;
    }
  }
  if (maxSlots > 0 && anyZero) {
    queryIds_.clear();
    valid_ = false;
  }
}

TimestampQueries::~TimestampQueries() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  if (!queryIds_.empty()) {
    iglDeleteQueries(static_cast<GLsizei>(queryIds_.size()), queryIds_.data());
  }
}

uint32_t TimestampQueries::capacity() const {
  return maxSlots_;
}

uint32_t TimestampQueries::count() const {
  return currentIndex_;
}

bool TimestampQueries::wasBegun(uint32_t slotIndex) const {
  return slotIndex < currentIndex_ && begun_[slotIndex];
}

void TimestampQueries::reset() {
  IGL_PROFILER_FUNCTION();
  currentIndex_ = 0;
  std::fill(begun_.begin(), begun_.end(), false);
  for (auto& framebuffer : retainedFramebuffers_) {
    framebuffer = {};
  }
}

bool TimestampQueries::isValid() const {
  return valid_;
}

bool TimestampQueries::requiresFramebufferRetention() const {
  return true;
}

bool TimestampQueries::resultsAvailable() const {
  IGL_PROFILER_FUNCTION();
  if (!valid_ || currentIndex_ == 0) {
    return false;
  }
  if (queryIds_[currentIndex_ - 1] == 0 || !begun_[currentIndex_ - 1]) {
    return false;
  }
  GLint available = 0;
  queryFunctions_.getQueryObjectiv(
      queryIds_[currentIndex_ - 1], GL_QUERY_RESULT_AVAILABLE, &available);
  return available != 0;
}

uint64_t TimestampQueries::getElapsedNanos(uint32_t slotIndex) const {
  const auto result = getElapsedNanosResult(slotIndex);
  return result.valid ? result.elapsedNanos : 0;
}

TimestampQueryResult TimestampQueries::getElapsedNanosResult(uint32_t slotIndex) const {
  IGL_PROFILER_FUNCTION();
  if (!valid_ || slotIndex >= currentIndex_) {
    return {};
  }
  if (queryIds_[slotIndex] == 0 || !begun_[slotIndex]) {
    return {};
  }

  GLint available = 0;
  queryFunctions_.getQueryObjectiv(queryIds_[slotIndex], GL_QUERY_RESULT_AVAILABLE, &available);
  if (!available) {
    return {};
  }

  GLuint64 result = 0;
  queryFunctions_.getQueryObjectui64v(queryIds_[slotIndex], GL_QUERY_RESULT, &result);
  return {.elapsedNanos = result, .valid = true};
}

bool TimestampQueries::readAndClearDisjoint() {
  IGL_PROFILER_FUNCTION();
  if (!DeviceFeatureSet::usesOpenGLES() ||
      !getContext().deviceFeatures().hasExtension(Extensions::TimerQuery)) {
    return false;
  }
  GLint disjoint = 0;
  getContext().getIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
  return disjoint != 0;
}

void TimestampQueries::beginElapsedQuery(uint32_t slotIndex,
                                         const std::shared_ptr<IFramebuffer>& framebuffer) {
  IGL_PROFILER_FUNCTION();
  if (!valid_ || slotIndex >= maxSlots_ || queryIds_[slotIndex] == 0) {
    return;
  }
  IGL_DEBUG_ASSERT(framebuffer);
  RetainedFramebuffer retained = {.framebuffer = framebuffer};
  for (const size_t index : framebuffer->getColorAttachmentIndices()) {
    retained.attachments.colorAttachments[index] = {
        .texture = framebuffer->getColorAttachment(index),
        .resolveTexture = framebuffer->getResolveColorAttachment(index)};
  }
  retained.attachments.depthAttachment = {.texture = framebuffer->getDepthAttachment(),
                                          .resolveTexture =
                                              framebuffer->getResolveDepthAttachment()};
  retained.attachments.stencilAttachment.texture = framebuffer->getStencilAttachment();
  beginElapsedQuery(slotIndex);
  retainedFramebuffers_[slotIndex] = std::move(retained);
}

void TimestampQueries::beginElapsedQuery(uint32_t slotIndex) {
  IGL_PROFILER_FUNCTION();
  if (!valid_ || slotIndex >= maxSlots_) {
    return;
  }
  if (queryIds_[slotIndex] == 0) {
    return; // guard against zero query ID - invalid per GL spec
  }
  queryFunctions_.beginQuery(GL_TIME_ELAPSED, queryIds_[slotIndex]);
  begun_[slotIndex] = true;
  if (slotIndex >= currentIndex_) {
    currentIndex_ = slotIndex + 1;
  }
}

void TimestampQueries::endElapsedQuery() {
  IGL_PROFILER_FUNCTION();
  if (!valid_) {
    return;
  }
  iglEndQuery(GL_TIME_ELAPSED);
}

} // namespace igl::opengl
