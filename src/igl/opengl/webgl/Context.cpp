/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @fb-only

#include <igl/opengl/webgl/Context.h>

#include <igl/Macros.h>
#include <igl/opengl/Texture.h>

namespace igl::opengl::webgl {

Context::Context(const char* canvasName) :
  Context({.flavor = BackendFlavor::OpenGL_ES, .majorVersion = 3, .minorVersion = 0}, canvasName) {}

Context::Context(BackendVersion backendVersion, const char* canvasName) : canvasName_(canvasName) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT(backendVersion.flavor == BackendFlavor::OpenGL_ES);
  IGL_DEBUG_ASSERT(backendVersion.majorVersion == 3 || backendVersion.majorVersion == 2);
  IGL_DEBUG_ASSERT(backendVersion.minorVersion == 0);
  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.majorVersion = backendVersion.majorVersion - 1;
  attrs.minorVersion = 0;
  attrs.premultipliedAlpha = false;
  attrs.alpha = false;
  attrs.powerPreference = EM_WEBGL_POWER_PREFERENCE_DEFAULT;
  initialize(attrs, canvasName, -1, -1);
}

Context::Context(EmscriptenWebGLContextAttributes& attributes,
                 const char* canvasName,
                 int width,
                 int height) :
  canvasName_(canvasName) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  initialize(attributes, canvasName, width, height);
}

void Context::initialize(EmscriptenWebGLContextAttributes& attributes,
                         const char* canvasName,
                         int width,
                         int height) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  context_ = emscripten_webgl_create_context(canvasName, &attributes);
  if (width > 0 && height > 0) {
    setCanvasBufferSize(width, height);
  }
  if (context_) {
    IContext::registerContext((void*)context_, this);
    setCurrent();

    igl::Result result;
    // Initialize through base class.
    IContext::initialize(&result);
    IGL_DEBUG_ASSERT(result.isOk());
  }
}

Context::~Context() {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_DESTROY);
  willDestroy((void*)context_);
  emscripten_webgl_destroy_context(context_);
}
void Context::setCurrent() {
  IGL_PROFILER_FUNCTION();
  emscripten_webgl_make_context_current(context_);
}

void Context::clearCurrentContext() const {
  // Intentionally does nothing. No such option on WebGL
}

bool Context::isCurrentContext() const {
  return emscripten_webgl_get_current_context() == context_;
}

bool Context::isCurrentSharegroup() const {
  return true;
}

std::unique_ptr<IContext> Context::createShareContext(Result* outResult) {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_CREATE);
  IGL_DEBUG_ASSERT_NOT_IMPLEMENTED();
  Result::setResult(outResult, Result::Code::Unimplemented, "Implement as needed");
  return nullptr;
}

void Context::setCanvasBufferSize(int width, int height) {
  IGL_PROFILER_FUNCTION();
  auto result = emscripten_set_canvas_element_size(canvasName_.c_str(), width, height);
  if (result != EMSCRIPTEN_RESULT_SUCCESS) {
    printf("emscripten_set_canvas_element_size failed: %d\n", result);
  }
}

void Context::present(std::shared_ptr<ITexture> surface) const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_PRESENT);
  emscripten_webgl_commit_frame();
}

} // namespace igl::opengl::webgl
