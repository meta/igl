/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <ldrutils/lutils/Pool.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <igl/CommandEncoder.h>
#include <igl/Common.h>
#include <igl/PlatformDevice.h>
#include <igl/opengl/ComputeCommandAdapter.h>
#include <igl/opengl/Config.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/GLFunc.h>
#include <igl/opengl/GLIncludes.h>
#include <igl/opengl/RenderCommandAdapter.h>
#include <igl/opengl/UnbindPolicy.h>
#include <igl/opengl/Version.h>
#include <igl/opengl/WithContext.h>

namespace igl::opengl {

///
/// Represents an pure abstract class that encapsulates in it an OpenGL context.
/// Individual types that implement this class are the ones that provide implementation
/// for a concrete OpenGL API implementation.
///
class IContext {
 public:
  virtual void setCurrent() = 0;
  virtual void clearCurrentContext() const = 0;
  virtual bool isCurrentContext() const = 0;
  virtual bool isCurrentSharegroup() const = 0;
  virtual std::string getCurrentContextDebugInfo() const {
    return {};
  }
  virtual void present(std::shared_ptr<ITexture> surface) const = 0;

  virtual std::unique_ptr<IContext> createShareContext(Result* IGL_NULLABLE outResult) = 0;

  IContext();
  virtual ~IContext();

  void flushDeletionQueue();

 protected:
  bool shouldQueueAPI() const;

 public:
  ///--------------------------------------
  /// MARK: - GL APIs
  void activeTexture(GLenum texture);
  void attachShader(GLuint program, GLuint shader);
  void bindBuffer(GLenum target, GLuint buffer);
  void bindBufferBase(GLenum target, GLuint index, GLuint buffer);
  void bindBufferRange(GLenum target,
                       GLuint index,
                       GLuint buffer,
                       GLintptr offset,
                       GLsizeiptr size);
  void bindFramebuffer(GLenum target, GLuint framebuffer);
  void bindRenderbuffer(GLenum target, GLuint renderbuffer);
  void bindTexture(GLenum target, GLuint texture);
  void bindImageTexture(GLuint unit,
                        GLuint texture,
                        GLint level,
                        GLboolean layered,
                        GLint layer,
                        GLenum access,
                        GLenum format);
  void bindVertexArray(GLuint vao);
  void blendColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
  void blendEquation(GLenum mode);
  void blendEquationSeparate(GLenum modeRGB, GLenum modeAlpha);
  void blendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha);
  virtual void blendFunc(GLenum sfactor, GLenum dfactor);
  void blendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha);
  void blendFuncSeparatei(GLuint buf,
                          GLenum srcRGB,
                          GLenum dstRGB,
                          GLenum srcAlpha,
                          GLenum dstAlpha);
  void blitFramebuffer(GLint srcX0,
                       GLint srcY0,
                       GLint srcX1,
                       GLint srcY1,
                       GLint dstX0,
                       GLint dstY0,
                       GLint dstX1,
                       GLint dstY1,
                       GLbitfield mask,
                       GLenum filter);
  void bufferData(GLenum target, GLsizeiptr size, const GLvoid* IGL_NULLABLE data, GLenum usage);
  void bufferSubData(GLenum target,
                     GLintptr offset,
                     GLsizeiptr size,
                     const GLvoid* IGL_NULLABLE data);
  virtual GLenum checkFramebufferStatus(GLenum target);
  void clear(GLbitfield mask);
  void clearBufferfv(GLenum buffer, GLint drawBuffer, const GLfloat* IGL_NULLABLE value);
  void clearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
  void clearDepthf(GLfloat depth);
  void clearStencil(GLint s);
  GLenum clientWaitSync(GLsync IGL_NULLABLE sync, GLbitfield flags, GLuint64 timeoutNs);
  void colorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
  void colorMaski(GLuint index, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
  void compileShader(GLuint shader);
  void compressedTexImage2D(GLenum target,
                            GLint level,
                            GLenum internalformat,
                            GLsizei width,
                            GLsizei height,
                            GLint border,
                            GLsizei imageSize,
                            const GLvoid* IGL_NULLABLE data);
  void compressedTexImage3D(GLenum target,
                            GLint level,
                            GLenum internalformat,
                            GLsizei width,
                            GLsizei height,
                            GLsizei depth,
                            GLint border,
                            GLsizei imageSize,
                            const GLvoid* IGL_NULLABLE data);
  void compressedTexSubImage2D(GLenum target,
                               GLint level,
                               GLint xoffset,
                               GLint yoffset,
                               GLsizei width,
                               GLsizei height,
                               GLenum format,
                               GLsizei imageSize,
                               const GLvoid* IGL_NULLABLE data);
  void compressedTexSubImage3D(GLenum target,
                               GLint level,
                               GLint xoffset,
                               GLint yoffset,
                               GLint zoffset,
                               GLsizei width,
                               GLsizei height,
                               GLsizei depth,
                               GLenum format,
                               GLsizei imageSize,
                               const GLvoid* IGL_NULLABLE data);
  void copyBufferSubData(GLenum readtarget,
                         GLenum writetarget,
                         GLintptr readoffset,
                         GLintptr writeoffset,
                         GLsizeiptr size);
  void copyTexSubImage2D(GLenum target,
                         GLint level,
                         GLint xoffset,
                         GLint yoffset,
                         GLint x,
                         GLint y,
                         GLsizei width,
                         GLsizei height);
  void createMemoryObjects(GLsizei n, GLuint* IGL_NULLABLE objects);
  GLuint createProgram();
  GLuint createShader(GLenum shaderType);
  virtual void cullFace(GLint mode);
  void debugMessageCallback(PFNIGLDEBUGPROC IGL_NULLABLE callback,
                            const void* IGL_NULLABLE userParam);
  void debugMessageInsert(GLenum source,
                          GLenum type,
                          GLuint id,
                          GLenum severity,
                          GLsizei length,
                          const GLchar* IGL_NULLABLE buf);
  void deleteBuffers(GLsizei n, const GLuint* IGL_NULLABLE buffers);
  void deleteFramebuffers(GLsizei n, const GLuint* IGL_NULLABLE framebuffers);
  void deleteMemoryObjects(GLsizei n, const GLuint* IGL_NULLABLE objects);
  void deleteRenderbuffers(GLsizei n, const GLuint* IGL_NULLABLE renderbuffers);
  void deleteVertexArrays(GLsizei n, const GLuint* IGL_NULLABLE vertexArrays);
  void deleteProgram(GLuint program);
  void deleteShader(GLuint shaderId);
  void deleteSync(GLsync IGL_NULLABLE sync);
  void deleteTextures(GLsizei n, const GLuint* IGL_NULLABLE textures);
  void depthFunc(GLenum func);
  void depthMask(GLboolean flag);
  void depthRangef(GLfloat n, GLfloat f);
  void detachShader(GLuint program, GLuint shader);
  virtual void disable(GLenum cap);
  void disablei(GLenum cap, GLuint index);
  void disableVertexAttribArray(GLuint index);
  void drawArrays(GLenum mode, GLint first, GLsizei count);
  void drawArraysIndirect(GLenum mode, const GLvoid* IGL_NULLABLE indirect);
  void drawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
  void drawBuffers(GLsizei n, GLenum* IGL_NULLABLE buffers);
  void drawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid* IGL_NULLABLE indices);
  void drawElementsInstanced(GLenum mode,
                             GLsizei count,
                             GLenum type,
                             const GLvoid* IGL_NULLABLE indices,
                             GLsizei instancecount);
  void drawElementsIndirect(GLenum mode, GLenum type, const GLvoid* IGL_NULLABLE indirect);
  void multiDrawArraysIndirect(GLenum mode,
                               const void* IGL_NULLABLE indirect,
                               GLsizei drawcount,
                               GLsizei stride);
  void multiDrawElementsIndirect(GLenum mode,
                                 GLenum type,
                                 const void* IGL_NULLABLE indirect,
                                 GLsizei drawcount,
                                 GLsizei stride);
  virtual void enable(GLenum cap);
  void enablei(GLenum cap, GLuint index);
  void enableVertexAttribArray(GLuint index);
  GLsync IGL_NULLABLE fenceSync(GLenum condition, GLbitfield flags);
  void finish();
  void flush();
  void framebufferRenderbuffer(GLenum target,
                               GLenum attachment,
                               GLenum renderbuffertarget,
                               GLuint renderbuffer);
  void framebufferTexture2D(GLenum target,
                            GLenum attachment,
                            GLenum textarget,
                            GLuint texture,
                            GLint level);
  void framebufferTexture2DMultisample(GLenum target,
                                       GLenum attachment,
                                       GLenum textarget,
                                       GLuint texture,
                                       GLint level,
                                       GLsizei samples);
  void framebufferTextureLayer(GLenum target,
                               GLenum attachment,
                               GLuint texture,
                               GLint level,
                               GLint layer);
  virtual void frontFace(GLenum mode);
  virtual void polygonFillMode(GLenum mode);
  void generateMipmap(GLenum target);
  void genBuffers(GLsizei n, GLuint* IGL_NULLABLE buffers);
  void genFramebuffers(GLsizei n, GLuint* IGL_NULLABLE framebuffers);
  void genRenderbuffers(GLsizei n, GLuint* IGL_NULLABLE renderbuffers);
  void genTextures(GLsizei n, GLuint* IGL_NULLABLE textures);
  void genVertexArrays(GLsizei n, GLuint* IGL_NULLABLE vertexArrays);
  void getActiveAttrib(GLuint program,
                       GLuint index,
                       GLsizei bufsize,
                       GLsizei* IGL_NULLABLE length,
                       GLint* IGL_NULLABLE size,
                       GLenum* IGL_NULLABLE type,
                       GLchar* IGL_NULLABLE name) const;
  void getActiveUniform(GLuint program,
                        GLuint index,
                        GLsizei bufsize,
                        GLsizei* IGL_NULLABLE length,
                        GLint* IGL_NULLABLE size,
                        GLenum* IGL_NULLABLE type,
                        GLchar* IGL_NULLABLE name) const;

  void getActiveUniformsiv(GLuint program,
                           GLsizei uniformCount,
                           const GLuint* IGL_NULLABLE uniformIndices,
                           GLenum pname,
                           GLint* IGL_NULLABLE params) const;
  void getActiveUniformBlockiv(GLuint program,
                               GLuint index,
                               GLenum pname,
                               GLint* IGL_NULLABLE params) const;
  void getActiveUniformBlockName(GLuint program,
                                 GLuint index,
                                 GLsizei bufSize,
                                 GLsizei* IGL_NULLABLE length,
                                 GLchar* IGL_NULLABLE uniformBlockName) const;
  GLint getAttribLocation(GLuint program, const GLchar* IGL_NULLABLE name) const;
  void getBooleanv(GLenum pname, GLboolean* IGL_NULLABLE params) const;
  void getBufferParameteriv(GLenum target, GLenum pname, GLint* IGL_NULLABLE params) const;
  GLuint getDebugMessageLog(GLuint count,
                            GLsizei bufSize,
                            GLenum* IGL_NULLABLE sources,
                            GLenum* IGL_NULLABLE types,
                            GLuint* IGL_NULLABLE ids,
                            GLenum* IGL_NULLABLE severities,
                            GLsizei* IGL_NULLABLE lengths,
                            GLchar* IGL_NULLABLE messageLog) const;
  virtual GLenum getError() const;
  void getFramebufferAttachmentParameteriv(GLenum target,
                                           GLenum attachment,
                                           GLenum pname,
                                           GLint* IGL_NULLABLE params) const;
  void getIntegerv(GLenum pname, GLint* IGL_NULLABLE params) const;
  /// Needs InternalFeatures::InternalFormatQuery
  void getInternalformativ(GLenum target,
                           GLenum internalformat,
                           GLenum pname,
                           GLsizei bufSize,
                           GLint* IGL_NULLABLE params) const;
  void getProgramiv(GLuint program, GLenum pname, GLint* IGL_NULLABLE params) const;
  void getProgramInterfaceiv(GLuint program,
                             GLenum programInterface,
                             GLenum pname,
                             GLint* IGL_NULLABLE params) const;
  void getProgramInfoLog(GLuint program,
                         GLsizei bufsize,
                         GLsizei* IGL_NULLABLE length,
                         GLchar* IGL_NULLABLE infolog) const;
  GLuint getProgramResourceIndex(GLuint program,
                                 GLenum programInterface,
                                 const GLchar* IGL_NULLABLE name) const;
  void getProgramResourceiv(GLuint program,
                            GLenum programInterface,
                            GLuint index,
                            GLsizei propCount,
                            const GLenum* IGL_NULLABLE props,
                            GLsizei bufSize,
                            GLsizei* IGL_NULLABLE length,
                            GLint* IGL_NULLABLE params);
  void getProgramResourceName(GLuint program,
                              GLenum programInterface,
                              GLuint index,
                              GLsizei bufSize,
                              GLsizei* IGL_NULLABLE length,
                              char* IGL_NULLABLE name) const;
  void getShaderiv(GLuint shader, GLenum pname, GLint* IGL_NULLABLE params) const;
  void getShaderInfoLog(GLuint shader,
                        GLsizei maxLength,
                        GLsizei* IGL_NULLABLE length,
                        GLchar* IGL_NULLABLE infoLog) const;
  virtual const GLubyte* IGL_NULLABLE getString(GLenum name) const;
  virtual const GLubyte* IGL_NULLABLE getStringi(GLenum name, GLuint index) const;
  void getSynciv(GLsync IGL_NULLABLE sync,
                 GLenum pname,
                 GLsizei bufSize,
                 GLsizei* IGL_NULLABLE length,
                 GLint* IGL_NULLABLE values) const;
  void getUniformiv(GLuint program, GLint location, GLint* IGL_NULLABLE params) const;
  GLint getUniformLocation(GLuint program, const GLchar* IGL_NULLABLE name) const;
  void importMemoryFd(GLuint memory, GLuint64 size, GLenum handleType, GLint fd);
  void invalidateFramebuffer(GLenum target,
                             GLsizei numAttachments,
                             const GLenum* IGL_NULLABLE attachments);
  GLboolean isEnabled(GLenum cap);
  GLboolean isTexture(GLuint texture);
  void linkProgram(GLuint program);
  void* IGL_NULLABLE mapBuffer(GLenum target, GLenum access);
  void* IGL_NULLABLE mapBufferRange(GLenum target,
                                    GLintptr offset,
                                    GLsizeiptr length,
                                    GLbitfield access);
  void objectLabel(GLenum identifier, GLuint name, GLsizei length, const char* IGL_NULLABLE label);
  void pixelStorei(GLenum pname, GLint param);
  void polygonOffsetClamp(GLfloat factor, GLfloat units, float clamp);
  void popDebugGroup();
  void pushDebugGroup(GLenum source, GLuint id, GLsizei length, const GLchar* IGL_NULLABLE message);
  void readBuffer(GLenum src);
  void readPixels(GLint x,
                  GLint y,
                  GLsizei width,
                  GLsizei height,
                  GLenum format,
                  GLenum type,
                  GLvoid* IGL_NULLABLE pixels);
  void renderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
  void renderbufferStorageMultisample(GLenum target,
                                      GLsizei samples,
                                      GLenum internalformat,
                                      GLsizei width,
                                      GLsizei height);
  /// Like renderbufferStorageMultisample(), for a renderbuffer that is attached to a framebuffer
  /// together with textures that use GL_EXT_multisampled_render_to_texture. That framebuffer can be
  /// incomplete with a renderbuffer whose storage renderbufferStorageMultisample() specified.
  void renderbufferStorageMultisampleImplicit(GLenum target,
                                              GLsizei samples,
                                              GLenum internalformat,
                                              GLsizei width,
                                              GLsizei height);
  /// The largest number of samples that GL_EXT_multisampled_render_to_texture, or its IMG version
  /// where that is what the context uses, accepts for any format
  [[nodiscard]] GLint getMaxMultisampledRenderToTextureSamples();
  /// Like getMaxMultisampledRenderToTextureSamples(), for textures and renderbuffers with
  /// `internalFormat`. A format that cannot be multisampled gets 0. Where the context cannot
  /// query it (before OpenGL ES 3.0), the limit for any format applies.
  [[nodiscard]] GLint getMaxMultisampledRenderToTextureSamples(GLenum internalFormat);
  void framebufferTextureMultiview(GLenum target,
                                   GLenum attachment,
                                   GLuint texture,
                                   GLint level,
                                   GLint baseViewIndex,
                                   GLsizei numViews);
  void framebufferTextureMultisampleMultiview(GLenum target,
                                              GLenum attachment,
                                              GLuint texture,
                                              GLint level,
                                              GLsizei samples,
                                              GLint baseViewIndex,
                                              GLsizei numViews);
  void scissor(GLint x, GLint y, GLsizei width, GLsizei height);
  virtual void setEnabled(bool shouldEnable, GLenum cap);
  void shaderSource(GLuint shader,
                    GLsizei count,
                    const GLchar * IGL_NULLABLE * IGL_NULLABLE string,
                    const GLint* IGL_NULLABLE length);
  void stencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask);
  void stencilMask(GLuint mask);
  void stencilMaskSeparate(GLenum face, GLuint mask);
  void stencilOpSeparate(GLenum face, GLenum fail, GLenum zfail, GLenum zpass);
  void texStorage2D(GLenum target,
                    GLsizei levels,
                    GLenum internalformat,
                    GLsizei width,
                    GLsizei height);
  void texStorage3D(GLenum target,
                    GLsizei levels,
                    GLenum internalformat,
                    GLsizei width,
                    GLsizei height,
                    GLsizei depth);
  void texStorageMem2D(GLenum target,
                       GLsizei levels,
                       GLenum internalFormat,
                       GLsizei width,
                       GLsizei height,
                       GLuint memory,
                       GLuint64 offset);
  void texStorageMem3D(GLenum target,
                       GLsizei levels,
                       GLenum internalFormat,
                       GLsizei width,
                       GLsizei height,
                       GLsizei depth,
                       GLuint memory,
                       GLuint64 offset);
  void texImage2D(GLenum target,
                  GLint level,
                  GLint internalformat,
                  GLsizei width,
                  GLsizei height,
                  GLint border,
                  GLenum format,
                  GLenum type,
                  const GLvoid* IGL_NULLABLE data);

  void texImage3D(GLenum target,
                  GLint level,
                  GLint internalformat,
                  GLsizei width,
                  GLsizei height,
                  GLsizei depth,
                  GLint border,
                  GLenum format,
                  GLenum type,
                  const GLvoid* IGL_NULLABLE data);

  void texParameteri(GLenum target, GLenum pname, GLint param);
  void texSubImage2D(GLenum target,
                     GLint level,
                     GLint xoffset,
                     GLint yoffset,
                     GLsizei width,
                     GLsizei height,
                     GLenum format,
                     GLenum type,
                     const GLvoid* IGL_NULLABLE pixels);
  void texSubImage3D(GLenum target,
                     GLint level,
                     GLint xoffset,
                     GLint yoffset,
                     GLint zoffset,
                     GLsizei width,
                     GLsizei height,
                     GLsizei depth,
                     GLenum format,
                     GLenum type,
                     const GLvoid* IGL_NULLABLE pixels);
  void uniform1f(GLint location, GLfloat x);
  void uniform1fv(GLint location, GLsizei count, const GLfloat* IGL_NULLABLE v);
  void uniform1i(GLint location, GLint x);
  void uniform1iv(GLint location, GLsizei count, const GLint* IGL_NULLABLE v);
  void uniform2f(GLint location, GLfloat x, GLfloat y);
  void uniform2fv(GLint location, GLsizei count, const GLfloat* IGL_NULLABLE v);
  void uniform2i(GLint location, GLint x, GLint y);
  void uniform2iv(GLint location, GLsizei count, const GLint* IGL_NULLABLE v);
  void uniform3f(GLint location, GLfloat x, GLfloat y, GLfloat z);
  void uniform3fv(GLint location, GLsizei count, const GLfloat* IGL_NULLABLE v);
  void uniform3i(GLint location, GLint x, GLint y, GLint z);
  void uniform3iv(GLint location, GLsizei count, const GLint* IGL_NULLABLE v);
  void uniform4f(GLint location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
  void uniform4fv(GLint location, GLsizei count, const GLfloat* IGL_NULLABLE v);
  void uniform4i(GLint location, GLint x, GLint y, GLint z, GLint w);
  void uniform4iv(GLint location, GLsizei count, const GLint* IGL_NULLABLE v);
  void uniformBlockBinding(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
  void uniformMatrix2fv(GLint location,
                        GLsizei count,
                        GLboolean transpose,
                        const GLfloat* IGL_NULLABLE value);
  void uniformMatrix3fv(GLint location,
                        GLsizei count,
                        GLboolean transpose,
                        const GLfloat* IGL_NULLABLE value);
  void uniformMatrix4fv(GLint location,
                        GLsizei count,
                        GLboolean transpose,
                        const GLfloat* IGL_NULLABLE value);
  void unmapBuffer(GLenum target);
  void useProgram(GLuint program);
  void validateProgram(GLuint program);
  void vertexAttribPointer(GLuint index,
                           GLint size,
                           GLenum type,
                           GLboolean normalized,
                           GLsizei stride,
                           const GLvoid* IGL_NULLABLE ptr);
  void vertexAttribDivisor(GLuint index, GLuint divisor);
  void viewport(GLint x, GLint y, GLsizei width, GLsizei height);
  void waitSync(GLsync IGL_NULLABLE sync, GLbitfield flags, GLuint64 timeoutNs);

  void dispatchCompute(GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);
  void dispatchComputeIndirect(GLintptr indirect);
  void memoryBarrier(GLbitfield barriers);
  GLuint64 getTextureHandle(GLuint texture);
  void makeTextureHandleResident(GLuint64 handle);
  void makeTextureHandleNonResident(GLuint64 handle);

  /** Returns current `callCounter_` value. Exposed for testing only. */
  unsigned int getCallCount() const;

  unsigned int getCurrentDrawCount() const;

  unsigned int getShaderCompilationCount() const;

  // Utility functions
  [[nodiscard]] const DeviceFeatureSet& deviceFeatures() const;
  /// Calls bindBuffer(target, 0) or enqueues to run when deletion queue is
  /// flushed
  void unbindBuffer(GLenum target);

  // Log the next N frames
  void apiLogNextNDraws(unsigned int n);

  // Log everything between start() and end()
  void apiLogStart();
  void apiLogEnd();

  void setShouldValidateShaders(bool shouldValidateShaders);
  bool shouldValidateShaders() const;
  inline bool isDestructionAllowed() const {
    return lockCount_ == 0;
  }

  void resetCounters();

  /** Manual reference counting.
   * In some cases, mostly for performance reasons, we hold unprotected
   * references to the IContext. When doing so, use the functions below to
   * signal such references so we can at least throw an error when those
   * references become invalid.
   */
  // @fb-only
  bool addRef();
  bool releaseRef();

  // @fb-only
  /**
   * The goal here is to try to check whether 'this' is a valid object and not
   * a zombie. Ideally, this should be handled elsewhere, but until we solve
   * that more difficult problem we can at least provide reasonable error
   * messages to users.
   *
   * The idea is to set a specific chunk of memory within the object to a
   * known valid value in the constructor and clear it in the destructor.
   * Invoking this method on a valid object always returns true, and invoking
   * this method on a zombie pointer will simply check that memory offset from
   * the base pointer and most likely return false, unless that memory happens
   * to match our not-so-secret pattern.
   */
  bool isLikelyValidObject() const {
    return zombieGuard_ == kNotAZombie;
  }

  virtual bool eglSupportssRGB() {
    return true; // we don't have egl so assume support is good.
  }

 public:
  UnbindPolicy getUnbindPolicy() const;

  /** Sets unbind policy for *subsequent* scopes/render passes.
   *
   * For example, only new instances of RenderCommandEncoder will honor the
   * new unbind policy. Previous instances, on the other hand, use the policy
   * that was in place when they were created. Similarly, the Device's unbind
   * policy will not change until the next beginScope().
   */
  void setUnbindPolicy(UnbindPolicy newValue);

  /** Enables or Disables calling getError() after GL call
   * This check is enabled by defaults in debug mode, and this function can be
   * used to turn it off. In release mode, the option is hard coded to false
   * and this function has no effect.
   */
  void enableAutomaticErrorCheck(bool enable);

  // Manages an adapter pool as recreating this every frame causes unwanted
  // memory allocations.
  // @fb-only
  // @fb-only
  auto& getAdapterPool() {
    return renderAdapterPool_;
  }

  auto& getComputeAdapterPool() {
    return computeAdapterPool_;
  }

  // Called to check if the last OGL call resulted in an error.
  GLenum checkForErrors(const char* IGL_NULLABLE callerName, size_t lineNum) const;
  Result getLastError() const;

 public:
  mutable ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> bindGroupBuffersPool;
  mutable ldr::Pool<BindGroupTextureTag, BindGroupTextureDesc> bindGroupTexturesPool;

 protected:
  static std::unordered_map<void* IGL_NULLABLE, IContext*>& getExistingContexts();
  static void registerContext(void* IGL_NULLABLE glContext, IContext* IGL_NULLABLE context);
  static void unregisterContext(void* IGL_NULLABLE glContext);
  void initialize(Result* IGL_NULLABLE result = nullptr);
  void willDestroy(void* IGL_NULLABLE glContext);

 private:
  GLVersion initializeGLVersion(Result* IGL_NULLABLE result);
  void loadExtensions(std::string& extensions,
                      std::unordered_set<std::string>& supportedExtensions);

  bool alwaysCheckError_ = false; // TRUE to check error after each OGL call
  // Objects created on this context call back into it from whichever thread binds, uploads or
  // releases them, so a context shared with a loader thread is entered from two threads at once.
  // The bookkeeping those calls touch is therefore atomic.
  mutable std::atomic<GLenum> lastError_{GL_NO_ERROR};
  mutable std::atomic<unsigned int> callCounter_{0};
  std::atomic<unsigned int> drawCallCount_{0};
  std::atomic<unsigned int> shaderCompilationCount_{0};
  std::atomic<int> lockCount_{0}; // used by DestructionGuard
  std::atomic<int> refCount_{0}; // used by addRef/releaseRef
  bool shouldValidateShaders_ = false;

  // API Logging
  unsigned int apiLogDrawsLeft_ = 0;
  bool apiLogEnabled_ = IGL_API_LOG != 0;
#if IGL_API_LOG
  void setBoundTexture(GLenum target, GLuint texture);
  GLuint boundTexture(GLenum target) const;
  GLuint boundTexture2D_ = 0;
  GLuint boundTexture3D_ = 0;
  GLuint boundTextureCubeMap_ = 0;
  GLuint boundTexture2DArray_ = 0;
  GLuint boundTextureExternalOes_ = 0;

  void setBoundBuffer(GLenum target, GLuint buffer);
  GLuint boundBuffer(GLenum target) const;
  GLuint boundArrayBuffer_ = 0;
  GLuint boundCopyReadBuffer_ = 0;
  GLuint boundCopyWriteBuffer_ = 0;
  GLuint boundDrawIndirectBuffer_ = 0;
  GLuint boundElementArrayBuffer_ = 0;
  GLuint boundPixelPackBuffer_ = 0;
  GLuint boundPixelUnpackBuffer_ = 0;
  GLuint boundShaderStorageBuffer_ = 0;
  GLuint boundUniformBuffer_ = 0;

  GLuint boundVao_ = 0;
  GLuint boundRenderbuffer_ = 0;

  void setBoundFramebuffer(GLenum target, GLuint framebuffer);
  GLuint boundFramebuffer(GLenum target) const;
  void setFramebufferAttachment(GLenum target,
                                GLenum attachment,
                                bool isRenderbuffer,
                                GLuint object);
  const char* framebufferAttachmentType(GLenum target, GLenum attachment) const;
  const char* framebufferAttachmentType(GLenum target, GLenum buffer, GLint drawbuffer) const;
  GLuint framebufferAttachment(GLenum target, GLenum attachment) const;
  GLuint framebufferAttachment(GLenum target, GLenum buffer, GLint drawbuffer) const;
  const char* framebufferReadBufferType(GLenum target) const;
  GLuint framebufferReadBuffer(GLenum target) const;
  std::string affectedFramebufferAttachments(GLenum target, GLbitfield mask) const;
  std::string boundFramebufferAttachments(GLenum target) const;
  GLuint boundReadFramebuffer_ = 0;
  GLuint boundWriteFramebuffer_ = 0;
  struct FramebufferAttachment {
    std::array<GLuint, 32> colorAttachment{};
    std::bitset<32> colorIsRenderbuffer;
    GLuint depthAttachment = 0;
    bool depthIsRenderbuffer = false;
    GLuint stencilAttachment = 0;
    bool stencilIsRenderbuffer = false;
    uint32_t readBufferIndex = 0;
  };
  std::unordered_map<GLuint, FramebufferAttachment> framebufferAttachments_;

  std::string boundDrawBuffers() const;
  std::bitset<32> enabledAttributes_;
  std::array<GLuint, 32> boundAttributes_{};

  std::string boundImageTextures() const;
  std::string boundDrawTextures() const;
  GLuint activeTextureUnit_ = 0;
  std::array<GLuint, 32> boundDrawTextures_{};
  std::array<GLuint, 32> boundImageTextures_{};

  GLuint boundProgram_ = 0;
  struct ProgramUniform {
    GLenum type;
    bool isTexture = false;
    bool isImage = false;
  };
  mutable std::unordered_map<GLuint, std::unordered_map<std::string, ProgramUniform>>
      programUniforms_;
  mutable std::unordered_map<GLuint, std::unordered_map<GLint, std::string>>
      programUniformsByLocation_;
  mutable std::unordered_map<GLuint, std::unordered_map<GLuint, std::string>> programUbosByIndex_;
  mutable std::unordered_map<GLuint, std::unordered_map<GLint, std::string>> programUbosByLocation_;
  mutable std::unordered_map<GLuint, std::unordered_map<GLuint, std::string>> programSsbosByIndex_;
  mutable std::unordered_map<GLuint, std::unordered_map<GLint, std::string>>
      programSsbosByLocation_;
  std::array<GLuint, 32> boundSsbos_{};
  std::array<GLuint, 32> boundUbos_{};
  void addProgramUniform(GLuint program, GLsizei* length, GLenum* type, GLchar* name) const;
  void setProgramUniformLocation(GLuint program, const GLchar* name, GLint location) const;
  void setProgramResourceIndex(GLuint program,
                               GLenum programInterface,
                               const GLchar* name,
                               GLuint index) const;
  std::string boundUniformTexture(GLuint program, GLint location, GLint unit) const;
  std::string boundBufferName(GLuint program, GLenum target, GLuint index) const;
  void setBoundBufferByIndex(GLenum target, GLuint index, GLuint buffer);
  GLuint boundBufferByIndex(GLenum target, GLuint index) const;
  std::string boundBuffersByIndex(bool ssbos = true, bool ubos = true) const;
  bool lastCommandWasCompute_ = false;
  std::string identifierLabel(GLuint identifier, GLuint name) const;

  std::string affectedMemoryBarrierObjects(GLbitfield bits) const;
  std::string boundPixelBufferObjects() const;
#endif

  PFNIGLBINDIMAGETEXTUREPROC IGL_NULLABLE bindImageTexturerProc_ = nullptr;
  PFNIGLBINDVERTEXARRAYPROC IGL_NULLABLE bindVertexArrayProc_ = nullptr;
  PFNIGLBLENDEQUATIONSEPARATEIPROC IGL_NULLABLE blendEquationSeparateiProc_ = nullptr;
  PFNIGLBLENDFUNCSEPARATEIPROC IGL_NULLABLE blendFuncSeparateiProc_ = nullptr;
  PFNIGLBLITFRAMEBUFFERPROC IGL_NULLABLE blitFramebufferProc_ = nullptr;
  PFNIGLCLEARDEPTHFPROC IGL_NULLABLE clearDepthfProc_ = nullptr;
  PFNIGLCLIENTWAITSYNCPROC IGL_NULLABLE clientWaitSyncProc_ = nullptr;
  PFNIGLCOLORMASKIPROC IGL_NULLABLE colorMaskiProc_ = nullptr;
  PFNIGLCOMPRESSEDTEXIMAGE3DPROC IGL_NULLABLE compressedTexImage3DProc_ = nullptr;
  PFNIGLCOMPRESSEDTEXSUBIMAGE3DPROC IGL_NULLABLE compressedTexSubImage3DProc_ = nullptr;
  PFNIGLDEBUGMESSAGECALLBACKPROC IGL_NULLABLE debugMessageCallbackProc_ = nullptr;
  PFNIGLDEBUGMESSAGEINSERTPROC IGL_NULLABLE debugMessageInsertProc_ = nullptr;
  PFNIGLDELETESYNCPROC IGL_NULLABLE deleteSyncProc_ = nullptr;
  PFNIGLDELETEVERTEXARRAYSPROC IGL_NULLABLE deleteVertexArraysProc_ = nullptr;
  PFNIGLDISABLEIPROC IGL_NULLABLE disableiProc_ = nullptr;
  PFNIGLDRAWBUFFERSPROC IGL_NULLABLE drawBuffersProc_ = nullptr;
  PFNIGLENABLEIPROC IGL_NULLABLE enableiProc_ = nullptr;
  PFNIGLFENCESYNCPROC IGL_NULLABLE fenceSyncProc_ = nullptr;
  PFNIGLFRAMEBUFFERTEXTURE2DMULTISAMPLEPROC IGL_NULLABLE framebufferTexture2DMultisampleProc_ =
      nullptr;
  PFNIGLINVALIDATEFRAMEBUFFERPROC IGL_NULLABLE invalidateFramebufferProc_ = nullptr;
  PFNIGLGENVERTEXARRAYSPROC IGL_NULLABLE genVertexArraysProc_ = nullptr;
  mutable PFNIGLGETDEBUGMESSAGELOGPROC IGL_NULLABLE getDebugMessageLogProc_ = nullptr;
  mutable PFNIGLGETSYNCIVPROC IGL_NULLABLE getSyncivProc_ = nullptr;
  PFNIGLGETTEXTUREHANDLEPROC IGL_NULLABLE getTextureHandleProc_ = nullptr;
  PFNIGLMAKETEXTUREHANDLERESIDENTPROC IGL_NULLABLE makeTextureHandleResidentProc_ = nullptr;
  PFNIGLMAKETEXTUREHANDLENONRESIDENTPROC IGL_NULLABLE makeTextureHandleNonResidentProc_ = nullptr;
  PFNIGLMAPBUFFERPROC IGL_NULLABLE mapBufferProc_ = nullptr;
  PFNIGLMAPBUFFERRANGEPROC IGL_NULLABLE mapBufferRangeProc_ = nullptr;
  PFNIGLMEMORYBARRIERPROC IGL_NULLABLE memoryBarrierProc_ = nullptr;
  PFNIGLOBJECTLABELPROC IGL_NULLABLE objectLabelProc_ = nullptr;
  PFNIGLPOPDEBUGGROUPPROC IGL_NULLABLE popDebugGroupProc_ = nullptr;
  PFNIGLPUSHDEBUGGROUPPROC IGL_NULLABLE pushDebugGroupProc_ = nullptr;
  PFNIGLREADBUFFERPROC IGL_NULLABLE readBufferProc_ = nullptr;
  PFNIGLRENDERBUFFERSTORAGEMULTISAMPLEPROC IGL_NULLABLE renderbufferStorageMultisampleProc_ =
      nullptr;
  PFNIGLRENDERBUFFERSTORAGEMULTISAMPLEPROC IGL_NULLABLE
      renderbufferStorageMultisampleImplicitProc_ = nullptr;
  PFNIGLTEXIMAGE3DPROC IGL_NULLABLE texImage3DProc_ = nullptr;
  PFNIGLTEXSTORAGE2DPROC IGL_NULLABLE texStorage2DProc_ = nullptr;
  PFNIGLTEXSTORAGE3DPROC IGL_NULLABLE texStorage3DProc_ = nullptr;
  PFNIGLTEXSUBIMAGE3DPROC IGL_NULLABLE texSubImage3DProc_ = nullptr;
  PFNIGLUNMAPBUFFERPROC IGL_NULLABLE unmapBufferProc_ = nullptr;
  PFNIGLVERTEXATTRIBDIVISORPROC IGL_NULLABLE vertexAttribDivisorProc_ = nullptr;
  PFNIGLWAITSYNCPROC IGL_NULLABLE waitSyncProc_ = nullptr;

  /// Responsible for holding onto operations queued for deletion when not in
  /// context. All operations to non-scratch queues are suyncronized by one
  /// mutex
  struct SynchronizedDeletionQueues {
   public:
    void flushDeletionQueue(IContext& context);

    void queueDeleteBuffers(GLsizei n, const GLuint* IGL_NULLABLE buffers);
    void queueUnbindBuffer(GLenum target);
    void queueDeleteFramebuffers(GLsizei n, const GLuint* IGL_NULLABLE framebuffers);
    void queueDeleteRenderbuffers(GLsizei n, const GLuint* IGL_NULLABLE renderbuffers);
    void queueDeleteVertexArrays(GLsizei n, const GLuint* IGL_NULLABLE vertexArrays);
    void queueDeleteProgram(GLuint program);
    void queueDeleteShader(GLuint shaderId);
    void queueDeleteTextures(GLsizei n, const GLuint* IGL_NULLABLE textures);

   private:
    /// This is called by flushDeletionQueue to swap fooQueue w/
    /// scratchFooQueue
    void swapScratchDeletionQueues();

    // These are swapped with the main queues then read to perform operations.
    // They can be read from flushDeletionQueue w/o synchronization.
    std::vector<GLuint> scratchBuffersQueue_;
    std::unordered_set<GLenum> scratchUnbindBuffersQueue_;
    std::vector<GLuint> scratchFramebuffersQueue_;
    std::vector<GLuint> scratchRenderbuffersQueue_;
    std::vector<GLuint> scratchVertexArraysQueue_;
    std::vector<GLuint> scratchProgramQueue_;
    std::vector<GLuint> scratchShaderQueue_;
    std::vector<GLuint> scratchTexturesQueue_;

    // Guards all the non-scratch queues.
    std::mutex deletionQueueMutex_;

    // Resources queued for deletion
    std::vector<GLuint> buffersQueue_;
    // These are the targets for the buffers that we enqueued for deletion
    // that we need to unbind
    std::unordered_set<GLenum> unbindBuffersQueue_;
    std::vector<GLuint> framebuffersQueue_;
    std::vector<GLuint> renderbuffersQueue_;
    std::vector<GLuint> vertexArraysQueue_;
    std::vector<GLuint> programQueue_;
    std::vector<GLuint> shaderQueue_;
    std::vector<GLuint> texturesQueue_;
  };

  SynchronizedDeletionQueues deletionQueues_;

  UnbindPolicy unbindPolicy_ = UnbindPolicy::Default;

  void getGLMajorAndMinorVersions(GLint& majorVersion, GLint& minorVersion) const;
  void getGlMajorAndMinorVersions(GLint& majorVersion, GLint& minorVersion) const;
  /// Whether the entry points of GL_EXT_multisampled_render_to_texture that the context uses are
  /// those of the IMG version. Like the calls that use them, this prefers the EXT version.
  [[nodiscard]] bool usesImgMultisampledRenderToTexture() const;
  friend class DestructionGuard;
  std::vector<std::unique_ptr<RenderCommandAdapter>> renderAdapterPool_;
  std::vector<std::unique_ptr<ComputeCommandAdapter>> computeAdapterPool_;

  DeviceFeatureSet deviceFeatureSet_;

  // For framebufferTexture2DMultisample and renderbufferStorageMultisampleImplicit
  GLint maxSamples_ = -1;
  std::unordered_map<GLenum, GLint> maxSamplesByFormat_;
  // Some drivers only take GL_FRAMEBUFFER as the target of glFramebufferTexture2DMultisampleEXT().
  // The first call with GL_DRAW_FRAMEBUFFER or GL_READ_FRAMEBUFFER finds out which kind this is.
  bool multisampleNeedsFramebufferTarget_ = false;
  bool multisampleDrawTargetVerified_ = false;
  GLint maxDebugStackSize_ = -1;
  GLint debugStackSize_ = 0;

  static constexpr uint64_t kNotAZombie = 0xdeadc0def3315badLL;
  uint64_t zombieGuard_ = kNotAZombie;
};

} // namespace igl::opengl
