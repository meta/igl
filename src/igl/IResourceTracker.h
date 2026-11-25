/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <igl/Core.h>

namespace igl {

class ITexture;
class IBuffer;
class IFramebuffer;
class ISamplerState;
class IShaderLibrary;
class IShaderModule;
class IShaderStages;

template<typename T>
class ITrackedResource;

/**
 * @brief The IResourceTracker interface allows clients to implement their own Resource Trackers.
 * Clients can track Textures, Buffers, Framebuffers or other TrackedResources
 */
class IResourceTracker {
 public:
  virtual ~IResourceTracker() = default;
  IResourceTracker() noexcept = default;

  /**
   * @brief Informs the tracker that the texture has been created
   *
   * @param texture Texture to be tracked
   */
  virtual void didCreate(const ITexture* texture) noexcept = 0;

  /**
   * @brief Informs the tracker that the texture will be deleted
   *
   * @param texture Texture which will be deleted
   */
  virtual void willDelete(const ITexture* texture) noexcept = 0;

  /**
   * @brief Informs the tracker that the buffer has been created
   *
   * @param buffer Buffer to be tracked
   */
  virtual void didCreate(const IBuffer* buffer) noexcept = 0;

  /**
   * @brief Informs the tracker that the buffer will be deleted
   *
   * @param buffer Buffer which will be deleted
   */
  virtual void willDelete(const IBuffer* buffer) noexcept = 0;

  /**
   * @brief Informs the tracker that the framebuffer has been created
   *
   * @param framebuffer Framebuffer to be tracked
   */
  virtual void didCreate(const IFramebuffer* framebuffer) noexcept = 0;

  /**
   * @brief Informs the tracker that the framebuffer will be deleted
   *
   * @param framebuffer Framebuffer which will be deleted
   */
  virtual void willDelete(const IFramebuffer* framebuffer) noexcept = 0;

  /**
   * @brief Informs the tracker that the sampler state has been created
   *
   * @param samplerState Sampler state to be tracked
   */
  virtual void didCreate(const ISamplerState* samplerState) noexcept = 0;

  /**
   * @brief Informs the tracker that the sampler will be deleted
   *
   * @param samplerState Sampler state which will be deleted
   */
  virtual void willDelete(const ISamplerState* samplerState) noexcept = 0;

  /**
   * @brief Informs the tracker that the shader library has been created
   *
   * @param shaderLibrary Shader library to be tracked
   */
  virtual void didCreate(const IShaderLibrary* shaderLibrary) noexcept = 0;

  /**
   * @brief Informs the tracker that the shader library be deleted
   *
   * @param shaderLibrary Shader library which will be deleted
   */
  virtual void willDelete(const IShaderLibrary* shaderLibrary) noexcept = 0;

  /**
   * @brief Informs the tracker that the shader module has been created
   *
   * @param shaderModule Shader module to be tracked
   */
  virtual void didCreate(const IShaderModule* shaderModule) noexcept = 0;

  /**
   * @brief Informs the tracker that the shader module will be deleted
   *
   * @param shaderModule Shader module which will be deleted
   */
  virtual void willDelete(const IShaderModule* shaderModule) noexcept = 0;

  /**
   * @brief Informs the tracker that the shader stages has been created
   *
   * @param shaderStages Shader stages to be tracked
   */
  virtual void didCreate(const IShaderStages* shaderStages) noexcept = 0;

  /**
   * @brief Informs the tracker that the shader stages will be deleted
   *
   * @param shaderStages Shader stages which will be deleted
   */
  virtual void willDelete(const IShaderStages* shaderStages) noexcept = 0;

  template<typename T>
  void didCreate([[maybe_unused]] const ITrackedResource<T>* resource) noexcept {
    IGL_DEBUG_ASSERT_NOT_REACHED();
  }
  template<typename T>
  void willDelete([[maybe_unused]] const ITrackedResource<T>* resource) noexcept {
    IGL_DEBUG_ASSERT_NOT_REACHED();
  }

  /**
   * @brief Associates a name tag with the next resources to be tracked
   *
   * @param tag Name of the tag
   */
  virtual void pushTag(const char* tag) noexcept = 0;

  /**
   * @brief Pops the last name tag in the stack
   */
  virtual void popTag() noexcept = 0;
};

/**
 * @brief Allows any resources being tracked by a ResourceTracker to be associated with a name tag.
 * Makes sure that the tag is popped in the destructor.
 */
class ResourceTrackerTagGuard {
 public:
  ResourceTrackerTagGuard(std::shared_ptr<IResourceTracker> tracker, const char* tag) :
    tracker_(std::move(tracker)) {
    tracker_->pushTag(tag);
  }
  ~ResourceTrackerTagGuard() {
    tracker_->popTag();
  }

 private:
  std::shared_ptr<IResourceTracker> tracker_;
};

} // namespace igl
