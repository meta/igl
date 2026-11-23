/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <igl/Framebuffer.h>
#include <igl/RenderPass.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/Texture.h>

namespace igl {
class ICommandBuffer;
namespace opengl {

///--------------------------------------
/// MARK: - FramebufferBindingGuard

class FramebufferBindingGuard {
 public:
  explicit FramebufferBindingGuard(IContext& context);
  ~FramebufferBindingGuard();

 private:
  IContext& context_;
  GLuint currentRenderbuffer_ = 0;
  GLuint currentFramebuffer_ = 0;
  GLuint currentReadFramebuffer_ = 0;
  GLuint currentDrawFramebuffer_ = 0;
};

// Framebuffer encapsulates an immutable render target (attachments) and per-render pass state.
class Framebuffer : public WithContext, public IFramebuffer {
 public:
  explicit Framebuffer(IContext& context);
  virtual Viewport getViewport() const = 0;
  virtual void bind(const RenderPassDesc& renderPass) const = 0;
  virtual void unbind() const = 0;

  void bindBuffer() const;
  void bindBufferForRead() const;
  void copyBytesColorAttachment(ICommandQueue& /* unused */,
                                size_t index,
                                void* pixelBytes,
                                const TextureRangeDesc& range,
                                size_t bytesPerRow = 0) const override;

  void copyBytesDepthAttachment(ICommandQueue& /* unused */,
                                void* pixelBytes,
                                const TextureRangeDesc& range,
                                size_t bytesPerRow = 0) const override;

  void copyBytesStencilAttachment(ICommandQueue& /* unused */,
                                  void* pixelBytes,
                                  const TextureRangeDesc& range,
                                  size_t bytesPerRow = 0) const override;

  void copyTextureColorAttachment(ICommandQueue& cmdQueue,
                                  size_t index,
                                  std::shared_ptr<ITexture> destTexture,
                                  const TextureRangeDesc& range) const override;
  inline GLuint getId() const {
    return frameBufferID_;
  }

  inline std::shared_ptr<Framebuffer> getResolveFramebuffer() const {
    return resolveFramebuffer_;
  }

  [[nodiscard]] bool isSwapchainBound() const override;

  /// Whether the color attachments are textures that IGL attached to a framebuffer object that it
  /// created. False for the framebuffer of the window system and for CurrentFramebuffer, whose
  /// color attachment is a placeholder.
  [[nodiscard]] virtual bool hasTextureAttachments() const = 0;

 protected:
  void attachAsColor(ITexture& texture,
                     uint32_t index,
                     const Texture::AttachmentParams& params) const;
  void attachAsDepth(ITexture& texture, const Texture::AttachmentParams& params) const;
  void attachAsStencil(ITexture& texture, const Texture::AttachmentParams& params) const;
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  GLuint frameBufferID_ = 0;

  struct CachedState {
    FramebufferMode mode = FramebufferMode::Mono;
    uint8_t layer = 0;
    uint8_t face = 0;
    uint8_t mipLevel = 0;

    bool needsUpdate(FramebufferMode newMode,
                     uint8_t newLayer,
                     uint8_t newFace,
                     uint8_t newMipLevel);
    void updateCache(FramebufferMode newMode,
                     uint8_t newLayer,
                     uint8_t newFace,
                     uint8_t newMipLevel);
  };

  constexpr static auto kNumCachedStates = 8; // We allow up to 8 color attachments
  mutable std::array<CachedState, kNumCachedStates> colorCachedState_;
  mutable CachedState depthCachedState_;
  mutable CachedState stencilCachedState_;

  std::shared_ptr<Framebuffer> resolveFramebuffer_ = nullptr;
};

// CustomFramebuffer enables caller-defined attachments
//
// There are several kinds of framebuffer color attachments that are encapsulated
// by the opengl::Texture interface. For the OpenGL backend, we have different
// implementations of that interface whose usage supports TextureUsageBits::Attachment:
//
// 1. `opengl::TextureBuffer`. These are regular old textures, e.g. loaded from an
// image file. They can be read from a shader (or written to by a compute shader)
// 2. `opengl::TextureTarget`. These are renderbuffers. For storage, normally IGL
// allocates storage via glRenderbufferStorage. On iOS, the view's storage is used
// instead — see ios::PlatformDevice::createTextureFromNativeDrawable() where
// glRenderBufferStorage is replaced by [EAGLContext renderbufferStorage:fromDrawable:].
// 3. `opengl::macos::ViewTextureTarget`. This represents the color attachment of
// the implicit framebuffer supplied by NSOpenGLView. Here, IGL skips prepareResource()
// since GPU resources are owned by the client, i.e. `hasImplicitColorAttachment()` == true.
class CustomFramebuffer final : public Framebuffer {
 public:
  using Framebuffer::Framebuffer;
  ~CustomFramebuffer() override;

  // Accessors
  std::vector<size_t> getColorAttachmentIndices() const override;
  std::shared_ptr<ITexture> getColorAttachment(size_t index) const override;
  std::shared_ptr<ITexture> getResolveColorAttachment(size_t index) const override;
  std::shared_ptr<ITexture> getDepthAttachment() const override;
  std::shared_ptr<ITexture> getResolveDepthAttachment() const override;
  std::shared_ptr<ITexture> getStencilAttachment() const override;
  [[nodiscard]] FramebufferMode getMode() const override;

  // Methods
  void updateDrawable(std::shared_ptr<ITexture> texture) override;
  void updateDrawable(SurfaceTextures surfaceTextures) override;
  void updateResolveAttachment(std::shared_ptr<ITexture> texture) override;

  bool isInitialized() const;
  bool hasImplicitColorAttachment() const;
  void initialize(const FramebufferDesc& desc, Result* outResult);

  // CommandBuffer
  Viewport getViewport() const override;
  void bind(const RenderPassDesc& renderPass) const override;
  void unbind() const override;

  // opengl::Framebuffer
  [[nodiscard]] bool hasTextureAttachments() const override;

  /// Whether GL_EXT_multisampled_render_to_texture resolves the multisampled color attachments. The
  /// framebuffer then attaches their resolve textures instead of them, and the GPU resolves in tile
  /// memory, so getResolveFramebuffer() is null.
  [[nodiscard]] bool resolvesImplicitly() const;

 private:
  /// The attachments of renderTarget_ that an update replaced
  struct Replaced {
    bool color = false;
    bool resolve = false;
    bool depthStencil = false;
  };

  void prepareResource(const std::string& debugName, Result* outResult);
  void updateDrawableInternal(SurfaceTextures surfaceTextures, bool updateDepthStencil);
  void applyClearMask(int targetCount) const;
  void setupResolveFramebuffer(Result* outResult);

  /// @returns the number of samples to resolve implicitly with, or 0 if the attachments of
  /// renderTarget_ do not allow it
  [[nodiscard]] uint32_t getImplicitResolveSamples() const;
  void attachTextures() const;
  void detachTextures() const;
  void attachColor(size_t index, Texture::AttachmentParams params) const;
  void attachDepth(Texture::AttachmentParams params) const;
  void attachStencil(Texture::AttachmentParams params) const;
  /// Attaches only the textures that `replaced` has an effect on
  void attachReplacedTextures(const Replaced& replaced) const;
  /// Detaches the color attachments of a framebuffer that resolved implicitly
  /// @param previousColor What was attached to color attachment 0, whose texture an update may
  /// have replaced in renderTarget_ already
  void detachImplicitColors(const std::shared_ptr<ITexture>& previousColor) const;
  /// Resolves with the resolve framebuffer after all, with the textures of renderTarget_
  void fallBackToBlit();
  /// Attaches the textures of renderTarget_ again after an update replaced those of `replaced`,
  /// and moves the framebuffer between resolving implicitly and with a blit if they require it
  /// @param previousColor What was attached to color attachment 0 before the update
  void reattachTextures(const std::shared_ptr<ITexture>& previousColor, const Replaced& replaced);

  bool initialized_ = false;
  // The number of samples of the multisampled color attachments if resolvesImplicitly()
  uint32_t implicitSamples_ = 0;

  friend class Framebuffer; // Needed to enable copyBytesColorAttachment
  FramebufferDesc renderTarget_; // attachments
  mutable RenderPassDesc renderPass_;
};

class CurrentFramebuffer final : public Framebuffer {
 public:
  using Super = Framebuffer;
  explicit CurrentFramebuffer(IContext& context);

  // IFramebuffer
  std::vector<size_t> getColorAttachmentIndices() const override;
  std::shared_ptr<ITexture> getColorAttachment(size_t index) const override;
  std::shared_ptr<ITexture> getResolveColorAttachment(size_t index) const override;
  std::shared_ptr<ITexture> getDepthAttachment() const override;
  std::shared_ptr<ITexture> getResolveDepthAttachment() const override;
  std::shared_ptr<ITexture> getStencilAttachment() const override;
  void updateDrawable(std::shared_ptr<ITexture> texture) override;
  void updateDrawable(SurfaceTextures surfaceTextures) override;
  void updateResolveAttachment(std::shared_ptr<ITexture> texture) override;
  [[nodiscard]] FramebufferMode getMode() const override;

  // opengl::Framebuffer
  Viewport getViewport() const override;
  void bind(const RenderPassDesc& renderPass) const override;
  void unbind() const override;
  [[nodiscard]] bool hasTextureAttachments() const override;

 private:
  Viewport viewport_;
  std::shared_ptr<ITexture> colorAttachment_;
};

} // namespace opengl
} // namespace igl
