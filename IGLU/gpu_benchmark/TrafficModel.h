/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <igl/Framebuffer.h>
#include <igl/RenderPass.h>
#include <igl/Texture.h>

namespace iglu::gpu_benchmark {

/// One attachment of a render pass, as the memory system sees it.
struct AttachmentTraffic {
  uint32_t bytesPerPixel = 4;
  uint32_t samples = 1;
  uint32_t width = 0;
  uint32_t height = 0;
  /// Array layers rendered to, for example 2 on a stereo framebuffer. Traffic scales with it.
  uint32_t layers = 1;
  igl::LoadAction loadAction = igl::LoadAction::DontCare;
  igl::StoreAction storeAction = igl::StoreAction::Store;
  /// Lives in tile memory only: no load or store traffic of its own; a MsaaResolve still writes the
  /// resolve target.
  bool memoryless = false;
  /// Bytes per pixel of the resolve target written by StoreAction::MsaaResolve. 0 means no resolve
  /// target. A resolve target is never memoryless.
  uint32_t resolveBytesPerPixel = 0;
  /// Resolves in addition to `storeAction`: with StoreAction::Store, the attachment is stored and
  /// resolved. StoreAction::MsaaResolve resolves already, so the resolve counts once.
  bool alsoResolve = false;
};

/// A texture read by the shaders of a pass.
struct SampledTexture {
  uint32_t bytesPerPixel = 4;
  uint32_t width = 0;
  uint32_t height = 0;
  /// Size of the texture in bytes. When non-zero it replaces `bytesPerPixel * width * height`,
  /// which is how block-compressed formats get an exact size. describeRenderPass() sets it.
  uint64_t bytes = 0;
};

/// A render pass: its attachments and the textures its shaders sample.
struct PassTraffic {
  std::string name;
  std::vector<AttachmentTraffic> attachments;
  std::vector<SampledTexture> sampledTextures;
};

/// Estimated external (DRAM) memory traffic, in bytes.
///
/// The model is an ideal tile-based GPU: an attachment reaches memory only through its load and
/// store actions, input attachment reads stay in tile memory, and nothing is cached. It is an
/// estimate, neither an upper nor a lower bound of what hardware measures. Real hardware adds
/// binning, vertex, uniform and cache-miss traffic, and framebuffer compression removes some.
struct TrafficEstimate {
  uint64_t readBytes = 0;
  uint64_t writeBytes = 0;

  [[nodiscard]] uint64_t totalBytes() const noexcept;
};

[[nodiscard]] inline uint64_t TrafficEstimate::totalBytes() const noexcept {
  return readBytes + writeBytes;
}

/// Sums the estimated traffic of `passes`. Per attachment that is not memoryless, with `samples`
/// and `layers` counted as at least 1:
///  - LoadAction::Load reads `bytesPerPixel * samples * width * height * layers`; Clear and
///    DontCare read 0
///  - StoreAction::Store writes `bytesPerPixel * samples * width * height * layers`
///  - StoreAction::MsaaResolve writes `resolveBytesPerPixel * width * height * layers`: the resolve
///    target only, the samples are discarded
///  - StoreAction::DontCare writes 0
///  - `alsoResolve` adds the resolve write to the store action, and a resolve counts once
///
/// A memoryless attachment reads 0 and has no store traffic of its own, but a resolve still writes
/// `resolveBytesPerPixel * width * height * layers`, because the resolve target is never
/// memoryless. For example, a 4x RGBA8 attachment resolved to RGBA8 costs
/// `4 * width * height` bytes when it is memoryless. Stored by one pass and resolved by a second,
/// it costs `(16 + 16 + 4) * width * height` bytes.
///
/// Each sampled texture is read once: `bytes` if that is set, `bytesPerPixel * width * height`
/// otherwise. Input attachment reads are free, so passes do not list them.
[[nodiscard]] TrafficEstimate estimateTraffic(const std::vector<PassTraffic>& passes);

/// Bytes per pixel of an uncompressed `format`. Compressed formats have no whole number of bytes
/// per pixel: debug builds assert and the result is 0.
[[nodiscard]] uint32_t bytesPerPixel(igl::TextureFormat format);

/// Describes `renderPass` as it runs on `framebuffer`, so the estimate matches what is submitted.
///
/// Lists every color attachment of the framebuffer. The Vulkan and Metal encoders reject a color
/// attachment that has no `renderPass.colorAttachments` entry: debug builds assert, and it is still
/// modeled with a default igl::RenderPassDesc::AttachmentDesc. Sizes are those of the attachment's
/// mip level, and every attachment of a stereo framebuffer counts 2 layers.
///
/// A depth-stencil texture attached as both depth and stencil is listed once. It loads if either
/// aspect loads and stores if either aspect stores. It also resolves if the depth aspect resolves,
/// because on Vulkan only the depth store action triggers the resolve. `resolveBytesPerPixel`
/// comes from the resolve texture of the attachment, if any.
///
/// Each of `sampledTextures` is read once and sized with
/// igl::TextureFormatProperties::getBytesPerLayer(), so block-compressed formats work.
///
/// A framebuffer without textures is not supported, such as the OpenGL backend's
/// `CurrentFramebuffer`, the placeholder for the default framebuffer.
[[nodiscard]] PassTraffic describeRenderPass(
    std::string name,
    const igl::RenderPassDesc& renderPass,
    const igl::IFramebuffer& framebuffer,
    const std::vector<std::shared_ptr<igl::ITexture>>& sampledTextures = {});

struct FootprintEstimate {
  /// Bytes the textures occupy in memory.
  uint64_t bytes = 0;
  /// Bytes of ResourceStorage::Memoryless textures.
  uint64_t memorylessBytes = 0;
  /// The memoryless textures are counted in `bytes` because the device has no memoryless
  /// attachments, so regular memory backs them.
  bool memorylessBacked = false;
};

/// Sums igl::ITexture::getEstimatedSizeInBytes() times the sample count over the unique textures
/// of `textures`. That size covers every mip level, array layer, cube face and 3D slice, and it
/// handles block-compressed formats. Textures are unique by object, so texture views over one
/// image are counted once per view.
///
/// Memoryless textures add to `memorylessBytes`. They add to `bytes` only when
/// `deviceHasMemorylessAttachments` is false, and `memorylessBacked` is then true.
[[nodiscard]] FootprintEstimate estimateFootprint(
    const std::vector<std::shared_ptr<igl::ITexture>>& textures,
    bool deviceHasMemorylessAttachments);

} // namespace iglu::gpu_benchmark
