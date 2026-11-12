/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <string>
#include <igl/PlatformDevice.h>
#include <igl/Texture.h>
#include <igl/vulkan/Common.h>

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
struct AHardwareBuffer;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

namespace igl {
class ICommandBuffer;
} // namespace igl

namespace igl::vulkan {

class Device;

/// @brief Describes a reusable command buffer (see PlatformDevice::createReusableCommandBuffer())
struct ReusableCommandBufferDesc {
  std::string debugName;
  /// How a run replays the recording of the current slot. If false, ICommandQueue::submit() submits
  /// the reusable command buffer. If true, PlatformDevice::executeCommands() records the recording
  /// into another command buffer, which runs it when it is submitted; that command buffer begins
  /// and ends the render passes of the recording, which therefore cannot have subpasses.
  bool executeInCommandBuffer = false;
};

/// @brief Implements the igl::IPlatformDevice interface
class PlatformDevice : public IPlatformDevice {
 public:
  static constexpr PlatformDeviceType kType = igl::PlatformDeviceType::Vulkan;

  explicit PlatformDevice(Device& device);
  ~PlatformDevice() override = default;

  /// Creates a Depth Texture through the underlying VulkanSwapChain
  /// This currently is for development purposes only and will be removed in
  /// the future;
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::shared_ptr<ITexture> createTextureFromNativeDepth(uint32_t width,
                                                         uint32_t height,
                                                         Result* IGL_NULLABLE outResult);

  /// Creates a texture from a native drawable surface
  /// @param outResult optional result
  /// @return pointer to generated Texture or nullptr
  std::shared_ptr<ITexture> createTextureFromNativeDrawable(Result* IGL_NULLABLE outResult);

#if defined(IGL_ANDROID_HWBUFFER_SUPPORTED)
  std::shared_ptr<ITexture> createTextureWithSharedMemory(const TextureDesc& desc,
                                                          Result* IGL_NULLABLE outResult) const;
  std::shared_ptr<ITexture> createTextureWithSharedMemory(AHardwareBuffer* buffer,
                                                          Result* IGL_NULLABLE outResult) const;
#endif // defined(IGL_ANDROID_HWBUFFER_SUPPORTED)

  /// @param handle The handle to the GPU Fence
  /// @return The Vulkan fence associated with the handle
  [[nodiscard]] VkFence getVkFenceFromSubmitHandle(SubmitHandle handle) const;

  /// Waits on the GPU Fence associated with the handle
  /// @param handle The handle to the GPU Fence
  /// @return 'true' when ready or 'false' on time out (can be ignored)
  // NOLINTNEXTLINE(modernize-use-nodiscard)
  bool waitOnSubmitHandle(SubmitHandle handle, uint64_t timeoutNanoseconds = UINT64_MAX) const;

  /// Invoke a task when a SubmitHandle is ready
  void deferredTask(std::packaged_task<void()>&& task, SubmitHandle handle) const;

  /// @brief Returns true if createReusableCommandBuffer() can succeed: neither descriptor indexing
  /// nor VK_EXT_descriptor_buffer is enabled, because the context rewrites their descriptors while
  /// a recording could still bind them.
  [[nodiscard]] bool supportsReusableCommandBuffers() const;

  /// @brief Creates a reusable command buffer, which records its commands once per slot and replays
  /// the recording of the current slot on every later run without encoding the commands again.
  ///
  /// The slot is the frame-in-flight index of the device, one of as many slots as the frames that
  /// the device can have in flight. Every reusable command buffer and every engine on the device
  /// share it. It moves to the next slot at the first submission after commands that read a Ring
  /// buffer (BufferDesc::BufferAPIHintBits::Ring) are encoded, whichever command buffer they are
  /// encoded into, a recording included, and with every run of a reusable command buffer: a run
  /// with ICommandQueue::submit() moves it, and a run with executeCommands() moves it at the first
  /// submission after it. So a reusable command buffer can skip slots between two of its runs.
  ///
  /// Encode into it only while isRecorded() returns false: the encoded commands become the
  /// recording of the current slot, and the first run of the slot ends the recording. A run either
  /// submits the reusable command buffer with ICommandQueue::submit() or, if it was created with
  /// ReusableCommandBufferDesc::executeInCommandBuffer, executes it from another command buffer
  /// with executeCommands(). A run fails, and the command buffer stays in its current slot, if the
  /// last run of the slot has not completed in time, if a render pass of its recording is being
  /// encoded, which fails with Result::Code::InvalidOperation and lets the recording go on, or if
  /// recording the slot failed, which includes a recording that is encoded into or run while
  /// another slot is the current one: then the recording fails with Result::Code::InvalidOperation
  /// rather than run in another slot. If the shared slot goes around all the slots back to the slot
  /// of a recording in between, nothing fails, and the recording runs in its own slot. A slot whose
  /// recording failed holds no recording and can be recorded again; the images that the recording
  /// transitioned are back in the layouts that they had before it, unless other commands changed
  /// them since.
  ///
  /// A recording keeps everything that was encoded: resources, bindings, dynamic state, push
  /// constants and image layout transitions. Its resources must stay alive and unchanged as long
  /// as the command buffer, and until every command buffer that executes one of its slots is
  /// submitted, and every image it uses must be in the same layout at the start of each run as at
  /// the start and at the end of the recording of its slot. Render passes leave their color
  /// attachments ready to be sampled, and the textures of their Dependencies that have
  /// TextureDesc::TextureUsageBits::Attachment ready to be rendered to. Render passes cannot have
  /// timestamp queries; a Timer of a command buffer that executes a slot times its run. A reusable
  /// command buffer cannot present a swapchain image, which every run would present again. The
  /// mipmaps of a texture whose UNORM image has sRGB views go through a scratch image that lives
  /// only as long as ITexture::generateMipmap(), so generating them fails the recording with
  /// Result::Code::Unsupported.
  ///
  /// Each run adds to the draw statistics of the queue the draws that
  /// ICommandBuffer::incrementCurrentDrawCount() counted while its slot recorded, while
  /// IDevice::getCurrentDrawCount() counts only the draws that recordings encode.
  /// @returns nullptr, with Result::Code::Unsupported, if supportsReusableCommandBuffers() is
  /// false.
  std::shared_ptr<ICommandBuffer> createReusableCommandBuffer(const ReusableCommandBufferDesc& desc,
                                                              Result* IGL_NULLABLE outResult) const;

  /// @brief Returns true if `commandBuffer` is a reusable command buffer of this device whose
  /// current slot holds a recording: running the command buffer replays the recording, and nothing
  /// can be encoded into it. Always false for a command buffer that is not reusable or that belongs
  /// to another device. `commandBuffer` must be a Vulkan command buffer: one of another backend
  /// cannot be detected.
  [[nodiscard]] bool isRecorded(const ICommandBuffer& commandBuffer) const;

  /// @brief Runs the current slot of `reusable`, a reusable command buffer created with
  /// ReusableCommandBufferDesc::executeInCommandBuffer, from `commandBuffer`, which a command queue
  /// of this device created: records into it the commands of the slot, after ending the recording
  /// of the slot if it is being recorded. Call it outside of encoders. The slot runs when
  /// `commandBuffer` is submitted, which also adds the draws of the slot to the draw statistics of
  /// the queue. Until then, `commandBuffer` keeps the recordings of `reusable` alive, even if
  /// `reusable` is destroyed, and no other command buffer can execute the slot. If `commandBuffer`
  /// is destroyed without being submitted, or its submission is discarded because its swapchain
  /// image cannot be presented, the slot does not run and can be executed again. Either way, the
  /// first submission after this call moves the shared slot (see createReusableCommandBuffer()).
  ///
  /// On failure, it records nothing and `reusable` stays in its current slot. It fails with
  /// Result::Code::ArgumentInvalid if `commandBuffer` or `reusable` belongs to another device, or
  /// if `reusable` was not created with ReusableCommandBufferDesc::executeInCommandBuffer, with
  /// Result::Code::InvalidOperation if a command buffer that has not been submitted executes the
  /// slot, and as a run of `reusable` fails. Fails with Result::Code::Unsupported if
  /// supportsReusableCommandBuffers() is false. Both must be Vulkan command buffers: one of another
  /// backend cannot be detected.
  void executeCommands(ICommandBuffer& commandBuffer,
                       ICommandBuffer& reusable,
                       Result* IGL_NULLABLE outResult) const;

  /// Android only for now - Creates the file descriptor for the underlying VkFence
  /// @param handle The handle to the GPU Fence
  /// @return The fd for the Vulkan Fence associated with the handle
#if defined(IGL_PLATFORM_ANDROID) && defined(VK_KHR_external_fence_fd)
  [[nodiscard]] int getFenceFdFromSubmitHandle(SubmitHandle handle) const;
#endif

  /// Clear the cached textures
  void clear() {
    nativeDrawableTextures_.clear();
    nativeDepthTexture_ = nullptr;
  }

 protected:
  [[nodiscard]] bool isType(PlatformDeviceType t) const noexcept override {
    return t == kType;
  }

 private:
  Device& device_;
  std::vector<std::shared_ptr<ITexture>> nativeDrawableTextures_;
  std::shared_ptr<ITexture> nativeDepthTexture_;
};

} // namespace igl::vulkan
