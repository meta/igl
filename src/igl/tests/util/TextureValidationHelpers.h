/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>

#include <glm/detail/qualifier.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <igl/CommandBuffer.h>
#include <igl/Common.h>
#include <igl/Device.h>

namespace igl::tests::util {
constexpr auto kTestPrecision = 0.0001f;

template<typename ColorType>
inline void testArray(std::vector<ColorType> actualData,
                      const ColorType* expectedData,
                      size_t expectedDataSize,
                      const char* message) {
  const auto kInnerTestPrecision = ColorType(kTestPrecision);
  for (size_t i = 0; i < expectedDataSize; i++) {
    std::stringstream ss;
    ss << message << ": Mismatch at index " << i << ": Expected: " << std::hex << expectedData[i]
       << " Actual: " << std::hex << actualData[i];
    ASSERT_NEAR(expectedData[i], actualData[i], kInnerTestPrecision) << ss.str();
  }
}

template<typename VectorUnit, glm::qualifier Precision>
inline void testArray(std::vector<glm::vec<4, VectorUnit, Precision>> actualData,
                      const glm::vec<4, VectorUnit, Precision>* expectedData,
                      size_t expectedDataSize,
                      const char* message) {
  const auto kInnerTestPrecision = VectorUnit(kTestPrecision);
  for (size_t i = 0; i < expectedDataSize; i++) {
    std::stringstream ss;
    ss << message << ": Mismatch at index " << i << ": Expected: " << std::hex << "("
       << expectedData[i].x << " " << expectedData[i].y << " " << expectedData[i].z << " "
       << expectedData[i].w << ")" << " Actual: " << std::hex << "(" << actualData[i].x << " "
       << actualData[i].y << " " << actualData[i].z << " " << actualData[i].w << ")";
    ASSERT_NEAR(expectedData[i].x, actualData[i].x, kInnerTestPrecision) << ss.str();
    ASSERT_NEAR(expectedData[i].y, actualData[i].y, kInnerTestPrecision) << ss.str();
    ASSERT_NEAR(expectedData[i].z, actualData[i].z, kInnerTestPrecision) << ss.str();
    ASSERT_NEAR(expectedData[i].w, actualData[i].w, kInnerTestPrecision) << ss.str();
  }
}

template<typename VectorUnit, glm::qualifier Precision>
inline void testArray(std::vector<glm::vec<3, VectorUnit, Precision>> actualData,
                      const glm::vec<3, VectorUnit, Precision>* expectedData,
                      size_t expectedDataSize,
                      const char* message) {
  const auto kInnerTestPrecision = VectorUnit(kTestPrecision);
  for (size_t i = 0; i < expectedDataSize; i++) {
    std::stringstream ss;
    ss << message << ": Mismatch at index " << i << ": Expected: " << std::hex << "("
       << expectedData[i].x << " " << expectedData[i].y << " " << expectedData[i].z << ")"
       << " Actual: " << std::hex << "(" << actualData[i].x << " " << actualData[i].y << " "
       << actualData[i].z << ")";
    ASSERT_NEAR(expectedData[i].x, actualData[i].x, kInnerTestPrecision) << ss.str();
    ASSERT_NEAR(expectedData[i].y, actualData[i].y, kInnerTestPrecision) << ss.str();
    ASSERT_NEAR(expectedData[i].z, actualData[i].z, kInnerTestPrecision) << ss.str();
  }
}

template<typename VectorUnit, glm::qualifier Precision>
inline void testArray(std::vector<glm::vec<2, VectorUnit, Precision>> actualData,
                      const glm::vec<2, VectorUnit, Precision>* expectedData,
                      size_t expectedDataSize,
                      const char* message) {
  const auto kInnerTestPrecision = VectorUnit(kTestPrecision);
  for (size_t i = 0; i < expectedDataSize; i++) {
    std::stringstream ss;
    ss << message << ": Mismatch at index " << i << ": Expected: " << std::hex << "("
       << expectedData[i].x << " " << expectedData[i].y << ")" << " Actual: " << std::hex << "("
       << actualData[i].x << " " << actualData[i].y << ")";
    ASSERT_NEAR(expectedData[i].x, actualData[i].x, kInnerTestPrecision) << ss.str();
    ASSERT_NEAR(expectedData[i].y, actualData[i].y, kInnerTestPrecision) << ss.str();
  }
}

/// Reads back a range of texture data
/// @param device The device the texture was created with
/// @param cmdQueue A command queue to submit any read requests on
/// @param texture The texture to validate
/// @param isRenderTarget True if the texture was the target of a render pass; false otherwise
/// @param range The range of data to validate. Must resolve to a single 2D texture region
/// @param expectedData The expected data in the specified range
/// @param message A message to print when validation fails
template<typename ColorType>
inline void validateTextureRange(IDevice& device,
                                 ICommandQueue& cmdQueue,
                                 const std::shared_ptr<ITexture>& texture,
                                 bool isRenderTarget,
                                 const TextureRangeDesc& range,
                                 const ColorType* expectedData,
                                 const char* message) {
  Result ret;
  // Dummy command buffer to wait for completion.
  auto cmdBuf = cmdQueue.createCommandBuffer({}, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(cmdBuf != nullptr);
  cmdQueue.submit(*cmdBuf);
  cmdBuf->waitUntilCompleted();

  ASSERT_EQ(range.numLayers, 1);
  ASSERT_EQ(range.numMipLevels, 1);
  ASSERT_EQ(range.depth, 1);

  const auto expectedDataSize = range.width * range.height;
  std::vector<ColorType> actualData;
  actualData.resize(expectedDataSize);

  const FramebufferDesc framebufferDesc{.colorAttachments = {{.texture = texture}}};
  auto fb = device.createFramebuffer(framebufferDesc, &ret);
  ASSERT_EQ(ret.code, Result::Code::Ok);
  ASSERT_TRUE(fb != nullptr);

  fb->copyBytesColorAttachment(cmdQueue, 0, actualData.data(), range);

  if (!isRenderTarget && (device.getBackendType() == BackendType::Metal ||
                          device.getBackendType() == BackendType::Vulkan ||
                          device.getBackendType() == BackendType::D3D12 ||
                          device.getBackendType() == BackendType::WebGPU)) {
    // The Vulkan, Metal, D3D12 and WebGPU implementations of copyBytesColorAttachment flip the
    // returned image vertically. This is the desired behavior for render targets, but for
    // non-render target textures, we want the unflipped data. This flips the output image again to
    // get the unmodified data.
    std::vector<ColorType> tmpData;
    tmpData.resize(actualData.size());
    for (size_t h = 0; h < range.height; ++h) {
      size_t src = (range.height - 1 - h) * range.width;
      size_t dst = h * range.width;
      for (size_t w = 0; w < range.width; ++w) {
        tmpData[dst++] = actualData[src++];
      }
    }
    actualData = std::move(tmpData);
  }

  testArray(actualData, expectedData, expectedDataSize, message);
}

template<typename ColorType>
inline void validateFramebufferTextureRange(IDevice& device,
                                            ICommandQueue& cmdQueue,
                                            const IFramebuffer& framebuffer,
                                            const TextureRangeDesc& range,
                                            const ColorType* expectedData,
                                            const char* message) {
  validateTextureRange(
      device, cmdQueue, framebuffer.getColorAttachment(0), true, range, expectedData, message);
}

template<typename ColorType>
inline void validateFramebufferTexture(IDevice& device,
                                       ICommandQueue& cmdQueue,
                                       const IFramebuffer& framebuffer,
                                       const ColorType* expectedData,
                                       const char* message) {
  validateFramebufferTextureRange(device,
                                  cmdQueue,
                                  framebuffer,
                                  framebuffer.getColorAttachment(0)->getFullRange(),
                                  expectedData,
                                  message);
}

template<typename ColorType>
inline void validateUploadedTextureRange(IDevice& device,
                                         ICommandQueue& cmdQueue,
                                         const std::shared_ptr<ITexture>& texture,
                                         const TextureRangeDesc& range,
                                         const ColorType* expectedData,
                                         const char* message) {
  validateTextureRange(device, cmdQueue, texture, false, range, expectedData, message);
}

template<typename ColorType>
inline void validateUploadedTexture(IDevice& device,
                                    ICommandQueue& cmdQueue,
                                    const std::shared_ptr<ITexture>& texture,
                                    const ColorType* expectedData,
                                    const char* message) {
  validateTextureRange(
      device, cmdQueue, texture, false, texture->getFullRange(), expectedData, message);
}

} // namespace igl::tests::util
