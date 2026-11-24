/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <igl/tests/util/device/webgpu/TestDevice.h>
#include <igl/tests/webgpu/TriangleRender.h>
#include <igl/webgpu/HWDevice.h>
#include <igl/webgpu/PlatformDevice.h>
#include <igl/webgpu/Readback.h>
#include <igl/webgpu/Texture.h>
#include <igl/webgpu/WebGPUContext.h>

namespace igl::tests {

namespace {

using igl::webgpu::ErrorMode;
using igl::webgpu::ErrorScopeKind;

// Passes the WGSL declaration parser but fails WebGPU validation.
constexpr const char* kInvalidWgsl = R"(
@compute @workgroup_size(1)
fn main() {
  let x : f32 = 1u;
}
)";

// Makes one validation error inside an error scope and returns what popErrorScope() reports.
Result makeValidationError(const igl::webgpu::WebGPUContext& ctx) {
  WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
  desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_Uniform;
  desc.size = 16;
  ctx.pushErrorScope(WGPUErrorFilter_Validation);
  const igl::webgpu::Handle<WGPUBuffer> buffer(wgpuDeviceCreateBuffer(ctx.getDevice(), &desc));
  return ctx.popErrorScope();
}

std::shared_ptr<IShaderModule> createInvalidShaderModule(IDevice& device, Result& outResult) {
  return device.createShaderModule(
      ShaderModuleDesc::fromStringInput(
          kInvalidWgsl, {.stage = ShaderStage::Compute, .entryPoint = "main"}, "invalid"),
      &outResult);
}

} // namespace

class WebGPUErrorModeTest : public ::testing::Test {
 public:
  void SetUp() override {
    setDebugBreakEnabled(false);
  }

 protected:
  static std::unique_ptr<igl::webgpu::Device> createDevice(ErrorMode mode) {
    return util::device::webgpu::createTestDevice({.errorMode = mode});
  }
};

TEST_F(WebGPUErrorModeTest, DefaultIsSyncNatively) {
  auto device = createDevice(ErrorMode::Default);
  ASSERT_NE(device, nullptr);
  const auto& ctx = device->getContext();
  EXPECT_EQ(ctx.getErrorMode(), ErrorMode::Sync);
  EXPECT_TRUE(ctx.canWait());
  EXPECT_TRUE(ctx.waitsForErrors(ErrorScopeKind::Resource));
  EXPECT_TRUE(ctx.waitsForErrors(ErrorScopeKind::Pipeline));
  EXPECT_EQ(makeValidationError(ctx).code, Result::Code::ArgumentInvalid);
}

TEST_F(WebGPUErrorModeTest, LatchedErrorsArriveLater) {
  auto device = createDevice(ErrorMode::Latched);
  ASSERT_NE(device, nullptr);
  auto& ctx = device->getContext();
  EXPECT_FALSE(ctx.waitsForErrors(ErrorScopeKind::Pipeline));

  EXPECT_TRUE(makeValidationError(ctx).isOk());
  ctx.processEvents();
  auto* platformDevice =
      static_cast<IDevice&>(*device).getPlatformDevice<igl::webgpu::PlatformDevice>();
  ASSERT_NE(platformDevice, nullptr);
  const std::vector<Result> errors = platformDevice->takeErrors();
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors[0].code, Result::Code::ArgumentInvalid);
  EXPECT_FALSE(errors[0].message.empty());
  EXPECT_TRUE(ctx.takeErrors().empty());
  EXPECT_EQ(ctx.getUncapturedErrorCount(), 0u);
}

TEST_F(WebGPUErrorModeTest, LatchedShaderModuleIsProvisionallyOk) {
  auto device = createDevice(ErrorMode::Latched);
  ASSERT_NE(device, nullptr);
  Result ret;
  auto module = createInvalidShaderModule(*device, ret);
  EXPECT_TRUE(ret.isOk()) << ret.message;
  EXPECT_NE(module, nullptr);
  device->getContext().processEvents();
  EXPECT_EQ(device->getContext().takeErrors().size(), 1u);
}

TEST_F(WebGPUErrorModeTest, SyncPipelinesWaitsOnlyForPipelines) {
  auto device = createDevice(ErrorMode::SyncPipelines);
  ASSERT_NE(device, nullptr);
  auto& ctx = device->getContext();
  EXPECT_FALSE(ctx.waitsForErrors(ErrorScopeKind::Resource));
  EXPECT_TRUE(ctx.waitsForErrors(ErrorScopeKind::Pipeline));

  Result ret;
  auto module = createInvalidShaderModule(*device, ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentInvalid);
  EXPECT_EQ(module, nullptr);

  EXPECT_TRUE(makeValidationError(ctx).isOk());
  ctx.processEvents();
  EXPECT_EQ(ctx.takeErrors().size(), 1u);
}

TEST_F(WebGPUErrorModeTest, DisallowedSuspensionFailsWaitsAndLatchesErrors) {
  auto device = createDevice(ErrorMode::Sync);
  ASSERT_NE(device, nullptr);
  auto& ctx = device->getContext();
  auto* platformDevice =
      static_cast<IDevice&>(*device).getPlatformDevice<igl::webgpu::PlatformDevice>();
  ASSERT_NE(platformDevice, nullptr);

  platformDevice->setSuspensionAllowed(false);
  EXPECT_FALSE(platformDevice->isSuspensionAllowed());
  EXPECT_FALSE(ctx.canWait());
  EXPECT_FALSE(ctx.waitsForErrors(ErrorScopeKind::Pipeline));
  EXPECT_EQ(ctx.checkCanWait().code, Result::Code::InvalidOperation);
  EXPECT_EQ(ctx.waitForSubmittedWork().code, Result::Code::InvalidOperation);

  // Rendering still works; only waiting for it does not.
  webgpu::TriangleRenderer renderer;
  ASSERT_TRUE(renderer.initialize(*device, TextureFormat::RGBA_UNorm8).isOk());
  Result ret;
  const auto target = device->createTexture(
      TextureDesc::new2D(
          TextureFormat::RGBA_UNorm8, 16, 16, TextureDesc::TextureUsageBits::Attachment),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  EXPECT_TRUE(renderer.render(target).isOk());
  EXPECT_TRUE(makeValidationError(ctx).isOk());
  auto module = createInvalidShaderModule(*device, ret);
  EXPECT_TRUE(ret.isOk()) << ret.message;
  ctx.processEvents();
  EXPECT_EQ(ctx.takeErrors().size(), 2u);

  platformDevice->setSuspensionAllowed(true);
  EXPECT_TRUE(ctx.waitForSubmittedWork().isOk());
  EXPECT_EQ(makeValidationError(ctx).code, Result::Code::ArgumentInvalid);
}

TEST_F(WebGPUErrorModeTest, DisallowedSuspensionFailsReadbacks) {
  auto device = createDevice(ErrorMode::Sync);
  ASSERT_NE(device, nullptr);
  auto& ctx = device->getContext();
  Result ret;
  auto texture = device->createTexture(
      TextureDesc::new2D(TextureFormat::RGBA_UNorm8, 4, 4, TextureDesc::TextureUsageBits::Sampled),
      &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  const igl::webgpu::TextureReadbackDesc desc = {
      .texture = static_cast<const igl::webgpu::Texture&>(*texture).getWGPUTexture(),
      .width = 4,
      .height = 4,
      .bytesPerTexel = 4,
  };
  std::vector<uint8_t> bytes(64);

  ctx.setSuspensionAllowed(false);
  EXPECT_EQ(igl::webgpu::readTexture(ctx, desc, bytes.data()).code, Result::Code::InvalidOperation);
  igl::webgpu::AsyncTextureReadback readback;
  ASSERT_TRUE(readback.begin(ctx, desc).isOk());
  EXPECT_EQ(readback.wait().code, Result::Code::InvalidOperation);

  ctx.setSuspensionAllowed(true);
  EXPECT_TRUE(readback.wait().isOk());
  EXPECT_TRUE(readback.copyTo(bytes.data()).isOk());
  EXPECT_TRUE(igl::webgpu::readTexture(ctx, desc, bytes.data()).isOk());
  EXPECT_EQ(ctx.getUncapturedErrorCount(), 0u);
}

TEST_F(WebGPUErrorModeTest, CreateWithWGPUDeviceWrapsAnExternalDevice) {
  Result ret;
  auto owner = igl::webgpu::HWDevice::createContext({}, &ret);
  ASSERT_NE(owner, nullptr) << ret.message;
  ASSERT_TRUE(owner->initDevice().isOk());

  auto device = igl::webgpu::HWDevice::createWithWGPUDevice(
      owner->getInstance(), owner->getDevice(), {}, &ret);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_NE(device, nullptr);
  EXPECT_EQ(device->getContext().getDevice(), owner->getDevice());
  EXPECT_EQ(device->getContext().getErrorMode(), ErrorMode::Sync);
  EXPECT_EQ(device->getContext().getBackendType(), owner->getBackendType());
  EXPECT_EQ(device->getContext().getAdapterType(), owner->getAdapterType());
  EXPECT_EQ(device->getContext().getVendorId(), owner->getVendorId());
  EXPECT_EQ(device->getContext().getAdapterName(), owner->getAdapterName());

  std::vector<uint8_t> rgba;
  ret = tests::webgpu::renderTriangle(*device, 16, rgba);
  ASSERT_TRUE(ret.isOk()) << ret.message;
  ASSERT_EQ(rgba.size(), 16u * 16u * 4u);
  EXPECT_NEAR(rgba[0], 51, 1);
  EXPECT_NEAR(rgba[1], 77, 1);
  EXPECT_NEAR(rgba[2], 102, 1);
  device.reset();
  EXPECT_EQ(owner->getUncapturedErrorCount(), 0u);
}

TEST_F(WebGPUErrorModeTest, CreateWithWGPUDeviceRejectsNull) {
  Result ret;
  auto device = igl::webgpu::HWDevice::createWithWGPUDevice(nullptr, nullptr, {}, &ret);
  EXPECT_EQ(ret.code, Result::Code::ArgumentNull);
  EXPECT_EQ(device, nullptr);
}

} // namespace igl::tests
