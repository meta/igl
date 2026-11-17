/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/Common.h>

#include "iglu/RecordingDevice.h"

#include <ldrutils/lutils/Pool.h>
#include <igl/CommandEncoder.h>

namespace igl::tests {

TEST(CommonTest, BackendTypeToStringTest) {
  ASSERT_EQ(BackendTypeToString(BackendType::Invalid), "Invalid");
  ASSERT_EQ(BackendTypeToString(BackendType::OpenGL), "OpenGL");
  ASSERT_EQ(BackendTypeToString(BackendType::Metal), "Metal");
  ASSERT_EQ(BackendTypeToString(BackendType::Vulkan), "Vulkan");
  // @fb-only
  ASSERT_EQ(BackendTypeToString(BackendType::D3D12), "D3D12");
  ASSERT_EQ(BackendTypeToString(BackendType::Custom), "Custom");
  ASSERT_EQ(BackendTypeToString(BackendType::WebGPU), "WebGPU");
}

TEST(CommonTest, BackendDebugColorTest) {
  RecordingDevice device;
  const auto colorFor = [&device](BackendType backendType) {
    device.backendType = backendType;
    return device.backendDebugColor();
  };
  const auto expectColor = [](const Color& color, float r, float g, float b, float a) {
    EXPECT_EQ(color.r, r);
    EXPECT_EQ(color.g, g);
    EXPECT_EQ(color.b, b);
    EXPECT_EQ(color.a, a);
  };
  expectColor(colorFor(BackendType::Invalid), 0.f, 0.f, 0.f, 0.f);
  expectColor(colorFor(BackendType::OpenGL), 1.f, 1.f, 0.f, 1.f);
  expectColor(colorFor(BackendType::Metal), 1.f, 0.f, 1.f, 1.f);
  expectColor(colorFor(BackendType::Vulkan), 0.f, 1.f, 1.f, 1.f);
  expectColor(colorFor(BackendType::D3D12), 0.f, 1.f, 1.f, 1.f);
  expectColor(colorFor(BackendType::WebGPU), 0.f, 1.f, 1.f, 1.f);
  // @fb-only
  expectColor(colorFor(BackendType::Custom), 0.f, 0.f, 1.f, 1.f);
}

TEST(CommonTest, ResultTest) {
  Result testResult;
  Result testResult2(Result::Code::Ok, "test message2");
  Result testResult3(Result::Code::Ok, std::string("test message3"));
  ASSERT_STREQ(testResult2.message.c_str(), "test message2");
  ASSERT_TRUE(testResult2.isOk());
  ASSERT_STREQ(testResult3.message.c_str(), "test message3");
  ASSERT_TRUE(testResult3.isOk());

  Result::setResult(&testResult, Result::Code::ArgumentInvalid, std::string("new test message"));
  ASSERT_STREQ(testResult.message.c_str(), "new test message");
  ASSERT_FALSE(testResult.isOk());

  Result::setResult(&testResult3, testResult);
  ASSERT_FALSE(testResult3.isOk());

  Result::setResult(&testResult2, std::move(testResult));
  ASSERT_FALSE(testResult2.isOk());
}

TEST(CommonTest, RectTest) {
  const ScissorRect testRect;
  ASSERT_TRUE(testRect.isNull());
  const ScissorRect testRect2{.x = 0, .y = 0, .width = 1, .height = 1};
  ASSERT_FALSE(testRect2.isNull());
}

TEST(CommonTest, ScissorRectIsNullSemantics) {
  // isNull() is true only when BOTH width and height are zero; a non-zero extent
  // in either dimension makes the rect non-null (guards against an accidental
  // || instead of &&).
  EXPECT_FALSE((ScissorRect{.width = 1, .height = 0}).isNull());
  EXPECT_FALSE((ScissorRect{.width = 0, .height = 1}).isNull());

  // x/y are not part of the emptiness test: a zero-extent rect is null
  // regardless of its origin.
  EXPECT_TRUE((ScissorRect{.x = 10, .y = 20, .width = 0, .height = 0}).isNull());
}

TEST(CommonTest, ScissorRectIsNullIgnoresOrigin) {
  // isNull() depends only on width/height, never on x/y — a zero-sized rect at a
  // non-zero origin is still considered null.
  const ScissorRect rect{.x = 10, .y = 20, .width = 0, .height = 0};
  EXPECT_TRUE(rect.isNull());
}

TEST(CommonTest, ScissorRectIsNullRequiresBothDimensionsZero) {
  // A rect with either dimension non-zero is not null.
  const ScissorRect wideOnly{.x = 0, .y = 0, .width = 5, .height = 0};
  EXPECT_FALSE(wideOnly.isNull());
  const ScissorRect tallOnly{.x = 0, .y = 0, .width = 0, .height = 5};
  EXPECT_FALSE(tallOnly.isNull());
}

TEST(CommonTest, SizeTest) {
  Size size;
  ASSERT_EQ(size.height, 0.0f);
  ASSERT_EQ(size.width, 0.0f);
  Size size2(2, 2);
  ASSERT_EQ(size2.height, 2.0f);
  ASSERT_EQ(size2.width, 2.0f);

  ASSERT_TRUE(size != size2);
  ASSERT_TRUE(size2 == size2);
  ASSERT_FALSE(size == size2);
  ASSERT_FALSE(size2 != size2);
}

TEST(CommonTest, SizeInequalityPerField) {
  // operator== must compare both fields; mutating each one in isolation should
  // make the sizes compare non-equal.
  const Size base(4.0f, 8.0f);
  {
    const Size other(5.0f, 8.0f);
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    const Size other(4.0f, 9.0f);
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
}

TEST(CommonTest, DimensionTest) {
  Dimensions dimension;
  ASSERT_EQ(dimension.height, 0);
  ASSERT_EQ(dimension.width, 0);
  ASSERT_EQ(dimension.depth, 0);
  Dimensions dimension2(2, 2, 2);
  ASSERT_EQ(dimension2.height, 2);
  ASSERT_EQ(dimension2.width, 2);
  ASSERT_EQ(dimension2.depth, 2);

  ASSERT_TRUE(dimension != dimension2);
  ASSERT_TRUE(dimension2 == dimension2);
  ASSERT_FALSE(dimension == dimension2);
  ASSERT_FALSE(dimension2 != dimension2);
}

TEST(CommonTest, DimensionInequalityPerField) {
  // operator== must compare width, height and depth; mutating each one in
  // isolation should make the dimensions compare non-equal.
  const Dimensions base(2, 3, 4);
  {
    Dimensions other = base;
    other.width = 20;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    Dimensions other = base;
    other.height = 30;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    Dimensions other = base;
    other.depth = 40;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
}

TEST(CommonTest, ViewportTest) {
  Viewport viewport;
  ASSERT_TRUE(viewport != kInvalidViewport);
  Viewport viewport2;
  ASSERT_TRUE(viewport == viewport2);
}

TEST(CommonTest, ViewportDefaultValues) {
  // The default viewport covers a unit region: width/height default to 1.0f (not
  // 0.0f) so a freshly constructed Viewport is immediately usable.
  const Viewport viewport;
  EXPECT_FLOAT_EQ(viewport.x, 0.0f);
  EXPECT_FLOAT_EQ(viewport.y, 0.0f);
  EXPECT_FLOAT_EQ(viewport.width, 1.0f);
  EXPECT_FLOAT_EQ(viewport.height, 1.0f);
  EXPECT_FLOAT_EQ(viewport.minDepth, 0.0f);
  EXPECT_FLOAT_EQ(viewport.maxDepth, 1.0f);
}

TEST(CommonTest, InvalidViewportValues) {
  // kInvalidViewport is the sentinel used by `!= kInvalidViewport` checks; every
  // field must be -1.0f so it can never be confused with a valid viewport.
  EXPECT_FLOAT_EQ(kInvalidViewport.x, -1.0f);
  EXPECT_FLOAT_EQ(kInvalidViewport.y, -1.0f);
  EXPECT_FLOAT_EQ(kInvalidViewport.width, -1.0f);
  EXPECT_FLOAT_EQ(kInvalidViewport.height, -1.0f);
  EXPECT_FLOAT_EQ(kInvalidViewport.minDepth, -1.0f);
  EXPECT_FLOAT_EQ(kInvalidViewport.maxDepth, -1.0f);
}

TEST(CommonTest, ViewportInequalityPerField) {
  // operator== must compare every field; mutating each one in isolation should
  // make the viewports compare non-equal.
  const Viewport base;
  {
    Viewport other = base;
    other.x = 5.0f;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    Viewport other = base;
    other.y = 5.0f;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    Viewport other = base;
    other.width = 5.0f;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    Viewport other = base;
    other.height = 5.0f;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    Viewport other = base;
    other.minDepth = 0.5f;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
  {
    Viewport other = base;
    other.maxDepth = 0.5f;
    EXPECT_TRUE(base != other);
    EXPECT_FALSE(base == other);
  }
}

TEST(CommonTest, EnumToValueTest) {
  auto val = EnumToValue(BackendType::Vulkan);
  ASSERT_EQ(val, 3);
}

TEST(CommonTest, ScopeGuardTest) {
  int testValue = 0;
  {
    auto scopeGuard = detail::ScopeGuardOnExit() + [&]() { ++testValue; };
  }
  ASSERT_EQ(testValue, 1);
}

TEST(CommonTest, ScopeExitMacro) {
  int count = 0;
  {
    IGL_SCOPE_EXIT {
      ++count;
    };
    IGL_SCOPE_EXIT {
      ++count;
    };
  }
  ASSERT_EQ(count, 2);
}

TEST(CommonTest, OptimizedMemCopyTest) {
  uint8_t buffer1[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  uint8_t buffer2[32] = {0};

  optimizedMemcpy(buffer1, buffer2, 4);
  ASSERT_EQ(*reinterpret_cast<uint32_t*>(buffer1), *reinterpret_cast<uint32_t*>(buffer2));
  optimizedMemcpy(buffer1, buffer2, 8);
  ASSERT_EQ(*reinterpret_cast<uint64_t*>(buffer1), *reinterpret_cast<uint64_t*>(buffer2));
  optimizedMemcpy(buffer1, buffer2, 16);
  ASSERT_EQ(*reinterpret_cast<uint64_t*>(buffer1), *reinterpret_cast<uint64_t*>(buffer2));
  ASSERT_EQ(*(reinterpret_cast<uint64_t*>(buffer1) + 1),
            *(reinterpret_cast<uint64_t*>(buffer2) + 1));
  optimizedMemcpy(buffer1, buffer2, 32);
  ASSERT_EQ(*reinterpret_cast<uint64_t*>(buffer1), *reinterpret_cast<uint64_t*>(buffer2));
  ASSERT_EQ(*(reinterpret_cast<uint64_t*>(buffer1) + 1),
            *(reinterpret_cast<uint64_t*>(buffer2) + 1));
  ASSERT_EQ(*(reinterpret_cast<uint64_t*>(buffer1) + 2),
            *(reinterpret_cast<uint64_t*>(buffer2) + 2));
  ASSERT_EQ(*(reinterpret_cast<uint64_t*>(buffer1) + 3),
            *(reinterpret_cast<uint64_t*>(buffer2) + 3));
}

TEST(CommonTest, OptimizedMemCopySize12) {
  // Exercises the dedicated 12-byte copy path (case 12 in optimizedMemcpy()), which performs an
  // 8-byte store followed by a 4-byte store. The general OptimizedMemCopyTest covers 4/8/16/32 but
  // not 12.
  alignas(4) const uint8_t src[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  alignas(4) uint8_t dst[12] = {0};

  optimizedMemcpy(dst, src, sizeof(src));

  for (size_t i = 0; i < sizeof(src); ++i) {
    EXPECT_EQ(dst[i], src[i]);
  }
}

TEST(CommonTest, OptimizedMemCopyMisalignedFallsBackToMemcpy) {
  // When dst or src is not 4-byte aligned, optimizedMemcpy() must fall back to the generic memcpy()
  // path instead of issuing aligned word stores. Offsetting by one byte guarantees misalignment of
  // an otherwise 4-byte copy.
  alignas(4) uint8_t srcStorage[8] = {0};
  alignas(4) uint8_t dstStorage[8] = {0};

  uint8_t* const src = srcStorage + 1;
  uint8_t* const dst = dstStorage + 1;
  for (uint8_t i = 0; i < 4; ++i) {
    src[i] = static_cast<uint8_t>(i + 1);
  }

  optimizedMemcpy(dst, src, 4);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(dst[i], src[i]);
  }
}

TEST(CommonTest, ResultSetOk) {
  Result result(Result::Code::ArgumentInvalid, "some error");
  ASSERT_FALSE(result.isOk());
  ASSERT_FALSE(result.message.empty());

  Result::setOk(&result);
  ASSERT_TRUE(result.isOk());
  ASSERT_TRUE(result.message.empty());
}

TEST(CommonTest, ResultSetOkNullSafe) {
  Result::setOk(nullptr);
}

TEST(CommonTest, ResultSetResultNullSafe) {
  Result::setResult(nullptr, Result::Code::RuntimeError, "should not crash");
  const Result src(Result::Code::Unsupported, "src");
  Result::setResult(nullptr, src);
  Result::setResult(nullptr, Result(Result::Code::Unimplemented, "tmp"));
}

TEST(CommonTest, ResultErrorCodes) {
  const Result r1(Result::Code::ArgumentNull, "null arg");
  ASSERT_FALSE(r1.isOk());
  ASSERT_EQ(r1.code, Result::Code::ArgumentNull);

  const Result r2(Result::Code::ArgumentOutOfRange);
  ASSERT_FALSE(r2.isOk());
  ASSERT_EQ(r2.code, Result::Code::ArgumentOutOfRange);

  const Result r3(Result::Code::Unsupported, "not supported");
  ASSERT_EQ(r3.code, Result::Code::Unsupported);

  const Result r4(Result::Code::DeviceLost, "gpu gone");
  ASSERT_EQ(r4.code, Result::Code::DeviceLost);
}

TEST(CommonTest, HandleTest) {
  const Holder<BindGroupTextureHandle> bindGroupHandle;
  const Holder<BindGroupBufferHandle> bindGroupBufferHandle;
  const Holder<TextureHandle> textureHandle;
  const Holder<SamplerHandle> samplerHandle;
}

TEST(CommonTest, PoolTest) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> bindGroupBuffersPool;
}

TEST(HolderTest, DefaultConstructedIsEmptyAndInvalid) {
  const Holder<BindGroupTextureHandle> holder;
  EXPECT_TRUE(holder.empty());
  EXPECT_FALSE(holder.valid());
}

TEST(HolderTest, DefaultConstructedIndexAndGenAreZero) {
  const Holder<TextureHandle> holder;
  EXPECT_EQ(holder.index(), 0u);
  EXPECT_EQ(holder.gen(), 0u);
}

TEST(HolderTest, MoveConstructionTransfersState) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> pool;
  const BindGroupBufferHandle handle = pool.create(BindGroupBufferDesc{});

  Holder<BindGroupBufferHandle> original(nullptr, handle);
  EXPECT_FALSE(original.empty());
  EXPECT_TRUE(original.valid());

  Holder<BindGroupBufferHandle> moved(std::move(original));
  EXPECT_FALSE(moved.empty());
  EXPECT_TRUE(moved.valid());
  EXPECT_TRUE(original.empty()); // NOLINT(bugprone-use-after-move)
}

TEST(HolderTest, MoveAssignmentTransfersState) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> pool;
  const BindGroupBufferHandle handle = pool.create(BindGroupBufferDesc{});

  Holder<BindGroupBufferHandle> a(nullptr, handle);
  Holder<BindGroupBufferHandle> b;
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(b.empty());

  b = std::move(a);
  EXPECT_FALSE(b.empty());
  EXPECT_TRUE(a.empty()); // NOLINT(bugprone-use-after-move)
}

TEST(HolderTest, AssignNullptrResetsHolder) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> pool;
  const BindGroupBufferHandle handle = pool.create(BindGroupBufferDesc{});

  Holder<BindGroupBufferHandle> holder(nullptr, handle);
  EXPECT_FALSE(holder.empty());

  holder = nullptr;
  EXPECT_TRUE(holder.empty());
  EXPECT_FALSE(holder.valid());
}

TEST(HolderTest, ReleaseReturnsHandleAndEmptiesHolder) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> pool;
  const BindGroupBufferHandle handle = pool.create(BindGroupBufferDesc{});

  Holder<BindGroupBufferHandle> holder(nullptr, handle);
  EXPECT_FALSE(holder.empty());

  const BindGroupBufferHandle released = holder.release();
  EXPECT_FALSE(released.empty());
  EXPECT_TRUE(holder.empty());
}

TEST(HolderTest, ResetEmptiesHolder) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> pool;
  const BindGroupBufferHandle handle = pool.create(BindGroupBufferDesc{});

  Holder<BindGroupBufferHandle> holder(nullptr, handle);
  EXPECT_FALSE(holder.empty());

  holder.reset();
  EXPECT_TRUE(holder.empty());
  EXPECT_FALSE(holder.valid());
}

TEST(HolderTest, ConvertsToUnderlyingHandle) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> pool;
  const BindGroupBufferHandle handle = pool.create(BindGroupBufferDesc{});

  const Holder<BindGroupBufferHandle> holder(nullptr, handle);

  // The implicit conversion to the underlying handle type must yield a handle
  // equal to the one the holder wraps (same index and gen), so a Holder can be
  // passed wherever the raw handle is expected.
  const BindGroupBufferHandle converted = holder;
  EXPECT_EQ(converted, handle);
  EXPECT_EQ(holder.index(), handle.index());
  EXPECT_EQ(holder.gen(), handle.gen());
}

TEST(HolderTest, IndexAsVoidMatchesHandle) {
  ldr::Pool<BindGroupBufferTag, BindGroupBufferDesc> pool;
  const BindGroupBufferHandle handle = pool.create(BindGroupBufferDesc{});

  const Holder<BindGroupBufferHandle> holder(nullptr, handle);

  // indexAsVoid() encodes the integer index as an opaque void* (no real
  // pointer dereference); it must match the wrapped handle's own encoding.
  EXPECT_EQ(holder.indexAsVoid(), handle.indexAsVoid());
}

} // namespace igl::tests
