/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <igl/opengl/Timer.h>

#include "../util/Common.h"

#include <igl/CommandBuffer.h>
#include <igl/Framebuffer.h>
#include <igl/RenderCommandEncoder.h>
#include <igl/RenderPass.h>
#include <igl/opengl/Device.h>
#include <igl/opengl/DeviceFeatureSet.h>
#include <igl/opengl/IContext.h>
#include <igl/opengl/TimestampQueries.h>

namespace igl::tests {

namespace {
class StubTimestampQueries final : public opengl::TimestampQueries {
 public:
  using TimestampQueries::QueryFunctions;
  StubTimestampQueries(opengl::IContext& context,
                       uint32_t capacity,
                       QueryFunctions queryFunctions) :
    TimestampQueries(context, capacity, queryFunctions) {}
};

// Exercise encoder retention on drivers without GL_TIME_ELAPSED support.
class NoElapsedTimestampQueries final : public opengl::TimestampQueries {
 public:
  using TimestampQueries::TimestampQueries;
  void beginElapsedQuery(uint32_t /*slotIndex*/) override {
    ++beginCalls;
  }
  void endElapsedQuery() override {}

  uint32_t beginCalls = 0;
};
} // namespace

//
// TimerOGLTest
//
// Tests for the OpenGL Timer.
//
class TimerOGLTest : public ::testing::Test {
 public:
  TimerOGLTest() = default;

  void SetUp() override {
    igl::setDebugBreakEnabled(false);
    util::createDeviceAndQueue(iglDev_, cmdQueue_);
    ASSERT_NE(iglDev_, nullptr);
    ASSERT_NE(cmdQueue_, nullptr);

    context_ = &static_cast<opengl::Device&>(*iglDev_).getContext();
    availabilityReadCount = 0;
    resultReadCount = 0;
    available = 1;
  }

  void TearDown() override {}

 protected:
  std::unique_ptr<opengl::TimestampQueries> createTimestampQueries(uint32_t capacity) {
    return std::make_unique<StubTimestampQueries>(
        *context_,
        capacity,
        StubTimestampQueries::QueryFunctions{.beginQuery = [](GLenum, GLuint) {},
                                             .getQueryObjectiv =
                                                 [](GLuint, GLenum, GLint* value) {
                                                   ++availabilityReadCount;
                                                   *value = available;
                                                 },
                                             .getQueryObjectui64v =
                                                 [](GLuint, GLenum, GLuint64* value) {
                                                   ++resultReadCount;
                                                   *value = 123;
                                                 }});
  }

  void createFramebuffer(std::shared_ptr<IFramebuffer>& framebuffer) {
    Result result;
    const auto textureDesc = TextureDesc::new2D(TextureFormat::RGBA_UNorm8,
                                                4,
                                                4,
                                                TextureDesc::TextureUsageBits::Attachment |
                                                    TextureDesc::TextureUsageBits::Sampled);
    const auto texture = iglDev_->createTexture(textureDesc, &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(texture, nullptr);
    const FramebufferDesc desc = {.colorAttachments = {{.texture = texture}}};
    framebuffer = iglDev_->createFramebuffer(desc, &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(framebuffer, nullptr);
  }

  void encodeTimedPass(const std::shared_ptr<ITimestampQueries>& queries,
                       const std::shared_ptr<IFramebuffer>& framebuffer,
                       uint32_t slotIndex = 0) {
    Result result;
    auto commandBuffer = cmdQueue_->createCommandBuffer({}, &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(commandBuffer, nullptr);
    const RenderPassDesc renderPass = {
        .colorAttachments = {{.loadAction = LoadAction::Clear, .storeAction = StoreAction::Store}},
        .timestampQuery = {.queries = queries, .slotIndex = slotIndex}};
    auto encoder = commandBuffer->createRenderCommandEncoder(renderPass, framebuffer, {}, &result);
    ASSERT_TRUE(result.isOk()) << result.message;
    ASSERT_NE(encoder, nullptr);
    encoder->endEncoding();
    cmdQueue_->submit(*commandBuffer);
  }

  inline static uint32_t availabilityReadCount = 0;
  inline static uint32_t resultReadCount = 0;
  inline static GLint available = 1;
  std::shared_ptr<IDevice> iglDev_;
  std::shared_ptr<ICommandQueue> cmdQueue_;
  opengl::IContext* context_ = nullptr;
};

TEST_F(TimerOGLTest, DisjointReadDoesNotGenerateGlError) {
  Result result;
  std::shared_ptr<ITimestampQueries> queries;
  if (iglDev_->hasFeature(DeviceFeatures::TimestampQueries)) {
    queries = iglDev_->createTimestampQueries(1, &result);
    ASSERT_TRUE(result.isOk()) << result.message;
  } else {
    queries = createTimestampQueries(0);
  }
  ASSERT_NE(queries, nullptr);
  ASSERT_TRUE(queries->isValid());
  ASSERT_EQ(context_->getError(), GL_NO_ERROR);

  context_->enableAutomaticErrorCheck(false);
  const bool disjoint = queries->readAndClearDisjoint();
  const GLenum error = context_->getError();
  context_->enableAutomaticErrorCheck(true);

  EXPECT_EQ(error, GL_NO_ERROR);
  if (!opengl::DeviceFeatureSet::usesOpenGLES() ||
      !context_->deviceFeatures().hasExtension(opengl::Extensions::TimerQuery)) {
    EXPECT_FALSE(disjoint);
  }
  EXPECT_TRUE(queries->isValid());
}

TEST_F(TimerOGLTest, NullQueryFunctionsAreInvalid) {
  for (const auto& queryFunctions :
       {StubTimestampQueries::QueryFunctions{.beginQuery = nullptr},
        StubTimestampQueries::QueryFunctions{.getQueryObjectiv = nullptr},
        StubTimestampQueries::QueryFunctions{.getQueryObjectui64v = nullptr}}) {
    const StubTimestampQueries queries(*context_, 1, queryFunctions);
    EXPECT_FALSE(queries.isValid());
    EXPECT_EQ(queries.capacity(), 1u);
    EXPECT_EQ(queries.count(), 0u);
    EXPECT_FALSE(queries.wasBegun(0));
    EXPECT_FALSE(queries.resultsAvailable());
    EXPECT_FALSE(queries.getElapsedNanosResult(0).valid);
  }
}

TEST_F(TimerOGLTest, UnbegunGapDoesNotReadQuery) {
  auto queries = createTimestampQueries(2);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  queries->beginElapsedQuery(1);
  EXPECT_EQ(queries->count(), 2u);
  EXPECT_FALSE(queries->wasBegun(0));
  EXPECT_TRUE(queries->wasBegun(1));
  EXPECT_FALSE(queries->wasBegun(queries->capacity()));

  const auto result = queries->getElapsedNanosResult(0);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.elapsedNanos, 0u);
  EXPECT_EQ(queries->getElapsedNanos(0), 0u);
  EXPECT_EQ(availabilityReadCount, 0u);
  EXPECT_EQ(resultReadCount, 0u);
}

TEST_F(TimerOGLTest, ResetClearsBegunSlots) {
  auto queries = createTimestampQueries(3);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  for (uint32_t slot = 0; slot < 3; ++slot) {
    queries->beginElapsedQuery(slot);
  }
  queries->reset();
  EXPECT_EQ(queries->count(), 0u);
  EXPECT_FALSE(queries->wasBegun(2));
  queries->beginElapsedQuery(2);
  EXPECT_EQ(queries->count(), 3u);

  for (uint32_t slot = 0; slot < 2; ++slot) {
    EXPECT_FALSE(queries->wasBegun(slot));
    EXPECT_FALSE(queries->getElapsedNanosResult(slot).valid);
    EXPECT_EQ(queries->getElapsedNanos(slot), 0u);
  }
  EXPECT_EQ(availabilityReadCount, 0u);
  EXPECT_EQ(resultReadCount, 0u);
}

TEST_F(TimerOGLTest, BegunSlotReadsQuery) {
  auto queries = createTimestampQueries(1);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  queries->beginElapsedQuery(0);

  const auto result = queries->getElapsedNanosResult(0);
  EXPECT_TRUE(result.valid);
  EXPECT_EQ(result.elapsedNanos, 123u);
  EXPECT_EQ(queries->getElapsedNanos(0), 123u);
  EXPECT_TRUE(queries->resultsAvailable());
  EXPECT_EQ(availabilityReadCount, 3u);
  EXPECT_EQ(resultReadCount, 2u);
}

TEST_F(TimerOGLTest, BegunUnavailableSlotDoesNotReadResult) {
  auto queries = createTimestampQueries(1);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  queries->beginElapsedQuery(0);
  available = 0;

  EXPECT_FALSE(queries->getElapsedNanosResult(0).valid);
  EXPECT_FALSE(queries->resultsAvailable());
  EXPECT_EQ(availabilityReadCount, 2u);
  EXPECT_EQ(resultReadCount, 0u);
}

TEST_F(TimerOGLTest, NoBegunSlotsHaveNoAvailableResults) {
  auto queries = createTimestampQueries(1);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }

  EXPECT_FALSE(queries->resultsAvailable());
  EXPECT_FALSE(queries->getElapsedNanosResult(0).valid);
  EXPECT_EQ(queries->getElapsedNanos(0), 0u);
  EXPECT_EQ(availabilityReadCount, 0u);
  EXPECT_EQ(resultReadCount, 0u);
}

//
// BasicTimerQuery
//
// Create a timer, end it, and check that resultsAvailable can be queried.
//
TEST_F(TimerOGLTest, BasicTimerQuery) {
  if (!iglDev_->hasFeature(DeviceFeatures::Timers)) {
    GTEST_SKIP() << "Timer queries not supported";
  }

  Result ret;
  auto timer = iglDev_->createTimer(&ret);
  ASSERT_TRUE(ret.isOk()) << ret.message.c_str();
  ASSERT_NE(timer, nullptr);

  // Cast to opengl::Timer and test the end method
  auto* oglTimer = static_cast<opengl::Timer*>(timer.get());
  ASSERT_NE(oglTimer, nullptr);

  // End the timer query
  oglTimer->end();

  // Check for GL errors
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);

  // resultsAvailable() should be callable without crashing
  // Results may or may not be available immediately
  const bool available = oglTimer->resultsAvailable();
  // We don't assert on the value, just that calling it doesn't crash or error
  (void)available;

  // If results are available, getElapsedTimeNanos should return a value
  if (available) {
    const uint64_t elapsed = oglTimer->getElapsedTimeNanos();
    // Elapsed time should be non-negative (it's uint64_t so always >= 0)
    (void)elapsed;
  }
}

TEST_F(TimerOGLTest, FramebufferSurvivesCallerReleaseUntilReset) {
  auto queries = std::make_shared<NoElapsedTimestampQueries>(*context_, 1);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  EXPECT_TRUE(queries->requiresFramebufferRetention());
  std::shared_ptr<IFramebuffer> framebuffer;
  ASSERT_NO_FATAL_FAILURE(createFramebuffer(framebuffer));
  const std::weak_ptr<IFramebuffer> weakFramebuffer = framebuffer;
  const std::weak_ptr<ITexture> weakTexture = framebuffer->getColorAttachment(0);

  ASSERT_NO_FATAL_FAILURE(encodeTimedPass(queries, framebuffer));
  EXPECT_EQ(queries->beginCalls, 1u);
  framebuffer.reset();
  EXPECT_FALSE(weakFramebuffer.expired());
  EXPECT_FALSE(weakTexture.expired());

  queries->reset();
  EXPECT_TRUE(weakFramebuffer.expired());
  EXPECT_TRUE(weakTexture.expired());
}

TEST_F(TimerOGLTest, AttachmentSnapshotSurvivesDrawableReplacement) {
  auto queries = std::make_shared<NoElapsedTimestampQueries>(*context_, 1);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  std::shared_ptr<IFramebuffer> framebuffer;
  std::shared_ptr<IFramebuffer> replacement;
  ASSERT_NO_FATAL_FAILURE(createFramebuffer(framebuffer));
  ASSERT_NO_FATAL_FAILURE(createFramebuffer(replacement));
  const std::weak_ptr<ITexture> originalTexture = framebuffer->getColorAttachment(0);

  ASSERT_NO_FATAL_FAILURE(encodeTimedPass(queries, framebuffer));
  framebuffer->updateDrawable(replacement->getColorAttachment(0));
  EXPECT_FALSE(originalTexture.expired());

  queries->reset();
  EXPECT_TRUE(originalTexture.expired());
}

TEST_F(TimerOGLTest, SlotOverwriteAndDestructionReleaseFramebuffers) {
  auto queries = std::make_shared<NoElapsedTimestampQueries>(*context_, 1);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  std::shared_ptr<IFramebuffer> first;
  std::shared_ptr<IFramebuffer> second;
  ASSERT_NO_FATAL_FAILURE(createFramebuffer(first));
  ASSERT_NO_FATAL_FAILURE(createFramebuffer(second));
  const std::weak_ptr<IFramebuffer> weakFirst = first;
  const std::weak_ptr<IFramebuffer> weakSecond = second;

  ASSERT_NO_FATAL_FAILURE(encodeTimedPass(queries, first));
  first.reset();
  EXPECT_FALSE(weakFirst.expired());
  ASSERT_NO_FATAL_FAILURE(encodeTimedPass(queries, second));
  EXPECT_TRUE(weakFirst.expired());
  second.reset();
  EXPECT_FALSE(weakSecond.expired());

  queries.reset();
  EXPECT_TRUE(weakSecond.expired());
}

TEST_F(TimerOGLTest, OutOfBoundsSlotDoesNotRetainFramebuffer) {
  auto queries = std::make_shared<NoElapsedTimestampQueries>(*context_, 1);
  if (!queries->isValid()) {
    GTEST_SKIP() << "Timer query objects are unavailable";
  }
  std::shared_ptr<IFramebuffer> framebuffer;
  ASSERT_NO_FATAL_FAILURE(createFramebuffer(framebuffer));
  const std::weak_ptr<IFramebuffer> weakFramebuffer = framebuffer;

  ASSERT_NO_FATAL_FAILURE(encodeTimedPass(queries, framebuffer, queries->capacity()));
  framebuffer.reset();
  EXPECT_EQ(queries->beginCalls, 0u);
  EXPECT_TRUE(weakFramebuffer.expired());
}

TEST_F(TimerOGLTest, RealElapsedQueryRetainsFramebufferUntilReset) {
  if (!iglDev_->hasFeature(DeviceFeatures::TimestampQueries)) {
    GTEST_SKIP() << "Elapsed timestamp queries not supported; lifetime coverage uses virtual hooks";
  }
  Result result;
  auto queries = iglDev_->createTimestampQueries(1, &result);
  if (!queries || !queries->isValid()) {
    GTEST_SKIP() << "Elapsed timestamp query objects are unavailable: " << result.message;
  }
  ASSERT_TRUE(result.isOk()) << result.message;
  std::shared_ptr<IFramebuffer> framebuffer;
  ASSERT_NO_FATAL_FAILURE(createFramebuffer(framebuffer));
  const std::weak_ptr<IFramebuffer> weakFramebuffer = framebuffer;

  ASSERT_NO_FATAL_FAILURE(encodeTimedPass(queries, framebuffer));
  ASSERT_EQ(context_->checkForErrors(__FILE__, __LINE__), GL_NO_ERROR);
  framebuffer.reset();
  EXPECT_FALSE(weakFramebuffer.expired());
  queries->reset();
  EXPECT_TRUE(weakFramebuffer.expired());
}

} // namespace igl::tests
