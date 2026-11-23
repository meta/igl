/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <shell/shared/platform/Platform.h>

#include <shell/shared/extension/Extension.h>
#include <shell/shared/extension/ExtensionLoader.h>
#include <shell/shared/imageLoader/ImageLoader.h>
#include <shell/shared/input/InputDispatcher.h>
#include <shell/shared/platform/DisplayContext.h>
#include <shell/shared/platform/PresentationRateController.h>

namespace {

int gArgc = 0;
char** gArgv = nullptr;
#if IGL_PLATFORM_ANDROID
// Android has no main()-supplied argc/argv. The IGL Android shell instead derives
// argv from Intent extras and calls initializeCommandLineArgs() once per backend
// init; mark args as initialized up front so accessing argc()/argv() before that
// call returns zero/nullptr instead of asserting, and allow the JNI bridge to
// re-call initializeCommandLineArgs() across backend tab switches with the same
// Intent.
bool gArgsInitialized = true;
#else
bool gArgsInitialized = false;
#endif

} // namespace

namespace igl::shell {

struct Platform::State {
  ExtensionLoader extensionLoader;
  InputDispatcher inputDispatcher;
  DisplayContext displayContext;
  PresentationRateController presentationRateController;
};

Platform::Platform() noexcept : state_(std::make_unique<State>()) {}

Platform::~Platform() = default;

Extension* Platform::createAndInitializeExtension(const char* name) noexcept {
  return state_->extensionLoader.createAndInitialize(name, *this);
}

InputDispatcher& Platform::getInputDispatcher() noexcept {
  return state_->inputDispatcher;
}

[[nodiscard]] DisplayContext& Platform::getDisplayContext() noexcept {
  return state_->displayContext;
}

[[nodiscard]] PresentationRateController& Platform::getPresentationRateController() noexcept {
  return state_->presentationRateController;
}

std::shared_ptr<ITexture> Platform::loadTexture(const char* filename,
                                                bool calculateMipmapLevels,
                                                TextureFormat format,
                                                TextureDesc::TextureUsageBits usage) {
  auto imageData = getImageLoader().loadImageData(filename);

  return loadTexture(imageData, calculateMipmapLevels, format, usage, filename);
}

std::shared_ptr<ITexture> Platform::loadTexture(const ImageData& imageData,
                                                bool calculateMipmapLevels,
                                                TextureFormat format,
                                                TextureDesc::TextureUsageBits usage,
                                                const char* debugName) {
  // @fb-only
  // For D3D12 backend: If
  // mipmaps are requested, we need Attachment usage so that generateMipmap can create render
  // targets for each mip level.
  auto effectiveUsage = usage;
  if (calculateMipmapLevels && getDevice().getBackendType() == BackendType::D3D12) {
    effectiveUsage = static_cast<TextureDesc::TextureUsageBits>(
        static_cast<uint32_t>(usage) |
        static_cast<uint32_t>(igl::TextureDesc::TextureUsageBits::Attachment));
  }

  TextureDesc texDesc =
      igl::TextureDesc::new2D(format, imageData.desc.width, imageData.desc.height, effectiveUsage);
  texDesc.numMipLevels =
      calculateMipmapLevels ? TextureDesc::calcNumMipLevels(texDesc.width, texDesc.height) : 1;
  texDesc.debugName = debugName;

  Result res;
  auto tex = getDevice().createTexture(texDesc, &res);
  IGL_DEBUG_ASSERT(res.isOk(), res.message.c_str());
  IGL_DEBUG_ASSERT(tex != nullptr, "createTexture returned null for some reason");
  tex->upload(tex->getFullRange(),
              imageData.data->data(),
              0,
              !imageData.mipLevelBytes.empty() ? imageData.mipLevelBytes.data() : nullptr);
  return tex;
}

int Platform::argc() {
  IGL_DEBUG_ASSERT(gArgsInitialized, "Accessing command line args before they are initialized.");
  return gArgc;
}

char** Platform::argv() {
  IGL_DEBUG_ASSERT(gArgsInitialized, "Accessing command line args before they are initialized.");
  return gArgv;
}

void Platform::initializeCommandLineArgs(int argc, char** argv) {
#if !IGL_PLATFORM_ANDROID
  IGL_DEBUG_ASSERT(!gArgsInitialized, "Must not initialize command line arguments more than once.");
#endif
  gArgc = argc;
  gArgv = argv;
  gArgsInitialized = true;
}

bool Platform::areCommandLineArgsInitialized() {
  return gArgsInitialized;
}

} // namespace igl::shell
