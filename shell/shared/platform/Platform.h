/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <igl/Device.h>

namespace igl::shell {

class Extension;
class FileLoader;
class ImageLoader;
struct ImageData;
class ImageWriter;
class InputDispatcher;
class DisplayContext;
class PresentationRateController;

class Platform {
 public:
  Platform() noexcept;
  virtual ~Platform();
  virtual IDevice& getDevice() noexcept = 0;
  [[nodiscard]] virtual std::shared_ptr<IDevice> getDevicePtr() const noexcept = 0;
  virtual ImageLoader& getImageLoader() noexcept = 0;
  [[nodiscard]] virtual const ImageWriter& getImageWriter() const noexcept = 0;
  [[nodiscard]] virtual FileLoader& getFileLoader() const noexcept = 0;

  virtual InputDispatcher& getInputDispatcher() noexcept;
  [[nodiscard]] virtual DisplayContext& getDisplayContext() noexcept;

  // Not virtual: the controller carries the requested/granted bookkeeping for this
  // platform, so handing out a second one would lose it. Platforms customize the seam
  // by installing a backend, not by overriding the accessor.
  [[nodiscard]] PresentationRateController& getPresentationRateController() noexcept;

  std::shared_ptr<ITexture> loadTexture(
      const char* filename,
      bool calculateMipmapLevels = true,
      TextureFormat format = TextureFormat::RGBA_SRGB,
      TextureDesc::TextureUsageBits usage = igl::TextureDesc::TextureUsageBits::Sampled);

  std::shared_ptr<ITexture> loadTexture(
      const ImageData& imageData,
      bool calculateMipmapLevels = true,
      TextureFormat format = TextureFormat::RGBA_SRGB,
      TextureDesc::TextureUsageBits usage = igl::TextureDesc::TextureUsageBits::Sampled,
      const char* debugName = "");
  // 'argc' and 'argv' are the exact arguments received in 'main()'.
  static int argc();
  static char** argv();

  // Don't call this from the application level. The shell framework will use this API to expose
  // command line arguments for the application.
  static void initializeCommandLineArgs(int argc, char** argv);

  // Returns whether initializeCommandLineArgs() has already been called, so independent entry
  // points can avoid double-initializing in a shared process.
  [[nodiscard]] static bool areCommandLineArgsInitialized();

 public:
  Extension* createAndInitializeExtension(const char* name) noexcept;

  /**
   * @brief Create a And Initialize object
   *
   * @return template <typename E>*
   */
  // E is the Extension type. Requirements on E
  // 1. The static method `const char* E::name() noexcept` must exist
  // 2. E must subclass from igl::shell::Extension
  template<typename E>
  E* createAndInitialize() noexcept {
    // TODO static_assert() to enforce subclass
    return static_cast<E*>(createAndInitializeExtension(E::name()));
  }

 private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace igl::shell
