/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <shell/shared/renderSession/RenderSessionConfig.h>
#include <shell/shared/renderSession/RenderSessionWindowConfig.h>
#include <shell/shared/renderSession/ShellType.h>

namespace igl::vulkan {
class VulkanFeatures;
struct VulkanContextConfig;
} // namespace igl::vulkan

namespace igl::shell {

class Platform;
class RenderSession;
struct ShellParams;

class IRenderSessionFactory {
 public:
  virtual ~IRenderSessionFactory() noexcept = default;

  // Used on desktop platforms to configure the window hosting render sessions
  virtual RenderSessionWindowConfig requestedWindowConfig(
      ShellType /* shellType */,
      RenderSessionWindowConfig suggestedConfig) {
    return suggestedConfig;
  }

  // Used to configure individual render sessions
  virtual std::vector<RenderSessionConfig> requestedSessionConfigs(
      ShellType /* shellType */,
      std::vector<RenderSessionConfig> suggestedConfigs) {
    std::vector<RenderSessionConfig> requestedConfigs;
    requestedConfigs.reserve(suggestedConfigs.size());
    for (auto& suggestedConfig : suggestedConfigs) {
      if (suggestedConfig.backendVersion.flavor != igl::BackendFlavor::Invalid) {
        requestedConfigs.push_back(suggestedConfig);
      }
    }
    return requestedConfigs;
  }

  virtual std::unique_ptr<RenderSession> createRenderSession(
      std::shared_ptr<Platform> platform) noexcept = 0;

  // Used on Android to get the system properties prefix for reading shell params
  [[nodiscard]] virtual const char* getAndroidSystemPropsPrefix() const noexcept {
    return "debug.iglshell.renderSession";
  }

  /// @brief Optional list of additional Vulkan device extensions the session requires.
  /// Names are appended to the IGL Vulkan device-creation call's `extraDeviceExtensions`
  /// parameter on backends that wire this through (currently the Android shell).
  /// Default = empty: existing shell consumers see no behavior change.
  [[nodiscard]] virtual std::vector<std::string> requestedVulkanDeviceExtensions() const noexcept {
    return {};
  }

  /// @brief Adds Vulkan feature requests before the Android shell creates its device.
  /// Use VulkanFeatures::requestFeature(); leave existing fields and requests intact.
  /// Extension names must also be supplied by requestedVulkanDeviceExtensions().
  /// The default leaves the shell's feature selection unchanged.
  virtual void addVulkanFeatureRequests(vulkan::VulkanFeatures& /* features */) const noexcept {}

  /// @brief Gives a session factory a chance to customize Android Vulkan context creation.
  /// The default implementation intentionally leaves the IGL configuration unchanged.
  virtual void configureVulkanContext(const ShellParams& /* shellParams */,
                                      vulkan::VulkanContextConfig& /* config */) const noexcept {}
};

} // namespace igl::shell
