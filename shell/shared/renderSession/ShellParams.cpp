/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <shell/shared/renderSession/ShellParams.h>

#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace igl::shell {

static_assert(sizeof(Fov) == 16);
static_assert(std::is_trivially_copyable_v<Fov>);

namespace {

// Returns true if arg matches any of the given flags.
bool matchArg(const std::string& arg, std::initializer_list<std::string_view> flags) {
  for (auto f : flags) {
    if (arg == f) {
      return true;
    }
  }
  return false;
}

// Tries to consume the next argument as a value. Returns true on success.
bool tryConsumeNext(const std::vector<std::string>& args, size_t& i) {
  if (i + 1 < args.size()) {
    ++i;
    return true;
  }
  return false;
}

std::optional<BenchmarkRenderSessionParams> parseBenchmarkRenderSessionParams(
    const std::vector<std::string>& args) {
  bool found = false;
  BenchmarkRenderSessionParams p;

  for (size_t i = 0; i < args.size(); i++) {
    const std::string& arg = args[i];

    if (matchArg(arg, {"--benchmark", "-b"})) {
      found = true;
    } else if (matchArg(arg, {"--timeout", "-t"}) && tryConsumeNext(args, i)) {
      p.renderSessionTimeoutMs = std::stoul(args[i]);
      found = true;
    } else if (matchArg(arg, {"--sessions", "-s"}) && tryConsumeNext(args, i)) {
      p.numSessionsToRun = std::stoul(args[i]);
      found = true;
    } else if (matchArg(arg, {"--log-reporter", "-l"})) {
      p.logReporter = true;
      found = true;
    } else if (matchArg(arg, {"--offscreen-only", "-o"})) {
      p.offscreenRenderingOnly = true;
      found = true;
    } else if (matchArg(arg, {"--benchmark-duration", "--run-time"}) && tryConsumeNext(args, i)) {
      p.benchmarkDurationMs = std::stoul(args[i]);
      found = true;
    } else if (arg == "--report-interval" && tryConsumeNext(args, i)) {
      p.reportIntervalMs = std::stoul(args[i]);
      found = true;
    } else if (arg == "--hiccup-multiplier" && tryConsumeNext(args, i)) {
      p.hiccupMultiplier = std::stod(args[i]);
      found = true;
    } else if (arg == "--render-buffer-size" && tryConsumeNext(args, i)) {
      p.renderTimeBufferSize = std::stoul(args[i]);
      found = true;
    } else if (arg == "--force-multiview") {
      // handled in parseShellParams; skip here
    } else if (arg.starts_with("--")) {
      std::string key = arg.substr(2);
      std::string value;
      if (i + 1 < args.size() && !args[i + 1].starts_with("--")) {
        value = args[++i];
      }
      p.customParams.emplace_back(key, value);
    }
  }

  return found ? std::optional<BenchmarkRenderSessionParams>(p) : std::nullopt;
}

} // namespace

std::vector<std::string> convertArgvToParams(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(argc);
  for (int i = 0; i < argc; i++) {
    args.emplace_back(argv[i]);
  }
  return args;
}

void parseShellParams(const std::vector<std::string>& args, ShellParams& shellParams) {
  auto parsedBenchmarkParams = parseBenchmarkRenderSessionParams(args);
  if (parsedBenchmarkParams.has_value()) {
    shellParams.benchmarkParams = parsedBenchmarkParams;
    if (parsedBenchmarkParams->offscreenRenderingOnly) {
      shellParams.shouldPresent = false;
    }
  }

  for (size_t i = 0; i < args.size(); i++) {
    const std::string& arg = args[i];

    if (arg == "--headless") {
      shellParams.isHeadless = true;
      shellParams.shouldPresent = false;
      shellParams.screenshotNumber =
          shellParams.screenshotNumber != ~0u ? shellParams.screenshotNumber : 0;
    } else if (arg == "--disable-vulkan-validation-layers") {
      shellParams.enableVulkanValidationLayers = false;
    } else if (arg == "--screenshot-file" && tryConsumeNext(args, i)) {
      shellParams.screenshotFileName = args[i];
    } else if (arg == "--screenshot-number" && tryConsumeNext(args, i)) {
      shellParams.screenshotNumber = static_cast<uint32_t>(std::stoi(args[i]));
    } else if (arg == "--viewport-size" && tryConsumeNext(args, i)) {
      unsigned int w = 0;
      unsigned int h = 0;
      if (std::sscanf(args[i].c_str(), "%ux%u", &w, &h) == 2 && w && h) {
        shellParams.viewportSize = glm::vec2(w, h);
      }
    } else if (arg == "--fps-throttle" && tryConsumeNext(args, i)) {
      shellParams.fpsThrottleMs = static_cast<uint32_t>(std::stoi(args[i]));
    } else if (arg == "--fps-throttle-random") {
      shellParams.fpsThrottleRandom = true;
    } else if (arg == "--freeze-at-frame" && tryConsumeNext(args, i)) {
      shellParams.freezeAtFrame = static_cast<uint32_t>(std::stoi(args[i]));
    } else if (arg == "--use-timer-rendering") {
      shellParams.useTimerRendering = true;
    } else if (arg == "--force-multiview") {
      shellParams.forceMultiview = true;
      shellParams.renderMode = RenderMode::SinglePassStereo;
      shellParams.viewParams.resize(2);
      shellParams.viewParams[0].viewIndex = 0;
      shellParams.viewParams[1].viewIndex = 1;
    }
  }
}

} // namespace igl::shell
