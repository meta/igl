# IGL WebGPU backend

`igl::webgpu` implements IGL on the C `webgpu.h` API: natively on Dawn (Metal on macOS, Vulkan on
Linux, the Null adapter in tests) and in the browser through Emscripten's `emdawnwebgpu` port
(JSPI or fully asynchronous builds). Shaders are WGSL.

## Status

| Area | Status | Notes |
|---|---|---|
| Device, buffers, textures (2D/array/cube/3D, sRGB views), samplers, framebuffers | Done | Formats follow the WebGPU spec plus optional features (float32-filterable/-blendable, depth32float-stencil8, BGRA storage, RG11B10 renderable, BC/ETC2/ASTC, unorm16) |
| Render pipelines, bind convention v1, `bindBytes()`, `bindPushConstants()` (group 3), bind groups | Done | Texture unit N at group 0 binding 2N, its sampler at 2N+1; buffer index N at group 1 binding N |
| Compute, storage buffers and textures, `dispatchThreadGroupsIndirect()` | Done | Storage texture N at group 2 binding N, for N up to 15; a stage uses at most `maxStorageTexturesPerShaderStage` of them (4 with the default limits) |
| `multiDrawIndirect()`, `multiDrawIndexedIndirect()` | Done | One `draw{Indexed}Indirect` per record (no core multi-draw); `DrawIndexedIndirect` is reported |
| Timers, timestamp queries | Done | When the adapter has `timestamp-query` |
| Mipmap generation, MSAA resolve, depth upload | Done | Renderable formats only |
| Error modes (Sync / SyncPipelines / Latched), error scopes on every `IDevice` create call, device loss | Done | After loss every GPU create call returns `Result::Code::DeviceLost` |
| IGLU texture accessor, texture scaler (WGSL path) | Done | |
| Shell sessions (offscreen) | Done | Every session renders (see "Shell sessions" below); static ones match Metal exactly apart from the known differences |
| macOS windowed shell (`CAMetalLayer` surface) | Done | Every `shell/apps` app (`shell_app(enable_webgpu = True)`) has a WebGPU tab; `--tab WebGPU` opens on it. Dawn and Tint add about 8 MB to a stripped opt app binary (HelloWorldSession: 16.2 MB without, 24.3 MB with) |
| iOS shell (`CAMetalLayer` surface, Dawn on Metal) | Done | Same apps, `<Session>Apple`; verified in the iOS simulator. The `iphoneos-arm64` binary builds (`:<Session>AppleBinary`); the signed app and device runs are unverified |
| Browser: JSPI and async wasm, canvas surface, imported page device | Done | Canvas checked in WebKit (`check_webkit.sh`); Chrome is checked manually |
| Dawn Null adapter test lane | Done (macOS) | Tests that read GPU results are denylisted |
| Linux (Dawn on lavapipe) unit, screenshot and Null lanes | Added, fixes unverified | Diff CI (asan-ubsan) crashed every WebGPU-specific test that calls `wgpu*()` directly: fbcode ignored `link_style`, so the shared-library link gave the test binary its own, never-filled `dawn_proc` table. The tests now link statically on fbcode, the screenshot lane uses the default fbcode platform (the sessions include OpenGL headers), and six Null tests that read GPU results are denylisted. Repro: `tests/webgpu/tools/linux_repro.sh` |
| CMake (`IGL_WITH_WEBGPU`), GLFW shell app, GitHub jobs | Added | CMake verified on macOS; Linux and Emscripten jobs need CI |

## Known gaps

- Not in WebGPU: multiview, 8-bit indices, 3-component and packed 16-bit formats, `BGR10_A2`,
  PVRTC, L8/LA8/A8, clamp-to-border, line widths, depth bias on lines and points.
- `Z_UNorm24` (depth24plus) cannot be copied or read back; use `Z_UNorm32`.
- Block-compressed textures whose size is not a whole number of blocks (e.g. 5x5 BC1, or 1024x1024
  ASTC 6x6) are allocated rounded up, so normalized coordinates span the padded size and such
  textures sample differently from Metal and Vulkan. Use whole-block sizes.
- WGSL only: SparkSL and GLSL sources need a WGSL counterpart (generated headers for the shell).
- No mesh shaders (MeshShaderTriangleSession draws its triangle from a vertex shader) and no
  line polygon fill (WireframeSession draws a line list of the triangle edges).
- Worker + OffscreenCanvas presentation is unverified; windowed GLFW surfaces (X11, Win32) are
  unverified; there is no Android surface.

## Testing guide

Buck commands run from the named package directory; the big macOS test bundles are run
with `xcrun xctest` because `buck2 test` on them does not finish locally.

| Feature | Status | Command | Expected |
|---|---|---|---|
| Backend unit tests (Dawn on Metal) | Verified | `src/igl`: `buck2 build :iglWebGPUTestsAppleMac --show-full-output`, then `xcrun xctest -XCTest All <bundle>` | 1490 tests, 0 failures |
| Device-free tables | Verified | `src/igl`: `buck2 test :iglWebGPUCommonTestsAppleMac` | Pass 37 |
| Host tests (no GPU; what the fbcode coverage lane runs) | Verified | `src/igl`: `buck2 test :iglHostTestsAppleMac` | Pass 45 |
| Null adapter lane | Verified | `src/igl`: `buck2 build :iglWebGPUTestsNullAppleMac --show-full-output`, then `xcrun xctest -XCTest All <bundle>` | 1491 tests, 0 failures (readback tests skipped) |
| Indirect and multi-draw indirect | Verified | the unit tests above, `WebGPURenderCommandEncoderTest.MultiDraw*` | pass |
| Device loss | Verified | the unit tests above, `WebGPUDeviceTest.CreateCallsFailAfterDeviceLoss` | pass |
| Shell sessions offscreen | Verified | `shell/renderSessionTests`: `buck2 build :offscreenWebGPUTestAppleMac --show-full-output`, then `xcrun xctest -XCTest All <bundle>` | 61 tests, 0 failures (also `offscreenMetalTestAppleMac`, `offscreenOpenGLTestAppleMac`: 61 each, sessions without a path on those backends skipped) |
| Rendering vs Metal/OpenGL | Verified | `src/igl/tests/webgpu/tools/render_compare.sh [out dir]` | maxAbs 0 vs Metal (including Textured3DCube, which also matches OpenGL exactly) except CheckerboardMipmap (Metal's mips are white; within 4 of OpenGL), Imgui (1), Uniform{Array,Packed} (Metal renders nothing; 0 vs OpenGL). Against OpenGL, GraphSample, MRT, MSAA, TQMultiRenderPass, TextureAccessor and TextureRotation differ by the same amounts as Metal vs OpenGL: the differences are OpenGL's. Animated and random sessions (BindGroup, Fluid) and SrgbMipmapGamma differ for the reasons in the Shell sessions table below |
| TextureAccessorSession | Verified | render_compare.sh | maxAbs 0 vs Metal |
| Screenshot tests (macOS) | Verified | `screenshot_tests`: `buck2 test :screenshot_testsWebGPUAppleMac` | Pass 20 |
| macOS window | Frames verified (FPS log), pixels unverified | `shell/apps`: `buck2 run :<Session>AppleMac -- --tab WebGPU` (add `--benchmark` for an FPS log) | Window opens on the WebGPU tab; all 46 apps run at display rate with no uncaptured WebGPU errors |
| iOS simulator | Verified (screenshots) | `shell/apps`: `buck2 build :<Session>Apple --target-platforms ovr_config//platform/iphoneos:iphonesimulator-arm64-local -c cxx.default_platform=iphonesimulator-arm64 --show-full-output`, `xcrun simctl install booted <app>`, `xcrun simctl launch booted com.facebook.<Session> --tab WebGPU` | WebGPU tab renders (44 of 46 apps; TinyMesh and TinyMeshBindGroup fail on every tab: they load `marble.png` from the source tree) |
| Node wasm lane | Verified (advisory) | `src/igl`: `buck2 test :iglWebGPUNodeTest -- --env WEBGPU_NODE_MODULE=<dir>/node_modules/webgpu/index.js --env NODE=/path/to/node` (Node >= 24) | Pass 1 (22 checks); without `WEBGPU_NODE_MODULE` it prints SKIP and passes |
| WebKit canvas (optional) | Verified (macOS 26) | `src/igl/tests/webgpu/wasm/check_webkit.sh [out dir]` (async flavor in a WKWebView; WebKit has no JSPI) | ALL PASSED, 8 checks: async readback and the 60th canvas frame maxAbs 0 vs the native frame |
| Chrome canvas | Unverified | `src/igl/tests/webgpu/wasm/check_chrome.sh` from a normal Terminal on a Mac | Page title reports ALL PASSED: JSPI frame matches the native frame, async readback, 60 canvas frames |
| Linux unit tests (lavapipe) | Unverified | `src/igl`: `buck2 test :iglWebGPUTestsFbcode` on a Linux host | all pass; failures point at lavapipe capability differences |
| Linux Null lane | Unverified | `src/igl`: `buck2 test :iglWebGPUTestsNullFbcode` | all pass |
| Linux screenshot tests | Unverified | `screenshot_tests`: `buck2 test :screenshot_testsWebGPUFbcode` | Pass 20 |
| CMake, installed Dawn | Verified (macOS) | `cmake -S . -B build -DIGL_WITH_WEBGPU=ON -DIGL_WEBGPU_DAWN_DIR=<dawn install> -DIGL_WITH_OPENGL=OFF -DIGL_WITH_VULKAN=OFF -DIGL_WITH_METAL=OFF -DIGL_WITH_TESTS=ON -DIGL_WITH_SHELL=OFF -DIGL_WITH_SAMPLES=OFF`, build `IGLTests`, run it | 1266 pass, 9 skipped, 1 unrelated failure (`TextureLoaderFactoryTest.loadKtx2`, also fails without WebGPU when Vulkan is off) |
| CMake, fetched Dawn | Verified (macOS, local source) | the same without `IGL_WEBGPU_DAWN_DIR` (FetchContent at the pinned revision) | `IGLWebGPU` builds |
| GLFW shell app | Unverified | Linux or Windows only (the macOS CMake shell has no WebGPU backend): `cmake ... -DIGL_WITH_WEBGPU=ON -DIGL_WITH_SHELL=ON -DIGL_WITH_SAMPLES=ON`, then `./build/shell/HelloWorldSession_webgpu --headless --screenshot-file out.png` | PNG with the triangle |
| GitHub jobs | Unverified | `.github/workflows/c-cpp.yml`: `cmake-webgpu-ubuntu`, `cmake-webgpu-emscripten` | green |

Building Dawn with CMake 4.x on Apple silicon fails in abseil's `randen_hwaes` (`-msse4.1` for
arm64) at the pinned revision; drop the x86 flags from `ABSL_RANDOM_HWAES_X64_FLAGS` in Dawn's
abseil or use a Dawn built elsewhere.

## Shell sessions

`offscreenWebGPUTestAppleMac` renders every session (`RenderSnapshotTests`, or `IGLSampleTests` for
the sessions without a frame to compare); `render_compare.sh` diffs the frames against Metal and
OpenGL. Time- or random-animated sessions (no frame-time injection in the shell) are checked for
running N frames without WebGPU errors rather than for pixels.

| Session | WebGPU offscreen | macOS app WebGPU tab | iOS sim WebGPU tab | vs Metal (maxAbs / differing) |
|---|---|---|---|---|
| BasicFramebufferSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| BindGroupSession | yes (snapshot) | yes | yes | 233 / 12.34% (time-animated (wall clock)) |
| BindlessBufferSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| BufferMappingSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| CheckerboardMipmapSession | yes (snapshot) | yes | yes | 255 / 53.89% (known: Metal mips white) |
| ClothSimulationSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| ColorSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| ComputeSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| CopyOperationsSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| DepthBiasSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| DrawIndirectSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| DrawInstancedSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| EmptySession | yes (IGLSampleTests) | yes | yes | - |
| FireworksSession | yes (snapshot) | yes | yes | 0 / 0.00% (time/random-animated; 30 frames) |
| FluidSimulationSession | yes (snapshot) | yes | yes | 255 / 10.11% (random seed + atomic order) |
| GPUStressSession | yes (snapshot) | yes | yes | no Metal ref (Metal/OpenGL have no path) |
| GPUTimerSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| GraphSampleSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| HandsOpenXRSession | n/a | n/a | n/a | n/a (OpenXR only) |
| HelloOpenXRSession | n/a | n/a | n/a | n/a (OpenXR only) |
| HelloSparkSLSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| HelloWorldSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| ImguiSession | yes (snapshot) | yes | yes | 1 / 0.00% (known (1 px)) |
| MRTSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| MSAASession | yes (snapshot) | yes | yes | 0 / 0.00% |
| MeshShaderTriangleSession | yes (snapshot) | yes | yes | 0 / 0.00% (WebGPU uses a vertex-shader fallback) |
| MultiDrawIndexedIndirectSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| ResourceTrackerSession | yes (IGLSampleTests) | yes | yes | - |
| ScissorTestSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| ShaderCompilationTestSession | yes (IGLSampleTests) | yes | yes | - |
| ShaderLibraryWithDataSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| SrgbMipmapGammaSession | yes (snapshot) | yes | yes | 64 / 10.64% (WebGPU has no sRGB storage: 3 quads vs 4) |
| StencilOutlineSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| SubpassDeferredSession | yes (snapshot) | not run | not run | not measured; matches the shared screenshot golden (WebGPU uses the two-pass fallback) |
| TQMultiRenderPassSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| TQSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| Texture3DSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| TextureAccessorSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| TextureFilteringSession | yes (snapshot) | yes | yes | no Metal ref; vs OpenGL 0 / 0.00% (no Metal shaders; 0 vs OpenGL) |
| TextureRotationSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| TextureViewSession | yes (snapshot) | yes | yes | no Metal ref (Metal has no texture views) |
| Textured3DCubeSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| TinyMeshBindGroupSession | yes (snapshot) | yes | crash (pre-existing) | 255 / 22.41% (Metal shader ignores the mesh; time/random) |
| TinyMeshSession | yes (snapshot) | yes | crash (pre-existing) | 255 / 22.41% (Metal shader ignores the mesh; time/random) |
| UniformArrayTestSession | yes (snapshot) | yes | yes | 255 / 100.00% (known: Metal renders nothing; 0 vs OpenGL) |
| UniformPackedTestSession | yes (snapshot) | yes | yes | 255 / 100.00% (known: Metal renders nothing; 0 vs OpenGL) |
| UniformTestSession | yes (snapshot) | yes | yes | 0 / 0.00% |
| WireframeSession | yes (snapshot) | yes | yes | 0 / 0.00% (WebGPU draws a line list of the edges) |
| YUVColorSession | yes (snapshot) | yes | yes | no Metal ref (Metal has no YUV path) |
