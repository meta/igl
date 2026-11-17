# IGL Android Shell

Sample Activity, JNI bridge, and renderer scaffolding that hosts an
`igl::shell::RenderSession` on Android (Vulkan or OpenGL ES). Used by the
MediaPrism demo and other IGL sample apps.

## Layout

- `java/com/facebook/igl/sample/` — `SampleActivity`, `SampleLib`,
  `SampleView`, `VulkanView`. `SampleActivity` passes its `Intent` to
  `SampleLib.getRenderSessionConfigs()` before creating the surface views.
  JNI initializes `Platform::argc()` / `argv()` from `args` before the
  factory requests session configs. The views pass the Intent to
  `SampleLib.init()` at first surface creation for `ShellParams` extras.
- `jni/` — `Jni.cpp` (Java ↔ native bridge, Intent extra parsing) and
  `TinyRenderer` (per-backend `IDevice` setup + frame loop).

## Passing CLI flags via adb

The shell exposes one Intent extra to native CLI parsers: a single string
named `args` whose value is forwarded to
`igl::shell::Platform::argc()` / `argv()` after whitespace tokenization.
`requestedSessionConfigs()` and RenderSessions consume those tokens through
the same shared flag parsers used on Mac, iOS, and Windows.

Example — launch the MediaPrism demo with the file input source instead of
the default Camera:

```sh
adb shell am start -n com.facebook.igl.sample/com.facebook.igl.shell.SampleActivity \
  --es "args" "--input-source=file"
```

Multiple flags go into the same string, separated by spaces:

```sh
adb shell am start -n com.facebook.igl.sample/com.facebook.igl.shell.SampleActivity \
  --es "args" "--input-source=file --frames=120"
```

Notes:

- `getRenderSessionConfigs(intent)` forwards `args` before any surface is
  created, so `requestedSessionConfigs()` sees the launch flags.
- Other Intent extras (e.g. `debug.iglshell.<session>.*` system properties or
  custom Bundle keys) reach `ShellParams` through `SampleLib.init()` at
  surface creation. `Platform::argv()` contains only `args` tokens.
- When no `args` extra is supplied, `Platform::argc()` returns `0` and
  `Platform::argv()` returns `nullptr`.
- The placeholder `argv[0]` ("`igl_android_shell`") is reserved as a program
  name so `findCliFlag()`-style parsers (which skip index 0) see the real
  tokens at indices ≥ 1.

## Assumptions

- Hosting activities follow `SampleActivity`'s order: request session
  configs before creating the surface views.
