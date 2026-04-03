# `imaging_skia`

Skia backend for the `imaging` command stream.

`SkiaGpuRenderer` is the owned Ganesh-backed GPU renderer. `SkiaGpuTargetRenderer` is the
direct-to-texture GPU variant. `SkiaCpuTargetRenderer` is the caller-targeted CPU backend, and
`SkiaCpuRenderer` is the owned CPU convenience wrapper. `CommonRendererState` remains the reusable
lower-level CPU raster state shared by those CPU entry points.

The Ganesh backend is adapted in part from `anyrender_skia` from the AnyRender project, with the
borrowed backend-initialization code carrying attribution in the copied source files.

## Building

`skia-safe` / `skia-bindings` normally download prebuilt Skia binaries at build time. In offline
or sandboxed environments, set `SKIA_BINARIES_URL` to a local `tar.gz` (downloaded ahead of time):

```sh
SKIA_BINARIES_URL='file:///absolute/path/to/skia-binaries-....tar.gz' cargo build
```
