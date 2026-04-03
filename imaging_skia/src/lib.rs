// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Skia backend for `imaging`.
//!
//! This crate connects the semantic `imaging` command stream to Skia.
//!
//! At a high level, there are three ways to use it:
//!
//! - [`SkiaGpuRenderer`] renders through Skia Ganesh into a GPU-backed surface.
//! - [`SkiaCpuTargetRenderer`] replays through Skia's raster backend into caller-provided CPU
//!   targets.
//! - [`SkiaCpuRenderer`] is the owned convenience wrapper for CPU raster rendering.
//! - [`SkCanvasSink`] and [`SkPictureRecorderSink`] let you stream `imaging` commands directly into
//!   native Skia targets instead of going through the owned renderers.
//!
//! # What The Crate Supports
//!
//! `imaging_skia` is useful when you want one of the following:
//!
//! - replay an [`imaging::record::Scene`] through Skia and read the result back as RGBA8 pixels
//! - render `imaging` scenes with Skia's GPU path when a Ganesh backend is available
//! - keep rendering CPU-only while still using Skia's paint, path, gradient, text, and filter
//!   implementations
//! - stream `imaging` commands directly into an existing `skia_safe::Canvas`
//! - record `imaging` commands into a native `skia_safe::Picture`
//! - attach Skia rendering to an existing backend-owned texture or image via Metal, Vulkan, or GL
//!   interop methods
//!
//! # CPU vs GPU
//!
//! [`SkiaCpuRenderer`] is the simpler choice when you just need pixels and do not need graphics API
//! interop. It allocates a raster surface internally and returns RGBA8 output after replay.
//!
//! [`CommonRendererState`] is the lower-level CPU replay engine. It keeps reusable raster-side
//! state such as path tolerance while callers provide the destination
//! [`skia_safe::Surface`]. [`SkiaCpuTargetRenderer`] is the corresponding `imaging_backend`
//! wrapper when you want the same CPU replay path behind a named backend type.
//!
//! [`SkiaGpuRenderer`] is the GPU path. It owns a Ganesh context and an offscreen GPU render surface
//! by default, but it can also be pointed at caller-owned backend textures through the backend-
//! specific constructors. Use it when you want GPU rendering, Skia backend texture access, or
//! integration with an existing graphics stack.
//!
//! # Backend Layout
//!
//! The GPU renderer chooses or exposes different backends depending on platform and enabled
//! features:
//!
//! - Apple platforms use Metal for the default GPU path.
//! - Non-Apple platforms use the internal OpenGL backend unless the `vulkan` feature is enabled,
//!   in which case the default GPU path is Vulkan.
//! - With the `wgpu` feature enabled, the renderer can also attach to existing backend resources:
//!   Metal on Apple, Vulkan on non-Apple builds with `vulkan`, and explicit GL/GLES textures via
//!   raw-context interop methods.
//!
//! # Which API To Reach For
//!
//! Use [`SkiaCpuRenderer`] when:
//!
//! - you want the lowest-friction offscreen renderer
//! - you do not need GPU interop
//! - deterministic CPU-side rendering is more important than backend integration
//!
//! Use [`CommonRendererState`] when:
//!
//! - you want to render into a caller-owned raster surface
//! - you want to reuse CPU-side caches across multiple wrapped surfaces
//! - your application already manages pixel storage and surface lifetime
//!
//! Use [`SkiaCpuTargetRenderer`] when:
//!
//! - you want the CPU renderer to implement `imaging_backend::Backend`
//! - you want naming symmetry with [`SkiaCpuRenderer`], [`SkiaGpuRenderer`], and
//!   [`SkiaGpuTargetRenderer`]
//! - you still want access to reusable [`CommonRendererState`] through a thin wrapper
//!
//! Use [`SkiaGpuRenderer::new`] or [`SkiaGpuRenderer::try_new`] when:
//!
//! - you want an owned offscreen GPU renderer
//! - `imaging_skia` should choose and own the underlying Ganesh backend
//! - you want to render scenes or pictures and optionally inspect the GPU surface afterward
//!
//! Use the backend-specific [`SkiaGpuTargetRenderer`] constructors when:
//!
//! - your application already owns the render target
//! - Skia must draw into an existing Metal texture, Vulkan image, or GL texture
//! - you need the renderer to follow an embedding application's backend/resource lifetime
//!
//! Use [`SkCanvasSink`] when:
//!
//! - you already have a Skia canvas
//! - you want to stream `imaging` commands directly into it
//! - you do not need `imaging_skia` to own any renderer state
//!
//! Use [`SkPictureRecorderSink`] when:
//!
//! - you want a retained native Skia recording
//! - downstream code already consumes `skia_safe::Picture`
//! - you want to separate authoring from later rendering
//!
//! # Render A Recorded Scene
//!
//! Record commands into [`imaging::record::Scene`], then hand the scene to [`SkiaGpuRenderer`].
//!
//! ```no_run
//! use imaging::{Painter, record};
//! use imaging_skia::SkiaGpuRenderer;
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_skia::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x2a, 0x6f, 0xdb));
//!     let mut scene = record::Scene::new();
//!
//!     {
//!         let mut painter = Painter::new(&mut scene);
//!         painter.fill_rect(Rect::new(0.0, 0.0, 128.0, 128.0), &paint);
//!     }
//!
//!     let mut renderer = SkiaGpuRenderer::new(128, 128);
//!     renderer.reset()?;
//!     renderer.render_scene(&scene)?;
//!     let image = renderer.read_image()?;
//!     assert_eq!(image.width, 128);
//!     Ok(())
//! }
//! ```
//!
//! # Render Into Wrapped Pixels
//!
//! If you already own the destination pixels, wrap them in a raster surface and use
//! [`CommonRendererState`].
//!
//! ```no_run
//! use imaging::{Painter, record};
//! use imaging_skia::CommonRendererState;
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//! use skia_safe as sk;
//!
//! fn main() -> Result<(), imaging_skia::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x16, 0xa3, 0x4a));
//!     let mut scene = record::Scene::new();
//!
//!     {
//!         let mut painter = Painter::new(&mut scene);
//!         painter.fill_rect(Rect::new(0.0, 0.0, 128.0, 128.0), &paint);
//!     }
//!
//!     let mut pixels = vec![0_u8; 128 * 128 * 4];
//!     let info = sk::ImageInfo::new(
//!         (128, 128),
//!         sk::ColorType::RGBA8888,
//!         sk::AlphaType::Premul,
//!         Some(sk::ColorSpace::new_srgb()),
//!     );
//!     let mut surface = sk::surfaces::wrap_pixels(&info, pixels.as_mut_slice(), Some(128 * 4), None)
//!         .expect("wrap raster pixels");
//!     let mut state = CommonRendererState::new();
//!     state.render_scene(&mut surface, &scene)?;
//!     Ok(())
//! }
//! ```
//!
//! # Draw Into An Existing `Canvas`
//!
//! If you already have a Skia canvas, wrap it with [`SkCanvasSink`] and stream commands directly.
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_skia::SkCanvasSink;
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//! use skia_safe::surfaces;
//!
//! fn main() -> Result<(), imaging_skia::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x1d, 0x4e, 0x89));
//!     let mut surface = surfaces::raster_n32_premul((128, 128)).unwrap();
//!
//!     {
//!         let mut sink = SkCanvasSink::new(surface.canvas());
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(Rect::new(0.0, 0.0, 128.0, 128.0), &paint);
//!         sink.finish()?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Record A `SkPicture`
//!
//! Use [`SkPictureRecorderSink`] when you want Skia's native retained recording format.
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_skia::SkPictureRecorderSink;
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_skia::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x7c, 0x3a, 0xed));
//!     let mut sink = SkPictureRecorderSink::new(Rect::new(0.0, 0.0, 128.0, 128.0));
//!
//!     {
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(Rect::new(16.0, 16.0, 112.0, 112.0), &paint);
//!     }
//!
//!     let picture = sink.finish_picture()?;
//!     assert_eq!(picture.cull_rect().right, 128.0);
//!     Ok(())
//! }
//! ```
//!
//! # Render A Native `SkPicture`
//!
//! If you already have a recorded picture, hand it directly to [`SkiaGpuRenderer`].
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_skia::{SkPictureRecorderSink, SkiaGpuRenderer};
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_skia::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0xd9, 0x77, 0x06));
//!     let mut sink = SkPictureRecorderSink::new(Rect::new(0.0, 0.0, 128.0, 128.0));
//!
//!     {
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(Rect::new(16.0, 16.0, 112.0, 112.0), &paint);
//!     }
//!
//!     let picture = sink.finish_picture()?;
//!     let mut renderer = SkiaGpuRenderer::new(128, 128);
//!     renderer.reset()?;
//!     renderer.render_picture(&picture)?;
//!     let image = renderer.read_image()?;
//!     assert_eq!(image.width, 128);
//!     Ok(())
//! }
//! ```

#![cfg_attr(not(test), warn(unused_crate_dependencies))]

#[cfg(all(feature = "wgpu", any(target_os = "macos", target_os = "ios")))]
use core::ffi::c_void;
#[cfg(all(feature = "wgpu", any(target_os = "macos", target_os = "ios")))]
use foreign_types_shared::ForeignType;

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metal;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod opengl;
mod sinks;
#[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios"))))]
mod vulkan;

mod font_cache;
mod ganesh;
use core::convert::Infallible;
use imaging::PaintSink;
use imaging::{
    Filter, GeometryRef,
    record::{Scene, ValidateError, replay},
};
use imaging_backend::{
    Backend as ImagingBackend, CpuBufferAlphaMode, CpuBufferChannelOrder, CpuBufferTarget,
    RenderSource,
};
#[cfg(feature = "wgpu")]
use imaging_backend::{GpuTextureTarget, RenderOutput};
use kurbo::{Affine, Shape as _, Size};
use peniko::color::{ColorSpaceTag, HueDirection, Srgb};
use peniko::{BrushRef, ImageAlphaType, ImageFormat, ImageQuality, InterpolationAlphaSpace};
use skia_safe as sk;
#[cfg(feature = "wgpu")]
use std::collections::HashSet;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::font_cache::skia_font_from_glyph_run;
use crate::ganesh::{GaneshBackend, create_surface as create_ganesh_surface};
use std::sync::Arc;

use sinks::{MaskImageCache, RetainedImageCache};
pub use sinks::{SkCanvasSink, SkPictureRecorderSink};

pub(crate) type ImageCacheHandle = Rc<RefCell<ImageCache>>;

#[derive(Debug, Default)]
pub(crate) struct ImageCache {
    images: HashMap<ImageCacheKey, sk::Image>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ImageCacheKey {
    blob_id: u64,
    format: core::mem::Discriminant<ImageFormat>,
    alpha_type: core::mem::Discriminant<ImageAlphaType>,
    width: u32,
    height: u32,
}

impl ImageCacheKey {
    fn new(image: &peniko::ImageData) -> Self {
        Self {
            blob_id: image.data.id(),
            format: core::mem::discriminant(&image.format),
            alpha_type: core::mem::discriminant(&image.alpha_type),
            width: image.width,
            height: image.height,
        }
    }
}

impl ImageCache {
    fn clear(&mut self) {
        self.images.clear();
    }

    fn get_or_create(&mut self, image: &peniko::ImageData) -> Option<sk::Image> {
        let key = ImageCacheKey::new(image);
        if let Some(cached) = self.images.get(&key) {
            return Some(cached.clone());
        }
        let sk_image = make_skia_image_from_peniko(image)?;
        self.images.insert(key, sk_image.clone());
        Some(sk_image)
    }
}

/// Errors that can occur when rendering via Skia.
#[derive(Debug)]
pub enum Error {
    /// The scene is invalid (unbalanced stacks).
    InvalidScene(ValidateError),
    /// No supported Ganesh backend was available on this platform.
    UnsupportedGpuBackend,
    /// A Ganesh backend could not be initialized.
    CreateGpuContext(&'static str),
    /// A Ganesh render target surface could not be created.
    CreateGpuSurface,
    /// An image brush was encountered; this backend does not support it.
    UnsupportedImageBrush,
    /// A filter configuration could not be translated.
    UnsupportedFilter,
    /// A glyph run used variable-font coordinates unsupported by this backend.
    UnsupportedGlyphVariations,
    /// A glyph run used a per-glyph transform unsupported by this backend.
    UnsupportedGlyphTransform,
    /// Font bytes could not be loaded by Skia.
    InvalidFontData,
    /// A glyph identifier could not be represented by Skia's glyph type.
    InvalidGlyphId,
    /// An internal invariant was violated.
    Internal(&'static str),
}

/// Reusable CPU raster replay state for Skia-backed rendering.
///
/// This type owns the persistent CPU-side caches and replay settings while callers provide the
/// destination raster surface for each render.
#[derive(Debug)]
pub struct CommonRendererState {
    tolerance: f64,
    mask_cache: Rc<RefCell<MaskImageCache>>,
    retained_image_cache: Rc<RefCell<RetainedImageCache>>,
}

impl Default for CommonRendererState {
    fn default() -> Self {
        Self::new()
    }
}

impl CommonRendererState {
    /// Create reusable CPU raster replay state.
    pub fn new() -> Self {
        Self {
            tolerance: 0.1,
            mask_cache: Rc::new(RefCell::new(MaskImageCache::default())),
            retained_image_cache: Rc::new(RefCell::new(RetainedImageCache::default())),
        }
    }
}

#[derive(Debug)]
struct GpuRendererState {
    backend: GaneshBackend,
    common: CommonRendererState,
    image_cache: ImageCacheHandle,
    #[cfg(feature = "wgpu")]
    wgpu_backend_keepalive: Option<WgpuDeviceQueueKeepalive>,
    #[cfg(feature = "wgpu")]
    initialized_wgpu_targets: HashSet<u64>,
}

#[derive(Debug)]
/// Generic Skia GPU renderer implementation parameterized by mode-specific storage.
pub struct SkiaRendererImpl<M> {
    state: GpuRendererState,
    #[allow(
        dead_code,
        reason = "Mode-specific state is only exercised by some concrete renderer variants."
    )]
    mode: M,
}

/// Marker for the Skia GPU renderer variant that owns an internal offscreen target.
#[derive(Debug)]
pub struct CopyMode {
    surface: sk::Surface,
    #[cfg(feature = "wgpu")]
    target_keepalive: Option<WgpuTextureHandle>,
    #[cfg(feature = "wgpu")]
    owned_wgpu_target_keepalive: Option<WgpuTextureHandle>,
}

/// Marker for the Skia GPU renderer variant that renders directly into caller-owned targets.
#[derive(Debug, Default)]
pub struct TargetMode;

/// Owned GPU renderer that allocates and retains its own offscreen Skia target.
pub type SkiaGpuRenderer = SkiaRendererImpl<CopyMode>;
/// GPU renderer that binds caller-provided targets directly.
pub type SkiaGpuTargetRenderer = SkiaRendererImpl<TargetMode>;

#[cfg(feature = "wgpu")]
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "These handles are retained only to extend native backend lifetimes."
)]
struct WgpuTextureHandle {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg(feature = "wgpu")]
impl WgpuTextureHandle {
    fn new(texture: wgpu::Texture) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }
}

#[cfg(feature = "wgpu")]
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "These handles are retained only to extend native backend lifetimes."
)]
struct WgpuDeviceQueueKeepalive {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<M> SkiaRendererImpl<M> {
    fn from_backend_and_mode(backend: GaneshBackend, mode: M) -> Self {
        Self {
            state: GpuRendererState {
                backend,
                common: CommonRendererState::new(),
                image_cache: Rc::new(RefCell::new(ImageCache::default())),
                #[cfg(feature = "wgpu")]
                wgpu_backend_keepalive: None,
                #[cfg(feature = "wgpu")]
                initialized_wgpu_targets: HashSet::new(),
            },
            mode,
        }
    }

    /// Set the geometric flattening tolerance used for path conversion.
    ///
    /// Lower values preserve curve fidelity more aggressively; higher values can reduce path
    /// complexity when rendering highly curved geometry.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.state.common.tolerance = tolerance;
        self.state.image_cache.borrow_mut().clear();
        self.state.common.mask_cache.borrow_mut().clear();
        self.state.common.retained_image_cache.borrow_mut().clear();
    }

    /// Drop any realized native mask images cached by the renderer.
    pub fn clear_cached_masks(&mut self) {
        self.state.image_cache.borrow_mut().clear();
        self.state.common.mask_cache.borrow_mut().clear();
        self.state.common.retained_image_cache.borrow_mut().clear();
    }
}

impl SkiaGpuRenderer {
    /// Build a renderer from an already-initialized Ganesh backend and wrapped target surface.
    ///
    /// Backend modules use this to share the same renderer initialization path without reaching
    /// into `SkiaRenderer`'s private fields directly.
    pub(crate) fn from_backend_surface(backend: GaneshBackend, surface: sk::Surface) -> Self {
        Self::from_backend_and_mode(
            backend,
            CopyMode {
                surface,
                #[cfg(feature = "wgpu")]
                target_keepalive: None,
                #[cfg(feature = "wgpu")]
                owned_wgpu_target_keepalive: None,
            },
        )
    }
}

#[cfg(feature = "wgpu")]
impl SkiaGpuTargetRenderer {
    pub(crate) fn from_backend(backend: GaneshBackend) -> Self {
        Self::from_backend_and_mode(backend, TargetMode)
    }
}

impl SkiaGpuRenderer {
    /// Create an offscreen GPU renderer for a fixed output size.
    ///
    /// This is the convenience entry point for the common case where `imaging_skia` owns the GPU
    /// backend and temporary render target for you.
    pub fn new(width: u16, height: u16) -> Self {
        Self::try_new(width, height).expect("create imaging_skia renderer")
    }

    /// Create an offscreen GPU renderer for a fixed output size.
    ///
    /// Use this when renderer construction may legitimately fail at runtime, such as when no
    /// compatible Ganesh backend can be created on the current machine.
    pub fn try_new(width: u16, height: u16) -> Result<Self, Error> {
        let width = i32::from(width);
        let height = i32::from(height);
        let mut backend = GaneshBackend::new()?;
        let surface = create_ganesh_surface(backend.direct_context(), width, height)?;
        Ok(Self::from_backend_surface(backend, surface))
    }
}

#[cfg(feature = "wgpu")]
impl SkiaRendererImpl<CopyMode> {
    fn try_new_from_wgpu_texture_impl(
        texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: wgpu::Texture,
    ) -> Result<Self, Error> {
        let mut initialized_wgpu_targets = HashSet::new();
        initialize_texture_for_wgpu_if_needed(
            &mut initialized_wgpu_targets,
            device,
            queue,
            &texture,
        );
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let hal_device = unsafe {
                device
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuContext("missing Metal device"))?
            };
            let hal_queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuContext("missing Metal queue"))?
            };
            let raw_device = hal_device.raw_device().as_ptr() as *mut c_void;
            let raw_command_queue = hal_queue.as_raw().lock().as_ptr() as *mut c_void;
            let mut target_renderer = unsafe {
                SkiaGpuTargetRenderer::try_new_metal_from_raw_pointers_without_texture(
                    raw_device,
                    raw_command_queue,
                )
            }?;
            let (surface, texture_keepalive) =
                target_renderer.create_wgpu_surface(texture_format, texture, device, queue)?;
            let mut renderer =
                SkiaGpuRenderer::from_backend_surface(target_renderer.state.backend, surface);
            renderer.state.initialized_wgpu_targets = initialized_wgpu_targets;
            renderer.state.wgpu_backend_keepalive = Some(WgpuDeviceQueueKeepalive {
                device: device.clone(),
                queue: queue.clone(),
            });
            renderer.mode.target_keepalive = Some(texture_keepalive);
            return Ok(renderer);
        }

        #[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios"))))]
        {
            use ash::vk::Handle as _;

            let hal_device = unsafe {
                device
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuContext("missing Vulkan device"))?
            };
            let hal_queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuContext("missing Vulkan queue"))?
            };
            let instance = hal_device.shared_instance().raw_instance().handle();
            let physical_device = hal_device.raw_physical_device();
            let raw_device = hal_device.raw_device().handle();
            let raw_queue = hal_queue.as_raw();
            let queue_family_index = hal_device.queue_family_index();
            let mut target_renderer = unsafe {
                SkiaGpuTargetRenderer::try_new_vulkan_from_raw_handles(
                    instance,
                    physical_device,
                    raw_device,
                    raw_queue,
                    queue_family_index,
                )
            }?;
            let (surface, texture_keepalive) =
                target_renderer.create_wgpu_surface(texture_format, texture, device, queue)?;
            let mut renderer =
                SkiaGpuRenderer::from_backend_surface(target_renderer.state.backend, surface);
            renderer.state.initialized_wgpu_targets = initialized_wgpu_targets;
            renderer.state.wgpu_backend_keepalive = Some(WgpuDeviceQueueKeepalive {
                device: device.clone(),
                queue: queue.clone(),
            });
            renderer.mode.target_keepalive = Some(texture_keepalive);
            return Ok(renderer);
        }

        #[allow(
            unreachable_code,
            unused_variables,
            reason = "Platform and feature cfgs intentionally leave unsupported backend paths empty."
        )]
        Err(Error::UnsupportedGpuBackend)
    }

    /// Create an offscreen renderer that shares the caller's `wgpu` device and queue.
    ///
    /// Unlike [`Self::try_new_from_wgpu_texture`], this path does not wrap a caller-owned texture.
    /// It rebuilds the platform backend from `wgpu` and then allocates an internal Skia render
    /// target, so it is the right choice when you care about backend sharing but not about drawing
    /// into a specific external texture. The renderer clones and retains the supplied `wgpu`
    /// handles so the wrapped backend objects stay alive for the renderer.
    pub fn try_new_from_wgpu_device(
        width: u16,
        height: u16,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_format: wgpu::TextureFormat,
    ) -> Result<Self, Error> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imaging_skia owned copy target"),
            size: wgpu::Extent3d {
                width: u32::from(width).max(1),
                height: u32::from(height).max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[texture_format],
        });
        let mut renderer =
            Self::try_new_from_wgpu_owned_texture(texture_format, device, queue, texture)?;
        renderer.mode.owned_wgpu_target_keepalive =
            renderer
                .mode
                .target_keepalive
                .as_ref()
                .map(|target| WgpuTextureHandle {
                    texture: target.texture.clone(),
                    view: target.view.clone(),
                });
        Ok(renderer)
    }

    /// Create an offscreen renderer around a caller-supplied `wgpu` texture that this renderer
    /// then owns and reuses internally.
    ///
    /// This is still the owned/copy renderer path. Use
    /// [`SkiaGpuTargetRenderer::try_new_from_wgpu_texture`] when you want direct rendering into a
    /// caller-managed texture instead of an internally owned offscreen target.
    pub fn try_new_from_wgpu_owned_texture(
        texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: wgpu::Texture,
    ) -> Result<Self, Error> {
        Self::try_new_from_wgpu_texture_impl(texture_format, device, queue, texture)
    }
}

#[cfg(feature = "wgpu")]
impl<M> SkiaRendererImpl<M> {
    fn create_wgpu_surface(
        &mut self,
        texture_format: wgpu::TextureFormat,
        texture: wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(sk::Surface, WgpuTextureHandle), Error> {
        initialize_texture_for_wgpu_if_needed(
            &mut self.state.initialized_wgpu_targets,
            device,
            queue,
            &texture,
        );
        let texture_keepalive = WgpuTextureHandle::new(texture);
        let texture_size = texture_keepalive.texture.size();
        let width = u16::try_from(texture_size.width)
            .map_err(|_| Error::Internal("texture width exceeds skia limit"))?;
        let height = u16::try_from(texture_size.height)
            .map_err(|_| Error::Internal("texture height exceeds skia limit"))?;

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let texture = unsafe {
                texture_keepalive
                    .texture
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuSurface)?
            };
            let texture = unsafe { texture.raw_handle() }.as_ptr() as *mut c_void;
            let surface =
                unsafe { self.create_metal_surface_raw(width, height, texture_format, texture) }?;
            return Ok((surface, texture_keepalive));
        }

        #[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios"))))]
        {
            let texture = unsafe {
                texture_keepalive
                    .texture
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuSurface)?
            };
            let raw_image = unsafe { texture.raw_handle() };
            let queue_family_index = match &self.state.backend {
                GaneshBackend::Vulkan(backend) => backend.queue_family_index(),
                _ => sk::gpu::vk::QUEUE_FAMILY_IGNORED,
            };
            let surface = unsafe {
                self.create_vulkan_surface(
                    width,
                    height,
                    texture_format,
                    raw_image,
                    ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    ash::vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    1,
                    queue_family_index,
                )
            }?;
            return Ok((surface, texture_keepalive));
        }

        #[allow(
            unreachable_code,
            unused_variables,
            reason = "Platform and feature cfgs intentionally leave unsupported backend paths empty."
        )]
        Err(Error::UnsupportedGpuBackend)
    }
}

#[cfg(feature = "wgpu")]
impl SkiaGpuTargetRenderer {
    /// Create a renderer that shares the caller's `wgpu` backend for later target binding.
    pub fn try_new_from_wgpu_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self, Error> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let hal_device = unsafe {
                device
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuContext("missing Metal device"))?
            };
            let hal_queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuContext("missing Metal queue"))?
            };
            let raw_device = hal_device.raw_device().as_ptr() as *mut c_void;
            let raw_command_queue = hal_queue.as_raw().lock().as_ptr() as *mut c_void;
            let mut renderer = unsafe {
                Self::try_new_metal_from_raw_pointers_without_texture(raw_device, raw_command_queue)
            }?;
            renderer.state.wgpu_backend_keepalive = Some(WgpuDeviceQueueKeepalive {
                device: device.clone(),
                queue: queue.clone(),
            });
            return Ok(renderer);
        }

        #[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios"))))]
        {
            use ash::vk::Handle as _;
            let hal_device = unsafe {
                device
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuContext("missing Vulkan device"))?
            };
            let hal_queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuContext("missing Vulkan queue"))?
            };
            let instance = hal_device.shared_instance().raw_instance().handle();
            let physical_device = hal_device.raw_physical_device();
            let raw_device = hal_device.raw_device().handle();
            let raw_queue = hal_queue.as_raw();
            let queue_family_index = hal_device.queue_family_index();
            let mut renderer = unsafe {
                Self::try_new_vulkan_from_raw_handles(
                    instance,
                    physical_device,
                    raw_device,
                    raw_queue,
                    queue_family_index,
                )
            }?;
            renderer.state.wgpu_backend_keepalive = Some(WgpuDeviceQueueKeepalive {
                device: device.clone(),
                queue: queue.clone(),
            });
            return Ok(renderer);
        }

        #[allow(
            unreachable_code,
            unused_variables,
            reason = "Platform and feature cfgs intentionally leave unsupported backend paths empty."
        )]
        Err(Error::UnsupportedGpuBackend)
    }

    /// Backward-compatible constructor that ignores the initial texture until render time.
    pub fn try_new_from_wgpu_texture(
        _texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _texture: wgpu::Texture,
    ) -> Result<Self, Error> {
        Self::try_new_from_wgpu_device(device, queue)
    }
}

impl GpuRendererState {
    fn begin_frame(&mut self) -> Result<(), Error> {
        self.backend.ensure_current()?;
        self.common.retained_image_cache.borrow_mut().flip_mark();
        Ok(())
    }

    fn reset_surface(&mut self, surface: &mut sk::Surface) -> Result<(), Error> {
        let canvas = surface.canvas();
        canvas.restore_to_count(1);
        canvas.reset_matrix();
        canvas.clear(sk::Color::TRANSPARENT);
        Ok(())
    }

    fn canvas_sink<'a>(&'a mut self, surface: &'a mut sk::Surface) -> SkCanvasSink<'a> {
        let mut sink = SkCanvasSink::new_with_mask_cache(
            surface.canvas(),
            Some(Rc::clone(&self.image_cache)),
            Rc::clone(&self.common.mask_cache),
            Rc::clone(&self.common.retained_image_cache),
        );
        sink.set_tolerance(self.common.tolerance);
        sink
    }

    fn with_canvas_sink_on_surface<R>(
        &mut self,
        surface: &mut sk::Surface,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        let mut sink = self.canvas_sink(surface);
        let out = f(&mut sink);
        let finish_result = sink.finish();
        finish_result?;
        Ok(out)
    }

    fn render_scene_on_surface(
        &mut self,
        surface: &mut sk::Surface,
        scene: &Scene,
    ) -> Result<(), Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        let mut sink = self.canvas_sink(surface);
        replay(scene, &mut sink);
        let finish_result = sink.finish();
        finish_result?;
        Ok(())
    }

    fn render_picture_on_surface(
        &mut self,
        surface: &mut sk::Surface,
        picture: &sk::Picture,
    ) -> Result<(), Error> {
        surface.canvas().draw_picture(picture, None, None);
        Ok(())
    }
}

impl SkiaGpuRenderer {
    #[cfg_attr(
        not(feature = "wgpu"),
        allow(dead_code, reason = "This shared GPU helper is currently exercised by wgpu paths.")
    )]
    fn with_paint_sink(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink)) {
        self.state
            .with_canvas_sink_on_surface(&mut self.mode.surface, |sink| f(sink))
            .expect("render into imaging_skia gpu canvas sink");
    }

    #[cfg_attr(
        not(feature = "wgpu"),
        allow(dead_code, reason = "This shared GPU helper is currently exercised by wgpu paths.")
    )]
    fn finish(&mut self) {
        self.state
            .common
            .retained_image_cache
            .borrow_mut()
            .evict_unmarked();
        self.state
            .backend
            .ensure_current()
            .expect("make imaging_skia gpu backend current");
        self.state.backend.flush_surface(&mut self.mode.surface);
    }

    #[cfg_attr(
        not(feature = "wgpu"),
        allow(dead_code, reason = "This shared GPU helper is currently exercised by wgpu paths.")
    )]
    fn reset_for_frame(&mut self) {
        Self::reset(self).expect("reset imaging_skia renderer");
    }

    /// Reset canvas state on the wrapped GPU surface before issuing new draw commands.
    ///
    /// This clears the surface to transparent, restores the canvas stack to its root save level,
    /// and resets the current transform. This method does not begin or finish a frame on its own;
    /// backend entry points and higher-level render helpers are responsible for frame ownership.
    ///
    /// Call this when you intentionally want to discard previous contents before a top-level render
    /// pass. Do not call it after you have started issuing commands for the frame you want to keep.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.state.reset_surface(&mut self.mode.surface)
    }

    /// Stream `imaging` commands directly into the current GPU surface.
    ///
    /// This is a low-level replay helper. It records into the bound surface but does not reset the
    /// canvas, begin a frame, or flush/present work on its own. Callers that use it directly are
    /// responsible for surface preparation and frame boundaries.
    pub fn with_canvas_sink<R>(
        &mut self,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        self.state
            .with_canvas_sink_on_surface(&mut self.mode.surface, f)
    }

    /// Replay an `imaging` scene into the current GPU surface.
    ///
    /// This validates and records the scene into the current surface but does not reset the
    /// surface, begin a frame, or flush the backend by itself. Use this after explicit frame
    /// setup when composing custom render flows.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<(), Error> {
        self.state
            .render_scene_on_surface(&mut self.mode.surface, scene)
    }

    /// Draw an existing native Skia picture into the current GPU surface.
    ///
    /// Like [`Self::render_scene`], this is a low-level draw helper. It does not reset the
    /// destination or finalize the frame for you.
    pub fn render_picture(&mut self, picture: &sk::Picture) -> Result<(), Error> {
        self.state
            .render_picture_on_surface(&mut self.mode.surface, picture)
    }

    /// Borrow the live GPU-backed `skia_safe::Surface`.
    pub fn surface(&mut self) -> &mut sk::Surface {
        &mut self.mode.surface
    }

    /// Snapshot the current GPU surface as a Skia image.
    pub fn image_snapshot(&mut self) -> sk::Image {
        let _ = self.state.backend.ensure_current();
        self.state.backend.flush_surface(&mut self.mode.surface);
        self.mode.surface.image_snapshot()
    }

    /// Expose Skia's backend texture for the current surface when the backend supports it.
    pub fn backend_texture(&mut self) -> Option<sk::gpu::BackendTexture> {
        let _ = self.state.backend.ensure_current();
        self.state.backend.flush_surface(&mut self.mode.surface);
        sk::gpu::surfaces::get_backend_texture(
            &mut self.mode.surface,
            sk::surface::BackendHandleAccess::FlushRead,
        )
    }

    /// Read back the current GPU surface into an unpremultiplied sRGB RGBA8 image.
    ///
    /// Rendering methods funnel through this helper after flushing work to the active backend.
    pub fn read_image(&mut self) -> Result<peniko::ImageData, Error> {
        self.state.backend.ensure_current()?;
        self.state.backend.flush_surface(&mut self.mode.surface);
        let info = sk::ImageInfo::new(
            (self.mode.surface.width(), self.mode.surface.height()),
            sk::ColorType::RGBA8888,
            sk::AlphaType::Unpremul,
            None,
        );
        let mut bytes =
            vec![
                0_u8;
                (self.mode.surface.width() as usize) * (self.mode.surface.height() as usize) * 4
            ];
        let ok = self.mode.surface.read_pixels(
            &info,
            bytes.as_mut_slice(),
            (4 * self.mode.surface.width()) as usize,
            (0, 0),
        );
        if !ok {
            return Err(Error::Internal("read_pixels failed"));
        }
        Ok(peniko::ImageData {
            data: peniko::Blob::new(Arc::new(bytes)),
            format: ImageFormat::Rgba8,
            width: self.mode.surface.width() as u32,
            height: self.mode.surface.height() as u32,
            alpha_type: ImageAlphaType::Alpha,
        })
    }
}

#[cfg(feature = "wgpu")]
impl SkiaGpuRenderer {
    /// Borrow the live owned `wgpu` texture when the renderer is targeting one.
    pub fn wgpu_texture(&self) -> Option<&wgpu::Texture> {
        self.mode
            .target_keepalive
            .as_ref()
            .map(|target| &target.texture)
    }

    /// Borrow the live `wgpu::TextureView` for the currently owned texture.
    pub fn wgpu_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.mode
            .target_keepalive
            .as_ref()
            .map(|target| &target.view)
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "Frame sizes are converted to whole pixels and then checked against `u16`."
    )]
    fn set_size(&mut self, size: Size) {
        let width = u16::try_from(size.width as u32).expect("skia width out of range");
        let height = u16::try_from(size.height as u32).expect("skia height out of range");
        let surface = &mut self.mode.surface;
        if surface.width() != i32::from(width) || surface.height() != i32::from(height) {
            let keepalive = self
                .state
                .wgpu_backend_keepalive
                .as_ref()
                .expect("copy skia renderer keeps wgpu backend alive");
            let texture_format = self
                .mode
                .owned_wgpu_target_keepalive
                .as_ref()
                .expect("copy skia renderer owns a wgpu target")
                .texture
                .format();
            *self = Self::try_new_from_wgpu_device(
                width,
                height,
                &keepalive.device,
                &keepalive.queue,
                texture_format,
            )
            .expect("recreate imaging_skia copy renderer");
        }
    }
}

#[cfg(feature = "wgpu")]
impl SkiaGpuTargetRenderer {
    fn render_target(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        target: GpuTextureTarget,
    ) -> Result<(), String> {
        let device = target.device;
        let queue = target.queue;
        let texture = target.texture_view.texture().clone();
        let texture_format = texture.format();
        let mut surface = self
            .create_wgpu_surface(texture_format, texture, &device, &queue)
            .map_err(|err| format!("{err:?}"))?
            .0;
        let width = size.width as i32;
        let height = size.height as i32;
        if surface.width() != width || surface.height() != height {
            return Err("skia target size must match bound texture size".to_string());
        }
        self.state.begin_frame().map_err(|err| format!("{err:?}"))?;
        self.state
            .reset_surface(&mut surface)
            .map_err(|err| format!("{err:?}"))?;
        self.state
            .with_canvas_sink_on_surface(&mut surface, |sink| source.paint_into(sink))
            .map_err(|err| format!("{err:?}"))?;
        self.state
            .common
            .retained_image_cache
            .borrow_mut()
            .evict_unmarked();
        self.state
            .backend
            .ensure_current()
            .map_err(|err| format!("{err:?}"))?;
        self.state.backend.flush_surface(&mut surface);
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
impl ImagingBackend for SkiaGpuTargetRenderer {
    type Error = String;
    type Image = peniko::ImageData;
    type BufferTarget<'a> = Infallible;
    type TextureTarget<'a> = GpuTextureTarget;

    fn render_to_buffer<'a>(
        &mut self,
        _size: Size,
        _source: &mut dyn RenderSource,
        target: Self::BufferTarget<'a>,
    ) -> Result<(), Self::Error> {
        match target {}
    }

    fn render_to_texture<'a>(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        target: Self::TextureTarget<'a>,
    ) -> Result<(), Self::Error> {
        self.render_target(size, source, target)
    }

    fn render_to_image(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        width: u32,
        height: u32,
    ) -> Result<Self::Image, Self::Error> {
        let keepalive = self.state.wgpu_backend_keepalive.as_ref().ok_or_else(|| {
            "direct skia target renderer requires wgpu backend for capture".to_string()
        })?;
        let device = keepalive.device.clone();
        let queue = keepalive.queue.clone();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imaging_skia direct target capture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_target(
            size,
            source,
            GpuTextureTarget {
                device: device.clone(),
                queue: queue.clone(),
                texture_view: texture_view.clone(),
            },
        )?;
        RenderOutput::GpuTexture(texture_view)
            .into_image_with(&device, &queue)
            .ok_or_else(|| "direct skia target renderer capture readback failed".to_string())
    }
}

#[cfg(feature = "wgpu")]
impl SkiaGpuRenderer {
    fn render_copy_to_texture(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        target: GpuTextureTarget,
    ) -> Result<(), String> {
        let owned_texture = self
            .mode
            .owned_wgpu_target_keepalive
            .as_ref()
            .ok_or_else(|| "copy skia renderer owns no internal target".to_string())?
            .texture
            .clone();
        let owned_format = owned_texture.format();
        let device = target.device;
        let queue = target.queue;
        let target_texture = target.texture_view.texture().clone();
        let target_format = target_texture.format();

        self.set_size(size);
        self.state.begin_frame().map_err(|err| format!("{err:?}"))?;
        self.reset_for_frame();
        self.with_paint_sink(&mut |sink| source.paint_into(sink));
        self.finish();
        let snapshot = self.image_snapshot();
        let mut target_surface = self
            .create_wgpu_surface(target_format, target_texture, &device, &queue)
            .map_err(|err| format!("{err:?}"))?
            .0;
        self.state
            .reset_surface(&mut target_surface)
            .map_err(|err| format!("{err:?}"))?;
        target_surface.canvas().draw_image(&snapshot, (0, 0), None);
        self.state
            .common
            .retained_image_cache
            .borrow_mut()
            .evict_unmarked();
        self.state
            .backend
            .ensure_current()
            .map_err(|err| format!("{err:?}"))?;
        self.state.backend.flush_surface(&mut target_surface);

        let (_, target_keepalive) = self
            .create_wgpu_surface(owned_format, owned_texture, &device, &queue)
            .map_err(|err| format!("{err:?}"))?;
        self.mode.target_keepalive = Some(target_keepalive);
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
impl ImagingBackend for SkiaGpuRenderer {
    type Error = String;
    type Image = peniko::ImageData;
    type BufferTarget<'a> = Infallible;
    type TextureTarget<'a> = GpuTextureTarget;

    fn render_to_buffer<'a>(
        &mut self,
        _size: Size,
        _source: &mut dyn RenderSource,
        _target: Self::BufferTarget<'a>,
    ) -> Result<(), Self::Error> {
        unreachable!()
    }

    fn render_to_texture<'a>(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        target: Self::TextureTarget<'a>,
    ) -> Result<(), Self::Error> {
        self.render_copy_to_texture(size, source, target)
    }

    fn render_to_image(
        &mut self,
        _size: Size,
        source: &mut dyn RenderSource,
        width: u32,
        height: u32,
    ) -> Result<Self::Image, Self::Error> {
        self.set_size(Size::new(width as f64, height as f64));
        self.reset_for_frame();
        self.with_paint_sink(&mut |sink| source.paint_into(sink));
        self.finish();
        self.read_image().map_err(|err| format!("{err:?}"))
    }
}

/// Reusable CPU raster replay state for Skia-backed rendering.
///
/// This type keeps renderer-side state such as tolerance while callers provide the destination
/// raster surface for each render.

impl CommonRendererState {
    /// Create a short-lived renderer view bound to a caller-provided raster surface.
    pub fn bind<'a>(&'a mut self, surface: &'a mut sk::Surface) -> SkiaCpuRendererRef<'a> {
        SkiaCpuRendererRef {
            state: self,
            mode: BoundCpuMode { surface },
        }
    }

    /// Reset canvas state on the provided raster surface.
    ///
    /// This clears the surface to transparent, restores the canvas stack to its root save level,
    /// and resets the current transform. It does not begin or finish a frame by itself.
    ///
    /// Call this before a top-level render pass when you want fresh contents. Do not call it
    /// mid-frame unless you intentionally want to discard earlier drawing.
    pub fn reset(surface: &mut sk::Surface) {
        let canvas = surface.canvas();
        canvas.restore_to_count(1);
        canvas.reset_matrix();
        canvas.clear(sk::Color::TRANSPARENT);
    }

    fn canvas_sink<'a>(&'a mut self, surface: &'a mut sk::Surface) -> SkCanvasSink<'a> {
        let mut sink = SkCanvasSink::new_with_mask_cache(
            surface.canvas(),
            None,
            Rc::clone(&self.mask_cache),
            Rc::clone(&self.retained_image_cache),
        );
        sink.set_tolerance(self.tolerance);
        sink
    }

    /// Stream `imaging` commands directly into the provided raster surface.
    ///
    /// This is the reusable raster replay primitive. It records into the supplied surface and
    /// finishes the sink, but it does not clear the surface first. Callers should pair it with an
    /// explicit [`Self::reset`] when they need a clean destination.
    pub fn with_canvas_sink<R>(
        &mut self,
        surface: &mut sk::Surface,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        let mut sink = self.canvas_sink(surface);
        let out = f(&mut sink);
        let finish_result = sink.finish();
        finish_result?;
        Ok(out)
    }

    /// Replay an `imaging` scene through the raster backend into the provided surface.
    ///
    /// This is a low-level draw helper. It validates and replays the scene, but it does not reset
    /// the surface first. Top-level render entry points should do that explicitly when they want a
    /// fresh frame.
    pub fn render_scene(&mut self, surface: &mut sk::Surface, scene: &Scene) -> Result<(), Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        self.with_canvas_sink(surface, |sink| replay(scene, sink))
            .map(|_| ())
    }

    /// Draw a native Skia picture through the raster backend into the provided surface.
    ///
    /// This draws into the current surface contents without resetting them first.
    pub fn render_picture(
        &mut self,
        surface: &mut sk::Surface,
        picture: &sk::Picture,
    ) -> Result<(), Error> {
        surface.canvas().draw_picture(picture, None, None);
        Ok(())
    }

    /// Read back the current raster surface into an unpremultiplied sRGB RGBA8 image.
    ///
    /// This is the raster counterpart to the GPU renderer's readback helper.
    pub fn read_image(surface: &mut sk::Surface) -> Result<peniko::ImageData, Error> {
        let image = surface.image_snapshot();
        let dims = image.dimensions();
        let info = sk::ImageInfo::new(
            (dims.width, dims.height),
            sk::ColorType::RGBA8888,
            sk::AlphaType::Unpremul,
            None,
        );
        let mut bytes = vec![0_u8; (dims.width as usize) * (dims.height as usize) * 4];
        let ok = image.read_pixels(
            &info,
            bytes.as_mut_slice(),
            (4 * dims.width) as usize,
            (0, 0),
            sk::image::CachingHint::Disallow,
        );
        if !ok {
            return Err(Error::Internal("read_pixels failed"));
        }
        Ok(peniko::ImageData {
            data: peniko::Blob::new(Arc::new(bytes)),
            format: ImageFormat::Rgba8,
            width: dims.width as u32,
            height: dims.height as u32,
            alpha_type: ImageAlphaType::Alpha,
        })
    }
}

/// Generic Skia CPU renderer implementation parameterized by mode-specific storage.
#[derive(Debug)]
pub struct SkiaCpuRendererImpl<S, M> {
    state: S,
    mode: M,
}

/// Marker for the CPU renderer variant that owns an internal raster surface.
#[derive(Debug)]
pub struct OwnedCpuMode {
    surface: sk::Surface,
}

/// Marker for the CPU renderer variant that binds a caller-provided raster surface.
#[derive(Debug)]
pub struct BoundCpuMode<'a> {
    surface: &'a mut sk::Surface,
}

/// Owned CPU raster renderer that allocates and retains its own raster surface.
pub type SkiaCpuRenderer = SkiaCpuRendererImpl<CommonRendererState, OwnedCpuMode>;
/// Borrowed CPU raster renderer view that binds reusable CPU state to a caller-owned surface.
pub type SkiaCpuRendererRef<'a> =
    SkiaCpuRendererImpl<&'a mut CommonRendererState, BoundCpuMode<'a>>;

/// CPU target renderer that implements `imaging_backend::Backend` over caller-provided CPU
/// targets.
#[derive(Debug, Default)]
pub struct SkiaCpuTargetRenderer {
    state: CommonRendererState,
}

impl SkiaCpuTargetRenderer {
    /// Create a CPU target renderer backed by reusable raster replay state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the reusable raster replay state.
    pub fn state(&self) -> &CommonRendererState {
        &self.state
    }

    /// Borrow the reusable raster replay state mutably.
    pub fn state_mut(&mut self) -> &mut CommonRendererState {
        &mut self.state
    }

    /// Consume the wrapper and return the underlying reusable raster replay state.
    pub fn into_state(self) -> CommonRendererState {
        self.state
    }
}

impl SkiaCpuRendererImpl<CommonRendererState, OwnedCpuMode> {
    /// Borrow the live raster `skia_safe::Surface`.
    pub fn surface(&mut self) -> &mut sk::Surface {
        &mut self.mode.surface
    }

    /// Set the geometric flattening tolerance used for path conversion.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.state.tolerance = tolerance;
        self.state.mask_cache.borrow_mut().clear();
        self.state.retained_image_cache.borrow_mut().clear();
    }

    /// Drop any realized mask artifacts cached by the renderer.
    pub fn clear_cached_masks(&mut self) {
        self.state.mask_cache.borrow_mut().clear();
        self.state.retained_image_cache.borrow_mut().clear();
    }

    /// Reset canvas state on the current raster surface.
    ///
    /// This is a surface-preparation helper only. It does not read back results or otherwise
    /// finalize the frame.
    pub fn reset(&mut self) {
        CommonRendererState::reset(&mut self.mode.surface);
    }

    /// Stream `imaging` commands directly into the current raster surface.
    ///
    /// This forwards to the reusable CPU replay path and does not clear the surface first.
    pub fn with_canvas_sink<R>(
        &mut self,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        self.state.with_canvas_sink(&mut self.mode.surface, f)
    }

    /// Replay an `imaging` scene through the current raster backend.
    ///
    /// This does not call [`Self::reset`]. Use the `*_rgba8` convenience methods when you want a
    /// fresh frame plus readback in one call.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<(), Error> {
        self.state.render_scene(&mut self.mode.surface, scene)
    }

    /// Reset, render a recorded scene, and return RGBA8 bytes.
    pub fn render_scene_rgba8(&mut self, scene: &Scene) -> Result<Vec<u8>, Error> {
        self.state.retained_image_cache.borrow_mut().flip_mark();
        self.reset();
        self.render_scene(scene)?;
        self.state
            .retained_image_cache
            .borrow_mut()
            .evict_unmarked();
        Ok(self.read_image()?.data.as_ref().to_vec())
    }

    /// Draw a native Skia picture through the current raster backend.
    ///
    /// This preserves existing contents unless the caller reset the surface earlier in the frame.
    pub fn render_picture(&mut self, picture: &sk::Picture) -> Result<(), Error> {
        self.state.render_picture(&mut self.mode.surface, picture)
    }

    /// Reset, draw a native Skia picture, and return RGBA8 bytes.
    pub fn render_picture_rgba8(&mut self, picture: &sk::Picture) -> Result<Vec<u8>, Error> {
        self.state.retained_image_cache.borrow_mut().flip_mark();
        self.reset();
        self.render_picture(picture)?;
        self.state
            .retained_image_cache
            .borrow_mut()
            .evict_unmarked();
        Ok(self.read_image()?.data.as_ref().to_vec())
    }

    /// Read back the current raster surface into an unpremultiplied RGBA8 image.
    pub fn read_image(&mut self) -> Result<peniko::ImageData, Error> {
        CommonRendererState::read_image(&mut self.mode.surface)
    }
}

impl SkiaCpuRendererImpl<&mut CommonRendererState, BoundCpuMode<'_>> {
    /// Set the geometric flattening tolerance used for path conversion.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.state.tolerance = tolerance;
        self.state.mask_cache.borrow_mut().clear();
        self.state.retained_image_cache.borrow_mut().clear();
    }

    /// Drop any realized mask artifacts cached by the renderer.
    pub fn clear_cached_masks(&mut self) {
        self.state.mask_cache.borrow_mut().clear();
        self.state.retained_image_cache.borrow_mut().clear();
    }

    /// Borrow the live raster `skia_safe::Surface`.
    pub fn surface(&mut self) -> &mut sk::Surface {
        self.mode.surface
    }

    /// Reset canvas state on the current raster surface.
    ///
    /// This is a surface-preparation helper only. It does not read back results or otherwise
    /// finalize the frame.
    pub fn reset(&mut self) {
        CommonRendererState::reset(self.mode.surface);
    }

    /// Stream `imaging` commands directly into the current raster surface.
    ///
    /// This forwards to the reusable CPU replay path and does not clear the surface first.
    pub fn with_canvas_sink<R>(
        &mut self,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        self.state.with_canvas_sink(self.mode.surface, f)
    }

    /// Replay an `imaging` scene through the current raster backend.
    ///
    /// This does not call [`Self::reset`]. Use the `*_rgba8` convenience methods when you want a
    /// fresh frame plus readback in one call.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<(), Error> {
        self.state.render_scene(self.mode.surface, scene)
    }

    /// Reset, render a recorded scene, and return RGBA8 bytes.
    pub fn render_scene_rgba8(&mut self, scene: &Scene) -> Result<Vec<u8>, Error> {
        self.state.retained_image_cache.borrow_mut().flip_mark();
        self.reset();
        self.render_scene(scene)?;
        self.state
            .retained_image_cache
            .borrow_mut()
            .evict_unmarked();
        Ok(self.read_image()?.data.as_ref().to_vec())
    }

    /// Draw a native Skia picture through the current raster backend.
    ///
    /// This preserves existing contents unless the caller reset the surface earlier in the frame.
    pub fn render_picture(&mut self, picture: &sk::Picture) -> Result<(), Error> {
        self.state.render_picture(self.mode.surface, picture)
    }

    /// Reset, draw a native Skia picture, and return RGBA8 bytes.
    pub fn render_picture_rgba8(&mut self, picture: &sk::Picture) -> Result<Vec<u8>, Error> {
        self.state.retained_image_cache.borrow_mut().flip_mark();
        self.reset();
        self.render_picture(picture)?;
        self.state
            .retained_image_cache
            .borrow_mut()
            .evict_unmarked();
        Ok(self.read_image()?.data.as_ref().to_vec())
    }

    /// Read back the current raster surface into an unpremultiplied RGBA8 image.
    pub fn read_image(&mut self) -> Result<peniko::ImageData, Error> {
        CommonRendererState::read_image(self.mode.surface)
    }
}

impl ImagingBackend for SkiaCpuTargetRenderer {
    type Error = String;
    type Image = peniko::ImageData;
    type BufferTarget<'a> = CpuBufferTarget<'a>;
    type TextureTarget<'a> = Infallible;

    fn render_to_buffer<'a>(
        &mut self,
        _size: Size,
        source: &mut dyn RenderSource,
        target: Self::BufferTarget<'a>,
    ) -> Result<(), Self::Error> {
        let color_type = match target.format.channel_order {
            CpuBufferChannelOrder::Rgba8 => sk::ColorType::RGBA8888,
            CpuBufferChannelOrder::Bgra8 => sk::ColorType::BGRA8888,
        };
        let alpha_type = match target.format.alpha_mode {
            CpuBufferAlphaMode::Opaque => sk::AlphaType::Opaque,
            CpuBufferAlphaMode::Premultiplied => sk::AlphaType::Premul,
        };
        let info = sk::ImageInfo::new(
            (target.width as i32, target.height as i32),
            color_type,
            alpha_type,
            None,
        );
        let mut surface =
            sk::surfaces::wrap_pixels(&info, target.buffer, Some(target.bytes_per_row), None)
                .ok_or_else(|| "wrap skia cpu target pixels".to_string())?;
        self.state.retained_image_cache.borrow_mut().flip_mark();
        CommonRendererState::reset(&mut surface);
        self.state.with_canvas_sink(&mut surface, |sink| source.paint_into(sink))
            .map_err(|err| format!("{err:?}"))?;
        self.state.retained_image_cache.borrow_mut().evict_unmarked();
        Ok(())
    }

    fn render_to_texture<'a>(
        &mut self,
        _size: Size,
        _source: &mut dyn RenderSource,
        target: Self::TextureTarget<'a>,
    ) -> Result<(), Self::Error> {
        match target {}
    }

    fn render_to_image(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        width: u32,
        height: u32,
    ) -> Result<Self::Image, Self::Error> {
        let info = sk::ImageInfo::new(
            (width as i32, height as i32),
            sk::ColorType::RGBA8888,
            sk::AlphaType::Premul,
            None,
        );
        let mut surface = sk::surfaces::raster(&info, None, None)
            .ok_or_else(|| "create skia raster RGBA8888/premul surface".to_string())?;
        self.state.retained_image_cache.borrow_mut().flip_mark();
        CommonRendererState::reset(&mut surface);
        self.state.with_canvas_sink(&mut surface, |sink| {
            let _ = size;
            source.paint_into(sink);
        })
        .map_err(|err| format!("{err:?}"))?;
        self.state.retained_image_cache.borrow_mut().evict_unmarked();
        CommonRendererState::read_image(&mut surface).map_err(|err| format!("{err:?}"))
    }
}

impl SkiaCpuRenderer {
    /// Create a CPU raster renderer for a fixed output size.
    ///
    /// This is the fallback path when callers want Skia rendering without any GPU backend or
    /// external graphics interop.
    pub fn new(width: u16, height: u16) -> Self {
        let width = i32::from(width);
        let height = i32::from(height);
        let info = sk::ImageInfo::new(
            (width, height),
            sk::ColorType::RGBA8888,
            sk::AlphaType::Premul,
            None,
        );
        let surface = sk::surfaces::raster(&info, None, None)
            .expect("create skia raster RGBA8888/premul surface");
        Self {
            state: CommonRendererState::new(),
            mode: OwnedCpuMode { surface },
        }
    }
}

#[cfg(feature = "wgpu")]
/// Map supported `wgpu` texture formats to the Skia color types used for wrapped surfaces.
pub(crate) fn color_type_for_wgpu_texture_format(
    texture_format: wgpu::TextureFormat,
) -> Result<sk::ColorType, Error> {
    match texture_format {
        wgpu::TextureFormat::Rgba8Unorm => Ok(sk::ColorType::RGBA8888),
        wgpu::TextureFormat::Rgba8UnormSrgb => Ok(sk::ColorType::RGBA8888),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            Ok(sk::ColorType::BGRA8888)
        }
        wgpu::TextureFormat::Rgb10a2Unorm => Ok(sk::ColorType::RGBA1010102),
        wgpu::TextureFormat::Rgba16Unorm => Ok(sk::ColorType::R16G16B16A16UNorm),
        wgpu::TextureFormat::Rgba16Float => Ok(sk::ColorType::RGBAF16),
        _ => Err(Error::Internal("unsupported wgpu texture format")),
    }
}

/// Initializes wgpu's internal resource tracking state for a texture before
/// handing its raw Metal handle to Skia.
///
/// On Metal, wgpu lazily initializes textures the first time they are used
/// through wgpu itself. If Skia renders into the raw `MTLTexture` before wgpu
/// has touched it, wgpu will later insert its own clear pass to "initialize"
/// the texture — overwriting whatever Skia drew and producing a black frame.
///
/// By submitting a render pass that clears to transparent here, we force wgpu
/// to mark the texture as initialized before Skia takes ownership of it, so
/// wgpu's deferred clear never fires. The clear value doesn't matter since
/// Skia will overwrite the entire texture, but transparent is the least
/// surprising default if anything goes wrong.
#[cfg(feature = "wgpu")]
fn wgpu_texture_init_key(texture: &wgpu::Texture) -> Option<u64> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let texture = unsafe { texture.as_hal::<wgpu::hal::api::Metal>()? };
        return Some(unsafe { texture.raw_handle() }.as_ptr() as usize as u64);
    }

    #[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios"))))]
    {
        use ash::vk::Handle as _;

        let texture = unsafe { texture.as_hal::<wgpu::hal::api::Vulkan>()? };
        return Some(texture.raw_handle().as_raw());
    }

    #[allow(
        unreachable_code,
        reason = "Feature-gated backend probes return early on supported platforms and fall through otherwise."
    )]
    None
}

#[cfg(feature = "wgpu")]
fn initialize_texture_for_wgpu_if_needed(
    initialized_targets: &mut HashSet<u64>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) {
    if let Some(key) = wgpu_texture_init_key(texture) {
        if initialized_targets.insert(key) {
            initialize_texture_for_wgpu(device, queue, texture);
        }
    } else {
        initialize_texture_for_wgpu(device, queue, texture);
    }
}

#[cfg(feature = "wgpu")]
fn initialize_texture_for_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) {
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("texture_init"),
    });
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("texture_init"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    drop(_pass);
    queue.submit([encoder.finish()]);
}

/// Narrow an `f64` value to `f32` for Skia APIs that operate in single precision.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Skia APIs consume f32; truncation from f64 geometry is acceptable"
)]
fn f64_to_f32(v: f64) -> f32 {
    v as f32
}

/// Convert a `kurbo` affine transform into Skia's row-major matrix representation.
fn affine_to_matrix(xf: Affine) -> sk::Matrix {
    let a = xf.as_coeffs();
    sk::Matrix::new_all(
        f64_to_f32(a[0]),
        f64_to_f32(a[2]),
        f64_to_f32(a[4]),
        f64_to_f32(a[1]),
        f64_to_f32(a[3]),
        f64_to_f32(a[5]),
        0.0,
        0.0,
        1.0,
    )
}

/// Translate `peniko` fill rules to the equivalent Skia path fill type.
fn sk_path_fill_type_from_fill_rule(rule: peniko::Fill) -> sk::PathFillType {
    match rule {
        peniko::Fill::NonZero => sk::PathFillType::Winding,
        peniko::Fill::EvenOdd => sk::PathFillType::EvenOdd,
    }
}

/// Return a path with the requested fill rule, cloning only when the rule must change.
fn path_with_fill_rule(path: &sk::Path, rule: peniko::Fill) -> sk::Path {
    let fill = sk_path_fill_type_from_fill_rule(rule);
    if path.fill_type() == fill {
        path.clone()
    } else {
        path.with_fill_type(fill)
    }
}

/// Normalize supported geometry inputs to a `kurbo::BezPath` for downstream conversion.
fn geometry_to_bez_path(geom: GeometryRef<'_>, tolerance: f64) -> Option<kurbo::BezPath> {
    Some(match geom {
        GeometryRef::Rect(r) => r.to_path(tolerance),
        GeometryRef::RoundedRect(rr) => rr.to_path(tolerance),
        GeometryRef::Path(p) => p.clone(),
        GeometryRef::OwnedPath(p) => p,
    })
}

/// Convert supported geometry inputs to a Skia path using the configured tolerance.
fn geometry_to_sk_path(geom: GeometryRef<'_>, tolerance: f64) -> Option<sk::Path> {
    let bez = geometry_to_bez_path(geom, tolerance)?;
    bez_to_sk_path(&bez)
}

/// Translate a `kurbo` bezier path into the equivalent Skia path commands.
fn bez_to_sk_path(bez: &kurbo::BezPath) -> Option<sk::Path> {
    let mut path = sk::Path::new();
    for el in bez.elements() {
        match el {
            kurbo::PathEl::MoveTo(p) => {
                path.move_to((f64_to_f32(p.x), f64_to_f32(p.y)));
            }
            kurbo::PathEl::LineTo(p) => {
                path.line_to((f64_to_f32(p.x), f64_to_f32(p.y)));
            }
            kurbo::PathEl::QuadTo(p1, p2) => {
                path.quad_to(
                    (f64_to_f32(p1.x), f64_to_f32(p1.y)),
                    (f64_to_f32(p2.x), f64_to_f32(p2.y)),
                );
            }
            kurbo::PathEl::CurveTo(p1, p2, p3) => {
                path.cubic_to(
                    (f64_to_f32(p1.x), f64_to_f32(p1.y)),
                    (f64_to_f32(p2.x), f64_to_f32(p2.y)),
                    (f64_to_f32(p3.x), f64_to_f32(p3.y)),
                );
            }
            kurbo::PathEl::ClosePath => {
                path.close();
            }
        }
    }
    Some(path)
}

/// Map `peniko` extend behavior to Skia tile modes for gradients and image shaders.
fn tile_mode_from_extend(extend: peniko::Extend) -> sk::TileMode {
    match extend {
        peniko::Extend::Pad => sk::TileMode::Clamp,
        peniko::Extend::Repeat => sk::TileMode::Repeat,
        peniko::Extend::Reflect => sk::TileMode::Mirror,
    }
}

/// Map `peniko` interpolation color spaces to Skia's gradient interpolation settings.
fn gradient_shader_cs_from_cs_tag(
    color_space: ColorSpaceTag,
) -> sk::gradient_shader::interpolation::ColorSpace {
    use sk::gradient_shader::interpolation::ColorSpace as SkCs;

    match color_space {
        ColorSpaceTag::Srgb => SkCs::SRGB,
        ColorSpaceTag::LinearSrgb => SkCs::SRGBLinear,
        ColorSpaceTag::Lab => SkCs::Lab,
        ColorSpaceTag::Lch => SkCs::LCH,
        ColorSpaceTag::Hsl => SkCs::HSL,
        ColorSpaceTag::Hwb => SkCs::HWB,
        ColorSpaceTag::Oklab => SkCs::OKLab,
        ColorSpaceTag::Oklch => SkCs::OKLCH,
        ColorSpaceTag::DisplayP3 => SkCs::DisplayP3,
        ColorSpaceTag::A98Rgb => SkCs::A98RGB,
        ColorSpaceTag::ProphotoRgb => SkCs::ProphotoRGB,
        ColorSpaceTag::Rec2020 => SkCs::Rec2020,
        _ => SkCs::SRGB,
    }
}

/// Map `peniko` hue interpolation direction to Skia's gradient hue method.
fn gradient_shader_hue_method_from_hue_direction(
    direction: HueDirection,
) -> sk::gradient_shader::interpolation::HueMethod {
    use sk::gradient_shader::interpolation::HueMethod as SkHue;

    match direction {
        HueDirection::Shorter => SkHue::Shorter,
        HueDirection::Longer => SkHue::Longer,
        HueDirection::Increasing => SkHue::Increasing,
        HueDirection::Decreasing => SkHue::Decreasing,
        _ => SkHue::Shorter,
    }
}

/// Convert a `peniko` color to Skia's packed 8-bit color representation.
fn color_to_sk_color(color: peniko::Color) -> sk::Color {
    let rgba = color.to_rgba8();
    sk::Color::from_argb(rgba.a, rgba.r, rgba.g, rgba.b)
}

/// Convert a `peniko` color to Skia's floating-point color representation.
fn color_to_sk_color4f(color: peniko::Color) -> sk::Color4f {
    let comps = color.components;
    sk::Color4f::new(comps[0], comps[1], comps[2], comps[3])
}

/// Build a configured Skia paint from a semantic `peniko` brush description.
///
/// This is the main translation point from `imaging` brush semantics into Skia shaders, colors,
/// image sampling, and opacity handling.
fn brush_to_paint(
    brush: BrushRef<'_>,
    opacity: f32,
    paint_xf: Affine,
    image_cache: Option<&ImageCacheHandle>,
) -> Option<sk::Paint> {
    let mut paint = sk::Paint::default();
    paint.set_anti_alias(true);
    let alpha_scale = opacity.clamp(0.0, 1.0);

    match brush {
        BrushRef::Solid(color) => {
            // Use float color to avoid quantizing alpha (important for Porter-Duff ops like XOR).
            let comps = color.components;
            let c = sk::Color4f::new(comps[0], comps[1], comps[2], comps[3] * alpha_scale);
            paint.set_color4f(c, None);
        }
        BrushRef::Gradient(grad) => {
            let stops = grad.stops.as_ref();
            if stops.is_empty() {
                paint.set_color(sk::Color::TRANSPARENT);
                return Some(paint);
            }

            let mut colors: Vec<sk::Color4f> = Vec::with_capacity(stops.len());
            let mut pos: Vec<f32> = Vec::with_capacity(stops.len());

            for s in stops {
                let color = s.color.to_alpha_color::<Srgb>().multiply_alpha(alpha_scale);
                colors.push(sk::Color4f::new(
                    color.components[0],
                    color.components[1],
                    color.components[2],
                    color.components[3],
                ));
                pos.push(s.offset.clamp(0.0, 1.0));
            }

            let tile_mode = tile_mode_from_extend(grad.extend);
            let local = affine_to_matrix(paint_xf);

            let interpolation = sk::gradient_shader::Interpolation {
                color_space: gradient_shader_cs_from_cs_tag(grad.interpolation_cs),
                in_premul: match grad.interpolation_alpha_space {
                    InterpolationAlphaSpace::Premultiplied => {
                        sk::gradient_shader::interpolation::InPremul::Yes
                    }
                    InterpolationAlphaSpace::Unpremultiplied => {
                        sk::gradient_shader::interpolation::InPremul::No
                    }
                },
                hue_method: gradient_shader_hue_method_from_hue_direction(grad.hue_direction),
            };

            match &grad.kind {
                peniko::GradientKind::Linear(line) => {
                    let p0 = sk::Point::new(f64_to_f32(line.start.x), f64_to_f32(line.start.y));
                    let p1 = sk::Point::new(f64_to_f32(line.end.x), f64_to_f32(line.end.y));
                    if let Some(shader) = sk::Shader::linear_gradient_with_interpolation(
                        (p0, p1),
                        (&colors[..], None),
                        &pos[..],
                        tile_mode,
                        interpolation,
                        Some(&local),
                    ) {
                        paint.set_shader(shader);
                    }
                }
                peniko::GradientKind::Radial(rad) => {
                    let start_center = sk::Point::new(
                        f64_to_f32(rad.start_center.x),
                        f64_to_f32(rad.start_center.y),
                    );
                    let start_radius = rad.start_radius;
                    let end_center =
                        sk::Point::new(f64_to_f32(rad.end_center.x), f64_to_f32(rad.end_center.y));
                    let end_radius = rad.end_radius;

                    if let Some(shader) = sk::Shader::two_point_conical_gradient_with_interpolation(
                        (start_center, start_radius),
                        (end_center, end_radius),
                        (&colors[..], None),
                        &pos[..],
                        tile_mode,
                        interpolation,
                        Some(&local),
                    ) {
                        paint.set_shader(shader);
                    }
                }
                peniko::GradientKind::Sweep(sweep) => {
                    let center =
                        sk::Point::new(f64_to_f32(sweep.center.x), f64_to_f32(sweep.center.y));
                    // `peniko` uses radians; Skia uses degrees for sweep gradient angles.
                    let start = {
                        let rad = sweep.start_angle;
                        rad.to_degrees()
                    };
                    let end = {
                        let rad = sweep.end_angle;
                        rad.to_degrees()
                    };
                    if let Some(shader) = sk::Shader::sweep_gradient_with_interpolation(
                        center,
                        (&colors[..], None),
                        Some(&pos[..]),
                        tile_mode,
                        Some((start, end)),
                        interpolation,
                        Some(&local),
                    ) {
                        paint.set_shader(shader);
                    }
                }
            }

            if paint.shader().is_none()
                && let Some(last_stop) = stops.last()
            {
                let color = last_stop
                    .color
                    .to_alpha_color::<Srgb>()
                    .multiply_alpha(alpha_scale);
                paint.set_color(color_to_sk_color(color));
            }
        }
        BrushRef::Image(image_brush) => {
            let image = skia_image_from_peniko(image_brush.image, image_cache)?;
            let shader = image.to_shader(
                Some((
                    tile_mode_from_extend(image_brush.sampler.x_extend),
                    tile_mode_from_extend(image_brush.sampler.y_extend),
                )),
                sampling_options_from_quality(image_brush.sampler.quality),
                Some(&affine_to_matrix(paint_xf)),
            )?;
            paint.set_shader(shader);
            paint.set_alpha_f((image_brush.sampler.alpha * alpha_scale).clamp(0.0, 1.0));
        }
    }

    Some(paint)
}

/// Convert a `peniko` image payload into a raster Skia image when its format is supported.
fn skia_image_from_peniko(
    image: &peniko::ImageData,
    image_cache: Option<&ImageCacheHandle>,
) -> Option<sk::Image> {
    if let Some(image_cache) = image_cache {
        return image_cache.borrow_mut().get_or_create(image);
    }
    make_skia_image_from_peniko(image)
}

fn make_skia_image_from_peniko(image: &peniko::ImageData) -> Option<sk::Image> {
    let color_type = match image.format {
        ImageFormat::Rgba8 => sk::ColorType::RGBA8888,
        ImageFormat::Bgra8 => sk::ColorType::BGRA8888,
        _ => return None,
    };
    let alpha_type = match image.alpha_type {
        ImageAlphaType::Alpha => sk::AlphaType::Unpremul,
        ImageAlphaType::AlphaPremultiplied => sk::AlphaType::Premul,
    };
    let info = sk::ImageInfo::new(
        (
            i32::try_from(image.width).ok()?,
            i32::try_from(image.height).ok()?,
        ),
        color_type,
        alpha_type,
        None,
    );
    let row_bytes = image.format.size_in_bytes(image.width, 1)?;
    sk::images::raster_from_data(&info, sk::Data::new_copy(image.data.data()), row_bytes)
}

/// Map `peniko` image quality hints to Skia sampling configuration.
fn sampling_options_from_quality(quality: ImageQuality) -> sk::SamplingOptions {
    match quality {
        ImageQuality::Low => sk::SamplingOptions::from(sk::FilterMode::Nearest),
        ImageQuality::Medium => sk::SamplingOptions::from(sk::FilterMode::Linear),
        ImageQuality::High => sk::SamplingOptions::from(sk::CubicResampler::mitchell()),
    }
}

/// Apply `kurbo` stroke settings to a Skia paint before stroke drawing.
fn apply_stroke_style(paint: &mut sk::Paint, style: &kurbo::Stroke) {
    paint.set_style(sk::PaintStyle::Stroke);
    paint.set_stroke_width(f64_to_f32(style.width));
    paint.set_stroke_miter(f64_to_f32(style.miter_limit));
    paint.set_stroke_join(match style.join {
        kurbo::Join::Bevel => sk::PaintJoin::Bevel,
        kurbo::Join::Miter => sk::PaintJoin::Miter,
        kurbo::Join::Round => sk::PaintJoin::Round,
    });
    let cap = match style.start_cap {
        kurbo::Cap::Butt => sk::PaintCap::Butt,
        kurbo::Cap::Square => sk::PaintCap::Square,
        kurbo::Cap::Round => sk::PaintCap::Round,
    };
    paint.set_stroke_cap(cap);
    if !style.dash_pattern.is_empty() {
        let intervals: Vec<f32> = style.dash_pattern.iter().map(|v| f64_to_f32(*v)).collect();
        if let Some(effect) =
            sk::PathEffect::dash(intervals.as_slice(), f64_to_f32(style.dash_offset))
        {
            paint.set_path_effect(effect);
        }
    }
}

/// Map semantic `peniko` blend and composite modes to Skia blend modes.
fn map_blend_mode(mode: &peniko::BlendMode) -> sk::BlendMode {
    use peniko::{Compose, Mix};

    match (mode.mix, mode.compose) {
        (_, Compose::Clear) => sk::BlendMode::Clear,
        (_, Compose::Copy) => sk::BlendMode::Src,
        (_, Compose::Dest) => sk::BlendMode::Dst,
        (_, Compose::SrcOver) => match mode.mix {
            Mix::Normal => sk::BlendMode::SrcOver,
            Mix::Multiply => sk::BlendMode::Multiply,
            Mix::Screen => sk::BlendMode::Screen,
            Mix::Overlay => sk::BlendMode::Overlay,
            Mix::Darken => sk::BlendMode::Darken,
            Mix::Lighten => sk::BlendMode::Lighten,
            Mix::ColorDodge => sk::BlendMode::ColorDodge,
            Mix::ColorBurn => sk::BlendMode::ColorBurn,
            Mix::HardLight => sk::BlendMode::HardLight,
            Mix::SoftLight => sk::BlendMode::SoftLight,
            Mix::Difference => sk::BlendMode::Difference,
            Mix::Exclusion => sk::BlendMode::Exclusion,
            Mix::Hue => sk::BlendMode::Hue,
            Mix::Saturation => sk::BlendMode::Saturation,
            Mix::Color => sk::BlendMode::Color,
            Mix::Luminosity => sk::BlendMode::Luminosity,
        },
        (_, Compose::DestOver) => sk::BlendMode::DstOver,
        (_, Compose::SrcIn) => sk::BlendMode::SrcIn,
        (_, Compose::DestIn) => sk::BlendMode::DstIn,
        (_, Compose::SrcOut) => sk::BlendMode::SrcOut,
        (_, Compose::DestOut) => sk::BlendMode::DstOut,
        (_, Compose::SrcAtop) => sk::BlendMode::SrcATop,
        (_, Compose::DestAtop) => sk::BlendMode::DstATop,
        (_, Compose::Xor) => sk::BlendMode::Xor,
        (_, Compose::Plus) => sk::BlendMode::Plus,
        (_, Compose::PlusLighter) => sk::BlendMode::Plus,
    }
}

/// Build the Skia image-filter chain used for group and layer effects.
fn build_filter_chain(filters: &[Filter]) -> Option<sk::ImageFilter> {
    use sk::image_filters;

    let mut current: Option<sk::ImageFilter> = None;
    for f in filters {
        current = Some(match *f {
            Filter::Flood { color } => {
                let shader = sk::shaders::color(color_to_sk_color(color));
                // Leaf filter: ignores any existing input chain.
                image_filters::shader(shader, None)?
            }
            Filter::Blur {
                std_deviation_x,
                std_deviation_y,
            } => image_filters::blur((std_deviation_x, std_deviation_y), None, current, None)?,
            Filter::DropShadow {
                dx,
                dy,
                std_deviation_x,
                std_deviation_y,
                color,
            } => image_filters::drop_shadow(
                (dx, dy),
                (std_deviation_x, std_deviation_y),
                color_to_sk_color4f(color),
                None,
                current,
                None,
            )?,
            Filter::Offset { dx, dy } => image_filters::offset((dx, dy), current, None)?,
        });
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use imaging::{GroupRef, MaskMode, Painter};
    use kurbo::Rect;
    use peniko::{Brush, Color};

    fn masked_scene(mode: MaskMode) -> Scene {
        let mask = Painter::<Scene>::record_mask(mode, |mask| {
            mask.fill(
                Rect::new(8.0, 8.0, 56.0, 56.0),
                Color::from_rgba8(255, 255, 255, 160),
            )
            .draw();
        });

        let mut scene = Scene::new();
        {
            let mut painter = Painter::new(&mut scene);
            painter.with_group(GroupRef::new().with_mask(mask.as_ref()), |content| {
                content
                    .fill(
                        Rect::new(0.0, 0.0, 64.0, 64.0),
                        Color::from_rgb8(0x2a, 0x6f, 0xdb),
                    )
                    .draw();
            });
        }

        scene
    }

    #[test]
    fn read_image_reads_native_picture() {
        let mut sink = SkPictureRecorderSink::new(Rect::new(0.0, 0.0, 32.0, 32.0));
        let paint = Brush::Solid(Color::from_rgb8(0x22, 0x66, 0xaa));
        {
            let mut painter = Painter::new(&mut sink);
            painter.fill_rect(Rect::new(0.0, 0.0, 32.0, 32.0), &paint);
        }

        let picture = sink.finish_picture().unwrap();
        let mut renderer = SkiaRenderer::new(32, 32);
        renderer.reset().unwrap();
        renderer.render_picture(&picture).unwrap();
        let image = renderer.read_image().unwrap();

        assert_eq!(image.width, 32);
        assert_eq!(image.height, 32);
        assert_eq!(&image.data.as_ref()[..4], &[0x22, 0x66, 0xaa, 0xff]);
    }

    #[test]
    fn render_scene_replays_masked_content_without_cached_mask_artifacts() {
        let scene = masked_scene(MaskMode::Alpha);
        let mut renderer = SkiaRenderer::new(64, 64);

        renderer.reset().unwrap();
        renderer.render_scene(&scene).unwrap();
        renderer.read_image().unwrap();

        renderer.render_scene(&scene).unwrap();
        renderer.read_image().unwrap();
    }

    #[test]
    fn clear_cached_masks_clears_realized_mask_cache() {
        let scene = masked_scene(MaskMode::Luminance);
        let mut renderer = SkiaRenderer::new(64, 64);

        renderer.reset().unwrap();
        renderer.render_scene(&scene).unwrap();
        renderer.read_image().unwrap();

        renderer.clear_cached_masks();

        renderer.reset().unwrap();
        renderer.render_scene(&scene).unwrap();
        renderer.read_image().unwrap();
    }

    #[test]
    fn changing_tolerance_keeps_masked_rendering_working() {
        let scene = masked_scene(MaskMode::Alpha);
        let mut renderer = SkiaRenderer::new(64, 64);

        renderer.reset().unwrap();
        renderer.render_scene(&scene).unwrap();
        renderer.read_image().unwrap();

        renderer.set_tolerance(0.25);
        renderer.reset().unwrap();
        renderer.render_scene(&scene).unwrap();
        renderer.read_image().unwrap();
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn maps_supported_wgpu_texture_formats_for_wrapped_surfaces() {
        assert_eq!(
            color_type_for_wgpu_texture_format(wgpu::TextureFormat::Rgba8Unorm).unwrap(),
            sk::ColorType::RGBA8888
        );
        assert_eq!(
            color_type_for_wgpu_texture_format(wgpu::TextureFormat::Rgba8UnormSrgb).unwrap(),
            sk::ColorType::RGBA8888
        );
        assert_eq!(
            color_type_for_wgpu_texture_format(wgpu::TextureFormat::Bgra8Unorm).unwrap(),
            sk::ColorType::BGRA8888
        );
        assert_eq!(
            color_type_for_wgpu_texture_format(wgpu::TextureFormat::Bgra8UnormSrgb).unwrap(),
            sk::ColorType::BGRA8888
        );
        assert_eq!(
            color_type_for_wgpu_texture_format(wgpu::TextureFormat::Rgb10a2Unorm).unwrap(),
            sk::ColorType::RGBA1010102
        );
        assert_eq!(
            color_type_for_wgpu_texture_format(wgpu::TextureFormat::Rgba16Unorm).unwrap(),
            sk::ColorType::R16G16B16A16UNorm
        );
        assert_eq!(
            color_type_for_wgpu_texture_format(wgpu::TextureFormat::Rgba16Float).unwrap(),
            sk::ColorType::RGBAF16
        );
    }
}
