// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Skia backend for `imaging`.
//!
//! This crate connects the semantic `imaging` command stream to Skia.
//!
//! At a high level, there are three ways to use it:
//!
//! - [`SkiaRenderer`] renders through Skia Ganesh into a GPU-backed surface.
//! - [`SkiaCpuRenderState`] replays through Skia's raster backend into any raster surface.
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
//! [`SkiaCpuRenderState`] is the lower-level CPU replay engine. It keeps reusable raster-side
//! state such as path tolerance while callers provide the destination
//! [`skia_safe::Surface`]. Use it when you want to render into caller-owned CPU memory via
//! `wrap_pixels` or when a host application owns raster-surface allocation.
//!
//! [`SkiaRenderer`] is the GPU path. It owns a Ganesh context and an offscreen GPU render surface
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
//! Use [`SkiaCpuRenderState`] when:
//!
//! - you want to render into a caller-owned raster surface
//! - you want to reuse CPU-side caches across multiple wrapped surfaces
//! - your application already manages pixel storage and surface lifetime
//!
//! Use [`SkiaRenderer::new`] or [`SkiaRenderer::try_new`] when:
//!
//! - you want an owned offscreen GPU renderer
//! - `imaging_skia` should choose and own the underlying Ganesh backend
//! - you want to render scenes or pictures and optionally inspect the GPU surface afterward
//!
//! Use the backend-specific `SkiaRenderer` constructors when:
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
//! Record commands into [`imaging::record::Scene`], then hand the scene to [`SkiaRenderer`].
//!
//! ```no_run
//! use imaging::{Painter, record};
//! use imaging_skia::SkiaRenderer;
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
//!     let mut renderer = SkiaRenderer::new(128, 128);
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
//! [`SkiaCpuRenderState`].
//!
//! ```no_run
//! use imaging::{Painter, record};
//! use imaging_skia::SkiaCpuRenderState;
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
//!         None,
//!     );
//!     let mut surface = sk::surfaces::wrap_pixels(&info, pixels.as_mut_slice(), Some(128 * 4), None)
//!         .expect("wrap raster pixels");
//!     let mut state = SkiaCpuRenderState::new();
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
//! If you already have a recorded picture, hand it directly to [`SkiaRenderer`].
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_skia::{SkPictureRecorderSink, SkiaRenderer};
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
//!     let mut renderer = SkiaRenderer::new(128, 128);
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
use imaging::{
    BeginFrame, CpuBufferFormat, CpuBufferTarget, Filter, GeometryRef, PaintSink, RenderCore,
    RenderOutput, Renderer, TargetRenderer,
    record::{Scene, ValidateError, replay},
};
use kurbo::{Affine, Shape as _};
use peniko::color::{ColorSpaceTag, HueDirection, Srgb};
use peniko::{BrushRef, ImageAlphaType, ImageFormat, ImageQuality, InterpolationAlphaSpace};
use skia_safe as sk;
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

/// Ganesh renderer that executes `imaging` commands into a GPU-backed Skia surface.
#[derive(Debug)]
pub struct SkiaRenderer {
    backend: GaneshBackend,
    surface: sk::Surface,
    tolerance: f64,
    image_cache: ImageCacheHandle,
    mask_cache: Rc<RefCell<MaskImageCache>>,
    retained_image_cache: Rc<RefCell<RetainedImageCache>>,
    frame_active: bool,
    #[cfg(feature = "wgpu")]
    wgpu_backend_keepalive: Option<WgpuDeviceQueueKeepalive>,
    #[cfg(feature = "wgpu")]
    wgpu_target_keepalive: Option<WgpuTextureHandle>,
}

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

impl SkiaRenderer {
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

    /// Build a renderer from an already-initialized Ganesh backend and wrapped target surface.
    ///
    /// Backend modules use this to share the same renderer initialization path without reaching
    /// into `SkiaRenderer`'s private fields directly.
    pub(crate) fn from_backend_surface(backend: GaneshBackend, surface: sk::Surface) -> Self {
        Self {
            backend,
            surface,
            tolerance: 0.1,
            image_cache: Rc::new(RefCell::new(ImageCache::default())),
            mask_cache: Rc::new(RefCell::new(MaskImageCache::default())),
            retained_image_cache: Rc::new(RefCell::new(RetainedImageCache::default())),
            frame_active: false,
            #[cfg(feature = "wgpu")]
            wgpu_backend_keepalive: None,
            #[cfg(feature = "wgpu")]
            wgpu_target_keepalive: None,
        }
    }

    /// Set the geometric flattening tolerance used for path conversion.
    ///
    /// Lower values preserve curve fidelity more aggressively; higher values can reduce path
    /// complexity when rendering highly curved geometry.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
        self.clear_cached_masks();
    }

    /// Drop any realized native mask images cached by the renderer.
    pub fn clear_cached_masks(&mut self) {
        self.image_cache.borrow_mut().clear();
        self.mask_cache.borrow_mut().clear();
        self.retained_image_cache.borrow_mut().clear();
    }

    fn begin_frame(&mut self) {
        if self.frame_active {
            return;
        }
        self.retained_image_cache.borrow_mut().flip_mark();
        self.frame_active = true;
    }

    fn finish_frame(&mut self) {
        if !self.frame_active {
            return;
        }
        self.retained_image_cache.borrow_mut().evict_unmarked();
        self.frame_active = false;
    }

    #[cfg(feature = "wgpu")]
    fn set_wgpu_backend_keepalive(&mut self, keepalive: WgpuDeviceQueueKeepalive) {
        self.wgpu_backend_keepalive = Some(keepalive);
    }

    #[cfg(feature = "wgpu")]
    fn set_wgpu_target_keepalive(&mut self, keepalive: WgpuTextureHandle) {
        self.wgpu_target_keepalive = Some(keepalive);
    }
}

#[cfg(feature = "wgpu")]
impl SkiaRenderer {
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
    ) -> Result<Self, Error> {
        #[allow(
            unused_variables,
            reason = "Platform cfgs may compile out all interop branches."
        )]
        let keepalive = WgpuDeviceQueueKeepalive {
            device: device.clone(),
            queue: queue.clone(),
        };

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let device = unsafe {
                device
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuContext("missing Metal device"))?
            };
            let queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuContext("missing Metal queue"))?
            };
            let device = device.raw_device().as_ptr() as *mut c_void;
            let command_queue = queue.as_raw().lock().as_ptr() as *mut c_void;
            let mut renderer = unsafe {
                Self::try_new_metal_from_raw_pointers_without_texture(
                    width,
                    height,
                    device,
                    command_queue,
                )
            }?;
            renderer.set_wgpu_backend_keepalive(keepalive);
            return Ok(renderer);
        }

        #[cfg(all(feature = "vulkan", not(any(target_os = "macos", target_os = "ios"))))]
        {
            use ash::vk::Handle as _;

            let device = unsafe {
                device
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuContext("missing Vulkan device"))?
            };
            let queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuContext("missing Vulkan queue"))?
            };
            let instance = device.shared_instance().raw_instance().handle();
            let physical_device = device.raw_physical_device();
            let raw_device = device.raw_device().handle();
            let raw_queue = queue.as_raw();
            let queue_family_index = device.queue_family_index();
            let mut renderer = unsafe {
                Self::try_new_vulkan_from_raw_handles(
                    width,
                    height,
                    instance,
                    physical_device,
                    raw_device,
                    raw_queue,
                    queue_family_index,
                )
            }?;
            renderer.set_wgpu_backend_keepalive(keepalive);
            return Ok(renderer);
        }

        #[allow(
            unreachable_code,
            unused_variables,
            reason = "Platform and feature cfgs intentionally leave unsupported backend paths empty."
        )]
        Err(Error::UnsupportedGpuBackend)
    }

    /// Create a renderer that targets an owned `wgpu` texture.
    ///
    /// This is the high-level interop entry point when the caller already works in `wgpu` terms and
    /// wants `imaging_skia` to attach to that device/queue/texture tuple while taking ownership of
    /// the render target handle. Use [`Self::try_new_from_wgpu_device`] instead when you only want
    /// to share the backend and let `imaging_skia` allocate its own offscreen target surface.
    pub fn try_new_from_wgpu_texture(
        texture_format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: wgpu::Texture,
    ) -> Result<Self, Error> {
        initialize_texture_for_wgpu(device, queue, &texture);
        let texture_keepalive = WgpuTextureHandle::new(texture);
        let texture_size = texture_keepalive.texture.size();
        let width = u16::try_from(texture_size.width)
            .map_err(|_| Error::Internal("texture width exceeds skia limit"))?;
        let height = u16::try_from(texture_size.height)
            .map_err(|_| Error::Internal("texture height exceeds skia limit"))?;

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
            let texture = unsafe {
                texture_keepalive
                    .texture
                    .as_hal::<wgpu::hal::api::Metal>()
                    .ok_or(Error::CreateGpuSurface)?
            };
            let raw_device = hal_device.raw_device().as_ptr() as *mut c_void;
            let raw_command_queue = hal_queue.as_raw().lock().as_ptr() as *mut c_void;
            let texture = unsafe { texture.raw_handle() }.as_ptr() as *mut c_void;
            let mut renderer = unsafe {
                Self::try_new_metal_from_raw_pointers(
                    width,
                    height,
                    texture_format,
                    raw_device,
                    raw_command_queue,
                    texture,
                )
            }?;
            renderer.set_wgpu_backend_keepalive(WgpuDeviceQueueKeepalive {
                device: device.clone(),
                queue: queue.clone(),
            });
            renderer.set_wgpu_target_keepalive(texture_keepalive);
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
            let texture = unsafe {
                texture_keepalive
                    .texture
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .ok_or(Error::CreateGpuSurface)?
            };
            let instance = hal_device.shared_instance().raw_instance().handle();
            let physical_device = hal_device.raw_physical_device();
            let raw_device = hal_device.raw_device().handle();
            let raw_queue = hal_queue.as_raw();
            let queue_family_index = hal_device.queue_family_index();
            let raw_image = unsafe { texture.raw_handle() };
            let image_layout = ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            let image_usage_flags = ash::vk::ImageUsageFlags::COLOR_ATTACHMENT;
            let mut renderer = unsafe {
                Self::try_new_vulkan_from_raw_handles_and_texture(
                    width,
                    height,
                    texture_format,
                    instance,
                    physical_device,
                    raw_device,
                    raw_queue,
                    queue_family_index,
                    raw_image,
                    image_layout,
                    image_usage_flags,
                    1,
                )
            }?;
            renderer.set_wgpu_backend_keepalive(WgpuDeviceQueueKeepalive {
                device: device.clone(),
                queue: queue.clone(),
            });
            renderer.set_wgpu_target_keepalive(texture_keepalive);
            return Ok(renderer);
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
impl SkiaRenderer {
    /// Retarget the renderer to a different owned `wgpu` texture on the same backend bridge.
    ///
    /// This is the `wgpu`-level companion to the explicit Metal and Vulkan replacement methods and
    /// is intended for integrations that manage resize or swapchain churn entirely through `wgpu`.
    pub fn replace_wgpu_texture(
        &mut self,
        texture_format: wgpu::TextureFormat,
        texture: wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Error> {
        initialize_texture_for_wgpu(device, queue, &texture);
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
            unsafe { self.replace_metal_texture_raw(width, height, texture_format, texture) }?;
            self.set_wgpu_target_keepalive(texture_keepalive);
            return Ok(());
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
            let queue_family_index = match &self.backend {
                GaneshBackend::Vulkan(backend) => backend.queue_family_index(),
                _ => sk::gpu::vk::QUEUE_FAMILY_IGNORED,
            };
            unsafe {
                self.replace_vulkan_texture(
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
            self.set_wgpu_target_keepalive(texture_keepalive);
            return Ok(());
        }

        #[allow(
            unreachable_code,
            unused_variables,
            reason = "Platform and feature cfgs intentionally leave unsupported backend paths empty."
        )]
        Err(Error::UnsupportedGpuBackend)
    }

    /// Borrow the live owned `wgpu` texture when the renderer is targeting one.
    pub fn wgpu_texture(&self) -> Option<&wgpu::Texture> {
        self.wgpu_target_keepalive
            .as_ref()
            .map(|target| &target.texture)
    }

    /// Borrow the live `wgpu::TextureView` for the currently owned texture.
    pub fn wgpu_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.wgpu_target_keepalive
            .as_ref()
            .map(|target| &target.view)
    }
}

impl SkiaRenderer {
    /// Reset canvas state before starting a new frame on the wrapped GPU surface.
    ///
    /// Rendering methods call this internally so each frame starts from a known transform, clip,
    /// and clear state regardless of what the previous frame left behind.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.backend.ensure_current()?;
        self.finish_frame();
        let canvas = self.surface.canvas();
        canvas.restore_to_count(1);
        canvas.reset_matrix();
        canvas.clear(sk::Color::TRANSPARENT);
        Ok(())
    }

    fn canvas_sink(&mut self) -> SkCanvasSink<'_> {
        let mut sink = SkCanvasSink::new_with_mask_cache(
            self.surface.canvas(),
            Some(Rc::clone(&self.image_cache)),
            Rc::clone(&self.mask_cache),
            Rc::clone(&self.retained_image_cache),
        );
        sink.set_tolerance(self.tolerance);
        sink
    }

    fn flush(&mut self) {
        self.backend.flush_surface(&mut self.surface);
    }

    /// Stream `imaging` commands directly into the current GPU surface.
    ///
    /// This is the low-level sink-oriented API for callers that want to drive Skia directly
    /// without first building an intermediate `Scene`.
    pub fn with_canvas_sink<R>(
        &mut self,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        self.backend.ensure_current()?;
        self.begin_frame();
        let mut sink = self.canvas_sink();
        let out = f(&mut sink);
        let finish_result = sink.finish();
        finish_result?;
        self.flush();
        Ok(out)
    }

    /// Replay an `imaging` scene into the current GPU surface.
    ///
    /// This is the main path for rendering the semantic `imaging::record::Scene` representation
    /// through Skia/Ganesh.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<(), Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        self.backend.ensure_current()?;
        self.begin_frame();
        let mut sink = self.canvas_sink();
        replay(scene, &mut sink);
        let finish_result = sink.finish();
        finish_result?;
        self.flush();
        Ok(())
    }

    /// Draw an existing native Skia picture into the current GPU surface.
    ///
    /// This is useful when higher layers already hold a recorded `SkPicture` and want to reuse the
    /// same renderer and readback path as scene-based rendering.
    pub fn render_picture(&mut self, picture: &sk::Picture) -> Result<(), Error> {
        self.backend.ensure_current()?;
        self.begin_frame();
        self.surface.canvas().draw_picture(picture, None, None);
        self.flush();
        Ok(())
    }

    /// Borrow the live GPU-backed `skia_safe::Surface`.
    ///
    /// Use this when you need direct Skia drawing access instead of going through scene replay.
    pub fn surface(&mut self) -> &mut sk::Surface {
        &mut self.surface
    }

    /// Snapshot the current GPU surface as a Skia image.
    ///
    /// This flushes pending work first so the returned image reflects the renderer's latest output.
    pub fn image_snapshot(&mut self) -> sk::Image {
        let _ = self.backend.ensure_current();
        self.flush();
        self.surface.image_snapshot()
    }

    /// Expose Skia's backend texture for the current surface when the backend supports it.
    ///
    /// This is primarily for advanced interop or inspection code that needs the underlying Ganesh
    /// texture handle after rendering.
    pub fn backend_texture(&mut self) -> Option<sk::gpu::BackendTexture> {
        let _ = self.backend.ensure_current();
        self.flush();
        sk::gpu::surfaces::get_backend_texture(
            &mut self.surface,
            sk::surface::BackendHandleAccess::FlushRead,
        )
    }

    /// Read back the current GPU surface into an unpremultiplied RGBA8 image.
    ///
    /// Rendering methods funnel through this helper after flushing work to the active backend.
    pub fn read_image(&mut self) -> Result<peniko::ImageData, Error> {
        self.backend.ensure_current()?;
        self.flush();
        let info = sk::ImageInfo::new(
            (self.surface.width(), self.surface.height()),
            sk::ColorType::RGBA8888,
            sk::AlphaType::Unpremul,
            None,
        );
        let mut bytes =
            vec![0_u8; (self.surface.width() as usize) * (self.surface.height() as usize) * 4];
        let ok = self.surface.read_pixels(
            &info,
            bytes.as_mut_slice(),
            (4 * self.surface.width()) as usize,
            (0, 0),
        );
        if !ok {
            return Err(Error::Internal("read_pixels failed"));
        }
        Ok(peniko::ImageData {
            data: peniko::Blob::new(Arc::new(bytes)),
            format: ImageFormat::Rgba8,
            width: self.surface.width() as u32,
            height: self.surface.height() as u32,
            alpha_type: ImageAlphaType::Alpha,
        })
    }
}

impl RenderCore for SkiaRenderer {
    fn render(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink)) {
        self.with_canvas_sink(|sink| f(sink))
            .expect("render into imaging_skia gpu canvas sink");
    }

    fn finish(&mut self) {
        self.finish_frame();
        self.backend
            .ensure_current()
            .expect("make imaging_skia gpu backend current");
        self.flush();
    }

    fn readback(&mut self) -> Option<RenderOutput> {
        self.read_image().ok().map(RenderOutput::Image)
    }

    fn debug_info(&self) -> String {
        "name: Skia\ninfo: imaging_skia::SkiaRenderer".to_string()
    }
}

impl Renderer for SkiaRenderer {
    type Target = peniko::ImageData;

    #[allow(
        clippy::cast_possible_truncation,
        reason = "Frame sizes are converted to whole pixels and then checked against `u16`."
    )]
    fn set_size(&mut self, frame: BeginFrame) {
        let width = u16::try_from(frame.size.width as u32).expect("skia width out of range");
        let height = u16::try_from(frame.size.height as u32).expect("skia height out of range");
        if self.surface.width() != i32::from(width) || self.surface.height() != i32::from(height) {
            *self = Self::try_new(width, height).expect("recreate imaging_skia renderer");
        }
    }

    fn reset(&mut self) {
        Self::reset(self).expect("reset imaging_skia renderer");
    }

    fn read_target(&mut self) -> Option<Self::Target> {
        self.read_image().ok()
    }
}

/// Reusable CPU raster replay state for Skia-backed rendering.
///
/// This type keeps renderer-side state such as tolerance while callers provide the destination
/// raster surface for each render.
#[derive(Debug)]
pub struct SkiaCpuRenderState {
    tolerance: f64,
    mask_cache: Rc<RefCell<MaskImageCache>>,
    retained_image_cache: Rc<RefCell<RetainedImageCache>>,
    frame_active: bool,
}

impl Default for SkiaCpuRenderState {
    fn default() -> Self {
        Self::new()
    }
}

impl SkiaCpuRenderState {
    /// Create reusable CPU raster replay state.
    pub fn new() -> Self {
        Self {
            tolerance: 0.1,
            mask_cache: Rc::new(RefCell::new(MaskImageCache::default())),
            retained_image_cache: Rc::new(RefCell::new(RetainedImageCache::default())),
            frame_active: false,
        }
    }

    /// Set the geometric flattening tolerance used for path conversion.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
        self.clear_cached_masks();
    }

    /// Drop any realized native mask images cached by the renderer state.
    pub fn clear_cached_masks(&mut self) {
        self.mask_cache.borrow_mut().clear();
        self.retained_image_cache.borrow_mut().clear();
    }

    fn begin_frame(&mut self) {
        if self.frame_active {
            return;
        }
        self.retained_image_cache.borrow_mut().flip_mark();
        self.frame_active = true;
    }

    fn finish_frame(&mut self) {
        if !self.frame_active {
            return;
        }
        self.retained_image_cache.borrow_mut().evict_unmarked();
        self.frame_active = false;
    }

    /// Create a short-lived renderer view bound to a caller-provided raster surface.
    pub fn bind<'a>(&'a mut self, surface: &'a mut sk::Surface) -> SkiaCpuRendererRef<'a> {
        SkiaCpuRendererRef {
            state: self,
            surface,
        }
    }

    /// Reset canvas state before starting a new frame on the provided raster surface.
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
    pub fn with_canvas_sink<R>(
        &mut self,
        surface: &mut sk::Surface,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        self.begin_frame();
        let mut sink = self.canvas_sink(surface);
        let out = f(&mut sink);
        let finish_result = sink.finish();
        finish_result?;
        Ok(out)
    }

    /// Replay an `imaging` scene through the raster backend into the provided surface.
    pub fn render_scene(&mut self, surface: &mut sk::Surface, scene: &Scene) -> Result<(), Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        self.with_canvas_sink(surface, |sink| replay(scene, sink))
            .map(|_| ())
    }

    /// Draw a native Skia picture through the raster backend into the provided surface.
    pub fn render_picture(
        &mut self,
        surface: &mut sk::Surface,
        picture: &sk::Picture,
    ) -> Result<(), Error> {
        self.begin_frame();
        surface.canvas().draw_picture(picture, None, None);
        Ok(())
    }

    /// Read back the current raster surface into an unpremultiplied RGBA8 image.
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

/// Owned CPU raster renderer that allocates and retains its own raster surface.
#[derive(Debug)]
pub struct SkiaCpuRenderer {
    state: SkiaCpuRenderState,
    surface: sk::Surface,
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
            state: SkiaCpuRenderState::new(),
            surface,
        }
    }

    /// Set the geometric flattening tolerance used for path conversion.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.state.set_tolerance(tolerance);
    }

    /// Drop any realized mask artifacts cached by the renderer.
    pub fn clear_cached_masks(&mut self) {
        self.state.clear_cached_masks();
    }

    /// Borrow the live raster `skia_safe::Surface`.
    pub fn surface(&mut self) -> &mut sk::Surface {
        &mut self.surface
    }

    /// Reset canvas state before starting a new frame on the owned surface.
    pub fn reset(&mut self) {
        self.state.finish_frame();
        SkiaCpuRenderState::reset(&mut self.surface);
    }

    /// Stream `imaging` commands directly into the owned raster surface.
    pub fn with_canvas_sink<R>(
        &mut self,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        self.state.with_canvas_sink(&mut self.surface, f)
    }

    /// Replay an `imaging` scene through the owned raster backend.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<(), Error> {
        self.state.render_scene(&mut self.surface, scene)
    }

    /// Reset, render a recorded scene, and return RGBA8 bytes.
    pub fn render_scene_rgba8(&mut self, scene: &Scene) -> Result<Vec<u8>, Error> {
        self.reset();
        self.render_scene(scene)?;
        Ok(self.read_image()?.data.as_ref().to_vec())
    }

    /// Draw a native Skia picture through the owned raster backend.
    pub fn render_picture(&mut self, picture: &sk::Picture) -> Result<(), Error> {
        self.state.render_picture(&mut self.surface, picture)
    }

    /// Reset, draw a native Skia picture, and return RGBA8 bytes.
    pub fn render_picture_rgba8(&mut self, picture: &sk::Picture) -> Result<Vec<u8>, Error> {
        self.reset();
        self.render_picture(picture)?;
        Ok(self.read_image()?.data.as_ref().to_vec())
    }

    /// Read back the current owned raster surface into an unpremultiplied RGBA8 image.
    pub fn read_image(&mut self) -> Result<peniko::ImageData, Error> {
        SkiaCpuRenderState::read_image(&mut self.surface)
    }
}

/// CPU target renderer that binds reusable Skia CPU state to a caller-provided pixel buffer.
pub struct SkiaCpuTargetRenderer<'a> {
    state: SkiaCpuRenderState,
    surface: sk::Borrows<'a, sk::Surface>,
}

impl<'a> core::fmt::Debug for SkiaCpuTargetRenderer<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SkiaCpuTargetRenderer")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl SkiaCpuTargetRenderer<'_> {
    fn with_renderer<R>(&mut self, f: impl FnOnce(&mut SkiaCpuRendererRef<'_>) -> R) -> R {
        let mut renderer = self.state.bind(&mut self.surface);
        f(&mut renderer)
    }

    fn readback_image(&mut self) -> Result<peniko::ImageData, Error> {
        self.with_renderer(|renderer| renderer.read_image())
    }

    fn with_canvas<R>(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink) -> R) -> R {
        let mut renderer = self.state.bind(&mut self.surface);
        renderer
            .with_canvas_sink(|sink| f(sink))
            .expect("render into imaging_skia cpu target sink")
    }
}

impl RenderCore for SkiaCpuTargetRenderer<'_> {
    fn render(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink)) {
        self.with_canvas(&mut |canvas| f(canvas));
    }

    fn finish(&mut self) {
        self.state.finish_frame();
    }

    fn readback(&mut self) -> Option<RenderOutput> {
        self.readback_image().ok().map(RenderOutput::Image)
    }

    fn debug_info(&self) -> String {
        "name: Skia CPU\ninfo: imaging_skia::SkiaCpuTargetRenderer".to_string()
    }
}

impl<'a> TargetRenderer for SkiaCpuTargetRenderer<'a> {
    type Target = CpuBufferTarget<'a>;

    fn create(_frame: BeginFrame, target: Self::Target) -> Result<Self, String> {
        let color_type = match target.format {
            CpuBufferFormat::Rgba8Opaque => sk::ColorType::RGBA8888,
            CpuBufferFormat::Bgra8Opaque => sk::ColorType::BGRA8888,
        };
        let info = sk::ImageInfo::new(
            (target.width as i32, target.height as i32),
            color_type,
            sk::AlphaType::Opaque,
            None,
        );
        let surface =
            sk::surfaces::wrap_pixels(&info, target.buffer, Some(target.bytes_per_row), None)
                .ok_or_else(|| "wrap skia cpu target pixels".to_string())?;
        Ok(Self {
            state: SkiaCpuRenderState::new(),
            surface,
        })
    }
}

/// Borrowed CPU raster renderer view that binds reusable CPU state to a caller-owned surface.
#[derive(Debug)]
pub struct SkiaCpuRendererRef<'a> {
    state: &'a mut SkiaCpuRenderState,
    surface: &'a mut sk::Surface,
}

impl SkiaCpuRendererRef<'_> {
    /// Reset canvas state before starting a new frame on the bound raster surface.
    pub fn reset(&mut self) {
        self.state.finish_frame();
        SkiaCpuRenderState::reset(self.surface);
    }

    /// Stream `imaging` commands directly into the bound raster surface.
    pub fn with_canvas_sink<R>(
        &mut self,
        f: impl FnOnce(&mut SkCanvasSink<'_>) -> R,
    ) -> Result<R, Error> {
        self.state.with_canvas_sink(self.surface, f)
    }

    /// Replay an `imaging` scene through the bound raster surface.
    pub fn render_scene(&mut self, scene: &Scene) -> Result<(), Error> {
        self.state.render_scene(self.surface, scene)
    }

    /// Draw a native Skia picture through the bound raster surface.
    pub fn render_picture(&mut self, picture: &sk::Picture) -> Result<(), Error> {
        self.state.render_picture(self.surface, picture)
    }

    /// Read back the current bound raster surface into an unpremultiplied RGBA8 image.
    pub fn read_image(&mut self) -> Result<peniko::ImageData, Error> {
        SkiaCpuRenderState::read_image(self.surface)
    }

    /// Borrow the currently bound raster `skia_safe::Surface`.
    pub fn surface(&mut self) -> &mut sk::Surface {
        self.surface
    }
}

#[cfg(feature = "wgpu")]
/// Map supported `wgpu` texture formats to the Skia color types used for wrapped surfaces.
pub(crate) fn color_type_for_wgpu_texture_format(
    texture_format: wgpu::TextureFormat,
) -> Result<sk::ColorType, Error> {
    match texture_format {
        wgpu::TextureFormat::Rgba8Unorm => Ok(sk::ColorType::RGBA8888),
        wgpu::TextureFormat::Rgba8UnormSrgb => Ok(sk::ColorType::SRGBA8888),
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            Ok(sk::ColorType::BGRA8888)
        }
        wgpu::TextureFormat::Rgb10a2Unorm => Ok(sk::ColorType::RGBA1010102),
        wgpu::TextureFormat::Rgba16Unorm => Ok(sk::ColorType::R16G16B16A16UNorm),
        wgpu::TextureFormat::Rgba16Float => Ok(sk::ColorType::RGBAF16),
        _ => Err(Error::Internal("unsupported wgpu texture format")),
    }
}

#[cfg(feature = "wgpu")]
/// Attach an explicit Skia color space when the wrapped `wgpu` texture is sRGB encoded.
pub(crate) fn color_space_for_wgpu_texture_format(
    texture_format: wgpu::TextureFormat,
) -> Option<sk::ColorSpace> {
    match texture_format {
        wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb => {
            Some(sk::ColorSpace::new_srgb())
        }
        _ => None,
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
            sk::ColorType::SRGBA8888
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
