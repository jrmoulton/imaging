// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello hybrid backend for `imaging`.
//!
//! This crate provides a headless CPU/GPU renderer that consumes `imaging::record::Scene` or a
//! native [`vello_hybrid::Scene`] and produces an RGBA8 image buffer using `vello_hybrid` +
//! `wgpu`.
//!
//! Recorded scenes with inline image brushes are uploaded through a renderer-scoped image registry
//! and translated to backend-managed opaque image ids. Use [`VelloHybridSceneSink::with_renderer`]
//! when recording directly into a native [`vello_hybrid::Scene`] and you want the same image
//! support.
//!
//! # Render A Recorded Scene
//!
//! Record commands into [`imaging::record::Scene`], then render them with
//! [`VelloHybridRenderer`].
//!
//! ```no_run
//! use imaging::{Painter, record};
//! use imaging_vello_hybrid::VelloHybridRenderer;
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_vello_hybrid::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x2a, 0x6f, 0xdb));
//!     let mut scene = record::Scene::new();
//!
//!     {
//!         let mut painter = Painter::new(&mut scene);
//!         painter.fill_rect(Rect::new(0.0, 0.0, 128.0, 128.0), &paint);
//!     }
//!
//!     let mut renderer = VelloHybridRenderer::try_new(128, 128)?;
//!     let image = renderer.render_scene_rgba8(&scene)?;
//!     assert_eq!(image.width, 128);
//!     Ok(())
//! }
//! ```
//!
//! # Record Into `vello_hybrid::Scene`
//!
//! If you want a backend-native retained scene without owning a full renderer, wrap an existing
//! [`vello_hybrid::Scene`] with [`VelloHybridSceneSink`].
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_vello_hybrid::VelloHybridSceneSink;
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_vello_hybrid::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x1d, 0x4e, 0x89));
//!     let mut scene = vello_hybrid::Scene::new(128, 128);
//!     scene.reset();
//!
//!     {
//!         let mut sink = VelloHybridSceneSink::new(&mut scene);
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(Rect::new(0.0, 0.0, 128.0, 128.0), &paint);
//!         sink.finish()?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! Use [`VelloHybridSceneSink::with_renderer`] instead when the scene uses image brushes.
//!
//! # Record Image Brushes Into `vello_hybrid::Scene`
//!
//! Use [`VelloHybridSceneSink::with_renderer`] when recording image brushes directly into a
//! native [`vello_hybrid::Scene`]. The sink uploads images through the renderer and reuses them
//! across later recordings and renders.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use imaging::Painter;
//! use imaging_vello_hybrid::{VelloHybridRenderer, VelloHybridSceneSink};
//! use kurbo::Rect;
//! use peniko::{Blob, Brush, ImageAlphaType, ImageBrush, ImageData, ImageFormat};
//!
//! fn main() -> Result<(), imaging_vello_hybrid::Error> {
//!     let image = ImageData {
//!         data: Blob::new(Arc::new([
//!             0xff, 0x20, 0x20, 0xff, 0x20, 0xff, 0x20, 0xff, 0x20, 0x20, 0xff, 0xff, 0xff,
//!             0xff, 0x20, 0xff,
//!         ])),
//!         format: ImageFormat::Rgba8,
//!         alpha_type: ImageAlphaType::Alpha,
//!         width: 2,
//!         height: 2,
//!     };
//!     let brush = Brush::Image(ImageBrush::new(image));
//!
//!     let mut renderer = VelloHybridRenderer::try_new(128, 128)?;
//!     let mut scene = vello_hybrid::Scene::new(128, 128);
//!     scene.reset();
//!
//!     {
//!         let mut sink = VelloHybridSceneSink::with_renderer(&mut scene, &mut renderer);
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(Rect::new(0.0, 0.0, 128.0, 128.0), &brush);
//!         sink.finish()?;
//!     }
//!
//!     let image = renderer.render_vello_hybrid_scene_rgba8(&scene)?;
//!     assert_eq!(image.width, 128);
//!     Ok(())
//! }
//! ```
//!
//! # Render A Native `vello_hybrid::Scene`
//!
//! If you already have a native hybrid scene, hand it directly to [`VelloHybridRenderer`].
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_vello_hybrid::{VelloHybridRenderer, VelloHybridSceneSink};
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_vello_hybrid::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0xd9, 0x77, 0x06));
//!     let mut scene = vello_hybrid::Scene::new(128, 128);
//!     scene.reset();
//!
//!     {
//!         let mut sink = VelloHybridSceneSink::new(&mut scene);
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(Rect::new(16.0, 16.0, 112.0, 112.0), &paint);
//!         sink.finish()?;
//!     }
//!
//!     let mut renderer = VelloHybridRenderer::try_new(128, 128)?;
//!     let image = renderer.render_vello_hybrid_scene_rgba8(&scene)?;
//!     assert_eq!(image.width, 128);
//!     Ok(())
//! }
//! ```

#![deny(unsafe_code)]
#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod image_registry;
mod scene_sink;

use image_registry::{HybridImageRegistry, HybridImageUploadSession};
use imaging::{
    PaintSink,
    record::{Scene, ValidateError, replay},
};
use core::convert::Infallible;
use imaging_backend::{
    Backend as ImagingBackend, CpuBufferAlphaMode, CpuBufferChannelOrder, CpuBufferTarget,
    GpuTextureTarget, GpuTextureTargetInfo, RenderOutput, RenderSource,
};
use kurbo::Size;
use peniko::{Blob, ImageAlphaType, ImageData, ImageFormat};
use std::sync::mpsc;
use vello_hybrid::{RenderError, RenderSize, RenderTargetConfig};
use wgpu::{
    CommandEncoderDescriptor, Extent3d, TextureDescriptor, TextureDimension, TextureFormat,
};

pub use scene_sink::VelloHybridSceneSink;

/// Errors that can occur when rendering via Vello hybrid.
#[derive(Debug)]
pub enum Error {
    /// The scene is invalid (unbalanced stacks).
    InvalidScene(ValidateError),
    /// An image brush was encountered on a sink path that has no renderer-backed image resolver.
    UnsupportedImageBrush,
    /// A filter configuration could not be translated.
    UnsupportedFilter,
    /// Masks are not supported by this backend yet.
    UnsupportedMask,
    /// Blurred rounded rect draws are not supported by this backend yet.
    UnsupportedBlurredRoundedRect,
    /// No suitable GPU adapter was found.
    NoAdapter,
    /// A GPU device could not be created.
    RequestDevice,
    /// Vello hybrid returned a render error.
    Render(RenderError),
    /// An internal invariant was violated.
    Internal(&'static str),
}

/// Describes the pixel format and alpha encoding requested for hybrid renderer readback.
#[derive(Clone, Copy, Debug)]
pub struct ImageOutputFormat {
    /// Channel order of the returned bytes.
    pub format: ImageFormat,
    /// Alpha encoding of the returned bytes.
    pub alpha_type: ImageAlphaType,
}

impl ImageOutputFormat {
    /// Unpremultiplied `RGBA8` image output.
    pub const RGBA8: Self = Self {
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
    };

    /// Unpremultiplied `BGRA8` image output.
    pub const BGRA8: Self = Self {
        format: ImageFormat::Bgra8,
        alpha_type: ImageAlphaType::Alpha,
    };

    /// Premultiplied `RGBA8` image output.
    pub const RGBA8_PREMULTIPLIED: Self = Self {
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::AlphaPremultiplied,
    };

    /// Premultiplied `BGRA8` image output.
    pub const BGRA8_PREMULTIPLIED: Self = Self {
        format: ImageFormat::Bgra8,
        alpha_type: ImageAlphaType::AlphaPremultiplied,
    };
}

/// Renderer that executes `imaging` commands using `vello_hybrid` + `wgpu`.
#[derive(Debug)]
pub struct VelloHybridRenderer {
    renderer: vello_hybrid::Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    bytes_per_row: u32,
    width: u16,
    height: u16,
    tolerance: f64,
    image_registry: HybridImageRegistry,
}

/// GPU copy renderer that records into a native hybrid scene and renders into an owned texture.
#[derive(Debug)]
pub struct VelloHybridGpuCopyRenderer {
    renderer: vello_hybrid::Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture_view: wgpu::TextureView,
    texture_format: TextureFormat,
    size: (u32, u32),
    scene: vello_hybrid::Scene,
    tolerance: f64,
}

impl VelloHybridRenderer {
    /// Create a renderer for a fixed-size target.
    pub fn new(width: u16, height: u16) -> Self {
        Self::try_new(width, height).expect("create imaging_vello_hybrid renderer")
    }

    /// Create a renderer for a fixed-size target.
    ///
    /// This is fallible because `wgpu` may not be able to find a compatible adapter/device
    /// in some sandboxed or headless environments.
    pub fn try_new(width: u16, height: u16) -> Result<Self, Error> {
        let (device, queue) = pollster::block_on(init_device_and_queue())?;
        Self::try_new_with_device_queue(device, queue, width, height)
    }

    /// Create a renderer for a fixed-size target using caller-provided `wgpu` state.
    pub fn try_new_with_device_queue(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u16,
        height: u16,
    ) -> Result<Self, Error> {
        let (texture, texture_view, readback, bytes_per_row) =
            create_targets(&device, width, height);

        let renderer = vello_hybrid::Renderer::new(
            &device,
            &RenderTargetConfig {
                format: TextureFormat::Rgba8Unorm,
                width: u32::from(width),
                height: u32::from(height),
            },
        );

        Ok(Self {
            renderer,
            device,
            queue,
            texture,
            texture_view,
            readback,
            bytes_per_row,
            width,
            height,
            tolerance: 0.1,
            image_registry: HybridImageRegistry::new(),
        })
    }

    /// Set the tolerance used when converting shapes to paths.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    pub(crate) fn begin_image_upload_session(
        &mut self,
        label: &'static str,
    ) -> HybridImageUploadSession<'_> {
        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });
        self.image_registry.begin_upload_session(
            &mut self.renderer,
            &self.device,
            &self.queue,
            encoder,
        )
    }

    /// Destroy all uploaded hybrid image resources cached by this renderer.
    pub fn clear_cached_images(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("imaging_vello_hybrid clear cached images"),
            });
        self.image_registry
            .clear(&mut self.renderer, &self.device, &self.queue, &mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    /// Render a recorded scene and return an image in the requested output format.
    ///
    /// Inline image brushes are uploaded on demand and cached for the lifetime of this renderer
    /// (or until [`Self::clear_cached_images`] is called).
    pub fn render_scene_image(
        &mut self,
        scene: &Scene,
        output: ImageOutputFormat,
    ) -> Result<ImageData, Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        let mut native = vello_hybrid::Scene::new(self.width, self.height);
        native.reset();
        let tolerance = self.tolerance;
        {
            let mut sink = VelloHybridSceneSink::with_renderer(&mut native, self);
            sink.set_tolerance(tolerance);
            replay(scene, &mut sink);
            sink.finish()?;
        }
        self.render_vello_hybrid_scene_image(&native, output)
    }

    /// Render a recorded scene into the requested output format.
    pub fn render_scene_into(
        &mut self,
        scene: &Scene,
        dst: &mut [u8],
        bytes_per_row: usize,
        output: ImageOutputFormat,
    ) -> Result<(), Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        let mut native = vello_hybrid::Scene::new(self.width, self.height);
        native.reset();
        let tolerance = self.tolerance;
        {
            let mut sink = VelloHybridSceneSink::with_renderer(&mut native, self);
            sink.set_tolerance(tolerance);
            replay(scene, &mut sink);
            sink.finish()?;
        }
        self.render_vello_hybrid_scene_into(&native, dst, bytes_per_row, output)
    }

    /// Render a native [`vello_hybrid::Scene`] and return an image in the requested output format.
    pub fn render_vello_hybrid_scene_image(
        &mut self,
        scene: &vello_hybrid::Scene,
        output: ImageOutputFormat,
    ) -> Result<ImageData, Error> {
        let mut bytes = vec![0_u8; usize::from(self.width) * usize::from(self.height) * 4];
        self.render_vello_hybrid_scene_into(
            scene,
            &mut bytes,
            usize::from(self.width) * 4,
            output,
        )?;
        Ok(ImageData {
            data: Blob::new(std::sync::Arc::new(bytes)),
            format: output.format,
            alpha_type: output.alpha_type,
            width: u32::from(self.width),
            height: u32::from(self.height),
        })
    }

    /// Render a native [`vello_hybrid::Scene`] into the requested output format.
    pub fn render_vello_hybrid_scene_into(
        &mut self,
        scene: &vello_hybrid::Scene,
        dst: &mut [u8],
        bytes_per_row: usize,
        output: ImageOutputFormat,
    ) -> Result<(), Error> {
        self.render_vello_hybrid_scene_into_output(scene, dst, bytes_per_row, output)
    }

    /// Render a recorded scene and return an RGBA8 image (unpremultiplied).
    pub fn render_scene_rgba8(&mut self, scene: &Scene) -> Result<ImageData, Error> {
        self.render_scene_image(scene, ImageOutputFormat::RGBA8)
    }

    /// Render a recorded scene into an opaque RGBA8 buffer.
    pub fn render_scene_into_rgba8_opaque(
        &mut self,
        scene: &Scene,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) -> Result<(), Error> {
        self.render_scene_into(
            scene,
            dst,
            bytes_per_row,
            ImageOutputFormat {
                format: ImageFormat::Rgba8,
                alpha_type: ImageAlphaType::AlphaPremultiplied,
            },
        )
    }

    /// Render a recorded scene into an opaque BGRA8 buffer.
    pub fn render_scene_into_bgra8_opaque(
        &mut self,
        scene: &Scene,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) -> Result<(), Error> {
        self.render_scene_into(
            scene,
            dst,
            bytes_per_row,
            ImageOutputFormat {
                format: ImageFormat::Bgra8,
                alpha_type: ImageAlphaType::AlphaPremultiplied,
            },
        )
    }

    /// Render a native [`vello_hybrid::Scene`] and return an RGBA8 image (unpremultiplied).
    pub fn render_vello_hybrid_scene_rgba8(
        &mut self,
        scene: &vello_hybrid::Scene,
    ) -> Result<ImageData, Error> {
        self.render_vello_hybrid_scene_image(scene, ImageOutputFormat::RGBA8)
    }

    /// Render a native [`vello_hybrid::Scene`] into an opaque RGBA8 buffer.
    pub fn render_vello_hybrid_scene_into_rgba8_opaque(
        &mut self,
        scene: &vello_hybrid::Scene,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) -> Result<(), Error> {
        self.render_vello_hybrid_scene_into(
            scene,
            dst,
            bytes_per_row,
            ImageOutputFormat {
                format: ImageFormat::Rgba8,
                alpha_type: ImageAlphaType::AlphaPremultiplied,
            },
        )
    }

    /// Render a native [`vello_hybrid::Scene`] into an opaque BGRA8 buffer.
    pub fn render_vello_hybrid_scene_into_bgra8_opaque(
        &mut self,
        scene: &vello_hybrid::Scene,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) -> Result<(), Error> {
        self.render_vello_hybrid_scene_into(
            scene,
            dst,
            bytes_per_row,
            ImageOutputFormat {
                format: ImageFormat::Bgra8,
                alpha_type: ImageAlphaType::AlphaPremultiplied,
            },
        )
    }

    fn render_vello_hybrid_scene_into_output(
        &mut self,
        scene: &vello_hybrid::Scene,
        dst: &mut [u8],
        bytes_per_row: usize,
        output: ImageOutputFormat,
    ) -> Result<(), Error> {
        let render_size = RenderSize {
            width: u32::from(self.width),
            height: u32::from(self.height),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("imaging_vello_hybrid render"),
            });

        self.renderer
            .render(
                scene,
                &self.device,
                &self.queue,
                &mut encoder,
                &render_size,
                &self.texture_view,
            )
            .map_err(Error::Render)?;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: u32::from(self.width),
                height: u32::from(self.height),
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);

        let slice = self.readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| Error::Internal("device poll failed"))?;
        rx.recv()
            .map_err(|_| Error::Internal("map_async callback dropped"))?
            .map_err(|_| Error::Internal("buffer map failed"))?;

        let mapped = slice.get_mapped_range();
        let width_bytes = usize::from(self.width) * 4;
        let mut pixels = Vec::with_capacity(usize::from(self.width) * usize::from(self.height));
        for row in mapped.chunks_exact(self.bytes_per_row as usize) {
            for px in row[..width_bytes].chunks_exact(4) {
                pixels.push(peniko::color::PremulRgba8::from_u8_array([
                    px[0], px[1], px[2], px[3],
                ]));
            }
        }
        drop(mapped);
        self.readback.unmap();

        let width = usize::from(self.width);
        let height = usize::from(self.height);
        if dst.len() < bytes_per_row.saturating_mul(height) || bytes_per_row < width * 4 {
            return Err(Error::Internal("destination buffer too small"));
        }

        match output.alpha_type {
            ImageAlphaType::Alpha => {
                let pixmap =
                    vello_common::pixmap::Pixmap::from_parts(pixels, self.width, self.height);
                let unpremul = pixmap.take_unpremultiplied();
                for (src_row, dst_row) in unpremul
                    .chunks_exact(width)
                    .zip(dst.chunks_exact_mut(bytes_per_row))
                {
                    for (src, out) in src_row.iter().zip(dst_row[..width * 4].chunks_exact_mut(4)) {
                        match output.format {
                            ImageFormat::Rgba8 => {
                                out.copy_from_slice(&[src.r, src.g, src.b, src.a]);
                            }
                            ImageFormat::Bgra8 => {
                                out.copy_from_slice(&[src.b, src.g, src.r, src.a]);
                            }
                            _ => return Err(Error::Internal("unsupported image format")),
                        }
                    }
                }
            }
            ImageAlphaType::AlphaPremultiplied => {
                for (src_row, dst_row) in pixels
                    .chunks_exact(width)
                    .zip(dst.chunks_exact_mut(bytes_per_row))
                {
                    for (src, out) in src_row.iter().zip(dst_row[..width * 4].chunks_exact_mut(4)) {
                        let rgba = src.to_u8_array();
                        match output.format {
                            ImageFormat::Rgba8 => out.copy_from_slice(&rgba),
                            ImageFormat::Bgra8 => {
                                out.copy_from_slice(&[rgba[2], rgba[1], rgba[0], rgba[3]]);
                            }
                            _ => return Err(Error::Internal("unsupported image format")),
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl ImagingBackend for VelloHybridRenderer {
    type Error = String;
    type Image = ImageData;
    type BufferTarget<'a> = CpuBufferTarget<'a>;
    type TextureTarget<'a> = Infallible;

    #[allow(
        clippy::cast_possible_truncation,
        reason = "Render target sizes are interpreted as whole pixels and then range-checked against `u16`."
    )]
    fn render_to_buffer<'a>(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        target: Self::BufferTarget<'a>,
    ) -> Result<(), Self::Error> {
        let width = u16::try_from(size.width as u32)
            .map_err(|_| "width exceeds vello_hybrid limit".to_string())?;
        let height = u16::try_from(size.height as u32)
            .map_err(|_| "height exceeds vello_hybrid limit".to_string())?;
        if self.width != width || self.height != height {
            *self = Self::try_new_with_device_queue(
                self.device.clone(),
                self.queue.clone(),
                width,
                height,
            )
            .map_err(|err| format!("{err:?}"))?;
        }
        let mut scene = vello_hybrid::Scene::new(width, height);
        {
            let mut sink = VelloHybridSceneSink::new(&mut scene);
            sink.set_tolerance(self.tolerance);
            source.paint_into(&mut sink);
            let _ = sink.finish();
        }
        let output = match (target.format.channel_order, target.format.alpha_mode) {
            (CpuBufferChannelOrder::Rgba8, CpuBufferAlphaMode::Opaque) => {
                ImageOutputFormat::RGBA8
            }
            (CpuBufferChannelOrder::Bgra8, CpuBufferAlphaMode::Opaque) => {
                ImageOutputFormat::BGRA8
            }
            (CpuBufferChannelOrder::Rgba8, CpuBufferAlphaMode::Premultiplied) => {
                ImageOutputFormat::RGBA8_PREMULTIPLIED
            }
            (CpuBufferChannelOrder::Bgra8, CpuBufferAlphaMode::Premultiplied) => {
                ImageOutputFormat::BGRA8_PREMULTIPLIED
            }
        };
        self.render_vello_hybrid_scene_into(&scene, target.buffer, target.bytes_per_row, output)
            .map_err(|err| format!("{err:?}"))
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
        let width = u16::try_from(width).map_err(|_| "width exceeds vello hybrid cpu limit")?;
        let height =
            u16::try_from(height).map_err(|_| "height exceeds vello hybrid cpu limit")?;
        let mut native = vello_hybrid::Scene::new(width, height);
        native.reset();
        let tolerance = self.tolerance;
        {
            let mut sink = VelloHybridSceneSink::with_renderer(&mut native, self);
            sink.set_tolerance(tolerance);
            let _ = size;
            source.paint_into(&mut sink);
            sink.finish().map_err(|err| format!("{err:?}"))?;
        }
        if self.width == width && self.height == height {
            self.render_vello_hybrid_scene_image(&native, ImageOutputFormat::RGBA8)
                .map_err(|err| format!("{err:?}"))
        } else {
            let mut renderer = VelloHybridRenderer::try_new_with_device_queue(
                self.device.clone(),
                self.queue.clone(),
                width,
                height,
            )
            .map_err(|err| format!("{err:?}"))?;
            renderer
                .render_vello_hybrid_scene_image(&native, ImageOutputFormat::RGBA8)
                .map_err(|err| format!("{err:?}"))
        }
    }
}

impl VelloHybridGpuCopyRenderer {
    /// Create a GPU-copy renderer for a fixed-size target using caller-provided `wgpu` state.
    pub fn try_new_with_device_queue(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u16,
        height: u16,
        format: TextureFormat,
    ) -> Result<Self, Error> {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("imaging_vello_hybrid gpu copy render target"),
            size: Extent3d {
                width: u32::from(width),
                height: u32::from(height),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[format],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let renderer = vello_hybrid::Renderer::new(
            &device,
            &RenderTargetConfig {
                format,
                width: u32::from(width),
                height: u32::from(height),
            },
        );
        Ok(Self {
            renderer,
            device,
            queue,
            texture_view,
            texture_format: format,
            size: (u32::from(width), u32::from(height)),
            scene: vello_hybrid::Scene::new(width, height),
            tolerance: 0.1,
        })
    }

    fn with_scene_sink<R>(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink) -> R) -> R {
        let mut sink = VelloHybridSceneSink::new(&mut self.scene);
        sink.set_tolerance(self.tolerance);
        let out = f(&mut sink);
        let _ = sink.finish();
        out
    }

    fn recreate_renderer(&mut self, width: u32, height: u32, format: TextureFormat) {
        self.renderer = vello_hybrid::Renderer::new(
            &self.device,
            &RenderTargetConfig {
                format,
                width,
                height,
            },
        );
        self.texture_format = format;
        self.size = (width, height);
        self.scene = vello_hybrid::Scene::new(
            u16::try_from(width).expect("width exceeds vello_hybrid limit"),
            u16::try_from(height).expect("height exceeds vello_hybrid limit"),
        );
    }
}

impl VelloHybridGpuCopyRenderer {
    fn with_paint_sink(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink)) {
        self.with_scene_sink(&mut |canvas| f(canvas));
    }

    fn finish(&mut self) {
        let render_size = RenderSize {
            width: self.size.0,
            height: self.size.1,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("imaging_vello_hybrid gpu target render"),
            });
        self.renderer
            .render(
                &self.scene,
                &self.device,
                &self.queue,
                &mut encoder,
                &render_size,
                &self.texture_view,
            )
            .expect("render into imaging_vello_hybrid gpu target");
        self.queue.submit([encoder.finish()]);
    }

    #[cfg(test)]
    fn readback(&mut self) -> Option<RenderOutput> {
        Some(RenderOutput::GpuTexture(self.texture_view.clone()))
    }
}

impl VelloHybridGpuCopyRenderer {
    fn set_target(&mut self, _size: Size, target: GpuTextureTarget) -> Result<(), String> {
        let texture = target.texture_view.texture();
        let size = texture.size();
        let width = u16::try_from(size.width)
            .map_err(|_| "width exceeds vello_hybrid limit".to_string())?;
        let height = u16::try_from(size.height)
            .map_err(|_| "height exceeds vello_hybrid limit".to_string())?;
        let format = texture.format();
        self.device = target.device;
        self.queue = target.queue;
        self.texture_view = target.texture_view;
        if self.size != (size.width, size.height) || self.texture_format != format {
            self.recreate_renderer(u32::from(width), u32::from(height), format);
        } else {
            self.scene.reset();
        }
        Ok(())
    }
}

impl ImagingBackend for VelloHybridGpuCopyRenderer {
    type Error = String;
    type Image = ImageData;
    type BufferTarget<'a> = Infallible;
    type TextureTarget<'a> = GpuTextureTarget;

    fn supports_texture_target(target: &GpuTextureTargetInfo) -> Result<(), Self::Error> {
        match target.format {
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => Ok(()),
            _ => Err("vello hybrid gpu backend only supports rgba8 texture targets directly".to_string()),
        }
    }

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
        self.set_target(size, target)?;
        self.with_paint_sink(&mut |sink| source.paint_into(sink));
        self.finish();
        Ok(())
    }

    fn render_to_image(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        width: u32,
        height: u32,
    ) -> Result<Self::Image, Self::Error> {
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("imaging_vello_hybrid gpu image target"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.set_target(
            size,
            GpuTextureTarget {
                device: self.device.clone(),
                queue: self.queue.clone(),
                texture_view,
            },
        )?;
        self.with_paint_sink(&mut |sink| source.paint_into(sink));
        self.finish();
        RenderOutput::GpuTexture(self.texture_view.clone())
            .into_image_with(&self.device, &self.queue)
            .ok_or_else(|| "vello hybrid gpu backend failed to read rendered image".to_string())
    }
}

async fn init_device_and_queue() -> Result<(wgpu::Device, wgpu::Queue), Error> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|_| Error::NoAdapter)?;

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("imaging_vello_hybrid device"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        })
        .await
        .map_err(|_| Error::RequestDevice)
}

fn create_targets(
    device: &wgpu::Device,
    width: u16,
    height: u16,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Buffer, u32) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("imaging_vello_hybrid render target"),
        size: Extent3d {
            width: u32::from(width),
            height: u32::from(height),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = (u32::from(width) * 4).next_multiple_of(256);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imaging_vello_hybrid readback buffer"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (texture, texture_view, readback, bytes_per_row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use imaging::Painter;
    use kurbo::Rect;
    use peniko::Color;

    fn try_init_device_and_queue() -> Result<(wgpu::Device, wgpu::Queue), ()> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .map_err(|_| ())?;
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("imaging_vello_hybrid test device"),
                    required_features: wgpu::Features::empty(),
                    ..Default::default()
                })
                .await
                .map_err(|_| ())
        })
    }

    #[test]
    fn gpu_copy_renderer_renders_into_owned_texture() {
        let Ok((device, queue)) = try_init_device_and_queue() else {
            return;
        };
        let _texture = device.create_texture(&TextureDescriptor {
            label: Some("imaging_vello_hybrid gpu target test texture"),
            size: Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback_device = device.clone();
        let readback_queue = queue.clone();
        let mut renderer = VelloHybridGpuCopyRenderer::try_new_with_device_queue(
            device,
            queue,
            32,
            32,
            TextureFormat::Rgba8Unorm,
        )
        .expect("create gpu copy renderer");

        renderer.with_paint_sink(&mut |canvas| {
            let mut painter = Painter::new(canvas);
            painter.fill_rect(
                Rect::new(0.0, 0.0, 32.0, 32.0),
                Color::from_rgb8(0x2a, 0x6f, 0xdb),
            );
        });
        renderer.finish();

        let image = renderer
            .readback()
            .and_then(|output| output.into_image_with(&readback_device, &readback_queue))
            .expect("read back gpu copy renderer image");
        assert_eq!(image.width, 32);
        assert_eq!(image.height, 32);
    }
}
