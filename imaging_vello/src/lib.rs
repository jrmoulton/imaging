// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello backend for `imaging`.
//!
//! This crate provides a headless CPU/GPU renderer that consumes `imaging::record::Scene` or a
//! native [`vello::Scene`] and produces an RGBA8 image buffer using `vello` + `wgpu`.
//!
//! Enable exactly one backend compatibility feature:
//!
//! - `vello-0-8` (default)
//! - `vello-0-7`
//!
//! # Render A Recorded Scene
//!
//! Record commands into [`imaging::record::Scene`], then render them with [`VelloRenderer`].
//!
//! ```no_run
//! use imaging::{Painter, record};
//! use imaging_vello::VelloRenderer;
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_vello::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x2a, 0x6f, 0xdb));
//!     let mut scene = record::Scene::new();
//!
//!     {
//!         let mut painter = Painter::new(&mut scene);
//!         painter.fill_rect(Rect::new(0.0, 0.0, 128.0, 128.0), &paint);
//!     }
//!
//!     let mut renderer = VelloRenderer::try_new(128, 128)?;
//!     let rgba = renderer.render_scene_rgba8(&scene)?;
//!     assert_eq!(rgba.len(), 128 * 128 * 4);
//!     Ok(())
//! }
//! ```
//!
//! # Record Into `vello::Scene`
//!
//! If you want a backend-native retained scene without going through [`VelloRenderer`], wrap a
//! mutable [`vello::Scene`] with [`VelloSceneSink`].
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_vello::{VelloSceneSink, vello};
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_vello::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0x1d, 0x4e, 0x89));
//!     let mut scene = vello::Scene::new();
//!
//!     {
//!         let bounds = Rect::new(0.0, 0.0, 128.0, 128.0);
//!         let mut sink = VelloSceneSink::new(&mut scene, bounds);
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(bounds, &paint);
//!         sink.finish()?;
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Render A Native `vello::Scene`
//!
//! If you already have a native Vello scene, hand it directly to [`VelloRenderer`].
//!
//! ```no_run
//! use imaging::Painter;
//! use imaging_vello::{VelloRenderer, VelloSceneSink, vello};
//! use kurbo::Rect;
//! use peniko::{Brush, Color};
//!
//! fn main() -> Result<(), imaging_vello::Error> {
//!     let paint = Brush::Solid(Color::from_rgb8(0xd9, 0x77, 0x06));
//!     let mut scene = vello::Scene::new();
//!
//!     {
//!         let bounds = Rect::new(0.0, 0.0, 128.0, 128.0);
//!         let mut sink = VelloSceneSink::new(&mut scene, bounds);
//!         let mut painter = Painter::new(&mut sink);
//!         painter.fill_rect(bounds, &paint);
//!         sink.finish()?;
//!     }
//!
//!     let mut renderer = VelloRenderer::try_new(128, 128)?;
//!     let rgba = renderer.render_vello_scene_rgba8(&scene)?;
//!     assert_eq!(rgba.len(), 128 * 128 * 4);
//!     Ok(())
//! }
//! ```
//!
//! Note: Vello uses a single layer stack for clipping and blending. Scenes that interleave clips
//! and groups in ways Vello cannot represent may return [`Error::UnbalancedLayerStack`].

#![deny(unsafe_code)]
#![cfg_attr(not(test), warn(unused_crate_dependencies))]

mod scene_sink;

#[cfg(all(feature = "vello-0-7", feature = "vello-0-8"))]
compile_error!("Enable exactly one of `vello-0-7` or `vello-0-8`.");

#[cfg(not(any(feature = "vello-0-7", feature = "vello-0-8")))]
compile_error!("Enable one of `vello-0-7` or `vello-0-8`.");

use imaging::record::{Scene, ValidateError, replay};
use kurbo::Rect;
use std::sync::mpsc;

#[cfg(feature = "vello-0-7")]
pub use vello_07 as vello;
#[cfg(all(not(feature = "vello-0-7"), feature = "vello-0-8"))]
pub use vello_08 as vello;

use crate::vello::wgpu;
use crate::vello::{AaConfig, RenderParams};
use imaging::PaintSink;
use core::convert::Infallible;
use imaging_backend::{
    Backend as ImagingBackend, GpuTextureTarget, GpuTextureTargetInfo, RenderOutput, RenderSource,
};
use kurbo::Size;

pub use scene_sink::VelloSceneSink;

/// Errors that can occur when rendering via Vello.
#[derive(Debug)]
pub enum Error {
    /// The scene is invalid (unbalanced stacks).
    InvalidScene(ValidateError),
    /// An image brush was encountered; this backend does not support it.
    UnsupportedImageBrush,
    /// A filter configuration could not be translated.
    UnsupportedFilter,
    /// A mask mode or masking primitive is not supported by this backend.
    UnsupportedMask,
    /// Glyph draws with non-default blend modes are not supported by this backend yet.
    UnsupportedGlyphBlend,
    /// Blurred rounded rect draws with non-default blend modes are not supported by this backend yet.
    UnsupportedBlurredRoundedRectBlend,
    /// The clip/group stack was not well-nested for this backend.
    ///
    /// Vello uses a single layer stack for both clipping and blending; `imaging` tracks these as
    /// separate stacks, so scenes that interleave them (e.g. `push_clip`, `push_group`, `pop_clip`)
    /// cannot be represented directly.
    UnbalancedLayerStack,
    /// No suitable GPU adapter was found.
    NoAdapter,
    /// A GPU device could not be created.
    RequestDevice,
    /// Vello returned a render error.
    Render(vello::Error),
    /// An internal invariant was violated.
    Internal(&'static str),
}

/// Renderer that executes `imaging` commands using `vello` + `wgpu`.
pub struct VelloRenderer {
    renderer: vello::Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    bytes_per_row: u32,
    width: u16,
    height: u16,
}

impl core::fmt::Debug for VelloRenderer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VelloRenderer")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// GPU copy renderer that records into a native Vello scene and renders into an owned texture.
pub struct VelloGpuCopyRenderer {
    renderer: vello::Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture_view: wgpu::TextureView,
    texture_format: wgpu::TextureFormat,
    size: (u32, u32),
    scene: vello::Scene,
}

impl core::fmt::Debug for VelloGpuCopyRenderer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VelloGpuCopyRenderer")
            .field("width", &self.size.0)
            .field("height", &self.size.1)
            .finish_non_exhaustive()
    }
}

impl VelloRenderer {
    /// Create a renderer for a fixed-size target.
    pub fn new(width: u16, height: u16) -> Self {
        Self::try_new(width, height).expect("create imaging_vello renderer")
    }

    /// Create a renderer for a fixed-size target.
    ///
    /// This is fallible because `wgpu` may not be able to find a compatible adapter/device
    /// in some sandboxed or headless environments.
    pub fn try_new(width: u16, height: u16) -> Result<Self, Error> {
        let (device, queue) = pollster::block_on(init_device_and_queue())?;
        let (texture, texture_view, readback, bytes_per_row) =
            create_targets(&device, width, height);

        let renderer = vello::Renderer::new(&device, vello::RendererOptions::default())
            .map_err(Error::Render)?;

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
        })
    }

    /// Render a recorded scene and return an RGBA8 buffer (unpremultiplied).
    pub fn render_scene_rgba8(&mut self, scene: &Scene) -> Result<Vec<u8>, Error> {
        scene.validate().map_err(Error::InvalidScene)?;
        let mut native = vello::Scene::new();
        let bounds = Rect::new(0.0, 0.0, f64::from(self.width), f64::from(self.height));
        let mut sink = VelloSceneSink::new(&mut native, bounds);
        replay(scene, &mut sink);
        sink.finish()?;
        self.render_vello_scene_rgba8(&native)
    }

    /// Render a native [`crate::vello::Scene`] and return an RGBA8 buffer (unpremultiplied).
    pub fn render_vello_scene_rgba8(&mut self, scene: &vello::Scene) -> Result<Vec<u8>, Error> {
        let params = RenderParams {
            base_color: peniko::Color::from_rgba8(0, 0, 0, 0),
            width: u32::from(self.width),
            height: u32::from(self.height),
            antialiasing_method: AaConfig::Area,
        };

        self.renderer
            .render_to_texture(
                &self.device,
                &self.queue,
                scene,
                &self.texture_view,
                &params,
            )
            .map_err(Error::Render)?;

        readback_rgba8(
            &self.device,
            &self.queue,
            &self.texture,
            &self.readback,
            self.bytes_per_row,
            self.width,
            self.height,
        )
    }
}

impl VelloGpuCopyRenderer {
    /// Create a GPU-copy renderer for a fixed-size target using caller-provided `wgpu` state.
    pub fn try_new_with_device_queue(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u16,
        height: u16,
        format: wgpu::TextureFormat,
    ) -> Result<Self, Error> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imaging_vello gpu copy render target"),
            size: wgpu::Extent3d {
                width: u32::from(width),
                height: u32::from(height),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING,
            format,
            view_formats: &[format],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let renderer = vello::Renderer::new(&device, vello::RendererOptions::default())
            .map_err(Error::Render)?;
        Ok(Self {
            renderer,
            device,
            queue,
            texture_view,
            texture_format: format,
            size: (u32::from(width), u32::from(height)),
            scene: vello::Scene::new(),
        })
    }

    fn with_scene_sink<R>(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink) -> R) -> R {
        let bounds = Rect::new(0.0, 0.0, f64::from(self.size.0), f64::from(self.size.1));
        let mut sink = VelloSceneSink::new(&mut self.scene, bounds);
        let out = f(&mut sink);
        let _ = sink.finish();
        out
    }
}

impl VelloGpuCopyRenderer {
    fn with_paint_sink(&mut self, f: &mut dyn FnMut(&mut dyn PaintSink)) {
        self.with_scene_sink(&mut |canvas| f(canvas));
    }

    fn finish(&mut self) {
        let params = RenderParams {
            base_color: peniko::Color::from_rgba8(0, 0, 0, 0),
            width: self.size.0,
            height: self.size.1,
            antialiasing_method: AaConfig::Area,
        };

        self.renderer
            .render_to_texture(
                &self.device,
                &self.queue,
                &self.scene,
                &self.texture_view,
                &params,
            )
            .expect("render into imaging_vello gpu copy target");
    }

    #[cfg(test)]
    fn readback(&mut self) -> Option<RenderOutput> {
        Some(RenderOutput::GpuTexture(self.texture_view.clone()))
    }

    fn set_target(&mut self, _size: Size, target: GpuTextureTarget) -> Result<(), String> {
        let texture = target.texture_view.texture();
        let size = texture.size();
        let format = texture.format();
        if self.texture_format != format {
            self.renderer = vello::Renderer::new(&target.device, vello::RendererOptions::default())
                .map_err(|err| format!("{err:?}"))?;
            self.texture_format = format;
        }
        self.device = target.device;
        self.queue = target.queue;
        self.texture_view = target.texture_view;
        self.size = (size.width, size.height);
        self.scene.reset();
        Ok(())
    }
}

impl ImagingBackend for VelloGpuCopyRenderer {
    type Error = String;
    type Image = peniko::ImageData;
    type BufferTarget<'a> = Infallible;
    type TextureTarget<'a> = GpuTextureTarget;

    fn supports_texture_target(target: &GpuTextureTargetInfo) -> Result<(), Self::Error> {
        match target.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => Ok(()),
            _ => Err("vello gpu backend only supports rgba8 texture targets directly".to_string()),
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
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imaging_vello gpu image target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
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
            .ok_or_else(|| "vello gpu backend failed to read rendered image".to_string())
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
            label: Some("imaging_vello device"),
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
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("imaging_vello render target"),
        size: wgpu::Extent3d {
            width: u32::from(width),
            height: u32::from(height),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = u32::from(width) * 4;
    let padded_bytes_per_row = bytes_per_row.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imaging_vello readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (texture, texture_view, readback, padded_bytes_per_row)
}

fn readback_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    readback: &wgpu::Buffer,
    bytes_per_row: u32,
    width: u16,
    height: u16,
) -> Result<Vec<u8>, Error> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("imaging_vello readback"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: u32::from(width),
            height: u32::from(height),
            depth_or_array_layers: 1,
        },
    );

    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| Error::Internal("device poll failed"))?;
    rx.recv()
        .map_err(|_| Error::Internal("map_async callback dropped"))?
        .map_err(|_| Error::Internal("buffer map failed"))?;

    let mapped = slice.get_mapped_range();
    let width_bytes = usize::from(width) * 4;

    let mut out = Vec::with_capacity(usize::from(width) * usize::from(height) * 4);
    for row in mapped.chunks_exact(bytes_per_row as usize) {
        out.extend_from_slice(&row[..width_bytes]);
    }
    drop(mapped);
    readback.unmap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use imaging::Painter;
    use kurbo::Size;
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
                    label: Some("imaging_vello test device"),
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
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imaging_vello gpu copy test texture"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback_device = device.clone();
        let readback_queue = queue.clone();
        let mut renderer = VelloGpuCopyRenderer::try_new_with_device_queue(
            device,
            queue,
            32,
            32,
            wgpu::TextureFormat::Rgba8Unorm,
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
