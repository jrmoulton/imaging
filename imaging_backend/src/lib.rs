//! Generic backend-facing render traits and target types for `imaging`.
//!
//! This crate intentionally sits above `imaging` and owns the renderer-facing
//! target, and readback types so the core `imaging` crate can stay backend-agnostic.

use std::{sync::Arc, sync::mpsc};

use imaging::{PaintSink, record::Scene, record::replay};
use kurbo::Size;
use peniko::{Blob, ImageAlphaType, ImageData, ImageFormat};

/// Backend-neutral render output returned from a renderer readback path.
#[derive(Debug)]
pub enum RenderOutput {
    /// CPU-readable image output.
    Image(ImageData),
    /// GPU texture output for backends that render to a GPU target.
    GpuTexture(wgpu::TextureView),
}

impl RenderOutput {
    /// Return the image payload when this output is CPU-backed.
    pub fn into_image(self) -> Option<ImageData> {
        match self {
            Self::Image(image) => Some(image),
            Self::GpuTexture(_) => None,
        }
    }

    /// Read a GPU texture output into an image using the supplied `wgpu` device and queue.
    pub fn into_image_with(self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<ImageData> {
        match self {
            Self::Image(image) => Some(image),
            Self::GpuTexture(texture) => read_texture_view_to_image(&texture, device, queue).ok(),
        }
    }
}

/// CPU target buffer byte order understood by target renderers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuBufferChannelOrder {
    /// RGBA8 byte order.
    Rgba8,
    /// BGRA8 byte order.
    Bgra8,
}

/// CPU target buffer alpha encoding understood by target renderers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuBufferAlphaMode {
    /// The destination should be treated as fully opaque.
    Opaque,
    /// The destination stores premultiplied alpha.
    Premultiplied,
}

/// CPU target buffer pixel layout understood by target renderers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuBufferFormat {
    /// Channel order of each 8-bit pixel.
    pub channel_order: CpuBufferChannelOrder,
    /// Alpha encoding of each pixel.
    pub alpha_mode: CpuBufferAlphaMode,
}

impl CpuBufferFormat {
    #[allow(
        non_upper_case_globals,
        reason = "Renderer format constants use type-like naming."
    )]
    /// Opaque RGBA8 pixels.
    pub const Rgba8Opaque: Self = Self {
        channel_order: CpuBufferChannelOrder::Rgba8,
        alpha_mode: CpuBufferAlphaMode::Opaque,
    };

    #[allow(
        non_upper_case_globals,
        reason = "Renderer format constants use type-like naming."
    )]
    /// Opaque BGRA8 pixels.
    pub const Bgra8Opaque: Self = Self {
        channel_order: CpuBufferChannelOrder::Bgra8,
        alpha_mode: CpuBufferAlphaMode::Opaque,
    };

    #[allow(
        non_upper_case_globals,
        reason = "Renderer format constants use type-like naming."
    )]
    /// Premultiplied RGBA8 pixels.
    pub const Rgba8Premultiplied: Self = Self {
        channel_order: CpuBufferChannelOrder::Rgba8,
        alpha_mode: CpuBufferAlphaMode::Premultiplied,
    };

    #[allow(
        non_upper_case_globals,
        reason = "Renderer format constants use type-like naming."
    )]
    /// Premultiplied BGRA8 pixels.
    pub const Bgra8Premultiplied: Self = Self {
        channel_order: CpuBufferChannelOrder::Bgra8,
        alpha_mode: CpuBufferAlphaMode::Premultiplied,
    };
}

/// CPU target metadata used for compatibility checks before binding a renderer.
#[derive(Clone, Copy, Debug)]
pub struct CpuBufferTargetInfo {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Distance in bytes between the start of adjacent rows.
    pub bytes_per_row: usize,
    /// Pixel format of the destination buffer.
    pub format: CpuBufferFormat,
}

/// Metadata describing a GPU texture-view render target.
#[derive(Clone, Copy, Debug)]
pub struct GpuTextureTargetInfo {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Texture format of the destination texture.
    pub format: wgpu::TextureFormat,
}

/// Metadata describing a GPU texture-copy render target.
#[derive(Clone, Copy, Debug)]
pub struct GpuTextureCopyTargetInfo {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Texture format of the destination texture.
    pub format: wgpu::TextureFormat,
}

/// Borrowed CPU target buffer passed to a renderer that writes directly into caller-owned memory.
#[derive(Debug)]
pub struct CpuBufferTarget<'a> {
    /// Destination pixel bytes.
    pub buffer: &'a mut [u8],
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Distance in bytes between the start of adjacent rows.
    pub bytes_per_row: usize,
    /// Pixel format of the destination buffer.
    pub format: CpuBufferFormat,
}

impl CpuBufferTarget<'_> {
    /// Return buffer metadata without exposing the caller-owned storage.
    pub fn info(&self) -> CpuBufferTargetInfo {
        CpuBufferTargetInfo {
            width: self.width,
            height: self.height,
            bytes_per_row: self.bytes_per_row,
            format: self.format,
        }
    }
}

/// Small trait implemented by caller-provided render targets.
pub trait Target {
    /// Metadata used to validate compatibility before binding a target.
    type Info;

    /// Return target metadata without transferring ownership of the target itself.
    fn info(&self) -> Self::Info;
}

impl Target for CpuBufferTarget<'_> {
    type Info = CpuBufferTargetInfo;

    fn info(&self) -> Self::Info {
        self.info()
    }
}

/// Owned GPU target description passed to a renderer that writes directly into a GPU texture.
#[derive(Debug)]
pub struct GpuTextureTarget {
    /// Device that owns the target texture.
    pub device: wgpu::Device,
    /// Queue used to submit rendering commands.
    pub queue: wgpu::Queue,
    /// Texture view to render into.
    pub texture_view: wgpu::TextureView,
}

impl Target for GpuTextureTarget {
    type Info = GpuTextureTargetInfo;

    fn info(&self) -> Self::Info {
        let texture = self.texture_view.texture();
        let size = texture.size();
        GpuTextureTargetInfo {
            width: size.width,
            height: size.height,
            format: texture.format(),
        }
    }
}

/// Owned GPU target description passed to a renderer that renders into a caller-owned texture.
#[derive(Debug)]
pub struct GpuTextureCopyTarget {
    /// Device that owns the target texture.
    pub device: wgpu::Device,
    /// Queue used to submit rendering commands.
    pub queue: wgpu::Queue,
    /// Texture to render into.
    pub texture: wgpu::Texture,
}

impl Target for GpuTextureCopyTarget {
    type Info = GpuTextureCopyTargetInfo;

    fn info(&self) -> Self::Info {
        let size = self.texture.size();
        GpuTextureCopyTargetInfo {
            width: size.width,
            height: size.height,
            format: self.texture.format(),
        }
    }
}

/// A source of paint commands that can be replayed into any `imaging` paint sink.
pub trait RenderSource {
    /// Stream this source into the provided sink.
    fn paint_into(&mut self, sink: &mut dyn PaintSink);
}

impl RenderSource for Scene {
    fn paint_into(&mut self, sink: &mut dyn PaintSink) {
        replay(self, sink);
    }
}

/// Adapter for immediate-mode paint closures.
#[derive(Debug)]
pub struct PaintFn<F>(pub F);

impl<F> RenderSource for PaintFn<F>
where
    F: FnMut(&mut dyn PaintSink),
{
    fn paint_into(&mut self, sink: &mut dyn PaintSink) {
        (self.0)(sink);
    }
}

/// Backend that can render a source into backend-defined targets or images.
pub trait Backend {
    /// Backend-specific error type.
    type Error;
    /// Backend-specific image output type.
    type Image;
    /// Backend-defined CPU buffer target handle.
    type BufferTarget<'a>;
    /// Backend-defined texture target handle.
    type TextureTarget<'a>;

    /// Report whether this backend should be used for the provided CPU buffer target.
    fn supports_buffer_target(_target: &CpuBufferTargetInfo) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        Ok(())
    }

    /// Report whether this backend should be used for the provided texture target.
    fn supports_texture_target(_target: &GpuTextureTargetInfo) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        Ok(())
    }

    /// Render `source` into a CPU buffer target.
    fn render_to_buffer<'a>(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        target: Self::BufferTarget<'a>,
    ) -> Result<(), Self::Error>;

    /// Render `source` into a texture target.
    fn render_to_texture<'a>(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        target: Self::TextureTarget<'a>,
    ) -> Result<(), Self::Error>;

    /// Render `source` into a new image of the requested size.
    fn render_to_image(
        &mut self,
        size: Size,
        source: &mut dyn RenderSource,
        width: u32,
        height: u32,
    ) -> Result<Self::Image, Self::Error>;
}

fn read_texture_view_to_image(
    texture_view: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<ImageData, String> {
    let texture = texture_view.texture();
    let size = texture.size();
    let width = size.width;
    let height = size.height;
    let (image_format, bytes_per_pixel) = match texture.format() {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
            (ImageFormat::Rgba8, 4_usize)
        }
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            (ImageFormat::Bgra8, 4_usize)
        }
        format => {
            return Err(format!(
                "unsupported texture format for readback: {format:?}"
            ));
        }
    };
    let width_bytes = width
        .checked_mul(
            u32::try_from(bytes_per_pixel)
                .expect("bytes per pixel fits in u32 for supported formats"),
        )
        .ok_or_else(|| "texture row byte count overflow".to_string())?;
    let padded_bytes_per_row = width_bytes
        .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .checked_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .ok_or_else(|| "texture row padding overflow".to_string())?;
    let total_size = padded_bytes_per_row
        .checked_mul(height)
        .ok_or_else(|| "texture readback buffer size overflow".to_string())?;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imaging readback"),
        size: u64::from(total_size),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("imaging readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("send imaging readback completion");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("receive imaging readback completion")
        .map_err(|err| format!("{err:?}"))?;

    let mapped = slice.get_mapped_range();
    let width_bytes_usize =
        usize::try_from(width_bytes).expect("width bytes fit usize for mapped readback");
    let padded_bytes_per_row_usize =
        usize::try_from(padded_bytes_per_row).expect("row pitch fits usize for mapped readback");
    let height_usize = usize::try_from(height).expect("height fits usize for mapped readback");
    let mut data = vec![0_u8; width_bytes_usize * height_usize];
    for y in 0..height_usize {
        let src_start = y * padded_bytes_per_row_usize;
        let src_end = src_start + width_bytes_usize;
        let dst_start = y * width_bytes_usize;
        let dst_end = dst_start + width_bytes_usize;
        data[dst_start..dst_end].copy_from_slice(&mapped[src_start..src_end]);
    }
    drop(mapped);
    readback.unmap();

    Ok(ImageData {
        data: Blob::new(Arc::new(data)),
        format: image_format,
        width,
        height,
        alpha_type: ImageAlphaType::AlphaPremultiplied,
    })
}
