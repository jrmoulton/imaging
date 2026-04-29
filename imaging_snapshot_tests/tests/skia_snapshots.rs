// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image snapshot tests for `imaging_skia` using `kompari`.

#![cfg(feature = "skia")]

use imaging_skia::SkiaCpuRenderer;
use imaging_snapshot_tests::cases::{
    DEFAULT_HEIGHT, DEFAULT_WIDTH, EXTERNAL_IMAGE_BRUSH_ID, EXTERNAL_IMAGE_BRUSH_PIXELS,
    EXTERNAL_IMAGE_BRUSH_SIZE, build_scene,
};

mod common;

fn render_case(case: &dyn imaging_snapshot_tests::cases::SnapshotCase) -> imaging::RgbaImage {
    let width = DEFAULT_WIDTH;
    let height = DEFAULT_HEIGHT;
    let w = f64::from(width);
    let h = f64::from(height);

    let scene = build_scene(case, w, h);
    let mut renderer = SkiaCpuRenderer::new();
    renderer
        .render_scene(&scene, width, height)
        .expect("render image")
}

fn render_external_image_case(
    case: &dyn imaging_snapshot_tests::cases::SnapshotCase,
) -> imaging::RgbaImage {
    struct Resolver {
        texture: imaging_skia::wgpu::Texture,
        view: imaging_skia::wgpu::TextureView,
    }

    impl imaging_wgpu::ExternalImageResolver for Resolver {
        fn resolve_external_image(
            &mut self,
            image: imaging::ExternalImage,
        ) -> Option<imaging_wgpu::ResolvedExternalImage> {
            (image.id == EXTERNAL_IMAGE_BRUSH_ID).then(|| imaging_wgpu::ResolvedExternalImage {
                texture: self.texture.clone(),
                view: self.view.clone(),
                format: self.texture.format(),
                width: self.texture.width(),
                height: self.texture.height(),
            })
        }
    }

    let width = DEFAULT_WIDTH;
    let height = DEFAULT_HEIGHT;
    let width_u32 = u32::from(width);
    let height_u32 = u32::from(height);
    let w = f64::from(width);
    let h = f64::from(height);
    let scene = build_scene(case, w, h);

    let instance = imaging_skia::wgpu::Instance::default();
    let adapter = pollster::block_on(
        instance.request_adapter(&imaging_skia::wgpu::RequestAdapterOptions::default()),
    )
    .expect("request wgpu adapter");
    let (device, queue) = pollster::block_on(
        adapter.request_device(&imaging_skia::wgpu::DeviceDescriptor::default()),
    )
    .expect("request wgpu device");
    let mut renderer = imaging_skia::SkiaRenderer::new(adapter, device.clone(), queue.clone())
        .expect("create Skia GPU renderer");

    let source_texture = device.create_texture(&imaging_skia::wgpu::TextureDescriptor {
        label: Some("imaging snapshot external image source"),
        size: imaging_skia::wgpu::Extent3d {
            width: EXTERNAL_IMAGE_BRUSH_SIZE.0,
            height: EXTERNAL_IMAGE_BRUSH_SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: imaging_skia::wgpu::TextureDimension::D2,
        format: imaging_skia::wgpu::TextureFormat::Rgba8Unorm,
        usage: imaging_skia::wgpu::TextureUsages::TEXTURE_BINDING
            | imaging_skia::wgpu::TextureUsages::COPY_DST
            | imaging_skia::wgpu::TextureUsages::COPY_SRC
            | imaging_skia::wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    queue.write_texture(
        imaging_skia::wgpu::TexelCopyTextureInfo {
            texture: &source_texture,
            mip_level: 0,
            origin: imaging_skia::wgpu::Origin3d::ZERO,
            aspect: imaging_skia::wgpu::TextureAspect::All,
        },
        EXTERNAL_IMAGE_BRUSH_PIXELS,
        imaging_skia::wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * EXTERNAL_IMAGE_BRUSH_SIZE.0),
            rows_per_image: Some(EXTERNAL_IMAGE_BRUSH_SIZE.1),
        },
        imaging_skia::wgpu::Extent3d {
            width: EXTERNAL_IMAGE_BRUSH_SIZE.0,
            height: EXTERNAL_IMAGE_BRUSH_SIZE.1,
            depth_or_array_layers: 1,
        },
    );
    device
        .poll(imaging_skia::wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll after source upload");
    let source_view =
        source_texture.create_view(&imaging_skia::wgpu::TextureViewDescriptor::default());

    let target = device.create_texture(&imaging_skia::wgpu::TextureDescriptor {
        label: Some("imaging snapshot external image target"),
        size: imaging_skia::wgpu::Extent3d {
            width: width_u32,
            height: height_u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: imaging_skia::wgpu::TextureDimension::D2,
        format: imaging_skia::wgpu::TextureFormat::Rgba8Unorm,
        usage: imaging_skia::wgpu::TextureUsages::RENDER_ATTACHMENT
            | imaging_skia::wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let mut resolver = Resolver {
        texture: source_texture,
        view: source_view,
    };
    let mut source = &scene;
    renderer
        .render_source_into_texture_with_external_images(&mut source, &target, &mut resolver)
        .expect("render external image scene");

    read_rgba_texture(&device, &queue, &target, width_u32, height_u32)
}

fn read_rgba_texture(
    device: &imaging_skia::wgpu::Device,
    queue: &imaging_skia::wgpu::Queue,
    texture: &imaging_skia::wgpu::Texture,
    width: u32,
    height: u32,
) -> imaging::RgbaImage {
    let row_bytes = width * 4;
    let padded_row_bytes = row_bytes.div_ceil(256) * 256;
    let buffer = device.create_buffer(&imaging_skia::wgpu::BufferDescriptor {
        label: Some("imaging snapshot readback"),
        size: u64::from(padded_row_bytes) * u64::from(height),
        usage: imaging_skia::wgpu::BufferUsages::MAP_READ
            | imaging_skia::wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&imaging_skia::wgpu::CommandEncoderDescriptor {
            label: Some("imaging snapshot readback"),
        });
    encoder.copy_texture_to_buffer(
        imaging_skia::wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: imaging_skia::wgpu::Origin3d::ZERO,
            aspect: imaging_skia::wgpu::TextureAspect::All,
        },
        imaging_skia::wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: imaging_skia::wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: None,
            },
        },
        imaging_skia::wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(imaging_skia::wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("send readback result");
    });
    device
        .poll(imaging_skia::wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll readback");
    rx.recv()
        .expect("receive readback result")
        .expect("map readback");

    let mapped = slice.get_mapped_range();
    let mut image = imaging::RgbaImage::new(width, height);
    for y in 0..height as usize {
        let src = y * padded_row_bytes as usize;
        let dst = y * row_bytes as usize;
        image.data[dst..dst + row_bytes as usize]
            .copy_from_slice(&mapped[src..src + row_bytes as usize]);
    }
    drop(mapped);
    buffer.unmap();
    image
}

#[test]
fn snapshots() {
    let mut errors = Vec::new();
    common::run_cases_with(
        "skia",
        |case| {
            if case.name() == "gm_external_image_brush" {
                render_external_image_case(case)
            } else {
                render_case(case)
            }
        },
        |case| case.skia_max_diff_pixels(),
        &mut errors,
    );
    common::assert_no_snapshot_errors(errors);
}
