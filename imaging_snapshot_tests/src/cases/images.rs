// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use imaging::{
    Brush as ImagingBrush, Composite, ExternalImage, ExternalImageId,
    ImageBrush as ImagingImageBrush, PaintSink, Painter, record::Geometry,
};
use kurbo::{Affine, BezPath, Point, Rect, RoundedRect, Stroke};
use peniko::{BlendMode, Brush, Color, Extend, ImageBrush, ImageQuality, Mix};

use super::SnapshotCase;
use super::util::{background, test_image};

pub(crate) struct GmImageBrushes;
impl SnapshotCase for GmImageBrushes {
    fn name(&self) -> &'static str {
        "gm_image_brushes"
    }

    fn supports_backend(&self, backend: &str) -> bool {
        matches!(
            backend,
            "skia" | "tiny_skia" | "vello_cpu" | "vello" | "vello_hybrid"
        )
    }

    fn run(&self, sink: &mut dyn PaintSink, width: f64, height: f64) {
        background(sink, width, height, Color::from_rgb8(244, 244, 240));
        let mut painter = Painter::new(sink);

        let left = Brush::Image(
            ImageBrush::from(test_image())
                .with_extend(Extend::Pad)
                .with_quality(ImageQuality::Medium),
        );
        painter
            .fill(
                Geometry::RoundedRect(RoundedRect::new(
                    width * 0.08,
                    height * 0.12,
                    width * 0.42,
                    height * 0.84,
                    24.0,
                )),
                &left,
            )
            .brush_transform(Some(Affine::translate((-18.0, -12.0))))
            .draw();

        let diamond_brush = Brush::Image(
            ImageBrush::from(test_image())
                .with_x_extend(Extend::Reflect)
                .with_y_extend(Extend::Pad)
                .with_quality(ImageQuality::Medium),
        );
        painter
            .fill(
                Geometry::Path(diamond(
                    width * 0.72,
                    height * 0.48,
                    width * 0.2,
                    height * 0.28,
                )),
                &diamond_brush,
            )
            .brush_transform(Some(Affine::translate((-104.0, -8.0))))
            .composite(Composite::new(BlendMode::from(Mix::Multiply), 1.0))
            .draw();

        let frame_stroke = Stroke::new(20.0);
        let frame_brush = Brush::Image(
            ImageBrush::from(test_image())
                .with_extend(Extend::Reflect)
                .with_quality(ImageQuality::Low),
        );
        painter
            .stroke(
                Geometry::RoundedRect(RoundedRect::new(
                    width * 0.5,
                    height * 0.58,
                    width * 0.9,
                    height * 0.9,
                    30.0,
                )),
                &frame_stroke,
                &frame_brush,
            )
            .brush_transform(Some(Affine::translate((-126.0, -120.0))))
            .draw();

        let glaze = Brush::Solid(Color::from_rgba8(255, 255, 255, 170));
        painter
            .fill(
                Geometry::Rect(Rect::new(
                    width * 0.46,
                    height * 0.54,
                    width * 0.94,
                    height * 0.94,
                )),
                &glaze,
            )
            .draw();
    }
}

/// External image id used by the Skia external-image snapshot resolver.
pub const EXTERNAL_IMAGE_BRUSH_ID: ExternalImageId = ExternalImageId(42);
/// Pixel size of the Skia external-image snapshot source texture.
pub const EXTERNAL_IMAGE_BRUSH_SIZE: (u32, u32) = (2, 2);
/// RGBA8 pixels uploaded into the Skia external-image snapshot source texture.
pub const EXTERNAL_IMAGE_BRUSH_PIXELS: &[u8] = &[
    0xff, 0x20, 0x10, 0xff, 0x20, 0xc8, 0x40, 0xff, 0x20, 0x60, 0xff, 0xff, 0xff, 0xe0, 0x20, 0xff,
];

pub(crate) struct GmExternalImageBrush;
impl SnapshotCase for GmExternalImageBrush {
    fn name(&self) -> &'static str {
        "gm_external_image_brush"
    }

    fn supports_backend(&self, backend: &str) -> bool {
        backend == "skia"
    }

    fn run(&self, sink: &mut dyn PaintSink, width: f64, height: f64) {
        background(sink, width, height, Color::from_rgb8(244, 244, 240));
        let mut painter = Painter::new(sink);
        let image = ExternalImage::new(
            EXTERNAL_IMAGE_BRUSH_ID,
            EXTERNAL_IMAGE_BRUSH_SIZE.0,
            EXTERNAL_IMAGE_BRUSH_SIZE.1,
            peniko::ImageAlphaType::AlphaPremultiplied,
        );
        let brush = ImagingBrush::Image(
            ImagingImageBrush::from(image)
                .with_extend(Extend::Reflect)
                .with_quality(ImageQuality::Low),
        );

        painter
            .fill(
                Geometry::RoundedRect(RoundedRect::new(
                    width * 0.15,
                    height * 0.18,
                    width * 0.85,
                    height * 0.82,
                    28.0,
                )),
                &brush,
            )
            .brush_transform(Some(Affine::scale_non_uniform(48.0, 32.0)))
            .draw();

        painter
            .stroke(
                Geometry::RoundedRect(RoundedRect::new(
                    width * 0.15,
                    height * 0.18,
                    width * 0.85,
                    height * 0.82,
                    28.0,
                )),
                &Stroke::new(6.0),
                &Brush::Solid(Color::from_rgba8(18, 28, 42, 220)),
            )
            .draw();
    }
}

fn diamond(cx: f64, cy: f64, rx: f64, ry: f64) -> BezPath {
    let mut path = BezPath::new();
    path.move_to(Point::new(cx, cy - ry));
    path.line_to(Point::new(cx + rx, cy));
    path.line_to(Point::new(cx, cy + ry));
    path.line_to(Point::new(cx - rx, cy));
    path.close_path();
    path
}
