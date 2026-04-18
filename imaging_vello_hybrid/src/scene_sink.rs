// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::Error;
use crate::{VelloHybridRenderer, image_registry::HybridImageUploadSession};
use imaging::{
    BlurredRoundedRect, Brush as ImagingBrush, BrushRef, ClipRef, Composite, FillRef, GeometryRef,
    GlyphRunRef, GroupRef, ImageBrush as ImagingImageBrush, ImageRef, MaskMode, PaintSink,
    StrokeRef,
    record::{Scene, replay_transformed},
};
use kurbo::{Affine, Shape as _};
use peniko::{BlendMode, Brush, Color, ColorStop, ImageAlphaType, ImageData, ImageFormat, Style};
use std::sync::Arc;
use vello_common::filter_effects::{EdgeMode, Filter as VelloFilter, FilterGraph, FilterPrimitive};
use vello_common::glyph::Glyph as VelloGlyph;

#[derive(Clone, Debug)]
struct PendingMask {
    scene: Scene,
    mode: MaskMode,
    transform: Affine,
}

#[derive(Clone, Debug)]
enum LayerFrame {
    Clip,
    Group { mask: Option<Box<PendingMask>> },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PaintMode {
    Normal,
    Mask(MaskMode),
}

/// Borrowed adapter that streams `imaging` commands into an existing [`vello_hybrid::Scene`].
pub struct VelloHybridSceneSink<'a> {
    scene: &'a mut vello_hybrid::Scene,
    image_upload: Option<HybridImageUploadSession<'a>>,
    tolerance: f64,
    error: Option<Error>,
    layer_stack: Vec<LayerFrame>,
    paint_mode: PaintMode,
}

impl core::fmt::Debug for VelloHybridSceneSink<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VelloHybridSceneSink")
            .field("tolerance", &self.tolerance)
            .field("error", &self.error)
            .field("layer_stack_depth", &self.layer_stack.len())
            .field("paint_mode", &self.paint_mode)
            .finish_non_exhaustive()
    }
}

impl<'a> VelloHybridSceneSink<'a> {
    /// Wrap an existing [`vello_hybrid::Scene`].
    pub fn new(scene: &'a mut vello_hybrid::Scene) -> Self {
        Self {
            scene,
            image_upload: None,
            tolerance: 0.1,
            error: None,
            layer_stack: Vec::new(),
            paint_mode: PaintMode::Normal,
        }
    }

    /// Wrap an existing [`vello_hybrid::Scene`] and use `renderer` to upload image brushes on
    /// demand.
    ///
    /// This is the native-scene path for image brushes. Uploaded images are cached on the
    /// renderer and reused across subsequent recordings and renders.
    pub fn with_renderer(
        scene: &'a mut vello_hybrid::Scene,
        renderer: &'a mut VelloHybridRenderer,
    ) -> Self {
        Self {
            scene,
            image_upload: Some(
                renderer.begin_image_upload_session("imaging_vello_hybrid scene upload images"),
            ),
            tolerance: 0.1,
            error: None,
            layer_stack: Vec::new(),
            paint_mode: PaintMode::Normal,
        }
    }

    /// Set the tolerance used when converting rounded rectangles to paths.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Return the first deferred translation error, if any, and ensure clip/group stacks are balanced.
    pub fn finish(&mut self) -> Result<(), Error> {
        let result = if let Some(err) = self.error.take() {
            Err(err)
        } else if !self.layer_stack.is_empty() {
            Err(Error::Internal("unbalanced layer stack"))
        } else {
            Ok(())
        };

        if let Some(mut image_upload) = self.image_upload.take() {
            image_upload.finish(result.is_ok());
        }

        result
    }

    fn set_error_once(&mut self, err: Error) {
        if self.error.is_none() {
            self.error = Some(err);
        }
    }

    fn brush_to_paint(
        &mut self,
        brush: BrushRef<'_>,
        composite: Composite,
    ) -> Option<vello_common::paint::PaintType> {
        let brush = brush.to_owned().multiply_alpha(composite.alpha);
        match self.paint_mode {
            PaintMode::Normal => match brush {
                ImagingBrush::Solid(color) => Some(Brush::Solid(color)),
                ImagingBrush::Gradient(gradient) => Some(Brush::Gradient(gradient)),
                ImagingBrush::Image(image) => self.resolve_image_brush(&image).map(Brush::Image),
            },
            PaintMode::Mask(mode) => self.mask_brush_to_paint(brush, mode),
        }
    }

    fn mask_brush_to_paint(
        &mut self,
        brush: ImagingBrush,
        mode: MaskMode,
    ) -> Option<vello_common::paint::PaintType> {
        match brush {
            ImagingBrush::Solid(color) => Some(Brush::Solid(mask_color(color, mode))),
            ImagingBrush::Gradient(mut gradient) => {
                for stop in gradient.stops.iter_mut() {
                    *stop = ColorStop {
                        offset: stop.offset,
                        color: mask_color(stop.color.to_alpha_color::<peniko::color::Srgb>(), mode)
                            .into(),
                    };
                }
                Some(Brush::Gradient(gradient))
            }
            ImagingBrush::Image(image) => self
                .mask_image_brush(&image, mode)
                .and_then(|image| self.resolve_image_brush(&image))
                .map(Brush::Image),
        }
    }

    fn mask_image_brush(
        &self,
        image: &ImagingImageBrush,
        mode: MaskMode,
    ) -> Option<ImagingImageBrush> {
        let ImageRef::Raster(raster) = image.image.as_ref() else {
            return None;
        };
        let transformed = mask_image_data(raster, mode)?;
        Some(ImagingImageBrush(peniko::ImageBrush {
            image: transformed.into(),
            sampler: image.sampler,
        }))
    }

    fn resolve_image_brush(
        &mut self,
        image: &ImagingImageBrush,
    ) -> Option<vello_common::paint::Image> {
        let image_upload = self.image_upload.as_mut()?;
        image_upload.resolve_image_brush(image).ok()
    }

    fn geometry_to_path(&self, geom: GeometryRef<'_>) -> kurbo::BezPath {
        match geom {
            GeometryRef::Rect(r) => r.to_path(self.tolerance),
            GeometryRef::RoundedRect(rr) => rr.to_path(self.tolerance),
            GeometryRef::Path(p) => p.clone(),
            GeometryRef::OwnedPath(p) => p,
        }
    }

    fn clip_to_path(&mut self, clip: ClipRef<'_>) -> (Affine, kurbo::BezPath, peniko::Fill) {
        match clip {
            ClipRef::Fill {
                transform,
                shape,
                fill_rule,
            } => (transform, self.geometry_to_path(shape), fill_rule),
            ClipRef::Stroke {
                transform,
                shape,
                stroke,
            } => {
                let path = self.geometry_to_path(shape);
                let outline = kurbo::stroke(
                    path.iter(),
                    stroke,
                    &kurbo::StrokeOpts::default(),
                    self.tolerance,
                );
                (transform, outline, peniko::Fill::NonZero)
            }
        }
    }

    fn draw_glyph_run(
        &mut self,
        glyph_run: GlyphRunRef<'_>,
        glyphs: &mut dyn Iterator<Item = imaging::record::Glyph>,
    ) {
        let Some(paint) = self.brush_to_paint(glyph_run.brush, glyph_run.composite) else {
            return;
        };
        self.scene.set_transform(glyph_run.transform);
        self.scene.set_blend_mode(glyph_run.composite.blend);
        self.scene.set_paint(paint);

        match glyph_run.style {
            Style::Fill(fill_rule) => {
                self.scene.set_fill_rule(*fill_rule);
                let builder = self
                    .scene
                    .glyph_run(glyph_run.font)
                    .font_size(glyph_run.font_size)
                    .hint(glyph_run.hint)
                    .normalized_coords(glyph_run.normalized_coords);
                let builder = if let Some(transform) = glyph_run.glyph_transform {
                    builder.glyph_transform(transform)
                } else {
                    builder
                };
                let glyphs = glyphs.map(|glyph| VelloGlyph {
                    id: glyph.id,
                    x: glyph.x,
                    y: glyph.y,
                });
                builder.fill_glyphs(glyphs);
            }
            Style::Stroke(stroke) => {
                self.scene.set_stroke(stroke.clone());
                let builder = self
                    .scene
                    .glyph_run(glyph_run.font)
                    .font_size(glyph_run.font_size)
                    .hint(glyph_run.hint)
                    .normalized_coords(glyph_run.normalized_coords);
                let builder = if let Some(transform) = glyph_run.glyph_transform {
                    builder.glyph_transform(transform)
                } else {
                    builder
                };
                let glyphs = glyphs.map(|glyph| VelloGlyph {
                    id: glyph.id,
                    x: glyph.x,
                    y: glyph.y,
                });
                builder.stroke_glyphs(glyphs);
            }
        }
    }

    fn draw_blurred_rounded_rect(&mut self, draw: BlurredRoundedRect) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "filter sigma is stored as f32"
        )]
        let blur_sigma = draw.std_dev as f32;
        self.scene.set_transform(draw.transform);
        self.scene.set_fill_rule(peniko::Fill::NonZero);
        self.scene.push_layer(
            None,
            Some(draw.composite.blend),
            Some(draw.composite.alpha),
            None,
            self.filters_to_vello(&[imaging::Filter::blur(blur_sigma)]),
        );
        self.scene.set_blend_mode(Composite::default().blend);
        let color = match self.paint_mode {
            PaintMode::Normal => draw.color,
            PaintMode::Mask(mode) => mask_color(draw.color, mode),
        };
        self.scene.set_paint(Brush::Solid(color));
        let path = draw.rect.to_rounded_rect(draw.radius).to_path(self.tolerance);
        self.scene.fill_path(&path);
        self.scene.pop_layer();
    }

    fn filters_to_vello(&self, filters: &[imaging::Filter]) -> Option<VelloFilter> {
        if filters.is_empty() {
            return None;
        }

        let mut graph = FilterGraph::new();
        let mut last = None;
        for filter in filters {
            let primitive = match *filter {
                imaging::Filter::Flood { color } => FilterPrimitive::Flood { color },
                imaging::Filter::Blur {
                    std_deviation_x,
                    std_deviation_y,
                } => FilterPrimitive::GaussianBlur {
                    std_deviation: std_deviation_x.max(std_deviation_y),
                    edge_mode: EdgeMode::None,
                },
                imaging::Filter::DropShadow {
                    dx,
                    dy,
                    std_deviation_x,
                    std_deviation_y,
                    color,
                } => FilterPrimitive::DropShadow {
                    dx,
                    dy,
                    std_deviation: std_deviation_x.max(std_deviation_y),
                    color,
                    edge_mode: EdgeMode::None,
                },
                imaging::Filter::Offset { dx, dy } => FilterPrimitive::Offset { dx, dy },
            };
            last = Some(graph.add(primitive, None));
        }
        if let Some(output) = last {
            graph.set_output(output);
        }
        Some(VelloFilter {
            graph: Arc::new(graph),
        })
    }

    fn push_clip_frame(&mut self) {
        self.layer_stack.push(LayerFrame::Clip);
    }

    fn push_group_frame(&mut self, mask: Option<Box<PendingMask>>) {
        self.layer_stack.push(LayerFrame::Group { mask });
    }

    fn pop_clip_frame(&mut self) -> bool {
        match self.layer_stack.pop() {
            Some(LayerFrame::Clip) => true,
            _ => {
                self.set_error_once(Error::Internal("pop_clip underflow"));
                false
            }
        }
    }

    fn pop_group_frame(&mut self) -> Option<Option<PendingMask>> {
        match self.layer_stack.pop() {
            Some(LayerFrame::Group { mask }) => Some(mask.map(|mask| *mask)),
            _ => {
                self.set_error_once(Error::Internal("pop_group underflow"));
                None
            }
        }
    }

    fn replay_masked_subscene(&mut self, scene: &Scene, transform: Affine, mode: MaskMode) {
        let old_mode = self.paint_mode;
        self.paint_mode = PaintMode::Mask(mode);
        replay_transformed(scene, self, transform);
        self.paint_mode = old_mode;
    }

    fn apply_mask(&mut self, mask: PendingMask) {
        self.scene.push_layer(
            None,
            Some(BlendMode::new(peniko::Mix::Normal, peniko::Compose::DestIn)),
            Some(1.0),
            None,
            None,
        );
        self.push_group_frame(None);
        self.replay_masked_subscene(&mask.scene, mask.transform, mask.mode);
        if self.pop_group_frame().is_none() {
            return;
        }
        self.scene.pop_layer();
    }

    fn fallback_brush(
        &self,
        brush: BrushRef<'_>,
        composite: Composite,
    ) -> vello_common::paint::PaintType {
        match brush.to_owned().multiply_alpha(composite.alpha) {
            ImagingBrush::Solid(color) => Brush::Solid(color),
            ImagingBrush::Gradient(gradient) => Brush::Gradient(gradient),
            ImagingBrush::Image(image) => Brush::Solid(average_image_color(&image)),
        }
    }
}

impl PaintSink for VelloHybridSceneSink<'_> {
    fn push_clip(&mut self, clip: ClipRef<'_>) {
        if self.error.is_some() {
            return;
        }
        let (xf, path, fill_rule) = self.clip_to_path(clip);
        self.scene.set_transform(xf);
        self.scene.set_fill_rule(fill_rule);
        self.scene.push_clip_path(&path);
        self.push_clip_frame();
    }

    fn pop_clip(&mut self) {
        if self.error.is_some() {
            return;
        }
        if !self.pop_clip_frame() {
            return;
        }
        self.scene.pop_clip_path();
    }

    fn push_group(&mut self, group: GroupRef<'_>) {
        if self.error.is_some() {
            return;
        }
        let clip_path = group.clip.map(|clip| {
            let (xf, path, fill_rule) = self.clip_to_path(clip);
            self.scene.set_transform(xf);
            self.scene.set_fill_rule(fill_rule);
            path
        });

        let blend = Some(group.composite.blend);
        let opacity = Some(group.composite.alpha);
        let filter = self.filters_to_vello(group.filters);
        self.scene
            .push_layer(clip_path.as_ref(), blend, opacity, None, filter);
        self.push_group_frame(group.mask.map(|mask| {
            Box::new(PendingMask {
                scene: mask.mask.scene.clone(),
                mode: mask.mask.mode,
                transform: mask.transform,
            })
        }));
    }

    fn pop_group(&mut self) {
        if self.error.is_some() {
            return;
        }
        let Some(mask) = self.pop_group_frame() else {
            return;
        };
        if let Some(mask) = mask {
            self.apply_mask(mask);
        }
        self.scene.pop_layer();
    }

    fn fill(&mut self, draw: FillRef<'_>) {
        if self.error.is_some() {
            return;
        }

        let paint = self
            .brush_to_paint(draw.brush, draw.composite)
            .unwrap_or_else(|| self.fallback_brush(draw.brush, draw.composite));
        self.scene.set_transform(draw.transform);
        self.scene.set_fill_rule(draw.fill_rule);
        self.scene
            .set_paint_transform(draw.brush_transform.unwrap_or(Affine::IDENTITY));

        let (blend, paint) = match (&paint, draw.composite.blend.compose) {
            (Brush::Solid(color), peniko::Compose::Copy) if color.components[3] == 0.0 => (
                BlendMode::new(peniko::Mix::Normal, peniko::Compose::Clear),
                Brush::Solid(Color::from_rgba8(0, 0, 0, 255)),
            ),
            _ => (draw.composite.blend, paint),
        };

        self.scene.set_blend_mode(blend);
        self.scene.set_paint(paint);

        match draw.shape {
            GeometryRef::Rect(r) => self.scene.fill_rect(&r),
            GeometryRef::RoundedRect(rr) => {
                let path = rr.to_path(self.tolerance);
                self.scene.fill_path(&path);
            }
            GeometryRef::Path(p) => self.scene.fill_path(p),
            GeometryRef::OwnedPath(p) => self.scene.fill_path(&p),
        }
    }

    fn stroke(&mut self, draw: StrokeRef<'_>) {
        if self.error.is_some() {
            return;
        }

        let paint = self
            .brush_to_paint(draw.brush, draw.composite)
            .unwrap_or_else(|| self.fallback_brush(draw.brush, draw.composite));
        self.scene.set_transform(draw.transform);
        self.scene.set_stroke(draw.stroke.clone());
        self.scene
            .set_paint_transform(draw.brush_transform.unwrap_or(Affine::IDENTITY));

        let (blend, paint) = match (&paint, draw.composite.blend.compose) {
            (Brush::Solid(color), peniko::Compose::Copy) if color.components[3] == 0.0 => (
                BlendMode::new(peniko::Mix::Normal, peniko::Compose::Clear),
                Brush::Solid(Color::from_rgba8(0, 0, 0, 255)),
            ),
            _ => (draw.composite.blend, paint),
        };

        self.scene.set_blend_mode(blend);
        self.scene.set_paint(paint);

        match draw.shape {
            GeometryRef::Rect(r) => self.scene.stroke_rect(&r),
            GeometryRef::RoundedRect(rr) => {
                let path = rr.to_path(self.tolerance);
                self.scene.stroke_path(&path);
            }
            GeometryRef::Path(p) => self.scene.stroke_path(p),
            GeometryRef::OwnedPath(p) => self.scene.stroke_path(&p),
        }
    }

    fn glyph_run(
        &mut self,
        draw: GlyphRunRef<'_>,
        glyphs: &mut dyn Iterator<Item = imaging::record::Glyph>,
    ) {
        if self.error.is_some() {
            return;
        }
        self.draw_glyph_run(draw, glyphs);
    }

    fn blurred_rounded_rect(&mut self, draw: BlurredRoundedRect) {
        if self.error.is_some() {
            return;
        }
        self.draw_blurred_rounded_rect(draw);
    }
}

fn average_image_color(image: &ImagingImageBrush) -> Color {
    let ImageRef::Raster(raster) = image.image.as_ref() else {
        return Color::TRANSPARENT;
    };
    let bytes = raster.data.as_ref();
    let mut r_sum = 0_u64;
    let mut g_sum = 0_u64;
    let mut b_sum = 0_u64;
    let mut a_sum = 0_u64;
    let mut count = 0_u64;

    for px in bytes.chunks_exact(4) {
        let (r, g, b, a) = match raster.format {
            ImageFormat::Rgba8 => (px[0], px[1], px[2], px[3]),
            ImageFormat::Bgra8 => (px[2], px[1], px[0], px[3]),
            _ => return Color::TRANSPARENT,
        };

        let (r, g, b) = match raster.alpha_type {
            ImageAlphaType::Alpha => (u32::from(r), u32::from(g), u32::from(b)),
            ImageAlphaType::AlphaPremultiplied if a == 0 => (0, 0, 0),
            ImageAlphaType::AlphaPremultiplied => {
                let alpha = u32::from(a);
                (
                    (u32::from(r) * 255 + alpha / 2) / alpha,
                    (u32::from(g) * 255 + alpha / 2) / alpha,
                    (u32::from(b) * 255 + alpha / 2) / alpha,
                )
            }
        };

        r_sum += u64::from(r);
        g_sum += u64::from(g);
        b_sum += u64::from(b);
        a_sum += u64::from(a);
        count += 1;
    }

    if count == 0 {
        return Color::TRANSPARENT;
    }

    Color::from_rgba8(
        u8::try_from(r_sum / count).expect("average red stays within u8"),
        u8::try_from(g_sum / count).expect("average green stays within u8"),
        u8::try_from(b_sum / count).expect("average blue stays within u8"),
        u8::try_from(a_sum / count).expect("average alpha stays within u8"),
    )
}

fn mask_color(color: Color, mode: MaskMode) -> Color {
    let coverage = match mode {
        MaskMode::Alpha => color.components[3],
        MaskMode::Luminance => {
            let alpha = color.components[3];
            let luminance = color.components[0] * 0.2126
                + color.components[1] * 0.7152
                + color.components[2] * 0.0722;
            alpha * luminance
        }
    }
    .clamp(0.0, 1.0);
    Color::from_rgba8(255, 255, 255, normalized_to_u8(coverage))
}

fn mask_image_data(image: &ImageData, mode: MaskMode) -> Option<ImageData> {
    let mut out = Vec::with_capacity(image.data.as_ref().len());
    for px in image.data.as_ref().chunks_exact(4) {
        let (r, g, b, a) = match image.format {
            ImageFormat::Rgba8 => (px[0], px[1], px[2], px[3]),
            ImageFormat::Bgra8 => (px[2], px[1], px[0], px[3]),
            _ => return None,
        };
        let coverage = mask_coverage_from_pixel(r, g, b, a, image.alpha_type, mode);
        out.extend_from_slice(&[255, 255, 255, coverage]);
    }
    Some(ImageData {
        data: peniko::Blob::from(out),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: image.width,
        height: image.height,
    })
}

fn mask_coverage_from_pixel(
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    alpha_type: ImageAlphaType,
    mode: MaskMode,
) -> u8 {
    match mode {
        MaskMode::Alpha => a,
        MaskMode::Luminance => match alpha_type {
            ImageAlphaType::Alpha => {
                let alpha = f32::from(a) / 255.0;
                let luminance = (f32::from(r) / 255.0) * 0.2126
                    + (f32::from(g) / 255.0) * 0.7152
                    + (f32::from(b) / 255.0) * 0.0722;
                normalized_to_u8(alpha * luminance)
            }
            ImageAlphaType::AlphaPremultiplied => {
                let premul_luma =
                    f32::from(r) * 0.2126 + f32::from(g) * 0.7152 + f32::from(b) * 0.0722;
                byte_value_to_u8(premul_luma)
            }
        },
    }
}

fn normalized_to_u8(value: f32) -> u8 {
    let scaled = (value.clamp(0.0, 1.0) * 255.0).round();
    byte_value_to_u8(scaled)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "value is clamped to the u8 range first"
)]
fn byte_value_to_u8(value: f32) -> u8 {
    u8::try_from(value.clamp(0.0, 255.0) as i16).expect("value is clamped to u8 range")
}

#[cfg(test)]
mod tests {
    use super::*;
    use imaging::MaskRef;
    use peniko::{Blob, ImageAlphaType, ImageData, ImageFormat};
    use std::sync::Arc;

    #[test]
    fn hybrid_scene_sink_reports_clip_underflow() {
        let mut scene = vello_hybrid::Scene::new(32, 32);
        scene.reset();
        let mut sink = VelloHybridSceneSink::new(&mut scene);
        sink.pop_clip();
        assert!(matches!(
            sink.finish(),
            Err(Error::Internal("pop_clip underflow"))
        ));
    }

    #[test]
    fn hybrid_scene_sink_supports_filters() {
        let mut scene = vello_hybrid::Scene::new(32, 32);
        scene.reset();
        let mut sink = VelloHybridSceneSink::new(&mut scene);
        sink.push_group(GroupRef::new().with_filters(&[imaging::Filter::blur(2.0)]));
        sink.pop_group();
        assert!(matches!(sink.finish(), Ok(())));
    }

    #[test]
    fn hybrid_scene_sink_falls_back_for_image_brushes_without_resolver() {
        let mut scene = vello_hybrid::Scene::new(32, 32);
        scene.reset();
        let mut sink = VelloHybridSceneSink::new(&mut scene);
        let image = ImagingBrush::Image(ImagingImageBrush::from(ImageData {
            data: Blob::new(Arc::new([255_u8; 16])),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::Alpha,
            width: 2,
            height: 2,
        }));
        sink.fill(FillRef::new(kurbo::Rect::new(0.0, 0.0, 8.0, 8.0), &image));
        assert!(matches!(sink.finish(), Ok(())));
    }

    #[test]
    fn hybrid_scene_sink_supports_luminance_masks() {
        let mut mask = Scene::new();
        mask.fill(FillRef::new(
            kurbo::Rect::new(0.0, 0.0, 8.0, 8.0),
            Color::WHITE,
        ));

        let mut scene = vello_hybrid::Scene::new(32, 32);
        scene.reset();
        let mut sink = VelloHybridSceneSink::new(&mut scene);
        sink.push_group(GroupRef::new().with_mask(MaskRef::new(MaskMode::Luminance, &mask)));
        sink.fill(FillRef::new(
            kurbo::Rect::new(1.0, 1.0, 7.0, 7.0),
            Color::BLACK,
        ));
        sink.pop_group();
        assert!(matches!(sink.finish(), Ok(())));
    }

    #[test]
    fn hybrid_scene_sink_supports_alpha_masks() {
        let mut mask = Scene::new();
        mask.fill(FillRef::new(
            kurbo::Rect::new(0.0, 0.0, 8.0, 8.0),
            Color::WHITE,
        ));

        let mut scene = vello_hybrid::Scene::new(32, 32);
        scene.reset();
        let mut sink = VelloHybridSceneSink::new(&mut scene);
        sink.push_group(GroupRef::new().with_mask(MaskRef::new(MaskMode::Alpha, &mask)));
        sink.fill(FillRef::new(
            kurbo::Rect::new(1.0, 1.0, 7.0, 7.0),
            Color::BLACK,
        ));
        sink.pop_group();
        assert!(matches!(sink.finish(), Ok(())));
    }
}
