// Copyright 2026 the Imaging Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{
    Error, affine_to_matrix, apply_stroke_style, bez_to_sk_path, brush_to_paint,
    build_filter_chain, f64_to_f32, geometry_to_bez_path, geometry_to_sk_path, map_blend_mode,
    path_with_fill_rule, skia_font_from_glyph_run,
};
use imaging::{
    BlurredRoundedRect, ClipRef, FillRef, GeometryRef, GlyphRunRef, GroupRef, MaskMode, PaintSink,
    RetainedDrawRef, RetainedEvictionPolicy, RetainedTransformPolicy, StrokeRef,
    record::{self, replay, replay_transformed},
};
use kurbo::{Affine, Rect, Shape as _};
use skia_safe as sk;
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
};

pub(crate) type MaskImageCacheHandle = Rc<RefCell<MaskImageCache>>;
pub(crate) type RetainedImageCacheHandle = Rc<RefCell<RetainedImageCache>>;
const MAX_MASK_CACHE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct MaskImageCache {
    images: HashMap<MaskImageCacheKey, sk::Image>,
    lru: VecDeque<MaskImageCacheKey>,
    bytes_used: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MaskImageCacheKey {
    retained_ptr: usize,
    mode: MaskMode,
    linear_transform: [u64; 4],
    tolerance_bits: u64,
    width: i32,
    height: i32,
}

impl MaskImageCache {
    pub(crate) fn new() -> Self {
        Self {
            images: HashMap::new(),
            lru: VecDeque::new(),
            bytes_used: 0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.images.clear();
        self.lru.clear();
        self.bytes_used = 0;
    }

    fn touch(&mut self, key: MaskImageCacheKey) {
        if let Some(idx) = self.lru.iter().position(|existing| *existing == key) {
            self.lru.remove(idx);
        }
        self.lru.push_back(key);
    }

    fn approx_bytes(width: i32, height: i32) -> usize {
        (width.max(0) as usize)
            .saturating_mul(height.max(0) as usize)
            .saturating_mul(4)
    }

    fn get(&mut self, key: MaskImageCacheKey) -> Option<sk::Image> {
        let image = self.images.get(&key)?.clone();
        self.touch(key);
        Some(image)
    }

    fn insert(&mut self, key: MaskImageCacheKey, image: sk::Image) {
        if let Some(old) = self.images.insert(key, image) {
            let dims = old.dimensions();
            self.bytes_used = self
                .bytes_used
                .saturating_sub(Self::approx_bytes(dims.width, dims.height));
        }
        self.bytes_used = self
            .bytes_used
            .saturating_add(Self::approx_bytes(key.width, key.height));
        self.touch(key);
        while self.bytes_used > MAX_MASK_CACHE_BYTES {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            if let Some(old) = self.images.remove(&oldest) {
                let dims = old.dimensions();
                self.bytes_used = self
                    .bytes_used
                    .saturating_sub(Self::approx_bytes(dims.width, dims.height));
            }
        }
    }
}

impl Default for MaskImageCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub(crate) struct RetainedImageCache {
    images: HashMap<RetainedImageCacheKey, CachedRetainedImage>,
    current_mark: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RetainedImageCacheKey {
    retained_id: u64,
    linear_transform: [u64; 4],
}

#[derive(Clone, Debug)]
struct CachedRetainedImage {
    image: sk::Image,
    used_mark: bool,
    eviction: RetainedEvictionPolicy,
}

impl RetainedImageCache {
    pub(crate) fn new() -> Self {
        Self {
            images: HashMap::new(),
            current_mark: false,
        }
    }

    pub(crate) fn flip_mark(&mut self) {
        self.current_mark = !self.current_mark;
    }

    pub(crate) fn evict_unmarked(&mut self) {
        let current_mark = self.current_mark;
        self.images.retain(|_, cached| match cached.eviction {
            RetainedEvictionPolicy::UntilUnused => cached.used_mark == current_mark,
            RetainedEvictionPolicy::Manual => true,
        });
    }

    pub(crate) fn clear(&mut self) {
        self.images.clear();
    }

    fn get(&mut self, key: RetainedImageCacheKey) -> Option<sk::Image> {
        let cached = self.images.get_mut(&key)?;
        cached.used_mark = self.current_mark;
        Some(cached.image.clone())
    }

    fn insert(
        &mut self,
        key: RetainedImageCacheKey,
        image: sk::Image,
        eviction: RetainedEvictionPolicy,
    ) {
        self.images.insert(
            key,
            CachedRetainedImage {
                image,
                used_mark: self.current_mark,
                eviction,
            },
        );
    }
}

impl Default for RetainedImageCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct StreamState {
    tolerance: f64,
    error: Option<Error>,
    clip_depth: u32,
    group_stack: Vec<GroupFrame>,
    mask_cache: Option<MaskImageCacheHandle>,
    retained_image_cache: Option<RetainedImageCacheHandle>,
}

#[derive(Debug)]
enum GroupFrame {
    Direct { restores: u8 },
    Masked(Box<MaskedGroupFrame>),
}

#[derive(Debug)]
struct MaskedGroupFrame {
    clip: Option<record::Clip>,
    filters: Vec<imaging::Filter>,
    composite: imaging::Composite,
    mode: MaskMode,
    transform: Affine,
    mask_retained_ptr: usize,
    mask: record::Scene,
    content: record::Scene,
    nested_group_depth: u32,
}

impl StreamState {
    fn new() -> Self {
        Self {
            tolerance: 0.1,
            error: None,
            clip_depth: 0,
            group_stack: Vec::new(),
            mask_cache: None,
            retained_image_cache: None,
        }
    }

    fn new_with_caches(
        mask_cache: MaskImageCacheHandle,
        retained_image_cache: RetainedImageCacheHandle,
    ) -> Self {
        Self {
            mask_cache: Some(mask_cache),
            retained_image_cache: Some(retained_image_cache),
            ..Self::new()
        }
    }

    fn set_error_once(&mut self, err: Error) {
        if self.error.is_none() {
            self.error = Some(err);
        }
    }

    fn finish(&mut self) -> Result<(), Error> {
        if let Some(err) = self.error.take() {
            return Err(err);
        }
        if self.clip_depth != 0 {
            return Err(Error::Internal("unbalanced clip stack"));
        }
        if !self.group_stack.is_empty() {
            return Err(Error::Internal("unbalanced group stack"));
        }
        Ok(())
    }
}

fn active_masked_group_mut(state: &mut StreamState) -> Option<&mut MaskedGroupFrame> {
    match state.group_stack.last_mut() {
        Some(GroupFrame::Masked(frame)) => Some(frame.as_mut()),
        _ => None,
    }
}

fn mask_image_cache_key(
    retained_ptr: usize,
    mode: MaskMode,
    transform: Affine,
    tolerance: f64,
    width: i32,
    height: i32,
) -> MaskImageCacheKey {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    MaskImageCacheKey {
        retained_ptr,
        mode,
        linear_transform: [a.to_bits(), b.to_bits(), c.to_bits(), d.to_bits()],
        tolerance_bits: tolerance.to_bits(),
        width,
        height,
    }
}

fn retained_image_cache_key_for_policy(
    retained: imaging::RetainedRef<'_>,
    transform: Affine,
) -> Option<RetainedImageCacheKey> {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let linear_transform = match retained.cache_policy.image_transform? {
        RetainedTransformPolicy::IdentityOnly => {
            if [a, b, c, d] != [1.0, 0.0, 0.0, 1.0] {
                return None;
            }
            [
                1.0_f64.to_bits(),
                0.0_f64.to_bits(),
                0.0_f64.to_bits(),
                1.0_f64.to_bits(),
            ]
        }
        RetainedTransformPolicy::TranslationOnly => [
            1.0_f64.to_bits(),
            0.0_f64.to_bits(),
            0.0_f64.to_bits(),
            1.0_f64.to_bits(),
        ],
        RetainedTransformPolicy::Linear => [a.to_bits(), b.to_bits(), c.to_bits(), d.to_bits()],
    };
    Some(RetainedImageCacheKey {
        retained_id: retained.stable_id(),
        linear_transform,
    })
}

fn split_transform_translation(transform: Affine) -> (Affine, sk::Point) {
    let [a, b, c, d, e, f] = transform.as_coeffs();
    (
        Affine::new([a, b, c, d, 0.0, 0.0]),
        sk::Point::new(f64_to_f32(e), f64_to_f32(f)),
    )
}

fn sk_rect(rect: Rect) -> sk::Rect {
    sk::Rect::new(
        f64_to_f32(rect.x0),
        f64_to_f32(rect.y0),
        f64_to_f32(rect.x1),
        f64_to_f32(rect.y1),
    )
}

fn render_mask_image(
    canvas: &sk::Canvas,
    state: &mut StreamState,
    retained_mask: &record::Scene,
    transform: Affine,
    width: i32,
    height: i32,
) -> Result<sk::Image, Error> {
    let info = canvas.image_info().with_dimensions((width, height));
    let mut surface = canvas
        .new_surface(&info, None)
        .or_else(|| sk::surfaces::raster(&info, None, None))
        .ok_or(Error::Internal("create surface for cached mask"))?;
    surface.canvas().clear(sk::Color::TRANSPARENT);
    let mut sink = SkCanvasSink::new_internal_with_mask_cache(
        surface.canvas(),
        state
            .mask_cache
            .as_ref()
            .expect("mask cache should exist while rendering cached mask")
            .clone(),
        state
            .retained_image_cache
            .as_ref()
            .expect("retained image cache should exist while rendering cached mask")
            .clone(),
    );
    sink.set_tolerance(state.tolerance);
    let (normalized_transform, _) = split_transform_translation(transform);
    replay_transformed(retained_mask, &mut sink, normalized_transform);
    sink.finish()?;
    Ok(surface.image_snapshot())
}

fn get_cached_mask_image(
    canvas: &sk::Canvas,
    state: &mut StreamState,
    retained_ptr: usize,
    mode: MaskMode,
    retained_mask: &record::Scene,
    transform: Affine,
) -> Result<sk::Image, Error> {
    let Some(mask_cache) = state.mask_cache.as_ref().cloned() else {
        let size = canvas.base_layer_size();
        return render_mask_image(
            canvas,
            state,
            retained_mask,
            transform,
            size.width,
            size.height,
        );
    };

    let size = canvas.base_layer_size();
    let key = mask_image_cache_key(
        retained_ptr,
        mode,
        transform,
        state.tolerance,
        size.width,
        size.height,
    );
    if let Some(image) = mask_cache.borrow_mut().get(key) {
        return Ok(image);
    }

    let image = render_mask_image(
        canvas,
        state,
        retained_mask,
        transform,
        size.width,
        size.height,
    )?;
    mask_cache.borrow_mut().insert(key, image.clone());
    Ok(image)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "The value is range-checked against i32::MAX above."
)]
fn ceil_dim_to_i32(value: f64) -> Result<i32, Error> {
    let ceil = value.ceil().max(1.0);
    if ceil > f64::from(i32::MAX) {
        return Err(Error::Internal("cached retained image dimension overflow"));
    }
    Ok(ceil as i32)
}

fn compute_retained_scene_bounds(scene: &record::Scene, tolerance: f64) -> Option<Rect> {
    let mut bounds: Option<Rect> = None;
    for command in scene.commands() {
        let record::Command::Draw(draw_id) = *command else {
            continue;
        };
        let draw_bounds = match scene.draw_op(draw_id) {
            record::Draw::Retained(draw) => {
                let retained = scene.retained(draw.retained);
                let child = compute_retained_scene_bounds(&retained.scene, tolerance)?;
                draw.transform.transform_rect_bbox(child)
            }
            record::Draw::Fill {
                transform, shape, ..
            } => {
                let base = match shape {
                    record::Geometry::Rect(rect) => *rect,
                    record::Geometry::RoundedRect(rect) => rect.rect(),
                    record::Geometry::Path(path) => path.bounding_box(),
                };
                transform.transform_rect_bbox(base)
            }
            record::Draw::Stroke {
                transform,
                stroke,
                shape,
                ..
            } => {
                let src = match shape {
                    record::Geometry::Rect(rect) => rect.to_path(tolerance),
                    record::Geometry::RoundedRect(rr) => rr.to_path(tolerance),
                    record::Geometry::Path(path) => path.clone(),
                };
                let outline =
                    kurbo::stroke(src.iter(), stroke, &kurbo::StrokeOpts::default(), tolerance);
                transform.transform_rect_bbox(outline.bounding_box())
            }
            record::Draw::BlurredRoundedRect(draw) => {
                let inflate = draw.std_dev * 3.0;
                draw.transform
                    .transform_rect_bbox(draw.rect.inflate(inflate, inflate))
            }
            record::Draw::GlyphRun(_) => return None,
        };
        bounds = Some(match bounds {
            Some(existing) => existing.union(draw_bounds),
            None => draw_bounds,
        });
    }
    bounds
}

fn retained_local_bounds(retained: imaging::RetainedRef<'_>, tolerance: f64) -> Option<Rect> {
    retained
        .bounds
        .or_else(|| compute_retained_scene_bounds(retained.scene, tolerance))
}

fn render_retained_image(
    canvas: &sk::Canvas,
    state: &mut StreamState,
    retained: imaging::RetainedRef<'_>,
    transform: Affine,
    local_bounds: Rect,
) -> Result<sk::Image, Error> {
    let (linear_transform, _) = split_transform_translation(transform);
    let transformed_bounds = linear_transform.transform_rect_bbox(local_bounds);
    let width = ceil_dim_to_i32(transformed_bounds.width())?;
    let height = ceil_dim_to_i32(transformed_bounds.height())?;
    let info = canvas.image_info().with_dimensions((width, height));
    let mut surface = canvas
        .new_surface(&info, None)
        .or_else(|| sk::surfaces::raster(&info, None, None))
        .ok_or(Error::Internal("create surface for cached retained image"))?;
    surface.canvas().clear(sk::Color::TRANSPARENT);
    let mut sink = SkCanvasSink::new_internal_with_mask_cache(
        surface.canvas(),
        state
            .mask_cache
            .as_ref()
            .expect("mask cache should exist while rendering cached retained image")
            .clone(),
        state
            .retained_image_cache
            .as_ref()
            .expect("retained image cache should exist while rendering cached retained image")
            .clone(),
    );
    sink.set_tolerance(state.tolerance);
    let normalized_transform =
        Affine::translate((-transformed_bounds.x0, -transformed_bounds.y0)) * linear_transform;
    replay_transformed(retained.scene, &mut sink, normalized_transform);
    sink.finish()?;
    Ok(surface.image_snapshot())
}

fn get_cached_retained_image(
    canvas: &sk::Canvas,
    state: &mut StreamState,
    retained: imaging::RetainedRef<'_>,
    transform: Affine,
    local_bounds: Rect,
) -> Result<sk::Image, Error> {
    let eviction = retained.cache_policy.eviction;
    let Some(key) = retained_image_cache_key_for_policy(retained.clone(), transform) else {
        return render_retained_image(canvas, state, retained, transform, local_bounds);
    };
    let Some(cache) = state.retained_image_cache.as_ref().cloned() else {
        return render_retained_image(canvas, state, retained, transform, local_bounds);
    };
    if let Some(image) = cache.borrow_mut().get(key) {
        return Ok(image);
    }
    let image = render_retained_image(canvas, state, retained, transform, local_bounds)?;
    cache.borrow_mut().insert(key, image.clone(), eviction);
    Ok(image)
}

fn set_matrix(canvas: &sk::Canvas, xf: Affine) {
    canvas.reset_matrix();
    canvas.concat(&affine_to_matrix(xf));
}

fn clip_path(canvas: &sk::Canvas, state: &mut StreamState, clip: ClipRef<'_>) -> Option<sk::Path> {
    match clip {
        ClipRef::Fill {
            transform,
            shape,
            fill_rule,
        } => {
            let mut path = geometry_to_sk_path(shape, state.tolerance)?;
            set_matrix(canvas, transform);
            path = path_with_fill_rule(&path, fill_rule);
            Some(path)
        }
        ClipRef::Stroke {
            transform,
            shape,
            stroke,
        } => {
            let src = geometry_to_bez_path(shape, state.tolerance)?;
            let outline = kurbo::stroke(
                src.iter(),
                stroke,
                &kurbo::StrokeOpts::default(),
                state.tolerance,
            );
            set_matrix(canvas, transform);
            bez_to_sk_path(&outline)
        }
    }
}

fn push_group_impl(canvas: &sk::Canvas, state: &mut StreamState, group: GroupRef<'_>) -> u8 {
    let filter = if group.filters.is_empty() {
        None
    } else {
        build_filter_chain(group.filters)
    };
    if !group.filters.is_empty() && filter.is_none() {
        state.set_error_once(Error::UnsupportedFilter);
    }

    let clip_path = group.clip.and_then(|clip| clip_path(canvas, state, clip));
    let mut restores = 0_u8;

    let mut paint = sk::Paint::default();
    let mut needs_layer = false;

    let blend = group.composite.blend;
    let alpha = group.composite.alpha.clamp(0.0, 1.0);
    if blend != peniko::BlendMode::default() || alpha != 1.0 {
        paint.set_blend_mode(map_blend_mode(&blend));
        paint.set_alpha_f(alpha);
        needs_layer = true;
    }

    if let Some(filter) = filter {
        paint.set_image_filter(filter);
        needs_layer = true;
    }

    if let Some(path) = clip_path.as_ref() {
        canvas.save();
        canvas.clip_path(path, None, true);
        restores += 1;
    }

    if needs_layer {
        // This is an isolated group layer: draw children into a fresh offscreen layer, then apply
        // the group's blend/alpha/filter when that layer is restored into the parent canvas.
        // True backdrop effects would need a separate path that first samples or copies the
        // existing destination content for the clipped region, applies the effect against that
        // backdrop, and then composites the group result back.
        canvas.save_layer(&sk::canvas::SaveLayerRec::default().paint(&paint));
        restores += 1;
    }

    restores
}

fn draw_glyph_run(
    canvas: &sk::Canvas,
    state: &mut StreamState,
    glyph_run: GlyphRunRef<'_>,
    glyphs: &mut dyn Iterator<Item = record::Glyph>,
) {
    if !glyph_run.normalized_coords.is_empty() {
        state.set_error_once(Error::UnsupportedGlyphVariations);
        return;
    }

    let Some(mut font) = skia_font_from_glyph_run(&glyph_run) else {
        state.set_error_once(Error::InvalidFontData);
        return;
    };

    set_matrix(canvas, glyph_run.transform);

    let Some(mut sk_paint) =
        brush_to_paint(glyph_run.brush, glyph_run.composite.alpha, Affine::IDENTITY)
    else {
        state.set_error_once(Error::Internal("invalid image brush"));
        return;
    };
    sk_paint.set_blend_mode(map_blend_mode(&glyph_run.composite.blend));

    match glyph_run.style {
        peniko::Style::Fill(_) => {
            sk_paint.set_style(sk::PaintStyle::Fill);
        }
        peniko::Style::Stroke(stroke) => apply_stroke_style(&mut sk_paint, stroke),
    }

    let mut glyph_ids = Vec::new();
    let mut positions = Vec::new();
    for glyph in glyphs {
        let Ok(glyph_id) = sk::GlyphId::try_from(glyph.id) else {
            state.set_error_once(Error::InvalidGlyphId);
            return;
        };
        glyph_ids.push(glyph_id);
        positions.push(sk::Point::new(glyph.x, glyph.y));
    }

    font.set_subpixel(true);
    canvas.draw_glyphs_at(
        &glyph_ids,
        positions.as_slice(),
        sk::Point::new(0.0, 0.0),
        &font,
        &sk_paint,
    );
}

fn draw_blurred_rounded_rect(
    canvas: &sk::Canvas,
    state: &mut StreamState,
    draw: BlurredRoundedRect,
) {
    set_matrix(canvas, draw.transform);

    let mut paint = sk::Paint::default();
    paint.set_anti_alias(true);
    paint.set_style(sk::PaintStyle::Fill);
    let color = draw.color.multiply_alpha(draw.composite.alpha);
    let comps = color.components;
    paint.set_color4f(
        sk::Color4f::new(comps[0], comps[1], comps[2], comps[3]),
        None,
    );
    paint.set_blend_mode(map_blend_mode(&draw.composite.blend));
    let Some(mask_filter) =
        sk::MaskFilter::blur(sk::BlurStyle::Normal, f64_to_f32(draw.std_dev), Some(true))
    else {
        state.set_error_once(Error::Internal("create blur mask filter"));
        return;
    };
    paint.set_mask_filter(mask_filter);

    let rect = sk::Rect::new(
        f64_to_f32(draw.rect.x0),
        f64_to_f32(draw.rect.y0),
        f64_to_f32(draw.rect.x1),
        f64_to_f32(draw.rect.y1),
    );
    let rrect = sk::RRect::new_rect_xy(rect, f64_to_f32(draw.radius), f64_to_f32(draw.radius));
    canvas.draw_rrect(rrect, &paint);
}

fn draw_masked_group(canvas: &sk::Canvas, state: &mut StreamState, masked: MaskedGroupFrame) {
    let mut group_paint = sk::Paint::default();
    group_paint.set_anti_alias(true);
    group_paint.set_blend_mode(map_blend_mode(&masked.composite.blend));
    group_paint.set_alpha_f(masked.composite.alpha);
    if !masked.filters.is_empty() {
        if let Some(filter) = build_filter_chain(&masked.filters) {
            group_paint.set_image_filter(filter);
        } else {
            state.set_error_once(Error::UnsupportedFilter);
            return;
        }
    }

    let clip_path = masked
        .clip
        .as_ref()
        .and_then(|clip| clip_path(canvas, state, clip.as_ref()));
    let content_bounds = compute_retained_scene_bounds(&masked.content, state.tolerance);
    if let Some(path) = clip_path.as_ref() {
        canvas.save();
        canvas.clip_path(path, None, true);
    }

    set_matrix(canvas, Affine::IDENTITY);
    let sk_bounds = content_bounds.map(sk_rect);
    let layer_rec = match sk_bounds.as_ref() {
        Some(bounds) => sk::canvas::SaveLayerRec::default()
            .bounds(bounds)
            .paint(&group_paint),
        None => sk::canvas::SaveLayerRec::default().paint(&group_paint),
    };
    canvas.save_layer(&layer_rec);

    {
        let mut sink = match (
            state.mask_cache.as_ref().cloned(),
            state.retained_image_cache.as_ref().cloned(),
        ) {
            (Some(mask_cache), Some(retained_image_cache)) => {
                SkCanvasSink::new_with_mask_cache(canvas, mask_cache, retained_image_cache)
            }
            _ => SkCanvasSink::new(canvas),
        };
        sink.set_tolerance(state.tolerance);
        replay(&masked.content, &mut sink);
        if let Err(err) = sink.finish() {
            state.set_error_once(err);
            canvas.restore();
            if clip_path.is_some() {
                canvas.restore();
            }
            return;
        }
    }

    let mask_image = match get_cached_mask_image(
        canvas,
        state,
        masked.mask_retained_ptr,
        masked.mode,
        &masked.mask,
        masked.transform,
    ) {
        Ok(mask_image) => mask_image,
        Err(err) => {
            state.set_error_once(err);
            canvas.restore();
            if clip_path.is_some() {
                canvas.restore();
            }
            return;
        }
    };
    let mut mask_paint = sk::Paint::default();
    mask_paint.set_anti_alias(true);
    mask_paint.set_blend_mode(sk::BlendMode::DstIn);
    if masked.mode == MaskMode::Luminance {
        mask_paint.set_color_filter(sk::ColorFilter::luma());
    }
    let (linear_transform, translation) = split_transform_translation(masked.transform);
    let mask_local_bounds = Rect::new(
        0.0,
        0.0,
        f64::from(mask_image.width()),
        f64::from(mask_image.height()),
    );
    let transformed_mask_bounds = linear_transform.transform_rect_bbox(mask_local_bounds);
    set_matrix(canvas, Affine::IDENTITY);
    canvas.draw_image(
        &mask_image,
        (
            f64_to_f32(transformed_mask_bounds.x0) + translation.x,
            f64_to_f32(transformed_mask_bounds.y0) + translation.y,
        ),
        Some(&mask_paint),
    );
    canvas.restore();

    if clip_path.is_some() {
        canvas.restore();
    }
}

fn paint_sink_push_clip(canvas: &sk::Canvas, state: &mut StreamState, clip: ClipRef<'_>) {
    if state.error.is_some() {
        return;
    }
    if let Some(frame) = active_masked_group_mut(state) {
        PaintSink::push_clip(&mut frame.content, clip);
        return;
    }
    let Some(path) = clip_path(canvas, state, clip) else {
        return;
    };
    canvas.save();
    canvas.clip_path(&path, None, true);
    state.clip_depth += 1;
}

fn paint_sink_pop_clip(canvas: &sk::Canvas, state: &mut StreamState) {
    if state.error.is_some() {
        return;
    }
    if let Some(frame) = active_masked_group_mut(state) {
        frame.content.pop_clip();
        return;
    }
    if state.clip_depth == 0 {
        state.set_error_once(Error::Internal("pop_clip underflow"));
        return;
    }
    canvas.restore();
    state.clip_depth -= 1;
}

fn paint_sink_push_group(canvas: &sk::Canvas, state: &mut StreamState, group: GroupRef<'_>) {
    if state.error.is_some() {
        return;
    }
    if let Some(frame) = active_masked_group_mut(state) {
        PaintSink::push_group(&mut frame.content, group);
        frame.nested_group_depth += 1;
        return;
    }
    if let Some(mask) = group.mask {
        state
            .group_stack
            .push(GroupFrame::Masked(Box::new(MaskedGroupFrame {
                clip: group.clip.map(ClipRef::to_owned),
                filters: group.filters.to_vec(),
                composite: group.composite,
                mode: mask.mask.mode,
                transform: mask.transform,
                mask_retained_ptr: mask.mask.retained.scene as *const _ as usize,
                mask: mask.mask.retained.scene.clone(),
                content: record::Scene::new(),
                nested_group_depth: 0,
            })));
        return;
    }
    let restores = push_group_impl(canvas, state, group);
    state.group_stack.push(GroupFrame::Direct { restores });
}

fn paint_sink_pop_group(canvas: &sk::Canvas, state: &mut StreamState) {
    if state.error.is_some() {
        return;
    }
    let Some(frame) = state.group_stack.pop() else {
        state.set_error_once(Error::Internal("pop_group underflow"));
        return;
    };
    match frame {
        GroupFrame::Direct { restores } => {
            for _ in 0..restores {
                canvas.restore();
            }
        }
        GroupFrame::Masked(mut frame) => {
            if frame.nested_group_depth != 0 {
                frame.content.pop_group();
                frame.nested_group_depth -= 1;
                state.group_stack.push(GroupFrame::Masked(frame));
                return;
            }
            draw_masked_group(canvas, state, *frame);
        }
    }
}

fn paint_sink_fill(canvas: &sk::Canvas, state: &mut StreamState, draw: FillRef<'_>) {
    if state.error.is_some() {
        return;
    }
    if let Some(frame) = active_masked_group_mut(state) {
        frame.content.fill(draw);
        return;
    }

    set_matrix(canvas, draw.transform);
    let Some(mut sk_paint) = brush_to_paint(
        draw.brush,
        draw.composite.alpha,
        draw.brush_transform.unwrap_or(Affine::IDENTITY),
    ) else {
        state.set_error_once(Error::Internal("invalid image brush"));
        return;
    };
    sk_paint.set_blend_mode(map_blend_mode(&draw.composite.blend));
    sk_paint.set_style(sk::PaintStyle::Fill);

    match draw.shape {
        GeometryRef::Rect(r) => {
            let rect = sk::Rect::new(
                f64_to_f32(r.x0),
                f64_to_f32(r.y0),
                f64_to_f32(r.x1),
                f64_to_f32(r.y1),
            );
            canvas.draw_rect(rect, &sk_paint);
        }
        GeometryRef::RoundedRect(rr) => {
            let path = rr.to_path(state.tolerance);
            let sk_path = bez_to_sk_path(&path).expect("rounded rect to sk path");
            let sk_path = path_with_fill_rule(&sk_path, draw.fill_rule);
            canvas.draw_path(&sk_path, &sk_paint);
        }
        GeometryRef::Path(p) => {
            let sk_path = bez_to_sk_path(p).expect("path to sk path");
            let sk_path = path_with_fill_rule(&sk_path, draw.fill_rule);
            canvas.draw_path(&sk_path, &sk_paint);
        }
        GeometryRef::OwnedPath(p) => {
            let sk_path = bez_to_sk_path(&p).expect("path to sk path");
            let sk_path = path_with_fill_rule(&sk_path, draw.fill_rule);
            canvas.draw_path(&sk_path, &sk_paint);
        }
    }
}

fn paint_sink_stroke(canvas: &sk::Canvas, state: &mut StreamState, draw: StrokeRef<'_>) {
    if state.error.is_some() {
        return;
    }
    if let Some(frame) = active_masked_group_mut(state) {
        frame.content.stroke(draw);
        return;
    }

    set_matrix(canvas, draw.transform);
    let Some(mut sk_paint) = brush_to_paint(
        draw.brush,
        draw.composite.alpha,
        draw.brush_transform.unwrap_or(Affine::IDENTITY),
    ) else {
        state.set_error_once(Error::Internal("invalid image brush"));
        return;
    };
    sk_paint.set_blend_mode(map_blend_mode(&draw.composite.blend));
    apply_stroke_style(&mut sk_paint, draw.stroke);

    match draw.shape {
        GeometryRef::Rect(r) => {
            let rect = sk::Rect::new(
                f64_to_f32(r.x0),
                f64_to_f32(r.y0),
                f64_to_f32(r.x1),
                f64_to_f32(r.y1),
            );
            canvas.draw_rect(rect, &sk_paint);
        }
        GeometryRef::RoundedRect(rr) => {
            let path = rr.to_path(state.tolerance);
            let sk_path = bez_to_sk_path(&path).expect("rounded rect to sk path");
            canvas.draw_path(&sk_path, &sk_paint);
        }
        GeometryRef::Path(p) => {
            let sk_path = bez_to_sk_path(p).expect("path to sk path");
            canvas.draw_path(&sk_path, &sk_paint);
        }
        GeometryRef::OwnedPath(p) => {
            let sk_path = bez_to_sk_path(&p).expect("path to sk path");
            canvas.draw_path(&sk_path, &sk_paint);
        }
    }
}

/// Borrowed adapter that streams `imaging` commands into an existing [`skia_safe::Canvas`].
pub struct SkCanvasSink<'a> {
    canvas: &'a sk::Canvas,
    state: StreamState,
}

impl core::fmt::Debug for SkCanvasSink<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SkCanvasSink")
            .field("tolerance", &self.state.tolerance)
            .field("error", &self.state.error)
            .field("clip_depth", &self.state.clip_depth)
            .field("group_depth", &self.state.group_stack.len())
            .finish_non_exhaustive()
    }
}

impl<'a> SkCanvasSink<'a> {
    /// Wrap an existing [`skia_safe::Canvas`].
    pub fn new(canvas: &'a sk::Canvas) -> Self {
        Self {
            canvas,
            state: StreamState::new(),
        }
    }

    pub(crate) fn new_with_mask_cache(
        canvas: &'a sk::Canvas,
        mask_cache: MaskImageCacheHandle,
        retained_image_cache: RetainedImageCacheHandle,
    ) -> Self {
        Self::new_with_caches(canvas, mask_cache, retained_image_cache)
    }

    pub(crate) fn new_internal_with_mask_cache(
        canvas: &'a sk::Canvas,
        mask_cache: MaskImageCacheHandle,
        retained_image_cache: RetainedImageCacheHandle,
    ) -> Self {
        Self::new_with_caches(canvas, mask_cache, retained_image_cache)
    }

    fn new_with_caches(
        canvas: &'a sk::Canvas,
        mask_cache: MaskImageCacheHandle,
        retained_image_cache: RetainedImageCacheHandle,
    ) -> Self {
        Self {
            canvas,
            state: StreamState::new_with_caches(mask_cache, retained_image_cache),
        }
    }

    /// Set the tolerance used when converting rounded rectangles to paths.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.state.tolerance = tolerance;
    }

    /// Return the first deferred translation error, if any, and ensure clip/group stacks are balanced.
    pub fn finish(&mut self) -> Result<(), Error> {
        self.state.finish()
    }
}

impl PaintSink for SkCanvasSink<'_> {
    fn push_clip(&mut self, clip: ClipRef<'_>) {
        paint_sink_push_clip(self.canvas, &mut self.state, clip);
    }

    fn pop_clip(&mut self) {
        paint_sink_pop_clip(self.canvas, &mut self.state);
    }

    fn push_group(&mut self, group: GroupRef<'_>) {
        paint_sink_push_group(self.canvas, &mut self.state, group);
    }

    fn pop_group(&mut self) {
        paint_sink_pop_group(self.canvas, &mut self.state);
    }

    fn retained(&mut self, draw: RetainedDrawRef<'_>) {
        if self.state.error.is_some() {
            return;
        }
        if let Some(frame) = active_masked_group_mut(&mut self.state) {
            PaintSink::retained(&mut frame.content, draw);
            return;
        }
        let Some(local_bounds) = retained_local_bounds(draw.retained.clone(), self.state.tolerance)
        else {
            self.state
                .set_error_once(Error::Internal("compute retained scene bounds"));
            return;
        };
        let cached = match get_cached_retained_image(
            self.canvas,
            &mut self.state,
            draw.retained.clone(),
            draw.transform,
            local_bounds,
        ) {
            Ok(cached) => cached,
            Err(err) => {
                self.state.set_error_once(err);
                return;
            }
        };
        let mut paint = sk::Paint::default();
        paint.set_alpha_f(draw.composite.alpha);
        paint.set_blend_mode(map_blend_mode(&draw.composite.blend));
        let (linear_transform, translation) = split_transform_translation(draw.transform);
        let transformed_bounds = linear_transform.transform_rect_bbox(local_bounds);
        set_matrix(
            self.canvas,
            Affine::translate((f64::from(translation.x), f64::from(translation.y))),
        );
        self.canvas.draw_image(
            &cached,
            (
                f64_to_f32(transformed_bounds.x0),
                f64_to_f32(transformed_bounds.y0),
            ),
            Some(&paint),
        );
    }

    fn fill(&mut self, draw: FillRef<'_>) {
        paint_sink_fill(self.canvas, &mut self.state, draw);
    }

    fn stroke(&mut self, draw: StrokeRef<'_>) {
        paint_sink_stroke(self.canvas, &mut self.state, draw);
    }

    fn glyph_run(
        &mut self,
        draw: GlyphRunRef<'_>,
        glyphs: &mut dyn Iterator<Item = record::Glyph>,
    ) {
        if self.state.error.is_some() {
            return;
        }
        if let Some(frame) = active_masked_group_mut(&mut self.state) {
            frame.content.glyph_run(draw, glyphs);
            return;
        }
        draw_glyph_run(self.canvas, &mut self.state, draw, glyphs);
    }

    fn blurred_rounded_rect(&mut self, draw: BlurredRoundedRect) {
        if self.state.error.is_some() {
            return;
        }
        if let Some(frame) = active_masked_group_mut(&mut self.state) {
            frame.content.blurred_rounded_rect(draw);
            return;
        }
        draw_blurred_rounded_rect(self.canvas, &mut self.state, draw);
    }
}

/// Owned sink that records `imaging` commands into a native [`skia_safe::Picture`].
pub struct SkPictureRecorderSink {
    recorder: sk::PictureRecorder,
    state: StreamState,
}

impl core::fmt::Debug for SkPictureRecorderSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SkPictureRecorderSink")
            .field("tolerance", &self.state.tolerance)
            .field("error", &self.state.error)
            .field("clip_depth", &self.state.clip_depth)
            .field("group_depth", &self.state.group_stack.len())
            .finish_non_exhaustive()
    }
}

impl SkPictureRecorderSink {
    /// Start recording a Skia picture with the given cull bounds.
    pub fn new(bounds: Rect) -> Self {
        Self::new_with_bbh(bounds, false)
    }

    /// Start recording a Skia picture with optional bounding-box hierarchy acceleration.
    pub fn new_with_bbh(bounds: Rect, use_bbh: bool) -> Self {
        let mut recorder = sk::PictureRecorder::new();
        let bounds = sk::Rect::new(
            f64_to_f32(bounds.x0),
            f64_to_f32(bounds.y0),
            f64_to_f32(bounds.x1),
            f64_to_f32(bounds.y1),
        );
        let _ = recorder.begin_recording(bounds, use_bbh);
        Self {
            recorder,
            state: StreamState::new(),
        }
    }

    /// Set the tolerance used when converting rounded rectangles to paths.
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.state.tolerance = tolerance;
    }

    /// Finish recording and return the resulting [`skia_safe::Picture`].
    pub fn finish_picture(mut self) -> Result<sk::Picture, Error> {
        self.state.finish()?;
        self.recorder
            .finish_recording_as_picture(None)
            .ok_or(Error::Internal("finish_recording_as_picture failed"))
    }
}

impl PaintSink for SkPictureRecorderSink {
    fn push_clip(&mut self, clip: ClipRef<'_>) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        paint_sink_push_clip(canvas, state, clip);
    }

    fn pop_clip(&mut self) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        paint_sink_pop_clip(canvas, state);
    }

    fn push_group(&mut self, group: GroupRef<'_>) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        paint_sink_push_group(canvas, state, group);
    }

    fn pop_group(&mut self) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        paint_sink_pop_group(canvas, state);
    }

    fn retained(&mut self, draw: RetainedDrawRef<'_>) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        if state.error.is_some() {
            return;
        }
        if let Some(frame) = active_masked_group_mut(state) {
            PaintSink::retained(&mut frame.content, draw);
            return;
        }
        let Some(local_bounds) = retained_local_bounds(draw.retained.clone(), state.tolerance)
        else {
            state.set_error_once(Error::Internal("compute retained scene bounds"));
            return;
        };
        let cached = match get_cached_retained_image(
            canvas,
            state,
            draw.retained.clone(),
            draw.transform,
            local_bounds,
        ) {
            Ok(cached) => cached,
            Err(err) => {
                state.set_error_once(err);
                return;
            }
        };
        let mut paint = sk::Paint::default();
        paint.set_alpha_f(draw.composite.alpha);
        paint.set_blend_mode(map_blend_mode(&draw.composite.blend));
        let (linear_transform, translation) = split_transform_translation(draw.transform);
        let transformed_bounds = linear_transform.transform_rect_bbox(local_bounds);
        set_matrix(
            canvas,
            Affine::translate((f64::from(translation.x), f64::from(translation.y))),
        );
        canvas.draw_image(
            &cached,
            (
                f64_to_f32(transformed_bounds.x0),
                f64_to_f32(transformed_bounds.y0),
            ),
            Some(&paint),
        );
    }

    fn fill(&mut self, draw: FillRef<'_>) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        paint_sink_fill(canvas, state, draw);
    }

    fn stroke(&mut self, draw: StrokeRef<'_>) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        paint_sink_stroke(canvas, state, draw);
    }

    fn glyph_run(
        &mut self,
        draw: GlyphRunRef<'_>,
        glyphs: &mut dyn Iterator<Item = record::Glyph>,
    ) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        if state.error.is_some() {
            return;
        }
        if let Some(frame) = active_masked_group_mut(state) {
            frame.content.glyph_run(draw, glyphs);
            return;
        }
        draw_glyph_run(canvas, state, draw, glyphs);
    }

    fn blurred_rounded_rect(&mut self, draw: BlurredRoundedRect) {
        let recorder = &mut self.recorder;
        let state = &mut self.state;
        let Some(canvas) = recorder.recording_canvas() else {
            state.set_error_once(Error::Internal("picture recorder not recording"));
            return;
        };
        if state.error.is_some() {
            return;
        }
        if let Some(frame) = active_masked_group_mut(state) {
            frame.content.blurred_rounded_rect(draw);
            return;
        }
        draw_blurred_rounded_rect(canvas, state, draw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imaging::Composite;
    use peniko::{Brush, Color};

    #[test]
    fn sk_canvas_sink_reports_clip_underflow() {
        let mut surface = sk::surfaces::raster_n32_premul((16, 16)).unwrap();
        let mut sink = SkCanvasSink::new(surface.canvas());
        sink.pop_clip();
        assert!(matches!(
            sink.finish(),
            Err(Error::Internal("pop_clip underflow"))
        ));
    }

    #[test]
    fn sk_picture_recorder_sink_finishes_picture() {
        let mut sink = SkPictureRecorderSink::new(Rect::new(0.0, 0.0, 32.0, 32.0));
        sink.fill(FillRef::new(
            Rect::new(0.0, 0.0, 16.0, 16.0),
            &Brush::Solid(Color::from_rgb8(0x11, 0x22, 0x33)),
        ));
        let picture = sink.finish_picture().unwrap();
        let cull = picture.cull_rect();
        assert_eq!(cull.left, 0.0);
        assert_eq!(cull.top, 0.0);
        assert_eq!(cull.right, 32.0);
        assert_eq!(cull.bottom, 32.0);
    }

    #[test]
    fn sk_picture_recorder_sink_rejects_unbalanced_group() {
        let mut sink = SkPictureRecorderSink::new(Rect::new(0.0, 0.0, 32.0, 32.0));
        sink.push_group(GroupRef::new().with_composite(Composite::default()));
        assert!(matches!(
            sink.finish_picture(),
            Err(Error::Internal("unbalanced group stack"))
        ));
    }
}
