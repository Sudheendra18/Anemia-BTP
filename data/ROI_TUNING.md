# Tuning ROI extraction on real images

The default HSV thresholds in `configs/config.yaml` are a reasonable
starting point, not a calibrated fit to your actual photos. Skin and moist
conjunctival tissue often sit in an overlapping hue band — what usually
separates them is **saturation** (the moist conjunctiva tends to read more
saturated/glossy than surrounding eyelid skin) more than hue alone. Expect to
retune, especially if `roi_status` comes back `fallback` for more than a
handful of images.

## 1. Run once and check the numbers

```bash
python scripts/run_pipeline.py
```

Look at the log line: `ROI extraction complete: N ok, M fallback, ...`
A high fallback count means the current thresholds aren't matching your
images' actual color distribution.

## 2. Look at the debug composites

`outputs/debug/` has `debug_sample_size` (default 20) random composites:
**original with detected box | mask | cropped result**. This is the fastest
way to see whether the box is landing on the conjunctiva, on eyelid skin, or
nothing at all.

## 3. Sample actual pixel values from your images

If the box is consistently wrong, pull HSV values directly from a few known
conjunctiva regions in your real photos to re-center the thresholds:

```python
import cv2
img = cv2.imread("data/raw/Anemic/Image_001.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# click-free quick check: print HSV at a pixel you can see is on the
# conjunctiva (open the image, note rough x,y)
x, y = 320, 240
print(hsv[y, x])  # (H, S, V)
```

Do this for ~10 images (a mix of Anemic and Non-anemic — anemic conjunctiva
is paler/less saturated by definition, which is exactly the biological
signal the model needs preserved, so the *segmentation* thresholds should be
generous enough to catch both, not tuned only on high-saturation Anemic=false
examples).

## 4. Adjust and re-run

Widen or shift `hsv_lower` / `hsv_upper` / `hsv_lower_wrap` / `hsv_upper_wrap`
in `configs/config.yaml` based on what you see, then re-run just Stage 2:

```bash
python scripts/run_pipeline.py --stage roi
```

## 5. If color segmentation isn't enough

If lighting varies a lot across devices/hospitals (likely, given the capture
protocol confound this project is already tracking), a fixed global HSV
range may not generalize. Two escalation paths, in order of effort:
- Per-hospital or per-batch threshold overrides (cheap: just parameterize by
  a `hospital` or `source` field before calling `_segment_mask`).
- A learned segmentation model (e.g. a small U-Net) if classical
  thresholding's fallback rate stays high after tuning — bigger effort, only
  worth it if this is genuinely a bottleneck once you see the real data.
