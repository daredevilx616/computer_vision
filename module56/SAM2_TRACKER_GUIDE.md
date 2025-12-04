# SAM2 Tracker - Complete Guide

## Overview

The SAM2 tracker uses **AI-powered segmentation** to find objects, then tracks them in real-time using color-based tracking.

### Why SAM2?

- **No manual selection needed** - AI finds the object automatically
- **Better than color tracking** - More accurate initial segmentation
- **Combines AI + real-time** - Best of both worlds

### How It Works:

```
1. Capture frame → 2. SAM2 segments object (offline) → 3. Track in real-time
```

---

## 🎯 Complete Workflow

### **Step 1: Prepare Your Environment**

Check if SAM2 is installed from module 3:
```bash
python -c "import sam2; print('SAM2 installed!')"
```

If not installed:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Check if you have the checkpoint:
```bash
ls module3/sam2_checkpoints/sam2_hiera_small.pt
```

If missing, download from: https://github.com/facebookresearch/segment-anything-2/tree/main/checkpoints

---

### **Step 2: Capture a Reference Frame**

1. Open the tracker: http://localhost:3000/assignment56
2. Click **"SAM2 Segmentation"** mode
3. Place your object in view (well-lit, clear background)
4. Click **"Capture Reference Frame"**
5. The frame is captured in memory

**Download the frame:**
- Right-click on the canvas → "Save image as..."
- Or add a download button (we can add this if needed)

For now, you can capture a frame by:
1. Taking a screenshot of the canvas
2. Or using your phone/camera to capture a test image

---

### **Step 3: Run SAM2 Segmentation**

Use the helper script I created:

```bash
python -m module56.run_sam2_single \
  --input path/to/captured_frame.jpg \
  --output sam2_mask.png \
  --checkpoint module3/sam2_checkpoints/sam2_hiera_small.pt
```

**Example with your existing setup:**
```bash
# If you saved a frame as "test_frame.jpg"
python -m module56.run_sam2_single \
  --input test_frame.jpg \
  --output my_object_mask.png
```

**What this does:**
- Loads your frame
- Runs SAM2 segmentation (finds all objects)
- Picks the largest/most stable object
- Saves a **binary mask** (white = object, black = background)
- Saves an **overlay** for visualization

**Output files:**
- `my_object_mask.png` - The mask to upload
- `my_object_mask_overlay.png` - Visual preview

---

### **Step 4: Upload Mask to Tracker**

1. Go back to the SAM2 tracker UI
2. Make sure you're still in **SAM2 mode**
3. Click **"Capture Reference Frame"** again (to sync)
4. Click the **file upload button**
5. Select `my_object_mask.png`
6. The tracker initializes!

Now move your object - the tracker follows it! 🎯

---

## 📝 **Example: Complete Walkthrough**

Let's track a **water bottle**:

### 1. Prepare
```bash
# Make sure SAM2 is ready
python -c "import sam2; print('Ready!')"
```

### 2. Capture Frame
- Open http://localhost:3000/assignment56
- Switch to **SAM2 mode**
- Place water bottle in front of camera
- Click "Capture Reference Frame"
- Take a screenshot or save the canvas as `bottle_frame.jpg`

### 3. Segment with SAM2
```bash
python -m module56.run_sam2_single \
  --input bottle_frame.jpg \
  --output bottle_mask.png
```

**Output:**
```
===========================================================
SAM2 Single Image Segmentation
===========================================================
Input         : bottle_frame.jpg
Output        : bottle_mask.png
...
Generated 15 masks

Top mask candidates:
  1. Area:  45000 pixels, Stability: 0.976
  2. Area:  12000 pixels, Stability: 0.894
  3. Area:   5000 pixels, Stability: 0.812

✓ Mask saved to: bottle_mask.png
✓ Overlay saved to: bottle_mask_overlay.png
===========================================================
```

### 4. Check the Overlay
```bash
# View the overlay to verify segmentation
start bottle_mask_overlay.png   # Windows
# or
open bottle_mask_overlay.png    # Mac
```

Make sure it correctly segmented your bottle!

### 5. Upload and Track
- Go back to UI
- Click "Capture Reference Frame" (important!)
- Upload `bottle_mask.png`
- Move the bottle → tracking starts! 🎉

---

## 🔧 **Advanced Options**

### Adjust Segmentation Quality

**More accurate (slower):**
```bash
python -m module56.run_sam2_single \
  --input frame.jpg \
  --output mask.png \
  --points-per-side 32
```

**Faster (less accurate):**
```bash
python -m module56.run_sam2_single \
  --input frame.jpg \
  --output mask.png \
  --points-per-side 8
```

### Use GPU (if available)
```bash
python -m module56.run_sam2_single \
  --input frame.jpg \
  --output mask.png \
  --device cuda
```

---

## ❓ **Troubleshooting**

### "SAM2 not installed"
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### "Checkpoint not found"
Download from: https://github.com/facebookresearch/segment-anything-2/tree/main/checkpoints

Place in: `module3/sam2_checkpoints/sam2_hiera_small.pt`

### "SAM2 produced no masks"
- **Image too dark** - increase lighting
- **Object not clear** - simplify background
- **Image too small** - use higher resolution
- Try adjusting `--points-per-side`

### "Wrong object detected"
SAM2 picks the largest/most stable mask. If it picks the wrong object:
1. Simplify the scene (remove other objects)
2. Make your object more prominent
3. Crop the image to focus on your object

### "Tracking doesn't work after upload"
- Make sure you clicked **"Capture Reference Frame"** before uploading
- The frame and mask must match (same scene)
- Check that mask is binary (black and white only)

---

## 🎨 **Best Practices**

### Good Objects for SAM2:
- ✅ Clear, solid-colored objects
- ✅ Well-defined boundaries
- ✅ Single prominent object in frame
- ✅ Good contrast with background

### Bad Objects:
- ❌ Transparent objects
- ❌ Objects matching background
- ❌ Multiple similar objects
- ❌ Very small objects

### Scene Setup:
- 💡 **Good lighting** - not too dark
- 🎨 **Plain background** - avoid clutter
- 📏 **Center the object** - SAM2 prefers centered objects
- 🔍 **Fill the frame** - object should be reasonably large

---

## 🆚 **Comparison: SAM2 vs Markerless**

| Feature | SAM2 Tracker | Markerless Tracker |
|---------|-------------|-------------------|
| **Setup** | Capture + segment offline | Drag box in real-time |
| **Accuracy** | AI segmentation (better) | Color-based (simpler) |
| **Speed** | Offline seg + fast tracking | Immediate |
| **Use Case** | Complex shapes | Simple objects |
| **Best For** | Demo/research | Quick testing |

---

## 📊 **Expected Results**

After setup, the SAM2 tracker should:
- ✅ Draw a bounding box around your object
- ✅ Update position as you move it (every frame)
- ✅ Handle slow-moderate movements
- ✅ Show confidence/color info

**Performance:**
- **Real-time tracking**: ~30-60 FPS (browser)
- **SAM2 segmentation**: ~5-30 seconds (offline, one-time)

---

## 💡 **Quick Test**

Don't have time for full setup? Use existing test images:

```bash
# Generate test ArUco marker masks from module 3
python -m module3.process_aruco_batch --input-dir module3/uploads/ArUco

# Or use any clear object photo
python -m module56.run_sam2_single \
  --input test_marker_DICT_4X4_50.png \
  --output test_mask.png
```

---

## 📚 **Additional Resources**

- SAM2 Paper: https://ai.meta.com/research/publications/segment-anything-2/
- SAM2 GitHub: https://github.com/facebookresearch/segment-anything-2
- Module 3 SAM2 Batch: `module3/run_sam2_batch.py`

---

## 🎯 **Summary**

SAM2 tracking is a **two-step process**:

1. **Offline** (one-time): SAM2 segments your object → creates mask
2. **Online** (real-time): Color tracker uses mask → follows object

**Advantages:**
- No manual selection needed
- AI finds object automatically
- Better than pure color tracking

**When to use:**
- Complex object shapes
- Research/demo purposes
- When you want AI-powered initialization
