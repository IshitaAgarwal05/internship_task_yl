# Intern Task: Visual Odometry Debug & Refactor

## Context
You're working on SLAM for a drone in GPS-denied environments. Visual odometry (estimating camera motion from images) is a foundational SLAM technique.

## The Task
This repository contains a visual odometry implementation that needs debugging. Your job is to:
1. **Fix the bugs** and get the pipeline working correctly
2. **Refactor** the code into clean, modular functions
3. **Add basic error handling**

You have **1 hour**.

## Prerequisites
Before running, ensure you have `sample_image.jpg` in this folder. Any image with distinct features (corners, edges, textures) will work - e.g., a photo of a room, outdoor scene, or object-rich environment.

**How it works**: Your image is upscaled 6x to ~6144x6144, then a moving 1024x1024 window extracts frames in a figure-8 pattern. This simulates camera motion without any artificial borders or black boxes.

## Setup
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run
```bash
python vo_pipeline.py
```

**Note**: The first run generates 200 frames from `sample_image.jpg` and saves them to `data/` as PNG files. Subsequent runs load these cached images (much faster). To regenerate data, delete the `data/` folder.

## What You're Given
- `vo_pipeline.py`: A script that:
  - Loads `sample_image.jpg` and transforms it to simulate camera motion
  - Detects features and estimates motion between frames
  - Computes trajectory error

## Deliverables
1. A working `vo_pipeline.py` with all bugs fixed
2. Code organized into clean functions/modules
3. Comments explaining your fixes
4. Output showing:
   - All bugs fixed (no crashes/assertions fail)
   - Reasonable trajectory error
   - Good motion correlation (r > 0.75)

## Evaluation Criteria
| Criteria | Points |
|----------|--------|
| All bugs fixed | 5 |
| Clean code organization | 2 |
| Error handling added | 2 |
| Trajectory error acceptable | 1 |

---

**Good luck!**
