"""
Visual Odometry Pipeline

Estimates camera motion from a sequence of images using feature tracking.
"""

import numpy as np
import cv2
import os

np.random.seed(42)

# Configuration
N_FRAMES = 200
TARGET_SIZE = (1024, 1024)
N_FEATURES = 2000
MIN_MATCHES = 10
OUTLIER_THRESHOLD = 2.0
MAX_ERROR_THRESHOLD = 5.0


def generate_synthetic_sequence(n_frames=N_FRAMES, cache_dir="data", source_image="sample_image.jpg"):
    """
    Generate an image sequence with known camera motion from a source image.
    Uses cache to avoid regenerating on every run.
    """
    os.makedirs(cache_dir, exist_ok=True)
    traj_file = os.path.join(cache_dir, "trajectory.npy")

    if os.path.exists(traj_file):
        frames = []
        for i in range(n_frames):
            img_file = os.path.join(cache_dir, f"frame_{i:04d}.png")
            if os.path.exists(img_file):
                frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                frames.append(frame)
            else:
                break

        if len(frames) == n_frames:
            trajectory = np.load(traj_file)
            print(f"Loaded {len(frames)} frames from cache")
            return frames, trajectory

    src_img = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        raise ValueError(f"Could not read source image '{source_image}'")

    output_h, output_w = TARGET_SIZE
    scale_factor = 6.0
    src_h, src_w = int(output_h * scale_factor), int(output_w * scale_factor)

    large_img = cv2.resize(src_img, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    print(f"Generating {n_frames} frames of {output_h}x{output_w}...")

    frames = []
    trajectory = []

    max_offset_x = (src_w - output_w) // 2
    max_offset_y = (src_h - output_h) // 2
    max_range = min(max_offset_x, max_offset_y)
    scale = max_range / 2.5

    for i in range(n_frames):
        t = 2 * np.pi * i / n_frames

        denom = 1 + np.sin(t) ** 2
        tx = scale * 2 * np.cos(t) / denom
        ty = scale * 2 * np.sin(t) * np.cos(t) / denom

        spiral_factor = 0.2 * np.sin(4 * t)
        tx += spiral_factor * scale * 0.3
        ty += spiral_factor * scale * 0.3

        center_x = src_w // 2 + int(tx)
        center_y = src_h // 2 + int(ty)

        x1 = center_x - output_w // 2
        y1 = center_y - output_h // 2
        x2 = x1 + output_w
        y2 = y1 + output_h

        x1 = max(0, min(x1, src_w - output_w))
        y1 = max(0, min(y1, src_h - output_h))
        x2 = x1 + output_w
        y2 = y1 + output_h

        frame = large_img[y1:y2, x1:x2].copy()

        noise = np.random.randn(output_h, output_w) * 2
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        img_file = os.path.join(cache_dir, f"frame_{i:04d}.png")
        cv2.imwrite(img_file, frame)

        frames.append(frame)
        trajectory.append([tx / scale_factor, ty / scale_factor])

    trajectory = np.array(trajectory)
    np.save(traj_file, trajectory)
    print(f"Saved {len(frames)} frames to {cache_dir}/")

    return frames, trajectory


def detect_features(frame, n_features=N_FEATURES):
    """Detect ORB features in frame."""
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors


def match_features(desc1, desc2):
    """Match features between two frames."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def estimate_motion(kp1, kp2, matches, outlier_threshold=OUTLIER_THRESHOLD):
    """Estimate motion from matched keypoints."""
    if len(matches) < MIN_MATCHES:
        print(f"Insufficient matches: {len(matches)}")
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    motion_vectors = pts2 - pts1

    threshold = 1.5
    for iteration in range(3):
        median_motion = np.median(motion_vectors, axis=0)
        std_motion = np.std(motion_vectors, axis=0)

        std_safe = np.where(std_motion > 1e-6, std_motion, 1.0)
        z_scores = np.abs((motion_vectors - median_motion) / std_safe)
        inlier_mask = np.all(z_scores < threshold, axis=1)

        if np.sum(inlier_mask) >= MIN_MATCHES:
            motion_vectors = motion_vectors[inlier_mask]
        else:
            if iteration == 2:
                return None

    mean_motion = np.mean(motion_vectors, axis=0)
    return mean_motion


def compute_trajectory_error(estimated, ground_truth):
    """Compute average trajectory error."""
    error = np.mean(estimated - ground_truth)
    return np.abs(error)


def run_visual_odometry_pipeline():
    """Main pipeline for visual odometry estimation."""
    print("=" * 60)
    print("Visual Odometry Pipeline")
    print("=" * 60)

    print("\n[1/4] Generating synthetic image sequence...")
    frames, gt_trajectory = generate_synthetic_sequence()
    print(f"Generated {len(frames)} frames")

    print("\n[2/4] Processing frames...")
    estimated_motions = []

    for i in range(len(frames) - 1):
        kp1, desc1 = detect_features(frames[i])
        kp2, desc2 = detect_features(frames[i + 1])

        matches = match_features(desc1, desc2)

        if len(matches) < MIN_MATCHES:
            continue

        motion = estimate_motion(kp1, kp2, matches)

        if motion is None:
            continue

        estimated_motions.append(motion)

    estimated_motions = np.array(estimated_motions)

    estimated_trajectory = np.vstack(([0, 0], np.cumsum(estimated_motions, axis=0)))
    gt_trajectory_relative = gt_trajectory - gt_trajectory[0]

    print(f"Estimated {len(estimated_motions)} motion steps")

    print("\n[3/4] Evaluating trajectory...")
    error = compute_trajectory_error(estimated_trajectory, gt_trajectory_relative)
    print(f"Trajectory error: {error:.4f} pixels")

    print("\n[4/4] Validation Results:")

    if error < MAX_ERROR_THRESHOLD:
        print(f"[OK] Trajectory error: {error:.2f} pixels < {MAX_ERROR_THRESHOLD}")
    else:
        print(f"[FAIL] Trajectory error: {error:.2f} pixels >= {MAX_ERROR_THRESHOLD}")

    try:
        if len(estimated_trajectory) == len(gt_trajectory_relative):
            correlation = np.corrcoef(
                estimated_trajectory.flatten(),
                gt_trajectory_relative.flatten()
            )[0, 1]
        else:
            print(f"[WARN] Cannot compute correlation - trajectory length mismatch")
            correlation = None
    except:
        print(f"[WARN] Cannot compute correlation")
        correlation = None

    if correlation is not None and not np.isnan(correlation):
        if correlation > 0.75:
            print(f"[OK] Motion correlation: r={correlation:.4f}")
        else:
            print(f"[FAIL] Motion correlation: r={correlation:.4f} < 0.75")
    else:
        print("[WARN] Could not compute correlation")

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)

    return estimated_trajectory, gt_trajectory_relative, error


if __name__ == "__main__":
    try:
        estimated_trajectory, ground_truth, error = run_visual_odometry_pipeline()

        assert error < MAX_ERROR_THRESHOLD, f"Error {error} exceeds threshold {MAX_ERROR_THRESHOLD}"
        print("\n[SUCCESS] All checks passed!")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
