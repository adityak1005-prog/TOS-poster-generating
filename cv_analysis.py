"""
Stage 2: CV trait extraction.
- MediaPipe Pose (BlazePose) -> 33 body keypoints, ~15-40ms on CPU
- MediaPipe Face Mesh -> 468 face landmarks, ~15-30ms on CPU
- Dominant color extraction -> plain numpy color quantization, no model at all
"""

import numpy as np
from PIL import Image
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# Reuse detector instances across requests -- loading the model graph is the
# expensive part (~200-400ms), running inference on a frame is cheap.
_pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,  # 0=lite, 1=full, 2=heavy -- 1 is the CPU sweet spot
    enable_segmentation=False,
    min_detection_confidence=0.5,
)
_face_detector = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,  # adds iris + eye-region landmarks, needed for glasses heuristic
    min_detection_confidence=0.5,
)

POSE_LANDMARKS = mp_pose.PoseLandmark


def _dominant_color(crop: np.ndarray, k: int = 3) -> tuple[int, int, int]:
    """
    Cheap dominant-color extraction without sklearn: downsample, bucket colors
    into a coarse grid, return the most populous bucket's mean color.
    This is O(pixels) and takes single-digit milliseconds on a torso crop.
    """
    if crop.size == 0:
        return (128, 128, 128)

    small = Image.fromarray(crop).resize((32, 32))
    arr = np.asarray(small).reshape(-1, 3).astype(np.int32)

    # bucket into 32-level bins per channel to merge near-duplicate colors
    bins = (arr // 32) * 32
    uniq, counts = np.unique(bins, axis=0, return_counts=True)
    dominant_bin = uniq[np.argmax(counts)]

    # refine: average actual pixels that fall in the winning bucket
    mask = np.all(bins == dominant_bin, axis=1)
    refined = arr[mask].mean(axis=0).astype(int)
    return tuple(int(c) for c in refined)


def _classify_color_name(rgb: tuple[int, int, int]) -> str:
    """Map an RGB triple to a small set of named colors used by the matcher."""
    r, g, b = rgb
    named = {
        "red": (200, 30, 30), "blue": (30, 60, 200), "green": (30, 140, 60),
        "black": (25, 25, 25), "white": (235, 235, 235), "yellow": (220, 200, 40),
        "gray": (130, 130, 130), "orange": (230, 120, 30), "purple": (120, 50, 160),
    }
    best_name, best_dist = "gray", float("inf")
    for name, ref in named.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def _pose_descriptor(landmarks) -> str:
    """
    Turn raw keypoints into a small set of semantic pose tags the character
    matcher can score against (e.g. "arm_extended", "hand_on_hip").
    Simple geometric rules on normalized landmark coordinates -- no model.
    """
    lm = landmarks.landmark
    tags = []

    l_shoulder, r_shoulder = lm[POSE_LANDMARKS.LEFT_SHOULDER], lm[POSE_LANDMARKS.RIGHT_SHOULDER]
    l_wrist, r_wrist = lm[POSE_LANDMARKS.LEFT_WRIST], lm[POSE_LANDMARKS.RIGHT_WRIST]
    l_hip, r_hip = lm[POSE_LANDMARKS.LEFT_HIP], lm[POSE_LANDMARKS.RIGHT_HIP]

    shoulder_width = abs(l_shoulder.x - r_shoulder.x) + 1e-6

    # arm extended sideways: wrist far from shoulder horizontally, similar height
    for side, shoulder, wrist in (("left", l_shoulder, l_wrist), ("right", r_shoulder, r_wrist)):
        horiz_reach = abs(wrist.x - shoulder.x) / shoulder_width
        vert_diff = abs(wrist.y - shoulder.y) / shoulder_width
        if horiz_reach > 1.3 and vert_diff < 0.8:
            tags.append(f"arm_extended_{side}")
        if wrist.y < shoulder.y - 0.15:
            tags.append(f"arm_raised_{side}")

    # hand on hip: wrist near the hip on the same side, close together
    for side, hip, wrist in (("left", l_hip, l_wrist), ("right", r_hip, r_wrist)):
        if abs(wrist.x - hip.x) / shoulder_width < 0.4 and abs(wrist.y - hip.y) / shoulder_width < 0.4:
            tags.append(f"hand_on_hip_{side}")

    return tags[0] if tags else "neutral_stance"


def _has_glasses(face_landmarks, image_np: np.ndarray) -> bool:
    """
    Heuristic, not a trained classifier: measure edge density in a thin band
    across the bridge of the nose / eye region. Glasses frames create a sharp,
    fairly horizontal edge that bare skin doesn't. Good enough for a fun booth,
    not medical grade -- swap for a small trained classifier if you want more
    reliability across lighting conditions.
    """
    import cv2

    lm = face_landmarks.landmark
    h, w = image_np.shape[:2]
    # landmark indices for left/right eye outer corners in refined FaceMesh
    left_eye, right_eye = lm[33], lm[263]
    x1, x2 = int(left_eye.x * w), int(right_eye.x * w)
    y = int(((left_eye.y + right_eye.y) / 2) * h)
    band_half = max(4, int(0.02 * h))
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    if x2 - x1 < 5:
        return False

    band = image_np[max(0, y - band_half):y + band_half, x1:x2]
    if band.size == 0:
        return False

    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    return edge_density > 0.06  # tuned empirically, adjust after a test run


def analyze(image_np: np.ndarray) -> dict:
    """
    Main entry point for stage 2. Takes an RGB numpy array (the captured
    photo), returns the structured trait JSON that feeds character matching.
    Target: well under 1 second on a laptop CPU for a single photo.
    """
    h, w = image_np.shape[:2]
    traits = {
        "pose": "neutral_stance",
        "outfit_color": "gray",
        "outfit_color_rgb": (128, 128, 128),
        "glasses": False,
        "pose_keypoints": None,  # kept for the split-screen skeleton overlay
        "face_found": False,
    }

    pose_result = _pose_detector.process(image_np)
    if pose_result.pose_landmarks:
        traits["pose"] = _pose_descriptor(pose_result.pose_landmarks)
        traits["pose_keypoints"] = [
            (lm.x * w, lm.y * h, lm.visibility) for lm in pose_result.pose_landmarks.landmark
        ]

        lm = pose_result.pose_landmarks.landmark
        l_sh, r_sh = lm[POSE_LANDMARKS.LEFT_SHOULDER], lm[POSE_LANDMARKS.RIGHT_SHOULDER]
        l_hip, r_hip = lm[POSE_LANDMARKS.LEFT_HIP], lm[POSE_LANDMARKS.RIGHT_HIP]
        top = int(min(l_sh.y, r_sh.y) * h)
        bottom = int(max(l_hip.y, r_hip.y) * h)
        left = int(min(l_sh.x, r_sh.x, l_hip.x, r_hip.x) * w)
        right = int(max(l_sh.x, r_sh.x, l_hip.x, r_hip.x) * w)
        torso_crop = image_np[max(0, top):max(0, bottom), max(0, left):max(0, right)]

        rgb = _dominant_color(torso_crop)
        traits["outfit_color_rgb"] = rgb
        traits["outfit_color"] = _classify_color_name(rgb)

    face_result = _face_detector.process(image_np)
    if face_result.multi_face_landmarks:
        traits["face_found"] = True
        traits["glasses"] = _has_glasses(face_result.multi_face_landmarks[0], image_np)

    return traits
