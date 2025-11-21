"""Constants for enrollment processing."""
import numpy as np

# Video processing
FRAME_SAMPLING_INTERVAL = 5  # Process every Nth frame from enrollment video

# Frame sampling distribution
SAMPLING_SEED = 42  # Random seed for reproducible sampling
DISTRIBUTION_MEAN_FRACTION = 0.5  # Sample around middle of sequence
DISTRIBUTION_STDDEV_FRACTION = 0.25  # Standard deviation as fraction of sequence length

# Head pose estimation - Facial landmark indices (MediaPipe Face Mesh)
NOSE_TIP_LANDMARK = 1
CHIN_LANDMARK = 152
LEFT_EYE_CORNER_LANDMARK = 33
RIGHT_EYE_CORNER_LANDMARK = 263
LEFT_MOUTH_CORNER_LANDMARK = 61
RIGHT_MOUTH_CORNER_LANDMARK = 291

# Head pose estimation - 3D face model points (in mm)
FACE_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -330.0, -65.0],      # Chin
    [-225.0, 170.0, -135.0],   # Left eye corner
    [225.0, 170.0, -135.0],    # Right eye corner
    [-150.0, -150.0, -125.0],  # Left mouth corner
    [150.0, -150.0, -125.0]    # Right mouth corner
])

# Direction classification thresholds (degrees)
YAW_THRESHOLD = 15   # Left/right rotation threshold
PITCH_THRESHOLD = 15  # Up/down rotation threshold
