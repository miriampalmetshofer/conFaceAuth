from face_auth.core.enrollment import HeadPose, HeadDirection

class DirectionClassifier:
    """Classifies head direction from pose angles."""

    def __init__(self, yaw_threshold: float, pitch_threshold: float):
        """Initialize direction classifier.

        Args:
            yaw_threshold: Threshold in degrees for left/right classification
            pitch_threshold: Threshold in degrees for up/down classification
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

    def classify(self, pose: HeadPose) -> HeadDirection:
        """Classify head direction from pose angles.

        Args:
            pose: HeadPose with pitch, yaw, roll angles

        Returns:
            Classified head direction
        """
        # Check if looking straight ahead
        if abs(pose.yaw) < self.yaw_threshold and abs(pose.pitch) < self.pitch_threshold:
            return HeadDirection.FRONT

        # Check horizontal direction (yaw has priority)
        if pose.yaw > self.yaw_threshold:
            return HeadDirection.RIGHT  # Person looking to their right
        elif pose.yaw < -self.yaw_threshold:
            return HeadDirection.LEFT  # Person looking to their left

        # Check vertical direction
        if pose.pitch > self.pitch_threshold:
            return HeadDirection.DOWN
        elif pose.pitch < -self.pitch_threshold:
            return HeadDirection.UP

        # Default to front for ambiguous cases
        return HeadDirection.FRONT
