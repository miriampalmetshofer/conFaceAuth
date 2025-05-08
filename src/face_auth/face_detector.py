from mtcnn import MTCNN
import cv2


class FaceDetector:
    def __init__(self, detector_name="MTCNN"):
        if detector_name == "MTCNN":
            self.detector = MTCNN()
        else:
            raise ValueError(f"Unsupported detector: {detector_name}")

    def detect_and_crop(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        if results:
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y + h, x:x + w]
            face = cv2.resize(face, (160, 200))

            return face, [x, y, w, h]

        return None
