import cv2
import mediapipe as mp
import time

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set low resolution (simulate always-on camera, e.g., VGA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_face_time = 0
face_present = False
FACE_TIMEOUT = 3  # seconds to wait before declaring "face lost"

print("Running face presence detector... Press 'q' to quit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run face detection every 500 ms
        current_time = time.time()
        results = face_detection.process(rgb)

        # Check if a face is detected
        if results.detections:
            if not face_present:
                print("[INFO] Face detected.")
                face_present = True
            last_face_time = current_time
        else:
            # If no face and timeout has passed, declare face lost
            if face_present and (current_time - last_face_time) > FACE_TIMEOUT:
                print("[INFO] Face lost.")
                face_present = False

        # OPTIONAL: Display the camera feed with boxes (for debugging)
        for detection in results.detections or []:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Presence Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
