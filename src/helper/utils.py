import cv2


def draw_detection_box(frame, points, ):
    x, y, w, h = points
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

def write_summary_frame(out, frame_count, unauthenticated_count, width, height):
    import numpy as np
    summary_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    text = f"Frames not authenticated: {unauthenticated_count} / {frame_count}"
    cv2.putText(summary_frame, text, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    out.write(summary_frame)
