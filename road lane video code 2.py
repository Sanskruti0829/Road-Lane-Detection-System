import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_edge_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny
def region_of_interest(frame):
    height = frame.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame


def detect_line_segments(frame):
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20
    line_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    lines = cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return line_frame


def lane_detector(frame):

    canny_frame = canny_edge_detector(frame)
    cropped_frame = region_of_interest(canny_frame)
    line_segments = detect_line_segments(cropped_frame)
    lane_frame = cv2.addWeighted(frame, 0.8, line_segments, 1, 1)

    return lane_frame

cap = cv2.VideoCapture("road_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    lane_frame = lane_detector(frame)

    cv2.imshow('Lane Detection', lane_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
