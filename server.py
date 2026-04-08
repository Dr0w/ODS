from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


HOST = "127.0.0.1"
PORT = 5100
COMMAND = "START_DETECTION"
WINDOW_NAME = "ODS Face and Eye Detection"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_ARRAY = NDArray[np.uint8]
Rect = tuple[int, int, int, int]


def open_camera() -> tuple[cv2.VideoCapture, int]:
    for camera_index in (0, 1):
        capture = cv2.VideoCapture(camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if capture.isOpened():
            return capture, camera_index
        capture.release()
    raise RuntimeError("Could not open a webcam on index 0 or 1.")


def cascade_path(filename: str) -> str:
    candidate_paths = [
        Path(__file__).resolve().parent / filename,
        Path(cv2.__file__).resolve().parent / "data" / filename,
    ]

    for path in candidate_paths:
        if path.exists():
            return str(path)

    raise RuntimeError(f"Could not find cascade file: {filename}")


def load_cascades() -> dict[str, cv2.CascadeClassifier]:
    cascade_paths = {
        "frontal": cascade_path("haarcascade_frontalface_default.xml"),
        "profile": cascade_path("haarcascade_profileface.xml"),
        "eyes": cascade_path("haarcascade_eye.xml"),
    }

    cascades = {name: cv2.CascadeClassifier(path) for name, path in cascade_paths.items()}
    failed = [name for name, cascade in cascades.items() if cascade.empty()]
    if failed:
        raise RuntimeError(f"Failed to load Haar cascades: {', '.join(failed)}.")
    return cascades


def preprocess_frame(frame: FRAME_ARRAY, clahe: cv2.CLAHE) -> FRAME_ARRAY:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    return blurred


def to_rectangles(
    detections: Sequence[Sequence[int]] | NDArray[np.int32] | tuple[Rect, ...] | list[Rect],
) -> list[Rect]:
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]


def intersection_over_union(first: Rect, second: Rect) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    intersection_width = max(0, xb - xa)
    intersection_height = max(0, yb - ya)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    first_area = w1 * h1
    second_area = w2 * h2
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area


def non_max_suppression(rectangles: list[Rect], overlap_threshold: float) -> list[Rect]:
    if not rectangles:
        return []

    sorted_rectangles = sorted(rectangles, key=lambda rect: rect[2] * rect[3], reverse=True)
    accepted: list[Rect] = []

    for candidate in sorted_rectangles:
        if all(intersection_over_union(candidate, kept) < overlap_threshold for kept in accepted):
            accepted.append(candidate)

    return accepted


def detect_raw_faces(
    gray: FRAME_ARRAY,
    frame_width: int,
    cascades: dict[str, cv2.CascadeClassifier],
) -> tuple[list[Rect], list[Rect]]:
    frontal_faces = to_rectangles(
        cascades["frontal"].detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=9,
            minSize=(70, 70),
            maxSize=(420, 420),
        )
    )
    profile_faces = to_rectangles(
        cascades["profile"].detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=11,
            minSize=(70, 70),
            maxSize=(420, 420),
        )
    )

    mirrored_gray = cv2.flip(gray, 1)
    mirrored_profiles = to_rectangles(
        cascades["profile"].detectMultiScale(
            mirrored_gray,
            scaleFactor=1.05,
            minNeighbors=11,
            minSize=(70, 70),
            maxSize=(420, 420),
        )
    )
    mirrored_profile_faces = [
        (frame_width - x - w, y, w, h) for (x, y, w, h) in mirrored_profiles
    ]

    return non_max_suppression(frontal_faces, 0.3), non_max_suppression(
        profile_faces + mirrored_profile_faces, 0.3
    )


def detect_eyes(face_gray: FRAME_ARRAY, eye_cascade: cv2.CascadeClassifier) -> list[Rect]:
    upper_face_limit = max(1, int(face_gray.shape[0] * 0.6))
    upper_face = face_gray[:upper_face_limit, :]
    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.03,
        minNeighbors=14,
        minSize=(16, 12),
        maxSize=(90, 70),
    )
    return to_rectangles(eyes)


def filter_eye_candidates(eyes: list[Rect], face_width: int, face_height: int) -> list[Rect]:
    filtered: list[Rect] = []

    for eye in eyes:
        ex, ey, ew, eh = eye
        aspect_ratio = ew / max(eh, 1)
        if ey > int(face_height * 0.45):
            continue
        if ew < int(face_width * 0.12) or ew > int(face_width * 0.42):
            continue
        if eh < int(face_height * 0.08) or eh > int(face_height * 0.28):
            continue
        if not 0.7 <= aspect_ratio <= 2.6:
            continue
        filtered.append(eye)

    return non_max_suppression(filtered, 0.2)


def pick_best_eye_pair(eyes: list[Rect], face_width: int, face_height: int) -> list[Rect]:
    if len(eyes) < 2:
        return eyes[:1]

    best_pair: list[Rect] = []
    best_score = -1.0

    for index, first in enumerate(eyes):
        for second in eyes[index + 1:]:
            left, right = sorted((first, second), key=lambda eye: eye[0])
            left_center_x = left[0] + left[2] / 2
            right_center_x = right[0] + right[2] / 2
            left_center_y = left[1] + left[3] / 2
            right_center_y = right[1] + right[3] / 2

            horizontal_gap = right_center_x - left_center_x
            vertical_gap = abs(left_center_y - right_center_y)
            average_width = (left[2] + right[2]) / 2
            average_height = (left[3] + right[3]) / 2
            size_similarity = min(left[2] * left[3], right[2] * right[3]) / max(
                left[2] * left[3], right[2] * right[3]
            )

            if horizontal_gap < face_width * 0.18 or horizontal_gap > face_width * 0.65:
                continue
            if vertical_gap > face_height * 0.12:
                continue
            if abs(left[2] - right[2]) > average_width * 0.45:
                continue
            if abs(left[3] - right[3]) > average_height * 0.45:
                continue

            score = horizontal_gap + size_similarity * 50 - vertical_gap * 2
            if score > best_score:
                best_score = score
                best_pair = [left, right]

    return best_pair


def validate_frontal_face(
    face: Rect,
    gray: FRAME_ARRAY,
    eye_cascade: cv2.CascadeClassifier,
) -> tuple[bool, list[Rect]]:
    x, y, w, h = face
    face_gray = gray[y:y + h, x:x + w]
    raw_eyes = detect_eyes(face_gray, eye_cascade)
    valid_eyes = filter_eye_candidates(raw_eyes, w, h)
    best_pair = pick_best_eye_pair(valid_eyes, w, h)
    return len(best_pair) == 2, best_pair


def validate_profile_face(
    face: Rect,
    gray: FRAME_ARRAY,
    eye_cascade: cv2.CascadeClassifier,
) -> tuple[bool, list[Rect]]:
    x, y, w, h = face
    face_gray = gray[y:y + h, x:x + w]
    raw_eyes = detect_eyes(face_gray, eye_cascade)
    valid_eyes = filter_eye_candidates(raw_eyes, w, h)

    if not valid_eyes:
        return False, []

    best_eye = sorted(
        valid_eyes,
        key=lambda eye: (eye[2] * eye[3], -abs((eye[1] + eye[3] / 2) - h * 0.25)),
        reverse=True,
    )[0]
    return True, [best_eye]


def detect_faces_and_valid_eyes(
    gray: FRAME_ARRAY,
    frame_width: int,
    cascades: dict[str, cv2.CascadeClassifier],
) -> list[tuple[Rect, list[Rect]]]:
    frontal_candidates, profile_candidates = detect_raw_faces(gray, frame_width, cascades)
    accepted_faces: list[tuple[Rect, list[Rect]]] = []

    for face in frontal_candidates:
        is_valid, eyes = validate_frontal_face(face, gray, cascades["eyes"])
        if is_valid:
            accepted_faces.append((face, eyes))

    if accepted_faces:
        return accepted_faces

    for face in profile_candidates:
        is_valid, eyes = validate_profile_face(face, gray, cascades["eyes"])
        if is_valid:
            accepted_faces.append((face, eyes))

    return accepted_faces


def draw_hud(
    frame: FRAME_ARRAY,
    face_count: int,
    eye_count: int,
    frames_processed: int,
    fps: float,
    camera_index: int,
) -> None:
    hud_lines = [
        f"Camera: {camera_index}",
        f"Faces: {face_count}",
        f"Eyes: {eye_count}",
        f"Frames: {frames_processed}",
        f"FPS: {fps:.1f}",
        "Press q to stop",
    ]

    panel_height = 28 * len(hud_lines) + 10
    cv2.rectangle(frame, (10, 10), (250, panel_height), (20, 20, 20), -1)

    for index, line in enumerate(hud_lines, start=1):
        cv2.putText(
            frame,
            line,
            (20, 10 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def detect_faces_and_eyes() -> str:
    capture, camera_index = open_camera()
    cascades = load_cascades()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    frames_processed = 0
    total_faces_detected = 0
    total_eyes_detected = 0
    peak_faces_in_frame = 0
    started_at = time.perf_counter()

    try:
        while True:
            ret, raw_frame = capture.read()
            if not ret:
                raise RuntimeError("Failed to read a frame from the webcam.")

            frame = raw_frame
            frames_processed += 1
            processed_gray = preprocess_frame(frame, clahe)
            accepted_faces = detect_faces_and_valid_eyes(
                processed_gray, frame.shape[1], cascades
            )

            frame_face_count = len(accepted_faces)
            frame_eye_count = sum(len(eyes) for _, eyes in accepted_faces)
            total_faces_detected += frame_face_count
            total_eyes_detected += frame_eye_count
            peak_faces_in_frame = max(peak_faces_in_frame, frame_face_count)

            for (x, y, w, h), eyes in accepted_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                cv2.putText(
                    frame,
                    "face",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

                for ex, ey, ew, eh in eyes:
                    eye_x1 = x + ex
                    eye_y1 = y + ey
                    eye_x2 = eye_x1 + ew
                    eye_y2 = eye_y1 + eh
                    cv2.rectangle(frame, (eye_x1, eye_y1), (eye_x2, eye_y2), (20, 80, 255), 2)

            elapsed = max(time.perf_counter() - started_at, 1e-6)
            fps = frames_processed / elapsed
            draw_hud(frame, frame_face_count, frame_eye_count, frames_processed, fps, camera_index)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    average_fps = frames_processed / elapsed
    return (
        f"Detection completed using camera index {camera_index}. "
        f"Frames processed: {frames_processed}, total faces detected: {total_faces_detected}, "
        f"total eyes detected: {total_eyes_detected}, peak faces in a frame: {peak_faces_in_frame}, "
        f"average FPS: {average_fps:.1f}."
    )


def start_server(host: str = HOST, port: int = PORT) -> None:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server is listening on {host}:{port}")
    print(f"Send '{COMMAND}' from client.py to start local detection. Press 'q' to stop.")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            with client_socket:
                print(f"Received connection from {client_address[0]}:{client_address[1]}")
                request = client_socket.recv(4096).decode("utf-8").strip()

                if request != COMMAND:
                    response = f"Unsupported command: {request or '<empty>'}"
                else:
                    try:
                        response = detect_faces_and_eyes()
                    except Exception as exc:
                        response = f"Detection failed: {exc}"

                client_socket.sendall(response.encode("utf-8"))
    finally:
        server_socket.close()


if __name__ == "__main__":
    start_server()
