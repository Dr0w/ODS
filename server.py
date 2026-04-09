from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


HOST = "127.0.0.1"
PORT = 5100
START_COMMAND = "START_DETECTION"
LIST_CAMERAS_COMMAND = "LIST_CAMERAS"
WINDOW_NAME = "ODS Face and Eye Detection"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_ARRAY = NDArray[np.uint8]
Rect = tuple[int, int, int, int]


def try_open_camera(camera_index: int) -> cv2.VideoCapture | None:
    system_name = platform.system()
    if system_name == "Darwin":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    elif system_name == "Windows":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if capture.isOpened():
        return capture
    capture.release()
    return None


def discover_macos_cameras() -> list[dict[str, int | str]]:
    if platform.system() != "Darwin":
        return []

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    raw_cameras = payload.get("SPCameraDataType", [])
    if not isinstance(raw_cameras, list):
        return []

    cameras: list[dict[str, int | str]] = []
    for index, item in enumerate(raw_cameras):
        if not isinstance(item, dict):
            continue
        name = item.get("_name")
        if isinstance(name, str) and name.strip():
            cameras.append({"index": index, "name": name.strip()})
    return cameras


def discover_windows_cameras() -> list[dict[str, int | str]]:
    if platform.system() != "Windows":
        return []

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPClass -eq 'Image' -or $_.Service -match 'usbvideo' } | "
            "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
        ),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return []

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, str):
        names = [payload]
    elif isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        names = payload
    else:
        return []

    return [{"index": index, "name": name.strip()} for index, name in enumerate(names) if name.strip()]


def available_cameras() -> list[dict[str, int | str]]:
    system_name = platform.system()
    if system_name == "Darwin":
        cameras = discover_macos_cameras()
    elif system_name == "Windows":
        cameras = discover_windows_cameras()
    else:
        cameras = []

    if cameras:
        return cameras
    return [{"index": 0, "name": "Default camera"}]


def log_available_cameras() -> None:
    cameras = available_cameras()
    if not cameras:
        print("No cameras discovered.")
        return

    print("Discovered cameras:")
    for camera in cameras:
        camera_index = camera.get("index")
        camera_name = camera.get("name")
        print(f"- {camera_name} (index {camera_index})")


def open_camera(camera_index: int | None = None) -> tuple[cv2.VideoCapture, int]:
    if camera_index is not None:
        capture = try_open_camera(camera_index)
        if capture is not None:
            return capture, camera_index
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    for discovered_camera in available_cameras():
        detected_camera_index = discovered_camera["index"]
        if not isinstance(detected_camera_index, int):
            continue
        capture = try_open_camera(detected_camera_index)
        if capture is not None:
            return capture, detected_camera_index
    raise RuntimeError("Could not open the selected camera.")


def cascade_path(filename: str) -> str:
    cv2_module_file = cv2.__file__
    if cv2_module_file is None:
        raise RuntimeError("OpenCV module path is unavailable; cannot locate cascade data files.")

    candidate_paths = [
        Path(__file__).resolve().parent / filename,
        Path(cv2_module_file).resolve().parent / "data" / filename,
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


def pick_best_single_eye(eyes: list[Rect], face_width: int, face_height: int) -> list[Rect]:
    if not eyes:
        return []

    best_eye = sorted(
        eyes,
        key=lambda eye: (
            eye[2] * eye[3],
            -abs((eye[0] + eye[2] / 2) - face_width * 0.5),
            -abs((eye[1] + eye[3] / 2) - face_height * 0.25),
        ),
        reverse=True,
    )[0]
    return [best_eye]


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
    if len(best_pair) == 2:
        return True, best_pair

    best_single_eye = pick_best_single_eye(valid_eyes, w, h)
    return len(best_single_eye) == 1, best_single_eye


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

    return True, pick_best_single_eye(valid_eyes, w, h)


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


def draw_face_outline(frame: FRAME_ARRAY, face: Rect) -> None:
    x, y, w, h = face
    center = (x + w // 2, y + h // 2)
    axes = (max(1, int(w * 0.42)), max(1, int(h * 0.52)))
    cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 220, 0), 2, cv2.LINE_AA)


def draw_eye_outline(frame: FRAME_ARRAY, face: Rect, eye: Rect) -> None:
    x, y, _, _ = face
    ex, ey, ew, eh = eye
    center = (x + ex + ew // 2, y + ey + eh // 2)
    axes = (max(1, int(ew * 0.5)), max(1, int(eh * 0.38)))
    cv2.ellipse(frame, center, axes, 0, 0, 360, (20, 80, 255), 2, cv2.LINE_AA)


def close_detection_window() -> None:
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # Give macOS a brief moment to tear down the AVFoundation window cleanly.
    time.sleep(0.05)


def detect_faces_and_eyes(camera_index: int | None = None) -> str:
    capture, active_camera_index = open_camera(camera_index)
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
                draw_face_outline(frame, (x, y, w, h))
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
                    draw_eye_outline(frame, (x, y, w, h), (ex, ey, ew, eh))

            elapsed = max(time.perf_counter() - started_at, 1e-6)
            fps = frames_processed / elapsed
            draw_hud(
                frame,
                frame_face_count,
                frame_eye_count,
                frames_processed,
                fps,
                active_camera_index,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            try:
                window_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                window_visible = False
            if key == ord("q") or key == 27 or not window_visible:
                break
    finally:
        capture.release()
        close_detection_window()

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    average_fps = frames_processed / elapsed
    return (
        f"Detection completed using camera index {active_camera_index}. "
        f"Frames processed: {frames_processed}, total faces detected: {total_faces_detected}, "
        f"total eyes detected: {total_eyes_detected}, peak faces in a frame: {peak_faces_in_frame}, "
        f"average FPS: {average_fps:.1f}."
    )


def parse_camera_index(request: str) -> int | None:
    parts = request.split(maxsplit=1)
    if len(parts) == 1:
        return None
    camera_index_text = parts[1].strip()
    if not camera_index_text:
        return None
    try:
        return int(camera_index_text)
    except ValueError as exc:
        raise RuntimeError(f"Invalid camera index: {camera_index_text}") from exc


def handle_request(request: str) -> str:
    if request == LIST_CAMERAS_COMMAND:
        payload = {
            "cameras": available_cameras(),
        }
        return json.dumps(payload)

    if request == START_COMMAND or request.startswith(f"{START_COMMAND} "):
        try:
            return detect_faces_and_eyes(parse_camera_index(request))
        except Exception as exc:
            return f"Detection failed: {exc}"

    return f"Unsupported command: {request or '<empty>'}"


def start_server(host: str = HOST, port: int = PORT) -> None:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server is listening on {host}:{port}")
    log_available_cameras()
    print(
        f"Send '{START_COMMAND}' from client.py to start local detection. "
        f"Use '{LIST_CAMERAS_COMMAND}' to inspect camera indices. Press 'q' to stop."
    )

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            with client_socket:
                print(f"Received connection from {client_address[0]}:{client_address[1]}")
                request = client_socket.recv(4096).decode("utf-8").strip()
                response = handle_request(request)
                client_socket.sendall(response.encode("utf-8"))
    finally:
        server_socket.close()


if __name__ == "__main__":
    start_server()
