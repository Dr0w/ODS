import argparse
import json
import socket
import sys


HOST = "127.0.0.1"
PORT = 5100
START_COMMAND = "START_DETECTION"
LIST_CAMERAS_COMMAND = "LIST_CAMERAS"
CONNECT_TIMEOUT_SECONDS = 5.0


def send_command(command: str) -> str:
    with socket.create_connection((HOST, PORT), timeout=CONNECT_TIMEOUT_SECONDS) as client_socket:
        client_socket.settimeout(None)
        client_socket.sendall(command.encode("utf-8"))
        return client_socket.recv(4096).decode("utf-8")


def list_cameras() -> dict[str, object]:
    response = send_command(LIST_CAMERAS_COMMAND)
    response_payload = json.loads(response)
    if not isinstance(response_payload, dict):
        raise RuntimeError("Server returned an invalid camera list response.")
    return response_payload


def parse_camera_entries(camera_list_response: dict[str, object]) -> list[dict[str, int | str]]:
    raw_camera_entries = camera_list_response.get("cameras", [])
    if not isinstance(raw_camera_entries, list):
        raise RuntimeError("Server returned an invalid camera list.")

    parsed_camera_entries: list[dict[str, int | str]] = []
    for raw_camera_data in raw_camera_entries:
        if not isinstance(raw_camera_data, dict):
            raise RuntimeError("Server returned an invalid camera entry.")
        camera_index = raw_camera_data.get("index")
        camera_name = raw_camera_data.get("name")
        if not isinstance(camera_index, int) or not isinstance(camera_name, str):
            raise RuntimeError("Server returned an invalid camera entry.")
        parsed_camera_entries.append({"index": camera_index, "name": camera_name})
    return parsed_camera_entries


def choose_camera_index(available_camera_entries: list[dict[str, int | str]]) -> int | None:
    if not available_camera_entries:
        print("No cameras were detected by the server.")
        return None

    print("Available cameras:")
    for position, listed_camera in enumerate(available_camera_entries, start=1):
        print(f"{position}. {listed_camera['name']} (index {listed_camera['index']})")

    if len(available_camera_entries) == 1:
        selected_index = available_camera_entries[0]["index"]
        if not isinstance(selected_index, int):
            raise RuntimeError("Server returned an invalid camera entry.")
        print(f"Using the only detected camera: {selected_index}")
        return selected_index

    while True:
        choice = input("Choose a camera number: ").strip()
        try:
            selected_position = int(choice)
        except ValueError:
            print("Enter a valid number.")
            continue
        if 1 <= selected_position <= len(available_camera_entries):
            selected_index = available_camera_entries[selected_position - 1]["index"]
            if not isinstance(selected_index, int):
                raise RuntimeError("Server returned an invalid camera entry.")
            return selected_index
        print("Selection out of range.")


def request_detection(camera_index: int | None = None) -> None:
    command = START_COMMAND if camera_index is None else f"{START_COMMAND} {camera_index}"
    print(send_command(command))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ODS client")
    parser.add_argument("--camera", type=int, help="Use a specific camera index")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List camera indices detected by the server and exit",
    )
    parser.add_argument(
        "--choose-camera",
        action="store_true",
        help="Prompt for a camera before detection starts",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Skip the interactive camera prompt and use automatic selection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        if args.list_cameras:
            listed_camera_response = list_cameras()
            listed_camera_entries = parse_camera_entries(listed_camera_response)
            if listed_camera_entries:
                print("Detected cameras:")
                for listed_camera in listed_camera_entries:
                    print(f"- {listed_camera['name']} (index {listed_camera['index']})")
            else:
                print("Detected cameras: []")
        else:
            selected_camera = args.camera
            should_prompt = (
                selected_camera is None
                and not args.no_prompt
                and (args.choose_camera or sys.stdin.isatty())
            )
            if should_prompt:
                prompt_camera_response = list_cameras()
                prompt_camera_entries = parse_camera_entries(prompt_camera_response)
                selected_camera = choose_camera_index(prompt_camera_entries)
            request_detection(selected_camera)
    except ConnectionRefusedError:
        print(f"Could not connect to {HOST}:{PORT}. Start server.py first.")
    except OSError as exc:
        print(f"Client error: {exc}")
    except RuntimeError as exc:
        print(exc)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse server response: {exc}")
