import socket


HOST = "127.0.0.1"
PORT = 5100
COMMAND = "START_DETECTION"


def request_detection():
    with socket.create_connection((HOST, PORT), timeout=5) as client_socket:
        client_socket.sendall(COMMAND.encode("utf-8"))
        response = client_socket.recv(4096).decode("utf-8")
        print(response)


if __name__ == "__main__":
    try:
        request_detection()
    except ConnectionRefusedError:
        print(f"Could not connect to {HOST}:{PORT}. Start server.py first.")
    except OSError as exc:
        print(f"Client error: {exc}")