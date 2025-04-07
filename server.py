import cv2
import socket


def detect_faces_and_eyes():
    capture = cv2.VideoCapture(1)  # Open the default camera (index 0)

    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Load the Haar cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if face_cascade.empty():
        raise IOError("Error: Failed to load face cascade classifier.")
    if eye_cascade.empty():
        raise IOError("Error: Failed to load eye cascade classifier.")

    while True:
        ret, frame = capture.read()  # Read a frame from the camera
        if not ret:
            break

        # Convert the frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face (green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Define the region of interest for eye detection within the face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes in the ROI
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                # Draw a rectangle around the eye (red)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # Display the processed frame
        cv2.imshow('Face and Eye Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def start_server(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))  # Bind to localhost and the specified port
    server_socket.listen(1)  # Listen for incoming connections

    print(f"Server is listening on port {port}")

    while True:
        client_socket, client_address = server_socket.accept()  # Accept a client connection
        print(f"Received connection from {client_address[0]}:{client_address[1]}")

        detect_faces_and_eyes()  # Call the face and eye detection function

        response = "Face and eye detection completed!"  # Example response
        client_socket.send(response.encode())  # Send response to the client
        client_socket.close()  # Close the client connection


start_server(5100)  # Start the server on port 5000