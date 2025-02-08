import cv2
import torch
import numpy as np
import time  # Import time module
from model import BlazeFace  # Replace with the actual import path for your BlazeFace implementation
from util import load_anchors, predict_on_image

# Initialize the BlazeFace model
model = BlazeFace()  # Set to True if you're using the back model
model.load_weights("blazeface.pth")  # Replace with the path to your weights
anchors= load_anchors("anchors.npy")  # Replace with the path to your anchor file
model.eval()


# Helper function to draw detections on the frame
def draw_detections(frame, detections):
    h, w, _ = frame.shape
    for detection in detections:
        ymin, xmin, ymax, xmax = detection[:4]
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


# Open webcam (front-facing camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'q' to exit.")

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    # Resize the frame to match the model's input size (128x128 or 256x256)
    input_size = 128 
    resized_frame = cv2.resize(frame, (input_size, input_size))

    # Run detection using predict_on_image
    detections = predict_on_image(resized_frame,model,anchors)


    # Draw detections on the original frame
    if len(detections) > 0:
        draw_detections(frame, detections)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Live Face Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()