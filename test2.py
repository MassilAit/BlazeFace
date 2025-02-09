import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
from model import BlazeFace
import cv2
import time  # Import time module
from util import load_anchors, predict_on_image

from torch.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
)
# Must be done before prepare/convert
torch.backends.quantized.engine = 'fbgemm'


# Suppose you already have your model defined (like your BlazeFace).
model_fp32 = BlazeFace()
model_fp32.eval()

# 1) Create custom observers for activation & weight
#    Here, we specify quant_min and quant_max to define the int8 range.
#    Activation is often quint8 (e.g. 0 to 255, or 0 to 127 if you want a narrower range).
#    Weight is often qint8 (e.g. -128 to +127).
activation_observer = MinMaxObserver.with_args(
    quant_min=0,
    quant_max=127,             # or 255, if you want the full 8-bit unsigned
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine
)

weight_observer = PerChannelMinMaxObserver.with_args(
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric
)

# 2) Build the QConfig 
my_qconfig = QConfig(activation=activation_observer, weight=weight_observer)

# 3) Assign this QConfig to your model
model_fp32.qconfig = my_qconfig

# 5) Prepare the model for calibration (inserting observers)
tq.prepare(model_fp32, inplace=True)

# 6) Calibration: Run some representative data through the model
with torch.no_grad():
    for _ in range(10):
        dummy_input = 2*torch.rand(1, 3, 128, 128) - 1  # shape (1,3,128,128), range [-1,1]
        model_fp32(dummy_input)

# 7) Convert to quantized model
model = tq.convert(model_fp32)

# That's it. Now model_int8 uses observers without reduce_range.


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




# At this point, model_int8 is your quantized version of the model
# which uses int8 weights and activations internally on a supported backend.
