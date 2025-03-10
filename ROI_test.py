from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as tf
import json
from Pre_Processing import FeatureExtraction

# Load model for German traffic sign recognition
german_model_path = r"German_Winning_99_75.h5"
german_model = tf.keras.models.load_model(german_model_path)

# Load label mappings for German traffic signs
label_map = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Load object recognition model
object_model_path = "v7.h5"
object_model = tf.keras.models.load_model(object_model_path)

# Load label mappings for object recognition
with open("v7.json", "r") as f:
    object_labels = json.load(f)


# Load YOLOv4-tiny weights and configuration
weights_path = "v1.weights"
config_path = "detector_yolov4_tiny.cfg"
classes_path = "classes.names"

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names for YOLO
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
image_path = "images.jpg"
orig_image = cv2.imread(image_path)

# Resize the image to 416x416
resized_image = cv2.resize(orig_image, (416, 416))

cv2.imwrite("resized_test.jpg", resized_image)

# Preprocess the image
blob = cv2.dnn.blobFromImage(resized_image, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
cv2.imwrite("resized_test_after_preprocess.jpg", resized_image)

# Set the input to the network
net.setInput(blob)
type(blob)
# Forward pass to get the detections
outs = net.forward(net.getUnconnectedOutLayersNames())

# Process the detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > 0.5:
            class_name = classes[classId]

            # Extract bounding box coordinates
            center_x = int(detection[0] * resized_image.shape[1])
            center_y = int(detection[1] * resized_image.shape[0])
            width = int(detection[2] * resized_image.shape[1])
            height = int(detection[3] * resized_image.shape[0])

            # Calculate bounding box coordinates
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            right = int(center_x + width / 2)
            bottom = int(center_y + height / 2)

            # Draw bounding box on original image
            cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(resized_image, class_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Crop the region of interest (ROI) from the original image
            roi = resized_image[top:bottom, left:right]

            # Save the ROI using PIL
            roi_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_image.save(f"ROI_{class_name}.jpg")

# Save the image with bounding boxes overlaid
cv2.imwrite("Detected_Objects_test_resized.jpg", resized_image)
