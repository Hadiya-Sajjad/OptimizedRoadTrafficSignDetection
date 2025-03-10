# Implementation based on:
# Abdel-Salam, R., Mostafa, R., & Abdel-Gawad, A.H. (2022).
# "RIECNN: real-time image enhanced CNN for traffic sign recognition."
# Neural Comput & Applic, 34, 6085–6096. DOI: 10.1007/s00521-021-06762-5

import cv2
import numpy as np
import tensorflow as tf
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

# Load video
video_path = "input_video_7.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0

# Get the width and height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output width and height for resized video
output_width = 416
output_height = 416

# Load the "no_sign.jpg" image
no_sign_image = cv2.imread("no_sign.jpg")

# Create windows for processed video and recognized sign
cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Recognized Sign", cv2.WINDOW_NORMAL)

# Resize the windows
cv2.resizeWindow("Processed Video", output_width, output_height)
cv2.resizeWindow("Recognized Sign", output_width * 2, output_height)

# Move the windows to different positions on the screen
cv2.moveWindow("Processed Video", 100, 100)  # Move "Processed Video" window to (100, 100) position
cv2.moveWindow("Recognized Sign", 100 + output_width + 10, 100)  # Move "Recognized Sign" window to the right of "Processed Video" window

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every 3rd frame to get 10fps
    if frame_count %  3 == 0:
        # Resize frame to 416x416
        resized_frame = cv2.resize(frame, (output_width, output_height))

        # Preprocess the frame
        blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)

        # Set the input to the network
        net.setInput(blob)

        # Forward pass to get the detections
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Create a blank canvas to display recognized sign images
        sign_canvas = np.zeros((output_height, output_width * 2, 3), dtype=np.uint8)

        # Process the detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > 0.5:
                    class_name = classes[classId]

                    # Extract bounding box coordinates
                    center_x = int(detection[0] * output_width)
                    center_y = int(detection[1] * output_height)
                    width = int(detection[2] * output_width)
                    height = int(detection[3] * output_height)

                    # Calculate bounding box coordinates
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    right = int(center_x + width / 2)
                    bottom = int(center_y + height / 2)

                    # Crop the region of interest (ROI) from the frame
                    roi = resized_frame[top:bottom, left:right]
                    
                    # Perform recognition using the German model
                    roi = FeatureExtraction(roi)
                    resized_image = cv2.resize(roi, (60, 60))
                    recognition_result = german_model.predict(np.expand_dims(resized_image, axis=0))
                    predicted_label_index = np.argmax(recognition_result)
                    recognized_class = label_map.get(predicted_label_index, 'Unknown')

                    # Load image of traffic sign
                    image_path = f"{predicted_label_index}.jpg"
                    sign_image = cv2.imread(image_path)

                    # Check if sign_image is loaded successfully
                    if sign_image is None:
                        print(f"Failed to load image: {image_path}")
                        continue

                    # Resize sign image to fit the canvas region
                    sign_image_resized = cv2.resize(sign_image, (output_width, output_height))

                    # Display recognized class name on the canvas
                    cv2.putText(sign_canvas, recognized_class, (output_width + 10, output_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Calculate start and end coordinates for placing sign image on canvas
                    start_x = output_width if start_x > output_width else start_x
                    end_x = output_width * 2 if end_x < output_width else output_width * 2

                    # Paste sign image onto canvas
                    sign_canvas[:, start_x:end_x, :] = sign_image_resized

        # Display processed video frame and recognized sign images
        cv2.imshow("Processed Video", resized_frame)
        cv2.imshow("Recognized Sign", sign_canvas)

        # Calculate the appropriate waitKey time based on the video's frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 30:
            wait_time = int(1000 / 30)
        elif fps == 25:
            wait_time = int(1000 / 25)
        else:
            wait_time = 1

        # Wait for the specified duration
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
