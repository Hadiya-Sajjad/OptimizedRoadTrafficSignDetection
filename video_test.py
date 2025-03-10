import cv2
import numpy as np

# Paths to the YOLOv4-tiny files
weights_path = "v1.weights"
config_path = "detector_yolov4_tiny.cfg"
classes_path = "classes.names"

# Load YOLOv4-tiny weights and configuration
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load video
cap = cv2.VideoCapture("input_video_8.mp4")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every 30th frame
    if frame_count % 30 == 0:
        # Resize frame to 416x416
        resized_frame = cv2.resize(frame, (416, 416))

        # Preprocess the frame
        blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)

        # Set the input to the network
        net.setInput(blob)

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
                    center_x = int(detection[0] * resized_frame.shape[1])
                    center_y = int(detection[1] * resized_frame.shape[0])
                    width = int(detection[2] * resized_frame.shape[1])
                    height = int(detection[3] * resized_frame.shape[0])

                    # Calculate bounding box coordinates
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    right = int(center_x + width / 2)
                    bottom = int(center_y + height / 2)

                    # Draw bounding box on the frame
                    cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(resized_frame, class_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Frame", resized_frame)
        key = cv2.waitKey(1)

        # Break the loop if 'Esc' key is pressed
        if key == 27:
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
