import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit , disconnect
from werkzeug.utils import secure_filename
from flask_cors import CORS
from Pre_Processing import FeatureExtraction
import base64
import io
from PIL import Image
# Load model for German traffic sign recognition
german_model_path = r"German_Winning_99_75.h5"
german_model = tf.keras.models.load_model(german_model_path)

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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, origins="http://localhost:5173") 
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def index():
    return jsonify('hi mom')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'message': 'No video part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process video
    process_video(filepath)

    return jsonify({'message': 'Video uploaded and processing started'}), 200

def upload_video():
    if 'video' not in request.files:
        return jsonify({'message': 'No video part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process video
    process_video(filepath)

    return jsonify({'message': 'Video uploaded and processing started'}), 200

def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection and recognition
        try:
            output_width, output_height = 416, 416
            resized_frame = cv2.resize(frame, (output_width, output_height))
            blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())

            detected_signs = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > 0.5:
                        class_name = classes[classId]
                        center_x = int(detection[0] * resized_frame.shape[1])
                        center_y = int(detection[1] * resized_frame.shape[0])
                        width = int(detection[2] * resized_frame.shape[1])
                        height = int(detection[3] * resized_frame.shape[0])
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        right = int(center_x + width / 2)
                        bottom = int(center_y + height / 2)

                        cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        detected_signs.append((class_name, [left, top, right, bottom]))
            print("Detected signs before merging:", detected_signs)


            merged_signs = []
            while len(detected_signs) > 0:
                class_name1, box1 = detected_signs.pop(0)
                for i, (class_name2, box2) in enumerate(detected_signs):
                    if box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1]:
                        merged_box = (min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3]))
                        merged_signs.append((class_name1, merged_box))
                        detected_signs.pop(i)
                        break
                else:
                    merged_signs.append((class_name1, box1))

            detected_signs = merged_signs

            print("Detected signs after merging:", detected_signs)
            final_detected_signs = []
            for class_name, bbox in detected_signs:
                left, top, right, bottom = bbox
                roi = resized_frame[top:bottom, left:right]
                roi = FeatureExtraction(roi)
                roi = cv2.resize(roi, (60, 60))
                recognition_result = german_model.predict(np.expand_dims(roi, axis=0))
                predicted_label_index = np.argmax(recognition_result)
                recognized_class = label_map.get(predicted_label_index, 'Unknown')

                final_detected_signs.append({
                    'recognized_class': recognized_class,
                    'image_path': f"{predicted_label_index}.jpg",
                    'bbox': bbox
                })

            # Encode frame back to base64
            _, buffer = cv2.imencode('.jpg', resized_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            prev_frame = encoded_frame
            socketio.emit('processed_frame', {'frame': encoded_frame, 'detected_signs': final_detected_signs})
        except Exception as e:
            print(f"Error processing frame: {e}")
            socketio.emit('processed_frame', {'frame': prev_frame, 'detected_signs': final_detected_signs})
            break


    print('emit')
    socketio.emit('video_ended')

    cap.release()

app.run()
