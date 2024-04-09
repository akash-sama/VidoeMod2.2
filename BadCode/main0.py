from __future__ import absolute_import, division, print_function

import os
import shutil
import math
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from datetime import timedelta
import statistics
import json

from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
import cv2
import time

# This part of the code is trying to do Nudity:


__labels = [
    "FEMALE_GENITALIA_COVERED",
    "SEXUAL_FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_CHEAST_EXPOSED",
    "ANUS_EXPOSED",
    "LEGS_FEETS_EXPOSED",
    "TIGHT_DRESS_BELLY",
    "LEGS_FEETS_COVERED",
    "SHOLDERS_ARMPITS_TIGHT_DRESS",
    "SHOLDERS_ARMPITS_EXPOSED",
    "FACE_SEX_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

flag_classes = {
    "FEMALE_GENITALIA_COVERED": 4, "SEXUAL_FACE_FEMALE": 1, "BUTTOCKS_EXPOSED": 4,
    "FEMALE_BREAST_EXPOSED": 4, "FEMALE_GENITALIA_EXPOSED": 4, "MALE_CHEAST_EXPOSED": 2,
    "ANUS_EXPOSED": 4, "LEGS_FEETS_EXPOSED": 2, "TIGHT_DRESS_BELLY": 1, "FEET_COVERED": 1,
    "SHOLDERS_ARMPITS_TIGHT_DRESS": 1, "SHOLDERS_ARMPITS_EXPOSED": 3, "FACE_SEX_MALE": 0, "BELLY_EXPOSED": 3,
    "MALE_GENITALIA_EXPOSED": 4, "ANUS_COVERED": 3, "FEMALE_BREAST_COVERED": 3,
    "BUTTOCKS_COVERED": 3,
}

# Preprocessing the image
def _read_image(image_path, target_size=320):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt(
        (img_width ** 2 + img_height ** 2) / (new_width ** 2 + new_height ** 2)
    )

    img = cv2.resize(img, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height

    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]

    img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    img = cv2.resize(img, (target_size, target_size))

    image_data = img.astype("float32") / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)

    return image_data, resize_factor, pad_left, pad_top


def _postprocess(output, resize_factor, pad_left, pad_top):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int(round((x - w * 0.5 - pad_left) * resize_factor))
            top = int(round((y - h * 0.5 - pad_top) * resize_factor))
            width = int(round(w * resize_factor))
            height = int(round(h * resize_factor))
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        detections.append(
            {"class": __labels[class_id], "score": float(score), "box": box}
        )

    return detections


# THE MAIN  CLASSFIER FUNCTION
class NudeDetector:
    def __init__(self, providers=None):
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "best.onnx"),
            providers=C.get_available_providers() if not providers else providers,
        )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]  # 320
        self.input_height = input_shape[3]  # 320
        self.input_name = model_inputs[0].name

    def detect(self, image_path):
        preprocessed_image, resize_factor, pad_left, pad_top = _read_image(
            image_path, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(outputs, resize_factor, pad_left, pad_top)

        return detections


# getting the frames of a video
def process_video(video_path, frames_dir='frames'):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Remove existing frames directory
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)

    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])[1].tobytes()
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        if frame_count % (fps // 5) == 0:
            timestamp = timedelta(seconds=frame_count / fps)
            frame_filename = f"{int(timestamp.total_seconds())}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

    return frames_dir


nude_detector = NudeDetector()


# result more clean and easy to read
def calculate_scores(data):
    if not data:
        return 0, 0

    average_score = sum(item['score'] for item in data) / len(data)
    max_contributor = max(data, key=lambda item: item['score'])['class']
    return average_score, max_contributor


def predict(video_dir, directory):
    process_video(video_dir)
    # Initialize a list to store the class values and confidence scores of detected flags
    detected_classes = []
    main_flags = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(directory, filename)
            result = calculate_scores(nude_detector.detect(filepath))
            if result[0]:
                # Add the class value and confidence score of the detected flag to the list
                main_flags.append((flag_classes.get(result[1], 0), result[1]))
                detected_classes.append(
                    (flag_classes.get(result[1], 0), result[0]))  # Default to 0 if flag is not found

    if detected_classes:
        mode_result = statistics.mode([item[0] for item in detected_classes])
        # Create a list of flags corresponding to the mode result
        main_flag = max([item[1] for item in main_flags if item[0] == mode_result])

        # Find the highest confidence score among the detected flags
        highest_confidence = max([item[1] for item in detected_classes if item[0] == mode_result])
    else:
        mode_result = 0
        main_flag = []
        highest_confidence = 0

    output = {
        'Result': mode_result,
        'Main_flag': main_flag,
        'Confidence': "{:.2f}".format(highest_confidence * 120) + "%",
    }
    json_output = json.dumps(output)
    print(json_output)


'''
*************************************************************************************************
This part of the code is trying to do Violence:
'''


model = load_model('Superdupemodel.h5')


def video_reader(filename, frame_count=30, resize_dim=(160, 160)):
    """
    Parameters:
    - filename: Path to the video file.
    - frame_count: Number of frames to read.
    - resize_dim: Dimension to resize frames to.

    Returns:
    - Numpy array of resized video frames.
    """
    frames = np.zeros((frame_count, *resize_dim, 3), dtype=np.float32)
    vc = cv2.VideoCapture(filename)

    if not vc.isOpened():
        raise IOError(f"Cannot open video file {filename}")

    for i in range(frame_count):
        rval, frame = vc.read()
        if not rval:
            break  # Break if no frame is read.
        frm = resize(frame, resize_dim, anti_aliasing=True)
        frames[i] = frm / 255.0 if np.max(frm) > 1 else frm

    vc.release()
    return frames


def pred_fight(model, video, accuracy_threshold=0.9):
    pred_test = model.predict(video)
    return pred_test[0][1] >= accuracy_threshold, pred_test[0][1]


def main_fight(video_path, accuracy=0.9):
    """
    The main function to detect violence, The accuracy of the  model should be modified for wider range

    Parameters:
    - video_path: Path to the video file.
    - accuracy: Accuracy threshold for predictions.(can be changed for cases of overfitting)

    """
    try:
        vid = video_reader(video_path)
        datav = np.expand_dims(vid, axis=0)
        start_time = time.time()
        f, percent = pred_fight(model, datav, accuracy_threshold=accuracy)
        processing_time = time.time() - start_time

        return {'Violence': int(f),
                'Confidence': "{:.2f}".format(percent *100) +"%",
                'processing_time': "{:.2f}".format(processing_time)
                }
    except Exception as e:
        return {'error': str(e)}



if __name__ == '__main__':
    # 10 mins video 'Test1.mp4'
    # 30 mins video 'Test30.,mp4'
    #predict2('Test1.mp4')
    print(main_fight('BloodSex.mp4'))
    predict('Fight.mp4', 'frames')

    # {"Result": 3, "Main_flag": "FEMALE_BREAST_COVERED", "confidence": "66.01"}
