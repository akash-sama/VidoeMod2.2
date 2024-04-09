from __future__ import absolute_import, division, print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import sys

sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

import json
import time
import onnxruntime
import numpy as np
from collections import Counter
from skimage.transform import resize
from tensorflow.keras.models import load_model
from onnxruntime.capi import _pybind_state as C

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__labels = [
    "FEMALE_GENITALIA_COVERED",
    "SAFE_Female_face",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "LEGS_FEET_EXPOSED",
    "BELLY_COVERED",
    "LEGS_FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "SAFE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


def preprocess_image(image_array, target_size=320):
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    scale = min(target_size / height, target_size / width)
    new_height, new_width = int(height * scale), int(width * scale)
    img_resized = cv2.resize(img, (new_width, new_height))

    img_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    top_left_x = (target_size - new_width) // 2
    top_left_y = (target_size - new_height) // 2

    img_square[top_left_y:top_left_y + new_height, top_left_x:top_left_x + new_width] = img_resized

    img_normalized = img_square.astype('float32') / 255.0
    img_preprocessed = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)

    return img_preprocessed


def _postprocess(output):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            class_ids.append(class_id)
            scores.append(max_score)

    detections = []
    safList = ["SAFE_FACE_FEMALE", "SAFETY_COVERED"]

    if not scores or any(item in scores for item in safList):
        detections.append(
            {"class": 'SAFE',
             "score": 0}
        )

    else:
        score = max(scores)
        class_id = max(class_ids)
        detections.append(
            {"class": __labels[class_id],
             "score": float(score)}
        )

    return detections


class NudeDetector:
    def __init__(self, providers=None):
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "Models/best.onnx"),
            providers=C.get_available_providers() if not providers else providers,
        )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]  # 320
        self.input_height = input_shape[3]  # 320
        self.input_name = model_inputs[0].name

    def detect(self, image_input):
        if isinstance(image_input, str):
            image_array = cv2.imread(image_input)
        else:
            image_array = image_input

        preprocessed_image = preprocess_image(
            image_array, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(outputs)

        return detections


def predict(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    detector = NudeDetector()
    start_time1 = time.time()

    detection_scores = []
    detection_classes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)

        for detection in detections:
            detection_scores.append(detection['score'])
            detection_classes.append(detection['class'])

    cap.release()

    max_sexiness_score = max(detection_scores) if detection_scores else 0
    class_counts = Counter(detection_classes)
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    total_detections = sum(class_counts.values())

    top_classes_data = [
        {"Class": item, "Percentage": round((count / total_detections) * 100, 2)}
        for item, count in sorted_class_counts[:5]
    ]

    Timetaken = time.time() - start_time1

    result = {
        "Top_SEXUAL_Flags": top_classes_data,
        "Confidence": "{:.2f}".format(max_sexiness_score * 102),
        "time": "{:.2f}".format(Timetaken) + 'Sec'
    }

    list0 = ["SAFE"]
    list1 = ["SAFE_Female_face", "ARMPITS_COVERED", "LEGS_FEET_COVERED", "BUTTOCKS_COVERED"]
    list2 = ["LEGS_FEET_EXPOSED", "ARMPITS_EXPOSED", "BELLY_EXPOSED","FEMALE_BREAST_COVERED"]
    list3 = ["ANUS_COVERED", "FEMALE_GENITALIA_COVERED"]
    list4 = ["FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_EXPOSED"]

    flags = {0: list0, 1: list1, 2: list2, 3: list3, 4: list4}

    flag = 0

    for item in result["Top_SEXUAL_Flags"]:
        for key, value in flags.items():
            if item["Class"] in value:
                flag = max(flag, key)

    result["flag"] = flag

    json_result = json.dumps(result, indent=2)
    return json_result


'''
*************************************************************************************************
This part of the code is trying to do Violence:
**************************************************************************************************
'''

model = load_model('Models/Superdupemodel.h5')


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
        raise IOError("Cannot open video file {filename}")

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


"""
    The main function to detect violence
"""
def main_fight(video_path, accuracy=0.9):
    """
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
        Result1 = {'Violence': int(f),
                   'Confidence': "{:.2f}".format(percent * 100) + "%",
                   'processing_time': "{:.2f}".format(processing_time)
                   }
        Result1 = json.dumps(Result1, indent=2)
        return Result1

    except Exception as e:
        return {'error': str(e)}


sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

if __name__ == '__main__':
    VideoPath = 'Videos/Test1.mp4'

    # print(main_fight(VideoPath))
    print(predict(VideoPath))
