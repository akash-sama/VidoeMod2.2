from __future__ import absolute_import, division, print_function

import json

from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
import cv2
import time

model = load_model('Superdupemodel.h5')


def video_reader(filename, frame_count=30, resize_dim=(160, 160)):
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

    try:
        vid = video_reader(video_path)
        datav = np.expand_dims(vid, axis=0)
        start_time = time.time()
        f, percent = pred_fight(model, datav, accuracy_threshold=accuracy)
        processing_time = time.time() - start_time

        result= {'Violence': int(f),
                'Confidence': "{:.2f}".format(percent *100) +"%",
                'processing_time': "{:.2f}".format(processing_time)
                }
        json_result = json.dumps(result, indent=2)
        return json_result

    except Exception as e:
        return {'error': str(e)}


# Example usage
result = main_fight('BloodSex.mp4')
print(result)
