import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

test_label_path = './ASL/ASL_Gestures_36_Classes/test'

x=[]
y=[]

class_names = sorted(os.listdir(test_label_path))
label_map = {label: idx for idx, label in enumerate(class_names)}

for label in tqdm(label_map):
    class_dir = os.path.join(test_label_path, label)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                landmarks=[]
                for lm in hand_landmark.landmark:
                    landmarks.extend([lm.x, lm.y])

                x.append(landmarks)
                y.append(label_map[label])
                break

x = np.array(x)
y = np.array(y)
np.save("test_landmarks.npy", x)
np.save("test_labels.npy", y)

