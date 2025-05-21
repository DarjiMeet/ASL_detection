import streamlit as st
import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load models globally (outside function to avoid reloading every loop)
landmark_model = load_model("landmark_model.h5")
image_model = load_model("best_asl_model.h5")
labels = sorted(os.listdir("./ASL/ASL_Gestures_36_Classes/train"))

# Initialize session state for ASL detection
if 'asl_running' not in st.session_state:
    st.session_state.asl_running = False

def run_asl_detection_streamlit():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    letter_delay = 1
    space_delay = 2

    letter_buffer = ""
    word_list = []
    last_letter = ""
    letter_start_time = 0
    last_detect_time = time.time()

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while st.session_state.asl_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_time = time.time()
        detected_letter = "N/A"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                landmarks_np = np.array(landmarks).reshape(1, -1)
                landmark_pred = landmark_model.predict(landmarks_np)
                landmark_label = labels[np.argmax(landmark_pred)]

                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)

                padding = 20
                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                x_max = min(x_max + padding, w)
                y_max = min(y_max + padding, h)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size != 0:
                    hand_img_resized = cv2.resize(hand_img, (224, 224))
                    input_img = hand_img_resized.astype("float32") / 255.0
                    input_img = np.expand_dims(input_img, axis=0)

                    image_pred = image_model.predict(input_img)
                    image_label = labels[np.argmax(image_pred)]
                else:
                    image_label = "N/A"
                    image_pred = np.zeros((1, len(labels)))

                ensemble_pred = (landmark_pred * 0.8) + (image_pred * 0.2)
                final_label = labels[np.argmax(ensemble_pred)]

                detected_letter = final_label
                last_detect_time = current_time

                if detected_letter == last_letter:
                    if current_time - letter_start_time >= letter_delay:
                        if detected_letter != "N/A":
                            if not letter_buffer.endswith(last_letter):
                                letter_buffer += detected_letter
                        letter_start_time = current_time
                else:
                    last_letter = detected_letter
                    letter_start_time = current_time
        else:
            if current_time - last_detect_time >= space_delay and letter_buffer:
                word_list.append(letter_buffer)
                letter_buffer = ""
                last_letter = ""
                letter_start_time = 0
                last_detect_time = current_time

        sentence = " ".join(word_list + ([letter_buffer] if letter_buffer else []))
        cv2.putText(frame, f"Detected: {sentence}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")


    cap.release()


# ------------------------- Streamlit UI --------------------------
st.set_page_config(page_title="ASL Recognition", layout="centered")
st.title("ASL Letter Scanner in Streamlit")

col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Start Scanning"):
        st.session_state.asl_running = True
        st.rerun()
with col2:
    if st.button("🛑 Stop Detection"):
        st.session_state.asl_running = False

# Call detection if running
if st.session_state.asl_running:
    run_asl_detection_streamlit()
