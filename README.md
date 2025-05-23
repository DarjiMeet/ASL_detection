
  ## 🎥 ASL Detection Demo

[![Watch the demo](https://img.youtube.com/vi/8eo3xt6L8sA/0.jpg)](https://youtu.be/8eo3xt6L8sA)



    
  <h1>ASL Detection</h1>
  <p>This project detects American Sign Language (ASL) gestures using two distinct models:</p>
  <ul>
    <li><strong>Image-Based Model</strong>: Trained on raw hand gesture images using transfer learning with the help of mobilenetv2 with 90% test accuracy.</li>
    <li><strong>Landmark-Based Model</strong>: Trained on hand landmarks extracted using MediaPipe and CNN architecture with 94% test accuracy.</li>
  </ul>

  <h2>app.py</h2>
  <div>UI is made of streamlit which integrate prediction of both image-based model and landmark-base model using ensemble technique by giving weights to each model prediction.</div>
  
  <h2>📁 Image Base Model - contains:</h2>
  <div>hand_sign_model.py</div>
  <div>Unfreeze_layers.py</div>
  <div>hand_sign_data.py</div>
  
  <h2>📁 Landmark-Based Model - contains:</h2>
  <div>sign_model.ipynb</div>
  <div>extract_handmark.py</div>
  <div>evluate.py</div>
  <div>x_landmarks.npy</div>
  <div>y_landmarks.npy</div>
  <div>test_labels.npy</div>
  <div>test_landmarks.npy</div>
  
  <h2>Requirements</h2>
  <ul>
    <li>Streamlit</li>
    <li>tensorflow</li>
    <li>mediapipe</li>
    <li>opencv-python</li>
    <li>numpy</li>
    <li>matplotlib</li>
  </ul>
