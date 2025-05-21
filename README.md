
  ## 🎥 ASL Detection Demo

[![Watch the demo](https://img.youtube.com/vi/8eo3xt6L8sA/0.jpg)](https://youtu.be/8eo3xt6L8sA)



    
  <h1>ASL Detection</h1>
  <p>This project detects American Sign Language (ASL) gestures using two distinct models:</p>
  <ul>
    <li><strong>Image-Based Model</strong>: Trained on raw hand gesture images with the help of mobilenetv2.</li>
    <li><strong>Landmark-Based Model</strong>: Trained on hand landmarks extracted using MediaPipe and CNN architecture.</li>
  </ul>
  
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
    <li>tensorflow==2.18.0</li>
    <li>mediapipe</li>
    <li>opencv-python</li>
    <li>numpy</li>
    <li>matplotlib</li>
  </ul>
