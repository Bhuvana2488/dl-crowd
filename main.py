# Sample code to load and preprocess crowd images
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset (example: ShanghaiTech Part_A)
def load_data(path):
    images = []
    for img_path in glob.glob(path + "/*.jpg"):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize for CNN input
        images.append(img)
    return np.array(images)

# Synthetic Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, Dense, TimeDistributed

model = Sequential([
    # CNN for spatial features
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(None, 224, 224, 3)),
    TimeDistributed(tf.keras.layers.MaxPooling2D(2,2)),
    
    # LSTM for temporal analysis
    TimeDistributed(tf.keras.layers.Flatten()),
    LSTM(64, return_sequences=True),
    
    # Output layer
    Dense(1, activation='sigmoid')  # Binary classification: "danger" or "safe"
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Compute optical flow between frames
def optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Add optical flow channels to input data
def augment_with_flow(X):
    X_flow = []
    for i in range(len(X)-1):
        X_flow.append(optical_flow(X[i], X[i+1]))
    return np.concatenate([X[:-1], X_flow], axis=-1)
# Train with synthetic data
train_generator = datagen.flow(X_train, y_train, batch_size=32)
model.fit(train_generator, epochs=20)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
# Convert to TF Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
tflite_model = converter.convert()

# Save for edge devices
with open('crowd_anomaly.tflite', 'wb') as f:
    f.write(tflite_model)

# Load on Raspberry Pi (example)
interpreter = tf.lite.Interpreter(model_path="crowd_anomaly.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # CCTV feed

while True:
    ret, frame = cap.read()
    frame_processed = preprocess(frame)  # Resize/Normalize
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], frame_processed)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    if prediction > 0.8:  # Danger threshold
        send_alert()  # SMS/API call to authorities




