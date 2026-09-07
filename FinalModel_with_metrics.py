import os
import cv2
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------------------------------------------------------
# METRICS LOGGING (added for resume/portfolio reporting)
# All numbers get written to metrics_report.json + printed to console.
# ---------------------------------------------------------------------------
metrics_report = {}

# Constants
FRAME_SIZE = (150, 150)
FRAME_SKIP = 2
DATASET_PATH = "./Data/video"
VIOLENT_PATH = os.path.join(DATASET_PATH, "violent")
NONVIOLENT_PATH = os.path.join(DATASET_PATH, "non-violent")
TRAINING_PATH = os.path.join(DATASET_PATH, "Training")

# Ensure directories exist
os.makedirs(TRAINING_PATH, exist_ok=True)


# Data Preprocessing Class
class FrameExtractor:
    def __init__(self, frame_size, frame_skip):
        self.frame_size = frame_size
        self.frame_skip = frame_skip

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % self.frame_skip == 0:
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
                print(f"Extracted frame {count}")
            count += 1
        cap.release()
        return np.array(frames)

# Improved CNN Model using Transfer Learning with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze the base model layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# --- log architecture stats ---
total_params = model.count_params()
trainable_params = sum(np.prod(v.shape) for v in model.trainable_weights)
metrics_report["architecture"] = {
    "base_model": "MobileNetV2 (ImageNet pretrained, frozen)",
    "total_params": int(total_params),
    "trainable_params": int(trainable_params),
    "input_shape": f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}x3",
}

# Compile Model with Early Stopping and Learning Rate Scheduler
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Data Augmentation for Training Data
# validation_split added so we can report a real held-out accuracy number,
# not just training accuracy (training accuracy alone is not resume-worthy).
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "Data/Frames",
    target_size=FRAME_SIZE,
    batch_size=20,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "Data/Frames",
    target_size=FRAME_SIZE,
    batch_size=20,
    class_mode='binary',
    shuffle=False,
    subset='validation'
)

metrics_report["dataset"] = {
    "train_samples": train_generator.samples,
    "val_samples": val_generator.samples,
    "classes": train_generator.class_indices,
}

# --- Train Model with Callbacks, timed ---
train_start = time.time()
history = model.fit(train_generator,
          epochs=30,
          steps_per_epoch=50,
          validation_data=val_generator,
          callbacks=[early_stopping, reduce_lr])
train_end = time.time()

# --- log training results ---
metrics_report["training"] = {
    "epochs_run": len(history.history["loss"]),
    "training_time_seconds": round(train_end - train_start, 2),
    "training_time_minutes": round((train_end - train_start) / 60, 2),
    "final_train_accuracy": round(float(history.history["accuracy"][-1]) * 100, 2),
    "final_val_accuracy": round(float(history.history["val_accuracy"][-1]) * 100, 2),
    "best_val_accuracy": round(float(max(history.history["val_accuracy"])) * 100, 2),
    "final_train_loss": round(float(history.history["loss"][-1]), 4),
    "final_val_loss": round(float(history.history["val_loss"][-1]), 4),
}

# --- save an accuracy/loss curve image for your portfolio/report ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("Saved training_curves.png — use this chart in your resume portfolio/report.")

# Save the trained model
model.save("crowd_anomaly_detection_2.h5")
model_size_mb = round(os.path.getsize("crowd_anomaly_detection_2.h5") / (1024 * 1024), 2)
metrics_report["model_size_mb"] = model_size_mb

# Detect Anomaly from Live Webcam
def predict_live(model, max_frames_for_fps_test=100):
    cap = cv2.VideoCapture(0)  # Use laptop webcam
    alert_sent = False

    frame_count = 0
    inference_times = []
    loop_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, FRAME_SIZE)
        frame_normalized = np.expand_dims(frame_resized / 255.0, axis=0)

        infer_start = time.time()
        prediction = model.predict(frame_normalized, verbose=0)[0][0]
        infer_end = time.time()
        inference_times.append(infer_end - infer_start)

        violent_percentage = round(prediction * 100, 2)
        nonviolent_percentage = round((1 - prediction) * 100, 2)
        
        label = f"Violent: {violent_percentage}% | Non-Violent: {nonviolent_percentage}%"
        print(f"Prediction: {label}")
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Live Camera Feed", frame)

        frame_count += 1

        # --- after N frames, print/save a live FPS + latency report ---
        if frame_count == max_frames_for_fps_test:
            elapsed = time.time() - loop_start
            avg_latency_ms = round((sum(inference_times) / len(inference_times)) * 1000, 2)
            fps = round(frame_count / elapsed, 2)

            metrics_report["live_inference"] = {
                "frames_tested": frame_count,
                "avg_inference_latency_ms": avg_latency_ms,
                "end_to_end_fps": fps,
            }
            with open("metrics_report.json", "w") as f:
                json.dump(metrics_report, f, indent=2)

            print("\n=== LIVE INFERENCE BENCHMARK ===")
            print(f"Frames tested: {frame_count}")
            print(f"Avg inference latency: {avg_latency_ms} ms/frame")
            print(f"End-to-end FPS (incl. capture+display): {fps}")
            print("Saved full metrics_report.json — use these numbers on your resume.")
            print("=================================\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    # Save metrics even if you quit early with 'q'
    if inference_times:
        elapsed = time.time() - loop_start
        avg_latency_ms = round((sum(inference_times) / len(inference_times)) * 1000, 2)
        fps = round(frame_count / elapsed, 2) if elapsed > 0 else 0
        metrics_report["live_inference"] = {
            "frames_tested": frame_count,
            "avg_inference_latency_ms": avg_latency_ms,
            "end_to_end_fps": fps,
        }
        with open("metrics_report.json", "w") as f:
            json.dump(metrics_report, f, indent=2)
        print(f"metrics_report.json updated with {frame_count} frames of benchmark data.")

# Run Live Webcam Detection
predict_live(model)
