from picamera2 import Picamera2
import tensorflow as tf
import numpy as np
import cv2
import json

# ===== LOAD CLASS (LIST) =====
with open("linhkien_classes.json", "r") as f:
    class_names = json.load(f)   # dạng list

# ===== LOAD MODEL =====
model = tf.keras.layers.TFSMLayer(
    "linhkien_saved",
    call_endpoint="serving_default"
)

# ===== CAMERA =====
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (640, 480)})
)
picam2.start()

print("Dang nhan dien...")

# ===== LOOP =====
while True:
    frame = picam2.capture_array()

    # ===== FIX MÀU =====
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # tăng sáng + giảm tím
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
    frame[:, :, 2] = frame[:, :, 2] * 0.9

    # ===== PREPROCESS =====
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ===== PREDICT =====
    pred = model(img)
    pred = list(pred.values())[0].numpy()

    class_id = int(np.argmax(pred))
    conf = float(np.max(pred))

    # ===== LOGIC CONF =====
    if conf < 0.7:
        label = "khong co linh kien"
        color = (0, 0, 255)   # đỏ
    else:
        if class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = "Unknown"
        color = (0, 255, 0)   # xanh

    text = f"{label} ({conf:.2f})"

    # ===== HIỂN THỊ =====
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("AI Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()