"""
Stage III (CV half) - Defect classification / anomaly detection.

Designed to work with the MVTec-AD dataset (https://www.kaggle.com/datasets/
ipythonx/mvtec-ad) for real defect-classification training via CNN transfer
learning (see `train_cnn_on_mvtec` -- requires TensorFlow + the dataset on
disk locally; this sandbox has neither internet nor TF, so that path is
written but not executed here).

For the live demo / any uploaded image, we ALSO provide a classical
computer-vision anomaly scorer (Laplacian-variance edge/texture analysis +
simple thresholding) that works immediately on any uploaded image with no
training data and no GPU, and reports a severity band per SOP-105. This is
the path wired into the Streamlit app and the Vision Agent by default.
"""

import numpy as np
import cv2

SEVERITY_BANDS = [
    (0.0, 0.02, "minor"),
    (0.02, 0.08, "major"),
    (0.08, 1.01, "critical"),
]


def classical_defect_score(image: np.ndarray) -> dict:
    """Unsupervised, training-free anomaly scoring:
    1) Convert to grayscale, blur to get a 'reference' smooth version.
    2) Absolute difference highlights high-frequency anomalies (scratches,
       cracks, contamination) vs the smoothed baseline.
    3) Otsu-threshold the diff map to segment anomalous pixels.
    4) Defect area fraction -> severity band per SOP-105.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    diff = cv2.absdiff(gray, blurred)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    defect_fraction = float(np.count_nonzero(mask)) / mask.size
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    severity = "minor"
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= defect_fraction < hi:
            severity = label
            break

    confidence = min(0.99, 0.5 + defect_fraction * 3)  # heuristic confidence proxy

    return {
        "defect_fraction": defect_fraction,
        "severity": severity,
        "texture_variance": laplacian_var,
        "confidence": round(confidence, 3),
        "mask": mask,  # for Grad-CAM-style overlay in XAI
        "is_defective": severity != "minor",
    }


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha=0.45) -> np.ndarray:
    """Produces a heatmap-style overlay (Grad-CAM-equivalent visual evidence
    for the classical detector) highlighting the flagged region."""
    color_mask = np.zeros_like(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    color_mask[mask > 0] = [255, 0, 0]
    base = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 1 - alpha, color_mask, alpha, 0)


# ---------------------------------------------------------------------------
# Optional real CNN transfer-learning path for the MVTec-AD dataset.
# Not executed in this sandbox (no TF / no dataset), included for the
# deliverable and to satisfy "CNN/transfer learning for image tasks".
# ---------------------------------------------------------------------------
def train_cnn_on_mvtec(data_dir: str, img_size=(160, 160), epochs=10):
    import tensorflow as tf
    from tensorflow.keras import layers, models as kmodels

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42,
        image_size=img_size, batch_size=32,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=img_size, batch_size=32,
    )
    class_names = train_ds.class_names

    base = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    model = kmodels.Sequential([
        layers.Rescaling(1.0 / 255),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("artifacts/mvtec_cnn.keras")
    return model, class_names


if __name__ == "__main__":
    # quick synthetic smoke test: a clean vs a "scratched" synthetic image
    clean = np.full((200, 200, 3), 180, dtype=np.uint8)
    defective = clean.copy()
    cv2.line(defective, (20, 20), (180, 180), (0, 0, 0), 3)
    print("Clean:", classical_defect_score(clean)["severity"])
    print("Defective:", classical_defect_score(defective)["severity"])
