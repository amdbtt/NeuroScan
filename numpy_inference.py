import time
import numpy as np
import h5py
from PIL import Image

# Empirically verified against the labeled Testing set (Kaggle-style folders):
#   meningioma -> idx 1 (~90%), notumor -> idx 2 (~99%),
#   glioma -> idx 3 (~70%), pituitary -> idx 4 (~96%)
# Index 0 is the residual 5th class (rarely activated on the 4-class test set).
CLASS_LABELS = [
    "Other / Non-MRI",  # 0
    "Meningioma",       # 1
    "No Tumor",         # 2
    "Glioma",           # 3
    "Pituitary Tumor",  # 4
]


def conv2d_same_3x3(x, kernel, bias):
    # x: (H, W, C_in), kernel: (3, 3, C_in, C_out), bias: (C_out,)
    H, W, C_in = x.shape
    C_out = kernel.shape[3]

    x_padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")
    slices = np.stack(
        [
            x_padded[0:H, 0:W, :],
            x_padded[0:H, 1 : W + 1, :],
            x_padded[0:H, 2 : W + 2, :],
            x_padded[1 : H + 1, 0:W, :],
            x_padded[1 : H + 1, 1 : W + 1, :],
            x_padded[1 : H + 1, 2 : W + 2, :],
            x_padded[2 : H + 2, 0:W, :],
            x_padded[2 : H + 2, 1 : W + 1, :],
            x_padded[2 : H + 2, 2 : W + 2, :],
        ],
        axis=0,
    )

    kernel_reshaped = kernel.reshape(9, C_in, C_out)
    out = np.tensordot(slices, kernel_reshaped, axes=((0, 3), (0, 1)))
    out += bias
    return out


def relu(x):
    return np.maximum(x, 0)


def maxpool2d_2x2(x):
    H, W, C = x.shape
    out = x.reshape(H // 2, 2, W // 2, 2, C)
    return out.max(axis=(1, 3))


def dense(x, kernel, bias, activation=None):
    out = np.dot(x, kernel) + bias
    if activation == "relu":
        out = relu(out)
    elif activation == "softmax":
        exp_x = np.exp(out - np.max(out))
        out = exp_x / np.sum(exp_x)
    return out


class NumPyVGG16Model:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.labels = list(CLASS_LABELS)
        self.weights = {}
        self.load_weights()

    def load_weights(self):
        print(f"Loading model weights from {self.h5_path}...")
        t0 = time.time()
        with h5py.File(self.h5_path, "r") as f:
            vgg16_grp = f["model_weights/vgg16"]
            for layer_name in vgg16_grp.keys():
                if "conv" in layer_name:
                    kernel = vgg16_grp[f"{layer_name}/kernel"][:]
                    bias = vgg16_grp[f"{layer_name}/bias"][:]
                    self.weights[layer_name] = (kernel, bias)

            self.weights["dense"] = (
                f["model_weights/dense/sequential/dense/kernel"][:],
                f["model_weights/dense/sequential/dense/bias"][:],
            )
            self.weights["dense_1"] = (
                f["model_weights/dense_1/sequential/dense_1/kernel"][:],
                f["model_weights/dense_1/sequential/dense_1/bias"][:],
            )
        print(f"Weights loaded successfully in {time.time() - t0:.2f} seconds.")

    def preprocess_image(self, image_path):
        # Match Keras training pipeline: RGB, 128x128, scale to [0, 1]
        img = Image.open(image_path).convert("RGB")
        img = img.resize((128, 128), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def predict(self, image_path):
        x = self.preprocess_image(image_path)

        t_start = time.time()
        x = relu(conv2d_same_3x3(x, *self.weights["block1_conv1"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block1_conv2"]))
        x = maxpool2d_2x2(x)

        x = relu(conv2d_same_3x3(x, *self.weights["block2_conv1"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block2_conv2"]))
        x = maxpool2d_2x2(x)

        x = relu(conv2d_same_3x3(x, *self.weights["block3_conv1"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block3_conv2"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block3_conv3"]))
        x = maxpool2d_2x2(x)

        x = relu(conv2d_same_3x3(x, *self.weights["block4_conv1"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block4_conv2"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block4_conv3"]))
        x = maxpool2d_2x2(x)

        x = relu(conv2d_same_3x3(x, *self.weights["block5_conv1"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block5_conv2"]))
        x = relu(conv2d_same_3x3(x, *self.weights["block5_conv3"]))
        x = maxpool2d_2x2(x)

        x_flat = x.flatten()
        x_dense = dense(x_flat, *self.weights["dense"], activation="relu")
        probabilities = dense(x_dense, *self.weights["dense_1"], activation="softmax")

        inference_time = time.time() - t_start
        probs = probabilities.astype(np.float64)
        probs = probs / probs.sum()  # numerical safety
        return probs.tolist(), inference_time

    def predict_detailed(self, image_path):
        probs, inference_time = self.predict(image_path)
        predicted_index = int(np.argmax(probs))
        predicted_label = self.labels[predicted_index]
        confidence = float(probs[predicted_index])
        return {
            "probabilities": probs,
            "labels": list(self.labels),
            "class_probabilities": {
                label: float(prob) for label, prob in zip(self.labels, probs)
            },
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "inference_time_ms": round(inference_time * 1000, 2),
        }
