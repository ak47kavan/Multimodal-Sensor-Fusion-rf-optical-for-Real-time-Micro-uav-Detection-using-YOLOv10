# Micro-UAV Detection using Custom YOLOv10 from Scratch

### 🛸 Project Overview
This repository contains a specialized implementation of **YOLOv10** built specifically for the detection of **Micro-UAVs (Drones)**. Developed as a final-year project, this model focuses on high-resolution spatial feature extraction and integrated attention mechanisms to identify small, fast-moving objects against complex sky and urban backgrounds.

### 🛠️ Key Features
* **Custom Architecture:** Built entirely in PyTorch with specialized `SCDown` (Spatial Context Downsampling) and `PSA` (Partial Self-Attention) layers.
* **Mathematical Transparency:** Includes specialized diagnostic tools to trace tensors from raw pixel integers (0-255) to final detection probabilities (0-1).
* **Defense & EW Ready:** Optimized for integration into Electronic Warfare systems, focusing on low-latency optical verification.

---

### 🧠 Architecture: The Mathematical Flow
The model follows a rigorous pipeline to ensure tiny drones aren't "lost" during downsampling:

1.  **Backbone:** * **ConvBlocks:** Initial texture and edge detection.
    * **SCDown Layers:** Advanced downsampling that preserves depthwise information while reducing resolution from **640x640** down to **160x160**.
2.  **Neck (Attention):** * **PSA (Partial Self-Attention):** Uses $Attention(Q,K,V) = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V$ to weight drone pixels higher than background "noise" like clouds or birds.
3.  **Head:** * **Detection Head:** A 1x1 Convolutional layer predicting 6 values per grid cell: $[x, y, w, h, \text{confidence}, \text{class}]$.

---

### 📉 Mathematical Trace (Data Transformation)

| Stage | Input Shape | Output Shape | Operation |
| :--- | :--- | :--- | :--- |
| **Input** | `(640, 640, 3)` | `(3, 640, 640)` | Normalization ($x/255$) |
| **Backbone L1** | `(3, 640, 640)` | `(16, 640, 640)` | Feature Extraction |
| **Spatial Down** | `(16, 640, 640)` | `(32, 320, 320)` | Stride-2 Convolution |
| **Attention** | `(256, 160, 160)` | `(256, 160, 160)` | PSA Self-Attention |
| **Detection** | `(256, 160, 160)` | `(6, 160, 160)` | Sigmoid Confidence |

---

### 🚀 Getting Started

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/micro-uav-detection.git](https://github.com/yourusername/micro-uav-detection.git)
cd micro-uav-detection

2. Install Dependencies

Bash
pip install torch torchvision opencv-python matplotlib numpy
3. Run Mathematical Trace
Visualize exactly how the model "thinks" by running the trace script:

Bash
python trace_math.py
4. Inference
Run detection on a local image:

Bash
python inference.py
🧪 Performance Details
Detection Grid: 160x160 (Providing 25,600 individual decision points).

Activation Function: SiLU (Sigmoid Linear Unit) for robust gradient flow during backpropagation.

Loss Function: Weighted CIoU (Complete Intersection over Union) for high-precision bounding box regression.

👨‍💻 Author
AK Kavan
Final Year B.Tech IT Student | Srinivas University
Specializing in ML/DL, Radar Signal Processing, and Space Technology.
