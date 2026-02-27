import torch
import torch.nn as nn
import cv2
import numpy as np

def trace_math(image_path, model_path):
    device = torch.device("cpu") 
    
    # 1. LOAD THE BRAIN
    from yolo_scratch import YOLOv10
    model = YOLOv10(nc=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. STEP 1: RAW PIXEL DATA
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_640 = cv2.resize(img_rgb, (640, 640))
    
    print(f"--- [STEP 1: RAW IMAGE] ---")
    print(f"Original Pixel (at center 320, 320): {img_640[320, 320]}") # Shows [R, G, B] integers
    
    x = torch.from_numpy(img_640).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    print(f"Normalized Tensor (at 320, 320): {x[0, :, 320, 320].numpy()}") # Shows 0.0-1.0 floats

    # 3. STEP 2: INITIAL FEATURES (640x640)
    with torch.no_grad():
        x1 = model.layer1(x)
    print(f"\n--- [STEP 2: BACKBONE L1 - EDGE DETECTION] ---")
    print(f"Shape: {x1.shape} (Resolution still 640)")
    # Printing 4 features out of 16 for one specific pixel
    print(f"First 4 Feature 'Thoughts' for pixel (320,320): {x1[0, :4, 320, 320].numpy()}")

    # 4. STEP 3: AFTER FIRST SHRINK (320x320)
    with torch.no_grad():
        x2 = model.layer2(x1)
    print(f"\n--- [STEP 3: BACKBONE L2 - SPATIAL REDUCTION] ---")
    print(f"Shape: {x2.shape} (Resolution cut to 320)")
    print(f"Math: Pixel (320,320) is now represented at grid (160,160)")

    # 5. STEP 4: FINAL FEATURES (160x160)
    with torch.no_grad():
        x5 = model.layer5(model.layer4(model.layer3(x2)))
    print(f"\n--- [STEP 4: BACKBONE L5 - DEEP KNOWLEDGE] ---")
    print(f"Shape: {x5.shape} (Resolution cut to 160)")
    # Checking the specific cell from your label (48% of 160 = 77, 56% of 160 = 89)
    print(f"Deep Feature Values at Drone Location (77, 89): {x5[0, :5, 89, 77].numpy()}...")

    # 6. STEP 5: ATTENTION EFFECT
    with torch.no_grad():
        attn_out = model.neck(x5)
    print(f"\n--- [STEP 5: PSA ATTENTION] ---")
    val_before = x5[0, 0, 89, 77].item()
    val_after = attn_out[0, 0, 89, 77].item()
    print(f"Math: Attention changed value at drone cell from {val_before:.4f} to {val_after:.4f}")

    # 7. STEP 6: RAW DETECTION HEAD
    with torch.no_grad():
        raw_output = model.head(attn_out)
    print(f"\n--- [STEP 6: DETECTION HEAD - THE 6 PREDICTIONS] ---")
    # For grid (77, 89), show the 6 values: [x, y, w, h, conf, class]
    prediction_vector = raw_output[0, :, 89, 77].numpy()
    print(f"Raw Output at Drone Cell: {prediction_vector}")

    # 8. STEP 7: FINAL PROBABILITY (SIGMOID)
    probs = torch.sigmoid(raw_output)
    conf_score = probs[0, 4, 89, 77].item()
    
    print(f"\n--- [STEP 7: FINAL CALCULATION] ---")
    print(f"Label said: Drone is here.")
    print(f"Model Raw Confidence: {prediction_vector[4]:.4f}")
    print(f"Sigmoid Math: 1 / (1 + e^-({prediction_vector[4]:.4f}))")
    print(f"RESULT: {conf_score*100:.2f}% Probability of Drone at this exact label coordinate.")

if __name__ == "__main__":
    # Using your specific training image
    path = r"C:\Users\kavan\OneDrive\Documents\drone_detection\datasets\train\images\1-276-_jpeg_jpg.rf.96eeae2e99bfb328b5172b9132c3e5ea.jpg"
    trace_math(path, "micro_uav_v10_scratch.pt")