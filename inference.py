import torch
import cv2
import numpy as np
from yolo_scratch import YOLOv10 # Import your custom brain

def run_local_inference(image_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Brain
    model = YOLOv10(nc=1).to(device)
    
    # 2. Load the trained math (Weights)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Important: Tells the model we are TESTING, not training
    print("Model loaded successfully!")

    # 3. Prepare your Image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 640x640 because that's what the math expects
    img_input = cv2.resize(img_rgb, (640, 640))
    # Math: (H, W, C) -> (C, H, W) and normalize 0-1
    tensor = torch.from_numpy(img_input).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # 4. Predict
    with torch.no_grad():
        output = model(tensor) # Result shape: [1, 6, 160, 160]

    # 5. Interpret the Output
   # 5. Interpret the Output
    # We apply torch.sigmoid to turn raw numbers into 0-1 probabilities
    output_probs = torch.sigmoid(output)
    
    conf_grid = output_probs[0, 4, :, :] # index 4 is our confidence
    max_val, max_idx = torch.max(conf_grid.view(-1), 0)
    
    # Math: Convert flattened index back to 2D grid coordinates
    grid_y, grid_x = divmod(max_idx.item(), 160)
    
    # Multiply by 100 to show as a percentage
    print(f"Detection confidence: {max_val.item()*100:.2f}% at Grid Cell: ({grid_x}, {grid_y})")

    if max_val > 0.4: # Threshold of 40%
        print("🎯 Micro-UAV Detected!")
        # Math: Map the 160x160 grid back to the 640x640 image size
        # Each grid cell represents a 4x4 pixel area (640 / 160 = 4)
        cv2.circle(img_input, (grid_x * 4, grid_y * 4), 20, (0, 255, 0), 2)

if __name__ == "__main__":
    # Update these with your local paths
    run_local_inference("C:\\Users\\kavan\\OneDrive\\Documents\\drone_detection\\datasets\\train\\images\\1-276-_jpeg_jpg.rf.96eeae2e99bfb328b5172b9132c3e5ea.jpg", "micro_uav_v10_scratch.pt")