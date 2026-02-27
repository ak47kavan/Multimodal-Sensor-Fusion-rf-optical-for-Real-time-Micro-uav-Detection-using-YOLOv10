from ultralytics import YOLOv10

def main():
    # 1. Load the YOLOv10n pre-trained model
    model = YOLOv10("yolov10n.pt") 

    # 2. Start training
    # Note: imgsz=640 is standard; decrease to 320 for even faster real-time speed
    model.train(
        data="data.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=16,     # Adjust based on your GPU memory
        device=0      # Use 0 for GPU, 'cpu' if no GPU available
    )

if __name__ == "__main__":
    main()