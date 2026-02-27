import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from yolo_scratch import YOLOv10 
from loss import YOLOv10Loss 
from data_loader import UAVDataset

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    print(f"\n--- Starting Epoch {epoch+1} ---")
    
    for batch_idx, batch in enumerate(dataloader):
        # 1. Unpacking the Batch
        images, targets = batch 
        print(f"[Batch {batch_idx}] Loaded {len(images)} images from disk.")
        
        # 2. Mathematical Stacking
        # We turn a list of images into a single 4D block (Tensor)
        images = torch.stack(images).to(device)
        print(f"  -> Images stacked into Tensor shape: {images.shape} (Batch, Channel, H, W)")
        
        # 3. Moving Targets to Device
        targets = [t.to(device) for t in targets]
        print(f"  -> Moved {len(targets)} target labels to {device}.")
        
        # 4. Forward Pass (The 'Prediction' Step)
        # The image enters the Brain and comes out as a grid of numbers
        preds = model(images)
        print(f"  -> Forward Pass complete. Prediction grid shape: {preds.shape}")
        
        # 5. Loss Calculation (The 'Penalty' Step)
        # We compare predictions to targets using CIoU math
        loss = criterion(preds, targets)
        print(f"  -> Loss calculated: {loss.item():.4f} (Lower is better)")
        
        # 6. Backward Pass (The 'Learning' Step)
        optimizer.zero_grad() # Clear old gradients
        loss.backward()      # Math: Calculate how to adjust weights (Calculus)
        print("  -> Backward pass: Computed gradients for all neurons.")
        
        # 7. Optimizer Step (The 'Adjustment' Step)
        optimizer.step()     # Math: Update weights using w = w - lr * grad
        print("  -> Optimizer: Updated model weights.")
        
        total_loss += loss.item()
        
        # We only print the full detail for the first batch to avoid flooding the screen
        if batch_idx == 0:
            print("  ... Continuing rest of batches for this epoch ...")
            
    return total_loss / len(dataloader)

def main():
    print("Setting up environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    train_images = r"C:\Users\kavan\OneDrive\Documents\drone_detection\datasets\train\images"
    train_labels = r"C:\Users\kavan\OneDrive\Documents\drone_detection\datasets\train\labels"
    
    # Initialize Brain and Penalty logic
    print("Initializing YOLOv10 Brain and Loss function...")
    model = YOLOv10(nc=1).to(device)
    criterion = YOLOv10Loss(nc=1)
    
    # Dataset setup
    print("Loading Dataset...")
    train_dataset = UAVDataset(train_images, train_labels)
    # The lambda below is the 'glue' that bundles different numbers of drones together
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    print(f"Dataset ready with {len(train_dataset)} images.")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Main Training Loop
    for epoch in range(100):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"--- Epoch {epoch+1} Results: Average Loss = {avg_loss:.4f} ---")

    # Final Save
    print("Training complete! Saving weights to 'micro_uav_v10_scratch.pt'...")
    torch.save(model.state_dict(), "micro_uav_v10_scratch.pt")
    print("Done.")

if __name__ == "__main__":
    main()