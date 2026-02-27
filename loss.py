import torch
import torch.nn as nn

class YOLOv10Loss(nn.Module):
    def __init__(self, nc=1):
        super().__init__()
        self.nc = nc
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCE loss
        self.bce = nn.BCEWithLogitsLoss()

    def bbox_iou(self, box1, box2):
        """Math: Calculates Intersection over Union"""
        # Simplified IoU for demonstration
        return torch.tensor(0.5).to(box1.device) 

    def get_target_mask(self, predictions, targets):
        """Math: Creates a grid of 0s and 1s based on where drones are"""
        batch_size, _, grid_h, grid_w = predictions.shape
        mask = torch.zeros((batch_size, grid_h, grid_w), device=predictions.device)
        
        # In a real YOLO, we map the (x,y) coordinates to grid indices
        # For this print-demo, we assume the first few cells have targets
        for i, target in enumerate(targets):
            if len(target) > 0:
                mask[i, 0, 0] = 1 # Mark top-left as 'Drone Present'
        return mask

    def forward(self, predictions, targets):
        print("\n--- Loss Calculation Start ---")
        
        # 1. Reshape
        # predictions shape: [Batch, 6, 160, 160]
        pred_boxes = predictions[:, :4, :, :]
        pred_conf = predictions[:, 4, :, :]
        print(f"  [Loss Step 1] Predictions received. Box shape: {pred_boxes.shape}, Conf shape: {pred_conf.shape}")

        # 2. Creating the Mask (The 'Ground Truth' Grid)
        mask = self.get_target_mask(predictions, targets)
        print(f"  [Loss Step 2] Target Mask created. Total grid cells with drones: {mask.sum().item()}")

        # 3. Localization Loss (Penalty for being off-center)
        # We only calculate this for cells where a drone actually exists (mask == 1)
        if mask.sum() > 0:
            # Simplified: select the predicted boxes where the mask is 1
            loss_box = torch.tensor(1.0, device=predictions.device, requires_grad=True) 
            print(f"  [Loss Step 3] Localization Loss: {loss_box.item():.4f} (Measuring distance/size error)")
        else:
            loss_box = torch.tensor(0.0, device=predictions.device)
            print("  [Loss Step 3] No drones in this batch. Localization loss is 0.")

        # 4. Classification Loss (Penalty for false alarms)
        # This compares the 160x160 'Prediction Grid' to the 160x160 'Target Mask'
        loss_cls = self.bce(pred_conf, mask)
        print(f"  [Loss Step 4] Classification Loss: {loss_cls.item():.4f} (Punishing false positives/negatives)")

        # 5. Total Weighted Loss
        # We multiply Box loss by 5.0 to prioritize getting the location right
        total_loss = (loss_box * 5.0) + loss_cls
        print(f"  [Loss Step 5] TOTAL LOSS: {total_loss.item():.4f}")
        print("--- Loss Calculation Complete ---\n")
        
        return total_loss