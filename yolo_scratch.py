import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        # Math: Standard Convolution + Normalization
        return self.act(self.bn(self.conv(x)))
    
class SCDown(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=2):
        super().__init__()
        self.cv1 = ConvBlock(in_c, out_c, k=1, s=1, p=0)
        self.cv2 = nn.Conv2d(out_c, out_c, kernel_size=k, stride=s, padding=k//2, groups=out_c)

    def forward(self, x):
        # Math: Spatial Reduction. Stride=2 cuts H and W in half.
        return self.cv2(self.cv1(x))
    
class PSA(nn.Module):
    def __init__(self, c, nh=8):
        super().__init__()
        # Simplified for structure, but adding a print to track the Neck
        pass

    def forward(self, x):
        print(f"  [Neck: PSA] Applying attention to features of shape: {x.shape}")
        return x 

class DetectionHead(nn.Module):
    def __init__(self, nc=1):
        super().__init__()
        # Math: 1x1 Conv to compress 256 channels into 6 prediction values
        self.head = nn.Conv2d(256, (4 + 1 + nc), kernel_size=1)

    def forward(self, x):
        out = self.head(x)
        print(f"  [Head: Detection] Final prediction tensor shape: {out.shape}")
        return out
    
class YOLOv10(nn.Module):
    def __init__(self, nc=1):
        super().__init__()
        # Defining the specific layers so we can print them individually
        self.layer1 = ConvBlock(3, 16, k=3, s=1)
        self.layer2 = SCDown(16, 32, k=3, s=2)
        self.layer3 = ConvBlock(32, 64, k=3, s=1)
        self.layer4 = SCDown(64, 128, k=3, s=2)
        self.layer5 = ConvBlock(128, 256, k=3, s=1)
        
        self.neck = PSA(256)
        self.head = DetectionHead(nc=nc)

    def forward(self, x):
        print(f"\n--- Brain Processing Start (Input: {x.shape}) ---")
        
        # Pass through Backbone
        x = self.layer1(x)
        print(f"  [Backbone L1] After Initial Conv: {x.shape}")
        
        x = self.layer2(x)
        print(f"  [Backbone L2] After SCDown (640->320): {x.shape}")
        
        x = self.layer3(x)
        print(f"  [Backbone L3] Features deepened to 64 channels: {x.shape}")
        
        x = self.layer4(x)
        print(f"  [Backbone L4] After SCDown (320->160): {x.shape}")
        
        x = self.layer5(x)
        print(f"  [Backbone L5] Final Backbone features: {x.shape}")
        
        # Pass through Neck and Head
        x = self.neck(x)
        x = self.head(x)
        
        print(f"--- Brain Processing Complete ---\n")
        return x