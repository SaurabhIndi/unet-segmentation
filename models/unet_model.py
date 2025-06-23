import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    # Changed parameters: 
    # in_channels_from_prev_decoder: Channels from the previous layer in the expanding path.
    # skip_channels: Channels from the corresponding skip connection in the contracting path.
    # out_channels: Desired output channels for this upsampling block.
    def __init__(self, in_channels_from_prev_decoder, skip_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # The DoubleConv's input channels will be the sum of upsampled channels and skip channels
            self.conv = DoubleConv(in_channels_from_prev_decoder + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_from_prev_decoder, in_channels_from_prev_decoder // 2, kernel_size=2, stride=2)
            # If using ConvTranspose, the upsampled output has half channels, then concatenated with skip_channels
            self.conv = DoubleConv((in_channels_from_prev_decoder // 2) + skip_channels, out_channels)

    def forward(self, x1, x2): # x1 from previous decoder stage, x2 is skip connection
        x1 = self.up(x1)
        
        # input is CHW - handle size mismatch
        # This padding ensures that dimensions match for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1) # Concatenate along the channel dimension
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Up(in_channels_from_prev_decoder, skip_channels, out_channels, bilinear)
        # up1: Input from down4 (x5, 1024 ch), skip from down3 (x4, 512 ch). Output 512 ch.
        self.up1 = Up(1024, 512, 512, bilinear) 
        
        # up2: Input from up1 (512 ch), skip from down2 (x3, 256 ch). Output 256 ch.
        self.up2 = Up(512, 256, 256, bilinear)
        
        # up3: Input from up2 (256 ch), skip from down1 (x2, 128 ch). Output 128 ch.
        self.up3 = Up(256, 128, 128, bilinear)
        
        # up4: Input from up3 (128 ch), skip from inc (x1, 64 ch). Output 64 ch.
        self.up4 = Up(128, 64, 64, bilinear)
        
        self.outc = OutConv(64, n_classes) # Output from up4 is 64 channels

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Pass the contracting path output (skip connection) to the upsampling layers
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor (batch_size, channels, height, width)
    # For grayscale images, channels = 1. For RGB, channels = 3.
    # U-Net paper uses 512x512 images.
    input_tensor = torch.randn(1, 1, 512, 512) 
    
    # Initialize U-Net model for grayscale input (1 channel) and binary segmentation (1 class output, e.g., cell vs background)
    model = UNet(n_channels=1, n_classes=1) 
    
    # Pass input through the model
    output = model(input_tensor)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output tensor shape: {output.shape}")

    # Verify that the output shape matches the input shape for segmentation
    assert output.shape == input_tensor.shape, "Output shape does not match input shape!"
    print("U-Net model created successfully and output shape matches input shape.")