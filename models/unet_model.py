import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2 with UNPADDED convolutions"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Changed padding to 0 for unpadded convolutions
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Changed padding to 0 for unpadded convolutions
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0), 
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
    """Upscaling then double conv, with cropping of skip connection"""
    def __init__(self, in_channels_from_prev_decoder, skip_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Input channels for conv will be sum of upsampled channels and cropped skip channels
            self.conv = DoubleConv(in_channels_from_prev_decoder + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_from_prev_decoder, in_channels_from_prev_decoder // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((in_channels_from_prev_decoder // 2) + skip_channels, out_channels)

    # Note: The cropping logic will be handled outside in the UNet forward pass
    # or passed via a helper. We'll add it in the UNet class for simplicity.
    def forward(self, x1, x2_cropped): # x1 from previous decoder stage, x2_cropped is already cropped skip connection
        x1 = self.up(x1)
        # No padding needed here; x2_cropped already matches x1's size
        x = torch.cat([x2_cropped, x1], dim=1) # Concatenate along the channel dimension
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False): # Changed default to False
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512, 512, bilinear) 
        self.up2 = Up(512, 256, 256, bilinear)
        self.up3 = Up(256, 128, 128, bilinear)
        self.up4 = Up(128, 64, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)

    # Helper function for center cropping
    def _center_crop(self, feature_map, target_size):
        """
        Crops the center of a feature map to a target size.
        Assumes feature_map is (N, C, H, W) and target_size is (H_target, W_target).
        """
        _, _, h, w = feature_map.size()
        th, tw = target_size
        
        # Calculate start and end indices for cropping
        h_start = max(0, (h - th) // 2)
        h_end = h_start + th
        w_start = max(0, (w - tw) // 2)
        w_end = w_start + tw
        
        return feature_map[:, :, h_start:h_end, w_start:w_end]


    def forward(self, x):
        x1 = self.inc(x)     # Output size: (H-4, W-4) for 2x3x3 convs
        x2 = self.down1(x1)  # MaxPool (H-4)/2, (W-4)/2; then 2x3x3 convs. Output size: ((H-4)/2 - 4, (W-4)/2 - 4)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Smallest feature map

        # Now for the expansive path, we crop the skip connections (x1, x2, x3, x4)
        # to match the size of the upsampled feature map before concatenation.
        # This requires knowing the exact size of the upsampled feature map (x_up)

        # Example for x_up from x5:
        # x_up_from_x5 = self.up1.up(x5) # This will be upsampled from x5.size()
        # We need to crop x4 to match x_up_from_x5.size()[2:]

        # Let's trace dimensions and add cropping
        # Initial input size: (H, W)
        # x1_size: (H-4, W-4)
        # x2_size: ( (H-4)/2 - 4, (W-4)/2 - 4 )
        # x3_size: ( ( (H-4)/2 - 4 )/2 - 4, ... )
        # x4_size: ( ( ( (H-4)/2 - 4 )/2 - 4 )/2 - 4, ... )
        # x5_size: ( ( ( ( (H-4)/2 - 4 )/2 - 4 )/2 - 4 )/2 - 4, ... )

        # Pass x5 (bottleneck) to first Up block
        x_up = self.up1.up(x5) # x_up is the upsampled feature map from x5
        x4_cropped = self._center_crop(x4, x_up.size()[2:]) # Crop x4 to match x_up
        x = self.up1.conv(torch.cat([x4_cropped, x_up], dim=1)) # Concatenate and convolve

        x_up = self.up2.up(x) # x_up is the upsampled feature map from previous Up block output
        x3_cropped = self._center_crop(x3, x_up.size()[2:])
        x = self.up2.conv(torch.cat([x3_cropped, x_up], dim=1))

        x_up = self.up3.up(x)
        x2_cropped = self._center_crop(x2, x_up.size()[2:])
        x = self.up3.conv(torch.cat([x2_cropped, x_up], dim=1))

        x_up = self.up4.up(x)
        x1_cropped = self._center_crop(x1, x_up.size()[2:])
        x = self.up4.conv(torch.cat([x1_cropped, x_up], dim=1))

        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    # Example usage:
    # U-Net paper uses 512x512 images.
    input_tensor = torch.randn(1, 1, 572, 572) # For a 572x572 input, output is 388x388 as per original paper.
                                                # Or, for 512x512, let's verify.
    
    # Trace for a 512x512 input:
    # Input: 512x512
    # Inc (2x3x3 conv): 512 - 2*2 = 508 (after 2 convs) -> x1 (508x508)
    # Down1: MaxPool (508/2 = 254); 2x3x3 conv: 254 - 4 = 250 -> x2 (250x250)
    # Down2: MaxPool (250/2 = 125); 2x3x3 conv: 125 - 4 = 121 -> x3 (121x121)
    # Down3: MaxPool (121/2 = 60.5 -> 60); 2x3x3 conv: 60 - 4 = 56 -> x4 (56x56)
    # Down4: MaxPool (56/2 = 28); 2x3x3 conv: 28 - 4 = 24 -> x5 (24x24) -- Bottleneck

    # Expansive Path
    # Up1: Upsample x5 (24 -> 48); 2x3x3 conv. Needs to crop x4 (56x56) to 48x48.
    #      Output after 2x3x3 conv: 48-4 = 44 -> (44x44)
    # Up2: Upsample (44 -> 88); 2x3x3 conv. Needs to crop x3 (121x121) to 88x88.
    #      Output after 2x3x3 conv: 88-4 = 84 -> (84x84)
    # Up3: Upsample (84 -> 168); 2x3x3 conv. Needs to crop x2 (250x250) to 168x168.
    #      Output after 2x3x3 conv: 168-4 = 164 -> (164x164)
    # Up4: Upsample (164 -> 328); 2x3x3 conv. Needs to crop x1 (508x508) to 328x328.
    #      Output after 2x3x3 conv: 328-4 = 324 -> (324x324)
    # OutConv: 1x1 conv on 324x324 -> final output is 324x324

    # The original U-Net paper used 572x572 input to get a 388x388 output.
    # Let's verify this example:
    # Input: 572x572
    # x1: 572 - 4 = 568
    # x2: (568/2) - 4 = 284 - 4 = 280
    # x3: (280/2) - 4 = 140 - 4 = 136
    # x4: (136/2) - 4 = 68 - 4 = 64
    # x5: (64/2) - 4 = 32 - 4 = 28

    # Expansive Path (target size for cropping in parentheses)
    # x_up_from_x5 = 28 * 2 = 56. Crop x4 (64x64) to (56x56). Output after conv: 56 - 4 = 52
    # x_up = 52 * 2 = 104. Crop x3 (136x136) to (104x104). Output after conv: 104 - 4 = 100
    # x_up = 100 * 2 = 200. Crop x2 (280x280) to (200x200). Output after conv: 200 - 4 = 196
    # x_up = 196 * 2 = 392. Crop x1 (568x568) to (392x392). Output after conv: 392 - 4 = 388
    # Final output is 388x388. This matches the paper for a 572x572 input!

    # So, for an input size of (H, W), the output size will be (H - 184, W - 184) for 5 levels of down/up.
    # This is (572 - 184) = 388.

    input_height = 512 # Let's stick with 512 if that's your typical input
    input_width = 512
    # Calculate expected output size for 512x512
    # 512 -> x1 (508)
    # 508 -> x2 (250)
    # 250 -> x3 (121)
    # 121 -> x4 (56)
    # 56 -> x5 (24)
    # Up1 target: 24*2 = 48. Crop x4 (56) to 48. Up1 conv output: 48-4 = 44
    # Up2 target: 44*2 = 88. Crop x3 (121) to 88. Up2 conv output: 88-4 = 84
    # Up3 target: 84*2 = 168. Crop x2 (250) to 168. Up3 conv output: 168-4 = 164
    # Up4 target: 164*2 = 328. Crop x1 (508) to 328. Up4 conv output: 328-4 = 324
    # Final output size for 512x512 input: 324x324

    input_tensor = torch.randn(1, 1, input_height, input_width) 
    
    model = UNet(n_channels=1, n_classes=1) 
    
    output = model(input_tensor)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output tensor shape: {output.shape}")

    expected_output_height = 324
    expected_output_width = 324
    assert output.shape[2] == expected_output_height and output.shape[3] == expected_output_width, \
        f"Output shape ({output.shape[2]}x{output.shape[3]}) does not match expected ({expected_output_height}x{expected_output_width})!"
    print("U-Net model created successfully and output shape matches expected size for unpadded convolutions.")