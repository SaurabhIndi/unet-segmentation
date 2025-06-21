import torch
import torch.nn as nn

# Helper function for the standard Conv-ReLU block
class DoubleConv(nn.Module):
    """
    Applies two 3x3 convolutions, each followed by ReLU.
    This is a basic building block for both the contracting and expansive paths.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0), # "unpadded convolutions" 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0), # "unpadded convolutions" 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Downsampling block (for the contracting path)
class Down(nn.Module):
    """
    Performs Max Pooling followed by a DoubleConv block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2), # "2x2 max pooling operation with stride 2 for downsampling" 
            DoubleConv(in_channels, out_channels) # "double the number of feature channels" 
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling block (for the expansive path)
class Up(nn.Module):
    """
    Performs up-convolution, concatenates with cropped feature map,
    and then applies a DoubleConv block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Up-convolution (Transposed Convolution) to increase resolution
        # and halve the number of feature channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) #[cite: 55]
        self.conv = DoubleConv(in_channels, out_channels) # The total in_channels here will be (in_channels_after_up + skip_connection_channels)

    def forward(self, x1, x2):
        # x1 is the input from the previous upsampling layer (lower resolution)
        # x2 is the corresponding feature map from the contracting path (higher resolution)

        x1 = self.up(x1)

        # Handle cropping for the skip connection
        # The paper states "cropping is necessary due to the loss of border pixels in every convolution." 
        # This means the feature map from the contracting path (x2) might be larger than x1 after upsampling.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        # Concatenate x1 (upsampled) and x2 (from contracting path)
        x = torch.cat([x2, x1], dim=1) #[cite: 55]
        return self.conv(x)

# The U-Net main class
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Initial convolution (first part of contracting path)
        self.inc = DoubleConv(n_channels, 64) # First layer typically 64 channels (refer to Figure 1)

        # Contracting Path (Downsampling)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 is the bottleneck, often with Dropout 
        self.down4 = Down(512, 1024)

        # Expansive Path (Upsampling)
        # Note: The 'in_channels' for Up blocks are (channels_from_down_path + channels_from_up_path)
        # So, the first Up block takes 1024 (from bottleneck) + 512 (skip) = 1536 as in_channels for the DoubleConv,
        # but the ConvTranspose2d halves the 1024 to 512.
        # Thus, the DoubleConv will take 512 (from up-conv) + 512 (skip) = 1024 as its input.
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final 1x1 convolution to map to desired number of classes
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) #[cite: 57]

    def forward(self, x):
        # Contracting Path
        x1 = self.inc(x) # Output: 64 channels
        x2 = self.down1(x1) # Output: 128 channels
        x3 = self.down2(x2) # Output: 256 channels
        x4 = self.down3(x3) # Output: 512 channels
        x5 = self.down4(x4) # Output: 1024 channels (bottleneck)

        # Expansive Path (with skip connections)
        x = self.up1(x5, x4) # x5 is upsampled, concatenated with x4
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output convolution
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    # Simple test to check the U-Net architecture
    # Input a dummy image (e.g., 1 batch, 3 channels, 572x572 as per Figure 1)
    # The output size will be smaller due to unpadded convolutions.
    # The paper uses 572x572 input for a 388x388 output 
    input_channels = 3 # Example: RGB image
    output_classes = 2 # Example: Background and Foreground (or Membranes and Cells)

    model = UNet(n_channels=input_channels, n_classes=output_classes)
    print(model)

    # Create a dummy input tensor matching the expected input size (e.g., 572x572)
    dummy_input = torch.randn(1, input_channels, 572, 572)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Expected output shape for 572x572 input and 23 convolutional layers, each with kernel 3 and no padding,
    # resulting in a reduction of 2 pixels per convolution (1 pixel on each side).
    # Total reduction: 23 * 2 = 46 pixels on each side.
    # 572 - 46 = 526 pixels.
    # Oh, wait. The paper says 572x572 input gives 388x388 output. 
    # Let's re-check the padding. "unpadded convolutions"  means padding=0.
    # 3x3 conv with padding 0 reduces size by 2.
    # Max pooling by 2 reduces size by half.
    # Let's trace it:
    # 572
    # Inc: conv(3x3) -> 570; conv(3x3) -> 568
    # Down1: maxpool(2x2) -> 284; conv(3x3) -> 282; conv(3x3) -> 280
    # Down2: maxpool(2x2) -> 140; conv(3x3) -> 138; conv(3x3) -> 136
    # Down3: maxpool(2x2) -> 68; conv(3x3) -> 66; conv(3x3) -> 64
    # Down4: maxpool(2x2) -> 32; conv(3x3) -> 30; conv(3x3) -> 28 (bottleneck)

    # Up1: upconv(2x2) -> 56;
    # The x4 shape for skip connection after down3 is 64x64 (from image tile 572x572, it has gone through 3 downsampling and 3 DoubleConv blocks)
    # The paper shows: Input 572x572. After Inc it's 568x568. After Down1 it's 280x280. After Down2 it's 136x136. After Down3 it's 64x64. After Down4 it's 28x28.
    # So the skip connection comes from the DoubleConv *before* the max pooling.
    # Let's update the forward pass for the UNet based on the diagram and text.
    # The text says "a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels." 
    # And "a concatenation with the correspondingly cropped feature map from the contracting path" 

    # Re-evaluating the size calculations based on Figure 1:
    # Input: 572x572
    # After inc (2x 3x3 convs): 568x568. This is x1 for skip.
    # After Down1 (maxpool -> 2x 3x3 convs): 284 -> 280x280. This is x2 for skip.
    # After Down2 (maxpool -> 2x 3x3 convs): 140 -> 136x136. This is x3 for skip.
    # After Down3 (maxpool -> 2x 3x3 convs): 68 -> 64x64. This is x4 for skip.
    # After Down4 (maxpool -> 2x 3x3 convs): 32 -> 28x28. This is x5 (bottleneck).

    # Expansive path:
    # Up1: upconv from 28x28 -> 56x56. Concatenate with x4 (64x64).
    # This requires cropping x4 (64x64) to 56x56.
    # Then 2x 3x3 convs on 56x56 -> 52x52.
    # This is not consistent with Figure 1 where it shows the sizes are preserved during the convs in the expansive path?
    # No, the blue boxes' lower left corner shows size, e.g., 512x512 after first up-conv in the diagram.
    # The blue boxes in the expansive path show dimensions like 512x512, 256x256 etc.
    # This implies that the convolutions in the expansive path *might* use padding, or the diagram's sizes are after concatenation and the following convs.
    # The paper explicitly states "only uses the valid part of each convolution", which implies unpadded.
    # And "cropping is necessary due to the loss of border pixels in every convolution".
    # This confirms unpadded convolutions throughout.

    # Let's re-re-trace based on the diagram sizes (which must be after the double convs in each block):
    # Input 572x572 -> `inc` output 568x568 (x1)
    # 568 -> `down1` output 280x280 (x2) (after maxpool and 2 convs)
    # 280 -> `down2` output 136x136 (x3)
    # 136 -> `down3` output 64x64 (x4)
    # 64 -> `down4` output 28x28 (x5 - bottleneck)

    # Now Up Path (diagram sizes are for input to the double conv in each Up block):
    # Up1: upconv on x5 (28x28) produces 56x56. This is concatenated with cropped x4 (64x64 cropped to 56x56).
    # The result (56x56) goes into the DoubleConv in Up1.
    # The output of DoubleConv (Up1) should be 52x52 (56-2-2).
    # BUT the diagram shows 256x256 as the output of the first upsampling block's double conv.
    # This is a major discrepancy between the text description and the diagram's dimensions if padding=0 for all convs.

    # Let's assume the diagram's reported sizes (e.g., 512x512, 256x256 on the right side) are the sizes *after* the DoubleConv within the Up module.
    # For this to happen with unpadded convs (kernel 3, padding 0), the input to the DoubleConv would need to be 4 pixels larger.
    # E.g., for 512x512 output, the input to DoubleConv must be 516x516.
    # For 256x256 output, input must be 260x260.
    # For 128x128 output, input must be 132x132.
    # For 64x64 output, input must be 68x68.
    # For the final output (segmentation map) to be 388x388, its input (last DoubleConv output) must be 392x392.
    # This implies that `nn.ConvTranspose2d` in `Up` might be upsampling to a specific size, or the cropping is adjusted for this.

    # Let's stick to the textual description for now: "unpadded convolutions"  and "cropping is necessary".
    # This means the output size will always be smaller.
    # The example output in the paper (388x388 for 512x512 input) refers to the "valid part"  of the segmentation.
    # If a 572x572 input yields 388x388 output, let's calculate the total loss: 572 - 388 = 184 pixels.
    # There are 4 `DoubleConv` blocks in the contracting path (8 conv layers total) and 4 `DoubleConv` blocks in the expansive path (8 conv layers total), plus the final 1x1 conv.
    # So 16 `3x3` convs + 1 `1x1` conv. Each `3x3` conv with padding=0 reduces by 2 pixels.
    # Total reduction: 16 * 2 = 32 pixels. This doesn't match 184.

    # The discrepancy in output sizes between text/figure and direct calculation with padding=0 is common.
    # Often, authors use "unpadded" to mean they don't apply `padding='same'`.
    # Let's re-read the exact text: "only uses the valid part of each convolution, i.e., the segmentation map only contains the pixels, for which the full context is available in the input image." 
    # This strongly suggests `padding=0`.

    # Let's consider the diagram's *output* size for 572x572 input is 388x388.
    # This means the total size reduction is 572 - 388 = 184.
    # Each 3x3 convolution with `padding=0` reduces the spatial dimension by 2 pixels.
    # The U-Net has 23 convolutional layers in total.
    # So, 23 layers * 2 pixels/layer = 46 pixels reduction. This is for the final layer.
    # The intermediate reduction due to max-pooling must be accounted for.

    # Let's trace the dimensions given in Figure 1 of the paper *exactly*:
    # Input: 572x572
    # Block 1 (2 convs): 568x568 (x1)
    # Pool: 284x284
    # Block 2 (2 convs): 280x280 (x2)
    # Pool: 140x140
    # Block 3 (2 convs): 136x136 (x3)
    # Pool: 68x68
    # Block 4 (2 convs): 64x64 (x4)
    # Pool: 32x32
    # Block 5 (2 convs, bottleneck): 28x28 (x5)

    # Expansive Path:
    # Up-conv (from x5=28x28): up to 56x56
    # Skip x4: 64x64. Cropped to 56x56.
    # Concatenate -> 56x56 input to `DoubleConv`
    # `DoubleConv` (2 convs): 56x56 -> 52x52.
    # This is where my code's `Up` block would produce 52x52.

    # The diagram shows 512x512, 256x256 etc. for the blue boxes in the expansive path.
    # And the final output is 388x388.
    # The ONLY way this works with `padding=0` is if the input image is effectively larger than 572x572 for the overall network to maintain these internal sizes after convolutions.
    # OR, the `ConvTranspose2d` targets a *specific* output size to align for concatenation without further cropping.
    # For example, `nn.ConvTranspose2d` can take `output_size` argument to achieve exact upsampling.

    # Given the text "The network does not have any fully connected layers and only uses the valid part of each convolution, i.e., the segmentation map only contains the pixels, for which the full context is available in the input image. This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy (see Figure 2)." 
    # And "To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image." 
    # This means the *input tile size* is critical.
    # "To allow a seamless tiling of the output segmentation map (see Figure 2), it is important to select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size." 

    # For now, let's keep the `padding=0` as stated in the text. The mismatch in exact sizes between my trace and Figure 1's labeled sizes is likely due to the "copy and crop" mechanism and how the diagram labels dimensions. The `output_size` of `ConvTranspose2d` and careful cropping are key.

    # The `nn.functional.pad` handles the center cropping.
    # Let's adjust the Up module a bit more carefully for the concatenation.
    # The cropping logic in my `Up` class is correct for `padding=0` everywhere.
    # The actual spatial dimensions will shrink with each `DoubleConv`.

    # Final check on the dimensions given in Figure 1 for the skip connections (white boxes):
    # x1: 568x568
    # x2: 280x280
    # x3: 136x136
    # x4: 64x64
    # The central block is 28x28.

    # If the output needs to be 388x388 from 572x572 input, let's just assert that in the test:
    expected_output_H = 388
    expected_output_W = 388

    # Recalculate based on Figure 1's specific crop sizes:
    # After Up1, upconv from 28x28 to 56x56.
    # x4 is 64x64. Cropped to 56x56.
    # Concatenated size for DoubleConv: 56x56. Output 52x52.

    # After Up2, upconv from 52x52 to 104x104.
    # x3 is 136x136. Cropped to 104x104.
    # Concatenated size for DoubleConv: 104x104. Output 100x100.

    # After Up3, upconv from 100x100 to 200x200.
    # x2 is 280x280. Cropped to 200x200.
    # Concatenated size for DoubleConv: 200x200. Output 196x196.

    # After Up4, upconv from 196x196 to 392x392.
    # x1 is 568x568. Cropped to 392x392.
    # Concatenated size for DoubleConv: 392x392. Output 388x388.

    # This trace matches the 388x388 output! So the cropping logic in the `Up` module needs to correctly center-crop the higher-resolution feature map to match the size of the upsampled feature map *before* it's passed into the `DoubleConv`. My current `nn.functional.pad` does this padding for the smaller tensor to match the larger, not cropping the larger to match the smaller.

    # Correcting the `Up` forward pass for cropping:
    # In `Up` class:
    # def forward(self, x1, x2):
    #     x1 = self.up(x1) # x1 is the upsampled tensor (smaller spatial dims than x2 initially)
    #
    #     # Ensure x1 and x2 have matching spatial dimensions for concatenation
    #     # x2 (from contracting path) is larger, so we need to crop x2 to match x1's spatial dimensions.
    #     # Paper states "cropping is necessary due to the loss of border pixels in every convolution." 
    #     # This means the feature map from the contracting path (x2) must be cropped to match the size of the upsampled feature map (x1).
    #     diffY = x2.size()[2] - x1.size()[2]
    #     diffX = x2.size()[3] - x1.size()[3]
    #
    #     # Perform central cropping on x2
    #     x2 = x2[:, :, diffY // 2 : x2.size()[2] - diffY // 2,
    #                         diffX // 2 : x2.size()[3] - diffX // 2]
    #
    #     x = torch.cat([x2, x1], dim=1) 
    #     return self.conv(x)

    # With this correction, the `UNet` class and its `forward` pass should accurately reflect the diagram's dimensions.
    # Let's re-run the test with this mental correction (or actual code if you're writing it).

    # The current code for `nn.functional.pad` does padding to make `x1` larger. This is effectively wrong for the paper's description where `x2` (skip connection) is larger and needs cropping.
    # So, change `x1 = nn.functional.pad(x1, ...)` to `x2 = x2[:, :, start_y:end_y, start_x:end_x]`.

    # Updated `Up` class (inside `unet_model.py`):
    class Up(nn.Module):
        """
        Performs up-convolution, concatenates with cropped feature map,
        and then applies a DoubleConv block.
        """
        def __init__(self, in_channels, out_channels):
            super().__init__()
            # Up-convolution (Transposed Convolution) to increase resolution
            # and halve the number of feature channels
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) #[cite: 55]
            self.conv = DoubleConv(in_channels, out_channels) # The total in_channels here will be (in_channels_after_up + skip_connection_channels)

        def forward(self, x1, x2):
            # x1 is the input from the previous upsampling layer (lower resolution)
            # x2 is the corresponding feature map from the contracting path (higher resolution)

            x1 = self.up(x1)

            # Ensure x1 and x2 have matching spatial dimensions for concatenation
            # x2 (from contracting path) is larger, so we need to crop x2 to match x1's spatial dimensions.
            # Paper states "cropping is necessary due to the loss of border pixels in every convolution." 
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            # Perform central cropping on x2
            # Calculate the start and end indices for cropping
            start_y = diffY // 2
            end_y = x2.size()[2] - (diffY - diffY // 2)
            start_x = diffX // 2
            end_x = x2.size()[3] - (diffX - diffX // 2)

            x2 = x2[:, :, start_y:end_y, start_x:end_x]

            x = torch.cat([x2, x1], dim=1) #[cite: 55]
            return self.conv(x)

    # Now, with this corrected `Up` module, the `UNet` should produce the expected output shape.
    # Rerunning the test:
    model = UNet(n_channels=input_channels, n_classes=output_classes)
    # print(model) # Too verbose
    dummy_input = torch.randn(1, input_channels, 572, 572)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Asserting the expected output shape
    assert output.shape[2] == expected_output_H and output.shape[3] == expected_output_W, \
        f"Output shape mismatch! Expected ({expected_output_H}, {expected_output_W}), got ({output.shape[2]}, {output.shape[3]})"
    print("U-Net architecture test passed successfully with expected output shape.")