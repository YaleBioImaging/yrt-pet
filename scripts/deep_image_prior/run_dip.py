#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
import pyyrtpet as yrt


# 3D Unet architecture for Deep image prior reconstruction using a forward model
# from pyyrtpet
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet3D, self).__init__()

        features = init_features
        # 16 features -> 32 features
        self.encoder1 = self._block(in_channels, features)
        self.downsampler1 = nn.Conv3d(
            features, features * 2, kernel_size=3, stride=2, padding=1
        )
        # 32 features -> 64 features
        self.encoder2 = self._block(features * 2, features * 2)
        self.downsampler2 = nn.Conv3d(
            features * 2, features * 4, kernel_size=3, stride=2, padding=1
        )
        # 64 features -> 128 features
        self.encoder3 = self._block(features * 4, features * 4)
        self.downsampler3 = nn.Conv3d(
            features * 4, features * 8, kernel_size=3, stride=2, padding=1
        )
        # 128 features -> 128 features
        self.bottleneck = self._block(features * 8, features * 8)

        # Transpose conv
        self.conv_trans3 = nn.ConvTranspose3d(
            features * 8,
            features * 4,
            kernel_size=3,
            stride=1,
        )
        # Upsampler here
        self.decoder3 = self._block(features * 8, features * 4)

        self.conv_trans2 = nn.ConvTranspose3d(
            features * 4,
            features * 2,
            kernel_size=3,
            stride=1,
        )
        # Upsampler here
        self.decoder2 = self._block(features * 4, features * 2)

        self.conv_trans1 = nn.ConvTranspose3d(
            features * 2,
            features,
            kernel_size=3,
            stride=1,
        )
        # Upsampler here
        self.decoder1 = self._block(features * 2, features)

        self.conv_and_relu = nn.Sequential(
            nn.Conv3d(features, out_channels, kernel_size=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_d = self.downsampler1(enc1)

        enc2 = self.encoder2(enc1_d)
        enc2_d = self.downsampler2(enc2)

        enc3 = self.encoder3(enc2_d)
        enc3_d = self.downsampler3(enc3)

        t_bottleneck = self.bottleneck(enc3_d)

        dec3 = self.conv_trans3(t_bottleneck)
        dec3 = F.interpolate(
            dec3, size=enc3.shape[2:], mode="trilinear", align_corners=False
        )
        dec3 = torch.cat((dec3, enc3), dim=1)  # 1 is channel dimension
        dec3 = self.decoder3(dec3)

        dec2 = self.conv_trans2(dec3)
        dec2 = F.interpolate(
            dec2, size=enc2.shape[2:], mode="trilinear", align_corners=False
        )
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.conv_trans1(dec2)
        dec1 = F.interpolate(
            dec1, size=enc1.shape[2:], mode="trilinear", align_corners=False
        )
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        y = self.conv_and_relu(dec1)

        return y

    @staticmethod
    def _block(in_channels, num_features):
        return nn.Sequential(
            nn.Conv3d(in_channels, num_features, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features),
            nn.LeakyReLU(inplace=True),
        )


# Forward model for the Deep image prior reconstruction
class ForwardProject(Function):
    @staticmethod
    def forward(ctx, image):
        # Perform the forward projection using op_proj.applyA
        projdata_device.setProjValuesDevicePointer(forward_projection.data_ptr())
        image_device.setDevicePointer(image.data_ptr())
        op_proj.applyA(image_device, projdata_device)
        ctx.save_for_backward(image)

        return forward_projection

    @staticmethod
    def backward(ctx, grad_output):
        (image,) = ctx.saved_tensors
        # Create a tensor to store the gradient
        grad_input = torch.zeros_like(image)
        # Set the device pointers
        projdata_device.setProjValuesDevicePointer(grad_output.data_ptr())
        image_device.setDevicePointer(grad_input.data_ptr())
        # Perform the back-projection (gradient computation)
        op_proj.applyAH(projdata_device, image_device)
        # Return the gradient
        return grad_input


# Define forward projection
def forward_project(image):
    return ForwardProject.apply(image)


# Log-likelihood loss between the forward projection of the prediction and
# the target under the assumption that the target has a Poisson distribution
# with handling of zero values
def compute_loss(prediction, target):
    # Forward projection of the prediction
    forward_prediction = forward_project(prediction)

    # multiplicative attenuation coorection factor and norm corrections on
    # the forward projection
    forward_prediction = forward_prediction * acf_torch * sensitivity_torch

    # add scatter and randoms to the forward projection
    forward_prediction = forward_prediction + scatter_torch + randoms_torch

    # Compute the Poisson log-likelihood loss
    loss = torch.sum(forward_prediction - target * torch.log(forward_prediction + 1e-7))

    # Compute MSE
    # loss = torch.sum(torch.pow(forward_prediction - target, 2))

    return loss


def save_array_to_nifti_image(
    img_params: yrt.ImageParams, array: np.ndarray, filename: str
):
    img_yrt = yrt.ImageAlias(img_params)
    array_np = np.require(array, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    img_yrt.bind(array_np)
    img_yrt.writeToFile(filename)


# Main
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="dip_histo",
        description="Reproducing: Direct PET Image Reconstruction "
        "Incorporating Deep Image Prior and a Forward Projection Model "
        "(Fumio Hashimoto and Kibo Ote).\n"
        "All Histograms given as arguments have to be in YRT-PET's Histogram3D format.\n"
        "All images given as arguments or generated by this script are in NIfTI format.",
    )

    parser.add_argument("-s", "--scanner", help="Scanner JSON file", required=True)
    parser.add_argument(
        "-p", "--params", help="Image parameters JSON file", required=True
    )
    parser.add_argument("--prompts", help="Prompts histogram file", required=True)
    parser.add_argument("--acf", help="ACF histogram file")
    parser.add_argument("--sensitivity", help="Sensitivity histogram file")
    parser.add_argument("--scatter", help="Scatter histogram file")
    parser.add_argument("--randoms", help="Randoms histogram file")
    parser.add_argument(
        "--nn_input",
        help="Input to the Neural network image"
        "file (NIfTI format). Leave empty to use noise as input.",
    )
    parser.add_argument(
        "--init_weights",
        help="Initial weights to the neural network ('.pth' file).",
    )
    parser.add_argument("-o", "--out", help="Output folder", required=True)
    parser.add_argument(
        "--projector", help="Projector to use " "(Siddon (S) or Distance-driven (DD))"
    )
    parser.add_argument(
        "--num_rays",
        help="Number of rays to use (for the Siddon projector only) (Default: 1)",
        type=int,
        default=1,
    )
    default_learning_rate = 0.1
    parser.add_argument(
        "--learning_rate",
        help="Learning rate (Default: " + str(default_learning_rate) + ")",
        type=float,
        default=default_learning_rate,
    )
    default_num_iterations = 1000
    parser.add_argument(
        "--num_iterations",
        help="Number of DIP iterations (Default: " + str(default_num_iterations) + ")",
        type=int,
        default=default_num_iterations,
    )
    default_save_iter_step = 100
    parser.add_argument(
        "--save_iter_step",
        help="Increment into which to save DIP iteration images "
        "(Default: " + str(default_save_iter_step) + ")",
        type=int,
        default=default_save_iter_step,
    )

    args = parser.parse_args()

    # Prepare output folder
    if not os.path.isdir(args.out):
        print("Path given does not exist, creating it...")
        os.mkdir(args.out)

    print("Saving current configuration...")
    # Save command line to text file
    command = " ".join(sys.argv[1:])
    with open(os.path.join(args.out, "command_arguments.txt"), "w") as f:
        f.write(__file__ + " " + command + "\n")
    with open(os.path.join(args.out, "working_directory.txt"), "w") as f:
        f.write(os.getcwd() + "\n")
    # Save this script to the output folder
    with open(__file__, "r") as f:
        current_script_content = f.read()
    with open(os.path.join(args.out, "script_used.py"), "w") as f:
        f.write(current_script_content)

    # General config
    device_number = 0
    torch.cuda.set_device(device_number)
    assert torch.cuda.is_available()
    device = torch.device(f"cuda:{device_number}")

    # Initialize scanner and image parameters
    scanner = yrt.Scanner(args.scanner)
    img_params = yrt.ImageParams(args.params)

    # Read histogram data
    print("Reading input data...")
    prompts = yrt.Histogram3DOwned(scanner, args.prompts)
    prompts_np = np.array(prompts, copy=False)
    prompts_torch = torch.from_numpy(prompts_np).float().to(device)

    if args.acf is not None:
        acf = yrt.Histogram3DOwned(scanner, args.acf)
        acf_np = np.array(acf, copy=False)
        acf_torch = torch.from_numpy(acf_np).float().to(device)
    else:
        acf_torch = torch.ones_like(prompts_torch).to(device)

    if args.sensitivity is not None:
        sensitivity = yrt.Histogram3DOwned(scanner, args.sensitivity)
        sensitivity_np = np.array(sensitivity, copy=False)
        sensitivity_torch = torch.from_numpy(sensitivity_np).float().to(device)
    else:
        sensitivity_torch = torch.ones_like(prompts_torch).to(device)

    if args.scatter is not None:
        scatter = yrt.Histogram3DOwned(scanner, args.scatter)
        scatter_np = np.array(scatter, copy=False)
        scatter_torch = torch.from_numpy(scatter_np).float().to(device)
    else:
        scatter_torch = torch.zeros_like(prompts_torch).to(device)

    if args.randoms is not None:
        randoms = yrt.Histogram3DOwned(scanner, args.randoms)
        randoms_np = np.array(randoms, copy=False)
        randoms_torch = torch.from_numpy(randoms_np).float().to(device)
    else:
        randoms_torch = torch.zeros_like(prompts_torch).to(device)

    # Prepare buffer
    forward_projection = torch.zeros(
        prompts_np.shape, layout=torch.strided, device=device
    )

    # Initialize the projector

    # Create bin iterator for subset index 0 out of 1
    print("Initializing projector...")
    bin_iter = prompts.getBinIter(1, 0)
    proj_params = yrt.OperatorProjectorParams(bin_iter, scanner, num_rays=args.num_rays)

    if args.projector == "S":
        op_proj = yrt.OperatorProjectorSiddon_GPU(proj_params)
    elif args.projector == "DD":
        op_proj = yrt.OperatorProjectorDD_GPU(proj_params)
    else:
        raise ValueError("Unknown projector given: " + str(args.projector))

    # Prepare device-side projection-space buffer
    # 1 is the number of subsets, 'prompts' is used as reference for calculating the LORs
    print("Initializing device-side projection-space buffers...")
    projdata_device = yrt.ProjectionDataDeviceAlias(scanner, prompts, 1)
    assert projdata_device.getNumBatches(0) == 1
    projdata_device.prepareBatchLORs(0, 0)  # Load batch 0 subset 0

    # Prepare image-side image-space buffer
    print("Initializing device-side image-space buffers...")
    image_device = yrt.ImageDeviceAlias(img_params)

    # Initialize U-Net and optimizer
    print("Initializing neural network...")
    input_channels = 1  # For a single-channel PET slice
    # Assuming the output is also a single-channel image
    output_channels = input_channels

    model = UNet3D().to(device)

    if args.init_weights is not None:
        model.load_state_dict(torch.load(args.init_weights, weights_only=True))

    # Initialize the network input as either prior or noise
    print("Initializing neural network input...")

    # Generate noise or load prior
    if args.nn_input is None:
        # 1 batch size, 1 channel, dimensions as required
        nn_input_torch = torch.rand(1, 1, img_params.nz, img_params.ny, img_params.nx)

        nn_input_np = nn_input_torch.squeeze().numpy()
        save_array_to_nifti_image(
            img_params, nn_input_np, os.path.join(args.out, "nn_input.nii.gz")
        )

        # Move to appropriate device
        nn_input_torch = nn_input_torch.to(device)
    else:
        nn_input = yrt.ImageOwned(args.nn_input)
        nn_input_np = np.array(nn_input, copy=False)

        # TODO: Apply this to the MR data given:
        # prior_normalized = (nn_input_np - np.min(nn_input_np)) / (np.max(nn_input_np) - np.min(nn_input_np))

        # Convert to PyTorch tensor and add batch and channel dimensions
        nn_input_torch = torch.from_numpy(nn_input_np).unsqueeze(0).unsqueeze(1).float()

        # Move to the appropriate device
        nn_input_torch = nn_input_torch.to(device)

    print("Initializing optimizer...")
    learning_rate = args.learning_rate
    optimizer = optim.LBFGS(
        model.parameters(), lr=learning_rate, max_iter=20, history_size=100
    )

    def closure():
        optimizer.zero_grad()
        output = model(nn_input_torch)
        loss = compute_loss(output, prompts_torch)
        loss.backward()
        return loss

    loss_values = []

    # Optimization loop
    print("Starting iterations...")
    num_iterations = args.num_iterations

    for step in range(num_iterations):

        # LBFGS requires calling optimizer.step(closure)
        # where 'closure' re-computes the loss and gradients.
        optimizer.step(closure)
        # After the step, retrieve the current loss from the closure
        loss = closure()

        print(f"Step {step}, Loss: {loss.item()}")
        loss_values.append(loss.item())

        # save output every few steps
        if step % args.save_iter_step == 0:
            output = model(nn_input_torch)
            output_np = output.squeeze().cpu().detach().numpy()
            save_array_to_nifti_image(
                img_params,
                output_np,
                os.path.join(
                    args.out, "recon_image_iteration" + str(step).zfill(4) + ".nii.gz"
                ),
            )

    # Save the final output
    print("Saving final image...")
    output = model(nn_input_torch)
    output_np = output.squeeze().cpu().detach().numpy()
    save_array_to_nifti_image(
        img_params, output_np, os.path.join(args.out, "recon_image.nii.gz")
    )

    # Save the loss curve as png
    loss_values_np = np.array(loss_values)
    np.save(os.path.join(args.out, "loss_values.npy"), loss_values_np)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(args.out, "model_weights.pth"))
