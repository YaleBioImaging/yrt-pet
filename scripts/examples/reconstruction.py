#!/usr/bin/env python
import pyyrtpet as yrt

# Note: This file is not to be executed, but to be used as a documentation

scanner = yrt.Scanner("<path to the scanner's json file>")
imgParams = yrt.ImageParams("<path to the image parameters file>")

# --- Reconstruction setup

# Create OSEM object
osem = yrt.createOSEM(scanner)
# or, alternatively, use GPU reconstruction (available only if yrt.compiledWithCuda() returns True)
osem = yrt.createOSEM(scanner, useGPU=True)

osem.setProjector("<Projector>") # Possible values: S (Siddon), DD (Distance-Driven), or DD_GPU (GPU Distance-Driven, available only if useGPU is 'True')
osem.setImageParams(imgParams) # Set the parameters of the output image
osem.num_MLEM_iterations = 10 # Number of MLEM iterations
osem.num_OSEM_subsets = 5 # Number of OSEM subsets
osem.setSensDataInput(...) # Dataset to use as input for the sensitivity image generation. Takes, as input, a ProjectionData object. Example: A histogram of sensitivity
osem.addTOF("""<TOF width in picoseconds>""", """<Number of STD deviations>""") # To enable Time-of-flight
osem.addProjPSF("<path to the PSF's CSV file>") # To add Projection-space PSF
osem.addImagePSF(...) # To add Image-space PSF. Takes, as input, a OperatorPsf object
osem.setListModeEnabled("""<True/False>""") # To enable if the dataset to use for reconstruction will be in ListMode format. This is important as it changes the way sensitivity image(s) are generated.
osem.attenuationImageForForwardProjection = ... # To add an attenuation image (Image object) to the forward model
osem.addHis = ... # To add an additive histogram (Histogram format) for example for Scatter and Randoms corrections.

# --- Generate the sensitivity image(s)

sens_imgs = osem.generateSensitivityImages() # Returns a python list of Image objects
osem.setSensitivityImages(sens_imgs) # Takes, as input, a python list of Image objects.

# Optionally, save the sensitivity images:
for i in range(len(sens_imgs)):
    sens_imgs[i].writeToFile("<path where to save the sensitivity images>/sens_img_subset"+str(i)+".nii")

# --- Reconstruction

# Read histogram data to reconstruct
dataset = yrt.Histogram3DOwned(scanner, "<path to the histogram file>")
# or, alternatively, read a ListMode file
dataset = yrt.ListModeLUTOwned(scanner, "<path to the listmode file>")

osem.setDataInput(dataset) # Dataset to use as input for the reconstruction.
outImg = osem.reconstruct() # Launch the reconstruction. It will return the reconstructed image

outImg.writeToFile("<path where to save the output image>")
