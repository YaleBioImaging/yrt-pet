#!/usr/bin/env python
import pyyrtpet as yrt

# Note: This file is not to be executed, but simply to be used as a documentation

scanner = yrt.Scanner("<path to the scanner's json file>")

imgParams = yrt.ImageParams("<path to the image parameters file>")

dataset = yrt.Histogram3DOwned(scanner, "<path to the histogram file>")
# or, alternatively, read a ListMode file
dataset = yrt.ListModeLUTOwned(scanner, "<path to the listmode file>")

# --- Reconstruction setup

# Create OSEM object
osem = yrt.createOSEM(scanner)
# or, alternatively, use GPU reconstruction (if compiled with CUDA)
osem = yrt.createOSEM(scanner, useGPU=True)

osem.setDataInput(dataset) # Dataset to use as input for the reconstruction.
osem.setProjector("<Projector>") # Possible values: S (Siddon), DD (Distance-Driven), or DD_GPU (GPU Distance-Driven, available only if useGPU is 'True')
osem.num_MLEM_iterations = 10 # Number of MLEM iterations
osem.num_OSEM_subsets = 5 # Number of OSEM subsets
osem.setSensDataInput(...) # Dataset to use as input for the sensitivity image generation. Takes, as input, a ProjectionData object.
osem.addTOF(<TOF width in picoseconds>, <Number of STD deviations>) # To enable Time-of-flight
osem.addProjPSF("<path to the PSF's CSV file>") # To add Projection-space PSF
osem.addImagePSF(...) # To add Image-space PSF. Takes, as input, a OperatorPsf object
osem.setListModeEnabled(<True/False>) # To enable if the dataset to use for reconstruction will be in ListMode format. This is important as it changes the way sensitivity images are generated.
osem.attenuationImage = ... # To add an attenuation image (Image object)
osem.addHis = ... # To add an additive histogram (Histogram format) for example for Scatter and Randoms correction.
osem.setImageParams(imgParams) # Set the parameters of the output image

# --- Generate the sensitivity images

# Here, "sens_imgs" will be a list of Image objects and
# they will be automatically registered for the reconstruction
sens_imgs = osem.generateSensitivityImages() # Returns a list of Image objects
# Note that the returned object should *not* be discarded. Otherwise it would cause Segmentation faults during the reconstruction because Python's garbage collector will invalidate the references
# or, alternatively. If you've already generated the sensitivity images:
osem.setSensitivityImages(...) # Takes, as input, a python list of Image objects.

# --- Reconstruction
outImg = osem.reconstruct() # Launch the reconstruction. It will fill 'outImg' with the reconstructed image

outImg.writeToFile("<path where to save the output image>")
