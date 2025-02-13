#!/usr/bin/env python
import pyyrtpet as yrt

# Note: This file is not to be executed, but simply to be used as a documentation

scanner = yrt.Scanner("<path to the scanner's json file>")
imgParams = yrt.ImageParams("<path to the image parameters file>")

# Read a histogram
dataset = yrt.Histogram3DOwned(scanner, "<path to an histogram to be backprojected>")
# or, alternatively, read a ListMode file
dataset = yrt.ListModeLUTOwned(scanner, "<path to a listmode to be backprojected>")

# Prepare an empty image
outImage = yrt.ImageOwned(imgParams)
outImage.allocate()

# Available projectors: SIDDON, DD, DD_GPU
projectorType = yrt.OperatorProjector.ProjectorType.DD

yrt.backProject(scanner, outImage, dataset, projectorType)

outImage.writeToFile("<path where to save the output image>")
