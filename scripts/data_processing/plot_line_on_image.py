#!/usr/bin/env python
import numpy as np
import cv2
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import argparse
import pyyrtpet as yrt

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Generate profile plot of image')
    parser.add_argument('--image', type=str, required=True, help='Image in RWD format')
    parser.add_argument('--slice', type=int, required=True, help='Slice to use')
    parser.add_argument('--p1', type=int, nargs=2, required=True, help='Starting Point to use for the plotting')
    parser.add_argument('--p2', type=int, nargs=2, required=True, help='End Point to use for the plotting')
    parser.add_argument('--out', type=str, required=False, help='Output figure to save')
    args = parser.parse_args()

    yrt_img = yrt.Array3Ddouble()
    yrt_img.readFromFile(args.image)
    img = np.array(yrt_img, copy=False)[args.slice]

    start_point = (args.p1[0],args.p1[1])
    end_point = (args.p2[0],args.p2[1])
    
    img_line = np.zeros(img.shape)
    img_line = cv2.line(img_line, start_point, end_point, color=1, thickness=1)

    points_idx = np.where(img_line>0)
    points = np.array([points_idx[1],points_idx[0]]).transpose()
    distances = np.sqrt((points[:,0] - start_point[0])**2 + (points[:,1] - start_point[1])**2)

    dist_sorting = np.argsort(distances)

    distances = distances[dist_sorting]
    vals = img[points_idx][dist_sorting]
    
    plotpoints = np.array([distances, vals])
    np.save(args.out, plotpoints)
