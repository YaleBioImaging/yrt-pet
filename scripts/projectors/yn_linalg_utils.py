#!/usr/bin/env python
FLAG_PLOT = False

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
if FLAG_PLOT:
    from matplotlib.ticker import MultipleLocator
    import matplotlib as mpl
    #mpl.use('Qt5Agg')
    mpl.use('TkAgg')



def plotpoint(ax, pt, lbl=None):
    ax.scatter([pt[0]], [pt[1]], label=lbl)

def plotline(ax, p1, p2, lbl=None, color=None):
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]], label=lbl, color=color)

def plotpoint3D(ax, pt, lbl=None):
    ax.scatter([pt[0]], [pt[1]], [pt[2]], label=lbl)

def plotline3D(ax, p1, p2, lbl=None, color=None):
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]], [p1[2],p2[2]], label=lbl, color=color)

def area_of_region(pts):
    numPoints = np.size(pts, 0)
    pts_t = np.transpose(pts)
    meanxy = [np.mean(pts_t[0]),np.mean(pts_t[1])]
    diff_pts = pts-meanxy
    #print(diff_pts)
    diff_pts_t = np.transpose(diff_pts)
    angles = np.arctan2(diff_pts_t[1], diff_pts_t[0])
    #print(angles)
    angles_order = np.argsort(angles)
    sorted_pts=np.array(pts)
    for i in np.arange(np.size(angles_order,0)):
        sorted_pts[i] = pts[angles_order[i]]
    #print(angles_order)
    #print(sorted_pts)
    
    area = 0 #Accumulates area 
    j = numPoints - 1
    i=0
    while i<numPoints:
        area += ((sorted_pts[j][0]+sorted_pts[i][0]) * (sorted_pts[j][1]-sorted_pts[i][1]))
        j = i #j is previous vertex to i
        i += 1
    return np.abs(area/2);

#This function does not work for lines parallel to x or y axis
def area_under_line_in_rect(p1, p2, d1, d2, plotting=True, above_or_under=False):
    # Based on the shoelace formula
    
    s=d1
    r=d2-d1
    
    p1xp2y = np.array([p1[0], p2[1]])
    p2xp1y = np.array([p2[0], p1[1]])
    sq_pts = np.array([p1,p2,p1xp2y,p2xp1y])

    #y=ax+b
    if r[0] == 0:
        return None

    a = r[1]/r[0]
    b = s[1]-a*s[0]

    maybe_intersections = np.zeros([4,2])
    maybe_intersections[0] = np.array([p1[0], a*p1[0]+b])
    maybe_intersections[1] = np.array([(p1[1]-b)/a, p1[1]])
    maybe_intersections[2] = np.array([p2[0], a*p2[0]+b])
    maybe_intersections[3] = np.array([(p2[1]-b)/a, p2[1]])
    intersections = np.zeros([0,2])
    it=0
    #Filter intersections outside square
    for i in np.arange(np.size(maybe_intersections,0)):
        if not (maybe_intersections[i][0]<p1[0] or maybe_intersections[i][1]<p1[1] or maybe_intersections[i][0]>p2[0] or maybe_intersections[i][1]>p2[1]) :
            intersections = np.insert(intersections, 0, maybe_intersections[i], axis=0)
            #print("intersect: "+str(intersections[it]))
            it+=1

    all_pts = np.array(intersections)
    for sq_pt in sq_pts:
        if above_or_under == False:
            if(a*sq_pt[0]+b>sq_pt[1]):
                all_pts = np.insert(all_pts, 0, sq_pt, axis=0)
        else:
            if(a*sq_pt[0]+b<sq_pt[1]):
                all_pts = np.insert(all_pts, 0, sq_pt, axis=0)
    #print(all_pts)

    if FLAG_PLOT and plotting:
        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = ax = fig.add_subplot(111)
        
        plotpoint(ax, [0,0], "Origin")
        plotpoint(ax, p1, "p1")
        plotpoint(ax, p2, "p2")

        for intersection in intersections:
            plotpoint(ax, intersection, "Line-square intersection "+str(intersection))
        plotpoint(ax, p2xp1y, "p2xp1y")
        plotpoint(ax, p1xp2y, "p1xp2y")
        
        plotline(ax, s, r, lbl="s", length=2)

        ax.grid(which="minor")
        ax.grid(which="major")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    return area_of_region(all_pts)

def planes_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz = D
           A,B,C,D in order  

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
    a_vec[3] = -a_vec[3]
    b_vec[3] = -b_vec[3]
    
    
    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]

#From three 3D points, gets the A,B,C and D of the plane
def plane_from_three_pts(p1,p2,p3):
    n = np.cross(p2-p1,p3-p1)
    n = n*1/(np.sqrt(n.dot(n))) # normalize
    nd = n.dot(p1)
    return np.array([n[0],n[1],n[2],nd])

def adjacent_prism_planes_from_three_pts(p1,p2,p3):
    p4 = p1+(p3-p2)
    dummy_normal_vector = np.cross(p2-p1,p3-p1)
    p1_prime = p1+dummy_normal_vector
    p2_prime = p2+dummy_normal_vector
    p3_prime = p3+dummy_normal_vector
    p4_prime = p4+dummy_normal_vector
    
    planes = np.zeros([4,4])
    planes[0]=plane_from_three_pts(p1,p1_prime,p4)
    planes[1]=plane_from_three_pts(p2,p2_prime,p1)
    planes[2]=plane_from_three_pts(p3,p3_prime,p2)
    planes[3]=plane_from_three_pts(p4,p4_prime,p3)
    return planes

def get_voxel_planes(p1,p2):
    planes = np.zeros([6,4])
    pts = np.zeros([18,3])
    #All combinations to get all eight vertices
    #Have been reordered to make the face generation easy
    pts[0] = np.array([p1[0], p1[1], p1[2]])
    pts[1] = np.array([p1[0], p1[1], p2[2]])
    pts[2] = np.array([p1[0], p2[1], p2[2]])

    pts[3] = np.array([p1[0], p1[1], p1[2]])
    pts[4] = np.array([p2[0], p1[1], p1[2]])
    pts[5] = np.array([p2[0], p1[1], p2[2]])

    pts[6] = np.array([p2[0], p2[1], p1[2]])
    pts[7] = np.array([p2[0], p1[1], p1[2]])
    pts[8] = np.array([p1[0], p1[1], p1[2]])

    pts[9] = np.array([p2[0], p2[1], p2[2]])
    pts[10] = np.array([p2[0], p1[1], p2[2]])
    pts[11] = np.array([p2[0], p1[1], p1[2]])

    pts[12] = np.array([p1[0], p2[1], p1[2]])
    pts[13] = np.array([p1[0], p2[1], p2[2]])
    pts[14] = np.array([p2[0], p2[1], p2[2]])

    pts[15] = np.array([p2[0], p2[1], p2[2]])
    pts[16] = np.array([p1[0], p2[1], p2[2]])
    pts[17] = np.array([p1[0], p1[1], p2[2]])

    
    for i in np.arange(np.size(planes,0)):
        planes[i] = plane_from_three_pts(pts[3*i],pts[3*i+1],pts[3*i+2])
        #print("pts["+str(i)+"]: "+str(pts[3*i]))
        #print("pts["+str(i+1)+"]: "+str(pts[3*i+1]))
        #print("pts["+str(i+2)+"]: "+str(pts[3*i+2]))
        #print("planes["+str(i)+"]: "+str(planes[i]))
        #print("----")
    return planes

def adjacent_prism_lines_from_three_pts(p1,p2,p3):
    p4 = p1+(p3-p2)
    dummy_normal_vector = np.cross(p2-p1,p3-p1)*3
    p1_prime = p1+dummy_normal_vector
    p2_prime = p2+dummy_normal_vector
    p3_prime = p3+dummy_normal_vector
    p4_prime = p4+dummy_normal_vector
    
    lines_p1 = np.zeros([4,3])
    lines_p2 = np.zeros([4,3])
    lines_p1[0] = p1
    lines_p2[0] = p1_prime
    lines_p1[1] = p2
    lines_p2[1] = p2_prime
    lines_p1[2] = p3
    lines_p2[2] = p3_prime
    lines_p1[3] = p4
    lines_p2[3] = p4_prime
    
    if FLAG_PLOT:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for i in np.arange(np.size(lines_p1, 0)):
            plotline3D(ax, lines_p1[i], lines_p2[i])
        
        ax.grid(which="minor")
        ax.grid(which="major")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    return (lines_p1, lines_p2)

def get_voxel_lines(p1,p2):
    lines_p1 = np.zeros([12,3])
    lines_p2 = np.zeros([12,3])
    pts = np.zeros([8,3])
    #All combinations to get all eight vertices
    #Have been reordered to make the face generation easy
    pts[0] = np.array([p1[0], p1[1], p1[2]])
    pts[1] = np.array([p1[0], p1[1], p2[2]])
    pts[2] = np.array([p1[0], p2[1], p2[2]])
    pts[3] = np.array([p1[0], p2[1], p1[2]])
    
    pts[4] = np.array([p2[0], p1[1], p1[2]])
    pts[5] = np.array([p2[0], p1[1], p2[2]])
    pts[6] = np.array([p2[0], p2[1], p2[2]])
    pts[7] = np.array([p2[0], p2[1], p1[2]])
    
    lines_p1[0] = pts[0]
    lines_p2[0] = pts[1]
    lines_p1[1] = pts[1]
    lines_p2[1] = pts[2]
    lines_p1[2] = pts[2]
    lines_p2[2] = pts[3]
    lines_p1[3] = pts[3]
    lines_p2[3] = pts[0]

    lines_p1[4] = pts[4]
    lines_p2[4] = pts[5]
    lines_p1[5] = pts[5]
    lines_p2[5] = pts[6]
    lines_p1[6] = pts[6]
    lines_p2[6] = pts[7]
    lines_p1[7] = pts[7]
    lines_p2[7] = pts[4]
    
    lines_p1[8]  = pts[0]
    lines_p2[8]  = pts[4]
    lines_p1[9]  = pts[1]
    lines_p2[9]  = pts[5]
    lines_p1[10] = pts[2]
    lines_p2[10] = pts[6]
    lines_p1[11] = pts[3]
    lines_p2[11] = pts[7]

    if FLAG_PLOT:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for i in np.arange(np.size(lines_p1, 0)):
            plotline3D(ax, lines_p1[i], lines_p2[i])
        
        ax.grid(which="minor")
        ax.grid(which="major")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    return (lines_p1, lines_p2)

def get_voxel_points(p1, p2):
    pts = np.zeros([8,3])
    pts[0] = np.array([p1[0], p1[1], p1[2]])
    pts[1] = np.array([p1[0], p1[1], p2[2]])
    pts[2] = np.array([p1[0], p2[1], p2[2]])
    pts[3] = np.array([p1[0], p2[1], p1[2]])
    pts[4] = np.array([p2[0], p1[1], p1[2]])
    pts[5] = np.array([p2[0], p1[1], p2[2]])
    pts[6] = np.array([p2[0], p2[1], p2[2]])
    pts[7] = np.array([p2[0], p2[1], p1[2]])
    return pts

def is_point_in_voxel(v_p1, v_p2, pt):
    bounds_x = np.array([np.min([v_p1[0],v_p2[0]]), np.max([v_p1[0],v_p2[0]])])
    bounds_y = np.array([np.min([v_p1[1],v_p2[1]]), np.max([v_p1[1],v_p2[1]])])
    bounds_z = np.array([np.min([v_p1[2],v_p2[2]]), np.max([v_p1[2],v_p2[2]])])
    
    if pt[0] < bounds_x[0] or pt[0] > bounds_x[1]:
        return False
    if pt[1] < bounds_y[0] or pt[1] > bounds_y[1]:
        return False
    if pt[2] < bounds_z[0] or pt[2] > bounds_z[1]:
        return False
    return True

def is_point_in_planes(planes,pt):
    epsilon = 10e-9
    for plane in planes:
        planeNormal = np.array([plane[0],plane[1],plane[2]])
        if plane[2] != 0:
            planePoint = np.array([1, 1, (plane[3]-plane[0]-plane[1])/(plane[2])])#Any point on the plane
        elif plane[1] != 0:
            planePoint = np.array([1, (plane[3]-plane[2]-plane[0])/(plane[1]), 1])#Any point on the plane
        elif plane[0] != 0:
            planePoint = np.array([(plane[3]-plane[1]-plane[2])/(plane[0]), 1, 1])#Any point on the plane
        else:
            return # Not a valid plane

        dist = planeNormal.dot(planePoint-pt)
        if dist < 0 and not(np.abs(dist)<epsilon):
            #print("rejected: "+str(pt)+"(dist="+str(dist)+")")
            return False
    return True

def intersect_line_plan(p1,p2,plane):
    #Define plane
    planeNormal = np.array([plane[0],plane[1],plane[2]])
    if plane[2] != 0:
        planePoint = np.array([1, 1, (plane[3]-plane[0]-plane[1])/(plane[2])])#Any point on the plane
    elif plane[1] != 0:
        planePoint = np.array([1, (plane[3]-plane[2]-plane[0])/(plane[1]), 1])#Any point on the plane
    elif plane[0] != 0:
        planePoint = np.array([(plane[3]-plane[1]-plane[2])/(plane[0]), 1, 1])#Any point on the plane
    else:
        return
    #Define ray
    rayDirection = p2-p1
    rayPoint = p2 #Any point along the ray
    
    ndotu = planeNormal.dot(rayDirection)

    if abs(ndotu) < 10e-8:
        print ("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint

    return Psi

def squared_area(p1, p2, p3):
    n = np.cross((p3-p1),(p2-p1))
    sq_area = 0.5*n.dot(n) # Square the vector
    return sq_area

def signed_volume_tetrahedron(a,b,c,d):
    return (1/6)*np.dot((d-a),np.cross(b-a,c-a))

def volume_of_points(pts):
    if np.size(pts,0) == 0:
        return 0
    else:
        return ConvexHull(pts).volume
