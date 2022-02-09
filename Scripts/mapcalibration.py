from typing import AsyncIterable
from matplotlib.colors import Colormap
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as mtransforms
from math import gamma, pi, atan, sin, cos, acos, sqrt
from scipy.ndimage import rotate, shift
import pickle
from sklearn import linear_model
from scipy.optimize import curve_fit, fmin
import copy

CALIBRATION_METHOD = 'MIN_SQUARES'


print('Nice')

with open('shared.pkl', 'rb') as fp:
    shared = pickle.load(fp)


NO_FLOORS = shared[-2]
NO_OF_POINTS = shared[-1]

MAP_IMAGES = []
floor_points = []

for i in range(NO_FLOORS):
    map_path = shared[i]['img']
    MAP_IMAGES.append(mpimg.imread(map_path))

# extract all floor points chosen in GUI

for i in range(NO_FLOORS):
    info = list(shared[i].values())
    points = []
    for j in range(NO_OF_POINTS):
        points.append(info[j])
    floor_points.append(points) # now floor points = [f1, f2, f3, ...]



#### FUNCTIONS TO BE USED LATER ####

def find_len(point1, point2):
    """find length using pythagoras theorem"""
    return sqrt((point2[1]-point1[1])**2 + (point2[0]-point1[0])**2)


def find_angle(point1, point2):
    """find angle between 2 points"""
    origin = (0, 0)
    A = find_len(origin, point1)
    B = find_len(point1, point2)
    C = find_len(origin, point2)

    angle = acos((A**2 + C**2 - B**2)/(2*A*C))

    return angle


def translation(trans_x, trans_y, x, y):
    """translation of point (x, y) by (trans_x, trans_y)"""
    xt = x + trans_x
    yt = y + trans_y
    return xt, yt


def rotation(angle, x, y):
    """rotation of point (x, y) about origin by angle"""
    xr = x*cos(angle) - y*sin(angle)
    yr = x*sin(angle) + y*cos(angle)
    return xr, yr


##### FINDING TRANSLATION AMOUNTS #####

translated_floor_points = copy.deepcopy(floor_points)
translation_amounts = []
trans_matrices = []


for f in translated_floor_points:
    translation_x = f[0][0]
    translation_y = f[0][1]
    translation_amounts.append([translation_x, translation_y])
    for point in f:
        point[0] -= translation_x
        point[1] -= translation_y

    # translation matrix for each floor
    # how to get matrix: https://en.wikipedia.org/wiki/Translation_(geometry)
    # trans_matrices.append(np.array(([[1, 0, 0, -translation_x],
    #                                 [0, 1, 0, -translation_y],
    #                                 [0, 0, 1, 0],
    #                                 [0, 0 ,0, 1]])))
    trans_matrices.append(np.array(([[-translation_x],
                                     [-translation_y],
                                     [0]])))

# for i in range(len(trans_matrices)):
#     print('Translation Matrix for Floor {}'.format(i+1))
#     print(trans_matrices[i])

###### FINDING ANGLES BETWEEN CORRESPONDING POINTS #####

all_angles_by_floor = []

for i in range(1, NO_OF_POINTS): # number of points
    angles = []    
    for j in range(1, NO_FLOORS): # number of floors 
        point1 = floor_points[0][i] # first floor point i
        point2 = floor_points[j][i]
        angles.append(find_angle(floor_points[0][i], floor_points[j][i]))

    all_angles_by_floor.append(angles)


##### METHOD 1: AVERAGE ANGLE CALIBRATION ####

if CALIBRATION_METHOD == 'AVERAGE_ANGLE':

    # initialise plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # find avg rotation angles
    rot_angles = [0] # initialise with 0 rotation angle for 1st floor

    for i in range(NO_FLOORS-1): # no of floors - 1 cause exclude 1st floor
        sum = 0
        for j in range(NO_OF_POINTS-1): # no of points - 1 cause excluding origin point after translation
            sum += all_angles_by_floor[j][i]
        average_rot = sum/(NO_OF_POINTS-1) # no of angles summed up (no of points - 1)
    
        rot_angles.append(average_rot)
        
            

    # translate and rotate

    for i in range(NO_FLOORS):
        x, y = np.mgrid[0:MAP_IMAGES[i].shape[0], 0:MAP_IMAGES[i].shape[1]]
        z = np.atleast_2d(i*10)
        
        xt, yt = translation(-translation_amounts[i][0], -translation_amounts[i][1], x, y)

        xtr, ytr = rotation(rot_angles[i], xt, yt)

        ax.plot_surface(xtr, ytr, z, facecolors=MAP_IMAGES[i], shade=False, rstride=20, cstride=20)


    plt.show()


elif CALIBRATION_METHOD == 'MIN_SQUARES':
    NO_OF_ANGLES = 100
    rot_angles = np.linspace(0,2*pi, NO_OF_ANGLES)


    # generating all rotated points for each angle for all floor
    all_rotated_points = []

    for floor in range(1, NO_FLOORS):

        floor_rotated_points = []

        for angle in rot_angles:
            by_angles = []
            for point in translated_floor_points[floor]:
                rot_point = []
                x, y = rotation(angle, point[0], point[1])
                rot_point.append(x)
                rot_point.append(y)
                by_angles.append(rot_point)
            floor_rotated_points.append(by_angles)
        
        all_rotated_points.append(floor_rotated_points)

    # finding distance between corresponding points

    all_distances = []

    for floor in range(0, NO_FLOORS-1):    
        floor_dist = []
        translated_f1 = translated_floor_points[0]

        for i in range(NO_OF_ANGLES): # no of angles in rot_angles
            sum = 0
            for j in range(NO_OF_POINTS): # no of points
                p2pdist = find_len(translated_f1[j], all_rotated_points[floor][i][j])
                
                sum += p2pdist

            floor_dist.append(sum)
        all_distances.append(floor_dist)



    # fitting model and getting rotation angle for each floor

    rotation_angles = [0] # initialise with 0 cause first floor wont rotate
    # rot_matrices = [np.array(([[1, 0, 0, 0],
    #                            [0, 1, 0, 0],
    #                            [0, 0, 1, 0],
    #                            [0, 0, 0, 1]]))]

    rot_matrices = [np.array(([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]))]


    def objective(theta, a, phi):
        return np.sqrt((2*a**2)*(1-np.cos(theta+phi)))



    for floor in all_distances:

        popt, pcov = curve_fit(objective, rot_angles, floor, maxfev=5000)

        def fitted_model(theta):
            popt, pcov = curve_fit(objective, rot_angles, floor, maxfev=5000)
            a = popt[0]
            phi = popt[1]
            return np.sqrt((2*a**2)*(1-np.cos(theta+phi)))

        min = fmin(fitted_model, 0) # this is our rotation angle
        rotation_angles.append(min)

        # rotation matrices
        # how to get rotation matrices: https://download.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        rot_matrices.append(np.array(([[cos(min), -sin(min), 0],
                                       [sin(min), cos(min), 0],
                                       [0, 0, 1]])))

    for i in range(len(rot_matrices)):
        print('Rotation matrix for Floor {}'.format(i+1))
        print(rot_matrices[i])

    # NEW METHOD WHERE ROTATION IS ABOUT BOTTOM LEFT OF IMAGE

    new_trans_floor_points = copy.deepcopy(floor_points)
    new_trans_amts = []
    new_trans_matrices = []

    for i, f in enumerate(new_trans_floor_points):
        # rotate the points first
        rot = rotation_angles[i]
        for j, point in enumerate(f):
            xr, yr = rotation(rot, point[0], point[1])
            new_trans_floor_points[i][j] = [xr, yr]

    for f in new_trans_floor_points:
        # use new rotated points to get translation amounts
        new_trans_x = f[0][0]
        new_trans_y = f[0][1]
        
        new_trans_amts.append([-new_trans_x, -new_trans_y])

        new_trans_matrices.append(np.array(([[new_trans_x],
                                             [new_trans_y],
                                             [0]])))

    transforms_dict = {}
    transforms_dict['translation'] = new_trans_matrices
    transforms_dict['rotations'] = rot_matrices

    with open('transformations2.pickle', 'wb') as handle:
        pickle.dump(transforms_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('transformations2.txt', 'w') as f:
        lines = []
        for i, j in zip(range(len(new_trans_matrices)),range(len(rot_matrices))):
            lines.append('Angle of rot. about x-axis (Floor {}): {}'.format(i+1, 0))
            lines.append('Angle of rot. about y-axis (Floor {}): {}'.format(i+1, 0))
            lines.append('Angle of rot. about z-axis (Floor {}): {}'.format(i+1, rotation_angles[i]))
            lines.append('Translation Matrix of Floor {}'.format(i+1))
            lines.append(str(new_trans_matrices[i]))
            lines.append('Rotation Matrix of Floor {}'.format(j+1))
            lines.append(str(rot_matrices[j]))

                
        f.write('\n'.join(lines))

    for i in range(len(new_trans_matrices)):
        print('Translation Matrix for Floor {}'.format(i+1))
        print(new_trans_matrices[i])

    # plotting

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(NO_FLOORS): # set 2 for faster load
        x, y = np.mgrid[0:MAP_IMAGES[i].shape[0], 0:MAP_IMAGES[i].shape[1]]
        z = np.atleast_2d(i*10)
        
        xr, yr = rotation(rotation_angles[i], x, y)
        xtr, ytr = translation(new_trans_amts[i][0], new_trans_amts[i][1], xr, yr)

        ax.plot_surface(xtr, ytr, z, facecolors=MAP_IMAGES[i], shade=False, rstride=20, cstride=20)

    plt.show()

