import re
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import Polygon
from scipy.spatial import distance

__author__ = "Lygerakis Fotios"
__license__ = "GPL"
__email__ = "ligerfotis@gmail.com"

title = "Index-Thumb Groundtruth"

plt.rcParams.update({'font.size': 16})
# box plot aperture set
top = 5
bottom = 0

# plot the average every frameAvg points
frameAvg = 1

# 1 is true
verbose = 0

# steps for tick printing on the axes
step_x = .01
step_y = .01
step_z = .01

# used in boxplot | positioned under each box
aperture_size = ['1cm', '2cm', '3cm', '4cm', '5cm']
msgsPerFile = ["291", "288", "290", "289", "291"]
img_raw_msgs = ["178", "176", "206", "200"]
pc_msgs = ["180", "174", "204", "198"]

# the z score value used to filter outliers
zscore = 2.698

x_points = []
y_points = []
z_points = []


# Function to extract all the distinct keypoints from a csv
def getKeyPoints(columns):
    columns = "".join(columns)
    array = re.findall(r"/keypoint_3d_matching/keypoints/[0-9]+/points/point/x", columns)
    return array


# Function to extract all the distinct keypoint names from a csv
def getKeypointNames(df):
    listOfNames = []
    for col in df.columns[1:]:
        if "name" in col:
            listOfNames.append(df[col].loc[0])
    return listOfNames


def get_cmap():
    return cycle('brgcmk')


def addPoint(scat, new_point, c='k'):
    old_off = scat.get_offsets()
    new_off = np.concatenate([old_off, np.array(new_point, ndmin=2)])
    old_c = scat.get_facecolors()
    new_c = np.concatenate([old_c, np.array(colors.to_rgba(c), ndmin=2)])

    scat.set_offsets(new_off)
    scat.set_facecolors(new_c)

    scat.axes.figure.canvas.draw_idle()


def keypointExtractor(filename, withOutliers=False):
    df = pd.read_csv(filename)

    listOfNames = getKeypointNames(df)
    numOfKeyPoints = len(getKeyPoints(df.columns))

    listOfKeyPoints_x = ["/keypoint_3d_matching/keypoints/" + str(i) + "/points/point/x" for i in range(numOfKeyPoints)]
    listOfKeyPoints_y = ["/keypoint_3d_matching/keypoints/" + str(i) + "/points/point/y" for i in range(numOfKeyPoints)]
    listOfKeyPoints_z = ["/keypoint_3d_matching/keypoints/" + str(i) + "/points/point/z" for i in range(numOfKeyPoints)]

    new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z = [], [], []
    new_listOfNames = []
    # create new dataframe containing only columns with point information
    new_df = pd.DataFrame()
    for i, point_list_names in enumerate(zip(listOfKeyPoints_x, listOfKeyPoints_y, listOfKeyPoints_z)):
        # check if keypoint contains mostly zero points
        count_valid_array = [(df[col_point] != 0).sum() for col_point in point_list_names]

        if all([point > numOfKeyPoints / 4 for point in count_valid_array]):
            new_listOfKeyPoints_x.append(point_list_names[0])
            new_listOfKeyPoints_y.append(point_list_names[1])
            new_listOfKeyPoints_z.append(point_list_names[2])
            new_listOfNames.append(listOfNames[i])

            for col in point_list_names:
                # rounding is crucial for comparing
                new_df[col] = df[col].astype(float).round(7)

    # replace zero values with NAN so they do not plot
    new_df.replace(0, np.nan, inplace=True)

    # # remove outliers
    # if not withOutliers:
    #     new_df = new_df[(np.abs(stats.zscore(new_df)) < zscore).all(axis=1)]

    return new_df, listOfKeyPoints_x, listOfKeyPoints_y, listOfKeyPoints_z, new_listOfNames


def new_keypointExtractor(filename, withOutliers=False):
    df = pd.read_csv(filename)

    listOfNames = getKeypointNames(df)
    numOfKeyPoints = len(getKeyPoints(df.columns))
    point_list_names = [
        ["/keypoint_3d_matching/keypoints/" + str(i) + "/points/point/x" for i in range(numOfKeyPoints)],
        ["/keypoint_3d_matching/keypoints/" + str(i) + "/points/point/y" for i in range(numOfKeyPoints)],
        ["/keypoint_3d_matching/keypoints/" + str(i) + "/points/point/z" for i in range(numOfKeyPoints)]]

    new_point_list_names = [
        ["keypoint_" + str(i) + "_x" for i in range(numOfKeyPoints)],
        ["keypoint_" + str(i) + "_y" for i in range(numOfKeyPoints)],
        ["keypoint_" + str(i) + "_z" for i in range(numOfKeyPoints)]]

    # create new dataframe containing only columns with point information
    new_df = pd.DataFrame()
    for i, (names, new_names) in enumerate(zip(point_list_names, new_point_list_names)):
        # check if keypoint contains mostly zero points
        count_valid_array = [(df[col_point] != 0).sum() for col_point in names]

        if all([point > numOfKeyPoints / 4 for point in count_valid_array]):

            for col, new_col in zip(point_list_names, new_point_list_names):
                # rounding is crucial for comparing
                new_df[new_col] = df[col].astype(float).round(7)

    # replace zero values with NAN so they do not plot
    new_df.replace(0, np.nan, inplace=True)

    # # remove outliers
    # if not withOutliers:
    #     new_df = new_df[(np.abs(stats.zscore(new_df)) < zscore).all(axis=1)]

    return new_df


def extractAperture(filename):
    df, new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z, _ = keypointExtractor(filename,
                                                                                                   withOutliers=True)

    # calculate aperture
    aperture = []

    # 2 lists the 2 fingers
    finger1 = np.asarray(
        [df[new_listOfKeyPoints_x[0]].values, df[new_listOfKeyPoints_y[0]].values, df[new_listOfKeyPoints_z[0]].values])
    finger2 = [df[new_listOfKeyPoints_x[1]].values, df[new_listOfKeyPoints_y[1]].values,
               df[new_listOfKeyPoints_z[1]].values]

    avgfinger1 = getAvg(finger1[0], finger1[1], finger1[2], frameAvg)
    avgfinger2 = getAvg(finger2[0], finger2[1], finger2[2], frameAvg)

    avgfinger1 = np.transpose(avgfinger1)
    avgfinger2 = np.transpose(avgfinger2)

    # for keypoint_x, keypoint_y, keypoint_z in zip(new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z):
    # 	fingers.append([df[keypoint_x].values, df[keypoint_y].values, df[keypoint_z].values])
    # susbtraction_x = point_list[0][0] - point_list[1][0]
    # substraction_y = point_list[0][1] - point_list[1][1]

    # euclidean distance
    # aperture = np.sqrt(np.add(np.power(susbtraction_x, 2), np.power(substraction_y,2)))
    # aperture = aperture[~np.isnan(aperture)]
    for point_f1, point_f2 in zip(avgfinger1, avgfinger2):
        aperture.append(distance.euclidean(point_f1, point_f2))

    # print(len(aperture))
    aperture = np.multiply(aperture, 1)
    # print(len(aperture))

    # aperture = aperture[~np.isnan(aperture)]

    return aperture


def boxPlot(filenames):
    apertures = [extractAperture(filename) for filename in filenames]

    data_cm = [aperture * 100 for aperture in apertures]

    labels = [ap + "\nMessages:\n" + msg for ap, msg in zip(aperture_size, msgsPerFile)]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.set_window_title('Aperture_Thumb_Fingertips_avg' + str(frameAvg))

    medianprops = dict(linestyle=None, linewidth=0, color='white')
    bp = ax.boxplot(data_cm, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, meanline=True, medianprops=medianprops)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.5)
    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title('Comparison of Aperture Discritability')
    ax.set_xlabel('Distripution')
    ax.set_ylabel('Aperture in cm')

    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data_cm)
    means = [np.mean(keypoint) for keypoint in data_cm]
    stds = [np.std(keypoint) for keypoint in data_cm]
    print("means")
    print(means)
    print("stdevs")
    print(stds)

    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # Set the axes ranges and axes labels
    ax.set_xlim(0.5, num_boxes + 0.5)

    plt.yticks(np.arange(bottom, top, 0.2))
    ax.set_xticklabels(labels, fontsize=18)

    plt.show()


def print2dPlot(axis1, axis2, new_df, new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z,
                new_listOfNames):
    fig, ax = plt.subplots(figsize=(10, 6))
    # gt_index_name = [k for k, v in globals().items() if v == gt_point_index][0]
    # gt_thumb_name = [k for k, v in globals().items() if v == gt_point_thumb][0]

    # fig.canvas.set_window_title(gt_index_name + gt_thumb_name + "_" + axis1 + axis2)

    cmap = get_cmap()
    scat = None
    color = 'brgcmk'
    point_list1 = []
    point_list2 = []

    if [axis1, axis2] == ['x', 'y']:
        new_listOfKeyPoints_ax1 = new_listOfKeyPoints_x
        new_listOfKeyPoints_ax2 = new_listOfKeyPoints_y
    elif [axis1, axis2] == ['x', 'z']:
        new_listOfKeyPoints_ax1 = new_listOfKeyPoints_x
        new_listOfKeyPoints_ax2 = new_listOfKeyPoints_z

    elif [axis1, axis2] == ['y', 'z']:
        new_listOfKeyPoints_ax1 = new_listOfKeyPoints_y
        new_listOfKeyPoints_ax2 = new_listOfKeyPoints_z

    for i, (list_ax1, list_ax2) in enumerate(zip(new_listOfKeyPoints_ax1, new_listOfKeyPoints_ax2)):
        point_list1.append(new_df[list_ax1])
        point_list2.append(new_df[list_ax2])
        avglist1, avglist2 = getAvg(array1=new_df[list_ax1], array2=new_df[list_ax2], avgNum=frameAvg)
        scat = ax.scatter(avglist1, avglist2, s=0.7, color=color[i], label=new_listOfNames[i])

    # fig.suptitle(title)
    ax.set_xlabel(axis1 + '-axis (in m)')
    ax.set_ylabel(axis2 + '-axis (in m)')

    # if ([axis1, axis2] == ['x', 'y']):
    #     plt.xticks(np.arange(axis_range_x[0], axis_range_x[1], step=step_x))
    #     plt.yticks(np.arange(axis_range_y[0], axis_range_y[1], step=step_y))
    #     # ax.set_xlim(axis_range_x)
    #     # ax.set_ylim(axis_range_y)
    #
    #     gt_points_ax1 = [gt_point_index[0], gt_point_thumb[0]]
    #     gt_points_ax2 = [gt_point_index[1], gt_point_thumb[1]]
    #
    # elif ([axis1, axis2] == ['x', 'z']):
    #     plt.xticks(np.arange(axis_range_x[0], axis_range_x[1], step=step_x))
    #     plt.yticks(np.arange(axis_range_z[0], axis_range_z[1], step=step_z))
    #     ax.set_xlim(axis_range_x)
    #     ax.set_ylim(axis_range_z)
    #
    #     # get two gaussian random numbers, mean=0, std=1, 2 numbers
    #     gt_points_ax1 = [gt_point_index[0], gt_point_thumb[0]]
    #     gt_points_ax2 = [gt_point_index[2], gt_point_thumb[2]]
    #
    # elif ([axis1, axis2] == ['y', 'z']):
    #     plt.xticks(np.arange(axis_range_y[0], axis_range_y[1], step=step_y))
    #     plt.yticks(np.arange(axis_range_z[0], axis_range_z[1], step=step_z))
    #     ax.set_xlim(axis_range_y)
    #     ax.set_ylim(axis_range_z)
    #
    #     # get two gaussian random numbers, mean=0, std=1, 2 numbers
    #     gt_points_ax1 = [gt_point_index[1], gt_point_thumb[1]]
    #     gt_points_ax2 = [gt_point_index[2], gt_point_thumb[2]]

    mean_ax1 = np.mean(new_df[new_listOfKeyPoints_ax1])
    mean_ax2 = np.mean(new_df[new_listOfKeyPoints_ax2])
    std_ax1 = np.std(new_df[new_listOfKeyPoints_ax1])
    std_ax2 = np.std(new_df[new_listOfKeyPoints_ax2])

    print('index point mean ' + axis1 + '-' + axis2 + ':' + str(mean_ax1[0]) + ',' + str(mean_ax2[0]))
    print('index point sted ' + axis1 + '-' + axis2 + ':' + str(std_ax1[0]) + ',' + str(std_ax2[0]))

    print('thumb point mean ' + axis1 + '-' + axis2 + ':' + str(mean_ax1[1]) + ',' + str(mean_ax2[1]))
    print('thumb point sted ' + axis1 + '-' + axis2 + ':' + str(std_ax1[1]) + ',' + str(std_ax2[1]))

    print('thumb point mean of ' + axis1 + '-' + axis2 + ':' + str(mean_ax1[0]) + ',' + str(mean_ax2[0]))
    print('thumb point sted of ' + axis1 + '-' + axis2 + ':' + str(std_ax1[0]) + ',' + str(std_ax2[0]))

    print('index point mean of' + axis1 + '-' + axis2 + ':' + str(mean_ax1[1]) + ',' + str(mean_ax2[1]))
    print('index point sted of' + axis1 + '-' + axis2 + ':' + str(std_ax1[1]) + ',' + str(std_ax2[1]))
    # print('Mean and stdev for axis:' + axis2 + 'is: (' + mean_ax2 + ', ' + std_ax2)

    # plot ground truth points
    # plt.scatter(gt_points_ax1, gt_points_ax2, s=30, color="green", label="Groundtruth")

    # ax.annotate("Ground Truth \nThumb Tip", (gt_points_ax1[0], gt_points_ax2[0]))
    # ax.annotate("Ground Truth \nIndex Tip", (gt_points_ax1[1], gt_points_ax2[1]))
    if verbose == 1:
        plt.errorbar(mean_ax1, mean_ax2, markersize=2, fmt='o', color='black', ecolor='black', ms=20, mfc='white',
                     label="Mean")

    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    if verbose == 1:
        ax.legend(markerscale=5)
    plt.show()

    return point_list1, point_list2


"""
get an average of the point over avgNum frames
"""


def getAvg(array1, array2, array3=None, avgNum=2):
    new_array1 = []
    new_array2 = []
    new_array3 = []

    for i in range(0, len(array1) - avgNum):

        step = i + avgNum
        new_array1.append(np.mean(array1[i:step]))
        new_array2.append(np.mean(array2[i:step]))

        if array3 is not None:
            new_array3.append(np.mean(array3[i:step]))

    if array3 is None:
        return new_array1, new_array2
    else:
        return new_array1, new_array2, new_array3


def plot(filename):
    new_df, new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z, new_listOfNames = keypointExtractor(
        filename, withOutliers=False)

    x_points, y_points = print2dPlot('x', 'y', new_df, new_listOfKeyPoints_x, new_listOfKeyPoints_y,
                                     new_listOfKeyPoints_z, new_listOfNames)
    _, z_points = print2dPlot('x', 'z', new_df, new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z,
                              new_listOfNames)
    print2dPlot('y', 'z', new_df, new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z, new_listOfNames)

    # dist_t = []
    # dist_i = []
    # # index
    # for point in zip(x_points[0], y_points[0], z_points[0]):
    #     dist_i.append(distance.euclidean(point, gt_point_index))
    # # thumb
    # for point in zip(x_points[1], y_points[1], z_points[1]):
    #     dist_t.append(distance.euclidean(point, gt_point_thumb))


#
# print("Mean euclidean distance for thumb")
# print(np.mean(dist_t))
#
# print("Mean euclidean distance for index")
# print(np.mean(dist_i))


def plot3D(filename):
    color = 'brgcmk'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_list1 = []
    point_list2 = []
    point_list3 = []
    new_df, new_listOfKeyPoints_x, new_listOfKeyPoints_y, new_listOfKeyPoints_z, new_listOfNames = keypointExtractor(
        filename, withOutliers=False)
    # ax.scatter(avglist1, avglist2, s=0.7, color=color[i], label=new_listOfNames[i])
    for i, (list_ax1, list_ax2, list_ax3) in enumerate(zip(new_listOfKeyPoints_x, new_listOfKeyPoints_y,
                                                           new_listOfKeyPoints_z)):
        point_list1.append(new_df[list_ax1])
        point_list2.append(new_df[list_ax2])
        point_list3.append(new_df[list_ax3])

        ax.scatter(new_df[list_ax1], new_df[list_ax2], new_df[list_ax3], s=0.7, color=color[i],
                   label=new_listOfNames[i])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
