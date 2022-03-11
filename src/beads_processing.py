from numpy import (uint8 as np_uint8, ones as np_ones, array as np_array, zeros as np_zeros, uint16 as np_uint16,
                   sum as np_sum, round as np_round, where as np_where, dstack as np_dstack, delete as np_delete,
                   bincount as np_bincount, zeros_like as np_zeros_like)
from pandas import (DataFrame as pd_DataFrame, read_csv as pd_read_csv)
from queue import Queue as queue_Queue
from cv2 import (cvtColor as cv2_cvtColor, COLOR_GRAY2RGB as cv2_COLOR_GRAY2RGB, threshold as cv2_threshold,
                 THRESH_BINARY as cv2_THRESH_BINARY, THRESH_OTSU as cv2_THRESH_OTSU, morphologyEx as cv2_morphologyEx,
                 connectedComponents as cv2_connectedComponents, RETR_LIST as cv2_RETR_LIST, watershed as cv2_watershed,
                 MORPH_OPEN as cv2_MORPH_OPEN, minEnclosingCircle as cv2_minEnclosingCircle,
                 CHAIN_APPROX_NONE as cv2_CHAIN_APPROX_NONE, findContours as cv2_findContours, imread as cv2_imread,
                 boundingRect as cv2_boundingRect
                 )
from skimage.measure import regionprops_table as measure_regionprops_table

from scipy.ndimage.measurements import center_of_mass as ndi_center_of_mass
from math import pi as math_pi
from matplotlib.pyplot import (Rectangle as plt_Rectangle, Circle as plt_Circle)

from sklearn.linear_model import LinearRegression


def train_beads(beads_path, df_beads):
    """
    Read beads information from csv file or beads dataFrame
    :param df_beads: beads dataframe
    :param beads_path:
    :return:
    """
    if len(beads_path) > 0:
        try:
            df_beads = pd_read_csv(beads_path, names=['red_x', 'red_y', 'green_x', 'green_y', 'blue_x', 'blue_y'])
        except Exception as e:
            raise e
    # green channel
    X_green = np_array(df_beads.loc[:, ['green_x', 'green_y']])
    Y_x_green = np_array(df_beads['red_x'] - df_beads['green_x'])
    Y_y_green = np_array(df_beads['red_y'] - df_beads['green_y'])

    lr_x_green = LinearRegression()
    lr_x_green.fit(X_green, Y_x_green)

    lr_y_green = LinearRegression()
    lr_y_green.fit(X_green, Y_y_green)

    pred_x_green = lr_x_green.predict(X_green)
    pred_y_green = lr_y_green.predict(X_green)

    # blue channel
    X_blue = np_array(df_beads.loc[:, ['blue_x', 'blue_y']])
    Y_x_blue = np_array(df_beads['red_x'] - df_beads['blue_x'])
    Y_y_blue = np_array(df_beads['red_y'] - df_beads['blue_y'])

    lr_x_blue = LinearRegression()
    lr_x_blue.fit(X_blue, Y_x_blue)

    lr_y_blue = LinearRegression()
    lr_y_blue.fit(X_blue, Y_y_blue)

    pred_x_blue = lr_x_blue.predict(X_blue)
    pred_y_blue = lr_y_blue.predict(X_blue)

    pred_beads = pd_DataFrame({'red_y': df_beads['red_y'], 'red_x': df_beads['red_x'],
                               'green_y': df_beads['green_y'] + pred_y_green,
                               'green_x': df_beads['green_x'] + pred_x_green,
                               'blue_y': df_beads['blue_y'] + pred_y_blue,
                               'blue_x': df_beads['blue_x'] + pred_x_blue})
    return lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, df_beads, pred_beads


def sub_mean(img):
    bead_mean = int(img.mean() + img.std())
    bead_bgst = np_zeros_like(img)
    for i, row in enumerate(img):
        for j, x in enumerate(row):
            if x < bead_mean:
                bead_bgst[i][j] = 0
            else:
                bead_bgst[i][j] = x - bead_mean
    return bead_bgst


def get_opening(bead_bgst):
    bead_bgst_c3 = cv2_cvtColor(bead_bgst, cv2_COLOR_GRAY2RGB)
    bead_bgst_float = bead_bgst / (bead_bgst + 1) * 255
    bead_bgst_uint8 = bead_bgst_float.astype(np_uint8)
    ret1, thresh = cv2_threshold(bead_bgst_uint8, 0, 255, cv2_THRESH_BINARY + cv2_THRESH_OTSU)
    kernel = np_ones((2, 2), np_uint8)
    opening = cv2_morphologyEx(thresh, cv2_MORPH_OPEN, kernel, iterations=2)
    # plt.figure(figsize=(15, 15))
    # imshow(opening)
    # plt.show()
    return bead_bgst_uint8, bead_bgst_c3, opening


shift_list = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, -1], [-1, 1], [1, 1], [1, -1]]
check_list = [[-1, 0], [0, 1], [1, 0], [0, -1]]


def find_contours(img_labeled, con_queue):
    # finish
    if con_queue.empty():
        return [-1, -1]
    point = con_queue.get()
    x_max, y_max, _ = img_labeled.shape
    x_min, y_min = (0, 0)

    x, y = point
    # check if contours is close to the edge of image
    if x == x_min or y == y_min or x == x_max or y == y_max:
        return [-2, -2]

    # check if point has been regarded as a contour
    if img_labeled[x, y, 3] == 1:
        return [0, 0]
    for shift in shift_list:
        x, y = [a + b for a, b in zip(shift, point)]
        if x < 0 or y < 0 or x == x_max or y == y_max:
            return [-2, -2]
        # check if point is in the img
        if np_sum(img_labeled[x, y, :3]) == 0:
            continue
        else:
            if check_contours(x_max, y_max, img_labeled, [x, y]):
                con_queue.put([x, y])
    return point


def check_contours(x_max, y_max, img_labeled, point):
    x, y = point
    if img_labeled[x, y, 3] != 0:
        return False
    for shift in check_list:
        x, y = [a + b for a, b in zip(shift, point)]
        if x < 0 or y < 0 or x == x_max or y == y_max:
            return False
        if np_sum(img_labeled[x, y, :3]) == 0:
            return True
    return False


def find_bead_contours(composited_img, coord_array):
    x_max, y_max, _ = composited_img.shape
    x_min = 0
    y_min = 0
    composted_labeled = np_zeros((x_max, y_max, 4), np_uint16)
    composted_labeled[:, :, :3] = composited_img
    counter_lists = []
    counter_list = []
    for i, coords in enumerate(coord_array):
        contours_queue = queue_Queue()
        counter_list = []
        x, y = (0, 0)
        for _, coord in enumerate(coords):
            x, y = coord
            if np_sum(composited_img[x][y]) == 0:
                continue
            else:
                break
        # step1: find most top contour point
        while True:
            up = x - 1
            if up > x_min:
                if np_sum(composited_img[up][y]) > 0:
                    x = up
                else:
                    break
            else:
                break
        start = [x, y]
        contours_queue.put(start)
        # step 3: find contour
        while True:
            x, y = find_contours(composted_labeled, contours_queue)
            if x == -1:
                break
            elif x == -2:
                counter_list = None
                break
            elif x == 0:
                continue
            else:
                counter_list.append([x, y])
                composted_labeled[x, y, 3] = 1
        # step 4: finish one golgi
        if counter_list is not None:
            counter_lists.append(counter_list)
    return counter_lists


def clean_contours(img, contours):
    h, w, _ = img.shape
    for x in range(h):
        y_array = np_where(contours[:, 0] == x)[0]
        y_min = w
        y_max = 0
        if len(y_array) > 0:
            y_min = contours[y_array][:, 1].min()
            y_max = contours[y_array][:, 1].max()
        for y in range(0, y_min):
            img[x, y] = [0, 0, 0]
        for y in range(y_max + 1, w):
            img[x, y] = [0, 0, 0]
    return img, True


def adjust_filter_bead_c1(bead):
    f = 0
    if np_sum(bead) == 0:
        return f
    if np_sum(bead[0]) > 0 or np_sum(bead[:, 0]) > 0 or np_sum(bead[-1]) > 0 or np_sum(bead[:, -1]) > 0:
        return f

    contours = cv2_findContours(bead, cv2_RETR_LIST, cv2_CHAIN_APPROX_NONE)
    if len(contours[0]) > 1:
        return f
    else:
        contours = contours[0][0].reshape(-1, 2)

    remove_index = set()

    b_count = np_bincount(contours[:, 1])
    b_index = np_where(b_count == 1)
    for index in b_index[0]:
        selected_index = np_where(contours[:, 1] == index)[0]
        for i in selected_index:
            remove_index.add(i)

    b_count = np_bincount(contours[:, 0])
    b_index = np_where(b_count == 1)
    for index in b_index[0]:
        selected_index = list(np_where(contours[:, 0] == index)[0])
        for i in selected_index:
            remove_index.add(i)

    contours = np_delete(contours, list(remove_index), axis=0)
    x, y, w, h = cv2_boundingRect(contours)
    rectangle = plt_Rectangle(xy=(x - 0.5, y - 0.5), width=w, height=h, alpha=0.5)

    (x_r, y_r), radius = cv2_minEnclosingCircle(contours)
    center = (x_r, y_r)
    circle1 = plt_Circle(xy=center, radius=radius + 0.5 ** 0.5, alpha=0.5)

    area = len(np_where(bead[y:y + h, x:x + w] > 0)[0])
    bg_circle_area = (radius + 0.5) ** 2 * math_pi
    f = area / bg_circle_area
    return f


def get_beads_df(beads_path_list, bgst=True):
    if len(beads_path_list) != 3:
        raise Exception("Lack beads images. Found {} images, but required 3 images.".format(len(beads_path_list)))
    bead_path_r, bead_path_g, bead_path_b = beads_path_list
    bead_r = cv2_imread(bead_path_r, -1)
    bead_g = cv2_imread(bead_path_g, -1)
    bead_b = cv2_imread(bead_path_b, -1)
    if not bgst:
        bead_r_bgst = sub_mean(bead_r)
        bead_g_bgst = sub_mean(bead_g)
        bead_b_bgst = sub_mean(bead_b)
    else:
        bead_r_bgst = bead_r
        bead_g_bgst = bead_g
        bead_b_bgst = bead_b

    bead_r_uint8, bead_r_c3, bead_r_opening = get_opening(bead_r_bgst)
    bead_g_uint8, bead_g_c3, bead_g_opening = get_opening(bead_g_bgst)
    bead_b_uint8, bead_b_c3, bead_b_opening = get_opening(bead_b_bgst)

    bgst_composited_uint8_c3 = np_dstack((bead_r_uint8, bead_g_uint8, bead_b_uint8))

    h, w = bead_r_opening.shape
    composited_c1 = np_zeros_like(bead_r_opening)
    for i in range(h):
        for j in range(w):
            r, g, b = bgst_composited_uint8_c3[i, j, 0:3]
            if r > 0 and b > 0 and g > 0:
                composited_c1[i, j] = 1
    _, markers = cv2_connectedComponents(composited_c1)
    markers = markers + 10
    markers = cv2_watershed(bgst_composited_uint8_c3, markers)
    props = measure_regionprops_table(markers, intensity_image=composited_c1,
                                      properties=['label', 'coords', 'centroid',
                                                  'area', 'mean_intensity'])

    df_props = pd_DataFrame(props)
    df_props = df_props[df_props.area <= 60]
    df_props = df_props[df_props.area >= 20]
    df_props.sort_values(by=['area'], ascending=False, inplace=True)

    center_mass = []
    beads_coords = np_array(df_props['coords'])
    composited_c3_contours = find_bead_contours(bgst_composited_uint8_c3, beads_coords)

    for i, contour in enumerate(composited_c3_contours):
        contour_np = np_array(contour)
        if len(contour_np) == 0:
            continue
        x, y, w, h = cv2_boundingRect(contour_np)
        if x == 0 or y == 0:
            continue
        if abs(w - h) > 2:
            continue

        select_bead = bgst_composited_uint8_c3[x - 1:x + w + 1, y - 1:y + h + 1]

        new_contours = contour_np - [x - 1, y - 1]
        select_bead, _ = clean_contours(select_bead, new_contours)

        r_bead = select_bead[:, :, 0]
        g_bead = select_bead[:, :, 1]
        b_bead = select_bead[:, :, 2]
        f_r, f_g, f_b = adjust_filter_bead_c1(r_bead), adjust_filter_bead_c1(g_bead), adjust_filter_bead_c1(b_bead)
        if f_r * f_b * f_g > 0 and 0.7 < (f_r + f_b + f_g) / 3 < 1:
            center_r = np_round(ndi_center_of_mass(r_bead), 3) + [x - 1, y - 1] + [0.5, 0.5]
            center_g = np_round(ndi_center_of_mass(g_bead), 3) + [x - 1, y - 1] + [0.5, 0.5]
            center_b = np_round(ndi_center_of_mass(b_bead), 3) + [x - 1, y - 1] + [0.5, 0.5]
            center_mass.append([center_r, center_g, center_b])
    center_mass = np_array(center_mass)
    if center_mass.shape[0] == 0:
        raise Exception("Beads not found.")

    X_red = np_array(center_mass[:, 0, :])
    X_green = np_array(center_mass[:, 1, :])
    X_blue = np_array(center_mass[:, 2, :])
    beads_df = pd_DataFrame({'red_y': X_red[:, 1],
                             'red_x': X_red[:, 0],
                             'green_y': X_green[:, 1],
                             'green_x': X_green[:, 0],
                             'blue_y': X_blue[:, 1],
                             'blue_x': X_blue[:, 0]})
    return beads_df
