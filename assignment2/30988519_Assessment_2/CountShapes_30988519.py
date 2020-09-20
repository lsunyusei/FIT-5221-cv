import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, draw, transform, feature, filters
from skimage import io
from skimage.color import *


def CountShapes_30988519(img):
    im = rgb2gray(img)
    image = filters.gaussian(im)
    edges = feature.canny(image, sigma=3.0)  # detect canny edge
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 15))
    ax0.imshow(edges, cmap=plt.cm.gray)  # show canny edge
    ax0.set_title('edge image')

    hough_radii = np.arange(15, 120, 2)  # Radius range
    hough_res = transform.hough_circle(edges,
                                       hough_radii)  # hough_res：3D array #radius,width_idx and height_idx of circle's center
    centers = []  # Store center point coordinates
    accums = []  # it is the prop
    radii = []  # radius

    for radius, h in zip(hough_radii, hough_res):
        peaks = feature.peak_local_max(h, threshold_abs=0.4)  # peak is a coordinate, namely the set of center
        centers.extend(peaks)  # add the center ; size: num_peaks per radius
        accums.extend(h[peaks[:, 0], peaks[:, 1]])  # size: num_peaks per radius
        radii.extend([radius] * len(peaks))  # size: num_peaks same radius per radius

    count = 0
    for idx in np.argsort(accums)[::-1][:]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = draw.circle_perimeter(center_y, center_x, radius)  # the lacation of pixels in the circle

        img[cy, cx] = (255, 255, 0)
        count += 1
    print("the number of objects is : ", count)
    ax1.imshow(img)
    plt.show()
    ax1.set_title('detected image')
    io.imsave('./out/CountShapes.jpg', img)


if __name__ == '__main__':
    # Read image
    img = io.imread("./test/ball-bearings.jpg")
    CountShapes_30988519(img)