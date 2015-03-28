import numpy as np
import pylab as pl
from skimage import transform as tf
from skimage.util.dtype import convert
from image.feature_extraction import process_image


def transform_from_corners(im):
    """ Allow the user to click on the sudoku grid corners, and then crop and warp it to get a squared grid """

    # Mark corners (1. Top left, 2. Bottom left, 3. Bottom right, 4. Top right)
    pl.figure()
    pl.title("Click on corners (1. Top left, 2. Bottom left, 3. Bottom right, 4. Top right)")
    pl.imshow(im)
    pl.gray()
    x = pl.ginput(4) # Wait on user input clicks
    pl.close()

    # Top left, top right, bottom right, bottom left
    fp = np.array([np.array([p[0], p[1]]) for p in x])
    dim = 1000 # New size for the croped grid
    tp = np.array([[0, 0], [0, dim], [dim, dim], [dim, 0]])

    # Compute projective transform, and apply it through warping
    tform = tf.ProjectiveTransform()
    tform.estimate(tp, fp)
    output_shape = (dim, dim)
    # TODO: use scikit-image warp function to apply the transformation
    warped = None

    # Convert result in the right format [0, 255]
    im = convert(warped, np.uint8)
    return im


def plot_extracted_cells(cells):
    """ Plot all extracted cells as a 9x9 grid """
    fig = pl.figure(frameon=False)

    for i, cell in enumerate(cells):
        ax = fig.add_subplot(9, 9, i)
        ax.xaxis.set_ticklabels([None])
        ax.yaxis.set_ticklabels([None])
        ax.xaxis.set_ticks([None])
        ax.yaxis.set_ticks([None])
        ax.imshow(cell)

    pl.gray()
    pl.show()


def extract_cells(im):
    """
    For a sudoku image, crop on the grid, and extract cells by dividing the grid in 9x9 cells, and store them in
    a list of 81 cells.
    """

    cells = []

    # Crop on the grid and transform (warping) to get a squared grid
    im = transform_from_corners(im)

    # Plot the transformed grid
    pl.figure()
    pl.title("After transformation")
    pl.imshow(im)
    pl.gray()
    pl.show()

    # Get dimensions (x, y) of the image, and compute dimensions of one single cell
    (dim_x, dim_y) = im.shape
    div_x = int(dim_x / 9)
    div_y = int(dim_y / 9)

    # Iterate and crop the cells
    for row in range(9):
        for col in range(9):
            crop = im[row*div_x:(row+1)*div_x, col*div_y:(col+1)*div_y]
            # TODO: process the image to remove borders and resize
            cells.append(crop)

    # Plot the extracted cells
    plot_extracted_cells(cells)

    return cells