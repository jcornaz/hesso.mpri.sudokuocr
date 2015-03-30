import numpy as np
from skimage.filters import median, threshold_otsu, threshold_yen
from skimage.morphology import disk, erosion

from tools import list_images, load_image
from skimage.transform import resize
from skimage.feature import hog

i = 0

def extract_features( im ):
    """ Returns a feature vector for an image patch. """
    hist = hog( im, orientations=8, pixels_per_cell=(5,5), cells_per_block=(1, 1) )
    flat = im.flatten()
    mean = np.mean( im )
    std = np.std( im )
    return np.concatenate( (hist, flat, [mean, std, len(im >= mean)] ) )


def process_image(im, border_size=10, im_size=50):
    """ Remove borders and resize """

    im = im[border_size:-border_size, border_size:-border_size]
    
    selem = disk( 1 )
    
    im = median( im, selem )
    
    t = threshold_yen( im )
    im = im > t
    
    im = erosion( im, selem )
    
    return resize(im, (im_size, im_size))


def load_data(path):
    """ Return labels and features for all jpg images in path. """

    # Create a list of all files ending in .jpg
    im_list = list_images(path, '.jpg')

    # Create labels
    labels = [int(im_name.split('/')[-1][0]) for im_name in im_list]

    # Create features from the images
    features = []
    for im_name in im_list:
        im = load_image( im_name )
        im = process_image( im )
        features.append( extract_features( im ) )
        
    return features, labels