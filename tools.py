import os
import numpy as np
from PIL import Image

def list_images(path, extension='.jpg'):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]
    
def load_image( im_name ):
    return np.array( Image.open( im_name ).convert( 'L' ) )