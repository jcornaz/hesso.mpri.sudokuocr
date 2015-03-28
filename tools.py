import os


def list_images(path, extension='.jpg'):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]