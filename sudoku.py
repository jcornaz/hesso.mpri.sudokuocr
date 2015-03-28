from classification.metrics import show_confusion_matrix, print_classification_report
from classification.svm import load_or_train
from image.cell_extraction import extract_cells, plot_extracted_cells
from PIL import Image
import numpy as np
from image.feature_extraction import extract_features

# Choose a sudoku grid number, and prepare paths (image and verification grid)
sudoku_nb = 18
im_path = './data/sudokus/sudoku{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)

# Get trained classifier
# TODO

# Load sudoku image
# TODO: load the sudoku image as a gray level image

# Extract cells
# TODO

# Add data for each cell
# TODO: iterate over cells and append features to a list

# Classification
# TODO: use the classifier to predict on the list of features

# Load solution to compare with, print metrics, and print confusion matrix
y_sudoku = np.loadtxt(ver_path).reshape(81)
# TODO: print classification report
# TODO: show confusion matrix

# Print resulting sudoku
# TODO: print the resulting sudoku grid (use reshape() function to get a 9x9 grid print!
