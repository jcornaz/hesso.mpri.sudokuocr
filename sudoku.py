from classification.metrics import show_confusion_matrix, print_classification_report
from classification.svm import load_or_train
from image.cell_extraction import extract_cells
import numpy as np
from image.feature_extraction import extract_features
from tools import load_image

# Choose a sudoku grid number, and prepare paths (image and verification grid)
sudoku_nb = 3
im_path = './data/sudokus/sudoku{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)

# Get trained classifier
clf = load_or_train()

# Load sudoku image
img = load_image( im_path )

# Extract cells
cells = extract_cells( img )

# Add data for each cell
features = []
for cell in cells:
    features.append( extract_features( cell ) )

# Classification
predicted = clf.predict( features )

# Load solution to compare with, print metrics, and print confusion matrix
y_sudoku = np.loadtxt(ver_path).reshape(81)
print_classification_report( y_sudoku, predicted, 'Classification report for \'' + im_path + '\'')
show_confusion_matrix( y_sudoku, predicted, 'Confusion matrix for \'' + im_path + '\'')

# Print resulting sudoku
print predicted.reshape((9,9))
