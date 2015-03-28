import os
import pickle
from sklearn import svm, cross_validation
from classification.metrics import show_confusion_matrix, print_classification_report
from image.feature_extraction import load_data


def train(clf, X_train, y_train):
    """ Train and return an SVM classifier """

    # TODO: train the classifier

    # TODO: return the classifier
    return None


def load_or_train(force_train=False):
    """
    Load an existing one or train a new SVM classifier, and return it.
    Once the classifier is trained, it is saved through pickle.
    """

    clf_path='./clf.pkl'
    data_path = "././data/ocr_data/"

    clf = None

    if not force_train and os.path.exists(clf_path):
        clf = pickle.load(open(clf_path, 'rb'))
    else:
        # Loading all data
        X, y = load_data(data_path)

        # Instantiate a classifier
        # TODO: instantiate a new classifier (choose an adapted kernel!)

        # Cross validation
        # TODO: do cross-validation and print cross-validation result (mean accuracy +/- standard deviation)

        # Train the classifier on the whole dataset, and save it
        # TODO: train the classifier on the whole dataset
        pickle.dump(clf, open(clf_path, 'wb'))

        # If you want, you can do validation, print classification report and show confusion matrix with this
        # trained classifier. But keep in mind that you will do it on the training set itself!

    # TODO: return the classifier
    return None