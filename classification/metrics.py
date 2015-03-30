from sklearn.metrics.metrics import confusion_matrix, classification_report
import pylab as pl


def show_confusion_matrix(y_true, y_predicted, title=''):
    """
    Plot (and print) a confusion matrix from y_true and y_predicted
    """
    confmat = confusion_matrix( y_true, y_predicted )
    pl.figure(figsize=(10, 10))
    
    ax = pl.subplot(111)
    cmim = ax.matshow( confmat, interpolation='nearest' )
    
    for i in xrange(confmat.shape[0]):
        for j in xrange(confmat.shape[1]):
            ax.annotate(str(confmat[i, j]), xy=(j, i),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=8)

    ax.set_title( title )
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    pl.colorbar(cmim, shrink=0.7, orientation='horizontal', pad=0.01)

def print_classification_report(y_true, y_pred, title=''):
    """
    Print a classification report
    """
    print title
    print classification_report( y_true, y_pred )