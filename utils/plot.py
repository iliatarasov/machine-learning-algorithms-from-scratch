import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np 

def compare_classifiers(models, X_test, y_test, xx, yy, X=None, y=None,
                        colormap='RB', X_alpha=0.3, figsize=None):
    '''
    Utility function that plots desicion rules for classifiers on a test 
    dataset
    '''
    cmap = {
        'PG': ('PiYG', 'PRGn'),
        'RB': ('coolwarm', 'seismic')
    }
    cm, cm_bright = cmap[colormap]
    if figsize is None:
        figsize = (len(models) * 6, 5)
    plt.figure(figsize=figsize)    
    for i, clf in enumerate(models):
        prediction = clf.predict(X_test)
        ax = plt.subplot(1, len(models), i + 1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        accuracy = (prediction == y_test).sum() * 1. / len(y_test)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.8, cmap=cm)
        if (X is not None) and (y is not None):
            ax.scatter(X[:, 0], X[:, 1], c=y, alpha=X_alpha, cmap=cm_bright) 
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(f'{clf.__class__.__name__} decision rule')
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        ax.text(0.65, 0.95, f'accuracy={accuracy:.2f}', 
                transform=ax.transAxes, bbox=props,
                verticalalignment='top', fontsize=figsize[0])