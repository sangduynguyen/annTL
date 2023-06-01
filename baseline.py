import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve


def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z

from tkinter import filedialog
     
if __name__ == '__main__':
    '''
    Example usage and testing
    '''
    print('Testing...')
    from scipy.stats import norm
    import matplotlib.pyplot as pl
    import pandas as pd
    
    import os
    #import tkinter as tk
    #from tkinter import filedialog
    import sys

    #file_string = r's1.csv'
    
    work_dir = ''
    work_file =''
    work_file = filedialog.askopenfilename(initialdir=work_dir, filetypes=[('CSV file', '*.csv')], title='Open CSV file')

    """ Read the curve CSV file.
    """
    file_string = work_file
    # if using a '.csv' file, use the following line:
    data_set = pd.read_csv(file_string).to_numpy()


    # x
    x = data_set[:, 0]

    # defines the independent variable.
    #
    y = data_set[:, 1]
    

    x1 = np.arange(0, 1000, 1)
    g1 = norm(loc=100, scale=1.0)  # generate three gaussian as a signal
    g2 = norm(loc=300, scale=3.0)
    g3 = norm(loc=750, scale=5.0)
    signal = g1.pdf(x1) + g2.pdf(x1) + g3.pdf(x1)
    baseline1 = 5e-4 * x1 + 0.2  # linear baseline
    #baseline2 = 0.2 * np.sin(np.pi * x / x.max())  # sinusoidal baseline
    noise = np.random.random(x1.shape[0]) / 500
    print('Generating simulated experiment')
    y1 = signal + baseline1 + noise
    #y2 = signal + baseline2 + noise
    y2=y
    print('Removing baselines')
    c1 = y1 - airPLS(y1)  # corrected values
    c2 = y2 - airPLS(y2)  # with baseline removed
    print('Plotting results')
    fig, ax = pl.subplots(nrows=1, ncols=1)
    #ax[0].plot(x1, y1, '-k')
    #ax[0].plot(x1, c1, '-r')
    #ax[0].set_title('Linear baseline')
    ax.plot(x, y2, '-k')
    ax.plot(x, c2, '-r')
    #ax.set_title('Sinusoidal baseline')
    pl.show()
    
    # add code
    data_new = {'T':x,'I':c2}
    # save to csv
    df = pd.DataFrame(data_new)
    # saving the dataframe
    #df.to_csv('C1.csv', sep=';', header=False, index=False)   
    df.to_csv('C.csv',header=False, index=False)   
    
    print('Done!')
    
    # close window tinker
    #destroy()
    sys.exit(0)
   
 