from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pylab as pl

from numpy import *
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # unequal plots
from scipy.optimize import leastsq
from pylab import *
from peakdetect import peakdetect

#import tkinter as tk
#from tkinter import filedialog

import sys
from tkinter import *
from tkinter.filedialog import askopenfilename   

fname = "unassigned"
#work_dir = ''

#work_file =''

def openFile():
    global fname
    fname = askopenfilename(initialdir=fname, filetypes=[('CSV file', '*.csv')])
    root.destroy()

if __name__ == '__main__':

    root = Tk()
    Button(root, text='File Open', command = openFile).pack(fill=X)
    mainloop()

    print (fname)

import builtins

import array as arr

from scipy.optimize import curve_fit
from numpy import arange



#file_string = work_file
file_string = fname

# reads the input file 

# if using a '.csv' file, use the following line:
data_set = pd.read_csv(file_string).to_numpy()

# add code
df = pd.read_csv(file_string)

# read in spectrum from data file
# T=nhietdo, I=cuongdo, dI=I uncertainty
# add code
n_value = len(df.columns)

#fig = plt.figure()
#ax = Axes3D(fig)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('TL')
ax.set_zlabel('Intensity  (a.u.)')
#colors = ['r', 'g', 'b', 'y']

for i in range(0,n_value, 2):
    x = data_set[:,i]
    y = data_set[:,i+1]
    #z = data_set[:,i+2]
    z = np.zeros(x.shape) + i
    #ax.bar3d(x, y, z, dx, dy, dz, verts='y', color='blue')

    
    #ax.add_collection3d(plt.fill_between(x,y,1, alpha=0.3,label="filled plot"),2, zdir='y')
    
    #verts = [(x[j],z[j],y[j]) for j in range(len(x))] + [(x.max(),0,0),(x.min(),0,0)]
    #verts = [(x[i],y[i],z[i])]
    #ax.add_collection3d(Poly3DCollection([verts])) # Add a polygon instead of fill_between
    ax.add_collection3d(pl.fill_between(x, y, i, alpha=0.3), zs=i, zdir='y')

        
    ax.plot(x,z,y,label='GL '+str(round((i/2)+1)))
    ax.legend()
    #ax.set_ylim(-1,1)
plt.show()

