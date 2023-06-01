import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# the raw string for the data file for analysis. 
# if the file is, for example, located on the desktop:

# Windows:  file_string = r'C:\Users\1mike\Desktop\Sample GC 1 pt acetone 1 pt cyclohexane.csv'
# Linux:    file_string = '/home/michael/Desktop/Sample GC 1 pt acetone 1 pt cyclohexane.csv'
# macOS:    file_string = ‘/Users/1mikegrn/Desktop/Sample GC 1 pt acetone 1 pt cyclohexane.csv’


#work_dir = ''

#work_file =''

#work_file = filedialog.askopenfilename(initialdir=work_dir, filetypes=[('CSV file', '*.csv')], title='Open CSV file')    

""" Read the curve CSV file.
"""

#file_string = work_file
file_string = fname

# reads the input file 

#file_string = r'Rg1_tach.csv'
#file_string = r'mr.csv'

# reads the input file 

# if data file is in an excel file, use the following line: 
# master = pd.read_excel(file_string).to_numpy()

# if using a '.csv' file, use the following line:
data_set = pd.read_csv(file_string).to_numpy()

# add code
df = pd.read_csv(file_string)

# read in spectrum from data file
# T=nhietdo, I=cuongdo, dI=I uncertainty
# add code
n_value = len(df.columns)

ledger1 = []
for i in range(n_value):
    ledger1.append('GL' + str(i+1))

# Create figure window to plot data
fig = plt.figure(1, figsize=(9.5, 6.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, 0])

# Top plot: data and fit
ax1 = fig.add_subplot(gs[0])

#fig, ax1 = plt.subplots(dpi = 300)
#adds the ledger to the graph.
#ax1.legend(ledger)

#sets the axis labels and parameters.
# add code NN--------------------
#yy = np.squeeze(zeros(len(x)))
#yy = y
#xx = np.linspace(min_x, max_x, len(x))

ax1.tick_params(direction = 'in', pad = 15)
ax1.set_xlabel('Temperature (K)', fontsize = 15)
ax1.set_ylabel('Intensity  (a.u.)', fontsize = 15)


# plots the first two data sets: the raw data and the GaussSum.

for i in range(0,n_value, 2):
    ax1.scatter(data_set[:,i]+273, data_set[:,i+1])
    #ax1.plot(data_set[:,i], data_set[:,i+1])
    #ax1.fill_between(data_set[:,i], data_set[:,i+1],alpha=0.25)
    print(i, i+1)
    #print(data_set[:,i]+273, data_set[:,i+1])

# format and show
#ledger1 = ['OTOR', 'MOK', 'GOK']
#adds the ledger to the graph.
ax1.legend(ledger1)

plt.tight_layout()
plt.show()