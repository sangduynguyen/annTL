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

# if using a '.csv' file, use the following line:
data_set = pd.read_csv(file_string).to_numpy()

# add code
df = pd.read_csv(file_string)

# read in spectrum from data file
# T=nhietdo, I=cuongdo, dI=I uncertainty
# add code
n_value = len(df.columns)

fig, axes = plt.subplots(nrows=int(n_value/2), ncols=1, figsize=(12,8))

#ledgeri = []
# ledger.append('TL' + str(i+1))

for j in range(0,int(n_value/2), 1):
    #ledger.append('TL')
    #ledger.append('TL' + str(j+1))   
    for i in range(0,n_value, 2):
        #axes[j].legend(ledger)
        x = data_set[:,i]
        y = data_set[:,i+1]
        #axes[j].legend(ledger) 
        
        #count = 0  
        ledger = []   
        if i == j*2:

            r = np.round(np.random.rand(),1)
            g = np.round(np.random.rand(),1)
            b = np.round(np.random.rand(),1)
                
            #axes[j].plot(x,y, color=[r,g,b])
            axes[j].scatter(x,y, color=[r,g,b])
            #axes[j].fill_between(x, y, color=[r,g,b], alpha=0.95)
            #for l in range(int(n_value/2)):
            ledger.append('GL'+ str(j+1))  
            print(ledger)
                      
            #axes[j].plot(x,y, color=[r,g,b])
            #axes[j].legend(ledger) 
            #ledger.append('TL'+ str(j+1))
            
            #print('TL'+ str(j+1))
            axes[j].legend(ledger) 
            axes[j].tick_params(direction = 'in', pad = 15)
            axes[j].set_xlabel('Temperature (K)', fontsize = 15)
            axes[j].set_ylabel('Intensity  (a.u.)', fontsize = 15)
            
        
fig.tight_layout()

plt.show()
