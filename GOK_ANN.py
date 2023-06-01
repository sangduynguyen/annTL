
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt1
import matplotlib.gridspec as gridspec  # unequal plots
from scipy.optimize import leastsq
from pylab import *

from matplotlib.widgets import Cursor

import os
import tkinter as tk
from tkinter import filedialog
import sys

import builtins


from scipy.optimize import curve_fit
from numpy import arange

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

# if data file is in an excel file, use the following line: 
# master = pd.read_excel(file_string).to_numpy()

# if using a '.csv' file, use the following line:
data_set = pd.read_csv(file_string).to_numpy()

#
x = data_set[:,0]
max_x=np.max(x)
min_x=np.min(x)
# defines the independent variable. 
#
y =  data_set[:,1]

# --- No changes below this line are necessary ---
# Plotting
fig = plt.figure()
ax = fig.subplots()
#sets the axis labels and parameters.
ax.tick_params(direction = 'in', pad = 15)
ax.set_xlabel('Temperature (K)', fontsize = 15)
ax.set_ylabel('Intensity  (a.u.)', fontsize = 15)
ax.plot(x,y,'bo')
#ax.scatter(x, y, s=15, color='blue', label='Data')
ax.grid()
# Defining the cursor
cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                color = 'r', linewidth = 1)

# Creating an annotating box
annot = ax.annotate("", xy=(0,0), xytext=(-40,40),textcoords="offset points",
                    bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                    arrowprops=dict(arrowstyle='-|>'))
annot.set_visible(True)

# Function for storing and showing the clicked values
#print("Import E:")
#z = 1.5  # import E
z = 0.85 # import E
#try:
# z = float(input('E: '))
# if z > 0:
 #   print('E = ',z)
#except:
 #z = 1.11
print("Default!, E = ",z)

# import b
bv = 1.50
try:
 bv = float(input('bv (1,2): '))
 if (bv > 1) & (bv <=2):
   print('bv = ',bv)
 else:
   bv = 1.61
   print('1<bv<2, bv=',bv)
except:
 print("1<bv<2", bv)

coord = []
def onclick(event):
    global coord
    coord.append((event.ydata,event.xdata, z, bv))
    x1 = event.xdata
    y1 = event.ydata
    #z = 1.61  
    # printing the values of the selected point
    print(y1,x1,z,bv) 
    annot.xy = (x1,y1)
    text = "[{:.3g}, {:.5}]".format(x1,y1)
    annot.set_text(text)
    annot.set_visible(True)
    fig.canvas.draw() #redraw the figure

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

initials = list(coord)
print(initials)


# determines the number of gaussian functions 
# to compute from the initial guesses
n_value = len(initials)

# defines a typical gaussian function, of independent variable x,
# amplitude a, position b, and width parameter c.
def gaussian(x,a,b,c,bv):
    kbz = 8.617385e-5
    return a*(bv**(bv/(bv-1.0)))*np.exp(c/kbz/x*((x-b)/b))*(((bv-1.0)*(x/b)**2)*np.exp(c/kbz/x*((x-b)/b))*(1.0-2.0*kbz*x/c)+1+(bv-1.0)*2.0*kbz*b/c)**(-bv/(bv-1.0))

# defines the expected resultant as a sum of intrinsic gaussian functions
def GaussSum(x, p, n):
    return builtins.sum(gaussian(x, p[4*k], p[4*k+1], p[4*k+2], p[4*k+3]) for k in range(n))

class NeuralNetworks(object):
    def __init__(self):  # Initializing the weight vectors
        self.n = 2  # Number of hidden neurons
        self.eta = 0.01  # Gradient step learning rate
        #self.w_1 = np.random.normal(0, 8, (self.n, 1))  # Input --> Hidden initial weight vector
        self.w_1 = np.linspace(0, 1, 2)[:, np.newaxis]
        self.b_1 = np.ones((self.n, 1))
        #self.w_2 = np.random.uniform(0, 0.01, (self.n, 1))  # Hidden --> Output initial weight vector
        self.w_2 = np.linspace(0, 1, 2)[:, np.newaxis]
        self.b_2 = np.ones((1, 1))
    
    def FeedForward(self, x, a, b, c, bv):  # This method feeds forward the input x and returns the predicted output
        # I use the same notation as Haykin book
        self.v_1 = x * (self.w_1) + self.b_1  # Local Induced Fileds of the hidden layer
        # edit code
        self.y_1 = gaussian(self.v_1, a, b, c, bv)  # hàm truyền

        self.v_2 = self.y_1.T.dot(self.w_2) + self.b_2 # x.T chuyển hàng thành cột, dot: vô hướng
        self.o = self.v_2  # output of the network
        
        return self.o
    
    def loss(self, x, d):  # Calculates the cost function of the network for a 'vector' of the inputs/outputs
        # x : input vector
        # d : desired output
        temp = np.zeros(len(x))
        for i in range(len(x)):
            temp[i] = d[i] - self.FeedForward(x[i])
        self.cost = np.mean(np.square(temp))
        return self.cost

    def BackPropagate(self, x, y, d):
        # Given the input, desired output, and predicted output
        # this method update the weights accordingly
        # I used the same notation as in Haykin: (4.13)
        self.delta_out = (d - y) * 1  # 1: phi' of the output at the local induced field
        self.w_2 += self.eta * self.delta_out * self.y_1
        self.b_2 += self.eta * self.delta_out

        # edit code
        self.delta_1 = (1 - np.power(np.tanh(self.v_1), 2)) * (self.w_2) * self.delta_out
        #self.delta_1 = (1 - np.power(np.maximum(self.v_1,0), 2)) * (self.w_2) * self.delta_out
        #edit code
        #self.delta_1 = (1 - np.power(np.exp(-(self.v_1)**2), 2)) * (self.w_2) * self.delta_out
        self.w_1 += self.eta * x * self.delta_1
        self.b_1 += self.eta * self.delta_1

    def train(self, x, d, epoch=100):  # Given a vector of input and desired output, this method trains the network
        iter = 0
        while (iter != epoch):
            for i in range(len(x)):
                o = self.FeedForward(x[i])  # Feeding forward
                self.BackPropagate(x[i], o, d[i])  # Backpropagating the error and updating the weights
            if iter % (epoch / 5) == 0:
                print("Epoch: %d\nLoss: %f" % (iter, self.loss(x, d)))
            iter += 1
# add code
    def findpeak(self, x, a, b, c, bv): 
        from scipy.signal import find_peaks
        #import matplotlib.pyplot as plt
        
        # Example data
        self.max_x=np.max(x)
        self.min_x=np.min(x)
        x = np.linspace(self.min_x, self.max_x)
        #y =  gaussian(x,a,b,c)
        #y = np.squeeze(zeros(len(x)))
        y =  np.squeeze(self.FeedForward(x,a,b,c, bv))
        #y = np.sin(x / 1000) + 0.1 * np.random.rand(*x.shape)
        
        # Find peaks
        self.i_peaks, _ = find_peaks(y)
        #print(i_peaks)
        
        # Find the index from the maximum peak
        #i_max_peak = i_peaks[np.argmax(y[i_peaks])]
        
        # Find the x value from that index
        #x_max = x[i_max_peak]
        Tm = x[self.i_peaks]
        
        # Plot the figure
        #plt.plot(x, y)
        #plt.plot(x[i_peaks], y[i_peaks], 'x')
        #plt.axvline(x=x_max, ls='--', color="k")
        #plt.show()
        return Tm
# add code
    def findpeakY(self, x, a, b, c, bv): 
        from scipy.signal import find_peaks
        #import matplotlib.pyplot as plt
        
        # Example data
        self.max_x=np.max(x)
        self.min_x=np.min(x)
        x = np.linspace(self.min_x, self.max_x)
        #y =  gaussian(x,a,b,c)
        #y = np.squeeze(zeros(len(x)))
        y =  np.squeeze(self.FeedForward(x,a,b,c, bv))
          
        # Find peaks
        self.i_peaks, _ = find_peaks(y)
        #print(i_peaks)
        
        # Find the index from the maximum peak
        #i_max_peak = i_peaks[np.argmax(y[i_peaks])]
        
        # Find the x value from that index
        #x_max = x[i_max_peak]
        Im = y[self.i_peaks]
        
        # Plot the figure
        #plt.plot(x, y)
        #plt.plot(x[i_peaks], y[i_peaks], 'x')
        #plt.axvline(x=x_max, ls='--', color="k")
        #plt.show()
        return Im


#--------------------
ndsang = NeuralNetworks()
#print ("Initial Loss: %f"%(ndsang.loss(x,d)))
#print("----|Training|----")
#ndsang.train(x,d,2000)
#print("----Training Completed----")

# add fit NN Gauss
def fitGauss(x,y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    gaussian_process.fit(x, y)

# defines condition of minimization, called the resudual, which is defined
# as the difference between the data and the function.
def residuals(p, y, x, n):
    return y - GaussSum(x,p,n)

# defines FOM
def FOM(p, y, x, n):
     return sum(y - GaussSum(x,p,n))/sum(GaussSum(x,p,n))

def E1(Tm,T1,T2):
    k = 8.617385e-5
    #return abs(1.51*((k*Tm**2)/(Tm - T1))-1.58*(2*k*Tm))
    return abs((2.52+10.2*((T2-Tm)/(T2-T1)-0.42))*((k*(Tm)**2)/(T2-T1))-(2*k*Tm))
    #return (0.976+7.3*((T2-Tm)/(T2-T1)-0.42))*((k*Tm**2)/(Tm - T1))	

# Convert decimal to a binary string
def den2bin(f):
	bStr = ''
	n = int(f)
	if n < 0: raise
	if n == 0: return '0'
	while n > 0:
		bStr = str(n % 2) + bStr
		n = n >> 1
	return bStr

#Convert decimal to a binary string of desired size of bits 
def d2b(f, b):
	n = int(f)
	base = int(b)
	ret = ""
	for y in range(base-1, -1, -1):
		ret += str((n >> y) & 1)
	return ret

#Invert Chromosome
def invchr(string, position):
	if int(string[position]) == 1:
		
		string = string[:position] + '0' + string[position+1:]
	else:
		string = string[:position] + '1' + string[position+1:]
	return string


#Roulette Wheel
def roulette(values, fitness):
	n_rand = random()*fitness
	sum_fit = 0
	for i in range(len(values)):
		sum_fit += values[i]
		if sum_fit >= n_rand:
			break
	return i	
# using least-squares optimization, minimize the difference between the data

def AE_gen(x,a,b,c,bv):
    #x = data_set[:,0]
    y=gaussian(x,a,b,c, bv)
    #Tm = geneticTL(x,a,b,c)
    #Im = geneticTLy(x,a,b,c)
    Tm = ndsang.findpeak(x,a,b,c,bv)
    Im = ndsang.findpeakY(x,a,b,c,bv)
   
    #
    y_half=np.zeros_like(x)+Im/2
    print("Img=",Im)

    #
    idx = np.argwhere(np.diff(np.sign(y -y_half))).flatten()

    #print("T1,T2",  x[idx])
    # Calculate E, method PS
    T1=x[idx][0]
    T2=x[idx][1]
    #Tm=higherPeaksY[:,0]
    #Tm=generations_x[num_gen-1]
    print("T1g=",T1)
    print("T2g=",T2)
    print("Tmg=",Tm)
    #omega=T2-T1
    #delta=T2-Tm
    #thau=Tm-T1
    #E=E1(Tm[0],T1,T2)
    E=E1(Tm,T1,T2)
    print("Eg=",E)
    return E

# define the true objective function IR
def IR(x, E, b):
    return E * x + b
# E: IR
def AE_IR(x,a,b,c,bv):
    #x = data_set[:,0]
    y=gaussian(x,a,b,c,bv)
    #Tm = geneticTL(x,a,b,c)
    #Im = geneticTLy(x,a,b,c)
    Tm = ndsang.findpeak(x,a,b,c,bv)
    Im = ndsang.findpeakY(x,a,b,c,bv)

    #Tm = generations_x[num_gen-1]
    #Im = generations_f[num_gen-1]
    
    #add IR
    #j=15
    j=10
    Tci=[]
    Ici=[]
    
    for i in range(1,j):
        
        yi=np.zeros_like(x)+Im*i/100
        idx = np.argwhere(np.diff(np.sign(y - yi))).flatten()
        Tc = x[idx][0]
        Ic = y[idx][0]
        #print("Tc,Ic=",Tc,Ic)
        Tci=np.append(Tci, Tc)
        Ici=np.append(Ici, Ic)

    kbz = 8.617385e-5
    # ND
    x_ND=1/(kbz*Tci)
    
    # ln(TL)
    y_ln=np.log(Ici)
    
    # curve fit
    popt, _ = curve_fit(IR, x_ND, y_ln)
    # summarize the parameter values
    E_IR, b_IR = popt
    print("E & b:",-E_IR,b_IR)
    print('y = %.5f * x + %.5f' % (E_IR,b_IR))
    
    #E=E2(Tc)
    print("E_IR=",-E_IR)
    
    #define function to calculate adjusted r-squared
    #def R2(x1_ND, y1_ln, degree):
    #results = {}
    coeffs = np.polyfit(x_ND, y_ln, 1)
    p = np.poly1d(coeffs)
    yhat = p(x_ND)
    ybar = np.sum(y_ln)/len(y_ln)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y_ln - ybar)**2)
    R2 = 1- (((1-(ssreg/sstot))*(len(y_ln)-1))/(len(y_ln)-1))
    print("R2=",R2)

    # Make predictions using the testing set
    y_pred = x_ND * E_IR + b_IR

    # format and show
    fig = plt.figure()
    bx = fig.subplots()
    # sets the axis labels and parameters.
    bx.tick_params(direction='in', pad=15)
    bx.set_xlabel('1/kT (eV)$^{-1}$', fontsize=15)
    bx.set_ylabel('lnI', fontsize=15)
    # bx.plot(x_ND, y_ln, 'r-')

    bx.scatter(x_ND, y_ln, color='blue', label='Data')

    bx.text(0.02, 0.07, r'Ei = {0:0.6f}'
            .format(-E_IR), transform=bx.transAxes)
    bx.text(0.02, 0.02, r'y = {0:0.5f}*x+ {1:0.5f}, R$^2$ = {2:0.5f}'
            .format(E_IR, b_IR, R2), transform=bx.transAxes)
    # add code
    bx.plot(x_ND, y_pred, color="red", linewidth=3)

    # truc hoanh
    y_0 = np.zeros_like(x)+ 0.01
    #
    idx = np.argwhere(np.diff(np.sign(y - y_0))).flatten()
    #
    # Calculate 2 chan cua TL
    #T10 = x[idx][0]
    #T20 = x[idx][1]

    # add code NN--------------------
    yy = np.squeeze(zeros(len(x)))
    #yy = y
    xx = np.linspace(min_x, max_x, len(x))
    #x0 = np.squeeze(linspace(min_x, max_x, len(x)))
    #xx = np.squeeze(linspace(T10, T20, len(x)))
    #xx = x

    for i in range(len(xx)):
        #yy[i] = ndsang.FeedForward(xx[i],Im,Tm,E_IR)
        yy[i] = ndsang.FeedForward(xx[i], Im, Tm, z, bv)
    fig = plt.figure()
    cx = fig.subplots()
    # add code
    #cx.scatter(T10, T20, label='Data-set')
    cx.scatter(xx, yy, label='Data-set i')
    #cx.plot(xx, yy, color='b', label='NN-Output')
    cx.plot(xx, gaussian(xx, Im, Tm, z, bv), color='red', label='NN-Output i')
    #cx.plot(xx, gaussian(xx, Im, Tm, E_IR),  color='red',label='NN-Output')
    cx.tick_params(direction='in', pad=15)
    cx.set_xlabel('Temperature (K)', fontsize=15)
    cx.set_ylabel('Intensity  (a.u.)', fontsize=15)
    cx.legend()
    # end code
    return -E_IR

cnsts =  leastsq(
            #geneticTL,
            residuals, 
            initials, 
            args=(
                data_set[:,1],          # y data
                data_set[:,0],          # x data
                n_value                 # n value
            )
        )[0]

# integrates the gaussian functions through gauss quadrature and saves the 
# results to a list, and each list is saved to its corresponding data file 

# Create figure window to plot data
fig = plt.figure(1, figsize=(9.5, 6.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

# Top plot: data and fit
ax1 = fig.add_subplot(gs[0])

#sets the axis labels and parameters.
ax1.tick_params(direction = 'in', pad = 15)
#ax1.set_xlabel('Temperature (K)', fontsize = 15)
ax1.set_ylabel('Intensity  (a.u.)', fontsize = 15)

# plots the first two data sets: the raw data and the GaussSum.
ax1.plot(data_set[:,0], data_set[:,1], 'ko')
ax1.plot(x,GaussSum(x,cnsts, n_value))

# adds a plot
#ax1.fill_between(x, GaussSum(x,cnsts, n_value), facecolor="yellow", alpha=0.25)of each individual gaussian to the graph.
for i in range(n_value):
    ax1.plot(
        x, 
        gaussian(
            x, 
            cnsts[4*i],
            cnsts[4*i+1],
            cnsts[4*i+2],
            cnsts[4*i+3]
        )
    )
# adds color a plot of each individual gaussian to the graph.
for i in range(n_value):
    ax1.fill_between(
        x, 
        gaussian(
            x, 
            cnsts[4*i],
            cnsts[4*i+1],
            cnsts[4*i+2],
            cnsts[4*i+3]
        ),alpha=0.25
    )

# adds a ffGOK of each individual gaussian to the graph.
AE2 = dict()
for i in range(n_value):
    AE2[i] = AE_gen(
            x, 
            cnsts[4*i],
            cnsts[4*i+1],
            cnsts[4*i+2],
            cnsts[4*i+3]
        )
# adds a ffGOK of each individual gaussian to the graph.
AE3 = dict()
for i in range(n_value):
    AE3[i] = AE_IR(
            x, 
            cnsts[4*i],
            cnsts[4*i+1],
            cnsts[4*i+2],
            cnsts[4*i+3]
        )


# creates ledger for each graph
ledger = ['Data', 'Resultant']
for i in range(n_value):
    ledger.append(
        'P' + str(i+1)
        + ', E'+'$_{IR}$' + str(i+1) +' = ' +str(round(AE3[i],3)) + ' eV'
        #+ ', E' +'$_{PS}$'+ str(i+1) +' = '+ str(round(AE2[i][0],3)) + ' eV'
         #+ ', t'+ str(i+1) +'$_{1/2}$ = ' + str(round(half1[i],3))+ ' s'
        #+ ', Tm'  + str(i+1) +' = '+ str(round(GA1[i],3)) + ' K'
        #+ ', Im'  + str(i+1) +' = '+ str(round(GA2[i],3)) + ' a.u.'
        #+ ', s = ' + str(round(ff[i])) + ' s$^{-1}$'        xxmxmxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        #+ '\ns = ' + str(round(ff[i]))[0] + '.' + str(round(ff[i]))[1] + str(round(ff[i]))[2] + 'x$10^{15}$' + ' s$^{-1}$'
    ) 

#adds the ledger to the graph.
ax1.legend(ledger)
plt.title('ANN approach for analyzing TL curve by GOK model')

#adds text FOM
#ax1.text(0.01, 0.25, r'GOK''\n FOM = {0:0.4f}'
ax1.text(0.01, 0.25, r'FOM = {0:0.9f}'
         .format(abs(FOM(cnsts, y, x, n_value))), transform=ax1.transAxes)

# Bottom plot: residuals
ax2 = fig.add_subplot(gs[1])
ax2.plot(x,residuals(cnsts, y, x, n_value))
#ax2.plot(x,GaussSum(x,cnsts, n_value))

ax2.set_xlabel('Temperature (K)',fontsize = 15)
ax2.set_ylabel('Residuals', fontsize = 15)
#ax2.set_ylim(-20, 20)
#ax2.set_yticks((-20, 0, 20))

# format and show
plt.tight_layout()
# format and show
#plt1.tight_layout()
#plot(fitGauss)

fig.savefig(r'ANN_GOK.png')

plt.show()

# add code
data_new = {'T':x,'I':GaussSum(x,cnsts, n_value)}
# save to csv
df = pd.DataFrame(data_new)
# saving the dataframe
#df.to_csv('C1.csv', sep=';', header=False, index=False)
df.to_csv('TL_GOK_new.csv',header=False, index=False)

#X= np.linspace(0, 600, 12000)[:, np.newaxis]
#y = np.squeeze(GaussSum(x,cnsts, n_value))


#X= np.linspace(0, 700, 200)[:, np.newaxis]
X= np.linspace(min_x, max_x, 200)[:, np.newaxis]
#X= np.array([min_x, max_x]).reshape(1, -1)
y1 = np.squeeze(GaussSum(X,cnsts, n_value))

rng = np.random.RandomState(0)
training_indices = rng.choice(np.arange(y1.size), size=1600)
X_train, y_train = X[training_indices], y1[training_indices]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

#plt.plot(X, y1, label=r"$f(x) = Gauss(x)$", linestyle="dotted", c='g')
plt.scatter(X_train, y_train, label="Observations", c='b')
plt.scatter(x, y, label="Data",s= 20, c='black')
plt.plot(X, mean_prediction, label="Mean prediction", c='r')
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.4,
    label=r"95% confidence interval",
)
#plt.show()
plt.legend()
#plt.xlabel("$x$")
#plt.ylabel("$f(x)$")
plt.xlabel('Temperature (K)', fontsize = 15)
plt.ylabel('Intensity  (a.u.)', fontsize = 15)
#_ = plt.title("Gaussian process regression")
plt.show()

# close window tinker
#destroy()
sys.exit(0)
#del x, X
#del y
