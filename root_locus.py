# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:53:40 2016
@author: G3-113 Mat-Tek 

In this a root locus plot for the critical damped system is simulated.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#System parameters
g = 9.82
m = 20
M = 10
c = 10
k = 5
l0 = 5


#Angle controll
Kp1 = 0
Ki1 = 0   
Kd1 = 169.4311 #Criticaly damped

#Position controll
Kp2 = 0.9394572088
Ki2 = 0.00095111
Kd2 = 0    

dim_system = 8


def regulator(m,M,k,c,g,l0,a):
    return np.matrix([
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [(-g*(m+M) - a*Kp1)/(M*l0),a*Kp2/(M*l0),0,-k/m -(a*Kd1)/(M*l0),(c+a*Kd2)/(M*l0)\
    ,0,-(a*Ki1)/(M*l0),a*Ki2/(M*l0)],
    [(m*g + a*Kp1)/M,-(a*Kp2)/M,0,a*Kd1/M,-(c+a*Kd2)/M,0,a*Ki1/M,-a*Ki2/M],
    [0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0]])

#==============================================================================
# Get eigenvalues and plot
#==============================================================================
alpha = np.linspace(0,3,5000)
#alpha = np.linspace(0,0.05,5000) #for plot 3
list_im = np.zeros((len(alpha),dim_system))
list_re = np.zeros((len(alpha),dim_system))
list_ei = np.zeros((len(alpha),dim_system), dtype = "complex128")

alpha_critical_wierd = 0.0428436

for i in range(len(alpha)):
    a = alpha[i]   
    eigval = np.linalg.eigvals(regulator(m,M,k,c,g,l0,a))
    list_ei[i] = eigval
    for q in range(len(eigval)):
        list_re[i,q] = np.real(eigval[q])
        list_im[i,q] = np.imag(eigval[q])

for j in range(dim_system):
    plt.plot(list_re.T[j],list_im.T[j],"k,")   
    minre = np.min(np.real(eigval))*1.2
    minim = np.min(np.imag(eigval))*1.2
    if minim == 0:
        minim = minre



plt.title("Spectrum 1")
plt.axis([-2.5,0.1,-2.5,2.5])
plt.arrow(-1.3,1.88,-0.01,-0.01, head_width=0.2, head_length=0.2, fc='b', ec='k')
plt.arrow(-1.3,-1.88,-0.01,0.01, head_width=0.2, head_length=0.2, fc='b', ec='k')
plt.arrow(-2,0,0.01,0, head_width=0.2, head_length=0.2, fc='r', ec='k')
plt.arrow(-2.2,0,-0.01,0, head_width=0.2, head_length=0.2, fc='r', ec='k')
plt.text(-1.6,1,"alpha = 1")
plt.arrow(-1.4,0.9,-0.5,-0.6, head_width=0.04, head_length=0.08, fc='k', ec='k')
#plt.savefig("grafer/spektrum1.pdf")

#plt.title("Spectrum 2")
#plt.axis([-0.35,0.01,-1,1])
#plt.text(-0.32,0.55,"alpha = 1")
#plt.arrow(-0.3,0.5,0.08,-0.38, head_width=0.01, head_length=0.03, fc='k', ec='k')
#plt.arrow(-0.3,0,0.01,0, head_width=0.1, head_length=0.03, fc='b', ec='k')
#plt.arrow(-0.11,0,-0.01,0, head_width=0.1, head_length=0.03, fc='b', ec='k')
#plt.arrow(-0.17,0.27,0.01,0.009, head_width=0.05, head_length=0.03, fc='r', ec='k')
#plt.arrow(-0.17,-0.27,0.01,-0.009, head_width=0.05, head_length=0.03, fc='r', ec='k')
##plt.savefig("grafer/spektrum2.pdf")    

#plt.title("Spectrum 3")
#plt.axis([-0.003,0.0001,-0.0012,0.0012])
#plt.arrow(-0.001,0.00102,-0.00005,0,width = 0.00002, head_width=0.00011, head_length=0.0002, fc='b', ec='k')
#plt.arrow(-0.001,-0.00102,-0.00005,0,width = 0.00002, head_width=0.00011, head_length=0.0002, fc='b', ec='k')
#plt.arrow(-0.0017,0,0.00005,0,width = 0.00002, head_width=0.00011, head_length=0.0002, fc='r', ec='k')
#plt.arrow(-0.00235,0,-0.00005,0,width = 0.00002, head_width=0.00011, head_length=0.0002, fc='g', ec='k')
#plt.text(-0.00274,0.0007,"alpha = %.5f" %(alpha_critical_wierd))
#plt.arrow(-0.0024,0.00065,0.00022,-0.00035,width = 0.00002, head_width=0.00005, head_length=0.0001, fc='k', ec='k')
##plt.savefig("grafer/spektrum3.pdf")   
#plt.show()  
