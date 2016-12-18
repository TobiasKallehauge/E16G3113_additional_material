# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:53:40 2016
@author: G3-113 Mat-Tek 

This script computes the numerical solution of 
The eigenvalue method, including the fundamental matrix, is used, and later
compred to the solution with the exsponential matrix and non linear solution.   
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import rungekutta as rk

#System parameters
g = 9.82
m = 20
M = 10
c = 10
k = 5
l0 = 5


#==============================================================================
# Find critical damping by emperic method
#==============================================================================
# K-values for no regulation
Kp1 = 0
Ki1 = 0
Kd1 = 0
Kp2 = 0
Ki2 = 0
Kd2 = 0

Kd1 = 166.43681010908933 #Critical damping of just angle

Kp2 = 0.932228905743981 #Critical damping just position

Kd1 = 169.4090185900067 #Second iteration

Kp2 = 0.9347331274204 #Second iteration

Kd1 = 169.416828484323403 #Third iteration

Kp2 = 0.934739755669057  #Third iteration

#Analyse cross cobling
#Kd1 = Kd1*1.01 

#Kp2 = Kp2*1.01


#inital conditions
t_start = 0
t_stop = 50


theta0 = np.deg2rad(0.5) # initial angle 
y0 = 0                   # position 
l0 = 5                   # linght of string
omega0 = 0               # angle velocity
my0 = 0                  # velocity
phi0 = 0                 # string velocity
Theta0 = 0               # Integral of theta0
Y0 = 0                   # Integral of y0
x0 = np.array([theta0,y0,l0,omega0,my0,phi0,Theta0,Y0])

#Reference position
ref = 5


#==============================================================================
# Initialise functions
#==============================================================================

sin = np.sin
cos = np.cos

def l_acc(t):
    return 0

# Linear regulated system (homegenious part)
def regulator(m,M,k,c,g,l0,Kp1,Kp2,Ki1,Ki2,Kd1,Kd2):
    return np.matrix([
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [(-g*(m+M) - Kp1)/(M*l0),Kp2/(M*l0),0,-k/m -Kd1/(M*l0),(c+Kd2)/(M*l0)\
    ,0,-(Ki1)/(M*l0),Ki2/(M*l0)],
    [(m*g + Kp1)/M,-Kp2/(M),0,Kd1/M,-(c+Kd2)/M,0,Ki1/M,-Ki2/M],
    [0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0]])

# Non linear system with regulator
def sys_reg(t,u):
    l = u[2]    
    return np.array([\
    u[3], \
    u[4], \
    u[5], \
    -(2*u[5]*u[3])/l - (k/m)*u[3] - (g/l)*sin(u[0]) + (cos(u[0])*(c*u[4] \
     + m*sin(u[0])*(l_acc(t) - g*cos(u[0]) - l*u[3]**2) - \
    (Ki1*u[6] + Kp1*u[0] + Kd1*u[3] + Ki2*(ref*t - u[7]) + Kp2*(ref - u[1]) - Kd2*u[4])))/ \
    (l*(M + m*sin(u[0])**2)), \
    ((Ki1*u[6] + Kp1*u[0] + Kd1*u[3] + Ki2*(ref*t - u[7]) + Kp2*(ref - u[1]) - Kd2*u[4]) \
    -c*u[4] + m*sin(u[0])*(l*u[3]**2 + g*cos(u[0]) - l_acc(t)))/(m*sin(u[0])**2 + M), \
    l_acc(t),
    u[0],
    u[1]])


#==============================================================================
# Calculate solution
#==============================================================================

#Get Matrix in order to calculate eigenvalues
S =  regulator(m,M,k,c,g,l0,Kp1,Kp2,Ki1,Ki2,Kd1,Kd2)

#Calculate algorithm
tol = 1e-12 # Tolerence
h0 = 0.5 #inital stepsize
epsilon = 0.04 #error margin for s
sol2, step = rk.rkf_algorithm(sys_reg,x0,t_start,t_stop, tol, h0,epsilon) #call RKF45
time = np.linspace(t_start,t_stop,len(sol2.T[0]))



#==============================================================================
# Calculate eigenvalues
#==============================================================================
eigval, eigvec = np.linalg.eig(S)
re = np.zeros(8)
im = np.zeros(8)

for i in range(len(eigval)):
    re[i] = np.real(eigval[i])
    im[i] = np.imag(eigval[i])

akse1 = np.matrix([[-3],
                   [1]]) 
if np.max(im) != 0:
    akse2 = np.matrix([[np.max(im)+0.5],
                       [np.min(im)-0.5]])
else:
    akse2 = np.matrix([[-0.5],
                       [0.5]])

#==============================================================================
# Plot simulation
#==============================================================================

plt.plot(time,np.rad2deg(sol2.T[0]))
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Angle of bob")
#plt.savefig("Grafer/Vinkel.pdf")
plt.show()

plt.plot(time,sol2.T[1], label = "hah")
plt.xlabel("Time [s]")
plt.ylabel("Pos [m]")
plt.title("Position of cart")
#plt.savefig("grafer/Position.pdf")
plt.show()


plt.title("Spectrum")
plt.plot(re,im,'ro')
plt.plot(akse1,[0,0],"k-")
plt.plot([0,0],akse2,"k-")
#plt.savefig("grafer/Spektrum.pdf")
plt.show()








