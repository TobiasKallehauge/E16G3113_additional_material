# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:53:40 2016
@author: G3-113 Mat-Tek 

This script implements 3 models (one with PID-regulator) in order to simulate
a container beeing moved over an obstacle. The simulation is based on a numerical
solution of a non-linear model.  
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rungekutta as rk

sin = np.sin
cos = np.cos

#==============================================================================
# Initialise parameters, intial conditions and the regulations matrices
#==============================================================================

g = 9.82
m = 5000
M = 120000
c = 750
k = 50
l0 = 25

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

#==============================================================================
# The non-linear model with and without regulation
#==============================================================================

def force(t):
    return fm

def l_acc(t):
    c1 = (l1 - l2)/2.
    c2 = np.pi/T
    if t <= T + dt:
        return -(c2**2)*c1*np.cos(c2*(t-dt))
    else:
        return 0
        
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
    
def sys(t,u):
    l = u[2]    
    return np.array([
    u[3], 
    u[4], 
    u[5], 
    -(2*u[5]*u[3])/l - (k/m)*u[3] - (g/l)*sin(u[0]) + (cos(u[0])*(c*u[4] \
    - force(t) + m*sin(u[0])*(l_acc(t) - g*cos(u[0]) - l*u[3]**2)))/ \
    (l*(M + m*sin(u[0])**2)), \
    (force(t) - c*u[4] + m*sin(u[0])*(l*u[3]**2 + g*cos(u[0]) -l_acc(t)))/ 
    (m*sin(u[0])**2 + M), 
    l_acc(t),
    u[0],
    u[1]])

#==============================================================================
# First model - Lift container and move cart over the containers
#==============================================================================
T = 18.27 # Time period for first lift
t_start0 = 0
t_stop0 = T
dt = t_start0


theta0 = np.deg2rad(0.0) # initial angle 
y0 = 0                   # position 
l0 = l0                   # linght of string
omega0 = 0               # angle velocity
my0 = 0                  # velocity
phi0 = 0                 # string velocity
Theta0 = 0               # Integral of theta0
Y0 = 0                   # Integral of y0
x0 = np.array([theta0,y0,l0,omega0,my0,phi0,Theta0,Y0])

l1 = l0
l2 = 15.7 #Lenght of string so container is moved over stack
fm = 4000
 
#Simulate
X0  = rk.rk_system(sys,x0,t_start0,t_stop0,0.05)

time0 = np.linspace(t_start0,t_stop0,len(X0.T[0]))

#==============================================================================
# Second model - Move the cart to end position
#==============================================================================

t_start1 = t_stop0
t_stop1 = 469


# Regulation parameters
Kp1 = 0
Ki1 = 0.00001
Kd1 = 80000 # OP == ud af reel akse (og ned af imaginær)
Kp2 = 33
Ki2 = 0.0000001
Kd2 = 3070 # OP == ud af reel akse (og ned af imaginær)

x1 = X0[-1]
x1[5]=0
ref = 100
l1 = l2

S = regulator(m,M,k,c,g,l0,Kp1,Kp2,Ki1,Ki2,Kd1,Kd2) #Get matrix for eigenvalues
 
#Simulate
X1  = rk.rk_system(sys_reg,x1,t_start1,t_stop1,0.1)

time1 = np.linspace(t_start1,t_stop1,len(X1.T[0]))

#==============================================================================
# Third model - Lower container to ground
#==============================================================================

t_start2 = t_stop1
t_stop2 = t_stop1+T


x2 = X1[-1]
dt = t_start2
l1 = l2
l2 = 25
T = t_stop2 - t_start2
fm = 0

X2  = rk.rk_system(sys,x2,t_start2,t_stop2,0.06)

time2 = np.linspace(t_start2,t_stop2,len(X2.T[0]))

#==============================================================================
# Get eigenvalues and plot
#==============================================================================

eigval = np.linalg.eigvals(S)

for i in range(len(eigval)):
    re = np.real(eigval[i])
    im = np.imag(eigval[i])
    plt.plot(re,im,"o", label = "%d" %(i))
    
minre = np.min(np.real(eigval))*1.2
minim = np.min(np.imag(eigval))*1.2
#minre = 1e-2
#minim = 1
if minim == 0:
    minim = minre
plt.plot([minre,-minre*0.5],[0,0],"k-")
plt.plot([0,0],[minim,-minim],"k-")
plt.legend(loc = "center right")
plt.title("Eigenvalue spectrum")
plt.savefig("Grafer/eigenvalueplot.pdf")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()

    
# Obs: Egenværdi 1 og 2 danner par og egenværdi 3 og 4 danner par. Egenværdi 5 er alene

#==============================================================================
# Plot different states
#==============================================================================

""" Stacking the three models and times """
X = np.vstack([X0,X1,X2])
time = np.hstack([time0,time1,time2])

""" Angle plot """
plt.plot(time,np.rad2deg(X.T[0]),'r-')
plt.title("Angles")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.savefig("Grafer/case_angle.pdf")
plt.show()

""" Angle velocity plot """
plt.plot(time,np.rad2deg(X.T[3]),'r-')
plt.title("Angles velocity")
plt.xlabel("Time [s]")
plt.ylabel("Angle velocity [deg/sec]")
plt.savefig("Grafer/case_angle_velocity.pdf")
plt.show()

""" Position of cart plot """
plt.plot(time,X.T[1],'r-')
plt.title("Position of cart")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.savefig("Grafer/case_position.pdf")
plt.show()

""" Velocity of cart plot """
plt.plot(time,X.T[4],'r-')
plt.title("Velocity of cart")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/sec]")
plt.savefig("Grafer/case_position_velocity.pdf")
plt.show()

""" Length of string plot """
plt.plot(time[:],X.T[2][:],'r-')
plt.title("Lenght of string")
plt.xlabel("Time [s]")
plt.ylabel("Lenght [m]")
plt.savefig("Grafer/case_string.pdf")
plt.show()

#==============================================================================
# Plot path of container
#==============================================================================

def pos_cargo(y,l,l0,theta):
    x = y + l*sin(theta)
    y = l0 - l*cos(theta)
    return x,y

path = np.zeros([len(X),2])

for i in range(len(X)):
    theta = X[i,0]    
    y = X[i,1]
    l = X[i,2]
    path[i] = pos_cargo(y,l,l0,theta)

dy = 26.3
plt.title("Path of container")
someX0, someY0 = 10, 0 -dy
someX1, someY1 = 10, 2.6 -dy
someX2, someY2 = 10, 5.2-dy 
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((someX0 , someY0 ), 2.438, 2.6, facecolor="grey"))
currentAxis.add_patch(Rectangle((someX1 , someY1 ), 2.438, 2.6, facecolor="grey"))
currentAxis.add_patch(Rectangle((someX2 , someY2 ), 2.438, 2.6, facecolor="grey"))
plt.arrow(1.1,4 -dy,0.2,0.5, head_width=3, head_length=1, fc='b', ec='k')
plt.arrow(50,9.3 -dy,0.1,0, head_width=0.6, head_length=6, fc='b', ec='k')
plt.arrow(100,5-dy,0 ,-0.1, head_width=3, head_length=1, fc='b', ec='k')
plt.axis([-10,110,-1 -dy,10 - dy])
plt.plot(path.T[0],path.T[1] -dy,"b-")
plt.plot([0,100],[0-dy,0-dy],"ro")
plt.xlabel("Pos y [m]")
plt.ylabel("Pos x [m]")
plt.savefig("Grafer/figuresBanekurve.pdf")
plt.show()