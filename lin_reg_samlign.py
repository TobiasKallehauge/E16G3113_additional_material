# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:53:40 2016
@author: G3-113 Mat-Tek 

In this script the linear system with regulator is compared with the non-linear
system. The systems are solved by numerical method. 
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

t_start = 0
t_stop = 50

#Tuning constants
Kp1 = 0
Ki1 = 0
Kd1 = 169.41691896 #Criticaly damped

#Position controll
Kp2 = 0.934739755669057  
Ki2 = 0.000005
Kd2 = 0


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

# ikke lineær model med regulator
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
    u[6],
    u[7]])

# lineær model med regulator
def lin_sys_reg(t,u):
    return np.array([
    u[3],
    u[4],
    u[5],
    ((-g*(m+M) - Kp1)/(M*l0))*u[0] + (Kp2/(M*l0))*u[1] -((Kp2 + Ki2*t)/(M*l0))*ref \
    - (k/m + Kd1/(M*l0))*u[3] + (c + Kd2)/(M*l0)*u[4]  - (Ki1/(M*l0))*u[6] \
    + (Ki2/(M*l0))*u[7],
    ((m*g + Kp1)/M)*u[0] -(Kp2/M)*u[1] + (Kd1/M)*u[3] - ((c+ Kd2)/M)*u[4] \
    + ((Kp2 + Ki2*t)/M)*ref + (Ki1/M)*u[6] - Ki2/M,
    l_acc(t),
    u[0],   
    u[1]]) 


#==============================================================================
# Run simolation with RK45
#==============================================================================
h0 = 0.01


X  = rk.rk_system(lin_sys_reg,x0,t_start,t_stop,h0) #Linear system
X2 = rk.rk_system(sys_reg,x0,t_start,t_stop,h0) #Non linear system

time = np.linspace(t_start,t_stop,len(X.T[0]))

#==============================================================================
# Plot simolation 
#==============================================================================


plt.plot(time,np.rad2deg(X.T[0]),'r-', label = "Linear")
plt.plot(time,np.rad2deg(X2.T[0]),'b-', label = "Non linear")
#plt.plot(time3,np.rad2deg(X3.T[0]),'g-', label = "Non linear")
plt.legend(loc= "lower right")
plt.title("Angles")
plt.xlabel("time [s]")
plt.ylabel("angle [deg]")
#plt.savefig("grafer/vinkel_samling_reg.pdf")
plt.show()


plt.plot(time,X.T[1],'r-', label = "Linear")
plt.plot(time,X2.T[1],'g-', label = "Non linear")
plt.legend(loc= "lower right")
plt.title("Position of cart")
plt.xlabel("time [s]")
plt.ylabel("position [m]")
#plt.savefig("grafer/position_samlign_reg.pdf")
plt.show()


#==============================================================================
# Calculation of the difference
#==============================================================================

maxdif = np.zeros([len(x0)]) # Maximum difference 
avgdif_sum = np.zeros([len(x0)]) # Average difference (sum)

for i in range(len(x0)):
    for j in range(len(time)):
        dif = abs(X.T[i,j]- X2.T[i,j] )
        avgdif_sum[i] += dif
        if dif > maxdif[i]:
            maxdif[i] = dif
        
avgdif = avgdif_sum/len(X)

endedif = abs(X[-1] - X2[-1]) # Difference in the last point

print("Max. difference for angle    : %.4e" %(maxdif[0])) 
print("Avg. difference for angle    : %.4e" %(avgdif[0]))
print("End  difference for angle    : %.4e" %(endedif[0]))

print("Max. difference for position : %.4e" %(maxdif[1])) 
print("Avg. difference for position : %.4e" %(avgdif[1]))
print("End  difference for position : %.4e" %(endedif[1]))






