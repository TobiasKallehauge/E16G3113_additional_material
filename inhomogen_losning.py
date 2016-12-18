# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:17:43 2016

@author: G3-113

This script compute the analytical solution to the inhomogeneous linear system 
by the exponential matrix. This is compared to the numerical somution to the non
linear system. 
To obtain specefic plot in section 7.3(repport) initial values has to be changed.    

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import rungekutta as rk
import scipy as sp
import time

cos = np.cos
sin = np.sin

#==============================================================================
# Initialise parameters, intial conditions and the coefficient matrices
#==============================================================================

m = 20      # Mass of load
M = 10      # Mass of cart
k = 5       # Friction on load
c = 10      # Friction on cart
g = 9.82    # Gravitaional acceleration

def coefM(m,M,k,c,g,l0):
    """ define coefficient matrix A """
    return np.matrix([    
    [0,0,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
    [-((g*(m+M))/(M*l0)),0,0,-k/m,c/(M*l0),0],
    [(m*g)/M,0,0,0,-c/M,0],
    [0,0,0,0,0,0]
    ])
    
def coefB(M,l0):
    """ define coefficient matrix B """
    return np.matrix([
    [0,0],
    [0,0],
    [0,0],
    [-(1/(M*l0)),0],
    [1/M,0],
    [0,1]])

#time interval
t_start = 0
t_stop = 30
    
""" Initial conditions """
#change for different outcome 
theta0 = np.deg2rad(0.5) # initial angle 
y0 = 0                   # position 
l0 = 5                   # linght of string
omega0 = 0               # angle velocity
my0 = 0                  # velocity
phi0 = 0                 # string velocity
fm = 5                   # external motor force
lacc = 0                 # acceleration of string
x0 = np.array([theta0,y0,l0,omega0,my0,phi0])
u = np.array([fm,lacc])

""" Coefficient matrices with the parameters """
A = coefM(m,M,k,c,g,l0)
B = coefB(M,l0)

#==============================================================================
# Calculate eigenvalues and eigenvectors and nonzero eigenvalues
#==============================================================================

eigval, eigvec = np.linalg.eig(A)
eigvec = eigvec.T

""" Pick linear independent eigenvectors """
print("Remember to manually appoint eigenvectors to eigenvalues, and complex conjugate")
e0 = np.array(eigvec[0])[0]
e1 = np.array(eigvec[1])[0]
e2 = np.array(eigvec[2])[0]
e3 = np.real(np.array(eigvec[3])[0])
e4 = np.array(eigvec[4])[0]

""" Index complex conjugate (pair) eigenvalue """
cmplx_eigval = [eigval[1]]
cmplx_eigvec = [e1]

""" Index real eigenvalue != 0 """
real_eigval = [eigval[3]]
real_eigvec = [e3]

""" Index zero eigenvalue """
e0 = np.real(e0)
e4 = np.real(e4)
e5 = np.array([0,0,0,0,0,1])
chains = [[e0],[e4,e5]]
nr_zero_sol = 3

#==============================================================================
# Inhomogenous solution
#==============================================================================

""" Define T and Jordan matrix  for transformation """
T = np.matrix(np.vstack([e4,e5,e0,e3,np.real(e1),np.imag(e1)])).T

J0 = np.matrix([[0,1,0],[0,0,0],[0,0,0]])
Jt = np.matrix([[eigval[3],0                  ,0                 ],
                [0        ,np.real(eigval[1]) ,np.imag(eigval[1])],
                [0        ,-np.imag(eigval[1]),np.real(eigval[1])]])


Jleft = np.vstack([J0,np.zeros([3,3])])
Jright = np.vstack([np.zeros([3,3]),Jt])
J = np.hstack([Jleft,Jright])

""" integrate Jordan matrix"""
def diag_integral(t,J):
    Jinv = np.linalg.inv(J)
    I = np.diag([1,1,1])
    expJ = sp.linalg.expm2(-J*t)
    return Jinv.dot(I - expJ)

def J0_integral(t,J0):
    I = np.diag([1,1,1])
    return I*t - ((t**2)/2)*J0

""" partiel solution of inhomogeneous linear system """
def inhomo_sol(a,b,J0,Jt,T,B,u):
    Tinv = np.linalg.inv(T)
    Jt_int = diag_integral(b,Jt)
    J0_int = J0_integral(b,J0)
    Jleft = np.vstack([J0_int,np.zeros([3,3])])
    Jright = np.vstack([np.zeros([3,3]),Jt_int])
    J_int = np.hstack([Jleft,Jright])
    return (T.dot(J_int).dot(Tinv).dot(B).dot(u)).T

""" Final solution of inhomogeneous linear system """
def exp_sol(t,A,J0,Jt,T,B,u,x0):
    expA = sp.linalg.expm(A*t)
    int_expA = inhomo_sol(0,t,J0,Jt,T,B,u)
    return expA.dot(x0) + np.array((expA.dot(int_expA)).T)[0]
#==============================================================================
# Numerical solutions of non linear system
#==============================================================================

""" Equations paramtres """

def force(t):
    return fm
    
def l_acc(t):
    return lacc
    
""" Definition of the function """

#Non-linear system
def sys(t,u):
    l = u[2]    
    return np.array([\
    u[3], \
    u[4], \
    u[5], \
    -(2*u[5]*u[3])/l - (k/m)*u[3] - (g/l)*sin(u[0]) + (cos(u[0])*(c*u[4] \
    - force(t) + m*sin(u[0])*(l_acc(t) - g*cos(u[0]) - l*u[3]**2)))/ \
    (l*(M + m*sin(u[0])**2)), \
    (force(t) - c*u[4] + m*sin(u[0])*(l*u[3]**2 + g*cos(u[0]) - l_acc(t)))/(m*sin(u[0])**2 + M), \
    l_acc(t)])

#Numerical system for linear system
def linsys(t,u):   
    return np.array([
    u[3],
    u[4],
    u[5],
    ((-g*(m+M))/(M*l0))*u[0] - (k/m)*u[3] + (c/(M*l0))*u[4] - (1/(M*l0))*force(t),
    ((m*g)/M)*u[0] - (c/M)*u[4] + force(t)/M,
    l_acc(t)])

tol = 1e-12 # Tolerance
h0 = 0.5 #Inital 
epsilon = 0.01

#Get RKF45 solution and get calculation time for it
t0 = time.time()
X, step  = rk.rkf_algorithm(sys, x0, t_start, t_stop, tol, h0,epsilon)
t1 = time.time()
t_RKF45 = t1 - t0


times = np.linspace(t_start,t_stop,len(X.T[0]))


sol3 = np.zeros([len(times),len(x0)],dtype = "complex64")

#Calculate inhomogenious system with eksponential matrix
for stepx in range(len(times)):
    sol3[stepx] = exp_sol(times[stepx],A,J0,Jt,T,B,u,x0)
    
sol3 = np.real(sol3)
#==============================================================================
# Comparison of analytical and numerical - Grahps and difference
#==============================================================================
""" details about  RK-F """
print ("RK4 calulation time: %g s" %(t_RKF45))

""" eigenvalue spektrum """
for i in range(len(eigval)):
    re = np.real(eigval[i])
    im = np.imag(eigval[i])
    plt.plot(re,im,"o", label =i)

minre = np.min(np.real(eigval))*1.2
minim = np.min(np.imag(eigval))*1.2
if minim == 0:
    minim = minre
plt.plot([minre,-minre*0.5],[0,0],"k-")
plt.plot([0,0],[minim,-minim],"k-")
plt.title("Eigenvalue spectrum")
plt.xlabel("[Re]")
plt.ylabel("[Im]")
#plt.savefig("grafer/spektrum.pdf")
plt.show()

""" Graph for Angles """
plt.plot(times,np.rad2deg(X.T[0]),'g-', label= "Non linear")
plt.plot(times,np.rad2deg(sol3.T[0]),'r-', label= "Linearized")
plt.title("Non linear and linearized solutions for angle")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.legend(loc="lower right")
#plt.savefig("grafer/vinkel.pdf")
plt.show()


""" Graph for position of cart """
plt.plot(times,X.T[1],'g-', label= "Non linear")
plt.plot(times,sol3.T[1],'r-', label= "Linearized")
plt.title("Non linear and linearized solutions for position of cart")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend(loc="lower right")
#plt.savefig("grafer/position.pdf")
plt.show()

""" Lenght of string """
plt.plot(times,X.T[2],'g-', label= "Non linear")
plt.title("Lenght of string")
plt.xlabel("time [s]")
plt.ylabel("Lenght [m]")
#plt.savefig("grafer/string.pdf")
plt.show()


#==============================================================================
# Calculation of the difference 
#==============================================================================

maxdif = np.zeros([len(x0)]) # Maximum difference 
avgdif_sum = np.zeros([len(x0)]) # Average difference (sum)

for i in range(len(x0)):
    for j in range(len(times)):
        dif = abs(sol3.T[i,j]-X.T[i,j] )
        avgdif_sum[i] += dif
        if dif > maxdif[i]:
            maxdif[i] = dif
        
avgdif = avgdif_sum/len(sol3)

endedif = abs(sol3[-1] - X[-1]) # Difference in the last point

print("Max. difference for the angle: %.4e" %(maxdif[0])) 
print("Avg. difference for the angle: %.4e" %(avgdif[0]))
print("End point difference for the angle: %.4e" %(endedif[0]))
print("")  
print("Max. difference for position of cart: %.4e" %(maxdif[1])) 
print("Avg. difference for position of cart: %.4e" %(avgdif[1]))
print("End point difference for position of cart: %.4e" %(endedif[1]))
