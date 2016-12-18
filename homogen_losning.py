# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:53:40 2016
@author: G3-113 Mat-Tek 

This script compute the analytical solution, to a linear homogeneous system
of first order ODEs. 
The eigenvalue method, including the fundamental matrix, is used, and later
compred to the solution with the exsponential matrix and non linear solution.   
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import rungekutta as rk
import scipy as sp

cos = np.cos
sin = np.sin

#==============================================================================
# Initialise parameters, intial conditions and the coefficient matrix
#==============================================================================

m = 20      #Mass of load
M = 10      #Mass of slate
k = 5       #Friction on load
c = 10      #Friction on slate
g = 9.82    #Gravitaional acceleration

def coefM(m,M,k,c,g,l0):
    "define the coefficient matrix"
    return np.matrix([    
    [0,0,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
    [-((g*(m+M))/(M*l0)),0,0,-k/m,c/(M*l0),0],
    [(m*g)/M,0,0,0,-c/M,0],
    [0,0,0,0,0,0]
    ])
    
t_start = 0
t_stop = 30
iterationer = 500 #number of evaluations in time period
    
#Initial conditions
theta0 = np.deg2rad(0.5) # initial angle 
y0 = 0                   # position 
l0 = 5                   # linght of string
omega0 = 0               # angle velocity
my0 = 0                  # velocity
phi0 = 0                 # string velocity
x0 = np.array([theta0,y0,l0,omega0,my0,phi0])

#Coefficient matrix
A = coefM(m,M,k,c,g,l0)

#==============================================================================
# Calculate eigenvalues and eigenvectors nonzero eigenvalues
#==============================================================================

eigval, eigvec = np.linalg.eig(A)
eigvec = eigvec.T

#Pick linear independent eigenvectors
print("Remember to manually appoint eigenvectors to eigenvalues, and complex conjugate")
e0 = np.array(eigvec[0])[0]
e1 = np.array(eigvec[1])[0]
e2 = np.array(eigvec[2])[0]
e3 = np.array(eigvec[3])[0]
e4 = np.array(eigvec[4])[0]

#Index complex conjugate (pair) eigenvalue
cmplx_eigval = [eigval[1]]
cmplx_eigvec = [e1]

#Index real eigenvalue != 0
real_eigval = [eigval[3]]
real_eigvec = [e3]

#Index for zero eigenvalue
e0 = np.real(e0)
e4 = np.real(e4)
e5 = np.array([0,0,0,0,0,1])
chains = [[e0],[e4,e5]]
nr_zero_sol = 3


#==============================================================================
# Complex eigenvalue solutions
#==============================================================================

def complex_sol(eigvec,eigval,t):
    x = np.zeros([len(cmplx_eigvec)*2,len(cmplx_eigvec[0])]) #Adds value of pairs of complex solutions    
    for i in range(len(cmplx_eigvec)):
        alpha = np.real(eigval[i])
        beta = np.imag(eigval[i])
        u = np.real(eigvec[i])
        v = np.imag(eigvec[i])
        x[i*2] = np.exp(alpha*t)*(u*cos(beta*t) - v*cos(beta*t))
        x[i*2 +1] = np.exp(alpha*t)*(v*cos(beta*t) + u*cos(beta*t))
    return x

#example 
example_cmplx = complex_sol(cmplx_eigvec,cmplx_eigval,0)

#==============================================================================
# Real eigenvalue solutions
#==============================================================================

def real_sol(eigvec,eigval,t):
    x = np.zeros([len(eigvec),len(eigvec[0])]) #Adds value of pairs of complex solutions    
    for i in range(len(eigvec)):
        lampda = np.real(eigval[i])
        x[i] = np.real(np.exp(lampda*t)*eigvec[i])
    return x

#example
example_real = real_sol(real_eigvec,real_eigval,0)

#==============================================================================
# Solutions for zero eigval
#==============================================================================

def zero_sol(nr_zero_sol,chains,t):
    dim = len(chains[0][0])
    x = np.zeros([nr_zero_sol,dim])
    for i in range(len(chains)):
        for j in range(len(chains[i])):
            for k in range(j+1):
                x[i+j] += (chains[i][k]*t**(j-k))/np.math.factorial(j-k)
    return x
    
#example
example_zero = zero_sol(nr_zero_sol,chains,2)

#==============================================================================
# Final soltion
#==============================================================================

def fundemental_matrix(complex_eigvec, cmplx_eigval,real_eigvec,real_eigval,nr_zero_sol,chains,t):
    """ Define fundamental matrix as phi(t) """
    cmplx = complex_sol(cmplx_eigvec,cmplx_eigval,t)    
    real = real_sol(real_eigvec,real_eigval,t)    
    zero = zero_sol(nr_zero_sol,chains,t)
    phi_t = np.vstack([cmplx,real,zero]).T
    return phi_t
  
def total_sol(phi_t,phi_0inv,x0):
    """ Define the solution x(t) from the fundamental matrix, and initial values """
    return phi_t.dot(phi_0inv).dot(x0)

#Get phi_0 to avoid recalculating it later on
phi_0 = fundemental_matrix(cmplx_eigvec,cmplx_eigval,real_eigvec,real_eigval,nr_zero_sol,chains,0)
phi_0_inv = np.linalg.inv(phi_0)

#Assign all solutions from t_start to t_slut to an array of lenght iterationer
time = np.linspace(t_start,t_stop,iterationer)
sol = np.zeros([len(time),len(x0)])

#Calculate solution by the fundemental matrix
for step in range(len(time)):
    phi_t = fundemental_matrix(cmplx_eigvec,cmplx_eigval,real_eigvec,real_eigval,nr_zero_sol,chains,time[step])
    sol[step] = total_sol(phi_t,phi_0_inv,x0)


#==============================================================================
# Eksponential matrix solution
#==============================================================================
"""Comparison to tjek relation between fundamental matrix and exponential matrix """
sol2 = np.zeros([len(time),len(x0)])

def exp_sol(t,A,x0):
    return sp.linalg.expm(A*t).dot(x0)

for step in range(len(time)):
    sol2[step] = exp_sol(time[step],A,x0)

#==============================================================================
# Numerical solutions of non linear system
#==============================================================================
""" Equations paramtre """
sin = np.sin
cos = np.cos

# definition of inhomogeneous part of system, therefore zero. 
def force(t):
    if 0 <= t <= 30:
        return 0
    else:
        return 0
    
def l_acc(t):
    return 0

# Definition of the function
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

tol = 1e-12 # Tolerance
h0 = 0.6
epsilon = 0.01
# numerical solution
X, step = rk.rkf_algorithm(sys,x0,t_start,t_stop,tol,h0,epsilon)
time2 = np.linspace(t_start,t_stop,len(X.T[0]))

#=============================================================================
# Comparison of linear solution (both fundamentalmatrix and exponentialmatrix)
# with non linear solution.    
#==============================================================================
plt.plot(time,np.rad2deg(sol.T[0]),"r",label="Linear fund")
plt.plot(time,np.rad2deg(sol2.T[0]),"b",label="Linear exp")
plt.plot(time2,np.rad2deg(X.T[0]),'g-', label= "Non linear")
plt.title("Angles")
plt.xlabel("time [s]")
plt.ylabel("angle [deg]")
plt.legend(loc="upper right")
#plt.savefig("grafer/vinkel.pdf")
plt.show()
 

plt.plot(time,sol.T[1],"r",label="Linear fund")
plt.plot(time,sol2.T[1],"b",label="Linear exp")
plt.plot(time2,X.T[1],'g-', label= "Non linear")
plt.title("Position of cart")
plt.xlabel("time [s]")
plt.ylabel("position [m]")
plt.legend(loc="lower right")
#plt.savefig("grafer/position.pdf")
plt.show()


    