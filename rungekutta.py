# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:00:01 2016

@author: Tobias
"""

 # -*- coding: utf-8 -*-

# solving an autonomous syste of ODEs
# using the Runge-Kutta 4 method
# the predator-prey model

from __future__ import division 
import numpy as np


#==============================================================================
# Definition af RK4
#==============================================================================
# one step RK4 adapted to system
def rk4(f, t, x, h):
    "Normal rk4 based on simpsons"
    k1 = f(t, x)
    k2 = f(t + 0.5*h, x + 0.5*h*k1)
    k3 = f(t + 0.5*h, x + 0.5*h*k2)
    k4 = f(t + h, x + h*k3)
    xp = x + h*(k1 + 2.0*(k2 + k3) + k4)/6.0
    return xp, t + h
    
def rk4_2(f,t,x,h):
    "rk4 based on the rkf45 method" 
    k1 = h*f(t,x)
    k2 = h*f(t + 0.25*h, x + 0.25*k1)
    k3 = h*f(t + (3/8)*h, x + (3/32)*k1 + (9/32)*k2)
    k4 = h*f(t + (12/13)*h, x + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = h*f(t + h, x + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
    xp = x + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5
    return xp , t + h

    
def rk5(f,t,x,h):
    "rk5 based on the rkf45 method"
    k1 = h*f(t,x)
    k2 = h*f(t + 0.25*h, x + 0.25*k1)
    k3 = h*f(t + (3/8)*h, x + (3/32)*k1 + (9/32)*k2)
    k4 = h*f(t + (12/13)*h, x + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = h*f(t + h, x + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
    k6 = h*f(t + 0.5*h, x - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
    xp = x + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
    return xp , t + h    

def rk45(f,t,x,h):
    "rk4 and rk5 based on the rkf45 method"
    k1 = h*f(t,x)
    k2 = h*f(t + 0.25*h, x + 0.25*k1)
    k3 = h*f(t + (3/8)*h, x + (3/32)*k1 + (9/32)*k2)
    k4 = h*f(t + (12/13)*h, x + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = h*f(t + h, x + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
    k6 = h*f(t + 0.5*h, x - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
    xp4 = x + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5    
    xp5 = x + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
    return xp4, xp5 , t + h    

def error(f,t,x,h):
    "rk4 and error between rk4 and rk5 basen on the rkf45 method"        
    k1 = h*f(t,x)
    k2 = h*f(t + 0.25*h, x + 0.25*k1)
    k3 = h*f(t + (3/8)*h, x + (3/32)*k1 + (9/32)*k2)
    k4 = h*f(t + (12/13)*h, x + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = h*f(t + h, x + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
    k6 = h*f(t + 0.5*h, x - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
    err = abs(1/360*k1 - (128/4275)*k3 - (2197/75240)*k4 + 1/50*k5 + 2/55*k6)
    ynew = x + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5     
    return err, ynew


def rkf_algorithm(f,u0,t_start,t_stop, tol, h0,epsilon):
    """
    f       :The function for the system of first order ODEs
    u0      :The initial conditions as a point (array)
    t_start :Time at the initial conditions
    t_stop  :Desired evaluation point (actual t_stop will vary if depending on
             on the found step size and how close it multiplies to desired t_stop)
    tol     :Tolerence for error between rk4 and rk5
    h0      :Is initial step size
    rk4     :Is function for runga kutta 4
    rk5     :IS function for runga kutta 5
    """
    s = 100
    step = h0
    counter = 0
    while not (1 - epsilon) <= s <= (1 + epsilon): #Continue until optimal step size is found
        N = int(np.ceil((t_stop - t_start)/step))
        #Number of steps for current stepsize

        #Set up rk4
        U0_rk4 = np.zeros([N + 1,len(u0)]) #Solution array 
        U0_rk4[0] = u0
        
        #Setup rk5
        U0_rk5 = np.zeros([N + 1,len(u0)])
        U0_rk5[0] = u0        
       
        t = t_start

        for i in range(N): #Perform rkf45
            Xp4, Xp5, t = rk45(f, t, U0_rk4[i], step)
            U0_rk4[i+1] = Xp4       
            U0_rk5[i+1] = Xp5      

        error = 0
        
        #Compare errors for alle system variabels and take greatest error
        for i in range(len(u0)):            
            error_variable = abs(U0_rk5[-1,i] - U0_rk4[-1,i])#calulate error for lowst order
            if error_variable > error:
                error = error_variable
                
        if error == 0:
            break
        s = (tol*step/(2*error))**(1/4) #s value acording to formula
        step = step*s #ajust stepsize
        print("Step is %.2e and s is %.2f" %(step, s))
        counter += 1
        
    return U0_rk4, step

    
def rk_system(f,u0,t_start,t_stop, h0):
    """
    f       :The function for the system of first order ODEs
    u0      :The initial conditions as a point (array)
    t_start :Time at the initial conditions
    t_stop  :Desired evaluation point (actual t_stop will vary if depending on
             on the found step size and how close it multiplies to desired t_stop)
    tol     :Tolerence for error between rk4 and rk5
    h0      :Is initial step size
    """
    step = h0
    N = int(np.ceil((t_stop - t_start)/step))

    U0 = np.zeros([N + 1,len(u0)]) #Solution array 
    U0[0] = u0
    t = t_start

    for i in range(N): #Perform rk4
        Xp, t = rk4_2(f, t, U0[i], step)
        U0[i+1] = Xp 

    return U0
    
def rkf45(f,t_start,t_stop,x0,step,tol):
    "Another approach for the adaptive method"
    big = 1e15 #Ensure calculation don't overflow
    h = (t_stop - t_start)/step #inital stepsize
    hmin = h/64 
    hmax = 64*h 
    N = int(np.ceil((t_stop - t_start)/hmin))
    Y = np.zeros([N + 1,len(x0)])
    T = np.zeros([N + 1])
    Y[0] = x0
    T[0] = t_start
    j = 0
    tj = T[0] #Time for j 
    br = t_stop - 0.00001*abs(t_stop); #Ensures that no function is call over h
    signal = False
    while T[j] < t_stop:
        if (T[j] + h) > br:
            h = t_stop - T[j] #Make the last step exactly hit the endpoint
        tj = T[j]
        yj = Y[j]
        err, ynew = error(f,tj,yj,h) #Get error and rk4 with h
        err_max = max(err)
        if (err_max < tol) or (h < 2*hmin) or signal:
        #continioue if tolerence is accepted or if h is less than h min
            signal = False
            Y[j+1] = ynew
            if (tj+h) > br: #If new step is larger than t_stop
                T[j+1] = t_stop
            else:
                T[j+1] = tj + h #Else take a step
            j = j+1;
            tj = T[j]
        else: #If error is over max, then calculate new by using x
            if err_max == 0: #Ensure no devision by zero
                s = 1;
            else:
                s = 0.84*((tol*h)/err_max)**(0.25)
            
            if (s < 0.75) and (h > 2*hmin): #half h if allowed
                h = h/2
            elif (s > 1.50) and (2*h < hmax): #half h if allowed
                h = 2*h
            else:
                signal = True
                # Let algorihm continue if s cannot be adjusted 
                #This is mainly a problem for high tolerences
            #So if s is does not satify the above, let it be
        if (big < abs(max(Y[j]))):
            break
        if t_stop > T[j]:
            step = j+1
        else:
            step = j
    idx = np.nonzero(Y.T[0])[0][-1]
    T = T[:idx + 1]
    Y = Y[:idx + 1]
    return T, Y, step