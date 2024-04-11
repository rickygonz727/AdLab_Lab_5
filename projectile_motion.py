"""Projectile Motion Lab

Author: Ricky Gonzales, Lily Howell, Ethan Polley
Version: April 5th, 2024

This code is used for the computations of the projectile motion arrays for displacement, velocity, and acceleration.
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

#%% Data Read-ins

x1, y1, t1 = np.genfromtxt("phys346_t1.csv",skip_header=1, delimiter=',', unpack=True)
x2, y2, t2 = np.genfromtxt("phys346_t2.csv",skip_header=1, delimiter=',', unpack=True)
x3, y3, t3 = np.genfromtxt("phys346_t3.csv",skip_header=1, delimiter=',', unpack=True)
x5, y5, t5 = np.genfromtxt("Phys346_t5.csv", skip_header=1, delimiter=',', unpack=True)


#%% Calculating Velocities

def x_velocity(x,t):
    return x/t


def y_velocity(y,t, g=9.8):
    return y[0] + ((y[2]-y[1])/(t[2]-t[1])) * t + 0.5*g*(t**2)


#%% System of Differential Equations in the x-direction

def dxdt(vx):
    return vx


def dvxdt(vx, bx, m=0.0027):
    return (bx*(vx**2))/m


#%% System of Differential Equations in the y-direction

def dydt(vy):
    return vy


def dvydt(vy, by, g=9.8, m=0.0027):
    return g - ((by*(vy**2))/m)


if __name__ == "__main__":
    #test
    vx = x_velocity(x1, t1)
    vy = y_velocity(y1,t1)
    
