"""Projectile Motion Lab

Author: Ricky Gonzales, Lily Howell, Ethan Polley
Version: April 5th, 2024

This code is used for the computations of the projectile motion arrays for displacement, velocity, and acceleration.
"""


import numpy as np
import sympy as sp

#%% Data Read-ins

m = 0.0027

x1, y1, t1 = np.genfromtxt("phys346_t1.csv",skip_header=1, delimiter=',', unpack=True)
x2, y2, t2 = np.genfromtxt("phys346_t2.csv",skip_header=1, delimiter=',', unpack=True)
x3, y3, t3 = np.genfromtxt("phys346_t3.csv",skip_header=1, delimiter=',', unpack=True)
x4, y4, t4 = np.genfromtxt("Phys346_t5.csv", skip_header=1, delimiter=',', unpack=True)
x5, y5, t5 = np.genfromtxt("phys346_t6.csv",skip_header=1, delimiter=',', unpack=True)
x6, y6, t6 = np.genfromtxt("phys346_t7.csv",skip_header=1, delimiter=',', unpack=True)
x7, y7, t7 = np.genfromtxt("phys346_t8.csv",skip_header=1, delimiter=',', unpack=True)
x8, y8, t8 = np.genfromtxt("Phys346_t9.csv", skip_header=1, delimiter=',', unpack=True)
x9, y9, t9 = np.genfromtxt("phys346_t10.csv",skip_header=1, delimiter=',', unpack=True)
x10, y10, t10 = np.genfromtxt("phys346_t12.csv",skip_header=1, delimiter=',', unpack=True)
x11, y11, t11 = np.genfromtxt("phys346_t13.csv",skip_header=1, delimiter=',', unpack=True)

y1d, t1d = np.genfromtxt("Phys346_dropt1.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y2d, t2d = np.genfromtxt("Phys346_dropt2.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y3d, t3d = np.genfromtxt("Phys346_dropt3.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y4d, t4d = np.genfromtxt("Phys346_dropt4.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y5d, t5d = np.genfromtxt("Phys346_dropt5.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y6d, t6d = np.genfromtxt("Phys346_dropt6.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y7d, t7d = np.genfromtxt("Phys346_dropt7.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y8d, t8d = np.genfromtxt("Phys346_dropt8.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)
y9d, t9d = np.genfromtxt("Phys346_dropt9.csv", usecols=(0,1), skip_header=1, delimiter=',', unpack=True)


x = np.zeros(13)
y = np.zeros(13)
t = np.zeros(13)

y_d = np.zeros(13)
t_d = np.zeros(13)

for i in range(13):
    x_avg = np.average([x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i],x8[i],x9[i],x10[i],x11[i]])
    y_avg = np.average([y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i],y8[i],y9[i],y10[i],y11[i]])
    t_avg = np.average([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],t7[i],t8[i],t9[i],t10[i],t11[i]])
    x[i] = x_avg
    y[i] = y_avg
    t[i] = t_avg
    
for i in range(13):
    yd_avg = np.average([y1d[i],y2d[i],y3d[i],y4d[i],y5d[i],y6d[i],y7d[i],y8d[i],y9d[i]])
    td_avg = np.average([t1d[i],t2d[i],t3d[i],t4d[i],t5d[i],t6d[i],t7d[i],t8d[i],t9d[i]])
    y_d[i] = yd_avg
    t_d[i] = td_avg

vx0 = (x[2]-x[0])/ (t[2]-t[0])
vy0 = (y[2]-y[0]) / (t[2] - t[0])
vx1 = (x[3]-x[1]) / (t[3] - t[1])

#%% Differntial Stuff

def velocity_approx(x_array, t_array):
    v_array = np.zeros(10)
    
    for i in range(1,11):
        if i != 11 and i != 0:
            calc_v = (x_array[i+1] - x_array[i-1]) / (t[i+1] - t[i-1])
            v_array[i-1] = calc_v
    return v_array


def accel_approx(v_array, t_array):
    a_array = np.zeros(9)
    
    for i in range(1,9):
        if i != 10 and i != 0:
            calc_v = (v_array[i+1] - v_array[i-1]) / (t[i+1] - t[i-1])
            a_array[i-1] = calc_v
    return a_array


#%% Calculations

vx_array = velocity_approx(x,t)
vy_array = velocity_approx(y,t)
ay_array = accel_approx(vy_array,t)

A = sp.Matrix([[vx0*t[1], vx0*m, -m],
              [vx1*t[2], vx1*m, -m]])

bx = A.rref()[0][2]

print(f"The Calcualted Acceleration due to Gravity from Numerical Differentiaion: {np.average(ay_array):.5f} m/s^2")
print(f"The calculated value of the x-coefficient of drag is: {bx:.5f}")
