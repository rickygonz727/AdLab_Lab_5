"""Projectile Motion Lab

Author: Ricky Gonzales, Lily Howell, Ethan Polley
Version: April 5th, 2024

This code is used for the computations of the projectile motion arrays for displacement, velocity, and acceleration.
"""


import numpy as np
import sympy as sp

#%% Data Read-ins

m = 0.0027 #The measured mass of the ping-pong ball

#The following chunks of code read-in the data
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


#%% Differntial Stuff

def velocity_approx(x_array, t_array):
    v_array = np.zeros(10) #This establishes
    
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

#We define some empty arrays for the projectile motion data
x = np.zeros(13)
y = np.zeros(13)
t = np.zeros(13)

#We then define some empty arrays for the dropping trials
y_d = np.zeros(13)
t_d = np.zeros(13)

#In order to make an array of averages, we had two loops to get the averages

for i in range(13):
    #This loop starts off by taking the averages of the i'th element
    x_avg = np.average([x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i],x8[i],x9[i],x10[i],x11[i]])
    y_avg = np.average([y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i],y8[i],y9[i],y10[i],y11[i]])
    t_avg = np.average([t1[i],t2[i],t3[i],t4[i],t5[i],t6[i],t7[i],t8[i],t9[i],t10[i],t11[i]])
    #Then we add the average values to the empty arrays
    x[i] = x_avg
    y[i] = y_avg
    t[i] = t_avg
    
#We do a similar process for the dropping trials
for i in range(13):
    yd_avg = np.average([y1d[i],y2d[i],y3d[i],y4d[i],y5d[i],y6d[i],y7d[i],y8d[i],y9d[i]])
    td_avg = np.average([t1d[i],t2d[i],t3d[i],t4d[i],t5d[i],t6d[i],t7d[i],t8d[i],t9d[i]])
    y_d[i] = yd_avg
    t_d[i] = td_avg

#After we had defined the x,y,t arrays for the projectile and dropping, we calculate some velocities using 
#Numerical Differentiation

vx0 = (x[2]-x[0])/ (t[2]-t[0]) #FOr the first value of velocity, we consider the 2nd and 0th data points
vx1 = (x[3]-x[1]) / (t[3] - t[1]) #For the second value of velocity, we consider the 3rd and 1st data points

vx_array = velocity_approx(x,t) #Then, using our function, we numerically differentiate to get the velocity in the x direction
vy_array = velocity_approx(y,t) #We use the function again to numerically differentiate to get the velocity in they direction
ay_array = accel_approx(vy_array,t) #Once we have the velocity in the y-direction, we conduct a similar process to get the acceleration

vy_d_array = velocity_approx(y_d,t_d) #We make similar calculations for the dropping trials
ay_d_array = accel_approx(vy_d_array,t)

#After analyzing the equations we had, we considered the second and first data points and create a matrix with two unknowns, bx and cx. 
#Since we also know the mass of the ping pong ball, we also considered it in our system of equations
#%% Matrices
A = sp.Matrix([[vx0*t[1], vx0*m, -m],
              [vx1*t[2], vx1*m, -m]])

B = sp.Matrix([[1,-(np.average(vy_array)**2)/m, np.average(ay_array)],
               [1,-(np.average(vy_d_array)**2)/m, np.average(ay_d_array)]])

#With these two matrices, the first to find bx and the second to find one version of g and by, we then row reduce to obtain our approximations
bx = A.rref()[0][2]  #This is the drag coefficient in the x direction
rref_avg = B.rref()[0][2] #This is one of the versions of the acceleration due to gravity
by = B.rref()[0][5] #This is the drag coefficient in the y direction


#%% Obtaining Final Values

#With our acceleration arrays for both the projectile motion and the dropping trials, we average them to get the average acceleration due to gravity
grav_p = np.average(ay_array)
grav_d = np.average(ay_d_array)

#Once we have all of our numbers for the gravity from row reduction, and averaging the acceleration arrays in the y-direction
#We then build our array of the acceleration due to gravity
accel_avg = np.zeros(3)
accel_avg[0] = grav_p
accel_avg[1] = grav_d
accel_avg[2] = rref_avg

#Once our array has been built, we then average the value and we have our approximation for the acceleration due to gravity, which is ~ 10.8. 

g = np.average(accel_avg)

#%% Printing Final Calculations

print(f"The Calcualted Acceleration due to Gravity: {g:.5f} m/s^2")
print(f"The calculated value of the x-coefficient of drag is: {bx:.5f}")
print(f"The calculated value of the x-coefficient of drag is: {by:.5f}")
