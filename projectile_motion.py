"""Projectile Motion Lab

Author: Ricky Gonzales, Lily Howell, Ethan Polley
Version: April 5th, 2024

This code is used for the computations of the projectile motion arrays for displacement, velocity, and acceleration.
"""


import numpy as np
import matplotlib.pyplot as plt

x1, y1, t1 = np.genfromtxt("phys346_t1.csv",skip_header=1, delimiter=',', unpack=True)
x2, y2, t2 = np.genfromtxt("phys346_t2.csv",skip_header=1, delimiter=',', unpack=True)
x3, y3, t3 = np.genfromtxt("phys346_t3.csv",skip_header=1, delimiter=',', unpack=True)
x5, y5, t5 = np.genfromtxt("Phys346_t5.csv", skip_header=1, delimiter=',', unpack=True)

