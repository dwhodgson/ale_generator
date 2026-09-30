import csv
import numpy as np
import cmath
from scipy.special import logsumexp
from math import exp, floor, log
import matplotlib.pyplot as plt
from tqdm import tqdm

def function(E):
    return E**2

def compute_rho(avec,midpoints,delta):
    rho = [None for _ in range(len(midpoints))]
    middle = floor(len(midpoints)/2)
    rho[middle] = 1
    mults = []
    mult = 0
    for i in range(middle-1,-1,-1):
        mult = mult-(delta/2)*(avec[i+1]+avec[i])
        mults.append(mult)
    mults.reverse()
    for i in range(middle):
        try:
            rho[i] = exp(mults[i])
        except(OverflowError):
            print(mults[i],'low')
    mult = 0
    for i in range(middle,len(midpoints)-1):
        mult = mult+(delta/2)*(avec[i+1]+avec[i])
        try:
            rho[i+1] = exp(mult)
        except(OverflowError):
            print(mult,'high')
    return rho

def partition_int(rho,midpoints,delta,etas):
    partition_ints = []
    rho =np.array(rho)
    midpoints = np.array(midpoints)
    for eta in etas:
        logs = np.log(delta) + np.log(rho) + eta*midpoints
        partition_ints.append(logsumexp(logs))
    return partition_ints

def integral(rho,midpoints,delta,etas,partition_ints):
    ints = []
    for j in range(len(etas)):
        logs = []
        for i in range(len(midpoints)):
            logs.append(np.log(delta) + np.log(function(midpoints[i])) + np.log(rho[i]) + etas[j]*midpoints[i])
        ints.append(exp(logsumexp(logs)-partition_ints[j]))
    return ints

def slit_map(c,z):
    return (exp(c) / (2 * z)) * (z ** 2 + 2 * z * (1 - exp(-c)) + 1 + (z + 1) ** 2 * cmath.sqrt(
        (z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1) / ((z + 1) ** 2)))

def building_block(c,z,theta):
    return cmath.exp(theta * 1j) * slit_map(c, cmath.exp(-theta * 1j) * z)

def slit_diff(c,z):
    sq_rt = cmath.sqrt((z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1) / ((z + 1) ** 2))
    prod1 = -(exp(c) / (2 * z ** 2)) * (z ** 2 + 2 * z * (1 - exp(-c)) + 1 + (z + 1) ** 2 * sq_rt)
    prod2 = (exp(c) / (2 * z)) * (2 * z + 2 * (1 - exp(-c)) + 2 * (z + 1) * sq_rt + (z + 1) ** 2 * 0.5 * (1 / sq_rt) * (
            ((2 * z + 2 * (1 - 2 * exp(-c))) * (z + 1) ** 2 - 2 * (z + 1) * (z ** 2 + 2 * z * (1 - 2 * exp(-c)) + 1)
             ) / ((z + 1) ** 4)))
    return prod1 + prod2

def map_diff(z,c,thetas,nn):
    diff = 1
    for i in range(nn - 1, -1, -1):
        diff = diff * slit_diff(c, cmath.exp(-thetas[i] * 1j) * z)
        z = building_block(c, z, thetas[i])
    return diff

# def action_init(angles,c,sigma):
#     return np.sum([-log(abs(map_diff(cmath.exp(sigma+1j*angles[i]),c,angles[:i],i))) for i in range(1,n)])

def action_init(angles,beta):
    total = 0
    for i in range(1,n):
        for j in range(i):
            total = total + np.cos(angles[i]-angles[j])/((i-j)**beta)
    return total

n = 100
# m1_1 = action_init(np.zeros(n),0.1,0.01)
# m1_2 = action_init(np.zeros(n),0.01,0.0001)
# m2_1 = action_init(np.array([0 if i%2==0 else 0.2*np.pi for i in range(n)]),0.1,0.01)
# m2_2 = action_init(np.array([0 if i%2==0 else 0.2*np.pi for i in range(n)]),0.01,0.0001)
# m1 = m1_1-(m1_1+m2_1)/2
# m2 = m1_2-(m1_2+m2_2)/2
m1 = action_init(np.zeros(n),1)
#m2 = action_init(np.zeros(n),2)
#m3 = action_init(np.zeros(n),3)
etas1 = [(-4 + 0.05*i)*m1 for i in range(161)]
#etas2 = [(-4 + 0.05*i)*m2 for i in range(161)]
#etas3 = [(-4 + 0.05*i)*m3 for i in range(161)]
plot_etas = [-4+0.05*i for i in range(161)]
midpoints = []
avec = []
delta = 0.01

energies = [1.0,0.97,0.94,0.91,0.88,0.85,0.82,0.79,0.76,0.73,0.7,0.67,0.64,0.61,0.58,0.55,0.52,0.49,0.46,0.43,0.4,0.37,0.34,0.31,0.28,0.25,0.22,0.19,0.16,0.13,0.1,0.07,0.04,0.01,-0.02,-0.05,-0.08,-0.11,-0.14,-0.17,-0.2,-0.23,-0.26,-0.29,-0.32,-0.35,-0.38,-0.41,-0.44,-0.47,-0.5,-0.53,-0.56,-0.59,-0.62,-0.65,-0.68,-0.71,-0.74,-0.77,-0.8,-0.83,-0.86,-0.89,-0.92,-0.95,-0.98]
loops = [10,50,100,250,500,1000]

for E in energies:
    with open(f'FullRuns/XY_n100_d0.01_mh500_{E}.csv',newline='\n') as file:
        reader = csv.reader(file)
        for value in reader.__next__():
            avec.append(float(value))
        for value in reader.__next__():
            midpoints.append(float(value))

avec.reverse()
midpoints.reverse()

plot_midpoints = []
plot_avec = []
for i in range(len(avec)):
    if avec[i]!=0:
        plot_midpoints.append(midpoints[i])
        plot_avec.append(avec[i])

plt.cla()
plt.plot(plot_midpoints,plot_avec)
plt.savefig('FullRunGraphs/XY_as.png')

rhos = compute_rho(avec,midpoints,delta)
plt.cla()
plt.plot(midpoints,rhos)
plt.savefig('FullRunGraphs/XY_rho.png')

part = partition_int(rhos,midpoints,delta,etas1)
plt.cla()
plt.plot(plot_etas,part)
plt.savefig('FullRunGraphs/XY_part.png')

ints = integral(rhos,midpoints,delta,etas1,part)
plt.cla()
plt.plot(plot_etas,ints)
plt.savefig('FullRunGraphs/XY_energy.png')