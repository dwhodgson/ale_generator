import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize
from csv import writer
from tqdm import trange, tqdm

rng = np.random.default_rng(65)

parser = argparse.ArgumentParser()
parser.add_argument('-n',type=int,default=100)
parser.add_argument('-d','--delta',type=float,default=0.01)
parser.add_argument('-E1',type=float,default=1.0)
parser.add_argument('-w', '--width',type=int,default=200)
parser.add_argument('-b','--mhburn',type=int,default=100)
parser.add_argument('-m','--mhloops',type=int,default=500)
parser.add_argument('-r','--rmloops',type=int,default=500)
args = parser.parse_args()

n = args.n
delta = args.delta
E1 = args.E1
width = args.width
mh_burn = args.mhburn
mh_loops = args.mhloops
rm_loops = args.rmloops

n = 10
beta = 1
mh_burn = 100
mh_loops = 100
rm_loops = 250

energies = [E1-i*delta for i in range(width)]
midpoints = [(E+E-delta)/2 for E in energies]

def wrap_add(x,y):
    total = x + y
    return total - np.round(total / (2 * np.pi)) * 2 * np.pi

def action_init(angles):
    return np.sum(np.cos(angles[1:] - angles[:-1]))

# def action_init(angles):
#     total = 0
#     for i in range(1,n):
#         for j in range(i):
#             total = total + np.cos(angles[i]-angles[j])/((i-j)**beta)
#     return total

def action(angles,m):
    return action_init(angles)/m

# def change(angles, jump, j,m):
#     if j==0:
#         return (np.cos(angles[1]-angles[0]-jump)-np.cos(angles[1]-angles[0]))/m
#     if j==n-1:
#         return (np.cos(angles[n-1]+jump-angles[n-2])-np.cos(angles[n-2]-angles[n-1]))/m
#     return (np.cos(angles[j]+jump-angles[j-1])+np.cos(angles[j+1]-angles[j]-jump)-np.cos(angles[j]-angles[j-1])-np.cos(angles[j+1]-angles[j]))/m

def change(angles,jump,j,m):
    total = 0
    for i in range(n):
        if i<j:
            total = total + np.cos(angles[j]+jump-angles[i])/((j-i)**beta) - np.cos(angles[j]-angles[i])/((j-i)**beta)
        elif j<i:
            total = total + np.cos(angles[i]-angles[j]-jump)/((i-j)**beta) - np.cos(angles[i]-angles[j])/((i-j)**beta)
    return total/m

def dist(angles,E1,E2,m):
    a = action(angles,m)
    return 0 if E1 <= a <= E2 else abs(a - (E1 + E2) / 2)

def MH(a,E1,E2,start_angles,start_act,loops,burn,n,m):
    angles = start_angles.copy()
    act = start_act
    total = 0
    result_list = []
#    out_count = 0
#    accepts = 0
    for i in range(burn+loops):
        noise = rng.uniform(-np.pi,np.pi,size=n)
        #noise = rng.normal(0,np.pi*jump,size=n)*jump
        idx = rng.integers(low=0, high=n, size=n)
        #idx = range(n)
        for j in range(n):
            new_angles = angles.copy()
            new_angles[idx[j]] = wrap_add(new_angles[idx[j]],noise[j])
            new_act = action(new_angles,m)
            # chng = change(angles,noise[j],idx[j],m)
            # new_act = act + chng
            if(E1<=new_act<=E2 and np.log(rng.random()) < a * (new_act-act)):
                #angles = new_angles.copy()
                angles[idx[j]] = wrap_add(angles[idx[j]],noise[j])
                act = new_act
#                accepts = accepts+1
#            elif (new_act<E1 or E2<new_act):
#                out_count = out_count+1
        if(i>=burn):
            d = (act-(E1+E2)/2)
            total = total + d
            result = total/(i+1-burn)
            result_list.append(result)
    return result_list[-1]

def RM(E,delta,loops,mh_loops,burn,n,m):
    a = 0
    a_vec = [a]
    gamma = (E - action(np.array([0 if i % 2 == 0 else np.pi for i in range(n)]),m)) / (
        action(np.zeros(n),m) - action(np.array([0 if i % 2 == 0 else np.pi for i in range(n)]),m))
    start = np.array([0 if i % 2 == 0 else (1-gamma)*np.pi for i in range(n)])
    act = action(start,m)
    if (E - delta > act or E < act):
        result = minimize(dist, start, (E - delta, E, m))
        start = result.x
    start_act = action(start,m)
    if (not (E - delta <= start_act <= E)):
        # print(start_act, start[:10])
        # print(result)
        # raise Exception('Starting value not found.')
        return 0
    for i in range(loops):
        e = MH(-a,E-delta,E,start,start_act,mh_loops,burn,n,m)
        a = a + (12/((delta**2)*(i+1)))*e
        a_vec.append(a)
    return a_vec[-1]

results = []
m = action_init(np.zeros(n))
for E in tqdm(energies):
    results.append(RM(E,delta,rm_loops,mh_loops,mh_burn,n,m))
plot_midpoints = []
plot_results = []
for i in range(len(results)):
    if results[i]!=0:
        plot_midpoints.append(midpoints[i])
        plot_results.append(results[i])
plt.cla()
plt.plot(plot_midpoints,plot_results)
plt.savefig('XYNN_check.png')
# filename = f'FullRuns/XYNN_n{n}_d{delta}_{E1}.csv'
# with open(filename,'w',newline='\n') as csvfile:
#     csvwriter = writer(csvfile)
#     csvwriter.writerows(results)
#     csvwriter.writerow(midpoints)
