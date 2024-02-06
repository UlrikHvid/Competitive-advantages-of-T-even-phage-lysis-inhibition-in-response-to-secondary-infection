import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares,minimize,root,brute,fsolve,approx_fprime
from scipy.linalg import eig
import sys
from Broth_model import *
from Initial_values import *
from Plotters import *
from tqdm import tqdm

################################################################################
#Fig. 1C
V = DV()

side = 20
ratarr = np.zeros((side,side))
ftauarr = np.linspace(1,10,side)
fbetaarr = np.linspace(1,10,side)

#Initial values
y00 = IV("M0",comp = 0)
y00[-2] = 10**3
y10 = IV("M1",comp = 0)
y10[-2] = 10**3

t = 20*60
for it,ftau in enumerate(tqdm(ftauarr)):
    for ib,fbeta in enumerate(ftauarr):
        V.f_tau = ftau
        V.f_beta = fbeta
        sol0 = solve_ivp(M0,[0,t],y00,args = Const(V,"M0"))
        sol1 = solve_ivp(M1,[0,t],y10,args = Const(V,"M1"))
        ratarr[ib,it] = sol1.y[-2,-1]/sol0.y[-2,-1]

################################################################################
#Fig. 2B
V = DV()
V.comp = 1

side = 20
ratarr = np.zeros((side,side)) #Figure data goes here
ftauarr = np.linspace(1,10,side)
fbetaarr = np.linspace(1,10,side)

#Initial values
y0 = IV("M1",comp = V.comp)
y0[-2] = 10**3
y0[-3] = 10**3

t = 20*60
for it,ftau in enumerate(tqdm(ftauarr)):
    for ib,fbeta in enumerate(ftauarr):
        V.f_tau = ftau
        V.f_beta = fbeta
        sol = solve_ivp(M1,[0,t],y0,args = Const(V,"M1"))
        ratarr[ib,it] = sol.y[-2,-1]/sol.y[-3,-1]

################################################################################
#Fig. S1 B
V = DV()
V.comp = 1
V.rtrig = False

side = 20
ratarr = np.zeros((side,side)) #Figure data goes here
ftauarr = np.linspace(1,10,side)
fbetaarr = np.linspace(1,10,side)

#Initial values
y0 = IV("M1",comp = V.comp)
y0[-2] = 10**3
y0[-3] = 10**3

t = 20*60
for it,ftau in enumerate(tqdm(ftauarr)):
    for ib,fbeta in enumerate(ftauarr):
        V.f_tau = ftau
        V.f_beta = fbeta
        sol = solve_ivp(M1,[0,t],y0,args = Const(V,"M1"))
        ratarr[ib,it] = sol.y[-2,-1]/sol.y[-3,-1]