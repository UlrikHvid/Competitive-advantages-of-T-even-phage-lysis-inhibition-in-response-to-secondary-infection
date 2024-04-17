import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import sys
sys.path.insert("/nbi/nbicmplx/cell/dhm160/Masters-thesis/")
from Plaque_model import *
from Initial_values import *
from Plotters import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor

size = 19
ftarr = np.linspace(1, 10, size)
fbarr = np.linspace(1, 10, size)

ZOIarr = np.empty((size, size))
sizearr = np.empty((size, size))

V = DVS(Rmax=2*10**3, dr=2)
y0 = IVS("MP1", V)
t = 6*60

def compute_for_ft(ft_index, ft):
    # Set the current ft value in the model environment
    local_V = DVS(Rmax=2*10**3, dr=2)
    local_V.f_tau = ft
    local_y0 = IVS("MP1", local_V)
    local_results = []

    for fb_index, fb in enumerate(fbarr):
        local_V.f_beta = fb
        sim = MPShell("MP1", local_y0, local_V, t)
        zoi = rhalf(sim, LIN=True, V=local_V, var="B")[-1]
        size = rhalf(sim, LIN=True, V=local_V, var="Btot")[-1]
        local_results.append((ft_index, fb_index, zoi, size))
    
    return local_results

# Use ProcessPoolExecutor to parallelize
with ProcessPoolExecutor() as executor:
    # Map function over ftarr with index
    results = list(executor.map(compute_for_ft, range(size), ftarr))
    
    # Gather results
    for local_results in results:
        for ft_index, fb_index, zoi, size in local_results:
            ZOIarr[ft_index, fb_index] = zoi
            sizearr[ft_index, fb_index] = size
            print(f"Progress = {ft_index/size}")


np.savetxt("MP1ZOI.csv", ZOIarr, delimiter=",")
np.savetxt("MP1plaquesize.csv", sizearr, delimiter=",")