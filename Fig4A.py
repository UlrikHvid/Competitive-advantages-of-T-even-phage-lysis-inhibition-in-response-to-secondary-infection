import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import sys
import os
sys.path.append(os.getcwd())
from Plaque_model import *
from Initial_values import *
from Plotters import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def main():
    length = 4 #ALSO DEFINED BELOW!
    ftarr = np.linspace(1, 10, length)

    ZOIarr      = np.empty((length, length)) #Zones of infection
    psizearr    = np.empty((length, length)) #Plaque sizes

    #r-mutant reference values. Calculated in "playground" notebook
    psize0  = 1192 #microns 
    zoi0    = 1306 #microns

    # Use ProcessPoolExecutor to parallelize
    with ProcessPoolExecutor() as executor:
        # Map function over ftarr with index
        results = list(executor.map(compute_for_ft, range(length), ftarr))

        # Gather results
        for local_results in results:
            for ft_index, fb_index, zoi, psize in local_results:
                ZOIarr[ft_index, fb_index] = zoi/zoi0
                psizearr[ft_index, fb_index] = psize/psize0
                print(f"Progress = {ft_index/length}",flush = True)
    
    np.savetxt("MP1ZOI.csv", ZOIarr, delimiter=",")
    np.savetxt("MP1plaquesize.csv", psizearr, delimiter=",")

def compute_for_ft(ft_index, ft):
    length = 4
    fbarr = np.linspace(1, 10, length)
    t = 5*60
    # Set the current ft value in the model environment
    local_V = DVS(Rmax=2*10**3, dr=2)
    local_V.f_tau = ft
    local_y0 = IVS("MP1", local_V)
    local_results = []

    for fb_index, fb in enumerate(fbarr):
        local_V.f_beta = fb
        result  = MPShell("MP1", local_y0, local_V, t, progress = True)[-1] #Simulate and pick out only last frame
        B       = result[0]
        zoi     = np.where(B > B[-1]/2)[0][0]*local_V.dr   
        Btot    = np.sum(result[:(1+LIN)*local_V.N+1],axis = 0)    #Sum all bacteria
        psize   = np.where(Btot >= Btot[-1]/2)[0][0]*local_V.dr   #Picks out the index, multiplied by dr
        local_results.append((ft_index, fb_index, zoi, psize))
    
    return local_results

if __name__ == "__main__":
    main()

