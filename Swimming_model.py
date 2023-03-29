import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from time import time
from tqdm import tqdm
import sys
sys.path.insert(-1,"/nbi/nbicmplx/cell/dhm160")
from Broth_model import *
from Plaque_model import *
from Initial_values import *

###########################################################################

#Subfunctions

def MatrixAd(l): #For computing the gradient for the advection speed
    subdiag = np.ones(l-1)*(-1/2)
    subdiag[-1] = 0
    supdiag = np.ones(l-1)*1/2
    supdiag[0] = 1
    diag = np.zeros(l)
    diag[0] = -1
    return sp.diags([subdiag,diag,supdiag],[-1,0,1])

def AdSpeed(chi,am,ap,n,n_matrix,dr,simple = False): #Find advection speed
    if simple:
        vector = n
    else:
        vector = np.log(  (1+n/am) / (1+n/ap)  )
    return chi/dr*n_matrix*vector

def Advect(nex,prev,vbord,dA,areaarr): #Advection algorithm
    dF        =  dA*prev[:-1]*(vbord>0) + dA*prev[1: ]*(vbord<0)
    nex[:-1] -= dF/areaarr[:-1] #Left of border
    nex[1:]  += dF/areaarr[1: ] #Right of border
    return nex

###########################################################################

#With swimming

def MSShell(model,y0,V,t,frames=False):
    if not frames:
        frames = int(t/V.tau0) #Default value for frames
    its          = int(t/V.dt)
    sim          = np.zeros((frames,len(y0),V.l))
    frameind     = np.linspace(0,its-1,frames,dtype = int)
    gn0          = Gamma(V.gnmax,V.n0,V.Kn)
    DMn,DMP,DMB  = V.Dn*Matrix(V.l)/V.dr**2, V.DP*Matrix(V.l)/V.dr**2,V.DB*Matrix(V.l)/V.dr**2
    rarr         = np.linspace(0,V.Rmax,V.l)
    areaarr      = Area(V.dr,np.arange(1,V.l+1))
    cellgrid     = np.meshgrid(rarr,indexing = "ij",sparse = True)
    n_matrix     = MatrixAd(V.l)
    ynext        = np.copy(y0)
    V.eta       *= 1/V.da
    if model == "MS0":
        for i in tqdm(range(its)):
            if i in frameind:
                sim[np.where(frameind == i)] = ynext
            ynext = MS0(ynext,V,gn0,DMB,DMP,DMn,rarr,areaarr,n_matrix,cellgrid)
    else:
        for i in tqdm(range(its)):
            if i in frameind:
                sim[np.where(frameind == i)] = ynext
            ynext = MS1(ynext,V,gn0,DMB,DMP,DMn,rarr,areaarr,n_matrix,cellgrid)
    return sim

def MS0(y,V,gn0,DMB,DMP,DMn,rarr,areaarr,n_matrix,cellgrid):
    #prev: First dimension represents variable (B,P etc.). Second represents space coordinate
    N,gnmax,Kn,eta,tau0,beta0,rl,rb,Y,delta,chi,am,ap,dt,dr,l = V.N,V.gnmax,V.Kn,V.eta,V.tau0,V.beta0,V.rl,V.rb,V.Y,V.delta,V.chi,V.am,V.ap,V.dt,V.dr,V.l
    B,P,n        = y[0],y[-2],y[-1]
    gn           = Gamma(gnmax,n,Kn)
    tau          = Tau(tau0   ,rl,gn0,gn)/N
    beta         = Beta(beta0 ,rb,gn0,gn)
    dydt         = np.zeros_like(y)
    dydt[0]      = (gn - eta*P)*B + DMB*B
    dydt[1]      = eta*P*B - y[1]/tau + DMB*y[1]
    for i in range(N-1): 
        dydt[2+i]= (y[1+i] - y[2+i])/tau + DMB*y[2+i]
    dydt[-2]     = beta*y[N]/tau  - (delta + eta*sum(y[:N+1]))*P + DMP*P
    dydt[-1]     = -gn*B/Y + DMn*n
    nex          = y+dt*dydt
    #Advection
    v            = AdSpeed(chi,am,ap,n,n_matrix,dr)
    vint         = RegularGridInterpolator(cellgrid,v)
    vbord        = vint((rarr[:l-1]+dr/2).reshape(l-1,1))
    dA           = np.pi*vbord*dt*(2*np.arange(1,l)*dr - vbord*dt)
    for i in range(N+1):
        nex[i] = Advect(nex[i],y[i],vbord,dA,areaarr)
    return nex

def MS1(y,V,gn0,DMB,DMP,DMn,rarr,areaarr,n_matrix,cellgrid):
    #prev: First dimension represents variable (B,P etc.). Second represents space coordinate
    N,gnmax,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,delta,chi,am,ap,dt,dr,l,comp = V.N,V.gnmax,V.Kn,V.eta,V.tau0,V.f_tau,V.beta0,V.f_beta,V.rl,V.rb,V.Y,V.delta,V.chi,V.am,V.ap,V.dt,V.dr,V.l,V.comp
    B,P,n          = y[0],y[-2],y[-1]
    gn             = Gamma(gnmax,n,Kn)
    tau            = Tau(tau0   ,rl,gn0,gn)/N
    tau_I          = f_tau*tau
    beta           = Beta(beta0 ,rb,gn0,gn)
    beta_I         = f_beta*beta
    P2             = y[-3] if comp else 0
    dydt           = np.zeros_like(y)
    dydt[0]        = (gn - eta*(P+P2))*B + DMB*B
    dydt[1]        = eta*P*B - (eta*(P+P2) + 1/tau)*y[1] + DMB*y[1] #First infected state
    dydt[N+1]      = eta*(P+P2)*sum(y[1:N+1])         - y[N+1]/tau_I  + DMB*y[N+1] #First inhibited state
    for i in range(2,N+1):  #Rest of infected and inhibited states
        dydt[i]    =  (y[i-1]  - y[i]  )/tau   - eta*(P+P2)*y[i]  + DMB*y[i]
        dydt[N+i]  = ((y[N+i-1]- y[N+i])/tau_I)+ DMB*y[N+i]
    if comp:
        dydt[2*N+1]= (eta*P2*B - y[2*N+1]/tau) + DMB*y[2*N+1] #First P2-infected state
        for i in range(2,N+1): #Rest of P2-infected states
            dydt[2*N+i]= (y[2*N+i-1] - y[2*N+i])/tau + DMB*y[2*N+i]
        dydt[-3]   = beta*y[3*N]/tau - P2*(eta*sum(y[:3*N+1])+delta) + DMP*P2
    dydt[-2]       = beta*y[N]/tau + beta_I*y[2*N]/tau_I - \
                         (delta + eta*sum(y[:(2+comp)*N+1]))*P + DMP*P
    dydt[-1]       = -gn*B/Y + DMn*n
    nex = y+dt*dydt
    #Advection
    v              = AdSpeed(chi,am,ap,n,n_matrix,dr)
    vint           = RegularGridInterpolator(cellgrid,v)
    vbord          = vint((rarr[:l-1]+dr/2).reshape(l-1,1))
    dA             = np.pi*vbord*dt*(2*np.arange(1,l)*dr - vbord*dt)
    for i in range(N*(2+comp)+1):
        nex[i] = Advect(nex[i],y[i],vbord,dA,areaarr)
    return nex

################################################################

