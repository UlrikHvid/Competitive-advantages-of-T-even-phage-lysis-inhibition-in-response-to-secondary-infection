import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
from time import time
from tqdm import tqdm
import sys
sys.path.insert(-1,"/nbi/nbicmplx/cell/dhm160")
from Broth_model import Gamma, Beta, Tau

####################################################################################

#Subfunctions

def Area(dr,i):
    return np.pi*dr**2*(2*i-1)

def Phi_l(i):
    return (i-1)/(i-1/2)
def Phi_u(i):
    return (i)/(i-1/2)

def Matrix(l):
    phi_l = Phi_l(np.arange(2,l))
    phi_l = np.append(phi_l,Phi_l(l) + Phi_u(l))
    phi_u = Phi_u(np.arange(1,l))
    diag = np.array([-2]*(l))
    return sp.diags([phi_l,diag,phi_u],[-1,0,1])

################################################################################################################

#Plaque models

def MPShell(model,y0,V,t,frames=False):
    if not frames:
        frames = int(t/V.tau0) #Default value for frames
    its = int(t/V.dt)
    sim = np.zeros((frames,len(y0),V.l)) #Holds the saved data
    frameind = np.linspace(0,its-1,frames,dtype = int)
    ynext = np.copy(y0)
    if model == "MP0":
        for i in tqdm(range(its)):
            ynext = MP0(ynext,V)
            if i in frameind:
                sim[np.where(frameind == i)] = ynext
    elif model == "MP1":
        for i in tqdm(range(its)):
            ynext = MP1(ynext,V)
            if i in frameind:
                sim[np.where(frameind == i)] = ynext
    return sim

def MP0(y,V):
    #prev: First dimension represents variable (B,P etc.). Second represents space coordinate
    N,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,DP,Dn,da,delta,dt,dr,l = V.N,V.gnmax,V.n0,V.Kn,V.eta,V.tau0,V.beta0,V.rl,V.rb,V.Y,V.DP,V.Dn,V.da,V.delta,V.dt,V.dr,V.l
    gn           = Gamma(gnmax,y[-1],Kn)
    gn0          = Gamma(gnmax,n0,Kn)
    tau          = Tau(tau0   ,rl,gn0,gn)/N
    beta         = Beta(beta0 ,rb,gn0,gn)
    B,P,n        = y[0],y[-2],y[-1]
    eta          = eta/da
    nex          = np.copy(y)
    nex[0]      += dt*(gn - eta*P)*B
    nex[1]      += dt*(eta*P*B - y[1]/tau)
    for i in range(N-1): 
        nex[2+i]+= dt*(y[1+i] - y[2+i])/tau
    nex[-2]     += dt*(beta*y[N]/tau  - (delta + eta*sum(y[:N+1]))*P + DP/dr**2*Matrix(l)*P)
    nex[-1]     += dt*(-gn*B/Y + Dn/dr**2*Matrix(l)*n)
    return nex

def MP1(y,V):
    #prev: First dimension represents variable (B,P etc.). Second represents space coordinate
    N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,DP,Dn,da,delta,dt,dr,l,comp =\
          V.N,V.gnmax,V.n0,V.Kn,V.eta,V.tau0,V.f_tau,V.beta0,V.f_beta,V.rl,V.rb,V.Y,V.DP,V.Dn,V.da,V.delta,V.dt,V.dr,V.l,V.comp
    gn             = Gamma(gnmax,y[-1],Kn)
    gn0            = Gamma(gnmax,n0,Kn)
    tau            = Tau(tau0   ,rl,gn0,gn)/N
    tau_I          = f_tau*tau
    beta           = Beta(beta0 ,rb,gn0,gn)
    beta_I         = f_beta*beta
    B,P,n          = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    eta            = eta/da
    nex            = np.copy(y)
    nex[0]        += dt*(gn - eta*(P+P2))*B
    nex[1]        += dt*(eta*P*B - (eta*(P+P2) + 1/tau)*y[1]) #First infected state
    nex[N+1]      += dt*(eta*(P+P2)*sum(y[1:N+1])         - y[N+1]/tau_I ) #First inhibited state
    for i in range(2,N+1):  #Rest of infected and inhibited states
        nex[i]  += dt*((y[i-1]  -  y[i]  )/tau - eta*(P+P2)*y[i])
        nex[N+i]+= dt*( y[N+i-1] - y[N+i])/tau_I
    if comp:
        nex[2*N+1] += dt*(eta*P2*B - y[2*N+1]/tau) #First P2-infected state
        for i in range(2,N+1): #Rest of P2-infected states
            nex[2*N+i]+= dt*(y[2*N+i-1] - y[2*N+i])/tau
        nex[-3] += dt*(beta*y[3*N]/tau - P2*(eta*sum(y[:3*N+1])+delta) + DP/dr**2*Matrix(l)*P2)
    nex[-2]       += dt*(beta*y[N]/tau + beta_I*y[2*N]/tau_I - \
                         (delta + eta*sum(y[:(2+comp)*N+1]))*P + DP/dr**2*Matrix(l)*P)
    nex[-1]       += dt*(-gn*B/Y + Dn/dr**2*Matrix(l)*n)
    return nex

