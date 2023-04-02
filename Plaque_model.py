import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
from tqdm import tqdm
import sys
sys.path.insert(-1,"/nbi/nbicmplx/cell/dhm160")
from Broth_model import Gamma, Beta, Tau

####################################################################################

#Subfunctions

def Area(dr,i): #Find area of a shell in the grid
    return np.pi*dr**2*(2*i-1)

def Phi_l(i):
    return (i-1)/(i-1/2)
def Phi_u(i):
    return (i)/(i-1/2)

def Matrix(l): #Diffusion matrix
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
    DMn,DMP = V.Dn*Matrix(V.l)/V.dr**2, V.DP*Matrix(V.l)/V.dr**2
    gn0          = Gamma(V.gnmax,V.n0,V.Kn)
    if model == "MP0":
        for i in tqdm(range(its)):
            ynext = MP0(ynext,V,gn0,DMP,DMn)
            if i in frameind:
                sim[np.where(frameind == i)] = ynext
    elif model == "MP1":
        for i in tqdm(range(its)):
            ynext = MP1(ynext,V,gn0,DMP,DMn)
            if i in frameind:
                sim[np.where(frameind == i)] = ynext
    return sim

def MP0(y,V,gn0,DMP,DMn):
    N,gnmax,Kn,eta,tau0,beta0,rl,rb,Y,da,delta,dt = V.N,V.gnmax,V.Kn,V.eta,V.tau0,V.beta0,V.rl,V.rb,V.Y,V.da,V.delta,V.dt
    gn           = Gamma(gnmax,y[-1],Kn)
    tau          = Tau(tau0   ,rl,gn0,gn)/N
    beta         = Beta(beta0 ,rb,gn0,gn)
    B,P,n        = y[0],y[-2],y[-1]
    eta          = eta/da
    dydt         = np.copy(y)
    dydt[0]      = (gn - eta*P)*B
    dydt[1]      = eta*P*B - y[1]/tau
    for i in range(N-1): 
        dydt[2+i]= (y[1+i] - y[2+i])/tau
    dydt[-2]     = beta*y[N]/tau  - (delta + eta*sum(y[:N+1]))*P + DMP*P
    dydt[-1]     = -gn*B/Y + DMn*n
    return y+dydt*dt

def MP1(y,V,gn0,DMP,DMn):
    N,gnmax,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,da,delta,dt,comp =\
          V.N,V.gnmax,V.Kn,V.eta,V.tau0,V.f_tau,V.beta0,V.f_beta,V.rl,V.rb,V.Y,V.da,V.delta,V.dt,V.comp
    gn             = Gamma(gnmax,y[-1],Kn)
    tau            = Tau(tau0   ,rl,gn0,gn)/N
    tau_I          = f_tau*tau
    beta           = Beta(beta0 ,rb,gn0,gn)
    beta_I         = f_beta*beta
    B,P,n          = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    eta            = eta/da
    dydt            = np.copy(y)
    dydt[0]         = (gn - eta*(P+P2))*B
    dydt[1]         = eta*P*B - (eta*(P+P2) + 1/tau)*y[1] #First infected state
    dydt[N+1]       = eta*(P+P2)*sum(y[1:N+1])         - y[N+1]/tau_I  #First inhibited state
    for i in range(2,N+1):  #Rest of infected and inhibited states
        dydt[i]   = (y[i-1]  -  y[i]  )/tau - eta*(P+P2)*y[i]
        dydt[N+i] = ( y[N+i-1] - y[N+i])/tau_I
    if comp:
        dydt[2*N+1]  = eta*P2*B - y[2*N+1]/tau #First P2-infected state
        for i in range(2,N+1): #Rest of P2-infected states
            dydt[2*N+i] = (y[2*N+i-1] - y[2*N+i])/tau
        dydt[-3]  = beta*y[3*N]/tau - P2*(eta*sum(y[:3*N+1])+delta) + DMP*P2
    dydt[-2]        = beta*y[N]/tau + beta_I*y[2*N]/tau_I - \
                         (delta + eta*sum(y[:(2+comp)*N+1]))*P + DMP*P
    dydt[-1]        = -gn*B/Y + DMn*n
    return y+dydt*dt

################################################################################################################################################

#Analytical functions

def rhalf(sim,LIN,V,var = "Btot"):
    rhalfarr = np.zeros(len(sim))
    if var == "Btot":
        for i,frame in enumerate(sim):
            Btot = np.sum(frame[:(1+LIN+V.comp)*V.N+1],axis = 0)
            halfmax = Btot[-1]/2
            for j in range(len(Btot)):
                if Btot[j] < halfmax and Btot[j+1] > halfmax:
                    rhalfarr[i] = j*V.dr/1000
    elif var == "B":
        for i,frame in enumerate(sim):
            B = frame[0]
            halfmax = B[-1]/2
            for j in range(len(B)):
                if B[j] < halfmax and B[j+1] > halfmax:
                    rhalfarr[i] = j*V.dr/1000
    return rhalfarr