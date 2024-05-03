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

def Matrix(l,absorbing = False): #Diffusion matrix
    phi_l = Phi_l(np.arange(2,l))
    phi_l = np.append(phi_l,Phi_l(l) + Phi_u(l))
    phi_u = Phi_u(np.arange(1,l))
    diag = np.array([-2]*(l))
    if absorbing:
        phi_l[-1],diag[-1] = 0,0
    return sp.diags([phi_l,diag,phi_u],[-1,0,1])

################################################################################################################

#Plaque models

def MPShell(model,y0,V,t,save_interval=60,absorbing = False,progress = True): #Outer function for the plaque model
    #By defaulting saving a frame every 60 min
    its           = int(t/V.dt)
    savetimes  = [0]
    saveiters = [0]
    for i in range(its):
        if i*V.dt > (savetimes[-1]+save_interval):
            saveiters.append(i)
            savetimes.append(savetimes[-1]+save_interval)
    sim           = np.zeros((len(savetimes),len(y0),V.l)) #Holds the saved data
    ynext         = np.copy(y0)
    DMn,DMP       = V.Dn*Matrix(V.l)/V.dr**2, V.DP*Matrix(V.l,absorbing)/V.dr**2 #Matrices for computing spatial differentials
    gn0           = Gamma(V.gnmax,V.n0,V.Kn)
    frameind = 0
    if progress:
        proglist = tqdm(range(its))
    else:
        proglist = range(its)
    if model      == "MP0":
        for i in proglist:
            ynext = MP0(ynext,V,gn0,DMP,DMn)
            if i in saveiters:
                sim[frameind] = ynext
                frameind += 1
    elif model    == "MP1":
        for i in proglist:
            ynext = MP1(ynext,V,gn0,DMP,DMn)
            if i in saveiters:
                sim[frameind] = ynext
                frameind += 1
    return sim,savetimes

#def MPShell(model,y0,V,t,frames=False,absorbing = False,progress = True):
#    if not frames:
#        frames    = int(t/V.tau0) #Default value for frames
#    its           = int(t/V.dt)
#    sim           = np.zeros((frames,len(y0),V.l)) #Holds the saved data
#    frameind      = np.linspace(0,its-1,frames,dtype = int)
#    ynext         = np.copy(y0)
#    DMn,DMP       = V.Dn*Matrix(V.l)/V.dr**2, V.DP*Matrix(V.l,absorbing)/V.dr**2
#    gn0           = Gamma(V.gnmax,V.n0,V.Kn)
#    if progress:
#        proglist = tqdm(range(its))
#    else:
#        proglist = range(its)
#    if model      == "MP0":
#        for i in proglist:
#            ynext = MP0(ynext,V,gn0,DMP,DMn)
#            if i in frameind:
#                sim[np.where(frameind == i)] = ynext
#    elif model    == "MP1":
#        for i in proglist:
#            ynext = MP1(ynext,V,gn0,DMP,DMn)
#            if i in frameind:
#                sim[np.where(frameind == i)] = ynext
#    return sim

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
            Btot        = np.sum(frame[:(1+LIN+V.comp)*V.N+1],axis = 0)    #Sum all bacteria
            rhalfarr[i] = np.where(Btot >= Btot[-1]/2)[0][0]*V.dr/1000   #Picks out the index, multiplied by dr
    elif var == "B":
        for i,frame in enumerate(sim):
            B           = frame[0]
            rhalfarr[i] = np.where(B > B[-1]/2)[0][0]*V.dr/1000                      #Picks out the index, multiplied by dr
    return rhalfarr

def Pfront(sim,V,LIN = False):
    Pfrontarr = np.zeros(len(sim))
    gn0          = Gamma(V.gnmax,V.n0,V.Kn)
    DMP =  V.DP*Matrix(V.l)/V.dr**2
    for i,frame in enumerate(sim):
        P = frame[-2]
        gn           = Gamma(V.gnmax,frame[-1],V.Kn)
        beta         = Beta(V.beta0 ,V.rb,gn0,gn)
        tau          = Tau(V.tau0   ,V.rl,gn0,gn)/V.N
        burst = beta*frame[V.N]/tau
        if LIN:
            burst += beta*V.f_beta*frame[2*V.N]/tau/V.f_tau
        diff = DMP*P
        for j in np.arange(len(P)-2,0,-1):
            if burst[j] > diff[j] and burst[j+1] < diff[j+1]:
                Pfrontarr[i] = j*V.dr/1000
                break
    return Pfrontarr

def Pdet(sim,V,B0):
    Pdet = np.zeros(len(sim)) #Radius of good statistics
    for i,frame in enumerate(sim):
        P = frame[-2]
        for j in np.arange(len(P)-2,0,-1):
            if P[j] > B0 and P[j+1] < B0:
                Pdet[i] = j*V.dr/1000
                break
    return Pdet

def rsuper(sim,V):
    rsuper = np.zeros(len(sim))
    for i,frame in enumerate(sim):
        B = frame[0]
        P = frame[-2]
        for j in range(len(P)-2):
            if P[j] > B[j] and P[j+1] < B[j+1]:
                rsuper[i] = j*V.dr/1000
    return rsuper
