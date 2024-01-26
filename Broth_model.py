import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
import matplotlib.animation as animation
from tqdm import tqdm

###########################################################################

def Gamma(gnmax, n, Kn): 
    return gnmax*n/(Kn + n)

def Beta(beta0,rb,gn0,gn):
    return beta0*(rb + (1-rb)*gn/gn0)

def Tau(tau0,rl,gn0,gn):
    return tau0/(rl + (1-rl)*gn/gn0)

def MatrixInfected(N): #For quickly integrating all infected states
    subdiag = np.ones(N-1)
    diag = np.ones(N)*-1
    return sp.diags([subdiag,diag],[-1,0])

###########################################################################

#These functions are formatted for use in scipy's solve_ivp

def M0(t,y,N,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,rr): #No LIN
    #y is vector og length (N + 3) containing all variables
    #y = [B,L1,...LN,P,n]
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    beta      = Beta(beta0 ,rb,gn0,gn)
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    #Susceptible
    dydt[0]   = (gn - eta*P)*B
    #First infected state
    dydt[1]   = eta*P*B - y[1]/tau
    #Rest of infected and inhibited
    for i in range(N-1): 
        dydt[2+i]   = (y[1+i] - y[2+i])/tau  
    dydt[-2] = beta*y[N]/tau  - eta*P*sum(y[:N+1])
    dydt[-1] = -gn*B/Y + rr*n0 #Nutrition
    dydt -= rr*y
    return dydt

def M1(t,y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,comp = 0,rtrig = True):
    #y is vector og length (2(N + comp) + 3) containing all variables
    #y[ = [B,L1,...LN,LI1,...LIN,(L21,...L2N,P2,)P,n]]
    #comp is for competition, and rtrig determines whether r mutants trigger LIN
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    tau_I     = f_tau*tau
    beta      = Beta(beta0 ,rb,gn0,gn)
    beta_I    = f_beta*beta
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    #Susceptible
    dydt[0]   = (gn - eta*(P+P2))*B
    #First infected state
    if rtrig:
        dydt[1]   = eta*P*B - (eta*(P+P2) + 1/tau)*y[1]
    else:
        dydt[1]   = eta*P*B - (eta*P      + 1/tau)*y[1]
    #First inhibited state
    if rtrig:
        dydt[N+1] = eta*(P+P2)*sum(y[1:N+1]) - y[N+1]/tau_I 
    else:
        dydt[N+1] = eta*P     *sum(y[1:N+1]) - y[N+1]/tau_I
    #Rest of infected and inhibited
    for i in range(2,N+1): 
        if rtrig:
            dydt[i]   = (y[i-1]   - y[i])/tau      - eta*(P+P2)*y[i]
        else:
            dydt[i]   = (y[i-1]   - y[i])/tau      - eta*P*y[i]
        dydt[N+i] = (y[N+i-1] - y[N+i])/tau_I
    if comp:
        dydt[2*N+1] = eta*P2*B - y[2*N+1]/tau #First P2-infected state
        for i in range(2,N+1): #Rest of P2-infected states
            dydt[2*N+i]= (y[2*N+i-1] - y[2*N+i])/tau
        dydt[-3] = beta*y[3*N]/tau - eta*P2*sum(y[:3*N+1])
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2+comp)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0 #Nutrition
    dydt    -= rr*y
    return dydt

def M2(t,y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,comp):
    #y is vector og length (2N + 3) containing all variables
    #y = [B,L1,...LN,LI1,...LIN,P,n]
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    tau_I     = f_tau*tau
    beta      = Beta(beta0 ,rb,gn0,gn)
    beta_I    = f_beta*beta
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    #Susceptible
    dydt[0]   = (gn - eta*(P+P2))*B
    #First infected state
    dydt[1]   = eta*P*B - eta*P*y[1]             - y[1]/tau
    #First inhibited state
    dydt[N+1] = eta*P*(sum(y[1:N+1])+sum(y[N+2:2*N+1])) - y[N+1]/tau_I 
    #Rest of infected and inhibited
    for i in range(2,N+1): 
        dydt[i]   = (y[i-1]   - y[i])/tau      - eta*P*y[i]
        dydt[N+i] = (y[N+i-1] - y[N+i])/tau_I  - eta*P*y[N+i]
    if comp:
        dydt[2*N+1] = eta*P2*B - y[2*N+1]/tau #First P2-infected state
        for i in range(2,N+1): #Rest of P2-infected states
            dydt[2*N+i]= (y[2*N+i-1] - y[2*N+i])/tau
        dydt[-3] = beta*y[3*N]/tau - eta*P2*sum(y[:3*N+1])
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2+comp)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0
    dydt -= rr*y #Empty out chemostat
    return dydt

def M3(t,y,N,M,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,rr,comp): 
    #y is vector of length (N*(M+comp) + 3 + comp) containing all variables
    #y = [B,L11,...L1N,L21,...L2N,...,LM1,...LMN,(LC1,...LCN,P2,)P,n] 
    def I(i,j): #For translating from 2D L matrix to 1D y vector
        return (i-1)*N + j
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    beta      = Beta(beta0 ,rb,gn0,gn)
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    #Susceptible
    dydt[0]   = (gn - eta*(P+P2))*B
    #L_(1,1)
    dydt[1]   = eta*P*B - eta*(P+P2)*y[1] - y[1]/tau
    #L_(i,1), 1<i<M
    for i in range(2,M):
        ind = I(i,1)
        dydt[ind] = eta*(P+P2)*( np.sum(y[I(i-1,1):ind]) ) - y[ind]/tau
    #L_(M,1)
    dydt[I(M,1)] = eta*(P+P2)*( y[I(M-1,1)] + \
                            np.sum(y[I(M-1,2):I(M,1)] + y[I(M,2):I(M+1,1)]) ) - \
                            y[I(M,1)]/tau
    #L[i,j], all i, j>1
    for i in range(1,M+1):
        for j in range(2,N+1):
            ind = I(i,j)
            dydt[ind] = (y[I(i,j-1)]-y[ind])/tau - eta*(P+P2)*y[ind]
    if comp:
        dydt[N*M+1] = eta*P2*B - y[N*M+1]/tau #First P2-infected state
        for i in range(N-1): #Rest of P2-infected states
            dydt[N*M+2+i]= (y[N*M+1+i] - y[N*M+2+i])/tau
        dydt[-3] = beta*y[N*(M+1)]/tau - eta*P2*sum(y[:N*(M+1)+1])
    #Phage
    dydt[-2] = - eta*P*np.sum(y[:N*(M+comp)+1])
    for i in range(1,M+1):
        dydt[-2] += i*beta*y[I(i,N)]/tau #Kan skrives som list comprehension eller andet
    #Nutrients
    dydt[-1] = -gn*y[0]/Y + rr*n0
    dydt -= rr*y #Empty out chemostat
    return dydt

##################################################

#Equation-versions, for finding root and derivative with scipy's fsolve and approx_fprime

def M0eq(y,N,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,rr,root = 1): 
    #y is vector og length (2N + 3) containing all variables
    #y = [B,L1,...LN,P,n]
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    beta      = Beta(beta0 ,rb,gn0,gn)
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    #Susceptible
    dydt[0]   = (gn - eta*P - rr) #Removed factor B so that it does not easily choose B=0 as a steady state
    dydt[0] *= 1 if root else B
    #First infected state
    dydt[1]   = eta*P*B - y[1]/tau 
    #Rest of infected and inhibited
    for i in range(N-1): 
        dydt[2+i]   = (y[1+i]   - y[2+i])/tau
    dydt[-2] = beta*y[N]/tau - eta*P*sum(y[:N+1]) #Phage
    dydt[-1] = -gn*B/Y + rr*n0 #Nutrition
    dydt[1:] -= rr*y[1:] #Depletion
    return dydt

def M1eq(y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,comp = 0,root = 1):
    #y is vector og length (2(N + comp) + 3) containing all variables
    #y = [B,L1,...LN,LI1,...LIN,(L21,...L2N,P2,)P,n]
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    tau_I     = f_tau*tau
    beta      = Beta(beta0 ,rb,gn0,gn)
    beta_I    = f_beta*beta
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    #Susceptible
    dydt[0]   = (gn - eta*(P+P2)-rr) #Removed factor B so that it does not easily choose B=0 as a steady state
    dydt[0] *= 1 if root else B
    #First infected state
    dydt[1]   = eta*P*B - (eta*P + 1/tau)*y[1]
    #First inhibited state
    dydt[N+1] = eta*P*sum(y[1:N+1])         - y[N+1]/tau_I 
    #Rest of infected and inhibited
    for i in range(2,N+1): 
        dydt[i]   = (y[i-1]   - y[i])/tau      - eta*P*y[i]
        dydt[N+i] = (y[N+i-1] - y[N+i])/tau_I
    if comp:
        dydt[2*N+1] = eta*P2*B - y[2*N+1]/tau #First P2-infected state
        for i in range(2,N+1): #Rest of P2-infected states
            dydt[2*N+i]= (y[2*N+i-1] - y[2*N+i])/tau
        dydt[-3] = beta*y[3*N]/tau - eta*P2*sum(y[:3*N+1])
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2+comp)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0 #Nutrition
    dydt[1:] -= rr*y[1:] #Obs
    return dydt

def M2eq(y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,comp = 0,root = 1):
    #y is vector og length (2N + 3) containing all variables
    #y = [B,L1,...LN,LI1,...LIN,P,n]
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    tau_I     = f_tau*tau
    beta      = Beta(beta0 ,rb,gn0,gn)
    beta_I    = f_beta*beta
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    #Susceptible
    dydt[0]   = (gn - eta*(P+P2)-rr) #Removed factor B so that it does not easily choose B=0 as a steady state
    dydt[0] *= 1 if root else B
    #First infected state
    dydt[1]   = eta*P*B - eta*P*y[1]             - y[1]/tau
    #First inhibited state
    dydt[N+1] = eta*P*(sum(y[1:N+1])+sum(y[N+2:2*N+1])) - y[N+1]/tau_I 
    #Rest of infected and inhibited
    for i in range(N-1): 
        dydt[2+i]   = (y[1+i]   - y[2+i])/tau      - eta*P*y[2+i]
        dydt[N+2+i] = (y[N+1+i] - y[N+2+i])/tau_I  - eta*P*y[N+2+i]
    if comp:
        dydt[2*N+1] = eta*P2*B - y[2*N+1]/tau #First P2-infected state
        for i in range(N-1): #Rest of P2-infected states
            dydt[2*N+2+i]= (y[2*N+1+i] - y[2*N+2+i])/tau
        dydt[-3] = beta*y[3*N]/tau - eta*P2*sum(y[:3*N+1])
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2+comp)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0
    dydt -= rr*y #Empty out chemostat
    return dydt
    
def M3eq(y,N,M,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,rr,comp = 0,root = 1):
    #y is vector of length (N*(M+comp) + 3 + comp) containing all variables
    #y = [B,L11,...L1N,L21,...L2N,...,LM1,...LMN,(LC1,...LCN,P2,)P,n] 
    def I(i,j): #For translating from 2D L matrix to 1D y vector
        return (i-1)*N + j
    gn        = Gamma(gnmax,y[-1],Kn)
    gn0       = Gamma(gnmax,n0,Kn)
    tau       = Tau(tau0   ,rl,gn0,gn)/N
    beta      = Beta(beta0 ,rb,gn0,gn)
    dydt      = np.zeros_like(y)
    B,P,n = y[0],y[-2],y[-1]
    P2 = y[-3] if comp else 0
    #Susceptible
    dydt[0]   = (gn - eta*(P+P2) - rr) #Removed factor B so that it does not easily choose B=0 as a steady state
    dydt[0] *= 1 if root else B
    #L_(1,1)
    dydt[1]   = eta*P*B - eta*P*y[1] - y[1]/tau
    #L_(i,1), 1<i<M
    for i in range(2,M):
        ind = I(i,1)
        dydt[ind] = eta*P*( np.sum(y[I(i-1,1):ind]) ) - y[ind]/tau
    #L_(M,1)
    dydt[I(M,1)] = eta*P*( y[I(M-1,1)] + \
                            np.sum(y[I(M-1,2):I(M,1)] + y[I(M,2):I(M+1,1)]) ) - \
                            y[I(M,1)]/tau
    #L[i,j], all i, j>1
    for i in range(1,M+1):
        for j in range(2,N+1):
            ind = I(i,j)
            dydt[ind] = (y[I(i,j-1)]-y[ind])/tau - eta*P*y[ind]
    if comp:
        dydt[N*M+1] = eta*P2*B - y[N*M+1]/tau #First P2-infected state
        for i in range(N-1): #Rest of P2-infected states
            dydt[N*M+2+i]= (y[N*M+1+i] - y[N*M+2+i])/tau
        dydt[-3] = beta*y[N*(M+1)]/tau - eta*P2*sum(y[:N*(M+1)+1])
    #Phage
    dydt[-2] = - eta*P*np.sum(y[:N*(M+comp)+1])
    for i in range(1,M+1):
        dydt[-2] += i*beta*y[I(i,N)]/tau
    #Nutrients
    dydt[-1] = -gn*y[0]/Y + rr*n0
    dydt[1:] -= rr*y[1:] #Empty out chemostat
    return dydt

####################################################

#Feast/famine simulations

def M0ShellFF(y0,const,t,runs=5):
    #T is the duration between nutrient additions
    sol = solve_ivp(M0,[0,t],y0,args = const)
    tarr,yarr = np.array(sol.t),np.array(sol.y)
    for _ in range(runs-1):
        ynext = sol.y[:,-1]/1000 #Dilute
        ynext[0],ynext[-1] = y0[0],y0[-1] #Add bacteria and nutrients
        sol = solve_ivp(M0,[0,t],ynext,args = const)
        tarr = np.append(tarr,np.array(sol.t + np.ones_like(sol.t)*tarr[-1]))
        yarr = np.append(yarr,sol.y,axis = 1)
    return tarr,yarr

def M1M2ShellFF(y0,const,t,solver,runs = 5):
    sol = solve_ivp(solver,[0,t],y0,args = const)
    tarr,yarr = np.array(sol.t),np.array(sol.y)
    for _ in range(runs-1):
        ynext = sol.y[:,-1]/1000 #Dilute
        ynext[0],ynext[-1] = y0[0],y0[-1] #Add bacteria and nutrients
        sol = solve_ivp(solver,[0,t],ynext,args = const)
        tarr = np.append(tarr,np.array(sol.t + np.ones_like(sol.t)*tarr[-1]))
        yarr = np.append(yarr,sol.y,axis = 1)
    return tarr,yarr


