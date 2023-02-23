import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import scipy
from ThesisModel import *

###########################################################################


def Gamma(gnmax, n, Kn): 
    return gnmax*n/(Kn + n)

def Beta(beta0,rb,gn0,gn):
    return beta0*(rb + (1-rb)*gn/gn0)

def Tau(tau0,rl,gn0,gn):
    return tau0/(rl + (1-rl)*gn/gn0)


###########################################################################

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

def M1(t,y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,comp = 0):
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
    dydt[0]   = (gn - eta*(P+P2))*B
    #First infected state
    dydt[1]   = eta*P*B - (eta*P + 1/tau)*y[1]
    #First inhibited state
    dydt[N+1] = eta*P*sum(y[1:N+1])         - y[N+1]/tau_I 
    #Rest of infected and inhibited
    for i in range(N-1): 
        dydt[2+i]   = (y[1+i]   - y[2+i])/tau      - eta*P*y[2+i]
        dydt[N+2+i] = (y[N+1+i] - y[N+2+i])/tau_I
    if comp:
        dydt[2*N+1] = eta*P2*B - y[2*N+1]/tau #First P2-infected state
        for i in range(N-1): #Rest of P2-infected states
            dydt[2*N+2+i]= (y[2*N+1+i] - y[2*N+2+i])/tau
        dydt[-3] = beta*y[3*N]/tau - eta*P2*sum(y[:3*N+1])
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2+comp)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0 #Nutrition
    dydt    -= rr*y
    return dydt

def M1compare(t,y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,p):
    #p is the probability of going into the I,0 state upon infection
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
    #Susceptible
    dydt[0]   = (gn - eta*P)*B
    #First infected state
    dydt[1]   = (1-p)*eta*P*B - y[1]/tau 
    #First inhibited state
    dydt[N+1] = p*eta*P*B       - y[N+1]/tau_I 
    #Rest of infected and inhibited
    for i in range(N-1): 
        dydt[2+i]   = (y[1+i]   - y[2+i])  /tau    
        dydt[N+2+i] = (y[N+1+i] - y[N+2+i])/tau_I
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0
    dydt -= rr*y
    return dydt

def M2(t,y,N,gnmax,n0,Kn,eta,tau0,tauI0,beta0,betaI0,rl,rb,Y,rr,comp):
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
    dydt -= rr*y #Empty out chemostat
    return dydt

##################################################

#Equation-versions, for finding root and derivative

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
    dydt[0]   = (gn - eta*P - rr) #Obs
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
    dydt[0]   = (gn - eta*(P+P2)-rr)
    dydt[0] *= 1 if root else B
    #First infected state
    dydt[1]   = eta*P*B - (eta*P + 1/tau)*y[1]
    #First inhibited state
    dydt[N+1] = eta*P*sum(y[1:N+1])         - y[N+1]/tau_I 
    #Rest of infected and inhibited
    for i in range(N-1): 
        dydt[2+i]   = (y[1+i]   - y[2+i])/tau      - eta*P*y[2+i]
        dydt[N+2+i] = (y[N+1+i] - y[N+2+i])/tau_I
    if comp:
        dydt[2*N+1] = eta*P2*B - y[2*N+1]/tau #First P2-infected state
        for i in range(N-1): #Rest of P2-infected states
            dydt[2*N+2+i]= (y[2*N+1+i] - y[2*N+2+i])/tau
        dydt[-3] = beta*y[3*N]/tau - eta*P2*sum(y[:3*N+1])
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2+comp)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0 #Nutrition
    dydt[1:] -= rr*y[1:] #Obs
    return dydt

def M1compareeq(y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,p,root = 1):
    #p is the probability of going into the I,0 state upon infection
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
    #Susceptible
    dydt[0]   = (gn - eta*P - rr)
    dydt[0] *= 1 if root else B
    #First infected state
    dydt[1]   = (1-p)*eta*P*B - y[1]/tau 
    #First inhibited state
    dydt[N+1] = p*eta*P*B       - y[N+1]/tau_I 
    #Rest of infected and inhibited
    for i in range(N-1): 
        dydt[2+i]   = (y[1+i]   - y[2+i])  /tau    
        dydt[N+2+i] = (y[N+1+i] - y[N+2+i])/tau_I
    dydt[-2] = beta*y[N]/tau + beta_I*y[2*N]/tau_I - eta*P*sum(y[:(2)*N+1])
    dydt[-1] = -gn*B/Y + rr*n0
    dydt[1:] -= rr*y[1:]
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
    dydt[0]   = (gn - eta*(P+P2)-rr)
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

def M2scalar(y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,comp,root = 1):
    dydt = M2eq(y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,rr,comp,root)
    return np.sum(dydt**2)
    
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
    dydt[0]   = (gn - eta*(P+P2) - rr)
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

def M3scalar(y,N,M,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,rr,comp):
    dydt = M3eq(y,N,M,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,rr,comp)
    return np.sum(dydt**2)


####################################################



def M0SimulateFF(y0,const,t,runs=5):
    #T is the duration between nutrient additions
    N = const[0]
    sol = solve_ivp(M0,[0,t],y0,args = const)
    tarr,yarr = np.array(sol.t),np.array(sol.y)
    for _ in range(runs-1):
        ynext = sol.y[:,-1]/1000 #Dilute
        ynext[0],ynext[-1] = y0[0],y0[-1] #Add bacteria and nutrients
        sol = solve_ivp(M0,[0,t],ynext,args = const)
        tarr = np.append(tarr,np.array(sol.t + np.ones_like(sol.t)*tarr[-1]))
        yarr = np.append(yarr,sol.y,axis = 1)
    return tarr,yarr

def M1M2SimulateFF(y0,const,t,solver,runs = 5):
    N = const[0]
    comp = const[-1]
    sol = solve_ivp(solver,[0,t],y0,args = const)
    tarr,yarr = np.array(sol.t),np.array(sol.y)
    for _ in range(runs-1):
        ynext = sol.y[:,-1]/1000 #Dilute
        ynext[0],ynext[-1] = y0[0],y0[-1] #Add bacteria and nutrients
        sol = solve_ivp(solver,[0,t],ynext,args = const)
        tarr = np.append(tarr,np.array(sol.t + np.ones_like(sol.t)*tarr[-1]))
        yarr = np.append(yarr,sol.y,axis = 1)
    return tarr,yarr



############################################################################


def IV(model,N=10,M=10,comp = 0):#Initial values
    if model == "M0":
        length = N+3
    elif model == "M1" or model == "M2":
        length = 2*N+3
    elif model == "M3":
        length = M*N+3
    length += 11 if comp else 0
    y0 = np.zeros(length)
    y0[0],y0[-2],y0[-1] = 10**6,10**8,10**9
    y0[-3] = y0[-2] if comp else 0
    return y0

class DefaultVal():
    def __init__(self):
        self.N = 10
        self.M = 10
        self.gnmax = np.log(2**(1/23))
        self.n0 = 10**9
        self.Kn= self.n0/5
        self.eta = 10**(-10)
        self.tau0 = 20
        self.f_tau = 2
        self.beta0 = 150
        self.f_beta = 2
        self.rl = 0.5
        self.rb = 0.1
        self.Y = 1
        self.rr = 0.01
        self.comp = 0

def Const(ob,M):
    if M == "M0":
        return [ob.N,ob.gnmax,ob.n0,ob.Kn,ob.eta,ob.tau0,ob.beta0,ob.rl,ob.rb,ob.Y,ob.rr]
    elif M == "M1" or M == "M2":
        return [ob.N,ob.gnmax,ob.n0,ob.Kn,ob.eta,ob.tau0,ob.f_tau,ob.beta0,ob.f_beta,ob.rl,ob.rb,ob.Y,ob.rr,ob.comp]
    elif M == "M3":
        return [ob.N,ob.M,ob.gnmax,ob.n0,ob.Kn,ob.eta,ob.tau0,ob.beta0,ob.rl,ob.rb,ob.Y,ob.rr,ob.comp]

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

############################################################################



def M0Plotter(tarr,yarr,N,scale = "log",lb = 1000,figtitle = 0):
    for i,var in enumerate(yarr):
        ls = "-"
        alpha = 1
        label = None
        if i == 0: #Susceptible
            label = "B"
            color = "orange"
        elif i == N + 1: #Phages
            label = "P"
            color = "black"
        elif i == N + 2: #Nutrition
            label = "n"
            color = "gray"
        if i > 0 and i <= N: #Infected
            color = "red"
            if i == 1:
                label = "L1"
            elif i == N:
                label = f"L{N}"
                ls = "--"
            else:
                alpha = 0.1
        plt.plot(tarr,var, label = label,ls = ls,alpha = alpha,color = color)
    if scale == "log":
        plt.ylim(lb,np.max(yarr)*1.5)
    plt.yscale(scale)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return

def M1M2Plotter(tarr,yarr,N,scale = "log",lb = 1000,figtitle = 0):
    for i,var in enumerate(yarr):
        ls = "-"
        alpha = 1
        label = None
        if i == 0: #Susceptible
            label = "B"
            color = "orange"
        elif i == 2*N + 1: #Phages
            label = "P"
            color = "black"
        elif i == 2*N + 2: #Nutrition
            label = "n"
            color = "gray"
        if i > 0 and i <=N: #Noninhibited infected
            color = "red"
            if i == 1:
                label = "L1"
            elif i == N:
                label = f"L{N}"
                ls = "--"
            else:
                alpha = 0.1
        elif i > N and i < 2*N+1: #Inhibited infected
            color = "blue"
            if i == N+1:
                label = "LI1"
            elif i == 2*N:
                label = f"LI{N}"
                ls = "--"
            else:
                alpha = 0.1
        plt.plot(tarr,var,label = label,ls = ls,alpha = alpha,color = color)
    if scale == "log":
        plt.ylim(lb,np.max(yarr)*1.5)
    plt.yscale(scale)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return

def CompPlotter(tarr,yarr,N,scale = "log",lb = 1000,figtitle = 0):
    for i,var in enumerate(yarr):
        ls = "-"
        alpha = 1
        label = None
        if i == 0: #Susceptible
            label = "B"
            color = "orange"
        elif i == 3*N + 2: #Phages
            label = "P"
            color = "black"
        elif i == 3*N + 1: #Phages of second type
            label = "P2"
            ls = "--"
            color = "black"
        elif i == 3*N + 3: #Nutrition
            label = "n"
            color = "gray"
        if i > 0 and i <=N: #Noninhibited infected
            #ls = "dashed"
            color = "red"
            if i == 1:
                label = "L1"
            elif i == N:
                label = f"L{N}"
                ls = "--"
            else:
                alpha = 0.1
        elif i > N and i < 2*N+1: #Inhibited infected
            color = "blue"
            if i == N+1:
                label = "LI1"
            elif i == 2*N:
                label = f"LI{N}"
                ls = "--"
            else:
                alpha = 0.1
        elif i > 2*N and i < 3*N+1: #Infected of second type
            color = "green"
            if i == 2*N+1:
                label = "L2_1"
            elif i == 3*N:
                label = f"L2_{N}"
                ls = "--"
            else:
                alpha = 0.1
        plt.plot(tarr,var,label = label,ls = ls,alpha = alpha,color = color)
    plt.yscale(scale)
    if scale == "log":
        plt.ylim(lb,np.max(yarr)*1.5)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return

def M3Plotter(sol,N,M,scale = "log",lb = 1000, figtitle = 0):
    count1,count2 = 1,1
    cmap = mpl.colormaps['winter']
    for i,var in enumerate(sol.y):
        ls = "-"
        label = None
        if i == 0: #Susceptible
            label = "B"
            color = "orange"
        elif i == N*M + 1: #Phages
            label = "P"
            color = "black"
        elif i == N*M + 2: #Nutrition
            label = "n"
            color = "gray"
        elif (i-1)%N == 0: #Just after infection
            label = f"L$_{{{count1},1}}$"
            color = cmap(count1/M)
            count1 += 1
        elif i%N == 0: #Just before lysis
            ls = "--"
            label = f"L$_{{{count2},{N}}}$"
            color = cmap(count2/M)
            count2 += 1
        else:
            continue
        plt.plot(sol.t,var,label = label,ls = ls,color = color)
    if scale == "log":
        plt.ylim(lb,np.max(sol.y)*1.5)
    plt.yscale(scale)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return

def M3Plotter_Simple(sol,N,M,scale = "log", lb = 1000,figtitle = 0):
    plt.plot(sol.t,sol.y[0],label = "B",color = "orange")
    plt.plot(sol.t,sol.y[-2],label = "P",color = "black")
    plt.plot(sol.t,sol.y[-1],label = "n",color = "gray")
    L_tot = sum(sol.y[1:N*M+1])
    plt.plot(sol.t,L_tot,label = "L$_{{tot}}$",color = "red")
    plt.yscale(scale)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if scale == "log":
            plt.ylim(lb,max(np.max(sol.y),np.max(L_tot))*1.5)
    if figtitle:
        plt.savefig(figtitle)
    return

def M3Plotter_Interm(sol,N,M,scale = "log",lb = 1000, figtitle = 0):
    count = 1
    cmap = mpl.colormaps['winter']
    for i,var in enumerate(sol.y):
        ls = "-"
        label = None
        if i == 0: #Susceptible
            label = "B"
            color = "orange"
        elif i == N*M + 1: #Phages
            label = "P"
            color = "black"
        elif i == N*M + 2: #Nutrition
            label = "n"
            color = "gray"
        elif i%N == 0: #Just before lysis
            label = f"L$_{{{count},{N}}}$"
            color = cmap(count/M)
            count += 1
        else:
            continue
        plt.plot(sol.t,var,label = label,ls = ls,color = color)
    if scale == "log":
        plt.ylim(lb,np.max(sol.y)*1.5)
    plt.yscale(scale)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return
    
def M3CPlotter(sol,N,M,scale = "log",lb = 1000,figtitle = 0):
    count1,count2 = 1,1
    cmap = mpl.colormaps['winter']
    for i,var in enumerate(sol.y):
        ls = "-"
        label = None
        if i == 0: 
            label = "B"
            color = "orange"
        elif i == N*(M+1) + 1: 
            label = "P"
            color = "black"
        elif i == N*(M+1) + 2:
            label = "P2"
            color = "black"
            ls = "--"
        elif i == N*M + 3: #Nutrition
            label = "n"
            color = "gray"
        elif (i-1)%N == 0 and i/M <= N: #P1-infected just after infection
            label = f"L$_{{{count1},1}}$"
            color = cmap(count1/M)
            count1 += 1
        elif i%N == 0 and i/M <= N: #Just before lysis
            ls = "--"
            label = f"L$_{{{count2},{N}}}$"
            color = cmap(count2/M)
            count2 += 1
        elif i == N*M+1: #First P2-infected
            ls = "-"
            label = f"LL$_{{1}}$"
            color = "blue"
        elif i == N*(M+1):
            ls = "--"
            label = f"LL$_{{N}}$"
            color = "blue"
        else:
            continue
        plt.plot(sol.t,var,label = label,ls = ls,color = color)
    if scale == "log":
        plt.ylim(lb,np.max(sol.y)*1.5)
    plt.yscale(scale)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return

def M3CPlotter_Simple(sol,N,M,scale = "log",lb = 1000,figtitle = 0):
    plt.plot(sol.t,sol.y[0],label = "B",color = "orange")
    plt.plot(sol.t,sol.y[-2],label = "P",color = "black")
    plt.plot(sol.t,sol.y[-3],label = f"P$_2$",ls = "--",color = "black")
    plt.plot(sol.t,sol.y[-1],label = "n",color = "gray")
    L_tot = sum(sol.y[1:N*M+1])
    plt.plot(sol.t,L_tot,label = f"L$_{{tot}}$",color = "red")
    L2 = sum(sol.y[N*M+1:N*(M+1)+1])
    plt.plot(sol.t,L2,label = f"L$_2$",ls = "--",color = "red")
    plt.yscale(scale)
    plt.ylabel("Count")
    plt.xlabel("Time [min]")
    plt.legend()
    if scale == "log":
            plt.ylim(lb,max(np.max(sol.y),np.max(L_tot))*1.5)
    if figtitle:
        plt.savefig(figtitle)
    return

####################################################################################

#Diffusion term - finite difference

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

def BMatrix(l): 
    phi_l = Phi_l(np.arange(2,l))
    phi_u = Phi_u(np.arange(1,l-1))
    diag = [-2]*(l-1)
    M1 = sp.diags([phi_l,diag,phi_u],[-1,0,1])
    M2 = np.vstack(np.zeros(l-1)) #Måske et scipy sparse array i stedet?
    M2[-1] = Phi_u(l-1)
    M3 = np.zeros(l-1)
    M4 = [0]
    return sp.bmat([[M1,M2],[M3,M4]])

def MP0(y,N,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,DP,Dn,da,delta,dt,dr,l,extinct = 1/500,threshold = 0):
    #prev: First dimension represents variable (B,P etc.). Second represents space coordinate
    gn           = Gamma(gnmax,y[-1],Kn)
    gn0          = Gamma(gnmax,n0,Kn)
    tau          = Tau(tau0   ,rl,gn0,gn)/N
    beta         = Beta(beta0 ,rb,gn0,gn)
    B,P,n        = y[0],y[-2],y[-1]
    eta          = eta/da
    nex          = np.copy(y)
    nex[0]      += dt*(gn - eta*P*(P>threshold))*B
    nex[1]      += dt*(eta*P*B - y[1]/tau)
    for i in range(N-1): 
        nex[2+i]+= dt*(y[1+i] - y[2+i])/tau
    nex[-2]     += dt*(beta*y[N]/tau  - (delta + eta*sum(y[:N+1]))*P + DP/dr**2*Matrix(l)*P)
    nex[-1]     += dt*(-gn*B/Y + Dn/dr**2*Matrix(l)*n)
    Btot = sum(nex[:N+1])
    for i in range(len(Btot)):
        if Btot[i] < extinct:
            nex[:N+1,i] = 0
    return nex

def MP0IVP(t,y,N,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,DP,Dn,da,delta,dr,l):
    #prev: First dimension represents variable (B,P etc.). Second represents space coordinate
    y            = y.reshape((N+3,l))
    dydt         = np.zeros_like(y)
    gn           = Gamma(gnmax,y[-1],Kn)
    gn0          = Gamma(gnmax,n0,Kn)
    tau          = Tau(tau0   ,rl,gn0,gn)/N
    beta         = Beta(beta0 ,rb,gn0,gn)
    B,P,n        = y[0],y[-2],y[-1]
    eta          = eta/da
    dydt[0]      = (gn - eta*P)*B
    dydt[1]      = (eta*P*B - y[1]/tau)
    for i in range(N-1): 
        dydt[2+i]= (y[1+i] - y[2+i])/tau
    dydt[-2]     = (beta*y[N]/tau  - (delta + eta*sum(y[:N+1]))*P + DP/dr**2*Matrix(l)*P)
    dydt[-1]     = (-gn*B/Y + Dn/dr**2*Matrix(l)*n)   
    return dydt.reshape((N+3)*l)

def MP1(y,N,gnmax,n0,Kn,eta,tau0,f_tau,beta0,f_beta,rl,rb,Y,DP,Dn,da,delta,dt,dr,l,extinct = 1/500):
    #prev: First dimension represents variable (B,P etc.). Second represents space coordinate
    gn             = Gamma(gnmax,y[-1],Kn)
    gn0            = Gamma(gnmax,n0,Kn)
    tau            = Tau(tau0   ,rl,gn0,gn)/N
    tau_I          = f_tau*tau
    beta           = Beta(beta0 ,rb,gn0,gn)
    beta_I         = f_beta*beta
    B,P,n          = y[0],y[-2],y[-1]
    eta            = eta/da
    nex            = np.copy(y)
    nex[0]        += dt*(gn - eta*P)*B
    nex[1]        += dt*(eta*P*B - (eta*P + 1/tau)*y[1])
    nex[N+1]      += dt*(eta*P*sum(y[1:N+1])         - y[N+1]/tau_I )
    for i in range(N-1): 
        nex[2+i]  += dt*((y[1+i]  - y[2+i])/tau - eta*P*y[2+i])
        nex[N+2+i]+= dt*(y[N+1+i] - y[N+2+i])/tau_I
    nex[-2]       += dt*(beta*y[N]/tau + beta_I*y[2*N]/tau_I - \
                         (delta + eta*sum(y[:2*N+1]))*P + DP/dr**2*Matrix(l)*P)
    nex[-1]       += dt*(-gn*B/Y + Dn/dr**2*Matrix(l)*n)
    Btot = sum(nex[:2*N+1])
    for i in range(l):
        if Btot[i] < extinct:
            nex[:2*N+1,i] = 0
    return nex

def MP3(y,N,M,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,DP,Dn,da,delta,dt,dr,l,extinct = 1/500):
    #prev: First dimension represents variable (B,L[i,j],P,n). Second represents space coordinate
    def I(i,j): #For translating from 2D L matrix to 1D y vector
        return (i-1)*N + j
    gn             = Gamma(gnmax,y[-1],Kn)
    gn0            = Gamma(gnmax,n0,Kn)
    tau            = Tau(tau0   ,rl,gn0,gn)/N
    beta           = Beta(beta0 ,rb,gn0,gn)
    B,P,n          = y[0],y[-2],y[-1]
    eta            = eta/da
    nex            = np.copy(y)
    nex[0]        += dt*(gn - eta*P)*B
    #L_(1,1)
    nex[1]        += dt*(eta*P*B - (eta*P + 1/tau)*y[1])
    #L_(i,1), 1<i<M
    for i in range(2,M):
        ind = I(i,1)
        nex[ind] += dt*(eta*P*( np.sum(y[I(i-1,1):ind]) ) - y[ind]/tau)
    #L_(M,1)
    nex[I(M,1)]  += dt*(eta*P*( y[I(M-1,1)] + \
                            np.sum(y[I(M-1,2):I(M,1)] + y[I(M,2):I(M+1,1)]) ) - \
                            y[I(M,1)]/tau)
    #L[i,j], all i, j>1
    for i in range(1,M+1):
        for j in range(2,N+1):
            ind = I(i,j)
            nex[ind] += dt*((y[ind-1]-y[ind])/tau - eta*P*y[ind])
    nex[-2]    += dt*(-(delta + eta*np.sum(y[:N*M+1]))*P + DP/dr**2*Matrix(l)*P)
    for i in range(1,M+1):
        ind = I(i,N)
        nex[-2]  += dt*(i*beta*y[ind]/tau)
    nex[-1]      += dt*(-gn*B/Y + Dn/dr**2*Matrix(l)*n)
    Btot = sum(nex[:N*M+1])
    for i in range(len(Btot)):
        if Btot[i] < extinct:
            nex[:N*M+1,i] = 0
    return nex

def MP3IMP(y,N,M,gnmax,n0,Kn,eta,tau0,beta0,rl,rb,Y,DP,Dn,da,dt,dr,l,extinct = 1/500):
    #prev: First dimension represents variable (B,L[i,j],P,n). Second represents space coordinate
    def I(i,j): #For translating from 2D L matrix to 1D y vector
        return (i-1)*N + j
    gn             = Gamma(gnmax,y[-1],Kn)
    gn0            = Gamma(gnmax,n0,Kn)
    tau            = Tau(tau0   ,rl,gn0,gn)/N
    beta           = Beta(beta0 ,rb,gn0,gn)
    B,P,n          = y[0],y[-2],y[-1]
    eta            = eta/da
    nex            = np.copy(y)
    nex[0]        += dt*(gn - eta*P)*B
    #L_(1,1)
    nex[1]        += dt*(eta*P*B - (eta*P + 1/tau)*y[1])
    #L_(i,1), 1<i<M
    for i in range(2,M):
        ind = I(i,1)
        nex[ind] += dt*(eta*P*( np.sum(y[I(i-1,1):ind]) ) - y[ind]/tau)
    #L_(M,1)
    nex[I(M,1)]  += dt*(eta*P*( y[I(M-1,1)] + \
                            np.sum(y[I(M-1,2):I(M,1)] + y[I(M,2):I(M+1,1)]) ) - \
                            y[I(M,1)]/tau)
    #L[i,j], all i, j>1
    for i in range(1,M+1):
        for j in range(2,N+1):
            ind = I(i,j)
            nex[ind] += dt*((y[ind-1]-y[ind])/tau - eta*P*y[ind])
    C = sum(y[N:M*N+1:N]*np.arange(1,M+1))*beta/tau
    D = sp.diags(eta*sum(y[:N*M+1],0) + DP/dr**2*Matrix(l))
    inverted = scipy.linalg.inv(sp.identity(l)-dt*D)
    nex[-2] = inverted*(y[-2] + dt*C)
    #nex[-2]    += dt*(-(delta + eta*np.sum(y[:N*M+1]))*P + DP/dr**2*Matrix(l)*P)
    #for i in range(1,M+1):
    #    ind = I(i,N)
    #    nex[-2]  += dt*(i*beta*y[ind]/tau) #Kan skrives pænere
    nex[-1]      += dt*(-gn*B/Y + Dn/dr**2*Matrix(l)*n)
    Btot = sum(nex[:N*M+1])
    for i in range(len(Btot)):
        if Btot[i] < extinct:
            nex[:N*M+1,i] = 0
    return nex

def IVP(model,N,M,l, dr, comp = 0):#Initial values
    if model == "MP0":
        length = N+3 
    elif model == "MP1" or model == "MP2":
        length = 2*N+3
    elif model == "MP3":
        length = M*N+3
    #length += 11 if comp else 0
    y0 = np.zeros((length,l))
    y0[0,:]   = 1/320 #micron**(-2)
    y0[1,0]   = 1/Area(dr,1) #micron**(-2)
    y0[-1,:]  = 30 #micron**(-2)
    #y0[-3] = y0[-2] if comp else 0
    return y0

class DefaultValP():
    def __init__(self):
        self.N      = 10
        self.M      = 10
        self.gnmax  = np.log(2**(1/20)) #min**-1
        self.n0     = 30 #micron**-1
        self.Kn     = self.n0/5
        self.eta    = 10**2 #micron**3/min
        self.tau0   = 20 #min
        self.f_tau  = 2 
        self.beta0  = 150
        self.f_beta = 2
        self.rl     = 0.5 
        self.rb     = 0.1
        self.Y      = 1
        self.comp   = 0
        self.da     = 500 #Microns
        self.DP     = 600 #Micron**2/min
        self.Dn     = 6000 #Micron**2/min
        self.Rmax   = 10**4 #Microns
        self.rspot  = 2*10**3 #Microns
        self.dr     = 20 #Microns
        self.dt     = 0.01 #min**-1
        self.l      = int(self.Rmax/self.dr)
        self.delta  = 0.003/60

class DefaultValRep():  #For replicating Namikos results
    def __init__(self):
        self.N      = 10
        self.gnmax  = np.log(2**(1/20)) #min**-1
        self.n0     = 30 #micron**-2
        self.Kn     = self.n0/5
        self.eta    = 8*10**4/60 #micron**3/min
        self.tau0   = 1.5/self.gnmax #Minimum value
        self.beta0  = 100
        self.rl     = 0.5 
        self.rb     = 0.1
        self.Y      = 1
        self.comp   = 0
        self.da     = 500 #Microns
        self.DP     = 600 #Micron**2/min
        self.Dn     = 6000 #Micron**2/min
        self.Rmax   = 10**4 #Microns
        self.rspot  = 2*10**3 #Microns
        self.dr     = 20 #Microns
        self.dt     = self.dr**2/self.Dn/10
        self.l      = int(self.Rmax/self.dr)
        self.delta  = 0.003/60

def ConstP(ob,M):
    if M == "MP0IVP":
        return [ob.N,ob.gnmax,ob.n0,ob.Kn,ob.eta,ob.tau0,ob.beta0,ob.rl,ob.rb,ob.Y,ob.DP,ob.Dn,ob.da,ob.delta,ob.dr,ob.l]
    #elif M == "M1" or M == "M2":
    #    return [ob.N,ob.gnmax,ob.n0,ob.Kn,ob.eta,ob.tau0,ob.f_tau,ob.beta0,ob.f_beta,ob.rl,ob.rb,ob.Y,ob.rr,ob.comp]
    #elif M == "M3":
    #    return [ob.N,ob.M,ob.gnmax,ob.n0,ob.Kn,ob.eta,ob.tau0,ob.beta0,ob.rl,ob.rb,ob.Y,ob.rr,ob.comp]

