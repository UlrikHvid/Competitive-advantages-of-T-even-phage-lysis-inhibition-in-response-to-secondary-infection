import numpy as np
from Plaque_model import Area
############################################################################

#Well-mixed models

def IV(model,N=10,M=10,comp = 0):#Initial values
    if model == "M0":
        length = N+3
    elif model == "M1" or model == "M2":
        length = 2*N+3
    elif model == "M3":
        length = M*N+3
    length += 11 if comp else 0
    y0 = np.zeros(length)
    y0[0],y0[-2],y0[-1] = 10**6,10**3,10**9
    y0[-3] = y0[-2] if comp else 0
    return y0

class DV():
    def __init__(self):
        self.N = 10
        self.M = 10
        self.gnmax = np.log(2**(1/20))
        self.n0 = 10**9
        self.Kn= self.n0/5
        self.eta = 5*10**(-10)
        self.tau0 = 20
        self.f_tau = 2
        self.beta0 = 150
        self.f_beta = 2
        self.rl = 0.5
        self.rb = 0.1
        self.Y = 1
        self.rr = 0
        self.comp = 0

def Const(ob,M): #Gather constants in an array of a length appropriate for the model
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

################################################################################################################

#Spatial models

def IVS(model,V): #Initial values for spatial models
    if   model in ["MP0","MS0"]:
        length = V.N+3 
    elif model in ["MP1","MS1"]:
        length = 2*V.N+3
    elif model == "MP3":
        length = V.M*V.N+3
    length += 11 if V.comp else 0
    y0 = np.zeros((length,V.l))
    swim   = True if model in ["MS0","MS1","MS2","MS3"] else False
    if swim:
        ispot = int(V.rspot/V.dr)
        y0[0,:ispot]  = 10**6/(np.pi*V.rspot**2)
        y0[-2,:ispot] = y0[0,0]
        y0[-3,:ispot] = y0[0,0] if V.comp else 0
        y0[-1]        = V.n0 #/micron**2
    else:
        y0[0]       = 1/40 #/micron**2
        y0[1,0]     = 1/Area(V.dr,1) #/micron**2
        if V.comp:
            y0[2*V.N+1,0] = 1/Area(V.V.dr,1) #/micron**2
        y0[-1]      = V.n0 #/micron**2
    return y0

class DVS(): #Default values for spatial models
    def __init__(self,rho = 10**9,dr = 20,Rmax = 10**4,da = 500,Dn = 50000):
        self.N      = 10
        self.M      = 10
        self.da     = da #Microns
        self.gnmax  = np.log(2**(1/20)) #/min
        self.n0     = rho*10**(-12)*self.da #/micron**2
        self.Kn     = self.n0/5
        self.eta    = 5*10**2 #micron**3/min
        self.tau0   = 20 #min
        self.f_tau  = 2 
        self.beta0  = 150
        self.f_beta = 2
        self.rl     = 0
        self.rb     = 0.1
        self.Y      = 1
        self.comp   = 0
        self.DP     = 600 #Micron**2/min
        self.DB     = 833 #Micron**2/min
        self.Dn     = Dn #Micron**2/min
        self.Rmax   = Rmax #Microns
        self.rspot  = 3*10**3 #Microns
        self.dr     = dr #Microns
        self.dt     = dr**2/Dn/3 #Ensures stability of diffusion algorithm
        self.l      = int(self.Rmax/self.dr)
        self.delta  = 0.003/60
        self.chi    = 10**5/6
        self.am     = self.n0/100
        self.ap     = self.n0*5
        self.extinct= 0

