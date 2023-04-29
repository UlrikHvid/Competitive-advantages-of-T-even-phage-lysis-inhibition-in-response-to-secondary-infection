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
    def __init__(self,n0 = 10**9):
        self.N = 10 #Number of substates
        self.M = 10 #Number of substates in M3
        self.gnmax = np.log(2**(1/20)) #Growth rate [/min]
        self.n0 = n0 #Initial nutrient concentration [/ml]
        self.Kn= n0/5 #Nutrient concentration of half max growth [/ml]
        self.eta = 5*10**(-10) #Adsorption rate [ml/min]
        self.tau0 = 20 #Min lysis time [min]
        self.f_tau = 2 #Ratio of LIN lysis time to normal
        self.beta0 = 150 #Max burst size
        self.f_beta = 2 #Ratio of LIN burst size to normal 
        self.rl = 0.5 #Ratio of minimum lysis time to max
        self.rb = 0.1 #Ratio of minimum burst size to max
        self.Y = 1 #Nutrient yield constant (choice of nutrient unit)
        self.rr = 0 #Dilution rate in chemostat model
        self.comp = 0 #Competition boolean

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
        y0[0,:ispot]  = 10**6/(np.pi*(ispot*V.dr)**2)
        y0[-2,:ispot] = y0[0,0]
        y0[-3,:ispot] = y0[0,0] if V.comp else 0
        y0[-1]        = V.n0 #/micron**2
    else:
        y0[0]         = 1/40 #/micron**2
        y0[1,0]       = 1/Area(V.dr,1) #/micron**2
        if V.comp:
            y0[2*V.N+1,0]= 1/Area(V.dr,1) #/micron**2
        y0[-1]        = V.n0 #/micron**2
    return y0

class DVS(): #Default values for spatial models
    def __init__(self,rho = 10**9,dr = 20,Rmax = 3*10**3,da = 500,Dn = 5*10**4):
        #rho [/ml] is the richness of the swimming medium
        self.N      = 10 #Number of substates
        self.M      = 10 #Number of substates in M3
        self.da     = da #Agar thickness [micron]
        self.gnmax  = np.log(2**(1/20)) #Mac growth rate [/min]
        self.n0     = rho*10**(-12)*self.da #Initial nutrient concentration [/micron**2]
        self.Kn     = self.n0/5 #Nut. concentration of half max growth rate [/min]
        self.eta    = 5*10**2 #Adsorption rate [micron**3/min]
        self.tau0   = 20 #Minimum lysis time [min]
        self.f_tau  = 2 #Ratio of LIN lysis time to normal 
        self.beta0  = 150 #Max burst size
        self.f_beta = 2 #Ratio of LIN burst size to normal
        self.rl     = 0 #Ratio of min lysis time to max 
        self.rb     = 0.1 #Ratio of min burst size to max
        self.Y      = 1 #Yield constant (choice of unit)
        self.comp   = 0 #Competition boolean
        self.DP     = 240 #Phage diffusion constant [micron**2/min]
        self.DB     = 800 #Bacteria diff. const. [micron**2/min]
        self.Dn     = Dn #Nutrient diff. const. [micron**2/min]
        self.Rmax   = Rmax #Plate radius [micron]
        self.rspot  = 3*10**3 #Inoculation radius [micron]
        self.dr     = dr #Pixel size [micron]
        self.dt     = dr**2/Dn/3 #Time step (Ensures stability of diffusion algorithm) [min]
        self.l      = int(self.Rmax/self.dr) #Number of pixels
        self.delta  = 0 #Phage decay rate
        self.chi    = 2*10**4 #Chemotactic coefficient
        self.am     = self.n0/100 #High nutrient sensing constant
        self.ap     = self.n0*5 #Low nutrient sensing constant

