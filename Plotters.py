import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np

############################################################################

#Plotter functions

def BrothPlotter(model,V,tarr,yarr,scale = "log",lb = 1000,figtitle = 0): #Not plotting infected substates
    N,comp = V.N,V.comp
    plt.plot(tarr,yarr[0], label = "B",color = "blue")
    plt.plot(tarr,sum(yarr[1:N+1]),label = "L",color = "darkviolet")
    plt.plot(tarr,yarr[-2],label = "P",color = "black")
    plt.plot(tarr,yarr[-1],label = "n",color = "gray")
    if model == "M1":
        LI = sum(yarr[N+1:2*N+1])
        plt.plot(tarr,LI,label = r"L$_I$",color = "crimson")
    if comp:
        Lr = sum(yarr[2*N+1:3*N+1])
        plt.plot(tarr,yarr[-3],label = r"P$_r$",ls = "--",color = "black")
        plt.plot(tarr,Lr,label = r"L$_r$",ls = "--",color = "darkviolet")
    plt.yscale(scale)
    if scale == "log":
        plt.ylim(lb,np.max(yarr)*1.5)
    plt.ylabel(r"Concentration [ml$^{-1}$]")
    plt.xlabel("Time [min]")
    plt.grid(axis="y", which = "major")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return

def BrothPlotter_AllStates(tarr,yarr,N,scale = "log",lb = 1000,figtitle = 0):
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
    plt.ylabel(r"Concentration [ml$^{-1}$]")
    plt.xlabel("Time [min]")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle)
    return


def M3Plotter(sol,N,M,scale = "log", lb = 1000,figtitle = 0):
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

def M3Plotter_Detailed(sol,N,M,scale = "log",lb = 1000, figtitle = 0):
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

################################################################################################################

#For outputting animations

def GifGenerator(sim,V,T,model,name,ylim = (1,200),r0 = 0,rf = False):
    frames = len(sim)
    arrlist = [frame for frame in sim]
    if not rf:
        rf = V.l #Plot whole system unless specified
    fig, ax = plt.subplots()
    if model == "MS1" or model == "MP1":
        LIN = True
    else:
        LIN = False
    rarr = np.linspace(r0,rf,rf-r0)*V.dr/1000 #mm
    lineB,  = ax.plot(rarr,arrlist[0][0,r0:rf],label = "B",color = "blue")
    lineP,  = ax.plot(rarr,arrlist[0][-2,r0:rf],label = "P",color = "black")
    linen,  = ax.plot(rarr,arrlist[0][-1,r0:rf],label = "n",color = "gray")
    lineL,  = ax.plot(rarr,sum(arrlist[0][ 1:11])[r0:rf],label = "L",color = "darkviolet")
    if LIN:
        lineLI, = ax.plot(rarr,sum(arrlist[0][11:21])[r0:rf],label = "LI",ls = "--",color = "darkviolet")
    if V.comp:
        linePr, = ax.plot(rarr,arrlist[0][-3,r0:rf],label = "Pr",ls = "--", color = "black")
        lineLr, = ax.plot(rarr,sum(arrlist[0][21:31])[r0:rf],label = "Lr",ls = "--",color = "darkviolet")
    lineList = [lineB,lineP,linen,lineL]
    if LIN:
        lineList.append(lineLI)
        if V.comp:
            lineList.append(linePr)
            lineList.append(lineLr)
    ax.set_ylim(ylim[0],ylim[1])
    ax.legend(loc = "upper left")
    ax.set_yscale("log")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel(r"Concentration [$\mu$m$^{-2}$]")
    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        lineList[0].set_ydata(arrlist[j][0, r0:rf])
        lineList[1].set_ydata(arrlist[j][-2,r0:rf])
        lineList[2].set_ydata(arrlist[j][-1,r0:rf])
        lineList[3].set_ydata(sum(arrlist[j][1:11])[r0:rf])
        if LIN:
            lineList[4].set_ydata(sum(arrlist[j][11:21])[r0:rf])
            if V.comp:
                lineList[5].set_ydata(arrlist[j][-3,r0:rf])
                lineList[6].set_ydata(sum(arrlist[j][21:31])[r0:rf])
        ax.set_title(f"t = {(j+1)*T/(frames-1):.2} min")
        # return the artists set
        return lineList
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(frames),interval = 50)
    writergif = animation.PillowWriter(fps=2) 
    _ = ani.save(name + ".gif", writer = writergif)
