import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np

############################################################################
#Plotter functions
############################################################################

def BrothPlotter(model,V,tarr,yarr,scale = "log",ylim=False,figtitle = 0,plotn = True,figsize = False): #Not plotting infected substates
    if figsize:
        plt.figure(figsize = figsize)
    if not ylim:
        ylim = (1E3,np.max(yarr)*1.5)
    N,comp = V.N,V.comp
    tarr = tarr/60
    if model == "M0":
        L_label = r"I$^R$"
        P_label = "R"
    elif model == "M1":
        L_label = r"I$^P$"
        P_label = "P"
        LI = sum(yarr[N+1:2*N+1])
    
    if plotn:
        plt.plot(tarr,yarr[-1],label = "n",color = "gray")
    plt.plot(tarr,yarr[-2],label = P_label,color = "black")
    plt.plot(tarr,yarr[0], label = "B",color = "blue")
    plt.plot(tarr,sum(yarr[1:N+1]),label = L_label,color = "darkviolet")
    plt.plot(tarr,LI,label = "L",color = "crimson") if model == "M1" else None
    if comp:
        Lr = sum(yarr[2*N+1:3*N+1])
        plt.plot(tarr,yarr[-3],label = r"R",ls = "--",color = "black")
        plt.plot(tarr,Lr,label = r"I$^R$",ls = "--",color = "darkviolet")
    plt.yscale(scale)
    plt.ylim(ylim)
    plt.ylabel(r"Concentration [ml$^{-1}$]",fontsize = 12)
    plt.xlabel("Time [h]", fontsize = 12)
    plt.grid(axis="y", which = "major")
    plt.legend()
    if figtitle:
        plt.savefig(figtitle + ".jpg",bbox_inches='tight')
    return

def BrothPlotter_AllStates(model,V,tarr,yarr,scale = "log",ylim = False,loc = None,figtitle = 0):
    if not ylim:
        ylim = (1E3,np.max(yarr)*1.5)
    N,comp = V.N,V.comp
    tarr = tarr/60
    if model == "M1":
        LIN = True
    else:
        LIN = False
    for i,var in enumerate(yarr):
        ls = "-"
        alpha = 1
        label = None
        if i == 0: #Susceptible
            label = r"$B$"
            color = "blue"
        elif i == len(yarr)-2: #Phages
            label = r"$P$"
            color = "black"
        elif i == len(yarr)-1: #Nutrition
            label = r"$n$"
            color = "gray"
        if i > 0 and i <=N: #Noninhibited infected
            color = "darkviolet"
            if i == 1:
                label = r"$L_1$"
            elif i == N:
                label = r"$L_{10}$"
                ls = "dotted"
            else:
                alpha = 0.1
        if LIN:
            if i > N and i < 2*N+1: #Inhibited infected
                color = "crimson"
                if i == N+1:
                    label = r"$L_{I,1}$"
                elif i == 2*N:
                    label = r"$L_{I,10}$"
                    ls = "dotted"
                else:
                    alpha = 0.1
            if comp:
                if i >2*N and i < 2*N+1:
                    color = "darkviolet"
                    ls = "--"
                    if i == 2*N+1:
                        label = "Lr1"
                    elif i == 3*N:
                        label = f"LI{N}"
                        ls = "dotted"
                    else:
                        alpha = 0.1
        plt.plot(tarr,var,label = label,ls = ls,alpha = alpha,color = color)
    if scale == "log":
        plt.ylim(ylim[0],ylim[1])
    plt.yscale(scale)
    plt.ylabel(r"Concentration [ml$^{-1}$]")
    plt.xlabel("Time [min]")
    plt.legend(loc = loc)
    plt.grid(axis="y", which = "major")
    if figtitle:
        plt.savefig(figtitle+".jpg")
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

def PlaquePlotter(sim,savetimes,V,T,model,ylim = (1e-3,20),r0 = 0,rf = False,legendloc="upper right",Btot = False,figtitle = 0):
    frames = len(sim)
    arrlist = [frame for frame in sim]
    if not rf:
        rf = V.l #Plot whole system unless specified
        
    # Define the number of rows and columns for the subplots
    rows = 4
    cols = 1
    fig, ax = plt.subplots(rows, cols,figsize=(5,9))# Create subplots
    fig.subplots_adjust(hspace=0)
    axs = ax.ravel()  # Flatten the array of subplots
     
    if model == "MS1" or model == "MP1":
        LIN = True
    else:
        LIN = False
    rarr = np.linspace(r0,rf,rf-r0)*V.dr/1000 #mm
    
    for i in range (1,frames):
        ax = axs[i-1]
        lineB,  = ax.plot(rarr,arrlist[i][0 ,r0:rf],label = r"$B$",color = "blue")
        lineP,  = ax.plot(rarr,arrlist[i][-2,r0:rf],label = r"$P$",color = "black")
        linen,  = ax.plot(rarr,arrlist[i][-1,r0:rf],label = r"$n$",color = "gray")
        if LIN:
            lineLI, = ax.plot(rarr,sum(arrlist[i][V.N+1:2*V.N+1])[r0:rf],label = r"$L$",color = "crimson")
            lineL,  = ax.plot(rarr,sum(arrlist[i][ 1:V.N+1])[r0:rf],label = r"$I^P$",color = "darkviolet")
        else:
            lineL,  = ax.plot(rarr,sum(arrlist[i][ 1:V.N+1])[r0:rf],label = r"$I^R$",color = "darkviolet")
        if V.comp:
            linePr, = ax.plot(rarr,arrlist[i][-3,r0:rf],label = r"$R$",ls = "--", color = "black")
            lineLr, = ax.plot(rarr,sum(arrlist[i][2*V.N+1:3*V.N+1])[r0:rf],label = r"$I^R$",ls = "--",color = "darkviolet")
        if Btot:
            lineBt,  = ax.plot(rarr,sum(arrlist[i][:(1+LIN+V.comp)*V.N+1])[r0:rf],label = r"$B_{tot}$",ls = (0,(1,3)),color = "g")
            
        _ = ax.set_ylim(ylim[0],ylim[1])
        _ = ax.set_xlim(0,2)
        ax.tick_params(axis = "both",labelsize = 9)
        if i != frames - 1:
            _ = ax.set_xticks(np.linspace(0,2,9),[None]*9)  
            ax.tick_params(axis='x', direction='in')
        else:
            _ = ax.set_xticks(np.linspace(0,2,9),np.linspace(0,2,9))
            ax.tick_params(axis='x', direction='out')
        
        _ = ax.set_yscale("log")
        if i == frames - 1:
            _ = ax.set_xlabel("r [mm]")
        _ = ax.set_ylabel(r"Concentration [$\mu$m$^{-2}$]",fontsize = 10)
        _ = ax.set_title(f"t = {savetimes[i]} min",y=1.0, pad=-12,loc = "left",position = (0.02,1),fontsize=10)
       
    _ = axs[0].legend(loc = legendloc)
        
    plt.tight_layout()  # Adjust the layout
    if figtitle:
        plt.savefig(figtitle + ".jpg")
    else: 
        plt.show()
    return


################################################################################################################
#For outputting animations
################################################################################################################

def GifGenerator(sim,savetimes,V,model,name,ylim = (1,200),r0 = 0,rf = False,legendloc="upper left",Btot = False):
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
    lineB,  = ax.plot(rarr,arrlist[0][0,r0:rf],label = r"$B$",color = "blue")
    lineP,  = ax.plot(rarr,arrlist[0][-2,r0:rf],label = r"$P$",color = "black")
    linen,  = ax.plot(rarr,arrlist[0][-1,r0:rf],label = r"$n$",color = "gray")
    lineL,  = ax.plot(rarr,sum(arrlist[0][ 1:V.N+1])[r0:rf],label = r"$L$",color = "darkviolet")
    if LIN:
        lineLI, = ax.plot(rarr,sum(arrlist[0][V.N+1:2*V.N+1])[r0:rf],label = r"$L_I$",color = "crimson")
    if V.comp:
        linePr, = ax.plot(rarr,arrlist[0][-3,r0:rf],label = r"$P_r$",ls = "--", color = "black")
        lineLr, = ax.plot(rarr,sum(arrlist[0][2*V.N+1:3*V.N+1])[r0:rf],label = r"$L_r$",ls = "--",color = "darkviolet")
    if Btot:
        lineBt,  = ax.plot(rarr,sum(arrlist[0][:(1+LIN+V.comp)*V.N+1])[r0:rf],label = r"$B_{tot}$",ls = (0,(1,3)),color = "g")
    lineList = [lineB,lineP,linen,lineL]
    if LIN:
        lineList.append(lineLI)
        if V.comp:
            lineList.append(linePr)
            lineList.append(lineLr)
    if Btot:
        lineList.append(lineBt)
    _ = ax.set_ylim(ylim[0],ylim[1])
    _ = ax.legend(loc = legendloc)
    _ = ax.set_yscale("log")
    _ = ax.set_xlabel("r [mm]")
    _ = ax.set_ylabel(r"Concentration [$\mu$m$^{-2}$]")
    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        lineList[0].set_ydata(arrlist[j][0, r0:rf])
        lineList[1].set_ydata(arrlist[j][-2,r0:rf])
        lineList[2].set_ydata(arrlist[j][-1,r0:rf])
        lineList[3].set_ydata(sum(arrlist[j][1:V.N+1])[r0:rf])
        if LIN:
            lineList[4].set_ydata(sum(arrlist[j][V.N+1:2*V.N+1])[r0:rf])
            if V.comp:
                lineList[5].set_ydata(arrlist[j][-3,r0:rf])
                lineList[6].set_ydata(sum(arrlist[j][2*V.N+1:3*V.N+1])[r0:rf])
        if Btot:
            lineList[4+LIN+2*V.comp].set_ydata(sum(arrlist[j][:(1+LIN+V.comp)*V.N+1])[r0:rf])
        ax.set_title(f"t = {savetimes[j]} min")
        # return the artists set
        return lineList
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(frames),interval = 50)
    writergif = animation.PillowWriter(fps=2) 
    _ = ani.save(name + ".gif", writer = writergif)

