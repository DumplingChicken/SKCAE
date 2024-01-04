import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib import ticker


#----------------------------------Visualization---------------------------------#

def plotContour_fast(x, y, numberx, numbery, path, time, snapshots, cmaps, pltT, levels, minz, maxz, title, ob):
    norm1 = colors.Normalize(vmin=minz, vmax=maxz, clip = True)
    # norm = colors.Normalize(minz, maxz)
    if isinstance(snapshots,list):
        series = enumerate(snapshots[::pltT], start=1)
    else:
        S_list =[]
        for m in range(np.shape(snapshots)[1]):
            S_list.append(snapshots[:,m])
        snapshots = S_list
        series = enumerate(snapshots[::pltT], start=1)
    for i, snapshot in series:
        fig, ax = plt.subplots(figsize=(32, 16), dpi=300)
        ax.set_title('\n {} field at {}s\n'.format(title, time[::pltT][i-1]),size=60) 
        # ss = np.reshape(snapshots[i], (numbery, numberx), order = 'C')
        ss = np.reshape(snapshots[::pltT][i-1], (numbery, numberx), order = 'C')
        plt.contourf(x, y, ss, 100, cmap = cmaps, norm=norm1)
        # cset1 = ax.contourf(x, y, ss, 100, cmap = cmaps, norm=norm1)
        cbar = plt.colorbar(ScalarMappable(norm=norm1, cmap=cmaps), spacing='uniform')
        # cbar = plt.colorbar(cset1, spacing='uniform')
        # cbar = plt.colorbar()
        plt.clim(minz, maxz)
        plt.xlabel('$\it X$',fontsize=60)
        plt.ylabel('$\it Y$',fontsize=60)
        ax.set_aspect(1)
        plt.xticks(np.linspace(-1, 8, 4), size=40)
        plt.yticks(np.linspace(-2, 2, 5), size=40)
        cbar.set_ticks(levels)
        cbar.set_label('{}'.format(ob), size=60)
        cbar.ax.tick_params(labelsize=40)
        
        circle = plt.Circle((0, 0), 0.5, color = 'gray')
        plt.gca().add_patch(circle)
        
        plt.show()      
        fig.savefig(path+r'{} contour_{}.jpg'.format(title, time[::pltT][i-1]))
        plt.close()
    return

def plotModes_fast(x, y, numberx, numbery, rank, path, snapshot, cmaps, ob):
    for i in range(rank):
        # norm1 = colors.Normalize(vmin=minz, vmax=maxz, clip = True)
        ss = np.reshape(snapshot[:,i].real, (numbery, numberx), order = 'C')
        fig, ax = plt.subplots(figsize=(32, 16), dpi=300)
        ax.set_title('\n {} {}\n'.format(ob, i+1),size=60) 
        plt.contourf(x,y,ss,100, cmap = cmaps, extend="both")
        # cset1 = ax.contourf(x,y,ss,100, cmap = cmaps, extend="both")
        plt.xlabel('$\it X$',fontsize=60)
        plt.ylabel('$\it Y$',fontsize=60)
        ax.set_aspect(1)
        plt.xticks(np.linspace(-1, 8, 4), size=40)
        plt.yticks(np.linspace(-2, 2, 5), size=40)
        # plt.clim(minz,maxz)
        cbar = plt.colorbar(cmap = cmaps, extend="both")
        # cbar.set_label('{}'.format(ob), size=30)
        
        cbar.ax.tick_params(labelsize=40)
        cbar.ax.yaxis.get_offset_text().set(size=30)
        circle = plt.Circle((0, 0), 0.5, color = 'gray')
        plt.gca().add_patch(circle)
        
        plt.show()      
        fig.savefig(path+r'{}_{}.jpg'.format(ob, i+1))
    return

def plot_temporal_rmse_fast(x, y, numberx, numbery, cmaps, rmse, path, method, ob):
    rmse = np.reshape(rmse, (numbery, numberx), order = 'C')
    fig, ax = plt.subplots(figsize=(32, 16), dpi=300)
    ax.set_title(('\n Temporal-averaged {} of '.format(ob) + method + ' solutions \n'),size=60, fontproperties='Times New Roman') 
    plt.contourf(x, y, rmse, 100, cmap = cmaps)
    cset1 = ax.contourf(x, y, rmse, 100, cmap = cmaps)
    # plt.xlabel('$\it X$',fontsize=60)
    # plt.ylabel('$\it Y$',fontsize=60)
    # plt.xticks(np.linspace(-1, 8, 4), size=40)
    # plt.yticks(np.linspace(-2, 2, 5), size=40)
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect(1)
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 3))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = plt.colorbar(cset1, format=fmt, cax=cax)
    cbar.set_label('{}'.format(ob), size=60, fontproperties='Times New Roman')
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.yaxis.get_offset_text().set_fontsize(40)
    circle = plt.Circle((0, 0), 0.5, color = 'gray')
    ax.add_patch(circle)
    plt.show()      
    fig.savefig(path+'Temporal-averaged {} of '.format(ob) + method + ' solutions.png')
    return

def plotOne_fast(x, y, numberx, numbery, path, snapshot, cmaps, ob):
    # norm1 = colors.Normalize(vmin=minz, vmax=maxz, clip = True)
    ss = np.reshape(snapshot.real, (numbery, numberx), order = 'C')
    fig, ax = plt.subplots(figsize=(32, 16), dpi=300)
    ax.set_title('\n {} contour \n'.format(ob),size=60) 
    plt.contourf(x,y,ss,100, cmap = cmaps)
    cset1 = ax.contourf(x,y,ss,100, cmap = cmaps)
    plt.xlabel('$\it X$',fontsize=60)
    plt.ylabel('$\it Y$',fontsize=60)
    ax.set_aspect(1)
    plt.xticks(np.linspace(-1, 8, 4), size=40)
    plt.yticks(np.linspace(-2, 2, 5), size=40)
    # plt.clim(minz,maxz)
    cbar = plt.colorbar(cset1)
    cbar.set_label('Vorticity', size=60)
    cbar.ax.tick_params(labelsize=40)
    circle = plt.Circle((0, 0), 0.5, color = 'gray')
    plt.gca().add_patch(circle)
    plt.show()      
    fig.savefig(path+r'{} contour.jpg'.format(ob))
    return

def plotContourErr_fast(x, y, numberx, numbery, path, time, snapshots, cmaps, pltT, levels, minz, maxz, ob):
    for i, snapshot in enumerate(snapshots[::pltT], start=1):
        ss = np.reshape(snapshots[i], (numbery, numberx), order = 'C')
        fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
        ax.set_title('\n {} contour of vortex shedding at {}s\n'.format(ob, time[::pltT][i-1]),size=32) 
        plt.contourf(x,y,ss,100, cmap = cmaps, level=levels)
        cset1 = ax.contourf(x,y,ss,100, cmap = cmaps, level=levels, extend="both")
        plt.xlabel('$\it X$',fontsize=60)
        plt.ylabel('$\it Y$',fontsize=60)
        ax.set_aspect(1)
        plt.xticks(np.linspace(-1, 8, 4), size=40)
        plt.yticks(np.linspace(-2, 2, 5), size=40)
        cbar = plt.colorbar(cset1)
        cbar.set_label('{}'.format(ob), size=30)
        cbar.set_ticks(levels)
        # cbar.set_ticks(np.linspace(minz, maxz, 10))
        # cbar.ax.yaxis.set_major_locator(plt.MultipleLocator(tick))
        # cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        plt.show()      
        fig.savefig(path+r'{} contour_{}.jpg'.format(ob, time[::pltT][i-1]))
    return
    

def get_levels(snapshots, num):
    maxz = 0
    minz = 0
    levels = 0
    if isinstance(snapshots, list):
        for i in range(len(snapshots)):
            if  max(snapshots[i]) > maxz:
                maxz = max(snapshots[i])
            if min(snapshots[i]) < minz:
                minz = min(snapshots[i])
    else:
        for i in range(np.shape(snapshots)[1]):
            if  max(snapshots[:,i]) > maxz:
                maxz = max(snapshots[:,i])
            if min(snapshots[:,i]) < minz:
                minz = min(snapshots[:,i])
    maxz = int(np.ceil(maxz))
    minz = int(np.floor(minz))
    levels = np.linspace(minz, maxz, num) 
    return maxz, minz, levels


