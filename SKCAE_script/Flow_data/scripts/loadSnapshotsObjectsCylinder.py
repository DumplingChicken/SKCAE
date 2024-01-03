import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
import random
from scipy.interpolate import griddata
from matplotlib.cm import ScalarMappable 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib import ticker
import cvxpy as cvx
import math
import cv2
import os
import pywt
from sklearn.metrics import mean_squared_error


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
    
def video(start, end, pltT, dltT, path, time, ob):
    fps = 5
    file_path = path+'{}.mp4'.format(ob) # 导出路径DIVX/mp4v
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX') # mp4
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX') # mp4
    img = cv2.imread(path+ob+str(time[0])+'.jpg')    
    size=(img.shape[1],img.shape[0])
    videoWriter = cv2.VideoWriter(file_path,fourcc,fps,size, isColor=True)
    # 这种情况更适合于照片是从"1.jpg" 开始，然后每张图片名字＋1的那种
    # for num in range(start, end, dt):
    #     frame = cv2.imread(path+ob+str(time[num])+'.jpg')
    #     videoWriter.write(frame) 
    # videoWriter.release()
    # return
    # timesmall = time[:int((end-start)/dltT)]
    for num in time[::pltT]:
        frame = cv2.imread(path+ob+str(num)+'.jpg')
        videoWriter.write(frame) 
    videoWriter.release()
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

def plotEigs(
    eigs,
    show_axes=True,
    show_unit_circle=True,
    figsize=(8, 8),
    title="",
    dpi=None,
    filename=None,
):

    rank = eigs.size
    if dpi is not None:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=figsize)

    plt.title('{} \n'.format(title), fontsize = 40)
    plt.gcf()
    ax = plt.gca()

    labellist = []     
    pointlist = []

    points, = ax.plot(
        eigs.real, eigs.imag, marker='o', 
        color='red', lw=0,
        )
    pointlist.append(points)
    labellist.append("Eigenvalues")  
    
    for i in range(rank):
        ax.annotate('{}'.format(i+1), xy=(eigs[i].real + 0.02, eigs[i].imag), )

        limit = 1.25*max(np.max(abs(eigs.real)),np.max(max(eigs.imag)),1)
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))
        
        # ax.set_xlim((-1, 2))
        # ax.set_ylim((-1, 2))   

        # x and y axes
        if show_axes:
            ax.annotate(
                "",
                xy=(np.max([limit * 0.8, 1.0]), 0.0),
                xytext=(np.min([-limit * 0.8, -1.0]), 0.0),
                arrowprops=dict(arrowstyle="->"),
            )
            ax.annotate(
                "",
                xy=(0.0, np.max([limit * 0.8, 1.0])),
                xytext=(0.0, np.min([-limit * 0.8, -1.0])),
                arrowprops=dict(arrowstyle="->"),
            )

    plt.ylabel("Imaginary part", fontsize = 40)
    plt.xlabel("Real part", fontsize = 40)
    plt.xticks(size=24)
    plt.yticks(size=24)

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="green",
            fill=False,
            label="Unit circle",
            linestyle="--",
        )
        ax.add_artist(unit_circle)

    # Dashed grid
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle("-.")
    ax.grid(True)

    ax.set_aspect("equal")
    a = pointlist
    a.append(unit_circle)
    b = labellist   
    b.append("Unit circle")
    ax.legend([pointlist,unit_circle],[labellist,"Unit circle"])
    ax.legend(a, b, loc = 'upper right', fontsize=20)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()   

#----------------------------------Error Analysis---------------------------------#

def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return mean_squared_error(records_real, records_predict)
    else:
        return None 

def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

def relative_error(y_true, y_pred):
    relative_error = (y_true - y_pred) / y_true
    return relative_error

def relative_abserror(y_true, y_pred):
    relative_abserror = abs(y_true - y_pred) / (1 + abs(y_true))
    return relative_abserror

def err(y_true, y_pred):
    err = (y_true - y_pred) 
    return err

def err_list(row, snapshots, snapshots2):
    err = []
    arr=np.zeros(row)
    for xx in range(min(len(snapshots), len(snapshots2))):
        for yy in range(row):
            arr[yy] = relative_abserror(snapshots[xx][yy], snapshots2[xx][yy])
        err.append(arr)
        arr=np.zeros(row)
    return err, arr

def get_NME(y_true, y_pred):
    NME = np.linalg.norm(y_true - y_pred, 2)
    NME /= np.linalg.norm(y_true, 2)
    return NME

def get_NRMSE(y_true, y_pred):
    NRMSE = get_rmse(y_true, y_pred)
    NRMSE /= np.linalg.norm(y_true, 2)
    # NRMSE /= len(y_true)
    return NRMSE

# def get_NRMSE(y_true, y_pred):
#     a = np.ones((len(y_true)))
#     NRMSE = mean_squared_error(a, y_pred / y_true)
#     return NRMSE

def get_NRMSE_safe(y_true, y_pred):
    NRMSE_safe = np.linalg.norm(1 - (y_pred + 20)/(y_true + 20), 2)
    NRMSE_safe /= y_true.size
    # NRMSE_safe /= len(y_true)
    return NRMSE_safe

def plot_spatial_NME(snapshots, snapshots2, startT, endT, dltT, path, method, ob):
    rmse=np.zeros(len(snapshots))
    fig,ax = plt.subplots(figsize=(26, 16), dpi=300)
    for i in range(len(snapshots)):
        if ob == 'NME':
            rmse[i] = get_NME(snapshots[i], snapshots2[i])
        elif ob == 'RMSE':
            rmse[i] = get_rmse(snapshots[i], snapshots2[i])
        elif ob == 'NRMSE':
            rmse[i] = get_NRMSE(snapshots[i], snapshots2[i])
        elif ob == 'NRMSEsafe':
            rmse[i] = get_NRMSE_safe(snapshots[i], snapshots2[i])
        else:
            raise ValueError('one certain error method must be defined') 
    plt.plot(np.linspace(startT,endT,len(snapshots)),rmse, lw=3)
    plt.xlabel('Time (s)',fontsize=50)
    plt.ylabel('{}'.format(ob),fontsize=50)
    plt.xticks(size=40)
    plt.yticks(size=40)
    ax.yaxis.get_offset_text().set(size=30)
    plt.title(('\n Spatial-averaged {} of '.format(ob) + method + ' solutions \n'),size=60) 
    plt.show()
    fig.savefig(path+'Spatial-averaged {} of '.format(ob) + method + ' solutions.png')   
    # with open(path+'rmse.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(rmse)
    fileObject = open(path+'{}.csv'.format(ob), 'w')  
    cc=0
    for ip in rmse:  
        fileObject.write(str(startT+dltT*cc)+',') 
        fileObject.write(str(ip))  
        fileObject.write('\n') 
        cc+=1
    fileObject.close()  
    return rmse

def get_spatial_NME(snapshots, snapshots2, ob):
    rmse=np.zeros(len(snapshots))
    for i in range(len(snapshots)):
        if ob == 'NME':
            rmse[i] = get_NME(snapshots[i], snapshots2[i])
        elif ob == 'RMSE':
            rmse[i] = get_rmse(snapshots[i], snapshots2[i])
        elif ob == 'NRMSE':
            rmse[i] = get_NRMSE(snapshots[i], snapshots2[i])
        elif ob == 'NRMSEsafe':
            rmse[i] = get_NRMSE_safe(snapshots[i], snapshots2[i])
        else:
            raise ValueError('one certain error method must be defined')  
    return rmse

def plot_spatial_rmse(snapshots, snapshots2, startT, endT, dltT, path, method):
    rmse=np.zeros(len(snapshots))
    fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
    for i in range(len(snapshots)):
        rmse[i]=get_rmse(snapshots[i], snapshots2[i])
    plt.plot(np.linspace(startT,endT,len(snapshots)),rmse, lw=3)
    plt.xlabel('Time (s)',fontsize=50)
    plt.ylabel('RMSE',fontsize=50)
    plt.xticks(size=40)
    plt.yticks(size=40)
    ax.yaxis.get_offset_text().set(size=30)
    plt.title(('\n Spatial-averaged RMSE of ' + method + ' solutions \n'),size=60) 
    plt.show()
    fig.savefig(path+'Spatial-averaged RMSE of ' + method + ' solutions.png')   
    # with open(path+'rmse.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(rmse)
    fileObject = open(path+'rmse.csv', 'w')  
    cc=0
    for ip in rmse:  
        fileObject.write(str(startT+dltT*cc)+',') 
        fileObject.write(str(ip))  
        fileObject.write('\n') 
        cc+=1
    fileObject.close()  
    return rmse

def get_tmporal_rmse(snapshots, snapshots2, arr):
    temporal_true=np.zeros((len(snapshots),len(arr)))
    temporal_predict=np.zeros((len(snapshots),len(arr)))
    rmse=np.zeros(len(arr))
    for i in range(len(snapshots)):
        for j in range(len(arr)):
            temporal_true[i][j] = snapshots[i][j]
            temporal_predict[i][j] = snapshots2[i][j]
    for i in range(len(arr)):
        rmse[i]=get_rmse(temporal_true[:,i], temporal_predict[:,i])
    return rmse

def get_tmporal_NME(snapshots, snapshots2, arr, ob):
    temporal_true=np.zeros((len(snapshots),len(arr)))
    temporal_predict=np.zeros((len(snapshots),len(arr)))
    rmse=np.zeros(len(arr))
    for i in range(len(snapshots)):
        for j in range(len(arr)):
            temporal_true[i][j] = snapshots[i][j]
            temporal_predict[i][j] = snapshots2[i][j]
    for i in range(len(arr)):
        if ob == 'NME':
            rmse[i] = get_NME(temporal_true[:,i], temporal_predict[:,i])
        elif ob == 'RMSE':
            rmse[i] = get_rmse(temporal_true[:,i], temporal_predict[:,i])
        elif ob == 'NRMSE':
            rmse[i] = get_NRMSE(temporal_true[:,i], temporal_predict[:,i])
        elif ob == 'NRMSEsafe':
            rmse[i] = get_NRMSE_safe(temporal_true[:,i], temporal_predict[:,i])
        else:
            raise ValueError('one certain error method must be defined') 
    return rmse

def plot_temporal_rmse(x, y, xi, yi, cmaps, snapshots, snapshots2, arr, path, method, ob):
    rmse = get_tmporal_rmse(snapshots, snapshots2, arr)
    fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
    ax.set_title(('\n Temporal-averaged {} of '.format(ob) + method + ' solutions \n'),size=32) 
    zi = griddata((x,y),rmse,(xi,yi),method='linear')
    plt.contourf(xi,yi,zi,100, cmap = cmaps)
    cset1 = ax.contourf(xi,yi,zi,100, cmap = cmaps, extend='both')
    plt.xlabel('X (m)',fontsize=30)
    plt.ylabel('Y (m)',fontsize=30)
    plt.xticks(size=24)
    plt.yticks(size=24)
    cbar = plt.colorbar(cset1)
    cbar.set_label('{}'.format(ob), size=30)
    plt.show()      
    fig.savefig(path+'Temporal-averaged {} of '.format(ob) + method + ' solutions.png')
    return


def get_rmseList(records_real, records_predict):    
    rmse = []
    m = min(len(records_real), len(records_predict))
    records_real = records_real[:m]
    records_predict = records_predict[:m]
    for i in range(m):
        rmse.append(get_rmse(records_real[:i], records_predict[:i]))
    return rmse

def get_l2err(y_true, y_pred):
    l2err = np.linalg.norm(y_pred - y_true, 2)
    return l2err

def l2err_list(row, snapshots, snapshots2):
    err = []
    arr=np.zeros(row)
    for xx in range(min(len(snapshots), len(snapshots2))):
        for yy in range(row):
            arr[yy] = get_l2err(snapshots[xx][yy], snapshots2[xx][yy])
        err.append(arr)
        arr=np.zeros(row)
    return err, arr

# =============================================================================
# Plot DMD
# =============================================================================
def compute_svd(X, svd_rank=0):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T

    def omega(x):
        return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

    if svd_rank == 0:
        beta = np.divide(*sorted(X.shape))
        tau = np.median(s) * omega(beta)
        rank = np.sum(s > tau)
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(s**2 / (s**2).sum())
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])
    else:
        rank = X.shape[1]
    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]
    return U, s, V

def plotDynamics(dynamics, time, title, path):
    fig, ax = plt.subplots(figsize=(24, 16), dpi=300)     
    M = ['o', '^', 'v', '<', '>', 's', '*', '+', 'x', 'd', 'D']
    C = ['red', 'green', 'blue', 'orange', 'yellow', 'brown', 'purple', 'orange', 'cyan']
    L = ['-', '--', '.', '-.']
    
    i=0
    c=0
    for dynamic in dynamics:
        i+=1
        c+=1        
        if c >= len(M):
            c -= len(M) 
        plt.plot(time[:dynamic.shape[0]], dynamic.real, label='Dynamics {}'.format(i), marker=M[c-1],ms=15, lw=2)
    plt.xlabel('Time (s)', size=60)
    plt.ylabel('Dynamics', size=60)
    plt.xticks(size=40)
    plt.yticks(size=40)
    # ax.set_yscale('log')
    plt.title('\n {} \n'.format(title), fontsize=60)
    plt.legend(fontsize=32, loc="best")
    fig.savefig(path+'{}.jpg'.format(title))
    
    np.savetxt(path + '{}.csv'.format(title), dynamics, delimiter = ',')
    return

def plotSigma(sigma, rank_r=0.99, title='', filename=None):
    energy = np.cumsum(sigma**2 / (sigma**2).sum())
    if -1.0 < rank_r < 1.0:  
        rank = np.searchsorted(energy, rank_r) + 1
    elif rank_r == -1:
        rank = sigma.shape[0] - 1
    else:
        rank = rank_r - 1
    normal = sigma / np.cumsum(sigma)
    r = np.linspace(1, sigma.size, sigma.size)
    
        
    # r = range(energy.size)
    
    plt.figure(figsize=(16, 10), dpi=300) 
    plt.title('\n{}\n'.format(title), fontsize = 40)
    plt.gcf()
    ax = plt.gca()
    
    interval = 5
    
    for i in range(rank):
        ax.plot(
            r[i], normal[i], marker='o', 
            color='b', lw=0, ms=8,
            )
          
    for i in range(rank, sigma.size):
        ax.plot(
            r[i], normal[i], marker='o', 
            color='gray', lw=0, ms=5,
            )  
        
        # for i in range(0, rank-1, interval):
        #     ax.scatter(
        #         x=r[i], y=normal[i], marker='1', s=10,  
        #         color='red', 
        #         )
        #     ax.annotate('mode {}'.format(i+1), xy=(r[i]+2, normal[i]), )
            
    # ax.annotate('mode {}'.format(rank+1), xy=(r[rank]+2, normal[rank]), )
    ax.plot(
        r[rank], normal[rank], marker='o', 
        color='red', lw=0, ms=10,
        )  
    ax.annotate('mode {}'.format(rank+1), xy=(r[rank], normal[rank]), 
                xytext=(r[rank]+10, normal[rank]+0.2), arrowprops=dict(facecolor='black', shrink=0.1), size=30,)
        
    plt.ylabel(r"$\sigma_r$", fontsize = 40)
    plt.xlabel("rank", fontsize = 40)
    plt.xticks(size=32)
    plt.yticks(size=32)
    ax.set_yscale('log')
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle("-.")
    ax.grid(True)
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()  
#----------------------------------Write CSV---------------------------------#

def txt_write(filename, ob):
    with open(filename, 'w') as f:
        f.write(str(ob))
        f.close
    return


def csv_write(path, folder, array, ob, pts):
    if not os.path.exists(path+folder):
        os.makedirs(path+folder) 
    fileObject = open(path+ob, 'w')  
    # cc=0
    for ip in range(len(array)):  
        fileObject.write(str(pts[ip,0])+',') 
        fileObject.write(str(pts[ip,1])+',') 
        fileObject.write(str(array[ip]))  
        fileObject.write('\n') 
        # cc+=1
    fileObject.close() 