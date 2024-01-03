import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import inv
from sklearn.metrics import mean_squared_error
import math

def Norm_ev(w,vt,v):
    m = vt.T @ v
    l,u = scipy.linalg.lu(m, permute_l=True)
    print(l @ u == vt.T @ v)
    print(inv(l) @ m @ inv(u))
    vt = (inv(l) @ vt.T).T
    v = v @ inv(u)
    print(vt.T @ v)
    return vt, v

def Rec_func(Gx,vt,v):
    mode = v
    eig_func_list = []
    Gx_rec_list = []
    for i in range(Gx.shape[1]):
        eig_func = vt.T @ Gx[:,i,:]
        eig_func_list.append(eig_func)
        Gx_rec_list.append(mode @ eig_func)
    eig_func = np.stack(eig_func_list,axis=1)
    Gx_rec = np.stack(Gx_rec_list,axis=1)
    return eig_func, Gx_rec
    
def validate_data(ori1, ori2, numberx, numbery):
    ori1 = np.reshape(ori1, (-1, numberx, numbery, 1))
    ori2 = np.reshape(ori2, (-1, numberx, numbery, 1))
    return ori1, ori2

def plot_latent(encoded, train_size, rank, path):
    plt.figure(figsize=(8,4), dpi=300)
    for i in range(rank):
        plt.plot(np.linspace(0,train_size,train_size), np.transpose(encoded[i,:]), label='latent dynamics {}'.format(i+1))
    plt.xlabel('Time (s)', size=24)
    plt.ylabel('Dynamics', size=24)
    plt.legend(fontsize=12)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig(path + 'latent dynamics' + '.jpg')
    return

def plot_trace(encoded, S_col, name):
    fig, ax = plt.subplots(figsize=(8,8), dpi=300, subplot_kw={"projection": "3d"})
    # cmap = plt.get_cmap("coolwarm")
    # from matplotlib.collections import LineCollection
    # dotColors = cmap(np.linspace(0,1,len(encoded[:,1])))
    ax.plot(encoded[:S_col,0], encoded[:S_col,1], encoded[:S_col,2], label='rec dynamics')
    ax.plot(encoded[S_col:,0], encoded[S_col:,1], encoded[S_col:,2], label='pred dynamics', linestyle='dashed')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')
    ax.set_zlabel(r'$y_3$')
    plt.tight_layout()
    plt.savefig(name)
    return
   
def plot_lstm(encoded, train_size, time, rank, path, name='lstm dynamics', label=False):
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    c = ['blue','orange','green','red','purple','brown','hotpink','aqua']
    for i in range(rank):
        ax.plot(np.array(time[:train_size]), np.transpose(encoded[i,:train_size]), 
                 label='latent dynamics {}'.format(i+1), c=c[i])
        try:
            ax.plot(np.array(time[train_size:]), np.transpose(encoded[i,train_size:]), ls='-.', 
                    label='predicted dynamics {}'.format(i+1), c=c[i])
        except:
            pass
    plt.xlabel('Time (s)', size=24)
    plt.ylabel('Dynamics', size=24)

    ax.annotate('Prediction', xy=(1.05*time[train_size], np.min(encoded)),size=16,)
    ax.vlines(time[train_size], np.min(encoded), np.max(encoded), colors='black', ls='-')
    if label:
        # plt.legend(fontsize=16, loc='upper center', ncol=rank)
        for i in range(rank):
            # ax.annotate('{}'.format(i+1), xy=(time[0]), np.transpose(encoded[i,0]))
            ax.text(time[0]-0.5, np.transpose(encoded[i,0]),str(i+1),size=16,color=c[i])
    
    plt.title(name,size=24)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig(path + name + '.jpg')
    return
    
def plotEncoded(encoded, train_size, pred_size, time, rank, path, name='lstm dynamics'):
    fig, ax = plt.subplots(figsize=(8,4), dpi=300)
    c = ['blue','orange','green','red','purple','brown','hotpink','aqua']
    for i in range(rank):
        ax.plot(np.linspace(0,train_size,train_size), np.transpose(encoded[i,:train_size]), 
                 label='latent dynamics {}'.format(i+1), c=c[i])
        ax.plot(np.linspace(0,pred_size,pred_size), np.transpose(encoded[i,train_size:]), ls='-.', 
                 label='predicted dynamics {}'.format(i+1), c=c[i])
    plt.xlabel('Time (s)', size=24)
    plt.ylabel('Dynamics', size=24)
    ax.annotate('Prediction', xy=(1.05*train_size, np.min(encoded)),size=16,)
    ax.vlines(train_size, np.min(encoded), np.max(encoded), colors='black', ls='-')
    # plt.legend(fontsize=12)
    plt.title(name,size=24)
    plt.xticks(ticks=time[:pred_size], size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig(path + name)
    return

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

def get_NME(y_true, y_pred):
    NME = np.linalg.norm(y_true - y_pred, 2)
    NME /= np.linalg.norm(y_true, 2)
    return NME

def get_NRMSE(y_true, y_pred):
    NRMSE = get_rmse(y_true, y_pred)
    NRMSE /= np.linalg.norm(y_true, 2)
    # NRMSE /= len(y_true)
    return NRMSE

def get_NRMSE_safe(y_true, y_pred):
    NRMSE_safe = np.linalg.norm(1 - (y_pred + 2)/(y_true + 2), 2)
    NRMSE_safe /= y_true.size
    # NRMSE_safe /= len(y_true)
    return NRMSE_safe

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

def sub_mean(S_full, S_mean):
    for i in range(S_full.shape[-1]):
        S_full[:,i] -= S_mean 
    return S_full

def plus_mean(S_full, S_mean):
    for i in range(S_full.shape[-1]):
        S_full[:,i] += S_mean 
    return S_full

def reshape(inp, numberx, numbery):
    return np.reshape(inp, (-1, numberx, numbery, 1))

def shape(inp, numberx, numbery):
    return np.reshape(inp, (numberx*numbery, -1))

def plot_loss(path, valid, method, loss_type='training'):
    import pandas as pd
    ml_head = pd.read_csv(path+'{}_log.csv'.format(loss_type), nrows=0)
    ml_head = list(ml_head)
    ml_loss = pd.read_csv(path+'{}_log.csv'.format(loss_type))
    ml_loss = np.array(ml_loss)
    epochs = ml_loss[:, 0]
    loss = ml_loss[:, 2]
    if type(valid) != type(None):
        loss_v = ml_loss[:, 4]  
        
    fig,ax = plt.subplots(figsize=(20, 16), dpi=300)
    plt.grid(True)
    plt.xticks(size=60)
    plt.yticks(size=60)
    plt.plot(epochs, loss, 'b', ls='--', lw=3, label='loss')
    if type(valid) != type(None):
        plt.plot(epochs, loss_v, 'r', ls='-.', lw=3, label='val loss')
    plt.xlabel('epoch',size=60)
    plt.ylabel('training loss',size=60)
    plt.legend(loc="upper right",fontsize=60)
    fig.savefig(path+'{} training loss.jpg'.format(method)) 
    return

def plot_loss2(path, method, loss_type='training'):
    import pandas as pd
    ml_head = pd.read_csv(path+'{}_log.csv'.format(loss_type), nrows=0)
    ml_head = list(ml_head)
    ml_loss = pd.read_csv(path+'{}_log.csv'.format(loss_type))
    ml_loss = np.array(ml_loss)
    epochs = ml_loss[:, 0]
    loss = ml_loss[:, 1]

    fig,ax = plt.subplots(figsize=(20, 16), dpi=300)
    plt.grid(True)
    plt.xticks(size=60)
    plt.yticks(size=60)
    plt.plot(epochs, loss, 'b', ls='--', lw=3, label='loss')

    plt.xlabel('epoch',size=60)
    plt.ylabel('training loss',size=60)
    plt.legend(loc="upper right",fontsize=60)
    fig.savefig(path+'{} training loss.jpg'.format(method)) 
    return

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
    return

def txt_write(filename, ob):
    with open(filename, 'w') as f:
        f.write(str(ob))
        f.close
    return