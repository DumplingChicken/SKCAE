import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcdefaults, rcParams
from sklearn.metrics import mean_squared_error
import math

plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'stix'
font={'family': 'Times New Roman', 'math_fontfamily':'stix'}

def plot_spatial_NME(snapshots, snapshots2, startT, trainT, endT, dltT, path, method, ob):
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
        if rmse[i] == np.nan:
            rmse[i] = 0
    plt.plot(np.linspace(startT,endT,len(snapshots)),rmse, lw=3)
    plt.xlabel('Time (s)',fontsize=50)
    plt.ylabel('{}'.format(ob),fontsize=50)
    plt.xticks(size=40)
    plt.yticks(size=40)
    ax.yaxis.get_offset_text().set(size=30)
    ax.annotate('Reconstruction', xy=(0.6*trainT, min(rmse)),size=60,)
    ax.annotate('Prediction', xy=(1.05*trainT, min(rmse)),size=60,)
    ax.vlines(trainT, min(rmse), max(rmse), colors='black', ls='-.')
    plt.title(('\n Spatial-averaged {} of '.format(ob) + method + ' solutions \n'),size=60) 
    plt.show()
    fig.savefig(path+'Spatial-averaged {} of '.format(ob) + method + ' solutions.png')  
    # fig.savefig(path+'Spatial-averaged {} of '.format(ob) + method + ' solutions.svg')  
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