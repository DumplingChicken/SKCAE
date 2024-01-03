import sys
sys.path.append('../')
sys.path.append('../../utils')
import numpy as np
import matplotlib.pyplot as plt
import os
import time as TT
import ML_plot
from sklearn.preprocessing import MinMaxScaler
import ML_utils as ut
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
import scipy

case = 'duffing'
method = 'Koopman'  
load = False 

# Hyperparams
offset = 5 
activ = 'tanh'
epoch = 1000 
batch_size = 32
patience = 500 
valid = True 
seed = 1
rank = 5 
    
# Plot
contour = True 
video = True 
plotOrigin = False 
plotPredict = True 
plotErr = False 

# Pre&post-process
normalize = False
writeData = True

# Noise
Noise = False
noise_sigma = 0.5

# Mask
Mask = False
mask_size = [64, 16] 

# Test
checkPoint = False 
test = True 

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# =============================================================================
# Load data
# =============================================================================
numberx, numbery = 32, 32  

if case == 'duffing':
    S_col_full = 600
    alltime = 30
    fileName = '../duffing600x32x32_30s.npy'
elif case == 'fixed':
    S_col_full = 500
    alltime = 10
    fileName = '../attractor500x32x32_10s.npy'
    
S_col = int(0.8*S_col_full)
dltT = alltime / S_col_full
    
startT = 0
trainT = startT + S_col * dltT
endT = startT + S_col_full * dltT

train_size = int((trainT - startT)/dltT)
full_size = int((trainT -endT)/dltT)
T_col = S_col_full - S_col

S_full = np.load(fileName)[:,:,:,:2]
S_t = np.load(fileName)[:,:,:,-1:]

S_train = S_full[:S_col,:,:,:]
S_test = S_full[S_col:,:,:,:]

maxz = float('%.2f' % np.max(S_full))
minz = float('%.2f' % np.min(S_full))
levels = np.linspace(minz, maxz, 11)
for i in range(levels.size):
    levels[i] = float('%.2f' % levels[i])

time = []
for i in np.arange(startT, endT, dltT):
    i = float('%.3f' % i)
    time.append(i)
snapshots = list(S_full)

cmaps = plt.cm.jet

# =============================================================================
# Constructor
# =============================================================================
projName = '{}_{}_'.format(case, method)

if 'Koopman' in method:
    projName += '_{}dt'.format(offset)

path_old = '../ML_' + projName +'_'+ activ + '_output/' 
if method == 'DMD':
    path_old = '../' + projName + '_rank' + str(rank)  + '_output/' 
 
if Noise or Mask:    
    if Noise:
        load = True
        if noise_sigma != 0:
            projName_old = projName
            projName += '_Noise{}'.format(noise_sigma)
            S_full_n = S_full * np.random.normal(0, noise_sigma, S_full.shape)
            S_full_n += S_full    
            S_train_n = S_full_n[:,:S_col]
        else:
            raise ValueError('Sigma cannot equal zero when switch_noise is ON')
    if Mask:
        load = True
        projName_old = projName
        projName += '_Mask{}'.format(mask_size)
        x_r = np.random.randint(numberx-mask_size[0])
        y_r = np.random.randint(numbery-mask_size[1])
        mask = np.ones((numberx, numbery))
        mask[x_r:x_r+mask_size[0], y_r:y_r+mask_size[1]] = 0
        mask = np.reshape(mask, (numberx*numbery), order='F')
        S_mask = np.ones(S_full.shape)
        for i in range(S_col_full):
            S_mask[:,i] = mask
        S_full_n = S_full * S_mask
        S_train_n = S_full_n[:,:S_col]
else:
    S_full_n = S_full
    S_train_n = S_train
    
snapshots_n = list(S_full_n.T)
        
path = '../ML_' + projName +'_'+ activ + '_output/'   
if method == 'DMD':
    path = '../' + projName + '_rank' + str(rank)  + '_output/'  

if not os.path.exists(path):
    os.makedirs(path) 
writepath = path + 'output_data'
if not os.path.exists(writepath):
    os.makedirs(writepath) 
    
plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'stix'
font={'family': 'Times New Roman', 'math_fontfamily':'stix'}

if checkPoint:    
    input('snapshots have been read, press Enter to continue')
offlineStart = TT.time()   

data = np.reshape(S_full, (-1, numberx*numbery, 2))

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
cmap = plt.get_cmap("coolwarm")
ax.tricontour(data[:,:,0].flatten(), data[:,:,1].flatten(), S_t.flatten(), cmap="RdBu_r")
plt.legend()
fig.savefig(path+'/trace_true.jpg')

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
cmap = plt.get_cmap("coolwarm")
ax.tricontour(data[:S_col,:,0].flatten(), data[:S_col,:,1].flatten(), S_t[:S_col,:,:,:].flatten(), cmap="RdBu_r")
plt.legend()
fig.savefig(path+'/trace_train.jpg')

# =============================================================================
# Koopman
# =============================================================================   
if 'Koopman' in method:
    import ML_Koopman_Time as ML_KPM 
    operator = 'Simple'
    from CL_Simple import Linear

    method = 'Koopman'

    if normalize:   
        from sklearn import preprocessing
        scaler = preprocessing.MinMaxScaler()
        S_full = scaler.fit_transform(S_full)

    if load:
        if Noise or Mask:
            encoder = load_model(path_old+'encoder.h5')
            decoder = load_model(path_old+'decoder.h5')
            autoencoder = load_model(path_old+'autoencoder.h5',custom_objects={"Linear": Linear})
            koopman = load_model(path_old+'koopman.h5',custom_objects={"Linear": Linear})
        else:
            encoder = load_model(path+'encoder.h5')
            decoder = load_model(path+'decoder.h5')
            autoencoder = load_model(path+'autoencoder.h5',custom_objects={"Linear": Linear})
            koopman = load_model(path+'koopman.h5',custom_objects={"Linear": Linear})
            KPO = koopman.get_weights()[0]
    else:  
        offlineStart = TT.time()
        autoencoder, encoder, decoder, koopman, KPO = \
        ML_KPM.ML_train(S_full, numberx, numbery, path, offset, activ, 
                        valid=valid, opt='adam', epoch=epoch, batch_size=batch_size, 
                        patience=patience, operator=operator, t=rank)
        
    offlineEnd = TT.time()
    onlineStart = TT.perf_counter()

    eigs, eigvec_l, eigvec = scipy.linalg.eig(KPO, left=True)
    idx = eigs.argsort()[::-1]   
    eigs = eigs[idx]
    eigvec = eigvec[:,idx] 
    eigvec_l = eigvec_l[:,idx] 

    eigvec_l, eigvec = ut.Norm_ev(eigs, eigvec_l, eigvec)

    encoded_true = ML_KPM.encoder_pred(S_full, numberx, numbery, encoder) 
    encoded_ori = ML_KPM.encoder_pred(S_train, numberx, numbery, encoder)
    
    encoded = encoded_ori
    encoded_pred = np.zeros((encoded_ori.shape[0],S_col_full+offset))

    for m, i in enumerate(range(S_col, S_col_full, offset), 0): 
        op_num = m + 1
        print('Koopman cycle:{}, {}-{} to {}-{}'.format(op_num,m*offset,i,op_num*offset,i+offset))
        encoded = ML_KPM.koopman_pred(encoded, numberx, numbery, koopman) 
        encoded_pred[:,i:i+offset] = encoded[:,-offset:] 
            
    encoded_pred = encoded_pred[:,:S_col_full]
    encoded_pred[:,:S_col] = encoded_ori
    
    eig_func = eigvec_l.T @ encoded_ori
    Gx_rec = eigvec @ eig_func
    
    if type(eigvec)!=type(None):
        mode_real = ML_KPM.decoder_pred(eigvec.real, numberx, numbery, decoder)
        mode_imag = ML_KPM.decoder_pred(eigvec.imag, numberx, numbery, decoder)
    
    S_pred = ML_KPM.decoder_pred(encoded_pred, numberx, numbery, decoder) 
    onlineEnd = TT.perf_counter()

    if normalize:
        S_pred = scaler.inverse_transform(S_pred)
        S_full = scaler.inverse_transform(S_full)

    snapshots2 = list(S_pred)
    
    # write and save
    if not load:
        np.save(path+'/encoded_true', encoded_true)
        np.save(path+'/encoded_pred', encoded_pred)
        np.save(path+'/encoded', encoded)
        np.save(path+'/KPO', KPO)
        np.save(path+'/eigs', eigs)
        np.save(path+'/eigvec', eigvec)
        np.save(path+'/mode_real', mode_real)
        np.save(path+'/mode_imag', mode_imag)
        np.savetxt(path + '/encoded.csv', encoded, delimiter=',')
        np.savetxt(path + '/KPO.csv', KPO, delimiter=',')
        np.savetxt(path + '/eigs.csv', eigs, delimiter=',')
        np.savetxt(path + '/eigvec.csv', eigvec, delimiter=',')
        ut.txt_write(path+'/seed.txt', seed)

    # plot
    print('the Koopman operator is {}'.format(KPO))
    print('the eigenvalue is {}'.format(eigs))

    ut.plotEigs(eigs,title='Eigenvalues',filename=path+'/Eigenvalues.jpg',dpi=300,show_unit_circle=True)

    plot_r = min(encoded_ori.shape[0], 8)
    ut.plot_lstm(encoded_true[idx,:], train_size, time, plot_r, path, name='Encoded true', label=True)
    ut.plot_lstm(encoded_pred[idx,:], train_size, time, plot_r, path, name='Encoded pred', label=True)
        

# =============================================================================
# Time
# =============================================================================
elif method == 'DMD':
    from pydmd import DMD
    offlineEnd = TT.time()
    onlineStart = TT.perf_counter()
    dmd = DMD(svd_rank=rank, exact=True)
    shape = S_train.shape
    S_train = np.reshape(S_train, (-1, S_col))
    dmd.fit(S_train)
    dmd.dmd_time['tend'] = int((endT-startT)/dltT) 
    dmd.dmd_time
    S_pred = dmd.reconstructed_data.real[:,:S_col_full]
    S_pred = np.reshape(S_pred, (-1, S_col_full))
    snapshots2 = list(np.transpose(S_pred))
    onlineEnd = TT.perf_counter()


if not load:
    file = open(path+'time.txt', 'w')
    offlineTime = offlineEnd - offlineStart
    onlineTime = onlineEnd - onlineStart
    print('offline time: '+ str(offlineTime))
    print('online time: '+ str(onlineTime))
    file.write('offline time: '+ str(offlineTime) + '\n')
    file.write('online time: '+ str(onlineTime) + '\n')
    file.close()

K.clear_session()

# =============================================================================
# Visualization
# =============================================================================
data_r = np.reshape(S_pred, (-1, numberx*numbery, 2))

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.tricontour(data_r[:,:,0].flatten(), data_r[:,:,1].flatten(), S_t.flatten(), cmap="RdBu_r")
plt.legend()
fig.savefig(path+'/trace_pred.jpg')

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.tricontour(data_r[:S_col,:,0].flatten(), data_r[:S_col,:,1].flatten(), S_t[:S_col,:,:,:].flatten(), cmap="RdBu_r")
plt.legend()
fig.savefig(path+'/trace_rec.jpg')

# =============================================================================
# Error estimation
# =============================================================================
snapshots = list(np.reshape(S_full, (-1, numberx*numbery*2)))
snapshots2 = list(np.reshape(S_pred, (-1, numberx*numbery*2)))

minT = int(min(len(snapshots), len(snapshots2)))    
snapshots = snapshots[:minT]
snapshots2 = snapshots2[:minT]

spatial_RMSE = ML_plot.plot_spatial_NME(snapshots, snapshots2, time[0],
                                   trainT, time[minT-1], dltT, path, 
                                   method='{}'.format(method), ob='RMSE')


arr = np.zeros((numberx*numbery))
temporal_RMSE = ut.get_tmporal_NME(snapshots[:minT], snapshots2[:minT], arr, 'RMSE') 

RMSE_train = ut.get_rmse(snapshots[:S_col], snapshots2[:S_col])
if minT > S_col:
    RMSE_pred = ut.get_rmse(snapshots[S_col:], snapshots2[S_col:])
    RMSE_all = ut.get_rmse(snapshots, snapshots2)
else:
    RMSE_pred = np.nan
    RMSE_all = np.nan
RMSE_list = [RMSE_train, RMSE_pred, RMSE_all]


loc = locals()
def get_variable_name(variable):
    for k,v in loc.items():
        if loc[k] is variable:
            return k
        
def write_NME(RMSE_list, writepath, ob):
    fileObject = open(writepath+'/{}_list.txt'.format(ob), 'w')  
    for i in range(len(RMSE_list)): 
        fileObject.write(get_variable_name(RMSE_list[i])+'\r') 
        fileObject.write(str(RMSE_list[i]) ) 
        fileObject.write('\n') 
    fileObject.close()  

write_NME(RMSE_list, path, 'RMSE')

if writeData:
    np.save(writepath+'/{}_reconstructed'.format(method), np.array(snapshots2).T)
    np.save(writepath+'/{}_spatial_RMSE'.format(method), spatial_RMSE)
    np.save(writepath+'/{}_temporal_RMSE'.format(method), temporal_RMSE)
 
if method != 'DMD': 
    try:
        ut.plot_loss(path, valid, method, loss_type='training')
    except:
        ut.plot_loss2(path, method, loss_type='training')
