import sys
sys.path.append('../')
sys.path.append('../../utils')
import numpy as np
import matplotlib.pyplot as plt
import os
import time as TT
import ML_utils as ut
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
import scipy

case = 'cylinder'
method = 'Koopman'  
ob = 'u'
load = False 

# Hyperparams
offset = 5 
activ = 'tanh'
patience = 500 #500 for 0.013
valid = True 
seed = 1
rank = 4 #Latent dim
epoch = 1000
batch_size = 16
    
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
mask_size = [16, 4]  

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# =============================================================================
# Load data
# =============================================================================
if case == 'cylinder':
    numberx, numbery = 192, 96
    S_full = np.load('../Cylinder_interpolate192x96y_order_C.npy')
    import loadSnapshotsObjectsCylinder as ls
    x = np.linspace(-1, 8, numberx)
    y = np.linspace(-2, 2, numbery)
    dltT = 0.2
    S_col = 120
    S_col_full = 151
    pltT = int(5) 
    startT = 0
    colorslist = ['aqua','blue','w','red','orange']
    from matplotlib import colors
    cmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist)
elif case == 'channel':
    import loadSnapshotsObjectsChannel as ls
    numberx, numbery = 256, 64  
    dltT = 0.13
    S_col = 120
    S_col_full = 151
    pltT = int(10)      
    startT = 4 
    x = np.linspace(0, 8*np.pi, numberx)
    y = np.linspace(-1, 1, numbery)
    fileName = 'channel_u_{}X{}p_4to24s_dt{}.npy'.format(numberx, numbery, dltT)
    data = np.load('../' +fileName)[:S_col_full,:,:,0]    
    S_full = np.reshape(data, (data.shape[0], -1), order='C')
    S_full = np.transpose(S_full)
    cmaps = plt.cm.jet
    
trainT = startT + S_col * dltT
endT = startT + S_col_full * dltT

train_size = int((trainT - startT)/dltT)
full_size = int((trainT -endT)/dltT)
T_col = S_col_full - S_col

S_train = S_full[:, :S_col]
S_test = S_full[:, S_col:]

maxz = float('%.2f' % np.max(S_full))
minz = float('%.2f' % np.min(S_full))
levels = np.linspace(minz, maxz, 11)
for i in range(levels.size):
    levels[i] = float('%.2f' % levels[i])
 
xi,yi = np.meshgrid(x,y)
x,y = np.meshgrid(x,y)
S_row = xi.size
dltx = (x[0,-1]-x[0,0])/numberx
dlty = (y[-1,0]-y[0,0])/numbery

S_mean = S_train.mean(axis=1)

time = []
for i in np.arange(startT, endT, dltT):
    i = float('%.3f' % i)
    time.append(i)
else:
    time_n = time

snapshots = list(S_full.T)
# =============================================================================
# Constructor
# =============================================================================
projName = '{}_{}_'.format(case, ob)

if 'Koopman' in method:
    projName += '_{}dt'.format(offset)
 
if Noise or Mask:   
    load = True 
    if Noise:
        if noise_sigma != 0:
            projName_old = projName
            projName += '_Noise{}'.format(noise_sigma)
            S_full_n = S_full * np.random.normal(0, noise_sigma, S_full.shape)
            S_full_n += S_full    
            S_train_n = S_full_n[:,:S_col]
        else:
            raise ValueError('Sigma cannot equal zero when switch_noise is ON')
    if Mask:
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
    projName_old = projName
    S_full_n = S_full
    S_train_n = S_train
    
snapshots_n = list(S_full_n.T)

path_old = '../ML_' + projName_old +'_'+ activ + '_output/'          
path = '../ML_' + projName +'_'+ activ + '_output/'   
if method == 'DMD':
    path_old = '../' + projName_old + '_rank' + str(rank)  + '_output/' 
    path = '../' + projName + '_rank' + str(rank)  + '_output/'  

if not os.path.exists(path):
    os.makedirs(path) 
writepath = path + 'output_data'
if not os.path.exists(writepath):
    os.makedirs(writepath) 
    
plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'stix'
font={'family': 'Times New Roman', 'math_fontfamily':'stix'}

# =============================================================================
# Koopman
# =============================================================================   
if 'Koopman' in method:
    method = 'Koopman'
    if 'Torch' in method:
        import torch
        import ML_Koopman_Torch as ML_KPM 
    else:
        import ML_Koopman_Time as ML_KPM 
        operator = 'Simple'
        from CL_Simple import Linear

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
            KPO = koopman.get_weights()[0].transpose()
        else:
            encoder = load_model(path+'encoder.h5')
            decoder = load_model(path+'decoder.h5')
            autoencoder = load_model(path+'autoencoder.h5',custom_objects={"Linear": Linear})
            koopman = load_model(path+'koopman.h5',custom_objects={"Linear": Linear})
            KPO = np.load(path_old+'/KPO.npy')
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

    if Noise or Mask:
        encoded_true = ML_KPM.encoder_pred(S_full_n, numberx, numbery, encoder) 
        encoded_ori = ML_KPM.encoder_pred(S_train_n, numberx, numbery, encoder)
        encoded_clean = ML_KPM.encoder_pred(S_full, numberx, numbery, encoder) 
        np.savetxt(path + '/encoded_clean.csv', encoded_clean, delimiter=',')
        np.savetxt(path + '/encoded_noise.csv', encoded_true, delimiter=',')
    else:
        encoded_true = ML_KPM.encoder_pred(S_full, numberx, numbery, encoder) 
        encoded_ori = ML_KPM.encoder_pred(S_train, numberx, numbery, encoder)

    encoded = encoded_ori
    encoded_pred = np.zeros((encoded_ori.shape[0],S_col_full+offset))
    Gx_pred = encoded_pred

    eig_func = eigvec_l.T @ encoded_ori
    Gx_rec = eigvec @ eig_func

    offset = 1
    for m, i in enumerate(range(S_col, S_col_full, offset), 0): 
        op_num = m + 1
        encoded = ML_KPM.koopman_pred(encoded, numberx, numbery, koopman) 
        encoded_pred[:,i:i+offset] = encoded[:,-offset:] 

    encoded_pred = encoded_pred[:,:S_col_full]
    encoded_pred[:,:S_col] = encoded_ori

    Gx_pred = np.zeros((encoded_ori.shape[0],S_col_full))

    Gx_pred[:,op_num:op_num+S_col] = (eigvec @ np.diag(eigs) ** op_num @ eig_func).real
    Gx_pred[:,:S_col] = encoded_ori
    Gx_pred = Gx_pred[:,:S_col_full]
    
    if type(eigvec)!=type(None):
        mode_real = ML_KPM.decoder_pred(eigvec.real, numberx, numbery, decoder)
        mode_imag = ML_KPM.decoder_pred(eigvec.imag, numberx, numbery, decoder)
        func_real = ML_KPM.decoder_pred(eig_func.real, numberx, numbery, decoder)
        func_imag = ML_KPM.decoder_pred(eig_func.imag, numberx, numbery, decoder)
        np.save(path+'/mode_real', mode_real)
        np.save(path+'/mode_imag', mode_imag)

    S_pred = ML_KPM.decoder_pred(Gx_pred, numberx, numbery, decoder) 
    onlineEnd = TT.perf_counter()

    if normalize:
        S_pred = scaler.inverse_transform(S_pred)
        S_full = scaler.inverse_transform(S_full)

    snapshots2 = list(np.transpose(S_pred))
    
    if not load:
        np.save(path+'/encoded_true', encoded_true)
        np.save(path+'/encoded', encoded)
        np.save(path+'/Gx_pred', Gx_pred)
        np.save(path+'/KPO', KPO)
        np.save(path+'/idx', idx)
        np.save(path+'/eigs', eigs)
        np.save(path+'/eigvec', eigvec)
        np.save(path+'/eigvec_l', eigvec_l)
        np.save(path+'/eig_func', eig_func)
        np.save(path+'/func_real', func_real)
        np.save(path+'/func_imag', func_imag)
        np.save(path+'/mode_real', mode_real)
        np.save(path+'/mode_imag', mode_imag)
        np.savetxt(path + '/encoded.csv', encoded, delimiter=',')
        np.savetxt(path + '/KPO.csv', KPO, delimiter=',')
        np.savetxt(path + '/eigs.csv', eigs, delimiter=',')
        np.savetxt(path + '/eigvec.csv', eigvec, delimiter=',')
        np.savetxt(path + '/idx.csv', idx, delimiter=',')
        ut.txt_write(path+'/seed.txt', seed)
    else:
        np.save(path+'/KPO', KPO)
        np.savetxt(path + '/KPO.csv', KPO, delimiter=',')
        np.save(path + '/Gx_pred', Gx_pred)
        np.savetxt(path + '/Gx_pred.csv', KPO, delimiter=',')


    # plot
    print('the Koopman operator is {}'.format(KPO))
    print('the eigenvalue is {}'.format(eigs))
    ls.plotModes_fast(xi, yi, numberx, numbery, encoded.shape[0], path, mode_real, cmaps, ob='Mode real')
    ls.plotModes_fast(xi, yi, numberx, numbery, encoded.shape[0], path, mode_imag, cmaps, ob='Mode imag')

    if not load:
        try:
            eig_func2d = np.load(path+'/eig_func2d.npy')
            Gx_2d = np.load(path+'/Gx_2d.npy')
        except:
            Gx_2d = []
            x_final = []
            if Noise or Mask:
                inp = np.transpose(S_train_n)
            else:
                inp = np.transpose(S_train)
            inp = np.reshape(inp, (-1, numberx*numbery, 1))
            for i in range(numberx*numbery):
                x_in = inp[:,i,:]
                x_out = np.repeat(x_in[:,np.newaxis,:], numberx*numbery, axis=1)
                x_out = x_out.reshape(-1, numberx, numbery, 1)
                Gx = encoder(x_out).numpy()
                Gx_2d.append(Gx)
            Gx_2d = np.stack(Gx_2d,axis=1).transpose(2,1,0)
            eig_func2d, ___ = ut.Rec_func(Gx_2d, eigvec_l, eigvec)
            np.save(path+'/eig_func2d', eig_func2d)
            np.save(path+'/Gx_2d', Gx_2d)

        eig_func2d_mean = np.mean(eig_func2d, -1).transpose()
        eig_func2d_mean_mag = np.sqrt((eig_func2d_mean.real)**2, (eig_func2d_mean.imag)**2)
        eig_func2d_mean_phs = np.arctan(eig_func2d_mean.imag/eig_func2d_mean.real)

        ls.plotModes_fast(xi, yi, numberx, numbery, encoded.shape[0], path, eig_func2d_mean.real, cmaps, ob='Eigenfunction')
        ls.plotModes_fast(xi, yi, numberx, numbery, encoded.shape[0], path, eig_func2d_mean.imag, cmaps, ob='Eigenfunction imag')
        ls.plotModes_fast(xi, yi, numberx, numbery, encoded.shape[0], path, eig_func2d_mean_mag, cmaps, ob='Eigenfunction magnitude')
        ls.plotModes_fast(xi, yi, numberx, numbery, encoded.shape[0], path, eig_func2d_mean_phs, cmaps, ob='Eigenfunction phase')

    ls.plotModes_fast(xi, yi, numberx, numbery, 1, path, func_real, cmaps, ob='SKCNN Eigenfunction')
    ls.plotModes_fast(xi, yi, numberx, numbery, 1, path, func_imag, cmaps, ob='SKCNN Eigenfunction imag')

    ut.plotEigs(eigs,title='Eigenvalues',filename=path+'/Eigenvalues.jpg',dpi=300,show_unit_circle=True)

    plot_r = min(encoded_ori.shape[0], 8)
    ut.plot_lstm(encoded_true[idx,:], train_size, time, plot_r, path, name='Observable true', label=True)
    ut.plot_lstm(encoded_pred[idx,:], train_size, time, plot_r, path, name='Observable pred', label=True)
    ut.plot_lstm(Gx_pred[idx,:], train_size, time, plot_r, path, name='Koopman Observable', label=True)

    ut.plot_trace(encoded_pred[idx,:], S_col, path+'/Trace.jpg')
# =============================================================================
# DMD
# =============================================================================
elif method == 'DMD':
    from pydmd import DMD
    offlineEnd = TT.time()
    onlineStart = TT.perf_counter()
    dmd = DMD(svd_rank=rank, exact=True)
    dmd.fit(S_train_n)
    dmd.dmd_time['tend'] = int((endT-startT)/dltT) 
    dmd.dmd_time
    S_pred = dmd.reconstructed_data.real
    snapshots2 = list(np.transpose(S_pred))
    onlineEnd = TT.perf_counter()
    
    __, sigma, __ = ut.compute_svd(S_train[:, :S_col-1], svd_rank=-1)
    POD_modes, __, __ = ut.compute_svd(S_train[:, :S_col], svd_rank=-1)
    cumulative_energy = np.cumsum(sigma**2 / (sigma**2).sum())
    np.savetxt(path + '/eig_cum.csv', cumulative_energy, delimiter=',')
    ut.plotSigma(sigma, rank, 'Normalized singular values', path + 'DMD Normalized singular values.png')

    modes = dmd.modes
    dynamics = dmd.dynamics
    eigs = dmd.eigs
    amplitudes = dmd.amplitudes
    growth = dmd.growth_rate
    frequency = dmd.frequency
    KPO = dmd.operator

    mode_real = modes.real
    mode_imag = modes.imag

    idx = eigs.argsort()[::-1]   
    eigs = eigs[idx]

    plot_r = min(dmd.eigs.shape[0], 4)

    ut.plotEigs(eigs,title='Eigenvalues',filename=path+'/Eigenvalues.jpg',dpi=300,show_unit_circle=True)

    ls.plotModes_fast(x, y, numberx, numbery, plot_r, path, dmd.modes.real, cmaps, ob='DMD mode')
    ls.plotModes_fast(x, y, numberx, numbery, plot_r, path, dmd.modes.imag, cmaps, ob='DMD mode imag')

    plot_time = 150
    ut.plotDynamics(dmd.dynamics[:plot_r:, :plot_time].real, time[:plot_time], 'DMD dynamics (real part)', path)
    ut.plotDynamics(dmd.dynamics[:plot_r, :plot_time].imag, time[:plot_time], 'DMD dynamics (imaginary part)', path)

    np.save(path+'/KPO', KPO)
    np.save(path+'/idx', idx)
    np.save(path+'/eigs', eigs)
    np.save(path+'/mode_real', mode_real)
    np.save(path+'/mode_imag', mode_imag)
    np.savetxt(path + '/eigs.csv', eigs, delimiter=',')
    np.savetxt(path + '/idx.csv', idx, delimiter=',')
# =============================================================================
# Time
# =============================================================================
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
if plotPredict:       
    #-------------------------contour-----------------------------# 
    if contour:
        if Noise or Mask:
            ls.plotContour_fast(xi, yi, numberx, numbery, path, time, snapshots_n, cmaps, pltT, 
                            levels, minz, maxz, title='Corrupted', ob=ob)
        if Mask:
            ls.plotOne_fast(xi, yi, numberx, numbery, path, mask, cmaps, 'Mask')
        if plotOrigin:
            ls.plotContour_fast(xi, yi, numberx, numbery, path, time, snapshots, cmaps, pltT, 
                            levels, minz, maxz, title='Original', ob=ob)
        ls.plotContour_fast(xi, yi, numberx, numbery, path, time, snapshots2, cmaps, pltT, 
                        levels, minz, maxz, title='{} reconstructed'.format(method), ob=ob)

#------------------------------plot--------------------------------#      
if plotErr:    
    err, __ = ls.err_list(x.size, snapshots, snapshots2)
    maxz, minz, levels = ls.get_levels(err, 11)
    #-------------------------contour-----------------------------# 
    if contour:
        ls.plotContourErr_fast(x, y, numberx, numbery, path, time, err, plt.cm.jet, pltT, 
                          levels, minz, maxz, ob='vorticity error')
        
#-------------------------video-----------------------------# 
if contour:
    if video:
        if plotOrigin:
            ut.video(startT, endT, pltT, dltT, path, time, ob='Original contour_')   
        if plotPredict:
            ut.video(startT, endT, pltT, dltT, path, time, ob='{} reconstructed contour_'.format(method))   
        if plotErr:
            ut.video(startT, endT, pltT, dltT, path, time, ob='vorticity error contour_')  
        if Noise or Mask:
            ut.video(startT, endT, pltT, dltT, path, time, ob='Corrupted contour_')  
        
# =============================================================================
# Error estimation
# =============================================================================
minT = int(min(len(snapshots), len(snapshots2)))    
snapshots = snapshots[:minT]
snapshots2 = snapshots2[:minT]

spatial_RMSE = ut.plot_spatial_NME(snapshots, snapshots2, time[0],
                                   trainT, time[minT-1], dltT, path, 
                                   method='{}'.format(method), ob='RMSE')

arr = np.zeros((x.size))
temporal_RMSE = ut.get_tmporal_NME(snapshots[:minT], snapshots2[:minT], arr, 'RMSE') 

ls.plot_temporal_rmse_fast(x, y, numberx, numbery, plt.cm.jet, 
                           temporal_RMSE, path, method='{}'.format(method), ob='RMSE')

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
 
if not 'DMD' in method: 
    if not Noise and not Mask:
        try:
            ut.plot_loss(path, valid, method, loss_type='training')
        except:
            ut.plot_loss2(path, method, loss_type='training')

print('Task completes')