import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Concatenate, Add, AveragePooling2D
from keras.layers import Flatten, Reshape, UpSampling2D, ZeroPadding2D, BatchNormalization, Lambda
from keras.models import Sequential, Model, load_model
from keras.regularizers import l1
from keras.losses import MeanSquaredError, mean_squared_error
from keras.utils.vis_utils import plot_model
from keras.activations import relu, tanh
from keras.optimizers import Adam, Nadam, Adamax
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger, RemoteMonitor
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import initializers

def ML_train(inp, numberx, numbery, path, offset = 1, activ = 'relu',
             valid = None, opt = 'adam', epoch = 1000, batch_size=32,
             patience=200, operator='', t = 4, pooling = 'max'): 

    lc = np.logspace(0,5,6,base=1/2)*32
    lx = np.logspace(0,5,6,base=1/2)*numberx
    ly = np.logspace(0,5,6,base=1/2)*numbery
    shape = []
    for i in range(6):
        shape.append(int(lx[i]*ly[i]*lc[i]))

    lc = [int(_) for _ in lc]
    lx = [int(_) for _ in lx]
    ly = [int(_) for _ in ly]
    shape = [int(_) for _ in shape]
    
    num_t = offset + 1
    
    x_train, x_test = train_test_split(inp, test_size=0.2, shuffle=False, random_state=1)
    x_train = tf.keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        None,
        sequence_length=num_t,
        batch_size=batch_size,
    )  

    X_train = np.concatenate(list(x_train.as_numpy_iterator()))
    x_0 = X_train[:,0,:,:]
    x_1 = X_train[:,1:,:,:]

    input_shape = x_0.shape[1:]

    if type(valid) == type(None) or valid == False:   
        validation_data = None
    else:
        try:
            x_test = tf.keras.preprocessing.timeseries_dataset_from_array(
            x_test,
            None,
            sequence_length=num_t,
            batch_size=batch_size
            )
            X_test =  np.concatenate(list(x_test.as_numpy_iterator()))
            x_0_test = X_test[:,0,:,:]
            x_1_test = X_test[:,1:,:,:]
            validation_data = (x_0_test, [x_0_test, x_1_test])
        except:
            validation_data = None

    print('x_train.shape is {} \n'.format(x_0.shape))
    print('the operator is {} \n'.format(operator))
    print('the time offset is {} \n'.format(offset))
    
    fs = 3
   
    input_img = Input(shape=input_shape, batch_size=None, name='Input')
    conv1 = (Conv2D(lc[0], (fs,fs), activation=activ, padding='same', name='Conv1'))(input_img)
    pool1 = MaxPooling2D((2,2), padding='same', name='Pooling1')(conv1) #128, 32
    conv2 = (Conv2D(lc[1], (fs,fs), activation=activ, padding='same', name='Conv2'))(pool1)
    pool2 = MaxPooling2D((2,2), padding='same', name='Pooling2')(conv2) #64, 16
    conv3 = (Conv2D(lc[2], (fs,fs), activation=activ, padding='same', name='Conv3'))(pool2)
    pool3 = MaxPooling2D((2,2), padding='same', name='Pooling3')(conv3) #32, 8
    conv4 = (Conv2D(lc[3], (fs,fs), activation=activ, padding='same', name='Conv4'))(pool3)
    pool4 = MaxPooling2D((2,2), padding='same', name='Pooling4')(conv4) #16, 4
    conv5 = (Conv2D(lc[4], (fs,fs), activation=activ, padding='same', name='Conv5'))(pool4)
    pool5 = MaxPooling2D((2,2), padding='same', name='Pooling5')(conv5) #8, 2
    conv6 = (Conv2D(lc[5], (fs,fs), activation=activ, padding='same', name='Conv6'))(pool5)

    fc1 = Reshape([shape[0]], name='Flatten1')(conv1)
    fc2 = Reshape([shape[1]], name='Flatten2')(conv2)
    fc3 = Reshape([shape[2]], name='Flatten3')(conv3)
    fc4 = Reshape([shape[3]], name='Flatten4')(conv4)
    fc5 = Reshape([shape[4]], name='Flatten5')(conv5)
    fc6 = Reshape([shape[5]], name='Flatten6')(conv6)

    fc1 = Dense(t, activation='tanh', name='FCe1')(fc1)
    fc2 = Dense(t, activation='tanh', name='FCe2')(fc2)
    fc3 = Dense(t, activation='tanh', name='FCe3')(fc3)
    fc4 = Dense(t, activation='tanh', name='FCe4')(fc4)
    fc5 = Dense(t, activation='tanh', name='FCe5')(fc5)
    fc6 = Dense(t, activation='tanh', name='FCe6')(fc6)

    # mode1
    encoded = Add(name='Add')([fc1,fc2,fc3,fc4,fc5,fc6])
    encoded = Lambda(lambda x: x/6, name='Lambda')(encoded)

    # mode2
    # encoded = Concatenate(axis=-1)([fc1,fc2,fc3,fc4,fc5,fc6])
    # encoded = Dense(t, activation='tanh')(encoded)

    input_Gx = Input(shape=t)
    if operator == 'Linear':
        from CL_Linear import Linear
        KPM = Linear()
    elif operator == 'Simple':
        from CL_Simple import Linear
        KPM = Linear()
    elif operator == 'Skew':
        from CL_Skew import Linear
        KPM = Linear()
    elif operator == 'Eigen':
        from CL_Eigen import Eigen
        KPM = Eigen(offset)
    elif operator == '':
        KPM = Dense(t, activation='linear', use_bias=False, name='Koopman')
    else:
        raise ValueError('choose one in Linear/Simple/KPM/')

    encoded_fw = KPM(input_Gx)

    input_KGx = Input(shape=t)
    # fc1x = Dense(t, activation=activ, name='FCd1')(input_KGx)
    # fc2x = Dense(t, activation=activ, name='FCd2')(input_KGx)
    # fc3x = Dense(t, activation=activ, name='FCd3')(input_KGx)
    # fc4x = Dense(t, activation=activ, name='FCd4')(input_KGx)
    # fc5x = Dense(t, activation=activ, name='FCd5')(input_KGx)
    # fc6x = Dense(t, activation=activ, name='FCd6')(input_KGx)

    fc1x = Dense(shape[0], activation=activ, name='Upd1')(input_KGx)
    fc2x = Dense(shape[1], activation=activ, name='Upd2')(input_KGx)
    fc3x = Dense(shape[2], activation=activ, name='Upd3')(input_KGx)
    fc4x = Dense(shape[3], activation=activ, name='Upd4')(input_KGx)
    fc5x = Dense(shape[4], activation=activ, name='Upd5')(input_KGx)
    fc6x = Dense(shape[5], activation=activ, name='Upd6')(input_KGx)

    fc1x = Reshape([lx[0],ly[0],lc[0]], name='Reshape1')(fc1x)
    fc2x = Reshape([lx[1],ly[1],lc[1]], name='Reshape2')(fc2x)
    fc3x = Reshape([lx[2],ly[2],lc[2]], name='Reshape3')(fc3x)
    fc4x = Reshape([lx[3],ly[3],lc[3]], name='Reshape4')(fc4x)
    fc5x = Reshape([lx[4],ly[4],lc[4]], name='Reshape5')(fc5x)
    fc6x = Reshape([lx[5],ly[5],lc[5]], name='Reshape6')(fc6x)

    conv7 = (Conv2D(lc[5], (fs,fs), activation=activ, padding='same', name='ConvT1'))(fc6x)
    up1 = UpSampling2D((2, 2), name='poolingT1')(conv7) #16, 4
    up1 = Concatenate(axis=-1, name='Concat1')([up1,fc5x])
    conv8 = (Conv2D(lc[4], (fs,fs), activation=activ, padding='same', name='ConvT2'))(up1)
    up2 = UpSampling2D((2, 2), name='poolingT2')(conv8) #32, 8
    up2 = Concatenate(axis=-1, name='Concat2')([up2,fc4x])
    conv9 = (Conv2D(lc[3], (fs,fs), activation=activ, padding='same', name='ConvT3'))(up2)
    up3 = UpSampling2D((2, 2), name='poolingT3')(conv9) #64, 16
    up3 = Concatenate(axis=-1, name='Concat3')([up3,fc3x])
    conv10 = (Conv2D(lc[2], (fs,fs), activation=activ, padding='same', name='ConvT4'))(up3)
    up4 = UpSampling2D((2, 2), name='poolingT4')(conv10) #128, 32
    up4 = Concatenate(axis=-1, name='Concat4')([up4,fc2x])
    conv11 = (Conv2D(lc[1], (fs,fs), activation=activ, padding='same', name='ConvT5'))(up4)
    up5 = UpSampling2D((2, 2), name='poolingT5')(conv11) #256, 64
    up5 = Concatenate(axis=-1, name='Concat5')([up5,fc1x])
    conv12 = (Conv2D(lc[0], (fs,fs), activation=activ, padding='same', name='ConvT6'))(up5)
    o = (Conv2D(input_shape[-1], (fs,fs), padding='same', name='Output'))(conv12)

    encoder = Model(inputs=[input_img], outputs=[encoded], name='encoder')
    koopman = Model(inputs=[input_Gx], outputs=[encoded_fw], name='koopman')
    decoder = Model(inputs=[input_KGx], outputs=[o], name='decoder')
    
    Gx_list = []
    KGx_list = []
    pred_list = []

    # x0_img = input_img
    Gx_img = encoder(input_img)
    if operator == '' or operator == 'Simple':
        KGx_img = Gx_img
        for i in range(offset):
            print('Koopman cycle in training:',i+1)
            KGx_img = koopman(KGx_img)
            KGx_list.append(KGx_img)
            pred_img = decoder(KGx_img)
            pred_list.append(pred_img)
    else:
        KGx_img = koopman(Gx_img)
        KGx_list.append(KGx_img)
        pred_img = decoder(KGx_img)
        pred_list.append(pred_img)
    rec_img = decoder(Gx_img)
    
    for pred_img in pred_list:
        Gx_pred = encoder(pred_img)
        Gx_list.append(Gx_pred)
    
    KGx_idx = [tf.expand_dims(KGx, axis=1, name='KGx') for KGx in KGx_list]
    KGx_c = tf.concat(KGx_idx, axis=1, name='l_KGx')

    Gx_idx = [tf.expand_dims(Gx, axis=1, name='KGx') for Gx in Gx_list]
    Gx_c = tf.concat(Gx_idx, axis=1, name='l_Gx')

    pred_idx = [tf.expand_dims(pred, axis=1, name='KGx') for pred in pred_list]
    pred_c = tf.concat(pred_idx, axis=1, name='l_pred')
    
    autoencoder = Model(inputs=[input_img], outputs=[rec_img, pred_c], name='autoencoder')
    autoencoder.summary()
    
    def l2(x, y):
        return tf.reduce_mean(tf.square(x - y))
    
    c_lin = 1.
    
    loss_lin = c_lin*l2(Gx_c, KGx_c)
    # loss_rec = c_rec*l2(rec_img, x0_img) # Xt to Xt
    # loss_pred = c_pred*l2(pred_img, x1_img) # Xt to Xt
    
    autoencoder.add_loss(c_lin*loss_lin)
    # autoencoder.add_loss(c_rec*loss_rec)
    # autoencoder.add_loss(c_pred*loss_pred)
    
    autoencoder.add_metric(c_lin*loss_lin, name='l_in')
    # autoencoder.add_metric(loss_rec, name='l_rec')
    # autoencoder.add_metric(loss_pred, name='l_pred')
    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([500,500,500],[1e-3,1e-4,1e-5,1e-6])
    # lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, 0.1*int(epoch))
    # lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, int(0.025*epoch), 0.8)
    # lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-3,decay_steps=int(0.1*epoch),decay_rate=10)
    # autoencoder.compile(optimizer=Adam(lr), loss='mse')
    autoencoder.compile(optimizer=Adam(), loss='mse')
    
    # if offset == 1:
    #     monitor = 'val_tf.identity_2_loss'
    # else:
    #     monitor = 'val_tf.concat_2_loss'
    monitor = 'loss'

    # tf.keras.utils.plot_model(autoencoder, expand_nested=True)
    # plot_model(autoencoder,to_file=path+'model.jpg', show_shapes=True, show_dtype=False, rankdir='TB', dpi=300)
    early_cb = EarlyStopping(monitor=monitor, mode='min', patience=patience, verbose=1)
    model_cb = ModelCheckpoint(path+'AE_weights.h5', monitor='loss',save_best_only=True,verbose=1,save_weights_only=True)
    csv_logger = CSVLogger(path+'training_log.csv', separator=',', append=False)
    history = autoencoder.fit([x_0], [x_0, x_1], epochs=epoch, batch_size=batch_size, verbose=1, validation_data=validation_data,
                    shuffle=True, callbacks=[csv_logger, early_cb, model_cb])

    autoencoder.load_weights(path+'AE_weights.h5')
    
    autoencoder.save(path+'autoencoder.h5', save_format='h5')
    encoder.save(path+'encoder.h5', save_format='h5')
    decoder.save(path+'decoder.h5', save_format='h5')
    koopman.save(path+'koopman.h5', save_format='h5')

    KPO = KPM.get_weights()[0]
    KPO = np.transpose(KPO)

    return autoencoder, encoder, decoder, koopman, KPO
 
    
def ML_pred(ori, numberx, numbery, model):
    pred = model.predict(ori)
    return pred

def koopman_pred(ori, numberx, numbery, model):
    ori = np.transpose(ori)
    pred = model.predict(ori)
    pred = np.transpose(pred)
    return pred

def encoder_pred(ori, numberx, numbery, model):
    pred = model.predict(ori)
    pred = np.transpose(pred)
    return pred

def decoder_pred(ori, numberx, numbery, model):
    ori = np.transpose(ori)
    pred = model.predict(ori)
    return pred
