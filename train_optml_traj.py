# first neural network with keras tutorial
import numpy as np
import matplotlib.pyplot as plt
from math import pi

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
MODEL_NUMBER = 1
seeds = [ 5460 , 12 , 445 , 2500, 1111 ]
np.random.seed( seeds[MODEL_NUMBER-1] )

train_path = "ML_data/train_data(" + str(seeds[MODEL_NUMBER-1]) + ").csv"
label_path = "ML_data/label_data(" + str(seeds[MODEL_NUMBER-1]) + ").csv"
model_path = "ML_data/ISOSC(" + str(seeds[MODEL_NUMBER-1]) + ")"


dt = 0.012 # [sec]
grvty = 9810 # [mm/sec^2]
SPACE_UP_LIMIT = 650 # [mm]
SPACE_DOWN_LIMIT = 0 # [mm]
SPACE_RIGHT_LIMIT = 650 # [mm]
SPACE_LEFT_LIMIT = 0 # [mm]
BOX_HALF_LENGTH = 125 # [mm]
v_avg_min = 100 # [mm/sec]
v_avg_max = 150 # [mm/sec]
theta_B_0_range = 30 * pi/180 # [rad]
omega_B_0_range = theta_B_0_range / dt


## DNN training parameter
valid_rate = 0.1
epoch_num = 100
batch_size = 32

######################
### Function Start ###
######################
##
def load_Y( label_path, train_num ):
    
    raw_Y = np.loadtxt( label_path, delimiter=',' )
    label_largest_len = raw_Y.shape[1]
    omega_B_0_max = np.max( raw_Y[:,0] )
#    print( np.argmax( raw_Y[:,0] ) )
    omega_B_0_min = np.min( raw_Y[:,0] )
#    print( np.argmin( raw_Y[:,0] ) )
    Y = []
    label_len_set = []
    
    for i in range( train_num ):
        
        j = 0        
        while(1):                        
            
            if( raw_Y[i,-1-j] != np.inf ):               
                label_len_set.append( label_largest_len - j )
                Y.append( np.copy( raw_Y[ i, :( label_largest_len - j ) ] ) )
                break
            
            j = j + 1
    
    return omega_B_0_max, omega_B_0_min, np.array( label_len_set ), Y


## normalized training dataset to 0 to 1
def normalize_X( raw_X ):
    
    # raw_x : via_x*4 [mm], via_y*4 [mm], v_avg*3 [mm/sec], theta_B_0 [rad], x_mB_0 [mm]
    nrmlz_X = np.copy( raw_X )
    nrmlz_X[:,:4] = ( raw_X[:,:4] - SPACE_LEFT_LIMIT ) / ( SPACE_RIGHT_LIMIT - SPACE_LEFT_LIMIT ) # via_x*4
    nrmlz_X[:,4:8] = ( raw_X[:,4:8] - SPACE_DOWN_LIMIT ) / ( SPACE_UP_LIMIT - SPACE_DOWN_LIMIT ) # via_y*4
    nrmlz_X[:,8:11] = ( raw_X[:,8:11] - v_avg_min ) / ( v_avg_max - v_avg_min ) # v_avg*3
    nrmlz_X[:,11] = ( raw_X[:,11] / theta_B_0_range + 1 ) / 2
    nrmlz_X[:,12] = ( raw_X[:,12] / BOX_HALF_LENGTH + 1 ) / 2
    
    return nrmlz_X


## normalized labelled dataset to 0 to 1
def normalize_Y( raw_Y ):
    
    # Y : omega_B_0, (time_stamp,theta_B_feat)*n
    nrmlz_Y = []
    for i in range( len(raw_Y) ):
        
        arr_temp = np.copy( raw_Y[i] )
        arr_temp[0] = ( arr_temp[0] + omega_B_0_range ) / ( 2 * omega_B_0_range ) # omega_B_0
        
        for j in range( 1, len(raw_Y[i]), 2 ):
            
            arr_temp[j] = arr_temp[j] / arr_temp[-2] # nmormalize time_stamp
            arr_temp[j+1] = ( arr_temp[j+1] + theta_B_0_range ) / ( 2 * theta_B_0_range ) # theta_B_feat
            
        nrmlz_Y.append( arr_temp )
    
    return nrmlz_Y


## make nrmlz_Y to 2D array
def creat_label_2Darr( raw_Y, label_len ):

    nrmlz_Y = []
    for i in range( len(raw_Y) ):
        
        arr_temp = np.copy( raw_Y[i] )
        
        while(len(arr_temp) > label_len ):
            arr_temp = np.delete( arr_temp, [ (len(arr_temp)-4), (len(arr_temp)-3) ] )
        
        while(len(arr_temp) < label_len ):
            arr_temp = np.append( arr_temp, arr_temp[-2:]  )
        
        # transform time stamp to incremental form
        for j in range( label_len-2, 2, -2 ):
            arr_temp[j] = arr_temp[j] - arr_temp[j-2]
        
        nrmlz_Y.append( arr_temp )
    
    return np.vstack( nrmlz_Y )


## Randomly split validation data from training data bcz "split" is before "shuffle" in "model.fit"
def cut_valid( data_X, data_Y, valid_num ):
    
    X = np.copy( data_X )
    Y = np.copy( data_Y )
    
    # Randomly rearrange the train data
    train_num = len( X )
    
    for i in range( train_num ):
        
        dice = np.random.randint( 0, train_num - 1 )
        X[ [ i, dice ] ] = X[ [ dice, i ] ]
        Y[ [ i, dice ] ] = Y[ [ dice, i ] ]

    X_valid = X[ ( train_num - valid_num ):, : ]
    X_train = X[ :( train_num - valid_num ), : ]
    Y_valid = Y[ ( train_num - valid_num ):, : ] 
    Y_train = Y[ :( train_num - valid_num ), : ]
    
    print('validation data built')
    return X_train , Y_train , X_valid , Y_valid

## construct DNN model
def get_DNN_Model( train_dim, label_dim ):
    
    model = Sequential()
    model.add( Dense( 1024,
                      activation='relu',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4),
                      input_dim = train_dim
                    ))
    model.add( Dropout(0.1) )
    model.add( BatchNormalization() )
    
    model.add( Dense( 1024,
                      activation='relu',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4)
                    ))
    model.add( Dropout(0.1) )
    model.add( BatchNormalization() )
    
    model.add( Dense( label_dim,
                      activation='sigmoid',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4)
                    ))
                    
#    adam = Adam( lr=0.001, decay=1e-6, clipvalue=0.5 )
    model.compile( loss='mse',
                   optimizer='adam',
                 )
    
    print('DNN model built')
    return model


## plot the DNN model training process
def PlotTrainingProcess( hist ):
    
    plt.plot( hist.history['loss'] )
    plt.plot( hist.history['val_loss'] )
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


## train the DNN model
def train_DNN_model( model, X_train, Y_train, X_valid, Y_valid, to_load_model ):
        
    if( to_load_model==False ):
        
        model.summary()
    
        ## some callbacks
        #    earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint( filepath=( model_path + "_w_best.hdf5" ),
                                      verbose=0,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      monitor='val_loss',
                                      mode='min'
                                    )
        hist = model.fit( X_train , Y_train,
                          validation_data=( X_valid, Y_valid ), 
                          epochs=epoch_num, batch_size=batch_size,
                          verbose=1,
                          shuffle=True,
        #                      callbacks=[ earlystopping, checkpoint ],
                          callbacks=[ checkpoint ]
                        )

        PlotTrainingProcess( hist )
        model.load_weights( model_path + "_w_best.hdf5" )
        model.save( model_path + '.h5' )
        DNN_model = model
        
    else:
        DNN_model = load_model( model_path + '.h5' )
        
    loss = DNN_model.evaluate( X_valid, Y_valid )
    print("\nValidation loss: " , loss )
    
    print('\nDNN model training and test data label prediction finished.')
    return DNN_model
    
########################
###       Main       ###
######################## 
# if __name__ == "__main__":
    
# load the dataset    
raw_X = np.loadtxt( train_path, delimiter=',' ) 
train_num = raw_X.shape[0]
train_dim = raw_X.shape[1]
omega_B_0_max, omega_B_0_min, label_len_set, raw_Y = load_Y( label_path, train_num ) # omega_B_0: [rad/sec]

# normalized dataset
nrmlz_X = normalize_X( raw_X )
nrmlz_Y = normalize_Y( raw_Y )

# decide dim of Y and make nrmlz_Y to 2D array
label_dim = int( np.mean( label_len_set ) )

if( label_dim % 2 == 0 ):
    label_dim = label_dim + 1

nrmlz_Y_ = creat_label_2Darr( nrmlz_Y, label_dim )
   
# Construct and train the DNN model
X_train , Y_train , X_valid , Y_valid = cut_valid( nrmlz_X, nrmlz_Y_, int( valid_rate * train_num ) ) # cut out validation data
DNN_model = get_DNN_Model( train_dim, label_dim )
DNN_model = train_DNN_model( DNN_model, X_train, Y_train, X_valid, Y_valid, to_load_model=False )
    
# validate based on training data


