'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Sep 19, 2018
Restructured May 10, 2019 - DNN training
Restructured Feb 3, 2020 - code revision
'''

import matplotlib
# matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
import os
import os.path
import datetime
import pickle
import skimage

# from https://github.com/BMClab/BMC/blob/master/functions/detect_peaks.py
from detect_peaks import detect_peaks

# additional requirement: SPTK 3.8 or above in PATH
import vocoder_LSP_sptk


from subprocess import run
import scipy

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))

import matplotlib.animation as animation


# read_ult reads in *.ult file from AAA
def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data


# read_psync_and_correct_ult reads *_sync.wav and finds the rising edge of the pulses
# if there was a '3 pulses bug' during the recording,
# it removes the first three frames from the ultrasound data
def read_psync_and_correct_ult(filename, ult_data):
    (Fs, sync_data_orig) = io_wav.read(filename)
    sync_data = sync_data_orig.copy()

    # clip
    sync_threshold = np.max(sync_data) * 0.6
    for s in range(len(sync_data)):
        if sync_data[s] > sync_threshold:
            sync_data[s] = sync_threshold

    # find peeks
    peakind1 = detect_peaks(sync_data, mph=0.9*sync_threshold, mpd=10, threshold=0, edge='rising')
    
    # this is a know bug: there are three pulses, after which there is a 2-300 ms silence, 
    # and the pulses continue again
    if (np.abs( (peakind1[3] - peakind1[2]) - (peakind1[2] - peakind1[1]) ) / Fs) > 0.2:
        bug_log = 'first 3 pulses omitted from sync and ultrasound data: ' + \
            str(peakind1[0] / Fs) + 's, ' + str(peakind1[1] / Fs) + 's, ' + str(peakind1[2] / Fs) + 's'
        print(bug_log)
        
        peakind1 = peakind1[3:]
        ult_data = ult_data[3:]
    
    for i in range(1, len(peakind1) - 2):
        # if there is a significant difference between peak distances, raise error
        if np.abs( (peakind1[i + 2] - peakind1[i + 1]) - (peakind1[i + 1] - peakind1[i]) ) > 1:
            bug_log = 'pulse locations: ' + str(peakind1[i]) + ', ' + str(peakind1[i + 1]) + ', ' +  str(peakind1[i + 2])
            print(bug_log)
            bug_log = 'distances: ' + str(peakind1[i + 1] - peakind1[i]) + ', ' + str(peakind1[i + 2] - peakind1[i + 1])
            print(bug_log)
            
            raise ValueError('pulse sync data contains wrong pulses, check it manually!')
    
    return ([p for p in peakind1], ult_data)




def calculate_FramesPerSec_from_psync(psync_data, Fs):
    psync_data = np.array(psync_data)
    T0s = psync_data[1:] - psync_data[:-1]
    pitch = Fs / T0s
    fps = np.mean(pitch[np.nonzero(pitch)])
    return fps


def get_ult_mgc_lf0(dir_file, filename_no_ext, NumVectors = 64, PixPerVector = 842):
    print('starting ' + dir_file + filename_no_ext)

    
    # NumVectors and PixPerVector could come from *US.txt,
    # but meta files are missing
    
    # read in raw ultrasound data
    ult_data = read_ult(dir_file + filename_no_ext + '.ult', NumVectors, PixPerVector)
    
    try:
        # read pulse sync data (and correct ult_data if necessary)
        (psync_data, ult_data) = read_psync_and_correct_ult(dir_file + filename_no_ext + '_sync.wav', ult_data)
    except ValueError as e:
        raise
    else:
        
        # works only with 22kHz sampled wav
        (Fs, speech_wav_data) = io_wav.read(dir_file + filename_no_ext + '_speech_volnorm.wav')
        assert Fs == 22050

        mgc_lsp_coeff = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.mgclsp', dtype=np.float32).reshape(-1, order + 1)
        lf0 = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.lf0', dtype=np.float32)

        (mgc_lsp_coeff_length, _) = mgc_lsp_coeff.shape
        (lf0_length, ) = lf0.shape
        assert mgc_lsp_coeff_length == lf0_length

        # cut from ultrasound the part where there are mgc/lf0 frames
        ult_data = ult_data[0 : mgc_lsp_coeff_length]

        print('finished ' + dir_file + filename_no_ext + ', altogether ' + str(lf0_length) + ' frames')

        return (ult_data, mgc_lsp_coeff, lf0)


## global parameters

# Parameters of vocoder
Fs = 22050
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 270 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
alpha = 0.42
stage = 3
n_mgc = order + 1

# there are 200 files in the 'PPBA' folder of each speaker
n_files = 200

# properties of UTI data
n_max_ultrasound_frames = n_files * 500
n_lines = 64
n_pixels = 842
n_pixels_reduced = 96


# TODO: modify this according to your data path
dir_base = "/shared/data_SSI2018/"


# train-validata-test on 2 female (048 & 049) and 2 male (102 & 103) speakers
speakers = ['spkr048', 'spkr049', 'spkr102', 'spkr103']

for speaker in speakers:
    
    ult = np.empty((n_max_ultrasound_frames, n_lines, n_pixels_reduced))
    mgc = np.empty((n_max_ultrasound_frames, n_mgc))
    ult_size = 0
    mgc_size = 0
    n_file = 0
    
    # load training and validation data
    # the 'PPBA' directory can be used for training & validation
    dir_train_val = dir_base + speaker + "/PPBA/"
    if os.path.isdir(dir_train_val):
        for file in sorted(os.listdir(dir_train_val)):
            if file.endswith(".ult") and n_file < n_files:
                try:
                    # load ULT and MGC data
                    (ult_data, mgc_lsp_coeff, lf0) = get_ult_mgc_lf0(dir_train_val, file[:-4])
                except ValueError as e:
                    # some of the ULT data is corrupted, leave them out from training
                    print("wrong data, check manually!", e)
                else:
                    
                    ult_len = len(ult_data)
                    if ult_size + ult_len > n_max_ultrasound_frames:
                        raise
                    
                    # resize ULT images
                    for i in range(ult_len):
                        # img0 = scipy.misc.imresize(ult_data[i], (n_lines, n_pixels_reduced), interp='bicubic') / 255
                        img0 = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
                        ult[ult_size + i] = img0
                    ult_size += ult_len
                    
                    mgc_len = len(mgc_lsp_coeff)
                    mgc[mgc_size : mgc_size + mgc_len] = mgc_lsp_coeff
                    mgc_size += mgc_len
                    
                    print('n_frames_all: ', ult_size)
                    n_file += 1

    ult = ult[0 : ult_size]
    mgc = mgc[0 : mgc_size]

    
    # reshape for FC-DNN output
    ult = np.reshape(ult, (-1, n_lines * n_pixels_reduced))
    
    # train-validation split
    ult_training, ult_validation, mgc_lsp_training, mgc_lsp_validation = \
        train_test_split(ult, mgc, test_size=0.1, random_state=19)

    # input: scaling to [0-1], feature by feature
    mgc_scalers = []
    for i in range(n_mgc):
        # mgc_scaler = MinMaxScaler(feature_range=(0, 1))
        mgc_scaler = StandardScaler(with_mean=True, with_std=True)
        mgc_scalers.append(mgc_scaler)
        mgc_lsp_training[:, i] = mgc_scalers[i].fit_transform(mgc_lsp_training[:, i].reshape(-1, 1)).ravel()
        mgc_lsp_validation[:, i] = mgc_scalers[i].transform(mgc_lsp_validation[:, i].reshape(-1, 1)).ravel()


    # target: already scaled to [0-1]
    
    # best parameters are chosen from hyperparameter optimization:
    # - input scaler: minmax
    # - target image resize: using sklearn
    # - optimizer: adamax
    # - activation: relu
    # - n_layers: 4
    # - n_neurons: 1000, 1500, 1500, 4000
    # - batch_size: 64
    
    ### single training
    model = Sequential()
    model.add(Dense(1000, input_dim=n_mgc, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(n_lines * n_pixels_reduced), kernel_initializer='normal', activation='linear'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adamax')

    print(model.summary())


    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    model_name = 'models/UTI_inversion_FC-DNN_baseline_' + speaker + '_' + current_date

    print('starting training', speaker, current_date)
    
    # save model
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

    # serialize scalers to pickle
    pickle.dump(mgc_scalers, open(model_name + '_mgc_scalers.sav', 'wb'))

    print(current_date)
    # early stopping to avoid over-training
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0), \
                 CSVLogger(model_name + '.csv', append=True, separator=';'), \
                 ModelCheckpoint(model_name + '_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

    # Run training
    history = model.fit(mgc_lsp_training, ult_training,
                            epochs = 100, batch_size = 64, shuffle = True, verbose = 1,
                            validation_data=(mgc_lsp_validation, ult_validation),
                            callbacks=callbacks)
    
    print('finished training', speaker, current_date)

