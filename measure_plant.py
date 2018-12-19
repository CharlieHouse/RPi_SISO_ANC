#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
________________________________________________

measure_plant.py
University of Southampton
Institute of Sound and Vibration Research

Charlie House - September 2018
Contact: c.house@soton.ac.uk
________________________________________________

"""

##### IMPORTS #####

import pyaudio
import numpy as np
import math
import atexit
import time
from scipy.io import savemat
import scipy.signal as sig

########## <<<<<<<<< VARIABLES >>>>>>>>> ##########
# Audio Stream Variables
fs = 44100
audio_dev_index = 2
frame_size = 2048

# Global Variables
gain = 1
duration = 10   # Duration of Measurement (in Seconds)
last_t = 0  # Counter used for Sine Wave

# Calculate HPF at 30Hz
normal_cutoff = 50 / (0.5 * fs)
b, a = sig.butter(6, normal_cutoff, btype='high', analog=False)

########## <<<<<<<<< FUNCTION DEFS >>>>>>>>> ##########

input_recording = np.zeros(((duration+1)*fs))
noise_recording = np.zeros(((duration+1)*fs))
# MAIN CALLBACK FUNCTION
def playingCallback(in_data, frame_count, time_info, status):
    global last_t
    global input_recording

    # Input Processing
    audio_frame_int = np.frombuffer(in_data,dtype=np.float32)  # Convert Bytes to Numpy Array
    audio_frame = np.reshape(audio_frame_int, (frame_size, 2))

    mic_in = audio_frame[:,0]
    mic_in = sig.lfilter(b, a, mic_in)  # Apply HPF

    # Record Input Channel 1
    input_recording[last_t : last_t+frame_size] = mic_in

    # Generate Noise Signals
    t = np.arange(last_t,last_t + frame_size)/fs
    sine = np.sin((2*np.pi*freq*t))
    
    # Play & Save Noise Signals
    noise_recording[last_t : last_t+frame_size] = sine*gain
    last_t = last_t+frame_size

    out_mat = (sine*gain,np.zeros(frame_size,))    # Channel1 = Control (Noise), Channel2 = Primary (Silence)
    
    # Ouput Processing
    out_mat = np.vstack(out_mat).reshape((-1,), order='F')
    out_data = out_mat.astype(np.float32)
    out_data = out_data.tobytes()
    return out_data, pyaudio.paContinue



########## <<<<<<<<< SETUP AUDIO STREAM >>>>>>>>> ##########
p = pyaudio.PyAudio()

stream = p.open(format = pyaudio.paFloat32,
    channels = 2,
    rate = fs,
    output = True,
    input = True,
    stream_callback=playingCallback,
    output_device_index=audio_dev_index,
    input_device_index=audio_dev_index,
    frames_per_buffer=frame_size)


########## <<<<<<<<< RUN >>>>>>>>> ##########


def measure_plant(freq):
    global noise_recording
    global input_recording
    
    # Run Measurement
    print('Measuring Plant Response')
    print('Starting Tone')
    stream.start_stream()
    time.sleep(duration)

    print('Stopping Tone')
    stream.stop_stream()

    # Truncate Recordings
    # noise_recording = noise_recording[1*fs:duration-2*fs]
    # input_recording = input_recording[1*fs:duration-2*fs]

    # Calculate TF
    print('Calculating Transfer Function')
    nfft = 16384
    [f,sxy] = sig.csd(noise_recording,input_recording,fs=fs,window='hanning',nperseg=nfft,noverlap=nfft/2,nfft=nfft)
    [f,sxx] = sig.welch(noise_recording,fs=fs,window='hanning',nperseg=nfft,noverlap=nfft/2,nfft=nfft)
    [f,coh] = sig.coherence(noise_recording, input_recording, fs=fs, window='hanning', nperseg=nfft, noverlap=nfft/2, nfft=nfft)
    H = np.divide(sxy,sxx)

    # Extract Frequency from TF & Print
    freq_ind = idx = (np.abs(f - freq)).argmin()
    plant_est = H[freq_ind]
    print('Coherence:')
    print(coh[freq_ind])
    print('Plant Estimate')
    print(plant_est)

    # Save
    print('Saving Data')
    savemat('plant.mat', {'mic':input_recording,'noise':noise_recording,'f':f,'H':H,'sxx':sxx,'sxy':sxy,'coh':coh})
    np.save('plantdata.npy',plant_est)
    return plant_est
    
measure_plant(250)

