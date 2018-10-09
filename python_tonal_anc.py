#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
________________________________________________

PYTHON TONAL FXLMS ACTIVE CONTROLLER
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
from scipy.signal import lfilter, butter
from guizero import App,PushButton,Slider,Text,CheckBox

########## <<<<<<<<< VARIABLES >>>>>>>>> ##########
# Audio Stream Variables
fs = 44100
output_index = 2
input_index = 2
frame_size = 1024
chan_count = 2

# LMS Variables
gamma = 0.999999
alpha = 1e-2

# Global Variables
freq = 250
control_gain = 0

last_t = 0  # Counter used for Sine Wave

# Estimate of Plant Response 
h = np.load('plantdata.npy')

# Calculate HPF at 30Hz
normal_cutoff = 50 / (0.5 * fs)
b, a = butter(6, normal_cutoff, btype='high', analog=False)

# Initial LMS Coefficients
u = np.zeros((2,1))
update = np.zeros((2,1))
r = np.zeros((2,frame_size))

########## <<<<<<<<< FUNCTION DEFS >>>>>>>>> ##########


# MAIN CALLBACK FUNCTION
def playingCallback(in_data, frame_count, time_info, status):
    global u
    global last_t
    global control_gain
    
    # Input Processing
    audio_frame_int = np.frombuffer(in_data,dtype=np.float32)  # Convert Bytes to Numpy Array
    
    # Extract Only Input Channel 1
    audio_frame = np.reshape(audio_frame_int, (frame_size, 2))
    mic_in = audio_frame[:,0]
    mic_in = lfilter(b, a, mic_in)  # APply HPF

    # Generate Ref Signals
    t = np.arange(last_t,last_t + frame_size)/fs
    ref_sine = np.sin((2*np.pi*freq*t))
    ref_cos = np.cos((2*np.pi*freq*t))
    last_t = last_t+frame_size

    # Calculate Output Signal for Current Sample 
    control_out = (ref_sine * u[0]) + (ref_cos * u[1])
    control_out = -control_out

    # Filter Ref Signal by Plant Estimate
    r[0,:] = (ref_sine * np.real(h)) - (ref_cos * np.imag(h))
    r[1,:] = (ref_sine * np.imag(h)) + (ref_cos * np.real(h))

    # Update Filter Coefficients for Next Sample Using Block LMS
    update[0] = (alpha/frame_size) * np.sum(r[0,:] * mic_in)
    update[1] = (alpha/frame_size) * np.sum(r[1,:] * mic_in)

    # Force LMS Filters to 0 when ANC is Turned Off
    if control_gain == 0:
        u = np.zeros((2,1))
    else:
        u[0] = (gamma * u[0]) - update[0] 
        u[1] = (gamma * u[1]) - update[1] 

    out_mat = (control_out,ref_sine)    # Channel1 = Control, Channel2 = Primary
    out_mat = np.vstack(out_mat).reshape((-1,), order='F')

    # Ouput Processing
    out_data = out_mat.astype(np.float32)
    out_data = out_data.tobytes()
    return out_data, pyaudio.paContinue

# Turn ANC on/off using tickbox
def change_gain():
    global control_gain
    global u
    if control_flag.value ==  1:
        control_gain = 1
        print('Control On')
    else:
        control_gain = 0
        u = np.zeros((2,1))
        print('Control Off')

def exit_function():
    stream.stop_stream()
    print('Function Terminated')



########## <<<<<<<<< SETUP AUDIO STREAM >>>>>>>>> ##########
p = pyaudio.PyAudio()

stream = p.open(format = pyaudio.paFloat32,
    channels = chan_count,
    rate = fs,
    output = True,
    input = True,
    stream_callback=playingCallback,
    output_device_index=output_index,
    input_device_index=input_index,
    frames_per_buffer=frame_size)



##### <<<<<<<<< CONFIGURE GUI >>>>>>>>> #####
control_app = App(title="Control App")
control_flag = CheckBox(control_app,text="ANC ON",command=change_gain)


########## <<<<<<<<< RUN >>>>>>>>> ##########
atexit.register(exit_function)
print('Starting Function')
print('Primary Disturbance Initiated')
stream.start_stream()
control_app.display()
time.sleep(60)
