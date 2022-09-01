from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from math import floor, log10

import pandas as pd
import numpy as np
import random

class FourierDataObject():
    """The FourierDataObject Class is used to generate time and frequency domain data based on user inputs"""
    
    __noise_types = ('none', 'white', 'pink', 'brown')
    __signal_types = ('sine', 'square', 'sawtooth', 'triangle')
    __window_types = ('none', 'bartlett', 'blackman-harris5', 'blackman-harris7', 'hamming', 'hanning', 'rectangular')
    __sawtooth_types = {'left': 0, 'right': 1}
    name = 'FourierDataObject'

    def __init__(self, signal_frequency = 1e3, sample_frequency= 1e6, amplitude = 1.0, duty_cycle = 0.5, dc_offset = 0.0):

        #Default Values
        self.time_axis_data = []
        self.signal_data = []
        self.signal_type =  self.__signal_types[0]
        self.noise_type = self.__noise_types[0]
        self.window_type = self.__window_types[0]
        self.start_time = 0.0
        self.end_time = 0.10
        self.sample_frequency = sample_frequency
        self.signal_frequency = signal_frequency
        self.sawtooth_type = self.__sawtooth_types['left']

        #Values calculated based on initialized values
        self.calc_sample_period()
        self.calc_num_samples()
        self.amplitude = amplitude
        self.dc_offset = dc_offset
        self.duty_cycle = duty_cycle
  
    def __repr__(cls) -> str:
        return cls.name

    def calc_sample_period(cls):
        cls.sample_period = 1.0 / cls.sample_frequency

    def calc_num_samples(cls) -> int:
        cls.num_samples = floor(abs(cls.end_time - cls.start_time) / cls.sample_period)

    def select_noise(cls, noise_type: str):
        noise_type = noise_type.lower()
        if noise_type not in cls.__noise_types:
            print('ERROR: Unexpected Input')
            print(f"The following are noise types are accepted inputs: {cls.__noise_types}")
        else:
            cls.noise_type = noise_type
    
    def select_signal(cls, signal_type: str, sawtooth_type = 'left'):
        signal_type = signal_type.lower()
        if signal_type not in cls.__signal_types:
            print('ERROR: Unexpected Input')
            print(f"The following are signal types are accepted inputs: {cls.__signal_types}")
        else:
            cls.signal_type = signal_type
            if cls.signal_type == cls.__signal_types[2]:
                cls.sawtooth_type = cls.__sawtooth_types[sawtooth_type]

    def select_window(cls, window_type: str):
        window_type = window_type.lower()
        if window_type not in cls.__window_types:
            print('ERROR: Unexpected Input')
            print(f"The following are FFT windows are accepted inputs: {cls.__window_types}")
        else:
            cls.window_type = window_type
    
    #Method to generate time domain data based on the 
    def generate_time_domain_data(cls):

        """
        This method will generate the time domain data base on the current signal configuration
        """
        cls.calc_sample_period()
        cls.calc_num_samples()

        cls.time_axis_data = np.linspace(cls.start_time, cls.end_time, cls.num_samples)

        ## TODO: Incorporate Noise Generation
        if cls.signal_type == 'sine':
            yAxis = np.sin(2 * np.pi * cls.signal_frequency * cls.time_axis_data)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset
    
        elif cls.signal_type =='square' :
            yAxis = signal.square(2 * np.pi * cls.signal_frequency * cls.time_axis_data, cls.duty_cycle)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset

        elif cls.signal_type ==  'sawtooth':
            
            yAxis = signal.sawtooth(2 * np.pi * cls.signal_frequency * cls.time_axis_data, cls.sawtooth_type)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset

        elif cls.signal_type == 'triangle':
            yAxis = signal.sawtooth(2 * np.pi * cls.signal_frequency * cls.time_axis_data, cls.duty_cycle)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset
    
    def generate_freq_domain_data(cls):
        
        # Calculate the size of the frequency bins
        cls.fft_bin_size = (cls.sample_frequency/cls.num_samples)
        
        # Two-Sided FFT data
        fft_data = fft(cls.signal_data)/cls.num_samples

        # Converted Two-Sided FFT to One-Sided
        one_sided_sample_limit = (cls.num_samples)//2
        fft_data_one_sided = (fft_data[0:one_sided_sample_limit]) * 2

        #Remove DC Bin
        fft_data_one_sided = np.delete(fft_data_one_sided, 0)

        #Generate the FFT Frequency Bins
        cls.fft_bins = np.arange(1, one_sided_sample_limit) * cls.fft_bin_size

        #Compute the fft magnitude
        cls.fft_magnitude = 20 * np.log10(np.absolute(fft_data_one_sided))

        

    def generate_noise_data(cls):
        
        if cls.noise_type == 'none':
            pass
        elif cls.noise_type == 'white':
            pass
        elif cls.noise_type == 'brown':
            pass
        elif cls.noise_type == 'pink':
            pass
        else:
            print('ERROR: Unexpected Noise type detected') 

    def fft_window_data(cls):
        if cls.window_type == 'none':
            pass
        elif cls.window_type == 'bartlett':
            pass
        elif cls.window_type == 'blackman-harris5':
            pass
        elif cls.window_type == 'blackman-harris7':
            pass
        elif cls.window_type == 'hamming':
            pass
        elif cls.window_type == 'hanning':
            pass
        elif cls.window_type == 'rectangular':
            pass
        else:
            print('ERROR: Unexpected Window type detected') 



    def plot_time_domain(cls):


        cls.amplitude = abs(max(cls.signal_data) - min(cls.signal_data))
        plt.plot(cls.time_axis_data, cls.signal_data)
        plt.title(f"Time Domain Data: Frequency: {cls.signal_frequency}Hz, Sampling Frequency: {cls.sample_frequency}Hz, Amplitude: {cls.amplitude}")
        plt.xlim(min(cls.time_axis_data), max(cls.time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.show()

    def plot_fft(cls):

        plt.semilogx(cls.fft_bins, cls.fft_magnitude)
        # plt.semilogx(cls.fft_magnitude)

        plt.title(f"FFT Plot: Frequency: {cls.signal_frequency}Hz, Sampling Frequency: {cls.sample_frequency}Hz, FFT Window: {cls.window_type.capitalize()}")
        # plt.xlim(min(cls.fft_bins), max(cls.fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.show()
        
    


