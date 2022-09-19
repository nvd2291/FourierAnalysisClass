from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from math import floor, log10, sqrt
from blackmanharris7 import blackmanharris7

import pandas as pd
import numpy as np
import random

class FourierDataObject():
    """The FourierDataObject Class is used to generate time and frequency domain data based on user inputs"""
    
    __noise_types = ('white', 'pink', 'brown')
    __signal_types = ('sine', 'square', 'sawtooth', 'triangle')
    __window_types = ('bartlett', 'blackman', 'blackmanharris4', 'blackmanharris7', 'boxcar', 'flatop', 'hamming', 'hanning', 'parzen', 'triangular', 'tukey')
    __sawtooth_types = {'left': 0, 'right': 1}
    name = 'FourierDataObject'

    def __init__(self, signal_frequency = 1e3, sample_frequency= 1e6, amplitude = 1.0, duty_cycle = 0.5, dc_offset = 0.0):

        #Default Values
        self.time_axis_data = []
        self.signal_data = []
        self._signal_type =  self.__signal_types[0]
        self._noise_type = self.__noise_types[0]
        self._window_type = self.__window_types[0]
        self.start_time = 0.0
        self.end_time = 0.10
        self.max_noise = 0.1
        self.window_enable = False
        self.noise_enable = False
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

    def equivalent_noise_bandwidth(cls, window, dB = False):
        enbw = len(window) * (np.sum(window**2)/ np.sum(window)**2)
        if not dB:
            return enbw
        else:
            return 10 * log10(enbw)
    
    def coherent_power_gain(cls, window, dB = False):
        cpg = np.sum(window)/ len(window)
        if not dB:
            return cpg
        else:
            return 20 * log10(cpg)

    def select_noise(cls, noise_type: str):
        noise_type = noise_type.lower()
        if noise_type not in cls.__noise_types:
            print('ERROR: Unexpected Input')
            print(f"The following are noise types are accepted inputs: {cls.__noise_types}")
        else:
            cls._noise_type = noise_type

    
    def select_signal(cls, signal_type: str, sawtooth_type = 'left'):
        signal_type = signal_type.lower()
        if signal_type not in cls.__signal_types:
            print('ERROR: Unexpected Input')
            print(f"The following are signal types are accepted inputs: {cls.__signal_types}")
        else:
            cls._signal_type = signal_type
            if cls._signal_type == cls.__signal_types[2]:
                cls.sawtooth_type = cls.__sawtooth_types[sawtooth_type]

    def select_window(cls, window_type: str):
        window_type = window_type.lower()
        if window_type not in cls.__window_types:
            print('ERROR: Unexpected Input')
            print(f"The following are FFT windows are accepted inputs: {cls.__window_types}")
        else:
            cls._window_type = window_type
    
    #Method to generate time domain data based on the 
    def generate_time_domain_data(cls):

        """
        This method will generate the time domain data base on the current signal configuration
        """
        cls.calc_sample_period()
        cls.calc_num_samples()

        cls.time_axis_data = np.linspace(cls.start_time, cls.end_time, cls.num_samples)

        if cls._signal_type == 'sine':
            yAxis = np.sin(2 * np.pi * cls.signal_frequency * cls.time_axis_data)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset
    
        elif cls._signal_type =='square' :
            yAxis = signal.square(2 * np.pi * cls.signal_frequency * cls.time_axis_data, cls.duty_cycle)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset

        elif cls._signal_type == 'sawtooth':
            
            yAxis = signal.sawtooth(2 * np.pi * cls.signal_frequency * cls.time_axis_data, cls.sawtooth_type)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset

        elif cls._signal_type == 'triangle':
            yAxis = signal.sawtooth(2 * np.pi * cls.signal_frequency * cls.time_axis_data, cls.duty_cycle)
            cls.signal_data = yAxis * cls.amplitude + cls.dc_offset
    
        if cls.noise_enable == True:
            cls.generate_noise_data()
            cls.signal_data += cls.noise_data
        


    def generate_freq_domain_data(cls):

        # Calculate the size of the frequency bins
        cls.fft_bin_size = (cls.sample_frequency/cls.num_samples)
        
        if cls.window_enable == True:
            window_data = cls.fft_window_data()
            enbw = cls.equivalent_noise_bandwidth(window_data)
            cpg = cls.coherent_power_gain(window_data)
            scaling_factor = 1/(cpg / sqrt(enbw))

            windowed_signal = window_data * cls.signal_data
            fft_data = np.absolute(fft(windowed_signal)/cls.num_samples) * scaling_factor

        else:
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
        cls.fft_magnitude = 20 * np.log10(fft_data_one_sided)

        

    def generate_noise_data(cls):
        
        #White noise = random uniform distribution
        if cls._noise_type == 'white':
            cls.noise_data = np.random.uniform(size = len(cls.signal_data)) * cls.max_noise

        elif cls._noise_type == 'brown':
            cls.noise_data = np.cumsum(np.random.uniform(size = len(cls.signal_data)))/ cls.num_samples * cls.max_noise

        elif cls._noise_type == 'pink':
            pass
        else:
            print('ERROR: Unexpected Noise type detected') 

    def fft_window_data(cls):
            
        if cls._window_type == 'blackmanharris4':
            return signal.get_window('blackmanharris', cls.num_samples)

        elif cls._window_type == 'blackmanharris7':
            return blackmanharris7(cls.num_samples)
        
        elif cls._window_type == 'hanning':
            return signal.get_window('hann', cls.num_samples)

        else:
            return signal.get_window(cls._window_type, cls.num_samples)
    
        print('ERROR: Unexpected Window type detected') 




    def plot_time_domain(cls):

        plt.figure(num=1)
        cls.amplitude = abs(max(cls.signal_data) - min(cls.signal_data))
        plt.plot(cls.time_axis_data, cls.signal_data)
        plt.title(f"Time Domain Data: Frequency: {cls.signal_frequency}Hz, Sampling Frequency: {cls.sample_frequency}Hz, Amplitude: {cls.amplitude}")
        plt.xlim(min(cls.time_axis_data), max(cls.time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')
        plt.show()

    def plot_fft(cls):

        plt.figure(num=2)
        plt.semilogx(cls.fft_bins, cls.fft_magnitude)
        plt.title(f"FFT Plot: Frequency: {cls.signal_frequency}Hz, Sampling Frequency: {cls.sample_frequency}Hz, FFT Window: {cls._window_type.capitalize()}")
        plt.xlim(min(cls.fft_bins), max(cls.fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, 'both')
        plt.show()
        
    
    def plot_time_and_fft(cls):
        
        #Plot Time Domain Data
        plt.figure(num=1)
        cls.amplitude = abs(max(cls.signal_data) - min(cls.signal_data))
        plt.subplot(2,1,1)
        plt.plot(cls.time_axis_data, cls.signal_data)
        plt.title(f"Time Domain Data: Frequency: {cls.signal_frequency}Hz, Sampling Frequency: {cls.sample_frequency}Hz, Amplitude: {cls.amplitude}")
        plt.xlim(min(cls.time_axis_data), max(cls.time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')

        #Plot Frequency Domain Data
        plt.subplot(2,1,2)
        plt.semilogx(cls.fft_bins, cls.fft_magnitude)
        if cls.window_enable:
            plt.title(f"FFT Plot: Frequency: {cls.signal_frequency}Hz, Sampling Frequency: {cls.sample_frequency}Hz, FFT Window: {cls._window_type.capitalize()}")
        else:
            plt.title(f"FFT Plot: Frequency: {cls.signal_frequency}Hz, Sampling Frequency: {cls.sample_frequency}Hz, FFT Window: No Window")
        plt.xlim(min(cls.fft_bins), max(cls.fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, 'both')

        #Show Plot
        plt.show()