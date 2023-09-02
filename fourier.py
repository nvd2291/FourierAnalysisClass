'''
Author: Naveed Naeem
Title: fourier.py
Purpose: Provides the capability to easily plot and display different types of
signal types along with their corresponding FFT plots
'''
from multipledispatch import dispatch
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from math import floor, log10, sqrt
from blackmanharris7 import blackmanharris7
from typing import Optional

import pandas as pd
import numpy as np
import random

class FourierDataObject():
    """
    The FourierDataObject Class is used to generate time and frequency domain data based on user inputs
    
    If a FourierDataObject is created without specifying any values then the following are the defaults:

    Signal Type: Sine Wave
    Signal Frequency: 1kHz
    Sampling Frequency: 1MHz
    Amplitude (Unitless): 1
    Duty Cycle: 50%
    DC Offset (Unitless): 0
    FFT Window: Rectangular(Flat-top)

    Noise: Disabled
    Windowing: Disabled
    """
    
    __noise_types = ('white', 'pink', 'brown')
    __signal_types = ('sine', 'square', 'sawtooth', 'triangle')
    __window_types = ('bartlett', 'blackman', 'blackmanharris4', 'blackmanharris7', 'boxcar', 'flattop', 'hamming', 'hanning', 'parzen', 'triangular', 'tukey')
    __sawtooth_types = {'left': 0, 'right': 1}
    __name = 'FourierDataObject'

    def __init__(self, signal_frequency = 1e3, sample_frequency= 1e6, amplitude = 1.0, duty_cycle = 0.5, dc_offset = 0.0):

        # self.__input_data = []
        self.__curr_sig_type =  self.__signal_types[0]
        self.__curr_noise_type = self.__noise_types[0]
        self.__curr_window_type = self.__window_types[4]
        self.__start_time = 0.0
        self.__end_time = 0.10
        self.__amplitude = amplitude
        self.__dc_offset = dc_offset
        self.__duty_cycle = duty_cycle
        self.__max_noise = 0.1
        self.__noise_enable = False
        self.__window_enable = False
        self.__sample_frequency = sample_frequency 
        self.__signal_frequency = signal_frequency
        self.__sawtooth_type = self.__sawtooth_types['left']
        self.__sample_period = 1        
        self.__num_samples = 32768
        self.__enbw = 1
        self.__cpg = 0
        #Values calculated based on initialized values
        self.calc_sample_period()
        self.calc_num_samples()

        #Default Values
        self.generate_time_axis()
        self.__signal_data = np.zeros(len(self.__time_axis_data))
        self.__fft_data_raw = []
        self.__fft_magnitude = []
        self.__fft_bins = []
  
    def __repr__(cls) -> str:
        return cls.__name

    def get_window_types(cls):
        return cls.__window_types

    def get_noise_types(cls):
        return cls.__noise_types

    def calc_sample_period(cls):
        """ Calculates the sample period based on the sampling frequency """
        cls.__sample_period = 1.0 / cls.__sample_frequency

    def calc_num_samples(cls):
        """
        Calculates an integer number of samples based on the current start time, end time, and sampled period
        """
        cls.__num_samples = floor(abs(cls.__end_time - cls.__start_time) / cls.__sample_period)

    def set_time(cls, start_time: int, stop_time: int):
        """
        Sets the start and stop time of the signal in seconds
        """
        cls.__start_time = start_time
        cls.__end_time =  stop_time

    def equivalent_noise_bandwidth(cls, window = None, dB = False):
        if window is None:
            window = cls.__window_data
        enbw = len(window) * (np.sum(window ** 2)/ np.sum(window) ** 2)
        if not dB:
            cls.__enbw = enbw
        else:
            cls.__enbw = 10 * log10(enbw)
    
    ## Calculate the Coherent Power Gain value based on the current window
    def coherent_power_gain(cls, window = None, dB = False):
        if window is None:
            window = cls.__window_data
        cpg = np.sum(window)/ len(window)
        if not dB:
            cls.__cpg = cpg
        else:
            cls.__cpg = 20 * log10(cpg)
    
    def set_noise_type(cls, noise_type: str, noise_magnitude: Optional[float] = None):
        noise_type = noise_type.lower()
        if noise_type not in cls.__noise_types:
            print('ERROR: Unexpected Input')
            print(f"The following are noise types are accepted inputs: {cls.__noise_types}")
        else:
            if noise_magnitude is not None:
                cls.__max_noise = noise_magnitude
            cls.__curr_noise_type = noise_type
            cls.generate_noise_data()

    def set_signal_type(cls, signal_type: str, sawtooth_type = 'left'):
        signal_type = signal_type.lower()
        if signal_type not in cls.__signal_types:
            print('ERROR: Unexpected Input')
            print(f"The following are signal types are accepted inputs: {cls.__signal_types}")
        else:
            cls.__curr_sig_type = signal_type
            if cls.__curr_sig_type == cls.__signal_types[2]:
                cls.__sawtooth_type = cls.__sawtooth_types[sawtooth_type]

    def set_window_type(cls, window_type: str):
        window_type = window_type.lower()
        if window_type not in cls.__window_types:
            print('ERROR: Unexpected Input')
            print(f"The following are FFT windows are accepted inputs: {cls.__window_types}")
        else:
            cls.__curr_window_type = window_type
            cls.fft_window_data()
    
    def enable_window(cls):
        cls.__window_enable = True

    def disable_window(cls):
        cls.__window_enable = False

    def enable_noise(cls):
        cls.__noise_enable = True

    def disable_noise(cls):
        cls.__noise_enable  = False

    def set_amplitude(cls, amplitude: float):
        if cls.__signal_data is None:
            cls.__amplitude = amplitude
        else:
            cls.__signal_data /= cls.__amplitude
            cls.__amplitude = amplitude
            cls.__signal_data *= amplitude

    def set_offset(cls, offset: float):
        if cls.__signal_data is None:
            cls.__dc_offset = offset
        else:
            cls.__signal_data -= cls.__dc_offset
            cls.__dc_offset = offset
            cls.__signal_data += offset
    
    def set_frequency(cls, frequencyVal):
        if frequencyVal > 0:
            cls.__signal_frequency = frequencyVal
    
    def generate_time_axis(cls):
        cls.__time_axis_data = np.linspace(cls.__start_time, cls.__end_time, cls.__num_samples)

    def generate_time_domain_data(cls, signal_type: Optional[str] = None):
        """
        This method will generate the time domain data base on the current signal configuration
        """
        if signal_type is not None:
            cls.set_signal_type(signal_type)

        cls.calc_sample_period()
        cls.calc_num_samples()

        cls.generate_time_axis()

        if cls.__curr_sig_type == 'sine':
            yAxis = np.sin(2 * np.pi * cls.__signal_frequency * cls.__time_axis_data)
            cls.__signal_data = yAxis * cls.__amplitude + cls.__dc_offset
    
        elif cls.__curr_sig_type =='square' :
            yAxis = signal.square(2 * np.pi * cls.__signal_frequency * cls.__time_axis_data, cls.__duty_cycle)
            cls.__signal_data = yAxis * cls.__amplitude + cls.__dc_offset

        elif cls.__curr_sig_type == 'sawtooth':
            yAxis = signal.sawtooth(2 * np.pi * cls.__signal_frequency * cls.__time_axis_data, cls.__sawtooth_type)
            cls.__signal_data = yAxis * cls.__amplitude + cls.__dc_offset

        elif cls.__curr_sig_type == 'triangle':
            yAxis = signal.sawtooth(2 * np.pi * cls.__signal_frequency * cls.__time_axis_data, cls.__duty_cycle)
            cls.__signal_data = yAxis * cls.__amplitude + cls.__dc_offset
            
        if cls.__noise_enable == True:
            cls.generate_noise_data()
            cls.__signal_data += cls.noise_data

    def construct_square_wave_from_sines(cls, 
                                        harmonics = 7, 
                                        amplitude: Optional[float] = None, 
                                        frequency: Optional[float] = None, 
                                        with_noise: Optional[bool] = None,
                                        noise_magnitude: Optional[float] = None):
        
        if amplitude is not None:
            cls.__amplitude = amplitude
        if frequency is not None:
            cls.__signal_frequency = frequency
        if noise_magnitude is not None:
            cls.__max_noise = noise_magnitude
        if with_noise is not None:
            cls.__noise_enable = with_noise

        cls.calc_sample_period()
        cls.calc_num_samples()

        four_over_pi = 4 / np.pi
        cls.generate_time_axis()

        sq_wave = np.zeros(len(cls.__time_axis_data))
        for n in range(1, (harmonics * 2 + 1), 2):

            sq_wave += (four_over_pi * ((1 / n) * 
                        np.sin( n * 2 * np.pi * 
                        cls.__signal_frequency * 
                        cls.__time_axis_data)))

        cls.__signal_data = (sq_wave * cls.__amplitude) + cls.__dc_offset
        
        if cls.__noise_enable == True:
            cls.generate_noise_data()
            cls.__signal_data += cls.noise_data


    def construct_triangle_wave_from_sines(cls, 
                                           harmonics = 7, 
                                           amplitude: Optional[float] = None, 
                                           frequency: Optional[float] = None, 
                                           with_noise: Optional[bool] = None,
                                           noise_magnitude: Optional[float] = None):
        
        if amplitude is not None:
            cls.__amplitude = amplitude
        if frequency is not None:
            cls.__signal_frequency = frequency
        if noise_magnitude is not None:
            cls.__max_noise = noise_magnitude
        if with_noise is not None:
            cls.__noise_enable = with_noise

        cls.calc_sample_period()
        cls.calc_num_samples()
        cls.generate_time_axis()

        scaling_factor = (8 / (np.pi ** 2)) * cls.__amplitude
        triangle_wave = np.zeros(len(cls.__time_axis_data))

        # Only sum odd number of harmonics
        for n in range(1, (harmonics * 2 + 1), 2):
            triangle_wave += (scaling_factor * ((-1) ** ((n - 1)/ 2)) * 
                              (1 / (n ** 2)) * 
                              np.sin(2 * np.pi * 
                              cls.__signal_frequency * 
                              cls.__time_axis_data * n))

        cls.__signal_data = triangle_wave + cls.__dc_offset


        if cls.__noise_enable:
            cls.generate_noise_data()
            cls.__signal_data += cls.noise_data

    def fft_window_data(cls):
            
        if cls.__curr_window_type == 'blackmanharris4':
            return signal.get_window('blackmanharris', cls.__num_samples)

        elif cls.__curr_window_type == 'blackmanharris7':
            return blackmanharris7(cls.__num_samples)
        
        elif cls.__curr_window_type == 'hanning':
            return signal.get_window('hann', cls.__num_samples)

        elif cls.__curr_window_type == 'triangular':
            return signal.get_window('triang', cls.__num_samples)

        else:
            return signal.get_window(cls.__curr_window_type, cls.__num_samples)
    
        print('ERROR: Unexpected Window type detected') 

    def generate_freq_domain_data(cls, is_windowed: Optional[bool] = None):

        # Calculate the size of the frequency bins
        cls.fft_bin_size = (cls.__sample_frequency/cls.__num_samples)

        if is_windowed is not None:
            cls.__window_enable = is_windowed

        if cls.__window_enable:
            cls.__window_data = cls.fft_window_data()
            cls.equivalent_noise_bandwidth()
            cls.coherent_power_gain()
            scaling_factor = 1/(cls.__cpg / sqrt(cls.__enbw))

            windowed_signal = cls.__window_data * cls.__signal_data
            fft_data = np.absolute(fft(windowed_signal)/cls.__num_samples) * scaling_factor

        else:
        # Two-Sided FFT data
            fft_data = fft(cls.__signal_data)/cls.__num_samples

        # Converted Two-Sided FFT to One-Sided
        one_sided_sample_limit = (cls.__num_samples)//2
        fft_data_one_sided = (fft_data[0:one_sided_sample_limit]) * 2

        #Remove DC Bin
        fft_data_one_sided = np.delete(fft_data_one_sided, 0)

        #Generate the FFT Frequency Bins
        cls.__fft_bins = np.arange(1, one_sided_sample_limit) * cls.fft_bin_size

        #Compute the fft magnitude
        cls.__fft_magnitude = 20 * np.log10(fft_data_one_sided)

    def generate_noise_data(cls, noise_type: Optional[str] = None, noise_magnitude: Optional[float] = None):

        if noise_type is not None:
            cls.set_noise_type(noise_type)
        if noise_magnitude is not None:
            cls.__max_noise = noise_magnitude

        #White noise = random uniform distribution
        if cls.__curr_noise_type == 'white':
            cls.noise_data = np.random.uniform(size = len(cls.__signal_data)) * cls.__max_noise

        elif cls.__curr_noise_type == 'brown':
            cls.noise_data = np.cumsum(np.random.uniform(size = len(cls.__signal_data)))/ cls.__num_samples * cls.__max_noise

        elif cls.__curr_noise_type == 'pink':
            pass
        else:
            print('ERROR: Unexpected Noise type detected') 

    def get_time_domain_data(cls):
        return [cls.__time_axis_data, cls.__signal_data]

    def get_fft_domain_data(cls):
        return [cls.__fft_bins, cls.__fft_magnitude]

    def get_freq(cls):
        return cls.__signal_frequency

    def get_fs(cls):
        return cls.__sample_frequency
    
    def get_amplitude(cls):
        return cls.__amplitude

    def get_noise_magnitude(cls):
        return cls.__max_noise

    def get_window_type(cls):
        return cls.__curr_window_type
    
    def get_window_state(cls):
        return cls.__window_enable
    def plot_time_domain(cls):

        plt.figure()
        cls.__amplitude = abs(max(cls.__signal_data) - min(cls.__signal_data))
        plt.plot(cls.__time_axis_data, cls.__signal_data)
        plt.title(f"Time Domain Data: Frequency: {cls.__signal_frequency}Hz, Sampling Frequency: {cls.__sample_frequency}Hz, Amplitude: {cls.__amplitude}")
        plt.xlim(min(cls.__time_axis_data), max(cls.__time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')
        plt.show()

    def plot_fft(cls):

        plt.figure()
        plt.semilogx(cls.__fft_bins, cls.__fft_magnitude)
        plt.title(f"FFT Plot: Frequency: {cls.__signal_frequency}Hz, Sampling Frequency: {cls.__sample_frequency}Hz, FFT Window: {cls.__curr_noise_type.capitalize()}")
        plt.xlim(min(cls.__fft_bins), max(cls.__fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, 'both')
        plt.show()
        
    def plot_time_and_fft(cls):
        
        #Plot Time Domain Data
        plt.figure()
        cls.__amplitude = abs(max(cls.__signal_data) - min(cls.__signal_data))
        plt.subplot(2,1,1)
        plt.plot(cls.__time_axis_data, cls.__signal_data)
        plt.title(f"Time Domain Data: Frequency: {cls.__signal_frequency}Hz, Sampling Frequency: {cls.__sample_frequency}Hz, Amplitude: {cls.__amplitude}")
        plt.xlim(min(cls.__time_axis_data), max(cls.__time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')

        #Plot Frequency Domain Data
        plt.subplot(2,1,2)
        plt.semilogx(cls.__fft_bins, cls.__fft_magnitude)
        if cls.__window_enable:
            plt.title(f"FFT Plot: Frequency: {cls.__signal_frequency}Hz, Sampling Frequency: {cls.__sample_frequency}Hz, FFT Window: {cls.__curr_noise_type.capitalize()}")
        else:
            plt.title(f"FFT Plot: Frequency: {cls.__signal_frequency}Hz, Sampling Frequency: {cls.__sample_frequency}Hz, FFT Window: No Window")
        plt.xlim(min(cls.__fft_bins), max(cls.__fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, 'both')

        #Show Plot
        plt.show()