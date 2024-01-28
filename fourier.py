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

class Fourier():    
    __noise_types = ('white', 'pink', 'brown')
    __signal_types = ('sine', 'square', 'sawtooth', 'triangle')
    __window_types = ('bartlett', 'blackman', 'blackmanharris4', 
                      'blackmanharris7', 'boxcar', 'flattop', 
                      'hamming', 'hanning', 'parzen', 'triangular', 
                      'tukey')

    __sawtooth_types = {'left': 0, 'right': 1}
    __name = 'Fourier'
    __current_figures = 0

    def __init__(self, signal_frequency = 1e3, 
                 fs= 1e6, amplitude = 1.0, 
                 duty_cycle = 0.5, dc_offset = 0.0):

        """
        The Fourier Class is used to generate time and frequency domain data based 
        on user inputs. If a Fourier instance is created without specifying any 
        values then the following are the defaults:

        Signal Type: Sine Wave
        Signal Frequency: 1kHz
        Sampling Frequency: 1MHz
        Amplitude (Unitless): 1
        Duty Cycle: 50% (Range is from 0-1)
        DC Offset (Unitless): 0
        FFT Window: Rectangular(Flat-top)

        Noise: Disabled
        Windowing: Disabled
        -signal_frequency: (Optional) Frequency of the desired signal in Hz. 
                            Defaults to 1kHz.
        - fs: (Optional) Sample Rate frequency. Defaults to 1MHz.
        - amplitude: (Optional) Amplitude of the signal. Defaults to 1.0.
        - duty_cycle: (Optional) Duty cycle of the signal. Defaults to 0.5.
        - dc_offset: (Optional) DC offset of the signal. Defaults to 0.0.
        """

        if duty_cycle < 0 or duty_cycle > 1:
            self.__duty_cycle = 0.5
        else:
            self.__duty_cycle = duty_cycle

        self.__curr_sig_type =  self.__signal_types[0]
        self.__curr_noise_type = self.__noise_types[0]
        self.__curr_window_type = self.__window_types[4]
        self.__start_time = 0.0
        self.__end_time = 0.10
        self.__amplitude = amplitude
        self.__dc_offset = dc_offset

        self.__max_noise = 0.1
        self.__noise_enable = False
        self.__window_enable = False
        self.__fs = fs 
        self.__sig_freq = signal_frequency
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


        self.generate_time_domain_data()
        self.generate_freq_domain_data()
  
    def __repr__(self) -> str:
        return self.__name

    def __format_freq_text(self, freq: float) -> str:

        if freq < 1e3:
            return f"{freq:.4f}Hz"
        elif freq >= 1e3 and freq < 1e6:
            return f"{(freq/1e3):.4f}kHz"
        elif freq >= 1e6 and freq < 1e9:
            return f"{(freq/1e6):.4f}MHz"
        else:
            return f"{(freq/1e9):.4f}GHz"

    def __increment_figures(self):
        self.__current_figures += 1
        return (self.__current_figures)

    @classmethod
    def get_window_types(cls):
        """
        Returns a tuple of the available window types.
        """
        return cls.__window_types

    @classmethod
    def get_noise_types(cls):
        """
        Returns a tuple of the available noise types.
        """
        return cls.__noise_types

    @classmethod
    def get_signal_types(cls):
        """
        Returns a tuple of the available signal types.
        """
        return cls.__signal_types
    
    def calc_sample_period(self):
        """ Calculates the sample period based on the sampling frequency. """
        self.__sample_period = 1.0 / self.__fs

    def calc_num_samples(self):
        """
        Calculates an integer number of samples based on the current start 
        time, end time, and sampled period.
        """
        self.__num_samples = int((abs(self.__end_time - self.__start_time) 
                             // self.__sample_period))

                    
    def set_time(self, start_time: int, stop_time: int):
        """
        Sets the start and stop time of the signal in seconds
        """
        self.__start_time = start_time
        self.__end_time =  stop_time

    def equivalent_noise_bandwidth(self, window = None, dB = False):
        """
        Calculate the ENBW based on the class instances window data or user
        provided window.
        """
        if window is None:
            window = self.__window_data

        enbw = len(window) * (np.sum(window ** 2)/ np.sum(window) ** 2)
        if not dB:
            self.__enbw = enbw
        else:
            self.__enbw = 10 * log10(enbw)
    
    def coherent_power_gain(self, window = None, dB = False):
        """
        Calculate the coherent power gain based on the either a window provided
        as an input or the window data currently in the class instance or a
        user provided window.
        """
        if window is None:
            window = self.__window_data
        cpg = np.sum(window)/ len(window)
        if not dB:
            self.__cpg = cpg
        else:
            self.__cpg = 20 * log10(cpg)
    
    def set_noise_type(self, noise_type: str, 
                       noise_mag: Optional[float] = None):
        """
        Set the current noise type.

        noise_type: @see: __noise_types
        noise_mag: Optionally provide a noise magnitude
        """
        noise_type = noise_type.lower()
        if noise_type not in self.__noise_types:
            print('ERROR: Unexpected Input')
            print(f"The following are noise types are accepted inputs: {self.__noise_types}")
        else:
            if noise_mag is not None:
                self.__max_noise = noise_mag
            self.__curr_noise_type = noise_type
            self.generate_noise_data()

    def set_signal_type(self, signal_type: str, sawtooth_type = 'left'):
        signal_type = signal_type.lower()
        if signal_type not in self.__signal_types:
            print('ERROR: Unexpected Input')
            print(f"The following are signal types are accepted inputs: {self.__signal_types}")
        else:
            self.__curr_sig_type = signal_type
            if self.__curr_sig_type == self.__signal_types[2]:
                self.__sawtooth_type = self.__sawtooth_types[sawtooth_type]

    def set_window_type(self, window_type: str):
        window_type = window_type.lower()
        if window_type not in self.__window_types:
            print('ERROR: Unexpected Input')
            print(f"The following are FFT windows are accepted inputs: {self.__window_types}")
        else:
            self.__curr_window_type = window_type
            self.fft_window_data()
    
    def enable_window(self):
        self.__window_enable = True

    def disable_window(self):
        self.__window_enable = False

    def enable_noise(self):
        self.__noise_enable = True

    def disable_noise(self):
        self.__noise_enable  = False

    def set_amplitude(self, amplitude: float):
        if self.__signal_data is None:
            self.__amplitude = amplitude
        else:
            self.__signal_data /= self.__amplitude
            self.__amplitude = amplitude
            self.__signal_data *= amplitude

    def set_offset(self, offset: float):
        if self.__signal_data is None:
            self.__dc_offset = offset
        else:
            self.__signal_data -= self.__dc_offset
            self.__dc_offset = offset
            self.__signal_data += offset
    
    def set_frequency(self, freq):
        if freq > 0:
            self.__sig_freq = freq
    
    def generate_time_axis(self):
        self.__time_axis_data = np.linspace(self.__start_time, self.__end_time, self.__num_samples)

    def generate_time_domain_data(self, signal_type: Optional[str] = None):
        """
        This method will generate the time domain data base on the current signal configuration
        """
        if signal_type is not None:
            self.set_signal_type(signal_type)

        self.calc_sample_period()
        self.calc_num_samples()

        self.generate_time_axis()

        if self.__curr_sig_type == 'sine':
            yAxis = np.sin(2 * np.pi * self.__sig_freq * self.__time_axis_data)
            self.__signal_data = yAxis * self.__amplitude + self.__dc_offset
    
        elif self.__curr_sig_type =='square' :
            yAxis = signal.square(2 * np.pi * self.__sig_freq * self.__time_axis_data, self.__duty_cycle)
            self.__signal_data = yAxis * self.__amplitude + self.__dc_offset

        elif self.__curr_sig_type == 'sawtooth':
            yAxis = signal.sawtooth(2 * np.pi * self.__sig_freq * self.__time_axis_data, self.__sawtooth_type)
            self.__signal_data = yAxis * self.__amplitude + self.__dc_offset

        elif self.__curr_sig_type == 'triangle':
            yAxis = signal.sawtooth(2 * np.pi * self.__sig_freq * self.__time_axis_data, self.__duty_cycle)
            self.__signal_data = yAxis * self.__amplitude + self.__dc_offset
            
        if self.__noise_enable == True:
            self.generate_noise_data()
            self.__signal_data += self.noise_data
            
        self.generate_freq_domain_data()

    def construct_square_wave_from_sines(self, 
                                        harmonics = 7, 
                                        amplitude: Optional[float] = None, 
                                        freq: Optional[float] = None,
                                        fs: Optional[float] = None,
                                        with_noise: Optional[bool] = None,
                                        noise_mag: Optional[float] = None):
        """
        Parameters:
        - harmonics (int): The number of harmonics that should be used to generate the signal.
        - amplitude (float): The desired amplitude of the signal.
        - freq (float): The signal frequency in Hz.
        - fs (float): Sampling Frequency used to Generate the signal.
        - with_noise (bool): True to generate with noise, False to generate without noise.
        - noise_mag (float): The amount of noise to add to the signal.
        
        Returns:
        Nothing. Updates the classes data members with the newly generated data
        """
        
        if amplitude is not None:
            self.__amplitude = amplitude
        if freq is not None:
            self.__sig_freq = freq
        if fs is not None:
            self. __sampling_freq = fs
        if noise_mag is not None:
            self.__max_noise = noise_mag
        if with_noise is not None:
            self.__noise_enable = with_noise

        self.calc_sample_period()
        self.calc_num_samples()

        four_over_pi = 4 / np.pi
        self.generate_time_axis()

        sq_wave = np.zeros(len(self.__time_axis_data))
        for n in range(1, (harmonics * 2 + 1), 2):

            sq_wave += (four_over_pi * ((1 / n) * 
                        np.sin( n * 2 * np.pi * 
                        self.__sig_freq * 
                        self.__time_axis_data)))

        self.__signal_data = (sq_wave * self.__amplitude) + self.__dc_offset
        
        if self.__noise_enable == True:
            self.generate_noise_data()
            self.__signal_data += self.noise_data
        
        # Regenerate the FFT data with the new signal
        self.generate_freq_domain_data()


    def construct_triangle_wave_from_sines(self, 
                                        harmonics = 7, 
                                        amplitude: Optional[float] = None, 
                                        freq: Optional[float] = None,
                                        fs: Optional[float] = None,
                                        with_noise: Optional[bool] = None,
                                        noise_mag: Optional[float] = None):
        
        """
        Parameters:
        - harmonics (int): The number of harmonics that should be used to generate the signal.
        - amplitude (float): The desired amplitude of the signal.
        - freq (float): The signal frequency in Hz.
        - fs (float): Sampling Frequency used to Generate the signal.
        - with_noise (bool): True to generate with noise, False to generate without noise.
        - noise_mag (float): The amount of noise to add to the signal.
        
        Returns:
        Nothing. Updates the classes data members with the newly generated data
        """
                
        if amplitude is not None:
            self.__amplitude = amplitude
        if freq is not None:
            self.__sig_freq = freq
        if fs is not None:
            self.__fs = fs
        if noise_mag is not None:
            self.__max_noise = noise_mag
        if with_noise is not None:
            self.__noise_enable = with_noise

        self.calc_sample_period()
        self.calc_num_samples()
        self.generate_time_axis()

        triangle_wave = np.zeros(len(self.__time_axis_data))
        
        scaling_factor = (8 / (np.pi ** 2)) * self.__amplitude
        
        # Only sum odd number of harmonics
        for n in range(1, (harmonics * 2 + 1), 2):
            triangle_wave += (scaling_factor * ((-1) ** ((n - 1)/ 2)) * 
                              (1 / (n ** 2)) * 
                              np.sin(2 * np.pi * 
                              self.__sig_freq * 
                              self.__time_axis_data * n))

        self.__signal_data = triangle_wave + self.__dc_offset


        if self.__noise_enable:
            self.generate_noise_data()
            self.__signal_data += self.noise_data

        # Regenerate the FFT data with the new signal
        self.generate_freq_domain_data()

    def fft_window_data(self):
        """
        Generates Window
        """
        if self.__curr_window_type == 'blackmanharris4':
            self.__window_data = signal.get_window('blackmanharris', self.__num_samples)

        elif self.__curr_window_type == 'blackmanharris7':
            self.__window_data = blackmanharris7(self.__num_samples)
        
        elif self.__curr_window_type == 'hanning':
            self.__window_data = signal.get_window('hann', self.__num_samples)

        elif self.__curr_window_type == 'triangular':
            self.__window_data = signal.get_window('triang', self.__num_samples)

        else:
            self.__window_data = signal.get_window(self.__curr_window_type, self.__num_samples)
    
        print('ERROR: Unexpected Window type detected') 

    def generate_freq_domain_data(self, is_windowed: Optional[bool] = None):

        if is_windowed is not None:
            self.__window_enable = is_windowed

        # Calculate the size of the frequency bins
        self.fft_bin_size = (self.__fs/self.__num_samples)
        
        if self.__window_enable:
            self.fft_window_data()
            self.equivalent_noise_bandwidth()
            self.coherent_power_gain()
            scaling_factor = 1/(self.__cpg / sqrt(self.__enbw))

            windowed_signal = self.__window_data * self.__signal_data
            fft_data = np.absolute(fft(windowed_signal)/self.__num_samples) * scaling_factor

        else:
        # Two-Sided FFT data
            fft_data = fft(self.__signal_data)/self.__num_samples

        # Converted Two-Sided FFT to One-Sided
        one_sided_sample_limit = (self.__num_samples)//2
        fft_data_one_sided = (fft_data[0:one_sided_sample_limit]) * 2

        #Remove DC Bin
        fft_data_one_sided = np.delete(fft_data_one_sided, 0)

        #Generate the FFT Frequency Bins
        self.__fft_bins = np.arange(1, one_sided_sample_limit) * self.fft_bin_size

        #Compute the fft magnitude
        self.__fft_magnitude = 20 * np.log10(fft_data_one_sided)

    def generate_noise_data(self, noise_type: Optional[str] = None, noise_mag: Optional[float] = None):

        if noise_type is not None:
            self.set_noise_type(noise_type)
        if noise_mag is not None:
            self.__max_noise = noise_mag

        #White noise = random uniform distribution
        if self.__curr_noise_type == 'white':
            self.noise_data = np.random.uniform(size = len(self.__signal_data)) * self.__max_noise

        elif self.__curr_noise_type == 'brown':
            self.noise_data = np.cumsum(np.random.uniform(size = len(self.__signal_data)))/ self.__num_samples * self.__max_noise

        elif self.__curr_noise_type == 'pink':
            pass
        else:
            print('ERROR: Unexpected Noise type detected') 

    def get_time_domain_data(self):
        return [self.__time_axis_data, self.__signal_data]

    def get_fft_domain_data(self):
        return [self.__fft_bins, self.__fft_magnitude]

    def get_freq(self):
        return self.__sig_freq

    def get_fs(self):
        return self.__fs
    
    def get_amplitude(self):
        return self.__amplitude

    def get_noise_magnitude(self):
        return self.__max_noise

    def get_window_type(self):
        return self.__curr_window_type
    
    def get_window_state(self):
        return self.__window_enable
    
    def plot_time_domain(self):

        plt.figure(self.__increment_figures())
        
        fs_text = self.__format_freq_text(self.__fs)
        f_text = self.__format_freq_text(self.__sig_freq)
        actual_amplitude = (max(self.__signal_data) - min(self.__signal_data)) / 2
        
        plt.plot(self.__time_axis_data, self.__signal_data)
        plt.title(f"Time Domain Data: Frequency: {f_text}, F$_{'S'}$: {fs_text}, Amplitude: {actual_amplitude:.4f}")

        plt.xlim(min(self.__time_axis_data), max(self.__time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')
        plt.show()

    def plot_fft(self):

        plt.figure(self.__increment_figures())

        fs_text = self.__format_freq_text(self.__fs)
        f_text = self.__format_freq_text(self.__sig_freq)
        
        plt.semilogx(self.__fft_bins, self.__fft_magnitude)
        if self.__window_enable:
            plt.title(f"FFT Plot: Frequency: {f_text}, F$_{'S'}$: {fs_text}, FFT Window: {self.__curr_noise_type.capitalize()}")
        else:
            plt.title(f"FFT Plot: Frequency: {f_text}, F$_{'S'}$: {fs_text}, FFT Window: No Window")
       
        plt.xlim(min(self.__fft_bins), max(self.__fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, 'both')
        plt.show()
        
    def plot_time_and_fft(self):
        
        #Plot Time Domain Data
        plt.figure(self.__increment_figures())

        fs_text = self.__format_freq_text(self.__fs)
        f_text = self.__format_freq_text(self.__sig_freq)
        actual_amplitude = (max(self.__signal_data) - min(self.__signal_data)) / 2
        
        plt.subplot(2,1,1)
        plt.plot(self.__time_axis_data, self.__signal_data)
        plt.title(f"Time Domain Data: Frequency: {f_text}, F$_{'S'}$: {fs_text}, Amplitude: {actual_amplitude:.4f}")
        plt.xlim(min(self.__time_axis_data), max(self.__time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')

        #Plot Frequency Domain Data
        plt.subplot(2,1,2)
        plt.semilogx(self.__fft_bins, self.__fft_magnitude)
        if self.__window_enable:
            plt.title(f"FFT Plot: Frequency: {f_text}, F$_{'S'}$: {fs_text}, FFT Window: {self.__curr_noise_type.capitalize()}")
        else:
            plt.title(f"FFT Plot: Frequency: {f_text}, F$_{'S'}$: {fs_text}, FFT Window: No Window")
       
        plt.xlim(min(self.__fft_bins), max(self.__fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, 'both')

        #Show Plot
        plt.tight_layout()
        plt.show()

def export_to_file(self, filename: str, filetype: str, data_type: str):
    pass

def import_from_file(self, filename: str):
    pass