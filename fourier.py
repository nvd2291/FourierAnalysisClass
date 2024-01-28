'''
Author: Naveed Naeem
Title: fourier.py
Purpose: Provides the capability to easily plot and display different types of
signal types along with their corresponding FFT plots.
Last Modified: 01/28/2024 
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

    def __init__(self, signal_frequency = 1.0e3, 
                 fs= 1.0e6, amplitude = 1.0, 
                 duty_cycle = 0.5, dc_offset = 0.0):

        """
        The Fourier Class is used to generate time and frequency domain data 
        based on user inputs.

        Default Values:
            - Signal Type: Sine Wave
            - Signal Frequency: 1kHz
            - Sampling Frequency: 1MHz
            - Amplitude (Unitless): 1
            - Duty Cycle: 50% (Range is from 0-1)
            - DC Offset (Unitless): 0
            - FFT Window: Rectangular(Flat-top)
            - Noise: Disabled
            - Windowing: Disabled
        """

        if duty_cycle < 0 or duty_cycle > 1:
            duty_cycle = 0.5
        if signal_frequency < 0:
            signal_frequency = abs(signal_frequency)
        if fs < 0:
            fs = abs(fs)

        self.__duty_cycle = duty_cycle
        self.__amplitude = amplitude
        self.__dc_offset = dc_offset
        self.__sig_freq = signal_frequency
        self.__fs = fs 

        self.__curr_sig_type =  self.__signal_types[0]
        self.__curr_noise_type = self.__noise_types[0]
        self.__curr_window_type = self.__window_types[4]
        self.__start_time = 0.0
        self.__end_time = 0.10
        self.__max_noise = 0.1
        self.__noise_enable = False
        self.__window_enable = False
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
        self.__sig_data = np.zeros(len(self.__time_axis_data))
        self.__fft_data_raw = []
        self.__fft_magnitude = []
        self.__fft_bins = []

        self.generate_time_domain_data()
        self.generate_freq_domain_data()
  
    def __repr__(self) -> str:
        return self.__name

    def __format_freq_text(self, freq: float) -> str:
        """ 
        Formats the input frequency in a string with the appropriate order
        of magnitude and returns
        """
        if freq < 1.0e3:
            return f"{freq:.4f}Hz"
        elif freq >= 1.0e3 and freq < 1.0e6:
            return f"{(freq/1.0e3):.4f}kHz"
        elif freq >= 1.0e6 and freq < 1.0e9:
            return f"{(freq/1.0e6):.4f}MHz"
        else:
            return f"{(freq/1.0e9):.4f}GHz"

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

                    
    def set_time(self, start_time: float, stop_time: float):
        """
        Sets the start and stop time of the signal in seconds.
        """
        self.__start_time = start_time
        self.__end_time =  stop_time

    def equivalent_noise_bandwidth(self, window: np.ndarray, dB = False):
        """
        Calculate the ENBW based on the class instances window data or user
        provided window. The default output is in bins but can also output
        the ENBW correction in dB by setting dB = True in the argument list

        Parameters:
        - window: (Optional) A FFT window in the form of a numpy array. If no
        input window is provided the ENBW will be calculated by the class
        instances current window array
        - dB (bool): (Optional): False: ENBW in bins, True: ENBW in dB
        """
        if window is None:
            window = self.__window_data

        enbw = len(window) * (np.sum(window ** 2)/ np.sum(window) ** 2)
        if not dB:
            self.__enbw = enbw
        else:
            self.__enbw = 10 * log10(enbw)
    
    def coherent_power_gain(self, 
                            window = None, 
                            dB = False):
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
    
    def set_noise_type(self, 
                       noise_type: str, 
                       noise_mag: Optional[float] = None):
        """
        Set the current noise type.

        Parameters:
            - noise_type: "white", "brown", "pink"
            - noise_mag: (Optional) Provide a noise magnitude
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

    def set_signal_type(self, 
                        signal_type: str, 
                        sawtooth_type: Optional[float] = 'left',
                        update_data: Optional[bool] = False):
        """
        Update the signal type to the input signal if it exists

        Parameters:
            - signal_type (string): Use the class method get_signal_types to
            see the available signal types
            - sawtooth_type (string): (Optional) Only needed if generating a 
            sawtooth. "left" or "right" are valid options.
            - update_data (bool): (Optional) True is provided the new signal
            immediately. Recommended to leave as False if multiple parameters
            need to be changed
        """

        signal_type = signal_type.lower()
        if signal_type not in self.__signal_types:
            print('ERROR: Unexpected Input')
            print(f"The following are signal types are accepted inputs: {self.__signal_types}")
        else:
            self.__curr_sig_type = signal_type
            if self.__curr_sig_type == self.__signal_types[2]:
                self.__sawtooth_type = self.__sawtooth_types[sawtooth_type]

            if update_data:
                self.generate_time_domain_data()


    def set_window_type(self, window_type: str):
        """
        Change the window type and regenerate the window.

        Parameters:
            - window_type (string): The desired window type.
            Call the class method get_window_types to see all the available
            window types.
        """
        window_type = window_type.lower()
        if window_type not in self.__window_types:
            print('ERROR: Unexpected Input')
            print(f"The following are FFT windows are accepted inputs: {self.__window_types}")
        else:
            self.__curr_window_type = window_type
            self.fft_window_data()
    

    def enable_window(self):
        """Enable FFT Windowing. Note: Doesn't regnerate data"""
        self.__window_enable = True

    def disable_window(self):
        """Disable FFT Windowing. Note: Doesn't regnerate data"""
        self.__window_enable = False

    def enable_noise(self):
        """Enable Noise. Note: Doesn't regnerate data"""
        self.__noise_enable = True

    def disable_noise(self):
        """Disable Noise. Note: Doesn't regnerate data"""
        self.__noise_enable  = False

    def set_amplitude(self, amplitude: float):
        if self.__sig_data is None:
            self.__amplitude = amplitude
        else:
            self.__sig_data /= self.__amplitude
            self.__amplitude = amplitude
            self.__sig_data *= amplitude

    def set_offset(self, offset: float):
        if self.__sig_data is None:
            self.__dc_offset = offset
        else:
            self.__sig_data -= self.__dc_offset
            self.__dc_offset = offset
            self.__sig_data += offset
    
    def set_frequency(self, freq):
        if freq > 0:
            self.__sig_freq = freq
    
    def generate_time_axis(self):
        """
        Generates the time axis vector based on the start time, end time, and
        number of samples
        """
        self.__time_axis_data = np.linspace(self.__start_time, 
                                            self.__end_time, 
                                            self.__num_samples)

    def generate_time_domain_data(self, signal_type: Optional[str] = None):
        """
        This method will generate the time domain data base on the current 
        signal configuration
        """
        if signal_type is not None:
            self.set_signal_type(signal_type)

        self.calc_sample_period()
        self.calc_num_samples()

        self.generate_time_axis()

        if self.__curr_sig_type == 'sine':
            yAxis = np.sin(2 * np.pi * self.__sig_freq * self.__time_axis_data)
            self.__sig_data = yAxis * self.__amplitude + self.__dc_offset
    
        elif self.__curr_sig_type =='square' :
            yAxis = signal.square(2 * np.pi * self.__sig_freq * self.__time_axis_data, self.__duty_cycle)
            self.__sig_data = yAxis * self.__amplitude + self.__dc_offset

        elif self.__curr_sig_type == 'sawtooth':
            yAxis = signal.sawtooth(2 * np.pi * self.__sig_freq * self.__time_axis_data, self.__sawtooth_type)
            self.__sig_data = yAxis * self.__amplitude + self.__dc_offset

        elif self.__curr_sig_type == 'triangle':
            yAxis = signal.sawtooth(2 * np.pi * self.__sig_freq * self.__time_axis_data, self.__duty_cycle)
            self.__sig_data = yAxis * self.__amplitude + self.__dc_offset
            
        if self.__noise_enable == True:
            self.generate_noise_data()
            self.__sig_data += self.noise_data
            
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
        Nothing. Updates the class instance's data members with the newly generated data
        """
        
        if amplitude is not None:
            self.__amplitude = amplitude
        if freq is not None:
            self.__sig_freq = freq
        if fs is not None:
            self.__sampling_freq = fs
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

        self.__sig_data = (sq_wave * self.__amplitude) + self.__dc_offset
        
        if self.__noise_enable == True:
            self.generate_noise_data()
            self.__sig_data += self.noise_data
        
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
        Nothing. Updates the class instance's data members with the newly generated data
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

        self.__sig_data = triangle_wave + self.__dc_offset


        if self.__noise_enable:
            self.generate_noise_data()
            self.__sig_data += self.noise_data

        # Regenerate the FFT data with the new signal
        self.generate_freq_domain_data()

    def fft_window_data(self):
        """
        Generates the Window Vector that's used to compute the windowed
        FFT
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

    def generate_freq_domain_data(self, is_windowed: Optional[bool] = False):
        """
        Generates the FFT data. 

        Parameters:
            - is_windowed (bool): If set to True the method will generate the 
            FFT with apply the windowing function to it
        """
        if is_windowed:
            self.__window_enable = is_windowed

        # Calculate the size of the frequency bins
        self.fft_bin_size = (self.__fs/self.__num_samples)
        
        if self.__window_enable:
            self.fft_window_data()
            self.equivalent_noise_bandwidth()
            self.coherent_power_gain()
            scaling_factor = 1/(self.__cpg / sqrt(self.__enbw))

            windowed_signal = self.__window_data * self.__sig_data
            fft_data = np.absolute(fft(windowed_signal)/self.__num_samples) * scaling_factor

        else:
        # Two-Sided FFT data
            fft_data = fft(self.__sig_data)/self.__num_samples

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
        """
        This method generates data based on either the user provided inputs or 
        the currently selected noise options

        Parameters:
            - noise_type
        """
        if noise_type is not None:
            self.set_noise_type(noise_type)
        if noise_mag is not None:
            self.__max_noise = noise_mag

        #White noise = random uniform distribution
        rand_noise = np.random.uniform(size = len(self.__sig_data))
        if self.__curr_noise_type == 'white':
            self.noise_data = (rand_noise * self.__max_noise)

        elif self.__curr_noise_type == 'brown':
            self.noise_data = (np.cumsum(rand_noise)
                              / self.__num_samples * self.__max_noise)

        elif self.__curr_noise_type == 'pink':
            pass
        else:
            print('ERROR: Unexpected Noise type detected') 

    def get_time_domain_data(self):
        """
        Returns a 2-Dimensional Array where the first column is the time axis 
        and the second column is the signal data 
        """
        return [self.__time_axis_data, self.__sig_data]

    def get_fft_domain_data(self):
        """
        Returns a 2-Dimensional Array where the first column is the fft bins 
        and the second column is the fft magnitude data
        """
        return [self.__fft_bins, self.__fft_magnitude]

    def get_freq(self):
        """ Returns the current frequency of the generated signal """
        return self.__sig_freq

    def get_fs(self):
        """ Returns the current sample frequency of the generated signal"""
        return self.__fs
    
    def get_amplitude(self):
        """ Returns the amplitude """
        return self.__amplitude

    def get_noise_magnitude(self):
        """ Returns the noise magnitude """
        return self.__max_noise

    def get_window_type(self):
        """ Returns a current window type """
        return self.__curr_window_type
    
    def get_window_state(self):
        """ Returns a bool indicating whether windowing is enabled/disabled"""
        return self.__window_enable
    
    def plot_time_domain(self):
        """
        Plots the current signal in the Time Domain.
        """
        plt.figure()
        
        fs_text = self.__format_freq_text(self.__fs)
        f_text = self.__format_freq_text(self.__sig_freq)
        actual_amplitude = (max(self.__sig_data) - min(self.__sig_data)) / 2

        plt.plot(self.__time_axis_data, self.__sig_data)
        plt.title((f"Time Domain Data: Frequency: {f_text}," 
                  f"F$_{'S'}$: {fs_text}, "
                  f"Amplitude: {actual_amplitude:.4f}"))

        plt.xlim(min(self.__time_axis_data), max(self.__time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')
        plt.show()

    def plot_fft(self):
        """ 
        Plots the current signal in the frequency domain
        """
        plt.figure()

        fs_text = self.__format_freq_text(self.__fs)
        f_text = self.__format_freq_text(self.__sig_freq)
        
        plt.semilogx(self.__fft_bins, self.__fft_magnitude)
        if self.__window_enable:
            plt.title(f"FFT Plot: Frequency: {f_text}, "
                      f"F$_{'S'}$: {fs_text}, "
                      f"FFT Window: {self.__curr_noise_type.capitalize()}")
        else:
            plt.title(f"FFT Plot: Frequency: {f_text}, "
                      f"F$_{'S'}$: {fs_text}, "
                      f"FFT Window: No Window")
       
        plt.xlim(min(self.__fft_bins), max(self.__fft_bins))
        plt.ylabel('Magnitude [dBFS]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True, 'both')
        plt.show()
        
    def plot_time_and_fft(self):
        """
        Plots the data of tuhe current signal in the time and the frequency domain
        """
        #Plot Time Domain Data
        plt.figure()

        fs_text = self.__format_freq_text(self.__fs)
        f_text = self.__format_freq_text(self.__sig_freq)
        actual_amplitude = (max(self.__sig_data) - min(self.__sig_data)) / 2
        
        plt.subplot(2,1,1)
        plt.plot(self.__time_axis_data, self.__sig_data)
        plt.title((f"Time Domain Data: Frequency: {f_text}," 
                  f"F$_{'S'}$: {fs_text}, "
                  f"Amplitude: {actual_amplitude:.4f}"))
        
        plt.xlim(min(self.__time_axis_data), max(self.__time_axis_data))
        plt.ylabel('Units')
        plt.xlabel('Seconds')
        plt.grid(True, 'both')

        #Plot Frequency Domain Data
        plt.subplot(2,1,2)
        plt.semilogx(self.__fft_bins, self.__fft_magnitude)
        if self.__window_enable:
            plt.title(f"FFT Plot: Frequency: {f_text}, "
                      f"F$_{'S'}$: {fs_text}, "
                      f"FFT Window: {self.__curr_noise_type.capitalize()}")
        else:
            plt.title(f"FFT Plot: Frequency: {f_text}, "
                      f"F$_{'S'}$: {fs_text}, "
                      f"FFT Window: No Window")
       
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