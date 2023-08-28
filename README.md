# FourierAnalysisClass

This python class can be used to create several different types of signals and plot the time domain and FFT signal. 

Available Signal Types:
 - Sine
 - Square
 - Sawtooth
 - Triangle

Available Noise Options:
  - White
  - Brown
  
  * To be added:
  - Pink
  - Blue
  - etc.

Available FFT Windows:
**Unless otherwise noted all FFT windows are from the SciPy Python Library**
- Bartlett
- Blackman
- Blackman-Harris 4-Term
- Blackman-Harris 7-Term (Implementation contained in *blackmanharris7.py*)
- Boxcar (rectangular)
- Flattop
- Hamming
- Hanning
- Parzen
- Triangular
- Tukey

Other Functionality:
- DC Offset Adjustment
- Noise Amplitude Adjustment
- Time Domain and FFT Plotting
- Square Wave and Triangle Wave Generation from Sine Waves

References:
- https://mathworld.wolfram.com/FourierSeriesTriangleWave.html
- https://en.wikipedia.org/wiki/Triangle_wave
