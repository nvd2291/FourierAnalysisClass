from fourier import FourierDataObject

fourierObject = FourierDataObject()

print(fourierObject)
fourierObject.select_signal('sine')
fourierObject.signal_frequency = 1000
fourierObject.sample_frequency = 2500000
fourierObject.end_time = 0.01
fourierObject.generate_time_domain_data()
fourierObject.generate_freq_domain_data()

# fourierObject.plot_time_domain()
fourierObject.plot_fft()
