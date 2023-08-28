from fourier import FourierDataObject

fourierObject = FourierDataObject()
fourierObject.set_time(0.001, 0.005)
fourierObject.construct_square_wave_from_sines(5,1.0, 20.0e3, True, 0.005)

fourierObject.generate_freq_domain_data()

fourierObject.plot_time_and_fft()
