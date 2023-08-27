from fourier import FourierDataObject

fourierObject = FourierDataObject()

print(fourierObject)
fourierObject.signal_frequency = 1000
fourierObject.sample_frequency = 2500000
fourierObject.end_time = 0.01
fourierObject.generate_time_domain_data()
fourierObject.generate_freq_domain_data()
fourierObject.construct_triangle_wave_from_sines(15)

fourierObject.noise_enable = True
fourierObject.max_noise = 0.4
fourierObject.generate_noise_data("white")
fourierObject.select_window("blackmanharris7")
fourierObject.enable_window()
fourierObject.generate_freq_domain_data()
# fourierObject.plot_time_domain()

fourierObject.plot_time_and_fft()
