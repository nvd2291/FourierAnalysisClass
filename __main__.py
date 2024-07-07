from fourier import Fourier

def main():
    print(Fourier.get_signal_types())
    myFSquare = Fourier()
    myFSquare.construct_square_wave_from_sines(harmonics = 10, freq = 440, fs = 44.1e3, with_noise = True, noise_mag = 0.01)
    myFSquare.plot_time_and_fft()

if __name__ == "__main__":
    main()