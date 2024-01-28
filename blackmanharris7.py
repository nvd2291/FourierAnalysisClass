import numpy as np

def blackmanharris7(N: int) -> np.ndarray[float]:
    """
    Generates a 7th order Blackam-Harris Window given the length
    of the current signal. Returns a numpy array containing the window data
    """
    a_coeffs = [0.27105140069342, 
    -0.43329793923448, 
    0.21812299954311, 
    -0.06592544638803, 
    0.01081174209837, 
    -0.00077658482522, 
    0.00001388721735]

    n = np.arange(0, N)

    #When k = 0, all values in the window == a_coeffs[0]
    w = np.full(N, a_coeffs[0]) 

    for k in range(1, len(a_coeffs)):
        w += a_coeffs[k] * np.cos((2 * np.pi * k * n) / N)
    
    return w