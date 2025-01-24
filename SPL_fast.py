import numpy as np
from estimate_energy_level import *
from cpp import CPP

def SPL_fast(x, Fs):
    '''
    Computes the mean of the Sound Pressure Level (SPL) using a fixed window duration
    and calibration constant. Used for processing calibration files.

    Parameters:
        x : np.ndarray
            Input audio signal
        Fs : int
            Sampling rate of x
    Returns:
        SPL_mean : float
            Mean SPL over all windows of time 
    '''
    C = 50  # default calibration constant (dB)
    time_step = 0.05  # duration of each window in seconds

    t = (1/Fs) * np.arange(len(x))  # time array
    
    # Calculate the length of each window in samples
    N = int(np.ceil(time_step * Fs))

    # Round N up to the nearest power of 2
    N = 2**int(np.ceil(np.log2(N)))

    windowStart = np.arange(0, len(x) - N, N)   # start index for each window
    SPL = np.zeros(len(windowStart))
    windowTime = t[windowStart + round((N - 1) / 2)]    # times at the middle of each window

    SPL_partial = np.zeros(len(windowStart))

    # Calculate the SPL at each window
    for i in range(len(windowStart)):
        SPL[i] = estimate_energy_level(x[windowStart[i] : windowStart[i] + N], Fs, C)
        SPL_partial[i] = SPL[i] * (windowTime[1] - windowTime[0])

    SPL_mean = np.sum(SPL_partial) / windowTime[-1]

    return SPL_mean

def SPL_fast_C_TH_CPP(x, Fs, C, time_step, f0min, f0max):
    '''
    Computes the Sound Pressure Level (SPL) and cepstral peak prominence (CPP)
    using a custom window duration and calibration constant. Used for processing the monitoring file.
    Parameters:
        x : np.ndarray
            Input audio signal
        Fs : int
            Sampling rate of x
        C : float
            Calibration Constant
        time_step : float
            Time step in seconds
        f0min : int
            Minimum frequency to search for CPP (Cepstral Peak Prominence)
        f0max : int
            Maximum frequency to search for CPP (Cepstral Peak Prominence)
    Returns:
        SPL_mean : float
            Mean SPL over all windows of time
        SPL : np.ndarray
            Array of SPL values
        windowTime:
            Time values at the center of each window
    '''
    t = (1/Fs) * np.arange(len(x))  # time array
    N = int(time_step * Fs) # length of each window in samples
    windowStart = np.arange(0, len(x)-N, N) # start index for each window
    SPL = np.zeros(len(windowStart))
    cpp = np.zeros_like(SPL)
    windowTime = t[windowStart + round((N - 1) / 2)]    # times at the middle of each window
    SPL_partial = np.zeros(len(windowStart))

    # Calculate the SPL and CPP at each window
    for i in range(len(windowStart)):
        SPL[i] = estimate_energy_level(x[windowStart[i] : windowStart[i] + N], Fs, C)
        SPL_partial[i] = SPL[i] * time_step
        cpp[i] = CPP(x[windowStart[i] : windowStart[i] + N], Fs, f0min, f0max, 2**13)

    SPL_mean = np.sum(SPL_partial) / windowTime[-1]

    return SPL_mean, SPL, cpp, windowTime