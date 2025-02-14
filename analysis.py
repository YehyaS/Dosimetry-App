from SPL_fast import *
from praat_pitch import *
from doses import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import soundfile as sf

def audioread(file):
    '''
    Custom function for reading audio files.

    Parameter:
        file : str 
            Path to the audio file
    Returns:
        Fs : int 
            The sample rate of the audio file
        x : np.ndarray 
            Audio data extracted from the channel with the highest RMS
    '''
    x, Fs = sf.read(file)

    # Check if audio is mono
    if len(x.shape) == 1:
        return Fs, x
    
    return Fs, x[:,np.argmax(np.mean(np.square(x), axis=0))]

def analysis(cal_files, cal_levels, monitoring_file, gender, save_folder=""):
    '''
    Performs acoustic analysis on calibration and monitoring files, and calculates
    sound pressure level (SPL), fundamental frequency (F0), and vocal doses, then saves
    the results in Excel files.

    Parameters:
        cal_files : list
            List of paths to calibration audio files
        cal_levels : list
            List of calibration levels for each calibration file
        monitoring_file : str
            Path to the monitoring file to be analyzed
        gender : str
            Speaker's gender (male, female, other)
        save_folder : str
            Path to the folder where the results will be stored
            Default: current directory (used for debugging purposes)
    
    Returns:
        time_audio : np.ndarray
            Time array corresponding to the monitoring data, in minutes
        audio : np.ndarray
            Audio data corresponding to the monitoring file
        time_SPL_F0 : np.ndarray
            Time array corresponding to SPL and F0 values, in seconds
        SPL : np.ndarray
            Array of calculated SPL values over time
        F0 : np.ndarray
            Array of calculated F0 values over time
        vocal_doses : pd.DataFrame
            DataFrame containing the calculated vocal doses
    '''
    # Step 1: Calibration
    calibration_constants = []
    for cal_file, cal_level in zip(cal_files, cal_levels):
        if(cal_file == ""):
            continue
        Fs, calibration = audioread(cal_file)
        SPL_mean = SPL_fast(calibration, Fs)
        c = 50+cal_level-SPL_mean
        calibration_constants.append(c)
    
    # Calculate the average calibration constant if calibration files were provided.
    # Otherwise, use the default calibration constant of 50.
    C = np.mean(calibration_constants) if len(calibration_constants) != 0 else 50

    # Step 2: Setting gender-specific F0 range
    if gender == "female":
        f0min = 100
        f0max = 400
    elif gender == "male":
        f0min = 50
        f0max = 300
    else:
        f0min = 50
        f0max = 400

    # Step 3: Monitoring File Analysis
    Fs, audio = audioread(monitoring_file)
    time_step = 0.05    # Time step in seconds
    SPL_mean, SPL, CPP, time_SPL_F0 = SPL_fast_C_TH_CPP(audio,Fs,C,time_step,f0min,f0max)

    # Step 4: Calculating F0 using Praat's algorithm
    F0 = praat_pitch(audio, Fs, time_step, f0min, f0max)

    # Step 5: Truncating time, SPL, F0 to the same length
    lim = min(len(SPL), len(F0), len(CPP))
    SPL = SPL[:lim]
    time_SPL_F0 = time_SPL_F0[:lim]
    F0 = F0[:lim]
    CPP = CPP[:lim]

    # Step 6: Adjusting SPL based on the distance to the microphone
    distance_cal = 0.30
    SPL=SPL-20*np.log(distance_cal/0.5)

    # Step 7: Filtering out small values of SPL and F0
    SPL = [0 if f < 50 else f for f in SPL]
    for i in range(len(F0)):
        if SPL[i] < 1e-10 or F0[i] < 1e-10:
            SPL[i] = 0
            F0[i] = 0
            CPP[i] = 0
    
    # Step 8: Creating the results directory
    results_directory = os.path.join(save_folder, os.path.splitext(os.path.basename(monitoring_file))[0] + "_results")
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    # Step 9: Saving SPL, F0, and CPP data to an Excel file
    results = {'Time' : time_SPL_F0, 'SPL' : SPL, 'F0' : F0, 'CPP' : CPP}
    df = pd.DataFrame(results)
    df.to_excel(os.path.join(save_folder, results_directory, "SPL_F0_CPP.xlsx"), index=False)

    # Step 10: Calulcating vocal doses and saving them to an Excel file
    vocal_doses = pd.DataFrame()
    vocal_doses.insert(0, "Doses", ['Dt', 'VLI', 'Dd', 'De', 'Dr', 'Dt_p', 'Dd_n', 'De_n', 'Dr_n', 'SPL_mean', 'F0_mean',  'SPL_sd', 'F0_sd', 'CPP'])
    vocal_doses.insert(1, "Values", doses(audio, Fs, time_SPL_F0, SPL, F0, gender, f0min, f0max, len(calibration_constants)==0))
    vocal_doses.to_excel(os.path.join(save_folder, results_directory, "Doses.xlsx"), index=False)

    # Step 11: Creating the time array corresponding to the monitoring data, in minutes
    time_audio = np.arange(len(audio))/(Fs*60)

    return time_audio, audio, time_SPL_F0, SPL, F0, CPP, vocal_doses


def display_data(time_audio, audio, time_SPL_F0, SPL, F0, CPP, vocal_doses):
    """
    Displays plots of the audio signal, SPL, F0, and vocal doses (defined in doses.py)
    in a 3x2 grid. Also calculates and displays the energetic average and moving average
    for SPL and F0.
    
    Parameters:
        time_audio : np.ndarray
            Time array corresponding to the monitoring data, in minutes
        audio : np.ndarray
            Audio data corresponding to the monitoring file
        time_SPL_F0 : np.ndarray
            Time array corresponding to SPL and F0 values, in seconds
        SPL : np.ndarray
            Array of calculated SPL values over time
        F0 : np.ndarray
            Array of calculated F0 values over time
        CPP : np.ndarray
            Array of calculated CPP values over time
        vocal_doses : pd.DataFrame
            DataFrame containing the calculated vocal doses
    """
    time_SPL_F0 = np.array(time_SPL_F0)
    SPL = np.array(SPL)
    F0 = np.array(F0)

    # Convert SPL to energy values
    energy_values = 10 ** (SPL / 10)

    # Pad with zeros
    energy_values_pad = np.concatenate((energy_values, np.zeros(1200)))
    F0_pad = np.concatenate((F0, np.zeros(1200)))

    # Initialize arrays for results
    sum_energy = np.zeros(len(SPL))
    Leq = np.zeros(len(SPL))
    F0_mean = np.zeros(len(SPL))

    # Compute the mean energy and Leq for each group
    for i in range(len(SPL)):
        # Compute the sum of energy over the window
        sum_energy[i] = (1 / 20) * np.sum(energy_values_pad[i:i + 1200])
        temp_F0 = F0_pad[i : i + 1200]
        temp_F0_filtered = temp_F0[temp_F0 != 0]

        # Convert back to dB (energetic average)
        Leq[i] = 10 * np.log10(sum_energy[i] / 60)
        F0_mean[i] = np.mean(temp_F0_filtered) if temp_F0_filtered.size != 0 else np.nan

    Leq = [np.nan if x < 50 else x for x in Leq]

   # Set all values near 0 to NaN to prevent plotting them 
    for i in range(len(SPL)):
        if SPL[i] < 1e-17 or F0[i] < 1e-17:
            SPL[i] = np.nan
            F0[i] = np.nan
            CPP[i] = np.nan

    plt.figure(figsize=(13,11))

    # Subplot 1: Table of vocal doses
    plt.subplot(421)
    plt.axis("off")
    table = plt.table(cellText=vocal_doses.values, colLabels=vocal_doses.columns, loc='center')
    table.scale(1,1.3)

    for (row, col), cell in table.get_celld().items():
        cell.set_fontsize(8)  # Adjust font size as needed
        cell.set_text_props(ha='center', va='center')  # Center the text

    # Subplot 2: Line plot of audio over time in minutes 
    plt.subplot(422)
    plt.plot(time_audio, audio)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (m)")
    plt.title("Audiowave")

    # Subplot 3: Scatter plot of SPL over time in minutes
    plt.subplot(423)
    plt.plot(time_SPL_F0/60, SPL, 'r+', alpha=0.1)
    plt.plot(time_SPL_F0/60, Leq, label='Energetic Average')
    plt.ylabel("SPL (dBA)")
    plt.xlabel("Time (m)")
    plt.title("SPL at 50 cm")
    plt.legend(loc='best')

    # Subplot 4: Histogram of SPL values
    plt.subplot(424)
    SPL_clean = SPL[~np.isnan(SPL)]
    bin_edges = np.arange(np.floor(min(SPL_clean)), np.ceil(max(SPL_clean)) + 2, 2)
    plt.hist(SPL, color="r", bins=bin_edges)#, bins=range(int(np.floor(min(~np.isnan(SPL)))), int(np.ceil(max(~np.isnan(SPL)))) + 2, 2))
    plt.ylabel("# of measurements")
    plt.xlabel("SPL (dBA)")
    plt.title("SPL at 50 cm")

    # Subplot 5: Scatter plot of F0 over time in minutes
    plt.subplot(425)
    plt.plot(time_SPL_F0/60, F0, 'c*', alpha=0.1)
    plt.plot(time_SPL_F0/60, F0_mean, label='Mean')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (m)")
    plt.title("Fundamental Frequency")
    plt.legend(loc='best')

    # Subplot 6: Histogram of F0 values
    plt.subplot(426)
    F0_clean = F0[~np.isnan(F0)]
    bin_edges = np.arange(np.floor(min(F0_clean)), np.ceil(max(F0_clean)) + 10, 10)
    plt.hist(F0, color="turquoise", bins=bin_edges)
    plt.ylabel("# of measurements")
    plt.xlabel("Frequency (Hz)")
    plt.title("Fundamental Frequency")

    # Subplot 7: Scatter plot of CPP over time in minutes
    plt.subplot(427)
    plt.plot(time_SPL_F0/60, CPP, '.', color='purple')
    plt.ylabel("CPP")
    plt.xlabel("Time (m)")
    plt.title("Cepstral Peak Prominence")

    # Subplot 8: Histogram of CPP values
    plt.subplot(428)
    CPP_clean = CPP[~np.isnan(CPP)]
    bin_edges = np.arange(np.floor(min(CPP_clean)), np.ceil(max(CPP_clean)) + 1, 1)
    plt.hist(CPP, color="purple", bins=bin_edges)
    plt.ylabel("# of measurements")
    plt.xlabel("CPP")
    plt.title("Cepstral Peak Prominence")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.075)

    # Make sure that the app pauses so that existing plot
    # window has to be closed to open a new one
    plt.show(block=True)