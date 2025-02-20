# ECG Dataset Class definition and methods for pre-processing ECG data.

import numpy as np
from scipy import signal
from sklearn.preprocessing import scale, robust_scale

# 12-LEAD ECG LEAD NAMES
ECG_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# --------------------------------------------ECG PRE-PROCESSING-------------------------------------------------------#


def lead_selection(ecg_signal, lead_names, lead_name_selection):
    """
    Selects the leads of the ECG data according to the lead_name_selection.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        lead_names (list): List of lead names in the ECG signal.
        lead_name_selection (list): List of lead names to select.
    Returns:
        ecg_signal_selected (ndarray): Selected ECG signal.
        lead_names_selected (list): Selected list of lead names.
    """
    # Check if lead_name_selection are in lead_names
    if not set(lead_name_selection).issubset(lead_names):
        raise ValueError(f'Invalid lead names: {lead_name_selection}')

    # Get indices of lead_name_selection in lead_names
    lead_indices = [lead_names.index(lead_name) for lead_name in lead_name_selection]
    # Select leads
    ecg_signal_selected = ecg_signal[:, lead_indices]

    return ecg_signal_selected, lead_name_selection


def lead_reordering(ecg_signal, lead_names):
    """
    Reorders the leads of the ECG data according to the order of the ECG_12_LEADS.
    ECG_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        lead_names (list): List of lead names.

    Returns:
        ecg_signal_reordered (ndarray): Reordered ECG signal.
        lead_names_reordered (list): Reordered list of lead names.
    """

    # Check if lead_names are in ECG_12_LEADS
    if not set(lead_names).issubset(ECG_12_LEADS):
        raise ValueError(f'Invalid lead names. Options: {ECG_12_LEADS}')

    # Find indices of lead_names in ECG_12_LEADS and the index order to sort them
    ecg_12_lead_indices = [ECG_12_LEADS.index(lead) for lead in lead_names]
    index_order = np.argsort(ecg_12_lead_indices)

    # Reorder ecg_signal and lead_names
    ecg_signal_reordered = ecg_signal[:, index_order]
    lead_names_reordered = [lead_names[i] for i in index_order]

    return ecg_signal_reordered, lead_names_reordered


def lead_augmentation(ecg_signal, lead_names, reordering=True):
    """
    Augments the ECG data by adding leads 'III', 'aVR', 'aVL', 'aVF' to the ECG data.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        lead_names (list): List of lead names.
        reordering (bool): If True, the leads will be reordered according to the ECG_12_LEADS.

    Returns:
        ecg_signal_augmented (ndarray): Augmented ECG signal.
        lead_names_augmented (list): Augmented list of lead names.
    """

    # Check if leads 'I' and 'II' are within the lead names
    if not {'I', 'II'}.issubset(lead_names):
        raise ValueError(f'Lead names must contain leads I and II for augmentation')

    # Calculate augmented leads
    augmented_leads = {
        'III': ecg_signal[:, lead_names.index('II')] - ecg_signal[:, lead_names.index('I')],
        'aVR': - ((ecg_signal[:, lead_names.index('I')] + ecg_signal[:, lead_names.index('II')]) / 2),
        'aVL': ecg_signal[:, lead_names.index('I')] - (ecg_signal[:, lead_names.index('II')] / 2),
        'aVF': ecg_signal[:, lead_names.index('II')] - (ecg_signal[:, lead_names.index('I')] / 2)
    }

    # Create augmented ecg_signal and lead_names
    ecg_signal_augmented = ecg_signal.copy()
    lead_names_augmented = lead_names.copy()

    # Check if augmented leads are in the data, if not, add them
    for lead_name, ecg in augmented_leads.items():
        if lead_name not in lead_names:
            ecg_signal_augmented = np.concatenate((ecg_signal_augmented, np.expand_dims(ecg, axis=-1)), axis=-1)
            lead_names_augmented.append(lead_name)

    # Reorder ecg_signal and lead_names
    if reordering:
        ecg_signal_augmented, lead_names_augmented = lead_reordering(ecg_signal_augmented, lead_names_augmented)

    return ecg_signal_augmented, lead_names_augmented


def baseline_correction(ecg_signal, sampling_rate, cutoff_freq=0.8):
    """
    Removes the baseline wander from the ECG signal using an IIR elliptic high-pass filter at cutoff_freq Hz.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal.
        cutoff_freq (float): Cutoff frequency of the high-pass filter in Hz.

    Returns:
        ecg_signal_no_baseline (ndarray): ECG signal with no baseline wander.
    """

    wn = cutoff_freq / (sampling_rate / 2)  # Passband edge frequency
    wst = 0.2 / (sampling_rate / 2)  # Stopband edge frequency
    rp = 0.5  # ripple in passband (dB)
    rs = 40  # attenuation in rejection band (dB)

    # Return the order of the lowest order digital or analog elliptic filter that loses no more than rp dB in the
    # passband and has at least rs dB attenuation in the stopband.
    filter_order, _ = signal.ellipord(wn, wst, rp, rs)

    # IIR digital elliptic high-pass filter of order 'filter_order'.
    sos = signal.iirfilter(filter_order, wn, rp, rs, btype='highpass', analog=False, ftype='ellip', output='sos')

    # A forward-backward digital filter using cascaded second-order sections.
    ecg_signal_no_baseline = signal.sosfiltfilt(sos, ecg_signal, axis=0, padtype='constant')

    return ecg_signal_no_baseline


# IMPLEMENT FILTERING AND POWER LINE INTERFERENCE REMOVAL

def bandpass_filtering(ecg_signal, sampling_rate, lowcut=None, highcut=None, order=3):
    """
    Applies a bandpass IIR butterworth filter to the ECG signal.
    The Gustafsson's method is used to determine the initial staes of forward-backward filtering (Fustafsson, 1996).

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal.
        lowcut (float or None): Low cutoff frequency in Hz. If None, a low-pass filter is applied.
        highcut (float or None): High cutoff frequency in Hz. If None, a high-pass filter is applied.
        order (int): Order of the bandpass filter.

    Returns:
        ecg_signal_filtered (ndarray): Filtered ECG signal.
    """
    if lowcut is None and highcut is None:
        raise ValueError('At least one of lowcut or highcut must be specified')

    # Lowpass filter
    if lowcut is None:
        wn = highcut / (sampling_rate / 2)
        ba = signal.iirfilter(order, wn, btype='lowpass', analog=False, ftype='butter', output='ba')
    # Highpass filter
    elif highcut is None:
        wn = lowcut / (sampling_rate / 2)
        ba = signal.iirfilter(order, wn, btype='highpass', analog=False, ftype='butter', output='ba')
    # Bandpass filter
    else:
        wn = [lowcut / (sampling_rate / 2), highcut / (sampling_rate / 2)]
        ba = signal.iirfilter(order, wn, btype='bandpass', analog=False, ftype='butter', output='ba')

    ecg_signal_filtered = signal.filtfilt(ba[0], ba[1], ecg_signal, axis=0, method='gust')

    return ecg_signal_filtered


def notch_filtering(ecg_signal, sampling_rate, powerline_freq=50, harmonics=True):
    """
    Applies a notch filter to the ECG signal to remove the powerline interference.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal.
        powerline_freq (int or None): Powerline frequency in Hz.
        harmonics (bool): If True, also remove the harmonics of the powerline frequency.

    Returns:
        ecg_signal_nopowerline (ndarray): ECG signal with no powerline interference.
    """
    q = 30.0  # Quality factor
    b, a = signal.iirnotch(powerline_freq, q, fs=sampling_rate)
    ecg_signal_nopowerline = signal.filtfilt(b, a, ecg_signal, axis=0, padtype='constant')

    # Remove the harmonics of the powerline frequency
    if harmonics:
        highest_harmonic_order = int(np.floor((sampling_rate / 2) / powerline_freq))
        for harmonic in range(2, highest_harmonic_order + 1):
            b, a = signal.iirnotch(harmonic * powerline_freq, q, fs=sampling_rate)
            ecg_signal_nopowerline = signal.filtfilt(b, a, ecg_signal_nopowerline, axis=0, padtype='constant')

    return ecg_signal_nopowerline


def resampling(ecg_signal, sampling_rate, new_sampling_rate, method='polyphase'):
    """
    Resamples the ECG signal to the new sampling rate. 'fourier' method uses the scipy.signal.resample function,
    while 'polyphase' uses the scipy.signal.resample_poly function.
    'polyphase' is preferable for small downsampling ratios, as well as to prevent aliasing effects.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal.
        new_sampling_rate (int): New sampling rate of the ECG signal.
        method (str): Resampling method. Can be 'fourier' or 'polyphase'.

    Returns:
        ecg_signal_resampled (ndarray): Resampled ECG signal.
    """

    # Check if method is valid
    if method not in ['fourier', 'polyphase']:
        raise ValueError(f'Invalid resampling method {method}')

    ecg_signal_resampled = None
    # Calculate new number of samples
    new_num_samples = int(ecg_signal.shape[0] * (new_sampling_rate / sampling_rate))

    # Resample ecg_signal
    if method == 'fourier':
        ecg_signal_resampled = signal.resample(ecg_signal, new_num_samples, axis=0)
    elif method == 'polyphase':
        ecg_signal_resampled = signal.resample_poly(ecg_signal, new_num_samples, ecg_signal.shape[0], axis=0)

    return ecg_signal_resampled


def zero_padding(ecg_signal):
    """
    Pads the ECG signal with zeros symmetrically so that num_samples gets to a power of two.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).

    Returns:
        ecg_signal_padded (ndarray): Padded ECG signal.
    """
    # Find next power of two from ECG signal length and calculate padding width
    next_power_of_two = int(2 ** np.ceil(np.log2(ecg_signal.shape[0])))

    # Find padding length
    pad_len = next_power_of_two - ecg_signal.shape[0]
    pad_len_left = int(np.floor(pad_len / 2))
    pad_len_right = int(np.ceil(pad_len / 2))

    ecg_signal_padded = np.pad(ecg_signal, ((pad_len_left, pad_len_right), (0, 0)), mode='constant', constant_values=0)

    return ecg_signal_padded


def normalization(ecg_signal, mode='instance-wise', method='StandardScaler', centering=True):
    """
    Normalizes the ECG signal based on the specified method. Instance-wise normalization normalizes the ECG signal
    across all leads and samples. Lead-wise normalization normalizes the ECG signal across all samples of each lead.

    Arguments:
        ecg_signal (ndarray): ECG data of shape (num_samples, num_leads).
        mode (str): Normalization mode. Options: 'instance-wise', 'lead-wise'.
        method (str): Normalization method. Options: 'StandardScaler', 'RobustScaler'.
        centering (bool): If True, the data will be centered.

    Returns:
        ecg_signal_norm (ndarray): Normalized ECG signal.
    """
    # Check Mode
    if mode not in ['instance-wise', 'lead-wise']:
        raise ValueError('Invalid mode. Options: instance-wise, lead-wise')
    # Check Method
    if method not in ['StandardScaler', 'RobustScaler']:
        raise ValueError('Invalid method. Options: StandardScaler, RobustScaler')

    ecg_signal_norm = None

    # Instance-wise normalization
    if mode == 'instance-wise':
        ecg_flatten = ecg_signal.flatten()
        if method == 'StandardScaler':
            ecg_flatten = scale(ecg_flatten.astype('float64'), axis=0, with_mean=centering)
        if method == 'RobustScaler':
            ecg_flatten = robust_scale(ecg_flatten.astype('float64'), axis=0, with_centering=centering,
                                       quantile_range=(25, 75))
        ecg_signal_norm = np.reshape(ecg_flatten, newshape=ecg_signal.shape)

    # Lead-wise normalization
    elif mode == 'lead-wise':
        if method == 'StandardScaler':
            ecg_signal_norm = scale(ecg_signal.astype('float64'), axis=0, with_mean=centering)
        if method == 'RobustScaler':
            ecg_signal_norm = robust_scale(ecg_signal.astype('float64'), axis=0, with_centering=centering,
                                           quantile_range=(25, 75))

    return ecg_signal_norm
