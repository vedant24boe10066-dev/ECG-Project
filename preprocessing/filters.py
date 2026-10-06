import numpy as np
import neurokit2 as nk
from preprocessing.base import BaseECGPreprocessor
from reporting.schemas import RawECGSignal, ProcessedECGSignal

class NeuroKitPreprocessor(BaseECGPreprocessor):
    """
    Standard preprocessor using NeuroKit2.
    Applies digital bandpass/notch filtering and extracts R-peaks (Pan-Tompkins/NeuroKit algorithm).
    """
    
    def __init__(self, method: str = 'neurokit'):
        self.method = method

    def preprocess(self, raw_signal: RawECGSignal) -> ProcessedECGSignal:
        if np.ptp(raw_signal.signal_data) == 0:
            raise ValueError("ECG signal is constant; no heartbeats can be detected.")
        # 1. Clean the signal (Bandpass filter usually 0.5 - 50Hz)
        cleaned_signal = nk.ecg_clean(
            raw_signal.signal_data, 
            sampling_rate=int(raw_signal.sampling_rate), 
            method=self.method
        )
        
        # 2. Detect R-peaks
        _, info = nk.ecg_peaks(
            cleaned_signal, 
            sampling_rate=int(raw_signal.sampling_rate),
            method=self.method,
            correct_artifacts=True
        )
        r_peaks = info['ECG_R_Peaks']
        if len(r_peaks) < 10:
            raise ValueError("Too few heartbeats detected for HRV analysis; provide a longer, clear ECG recording.")
        
        # 3. Calculate R-R intervals in milliseconds
        # Time difference between consecutive R-peaks
        rr_intervals = np.diff(r_peaks) / raw_signal.sampling_rate * 1000.0
        
        return ProcessedECGSignal(
            filtered_signal=cleaned_signal.astype(np.float64),
            r_peaks=np.array(r_peaks),
            rr_intervals=rr_intervals.astype(np.float64),
            sampling_rate=raw_signal.sampling_rate
        )
