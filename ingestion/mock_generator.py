import numpy as np
import neurokit2 as nk
from reporting.schemas import RawECGSignal

class MockECGGenerator:
    """Generates synthetic ECG data for testing without needing external files."""
    
    @staticmethod
    def generate(duration_seconds: int = 60, sampling_rate: int = 250, heart_rate: int = 70, noise_level: float = 0.1) -> RawECGSignal:
        # Generate synthetic ECG signal
        ecg_signal = nk.ecg_simulate(
            duration=duration_seconds, 
            sampling_rate=sampling_rate, 
            heart_rate=heart_rate,
            noise=noise_level
        )
        
        return RawECGSignal(
            signal_data=ecg_signal.astype(np.float64),
            sampling_rate=float(sampling_rate),
            channel_name="Synthetic_ECG",
            file_path="mock_internal",
            duration_seconds=float(duration_seconds)
        )
