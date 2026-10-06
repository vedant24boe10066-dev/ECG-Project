import numpy as np
import neurokit2 as nk
from features.base import BaseHRVExtractor
from reporting.schemas import ProcessedECGSignal, HRVMetrics

class NeuroKitHRVExtractor(BaseHRVExtractor):
    """
    Extracts HRV metrics using NeuroKit2.
    """
    
    def extract_features(self, processed_signal: ProcessedECGSignal) -> HRVMetrics:
        # NeuroKit requires just the R-peaks array to compute HRV features.
        # Ensure R-peaks are valid indices before passing to hrv
        r_peaks = processed_signal.r_peaks
        if (len(r_peaks) < 10 or not np.isfinite(r_peaks).all()
                or not np.equal(r_peaks, np.floor(r_peaks)).all()
                or np.any(np.diff(r_peaks) <= 0) or r_peaks[0] < 0
                or r_peaks[-1] >= len(processed_signal.filtered_signal)):
            raise ValueError("HRV requires at least 10 valid, increasing R-peak indices.")
        if not np.isfinite(processed_signal.sampling_rate) or processed_signal.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive and finite.")
        r_peaks = r_peaks.astype(int)
        
        # Calculate HRV across all domains using NeuroKit2
        hrv_results = nk.hrv(r_peaks, sampling_rate=int(processed_signal.sampling_rate))
        
        # Map Time Domain Features
        time_domain = {
            "SDNN": float(hrv_results["HRV_SDNN"].values[0]) if "HRV_SDNN" in hrv_results else None,
            "RMSSD": float(hrv_results["HRV_RMSSD"].values[0]) if "HRV_RMSSD" in hrv_results else None,
            "pNN50": float(hrv_results["HRV_pNN50"].values[0]) if "HRV_pNN50" in hrv_results else None,
            "MeanRR": float(hrv_results["HRV_MeanNN"].values[0]) if "HRV_MeanNN" in hrv_results else None,
        }
        
        # Map Frequency Domain Features
        freq_domain = {
            "LF": float(hrv_results["HRV_LF"].values[0]) if "HRV_LF" in hrv_results else None,
            "HF": float(hrv_results["HRV_HF"].values[0]) if "HRV_HF" in hrv_results else None,
            "LF_HF_Ratio": float(hrv_results["HRV_LFHF"].values[0]) if "HRV_LFHF" in hrv_results else None,
        }
        
        # Map Non-Linear Features
        nonlinear = {
            "SD1": float(hrv_results["HRV_SD1"].values[0]) if "HRV_SD1" in hrv_results else None,
            "SD2": float(hrv_results["HRV_SD2"].values[0]) if "HRV_SD2" in hrv_results else None,
            "SampEn": float(hrv_results["HRV_SampEn"].values[0]) if "HRV_SampEn" in hrv_results else None,
        }
        
        # Flattened feature vector for ML
        feature_vector = {**time_domain, **freq_domain, **nonlinear}
        
        warnings = []
        for k, v in feature_vector.items():
            if v is None or not np.isfinite(v):
                feature_vector[k] = None
                for domain in (time_domain, freq_domain, nonlinear):
                    if k in domain:
                        domain[k] = None
                warnings.append(f"{k} is unavailable for this recording; it is not a measured zero.")

        return HRVMetrics(
            warnings=warnings,
            time_domain=time_domain,
            frequency_domain=freq_domain,
            nonlinear=nonlinear,
            feature_vector=feature_vector
        )
