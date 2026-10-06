import pyedflib
import numpy as np
from reporting.schemas import RawECGSignal
from ingestion.base import BaseECGIngestor

class EDFECGIngestor(BaseECGIngestor):
    def ingest(self, file_path: str) -> RawECGSignal:
        """
        Reads a single .edf file.
        """
        f = pyedflib.EdfReader(file_path)
        
        # We assume the first signal is the ECG or the relevant channel
        # A more advanced version could search for 'ECG' in f.getSignalLabels()
        channel_to_read = 0
        labels = f.getSignalLabels()
        for i, label in enumerate(labels):
            if 'ecg' in label.lower() or 'ekg' in label.lower():
                channel_to_read = i
                break
                
        signal_data = f.readSignal(channel_to_read).astype(np.float64)
        sampling_rate = float(f.getSampleFrequency(channel_to_read))
        
        f._close()
        
        return RawECGSignal(
            signal_data=signal_data,
            sampling_rate=sampling_rate,
            duration_seconds=len(signal_data) / sampling_rate,
            source="EDF"
        )
