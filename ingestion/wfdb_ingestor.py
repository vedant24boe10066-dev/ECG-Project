import wfdb
import numpy as np
from reporting.schemas import RawECGSignal
from ingestion.base import BaseECGIngestor

class WFDBECGIngestor(BaseECGIngestor):
    def ingest(self, file_path: str) -> RawECGSignal:
        """
        Reads a WFDB record (requires both .hea and .dat files to be present).
        file_path should be the record name without the extension (e.g., 'data/raw/100').
        """
        # wfdb.rdrecord automatically reads both the .hea and .dat based on the base record name
        record = wfdb.rdrecord(file_path)
        
        # Typically, ECG is in the first channel (channel 0)
        # We can extract the signal data array
        signal_data = record.p_signal[:, 0].astype(np.float64)
        
        # The sampling rate is stored in the record header
        sampling_rate = float(record.fs)
        
        return RawECGSignal(
            signal_data=signal_data,
            sampling_rate=sampling_rate,
            duration_seconds=len(signal_data) / sampling_rate,
            source="WFDB"
        )
