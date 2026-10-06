import numpy as np
import pandas as pd
from typing import Optional
from ingestion.base import BaseECGIngestor
from reporting.schemas import RawECGSignal

class CSVECGIngestor(BaseECGIngestor):
    def __init__(self, sampling_rate: float, column_name: str = "ECG"):
        self.sampling_rate = sampling_rate
        self.column_name = column_name

    def ingest(self, file_path: str) -> RawECGSignal:
        df = pd.read_csv(file_path)
        if self.column_name not in df.columns:
            raise ValueError(f"CSV must have a '{self.column_name}' column with one raw ECG sample per row. HRV tables and rows of separate recordings are not supported.")
        signal_data = pd.to_numeric(df[self.column_name], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive and finite.")
        duration = len(signal_data) / self.sampling_rate
        
        return RawECGSignal(
            signal_data=signal_data,
            sampling_rate=self.sampling_rate,
            channel_name=self.column_name,
            file_path=file_path,
            duration_seconds=duration
        )
