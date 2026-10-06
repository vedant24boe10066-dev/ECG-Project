from abc import ABC, abstractmethod
from reporting.schemas import RawECGSignal, ProcessedECGSignal

class BaseECGPreprocessor(ABC):
    """Abstract base class for ECG signal filtering and R-peak detection."""
    
    @abstractmethod
    def preprocess(self, raw_signal: RawECGSignal) -> ProcessedECGSignal:
        """Filters the raw signal and extracts R-peaks and RR-intervals."""
        pass
