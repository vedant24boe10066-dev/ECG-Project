from abc import ABC, abstractmethod
from reporting.schemas import ProcessedECGSignal, HRVMetrics

class BaseHRVExtractor(ABC):
    """Abstract base class for Heart Rate Variability (HRV) feature extraction."""
    
    @abstractmethod
    def extract_features(self, processed_signal: ProcessedECGSignal) -> HRVMetrics:
        """Computes time, frequency, and non-linear domain HRV features from R-R intervals."""
        pass
