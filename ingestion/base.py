from abc import ABC, abstractmethod
from reporting.schemas import RawECGSignal

class BaseECGIngestor(ABC):
    """Abstract base class for all ECG data ingestors."""
    
    @abstractmethod
    def ingest(self, file_path: str) -> RawECGSignal:
        """Read a local ECG file and return a standardized RawECGSignal object."""
        pass
