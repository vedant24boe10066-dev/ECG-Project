import pandas as pd
import neurokit2 as nk
from features.hrv_extractor import NeuroKitHRVExtractor
from preprocessing.filters import NeuroKitPreprocessor
from ingestion.csv_ingestor import CSVECGIngestor

ingestor = CSVECGIngestor(sampling_rate=250.0)
raw = ingestor.ingest("sample_ecg.csv")
prep = NeuroKitPreprocessor()
proc = prep.preprocess(raw)

ext = NeuroKitHRVExtractor()
hrv = ext.extract_features(proc)
print("Freq Domain:", hrv.frequency_domain)
