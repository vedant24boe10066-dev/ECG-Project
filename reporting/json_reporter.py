import json
import os
from reporting.schemas import ECGStressReport

class JSONReportGenerator:
    """Serializes the ECGStressReport object into a structured offline JSON file."""
    
    @staticmethod
    def save_report(report: ECGStressReport, output_dir: str = "data/processed") -> str:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{report.report_id}.json")
        
        with open(file_path, 'w') as f:
            f.write(report.model_dump_json(indent=4))
            
        return file_path
