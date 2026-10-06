from datetime import datetime
import json
from pydantic import BaseModel

class Demo(BaseModel):
    timestamp: datetime

report = Demo(timestamp=datetime.now())
try:
    print(json.dumps(report.model_dump()))
except Exception as e:
    print(f"Error: {e}")

try:
    print(json.dumps(report.model_dump(mode='json')))
except Exception as e:
    print(f"Mode JSON Error: {e}")
