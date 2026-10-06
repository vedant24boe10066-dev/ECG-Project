import requests
import json

url = "http://localhost:8000/api/analyze?sampling_rate=250.0"
files = [
    ('files', ('sample_ecg.csv', open('sample_ecg.csv', 'rb'), 'text/csv'))
]

response = requests.post(url, files=files, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
