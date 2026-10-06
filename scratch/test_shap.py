import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from explainability.shap_engine import SHAPEngine

# Simulate what happens at the SHAP step
print("Training dummy model...")
n_features = 80
X_dummy = np.random.rand(10, n_features)
y_dummy = np.random.randint(0, 2, 10)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_dummy, y_dummy)

features = {f"feat_{i}": np.random.rand() for i in range(n_features)}

print("Init SHAPEngine...")
engine = SHAPEngine()
engine.model = model

print("Calling explain...")
res = engine.explain(features)
print(f"Done! {len(res)} features explained.")
