with open("api.py", "r") as f:
    lines = f.readlines()

with open("api.py", "w") as f:
    skip = False
    for line in lines:
        if line.startswith("# Train a dummy RF model since we don't have a real saved model file"):
            skip = True
            
        if line.startswith("shap_engine ="):
            f.write("shap_engine = SHAPEngine(model=predictor.get_model())\n")
            skip = False
            continue
            
        if not skip:
            f.write(line)
