import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  AI Supply Chain Platform -- Setup")
print("=" * 55)

print("\n[1/2] Generating sensor data & machine health...")
from models.lstm_maintenance import generate_sensor_data, score_all_machines
generate_sensor_data()
score_all_machines()
print("      [DONE]")

print("\n[2/2] Testing NLP reporter...")
try:
    import pandas as pd, numpy as np
    from models.nlp_reporter import build_daily_briefing
    h = pd.DataFrame([
        {"machine_id":"M-001","failure_probability":.78,"health_score":22,"rul_hours":12,"severity":"CRITICAL"},
        {"machine_id":"M-002","failure_probability":.31,"health_score":69,"rul_hours":48,"severity":"MEDIUM"},
    ])
    s = pd.DataFrame({"Predicted_Delay": np.random.choice([0,1],200,p=[.65,.35])})
    build_daily_briefing(h, s)
    print("      [DONE]")
except Exception as e:
    print(f"      [WARN] {e}")

print("\n" + "=" * 55)
print("  Setup complete!")
print("  Run: py -m streamlit run C:\\SUPPLY_CHAIN\\dashboard\\app.py")
print("=" * 55)
