"""
LSTM Predictive Maintenance Engine
===================================
Trains a multi-variate LSTM on equipment sensor data to predict failure
probability and estimate Remaining Useful Life (RUL).
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not installed. LSTM model will use mock predictions.")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "sensor")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "saved")
SENSOR_CSV  = os.path.join(DATA_DIR, "sensor_data.csv")
MODEL_PATH  = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "sensor_scaler.joblib")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 30          # 30 timesteps look-back
FEATURES     = ["temperature", "vibration", "pressure", "rpm", "current"]
TARGET       = "failure_in_24h"
N_MACHINES   = 10
N_TIMESTEPS  = 5000        # per machine


# ══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC SENSOR DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_sensor_data(n_machines=N_MACHINES, n_timesteps=N_TIMESTEPS, seed=42):
    """Generate realistic multi-variate sensor time-series with failure events."""
    np.random.seed(seed)
    records = []

    for machine_id in range(1, n_machines + 1):
        t = np.arange(n_timesteps)

        # Baseline sensor readings (healthy state)
        temp     = 60 + 5 * np.sin(t / 200) + np.random.normal(0, 1.5, n_timesteps)
        vibr     = 0.5 + 0.1 * np.sin(t / 150) + np.random.normal(0, 0.05, n_timesteps)
        pressure = 100 + 3 * np.cos(t / 300) + np.random.normal(0, 2, n_timesteps)
        rpm      = 1500 + 50 * np.sin(t / 250) + np.random.normal(0, 10, n_timesteps)
        current  = 10 + 0.5 * np.sin(t / 180) + np.random.normal(0, 0.3, n_timesteps)

        # Inject degradation events (3–6 failures per machine)
        failure_label = np.zeros(n_timesteps, dtype=int)
        n_failures = np.random.randint(3, 7)
        failure_times = np.sort(np.random.choice(range(200, n_timesteps - 50), n_failures, replace=False))

        for ft in failure_times:
            # 72-hour degradation ramp leading up to failure
            ramp_start = max(0, ft - 72)
            ramp = np.linspace(0, 1, ft - ramp_start)
            ramp_len = len(ramp)

            temp[ramp_start:ft]     += ramp * np.random.uniform(15, 30)
            vibr[ramp_start:ft]     += ramp * np.random.uniform(0.5, 1.5)
            pressure[ramp_start:ft] -= ramp * np.random.uniform(10, 25)
            rpm[ramp_start:ft]      -= ramp * np.random.uniform(100, 300)
            current[ramp_start:ft]  += ramp * np.random.uniform(2, 5)

            # Label: 1 if failure within next 24 hours
            label_start = max(0, ft - 24)
            failure_label[label_start:ft + 1] = 1

        # Clip to realistic ranges
        temp     = np.clip(temp, 30, 120)
        vibr     = np.clip(vibr, 0.1, 5.0)
        pressure = np.clip(pressure, 50, 200)
        rpm      = np.clip(rpm, 500, 2000)
        current  = np.clip(current, 2, 25)

        for i in range(n_timesteps):
            records.append({
                "machine_id":    machine_id,
                "timestamp":     pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
                "temperature":   round(temp[i], 2),
                "vibration":     round(vibr[i], 4),
                "pressure":      round(pressure[i], 2),
                "rpm":           round(rpm[i], 1),
                "current":       round(current[i], 3),
                "failure_in_24h": failure_label[i]
            })

    df = pd.DataFrame(records)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(SENSOR_CSV, index=False)
    print(f"[OK] Sensor data saved -> {SENSOR_CSV}  ({len(df):,} rows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — SLIDING WINDOW
# ══════════════════════════════════════════════════════════════════════════════
def create_sequences(df, window=WINDOW_SIZE):
    """Convert time-series DataFrame into (X, y) sliding windows per machine."""
    X_list, y_list = [], []
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    for machine_id in df["machine_id"].unique():
        machine_df = df[df["machine_id"] == machine_id].sort_values("timestamp")
        values = machine_df[FEATURES].values
        labels = machine_df[TARGET].values

        for i in range(window, len(values)):
            X_list.append(values[i - window:i])
            y_list.append(labels[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[OK] Scaler saved -> {SCALER_PATH}")
    print(f"[OK] Sequences: X={X.shape}, y={y.shape}, failure_rate={y.mean():.2%}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD LSTM MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_lstm(input_shape):
    """Multi-layer LSTM with Dropout for binary failure classification."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train(regenerate_data=False):
    if regenerate_data or not os.path.exists(SENSOR_CSV):
        df = generate_sensor_data()
    else:
        df = pd.read_csv(SENSOR_CSV, parse_dates=["timestamp"])
        print(f"[OK] Loaded existing sensor data ({len(df):,} rows)")

    X, y = create_sequences(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weight to handle imbalance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    class_weight = {0: 1.0, 1: pos_weight}

    model = build_lstm(input_shape=(X.shape[1], X.shape[2]))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_prob)
    print("\n" + "=" * 50)
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"]))

    model.save(MODEL_PATH)
    print(f"[OK] LSTM model saved -> {MODEL_PATH}")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE — FAILURE PROBABILITY + RUL
# ══════════════════════════════════════════════════════════════════════════════
def predict_failure(sensor_window: np.ndarray) -> dict:
    """
    Given a (WINDOW_SIZE, 5) sensor window, return:
      - failure_prob: float 0–1
      - rul_hours: estimated hours until failure
      - severity: LOW / MEDIUM / HIGH / CRITICAL
    """
    if not TF_AVAILABLE or not os.path.exists(MODEL_PATH):
        # Fallback mock prediction
        prob = float(np.random.beta(2, 5))
    else:
        scaler = joblib.load(SCALER_PATH)
        window_scaled = scaler.transform(sensor_window)
        model = load_model(MODEL_PATH)
        prob = float(model.predict(window_scaled[np.newaxis], verbose=0)[0][0])

    # Estimate RUL from failure probability
    rul_hours = max(1, int((1 - prob) * 72))

    if prob < 0.25:
        severity = "LOW"
    elif prob < 0.50:
        severity = "MEDIUM"
    elif prob < 0.75:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    return {
        "failure_probability": round(prob, 4),
        "rul_hours":           rul_hours,
        "severity":            severity
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. BATCH SCORING — ALL MACHINES (for dashboard)
# ══════════════════════════════════════════════════════════════════════════════
def score_all_machines() -> pd.DataFrame:
    """Load sensor data and return latest health score for all machines."""
    if not os.path.exists(SENSOR_CSV):
        generate_sensor_data()

    df = pd.read_csv(SENSOR_CSV, parse_dates=["timestamp"])

    if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH) and TF_AVAILABLE:
        scaler = joblib.load(SCALER_PATH)
        model  = load_model(MODEL_PATH)
        use_model = True
    else:
        use_model = False

    results = []
    for machine_id in df["machine_id"].unique():
        mdf = df[df["machine_id"] == machine_id].sort_values("timestamp")
        latest = mdf.tail(WINDOW_SIZE)[FEATURES].values

        if use_model and len(latest) == WINDOW_SIZE:
            scaled = scaler.transform(latest)
            prob = float(model.predict(scaled[np.newaxis], verbose=0)[0][0])
        else:
            # Heuristic fallback: normalize sensor deviation
            means = mdf[FEATURES].mean()
            stds  = mdf[FEATURES].std().replace(0, 1)
            z = ((mdf.tail(1)[FEATURES].values[0] - means.values) / stds.values)
            prob = float(np.clip(np.abs(z).mean() / 4, 0, 1))

        rul  = max(1, int((1 - prob) * 72))
        sev  = ["LOW", "MEDIUM", "HIGH", "CRITICAL"][min(3, int(prob * 4))]
        last = mdf.tail(1).iloc[0]

        results.append({
            "machine_id":          f"M-{machine_id:03d}",
            "failure_probability": round(prob, 4),
            "health_score":        round((1 - prob) * 100, 1),
            "rul_hours":           rul,
            "severity":            sev,
            "last_temperature":    round(last["temperature"], 1),
            "last_vibration":      round(last["vibration"], 4),
            "last_pressure":       round(last["pressure"], 1),
            "last_rpm":            round(last["rpm"], 1),
            "last_current":        round(last["current"], 3),
        })

    out = pd.DataFrame(results).sort_values("failure_probability", ascending=False)
    out_path = os.path.join(BASE_DIR, "data", "processed", "machine_health.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] Machine health scores saved -> {out_path}")
    return out


if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("TensorFlow not installed — generating data and mock scores only.")
        generate_sensor_data()
        score_all_machines()
    else:
        train(regenerate_data=True)
        score_all_machines()
