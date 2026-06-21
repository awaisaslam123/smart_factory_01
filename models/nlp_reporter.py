"""
Hugging Face NLP Report Generator
====================================
Converts raw analytics DataFrames into natural language executive summaries
using google/flan-t5-base (local, no API key needed, runs offline).
"""

import os
import json
import textwrap
from datetime import datetime

import pandas as pd
import numpy as np

# ── NLP Backend ───────────────────────────────────────────────────────────────
try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARN] transformers not installed. NLP reporter will use template mode.")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_NAME = "google/flan-t5-base"   # Free, runs locally, ~250 MB


# ══════════════════════════════════════════════════════════════════════════════
# 1. LAZY MODEL LOADER (cached in session)
# ══════════════════════════════════════════════════════════════════════════════
_generator = None

def _get_generator():
    global _generator
    if _generator is None and HF_AVAILABLE:
        try:
            print(f"[NLP] Loading {MODEL_NAME} (first run downloads ~250 MB)...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            _generator = hf_pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
                do_sample=False
            )
            print("[NLP] Model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load HF model: {e}. Using template fallback.")
            _generator = None
    return _generator


def _generate_text(prompt: str) -> str:
    """Generate text via HF model or fall back to template."""
    gen = _get_generator()
    if gen is not None:
        try:
            result = gen(prompt, max_new_tokens=200)[0]["generated_text"]
            return result.strip()
        except Exception as e:
            print(f"[WARN] HF generation failed: {e}")
    # Template fallback — returns a clean string directly
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. REPORT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_maintenance_report(health_df: pd.DataFrame) -> str:
    """Generate executive maintenance briefing from machine health scores."""
    if health_df.empty:
        return "No machine health data available."

    critical = health_df[health_df["severity"] == "CRITICAL"]
    high     = health_df[health_df["severity"] == "HIGH"]
    avg_health = health_df["health_score"].mean()
    worst = health_df.iloc[0]

    # Build structured prompt
    prompt = f"""Summarize this industrial maintenance report in 3 sentences for the Lahore Textile Plant manager:
- Total machines monitored: {len(health_df)}
- Average health score: {avg_health:.1f}/100
- Critical machines requiring immediate attention: {len(critical)} ({', '.join(critical['machine_id'].tolist()) if not critical.empty else 'None'})
- High-risk machines: {len(high)}
- Most at-risk machine: {worst['machine_id']} with failure probability {worst['failure_probability']:.1%} and {worst['rul_hours']} hours remaining useful life.
Write a professional 3-sentence executive summary with specific recommended actions."""

    ai_text = _generate_text(prompt)

    if ai_text:
        return ai_text
    else:
        # Template fallback
        status = "⚠️ ALERT" if len(critical) > 0 else "✅ STABLE"
        lines = [
            f"**Fleet Status: {status}**",
            f"",
            f"As of {datetime.now().strftime('%Y-%m-%d %H:%M')}, the plant is monitoring **{len(health_df)} machines** "
            f"with an average health score of **{avg_health:.1f}/100**.",
        ]
        if not critical.empty:
            lines.append(f"**{len(critical)} machine(s) are in CRITICAL condition** ({', '.join(critical['machine_id'].tolist())}) "
                         f"and require **immediate maintenance intervention** within the next {critical['rul_hours'].min()} hours.")
        if not high.empty:
            lines.append(f"An additional **{len(high)} machine(s)** are classified as HIGH risk and should be scheduled for inspection within 24–48 hours.")

        lines.append(f"The most at-risk unit is **{worst['machine_id']}** (failure probability: {worst['failure_probability']:.1%}, "
                     f"RUL: {worst['rul_hours']}h). Recommend prioritizing preventive maintenance to avoid unplanned downtime.")
        return "\n".join(lines)


def build_supply_chain_report(predictions_df: pd.DataFrame) -> str:
    """Generate supply chain risk executive summary."""
    if predictions_df.empty:
        return "No supply chain data available."

    total      = len(predictions_df)
    delayed    = predictions_df["Predicted_Delay"].sum() if "Predicted_Delay" in predictions_df.columns else 0
    delay_rate = delayed / total if total > 0 else 0

    high_risk_regions = []
    if "Order Region" in predictions_df.columns and "Predicted_Delay" in predictions_df.columns:
        region_risk = (
            predictions_df.groupby("Order Region")["Predicted_Delay"]
            .mean().sort_values(ascending=False).head(3)
        )
        high_risk_regions = [f"{r} ({v:.0%})" for r, v in region_risk.items()]

    prompt = f"""Write a 3-sentence raw material supply chain executive briefing for the Lahore Textile Plant:
- Total shipments analyzed: {total:,}
- Predicted delayed shipments: {int(delayed):,} ({delay_rate:.1%} of fleet)
- Highest-risk regions: {', '.join(high_risk_regions) if high_risk_regions else 'Data unavailable'}
- Recommended action: focus on high-risk routes and supplier negotiations.
Be professional and specific."""

    ai_text = _generate_text(prompt)

    if ai_text:
        return ai_text
    else:
        status = "🔴 HIGH RISK" if delay_rate > 0.4 else ("🟡 MODERATE" if delay_rate > 0.2 else "🟢 LOW RISK")
        lines = [
            f"**Supply Chain Status: {status}**",
            f"",
            f"Analysis of **{total:,} shipments** reveals a predicted delay rate of **{delay_rate:.1%}**, "
            f"affecting approximately **{int(delayed):,} orders**.",
        ]
        if high_risk_regions:
            lines.append(f"The highest-risk regions are **{', '.join(high_risk_regions)}**, "
                         f"which should be prioritized for carrier renegotiation and buffer stock allocation.")
        lines.append(f"Recommend activating contingency routing protocols for flagged shipments "
                     f"and initiating supplier performance reviews for the affected regions.")
        return "\n".join(lines)


def build_daily_briefing(health_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
    """
    Combine maintenance + supply chain data into a full daily AI briefing.
    Returns a dict with sections for the dashboard.
    """
    date_str = datetime.now().strftime("%A, %d %B %Y")

    maintenance_summary = build_maintenance_report(health_df)
    supply_chain_summary = build_supply_chain_report(predictions_df)

    # Overall risk score (composite)
    maint_risk   = health_df["failure_probability"].mean() if not health_df.empty else 0
    sc_risk      = (predictions_df["Predicted_Delay"].mean()
                    if not predictions_df.empty and "Predicted_Delay" in predictions_df.columns else 0)
    overall_risk = round((maint_risk * 0.5 + sc_risk * 0.5) * 100, 1)

    alert_count = 0
    if not health_df.empty:
        alert_count += len(health_df[health_df["severity"].isin(["CRITICAL", "HIGH"])])
    if not predictions_df.empty and "Predicted_Delay" in predictions_df.columns:
        alert_count += int(predictions_df["Predicted_Delay"].sum())

    briefing = {
        "date":                  date_str,
        "overall_risk_score":    overall_risk,
        "total_alerts":          alert_count,
        "maintenance_summary":   maintenance_summary,
        "supply_chain_summary":  supply_chain_summary,
        "executive_headline":    _build_headline(overall_risk, alert_count),
        "generated_at":          datetime.now().isoformat(),
        "model_used":            MODEL_NAME if HF_AVAILABLE else "Template Engine",
    }

    # Save to file
    report_path = os.path.join(REPORT_DIR, f"briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(briefing, f, indent=2, ensure_ascii=False)
    print(f"[OK] Daily briefing saved -> {report_path}")

    return briefing


def _build_headline(risk_score: float, alert_count: int) -> str:
    if risk_score >= 60:
        return f"🔴 HIGH OPERATIONAL RISK — {alert_count} active alerts require immediate attention."
    elif risk_score >= 35:
        return f"🟡 ELEVATED RISK — {alert_count} alerts detected. Monitor closely and prepare contingencies."
    else:
        return f"🟢 OPERATIONS NORMAL — {alert_count} minor alerts. System performing within expected parameters."


def list_saved_reports() -> list:
    """Return list of previously generated reports (for history tab)."""
    files = sorted(
        [f for f in os.listdir(REPORT_DIR) if f.endswith(".json")],
        reverse=True
    )
    reports = []
    for f in files[:20]:
        path = os.path.join(REPORT_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            reports.append({
                "filename":   f,
                "date":       data.get("date", "Unknown"),
                "risk_score": data.get("overall_risk_score", 0),
                "alerts":     data.get("total_alerts", 0),
                "path":       path,
            })
        except Exception:
            pass
    return reports


# ══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Mock data test
    health_mock = pd.DataFrame([
        {"machine_id": "M-001", "failure_probability": 0.82, "health_score": 18, "rul_hours": 8,  "severity": "CRITICAL"},
        {"machine_id": "M-002", "failure_probability": 0.55, "health_score": 45, "rul_hours": 32, "severity": "HIGH"},
        {"machine_id": "M-003", "failure_probability": 0.12, "health_score": 88, "rul_hours": 63, "severity": "LOW"},
    ])
    sc_mock = pd.DataFrame({
        "Order Region": ["Western Europe", "Central America", "Southeast Asia"] * 100,
        "Predicted_Delay": np.random.choice([0, 1], 300, p=[0.65, 0.35])
    })

    briefing = build_daily_briefing(health_mock, sc_mock)
    print("\n" + "=" * 60)
    print("DAILY AI BRIEFING")
    print("=" * 60)
    for k, v in briefing.items():
        print(f"\n[{k.upper()}]\n{v}")
