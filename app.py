"""Silent Churn — Flask API + static React UI.  Run:  python app.py  ->  http://localhost:5000"""
import io, os, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE / "model" / "churn_model_7030.pkl"))
DATA_PATH = BASE / "data" / "engineered_features.csv"
BUILD = "v4-detector"
app = Flask(__name__, static_folder="static", static_url_path="")

# ---- Feature schema (mirrors the encodings in UpdatedFeatureEngineering.ipynb) ----
FEATURES = ["tenure_group", "charge_level", "multiple_lines", "tech_support",
            "internet_service", "online_security", "contract"]
SCHEMA = [
    dict(key="tenure_group", label="Tenure", group="Loyalty", options=[
        dict(value=0, label="New", hint="0–12 months"), dict(value=1, label="Established", hint="13–36 months"),
        dict(value=2, label="Loyal", hint="36+ months")]),
    dict(key="charge_level", label="Monthly charges", group="Pricing", options=[
        dict(value=0, label="Low", hint="Below 35"), dict(value=1, label="Medium", hint="35 to 70"),
        dict(value=2, label="High", hint="70 and above")]),
    dict(key="internet_service", label="Internet service", group="Services", options=[
        dict(value=0, label="None", hint="No internet"), dict(value=1, label="DSL", hint="Fixed broadband"),
        dict(value=2, label="Fiber", hint="Fiber optic")]),
    dict(key="multiple_lines", label="Multiple lines", group="Services", options=[
        dict(value=0, label="No"), dict(value=1, label="Yes")]),
    dict(key="tech_support", label="Tech support", group="Services", options=[
        dict(value=0, label="No"), dict(value=1, label="Yes")]),
    dict(key="online_security", label="Online security", group="Services", options=[
        dict(value=0, label="No"), dict(value=1, label="Yes")]),
    dict(key="contract", label="Contract", group="Commitment", options=[
        dict(value=0, label="Month-to-month"), dict(value=1, label="One year"), dict(value=2, label="Two year")]),
]
MODEL_COLS = ["Tenure Group", "Charge Level", "Multiple Lines", "Tech Support", "Internet Service", "Online Security", "Contract"]
LABELS = {f["key"]: {o["value"]: o["label"] for o in f["options"]} for f in SCHEMA}
NAMES = {f["key"]: f["label"] for f in SCHEMA}

# Reported test metrics: tuned hybrid (LR + SVM + XGBoost), 70/30 split — HyperparameterTuning.ipynb
MODEL_INFO = dict(
    name="Tuned soft-voting hybrid", members=["Logistic Regression (C=0.01)", "SVM (linear, C=10)", "XGBoost (depth 5, 100 trees)"],
    split="70 / 30", balancing="SMOTETomek", threshold=0.5,
    metrics=dict(precision=0.50, recall=0.77, f1=0.60, auc=0.82))

# ---- Model loading (pickle first; retrain per ChurnPredictionApp.ipynb if unpickling fails) ----
DF = pd.read_csv(DATA_PATH)
DF.columns = FEATURES + ["churn"]

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f), f"pickle:{MODEL_PATH.name}"
    except Exception as e:
        print(f"[warn] could not load {MODEL_PATH.name} ({e}); retraining from CSV")
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
        from imblearn.combine import SMOTETomek
        X, y = DF[FEATURES].set_axis(MODEL_COLS, axis=1), DF["churn"]
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        Xb, yb = SMOTETomek(random_state=42).fit_resample(Xtr, ytr)
        m = VotingClassifier([("lr", LogisticRegression(C=0.01, random_state=42)),
                              ("svm", SVC(C=10, kernel="linear", probability=True, random_state=42)),
                              ("xgb", XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100,
                                                    random_state=42, eval_metric="logloss"))], voting="soft")
        return m.fit(Xb, yb), "retrained"

MODEL, MODEL_SOURCE = load_model()
MARGINALS = {k: DF[k].value_counts(normalize=True).to_dict() for k in FEATURES}  # reference population

def proba(X):
    return MODEL.predict_proba(pd.DataFrame(X, columns=MODEL_COLS))[:, 1]

def band(p):
    return "high" if p >= 0.5 else "steady" if p >= 0.3 else "low"

def drivers(x):
    """Model-agnostic attribution: p(x) minus expected p when one feature is drawn from the population.
    Positive = pushes churn risk up (percentage points)."""
    base = proba([x])[0]
    rows, spans = [], []
    for i, k in enumerate(FEATURES):
        vals = list(MARGINALS[k].items())
        spans.append((len(rows), vals))
        for v, _ in vals:
            r = list(x); r[i] = int(v); rows.append(r)
    ps = proba(rows)
    out = []
    for i, k in enumerate(FEATURES):
        start, vals = spans[i]
        expected = sum(w * ps[start + j] for j, (_, w) in enumerate(vals))
        out.append(dict(key=k, label=NAMES[k], value=LABELS[k][x[i]], impact=round(float(base - expected) * 100, 1)))
    return sorted(out, key=lambda d: -abs(d["impact"]))

def parse_input(body):
    try:
        x = [int(body[k]) for k in FEATURES]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Required integer fields: {', '.join(FEATURES)}")
    for k, v in zip(FEATURES, x):
        if v not in LABELS[k]: raise ValueError(f"Invalid value {v} for '{k}'")
    return x

# ---- API ----
@app.after_request
def no_cache(resp):  # always serve the latest UI files during development
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/api/health")
def health():
    return jsonify(status="ok", build=BUILD, model_source=MODEL_SOURCE, model_class=type(MODEL).__name__)

@app.get("/api/schema")
def schema():
    return jsonify(features=SCHEMA)

@app.get("/api/model")
def model_info():
    return jsonify(**MODEL_INFO, source=MODEL_SOURCE, training_rows=len(DF))

@app.post("/api/predict")
def predict():
    try: x = parse_input(request.get_json(force=True, silent=True) or {})
    except ValueError as e: return jsonify(error=str(e)), 400
    p = float(proba([x])[0])
    return jsonify(churn_probability=round(p * 100, 1), retention_probability=round((1 - p) * 100, 1),
                   prediction=int(p >= MODEL_INFO["threshold"]), risk_band=band(p),
                   drivers=drivers(x))

@app.get("/api/insights")
def insights():
    out = dict(customers=len(DF), churn_rate=round(DF.churn.mean() * 100, 1), features=[])
    for f in SCHEMA:
        g = DF.groupby(f["key"]).churn.agg(["mean", "count"])
        out["features"].append(dict(key=f["key"], label=f["label"], segments=[
            dict(label=LABELS[f["key"]][int(v)], churn_rate=round(r["mean"] * 100, 1), customers=int(r["count"]))
            for v, r in g.iterrows()]))
    return jsonify(out)

RAW = {"Multiple Lines": {"No phone service": 0, "No": 0, "Yes": 1}, "Tech Support": {"No internet service": 0, "No": 0, "Yes": 1},
       "Online Security": {"No internet service": 0, "No": 0, "Yes": 1}, "Internet Service": {"No": 0, "DSL": 1, "Fiber optic": 2},
       "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}}

def encode_upload(df):
    """Accepts raw Telco columns (Tenure, Monthly Charges, text categories) or already-encoded feature columns."""
    if set(FEATURES) <= set(df.columns): return df[FEATURES].astype(int)
    need = ["Tenure", "Monthly Charges", *RAW]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {', '.join(miss)}")
    out = pd.DataFrame({
        "tenure_group": pd.cut(df["Tenure"], [-1, 12, 36, 10**6], labels=[0, 1, 2]).astype(int),
        "charge_level": np.select([df["Monthly Charges"] < 35, df["Monthly Charges"] < 70], [0, 1], 2)})
    for col, key in zip(RAW, ["multiple_lines", "tech_support", "online_security", "internet_service", "contract"]):
        out[key] = df[col].map(RAW[col])
    if out.isnull().any().any(): raise ValueError("Unrecognised category values in upload")
    return out[FEATURES].astype(int)

@app.post("/api/predict/batch")
def predict_batch():
    f = request.files.get("file")
    if not f: return jsonify(error="Upload a CSV as form field 'file'"), 400
    try:
        raw = pd.read_csv(io.BytesIO(f.read()))
        if len(raw) > 5000: raise ValueError("Limit is 5,000 rows per upload")
        X = encode_upload(raw)
    except Exception as e: return jsonify(error=str(e)), 400
    p = proba(X.values.tolist())
    rows = [dict(row=i + 1, probability=round(float(v) * 100, 1), risk_band=band(v),
                 **{k: LABELS[k][int(X.iloc[i][k])] for k in FEATURES}) for i, v in enumerate(p)]
    summary = {b: sum(r["risk_band"] == b for r in rows) for b in ("high", "steady", "low")}
    return jsonify(count=len(rows), summary=summary, avg_probability=round(float(p.mean()) * 100, 1), rows=rows)

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    print(f"Build {BUILD} -> open http://localhost:{os.getenv('PORT', 5055)}")
    print(f"Serving UI from: {BASE / 'static'}")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5055)), debug=False, use_reloader=False)
