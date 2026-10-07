# Silent Churn Detector

    pip install -r requirements.txt
    python app.py            # http://localhost:5055

Uses `model/churn_model_7030.pkl` (tuned LR + SVM + XGBoost hybrid). If it can't be unpickled, the app retrains the
same model from `data/engineered_features.csv`. Set `MODEL_PATH` to serve another pickle. The UI loads React/Babel from cdnjs, so it needs internet.

| Endpoint | Purpose |
|---|---|
| GET /api/health | Liveness + model source |
| GET /api/schema | Form fields, options, encodings |
| POST /api/predict | Probability, risk band, drivers |
| POST /api/predict/batch | CSV upload (raw telco or encoded columns, ≤5,000 rows) |
| GET /api/insights | Churn rate by segment |
| GET /api/model | Model card + metrics |
