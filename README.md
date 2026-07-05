# Credit Card Fraud Detection Using Machine Learning

A BCA Minor Project: a Flask web application that uses a Random Forest
Classifier to predict whether a credit card transaction is **Genuine** or
**Fraudulent**.

- **GitHub Repository:** `<GitHub Link>`
- **Live Application:** `<Deployment Link>`

---

## 1. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | Scikit-Learn (Random Forest Classifier) |
| Data | Pandas, NumPy |
| Backend | Flask |
| Frontend | HTML5, CSS3 (Jinja2 templates) |
| Deployment | Gunicorn + any Python-friendly host (Render / Railway / PythonAnywhere) |

This is fundamentally a **Python** project. `npm` is included only as a thin,
convenient wrapper (`npm run dev`) around the Python commands, in case you're
more used to typing `npm run dev` out of habit. It does **not** require any
actual Node.js packages to function — `dev.js` just spawns `python`/`python3`
for you.

---

## 2. Project Structure

```
credit-card-fraud-detection/
├── app.py                       # Flask application (routes + prediction logic)
├── train_model.py                # Full ML training pipeline
├── dev.js / package.json         # npm wrapper around the Python app
├── requirements.txt              # Python dependencies
├── Procfile                      # For Gunicorn-based deployment
├── data/
│   ├── generate_synthetic_data.py
│   └── creditcard.csv            # (generated — or drop in the real Kaggle CSV)
├── model/                        # Saved model, scaler, metrics (generated)
├── static/css/style.css
└── templates/
    ├── base.html
    ├── index.html
    └── metrics.html
```

---

## 3. Quick Start (Local)

You need **Python 3.10+** installed. Node/npm is optional — only needed if you
want to use the `npm run ...` shortcuts.

### Option A — using npm shortcuts

```bash
# 1. Install Python dependencies
npm run setup

# 2. Generate a sample dataset (synthetic — see note below)
npm run generate-data

# 3. Train the model
npm run train

# 4. Start the app
npm run dev
```

Then open **http://127.0.0.1:5000** in your browser.

### Option B — plain Python (no npm at all)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
python data/generate_synthetic_data.py
python train_model.py
python app.py
```

Then open **http://127.0.0.1:5000**.

---

## 4. About the Dataset

The real dataset used in the project report is the Kaggle
**"Credit Card Fraud Detection"** dataset:
https://www.kaggle.com/mlg-ulb/creditcardfraud

That dataset requires a (free) Kaggle account to download, so it can't be
fetched automatically by a script. To let you run this project immediately
without signing up anywhere, `data/generate_synthetic_data.py` creates a
**synthetic** dataset with the exact same column structure
(`Time, V1...V28, Amount, Class`) and a similarly small fraud ratio.

**To use the real dataset (recommended before submitting/demoing this project):**

1. Download `creditcard.csv` from the Kaggle link above.
2. Replace `data/creditcard.csv` with the downloaded file (same filename, same columns).
3. Re-run `python train_model.py`.
4. Restart the app (`npm run dev` / `python app.py`).

Everything downstream (scaling, imbalance handling, training, the web app) is
identical either way.

---

## 5. How It Works

1. **`train_model.py`** loads `data/creditcard.csv`, cleans it, scales the
   `Time`/`Amount` columns, balances the classes via undersampling, trains a
   `RandomForestClassifier`, evaluates it (Accuracy, Precision, Recall,
   F1-score, Confusion Matrix, ROC-AUC), and saves:
   - `model/random_forest_model.pkl`
   - `model/scaler.pkl`
   - `model/scale_cols.pkl`
   - `model/feature_order.json`
   - `model/metrics.json`
2. **`app.py`** loads those artifacts on startup and exposes:
   - `GET /` — the prediction form
   - `POST /predict` — runs a prediction on submitted form values
   - `GET /metrics` — displays the saved evaluation metrics
   - `GET /health` — simple JSON health check (useful for deployment platforms)

---

## 6. Pushing This Project to GitHub

From inside the project folder:

```bash
git init
git add .
git commit -m "Initial commit: Credit Card Fraud Detection ML web app"

# Create an empty repo on GitHub first (via github.com/new), then:
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

Note: `.gitignore` excludes the generated `data/creditcard.csv` and the
`model/*.pkl` files by default (they're derived artifacts, not source code,
and can be large). If you want them included in the repo anyway — e.g. so a
grader can run the app without retraining — remove those lines from
`.gitignore` before committing.

---

## 7. Deploying Online

Any host that runs a Python/Flask app + Gunicorn works (Render, Railway,
PythonAnywhere, Fly.io, etc.). General steps (Render as an example):

1. Push the repo to GitHub (see above).
2. On Render: **New → Web Service** → connect your GitHub repo.
3. Build command: `pip install -r requirements.txt && python data/generate_synthetic_data.py && python train_model.py`
   (or skip the generate/train steps and commit the `model/` folder directly).
4. Start command: `gunicorn app:app`
5. Deploy, then copy the live URL into this README and into your project report's Appendix.

---

## 8. Disclaimer

This is an educational **BCA Minor Project**. It is not connected to any real
bank, is not PCI-DSS compliant, and should not be used to process real
financial transactions.
