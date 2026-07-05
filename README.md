# Credit Card Fraud Detection Using Machine Learning

A BCA Minor Project: a Flask web application that uses a Random Forest
Classifier to predict whether a credit card transaction is **Genuine** or
**Fraudulent**.


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

