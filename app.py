from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle, sqlite3, json, os, hashlib, secrets
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
DB_PATH = "oncoscan.db"

# ── Feature order ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean",
    "concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se",
    "concave_points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst",
    "smoothness_worst","compactness_worst","concavity_worst",
    "concave_points_worst","symmetry_worst","fractal_dimension_worst",
]

# WDBC: 0 = Malignant, 1 = Benign
LABELS = {0: "Malignant  (Cancer)", 1: "Benign  (No Cancer)"}


# ── Load models ───────────────────────────────────────────────────────────────
def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

svm_model = load_model("svm_model.pkl")
rf_model  = load_model("model.pkl")
lr_model  = load_model("lr_model.pkl")
scaler    = load_model("scaler.pkl")

MODELS = [
    ("SVM",                 svm_model),
    ("Random Forest",       rf_model),
    ("Logistic Regression", lr_model),
]


# ── DB setup ──────────────────────────────────────────────────────────────────
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = get_db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            role          TEXT    DEFAULT 'doctor',
            created_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cases (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER,
            patient_id    TEXT,
            patient_name  TEXT,
            patient_age   TEXT,
            physician     TEXT,
            sample_date   TEXT,
            notes         TEXT,
            features      TEXT,
            prediction    TEXT,
            confidence    REAL,
            model_votes   TEXT,
            created_at    TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    con.commit()
    con.close()

init_db()


# ── Auth helpers ──────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    con = get_db()
    user = con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    con.close()
    return dict(user) if user else None


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html", user=current_user())

@app.route("/predict")
def predict_page():
    return render_template("predict.html", user=current_user())

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)
    email    = request.form.get("email","").strip().lower()
    password = request.form.get("password","")
    con = get_db()
    user = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    con.close()
    if not user or user["password_hash"] != hash_password(password):
        return render_template("login.html", error="Invalid email or password.")
    session["user_id"] = user["id"]
    return redirect(url_for("predict_page"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", error=None)
    name     = request.form.get("name","").strip()
    email    = request.form.get("email","").strip().lower()
    password = request.form.get("password","")
    confirm  = request.form.get("confirm","")
    role     = request.form.get("role","doctor")
    if not name or not email or not password:
        return render_template("register.html", error="All fields are required.")
    if password != confirm:
        return render_template("register.html", error="Passwords do not match.")
    if len(password) < 6:
        return render_template("register.html", error="Password must be at least 6 characters.")
    try:
        con = get_db()
        con.execute(
            "INSERT INTO users (name,email,password_hash,role,created_at) VALUES (?,?,?,?,?)",
            (name, email, hash_password(password), role, datetime.now().isoformat())
        )
        con.commit()
        user = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        con.close()
        session["user_id"] = user["id"]
        return redirect(url_for("predict_page"))
    except sqlite3.IntegrityError:
        return render_template("register.html", error="An account with this email already exists.")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ── Prediction API ────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing: {missing}"}), 400

        features_raw    = np.array([[float(data[f]) for f in FEATURE_NAMES]])
        features_scaled = scaler.transform(features_raw) if scaler else features_raw

        votes = []
        proba_sum    = np.zeros(2)
        valid_models = 0

        for name, mdl in MODELS:
            if mdl is None:
                continue
            try:
                pred  = mdl.predict(features_scaled)[0]
                proba = mdl.predict_proba(features_scaled)[0]
                proba_sum += proba
                valid_models += 1
                votes.append({
                    "model":      name,
                    "prediction": LABELS[int(pred)],
                    "confidence": round(float(proba[int(pred)]), 4),
                })
            except Exception as e:
                votes.append({"model": name, "prediction": "Error", "confidence": 0})

        if valid_models == 0:
            return jsonify({"error": "No models loaded. Run model_training.ipynb first."}), 500

        avg_proba  = proba_sum / valid_models
        pred_class = int(np.argmax(avg_proba))
        label      = LABELS[pred_class]
        confidence = round(float(avg_proba[pred_class]), 4)

        # Save to DB if logged in
        user = current_user()
        if user:
            patient = data.get("patient", {})
            features_dict = {f: float(data[f]) for f in FEATURE_NAMES}
            con = get_db()
            con.execute("""
                INSERT INTO cases
                (user_id,patient_id,patient_name,patient_age,physician,
                 sample_date,notes,features,prediction,confidence,model_votes,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                user["id"],
                patient.get("id",""),   patient.get("name",""),
                patient.get("age",""),  patient.get("physician",""),
                patient.get("date",""), patient.get("notes",""),
                json.dumps(features_dict), label, confidence,
                json.dumps(votes), datetime.now().isoformat()
            ))
            con.commit()
            con.close()

        return jsonify({
            "prediction":  label,
            "confidence":  confidence,
            "model_votes": votes,
            "saved":       user is not None,
            "probabilities": {
                "malignant": round(float(avg_proba[0]), 4),
                "benign":    round(float(avg_proba[1]), 4),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── History API ───────────────────────────────────────────────────────────────
@app.route("/api/history")
def api_history():
    user = current_user()
    if not user:
        return jsonify({"error": "Login required"}), 401
    con = get_db()
    rows = con.execute(
        "SELECT * FROM cases WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
        (user["id"],)
    ).fetchall()
    con.close()
    cases = []
    for r in rows:
        cases.append({
            "id":           r["id"],
            "date":         r["sample_date"] or r["created_at"][:10],
            "patient_id":   r["patient_id"],
            "patient_name": r["patient_name"],
            "patient_age":  r["patient_age"],
            "physician":    r["physician"],
            "prediction":   r["prediction"],
            "confidence":   r["confidence"],
            "notes":        r["notes"],
            "created_at":   r["created_at"][:16].replace("T"," "),
        })
    return jsonify({"cases": cases, "user": user["name"]})


@app.route("/api/me")
def api_me():
    user = current_user()
    if not user:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "name": user["name"], "email": user["email"], "role": user["role"]})


@app.route("/health")
def health():
    loaded = {name: (mdl is not None) for name, mdl in MODELS}
    return jsonify({"status": "ok", "models": loaded, "scaler": scaler is not None})


if __name__ == "__main__":
    app.run(debug=True)


@app.route("/api/extract_pdf", methods=["POST"])
def extract_pdf():
    import io, re
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400
    pdf_bytes = request.files["pdf"].read()
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception:
        pass
    if not text.strip():
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            for img in convert_from_bytes(pdf_bytes):
                text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            return jsonify({"error": "Install pdfplumber: pip install pdfplumber"}), 400
    if not text.strip():
        return jsonify({"error": "No text found in PDF."}), 400

    text = text.lower()
    name_map = {
        "radius mean":"radius_mean","texture mean":"texture_mean","perimeter mean":"perimeter_mean",
        "area mean":"area_mean","smoothness mean":"smoothness_mean","compactness mean":"compactness_mean",
        "concavity mean":"concavity_mean","concave points mean":"concave_points_mean",
        "symmetry mean":"symmetry_mean","fractal dimension mean":"fractal_dimension_mean",
        "radius se":"radius_se","texture se":"texture_se","perimeter se":"perimeter_se",
        "area se":"area_se","smoothness se":"smoothness_se","compactness se":"compactness_se",
        "concavity se":"concavity_se","concave points se":"concave_points_se",
        "symmetry se":"symmetry_se","fractal dimension se":"fractal_dimension_se",
        "radius worst":"radius_worst","texture worst":"texture_worst","perimeter worst":"perimeter_worst",
        "area worst":"area_worst","smoothness worst":"smoothness_worst","compactness worst":"compactness_worst",
        "concavity worst":"concavity_worst","concave points worst":"concave_points_worst",
        "symmetry worst":"symmetry_worst","fractal dimension worst":"fractal_dimension_worst",
    }
    features = {f: None for f in FEATURE_NAMES}
    num = r"[-+]?\d*\.?\d+"
    for readable, key in name_map.items():
        for term in [readable, readable.replace(" ","_")]:
            m = re.search(rf"{re.escape(term)}\s*[:\-=]?\s*({num})", text)
            if m:
                try: features[key] = float(m.group(1)); break
                except: pass
    return jsonify({"features": features})
