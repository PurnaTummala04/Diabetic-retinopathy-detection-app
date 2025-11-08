import os
import re
import io
import json
import datetime
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, abort
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from joblib import load
from xhtml2pdf import pisa # Used for PDF generation

# ✅ --- GridFS INCLUDED ---
import gridfs

# ============================================
# FLASK CONFIGURATION
# ============================================
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "supersecretkey")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/drapp")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "drapp")

bcrypt = Bcrypt(app)

# ============================================
# SIMPLIFIED, ROBUST MONGODB CONNECTION
# ============================================
client = None
db = None
fs = None

def connect_mongo():
    """
    Robust connection logic:
    1) Try cloud MONGO_URI if provided
    2) Else try local MongoDB
    3) Else try mongomock (in-memory)
    """
    global client, db, fs

    mongo_uri = os.environ.get("MONGO_URI")
    mongo_db_name = os.environ.get("MONGO_DB_NAME", "drapp")

    # 1) Try cloud (Atlas / MONGO_URI)
    if mongo_uri:
        try:
            app.logger.info(f"Attempting to connect to MongoDB Atlas: {mongo_uri}")
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
            client.admin.command("ping")
            db = client[mongo_db_name]
            fs = gridfs.GridFS(db)
            app.logger.info("Connected to MongoDB Atlas.")
            return client, db
        except Exception as e:
            app.logger.error(f"MongoDB Atlas connection failed: {e}")

    # 2) Try local MongoDB
    try:
        local_uri = f"mongodb://localhost:27017/{mongo_db_name}"
        app.logger.info(f"Attempting to connect to local MongoDB: {local_uri}")
        client = MongoClient(local_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[mongo_db_name]
        fs = gridfs.GridFS(db)
        app.logger.warning("Connected to local MongoDB.")
        return client, db
    except Exception as e:
        app.logger.error(f"Local MongoDB connection failed: {e}")

    # 3) Fallback to mongomock (in-memory)
    try:
        import mongomock
        app.logger.warning("Falling back to in-memory mongomock (data will NOT persist).")
        client = mongomock.MongoClient()
        db = client[mongo_db_name]
        fs = gridfs.GridFS(db)
        return client, db
    except Exception as e:
        app.logger.critical(f"mongomock fallback failed: {e}")
        raise RuntimeError("Could not connect to MongoDB (Atlas/local/mongomock failed).")

# Establish connection on import
try:
    client, db = connect_mongo()
    users_col = db["users"]
    reports_col = db["reports"]
    batches_col = db["batches"]
    batch_rows_col = db["batch_rows"]
except RuntimeError as e:
    # Handle the case where all connection attempts failed
    print(f"FATAL ERROR: {e}")
    users_col = None
    reports_col = None
    batches_col = None
    batch_rows_col = None

# ✅ FIX 1: Use 'is not None' for boolean comparison on collection objects
if users_col is not None and reports_col is not None and batches_col is not None and batch_rows_col is not None:
    try:
        users_col.create_index("username", unique=True)
        reports_col.create_index([("probability", DESCENDING), ("created_at", DESCENDING)])
        reports_col.create_index("patient_id")
        reports_col.create_index("doctor_id")
        batches_col.create_index([("doctor_id", ASCENDING), ("created_at", DESCENDING)])
        batch_rows_col.create_index([("batch_id", ASCENDING), ("ensemble_avg_prob", DESCENDING)])
    except Exception as e:
        app.logger.warning(f"Index creation warning: {e}")
else:
    app.logger.critical("Database collections not initialized due to connection failure.")


# ============================================
# PATHS AND DIRECTORIES
# ============================================
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# ✅ THRESHOLD
ENSEMBLE_THRESHOLD_CUSTOM = 0.371

# ============================================
# SEED ADMIN USER
# ============================================
def seed_admin():
    # ✅ FIX 2: Use 'is None' for safe check
    if users_col is None: return 
    
    admin_user = os.environ.get("ADMIN_USERNAME", "admin").strip().lower()
    admin_pass = os.environ.get("ADMIN_PASSWORD", "a")
    admin_name = os.environ.get("ADMIN_NAME", "admin")
    
    if not admin_user or not admin_pass:
        return
    
    doc = users_col.find_one({"username": admin_user})
    pw_hash = bcrypt.generate_password_hash(admin_pass).decode("utf-8")
    
    if not doc:
        users_col.insert_one({
            "role": "admin",
            "name": admin_name,
            "username": admin_user,
            "password_hash": pw_hash,
            "created_at": datetime.datetime.utcnow(),
        })
        app.logger.info("Admin user created from env.")
    else:
        if doc.get("role") != "admin":
            users_col.update_one({"_id": doc["_id"]}, {"$set": {"role": "admin", "name": admin_name}})
            app.logger.info("Existing user promoted to admin.")
        
        if os.environ.get("ADMIN_RESET") == "1":
            users_col.update_one({"_id": doc["_id"]}, {"$set": {"password_hash": pw_hash}})
            app.logger.info("Admin password reset from env.")

seed_admin()

# ============================================
# AUTHENTICATION
# ============================================
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, _id, username, name, role):
        self.id = str(_id)
        self.username = username
        self.name = name
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    if users_col is None: return None # Safety check if DB failed
    try:
        doc = users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    if not doc:
        return None
    return User(doc["_id"], doc["username"], doc["name"], doc["role"])

def is_admin():
    return current_user.is_authenticated and current_user.role == "admin"

# ============================================
# TIMEZONE HELPERS (IST)
# ============================================
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

def to_ist_str(dt: datetime.datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(IST).strftime(fmt)

@app.template_filter("ist")
def ist_filter(dt, fmt="%Y-%m-%d %H:%M"):
    return to_ist_str(dt, fmt)

# ============================================
# ML ARTIFACTS LOADING
# ============================================
SCALER = None
FEATURES = []
CLINICAL_FEATURES = ["fasting_glucose", "hba1c", "diabetes_duration"]
MODEL_ORDER = []
MODEL_FILES = {}
MODELS = {}
PER_MODEL_THRESH = {}
PER_MODEL_BALACC = {}
ENSEMBLE_BALACC = 0.0
ENSEMBLE_LABEL = ""
METRICS_ALL = {}

def load_artifacts():
    global SCALER, FEATURES, CLINICAL_FEATURES, MODEL_ORDER, MODEL_FILES, MODELS
    global PER_MODEL_THRESH, PER_MODEL_BALACC, ENSEMBLE_BALACC, ENSEMBLE_LABEL, METRICS_ALL
    
    meta_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    
    if not os.path.exists(meta_path) or not os.path.exists(scaler_path):
        app.logger.warning("Artifacts missing. Run train.py first.")
        return
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # ✅ 8 FEATURES ONLY (NO derived features)
    FEATURES = meta.get("features", [
        "exudates_count", "hemorrhages_count", "microaneurysms_count",
        "vessel_tortuosity", "macular_thickness", "fasting_glucose",
        "hba1c", "diabetes_duration"
    ])
    
    MODEL_FILES = meta.get("model_filenames", {})
    if MODEL_FILES:
        MODEL_ORDER = meta.get("model_order", list(MODEL_FILES.keys()))
    else:
        MODEL_ORDER = []
    
    try:
        with open(metrics_path, "r") as f:
            METRICS_ALL = json.load(f)
            per_model = METRICS_ALL.get("per_model", {})
            ensemble = METRICS_ALL.get("ensemble", {})
            
            for name in MODEL_ORDER:
                if name in per_model:
                    PER_MODEL_THRESH[name] = float(per_model[name].get("best_threshold", 0.371))
                    PER_MODEL_BALACC[name] = float(per_model[name].get("balanced_accuracy", 0.0))
                else:
                    PER_MODEL_THRESH[name] = 0.371
                    PER_MODEL_BALACC[name] = 0.0
            
            ENSEMBLE_BALACC = float(ensemble.get("balanced_accuracy", 0.0))
    except Exception as e:
        app.logger.warning(f"Could not load metrics: {e}")
        for name in MODEL_ORDER:
            PER_MODEL_THRESH[name] = 0.371
            PER_MODEL_BALACC[name] = 0.0
    
    try:
        SCALER = load(scaler_path)
    except Exception as e:
        app.logger.error(f"Failed to load scaler: {e}")
        SCALER = None
    
    MODELS = {}
    for name in MODEL_ORDER:
        fname = MODEL_FILES.get(name)
        if not fname:
            app.logger.error(f"No filename for model '{name}' in metadata.")
            continue
        
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            app.logger.error(f"Missing model file: {path}")
            continue
        
        try:
            MODELS[name] = load(path)
        except Exception as e:
            app.logger.error(f"Failed to load model {name} at {path}: {e}")
    
    ENSEMBLE_LABEL = f"Ensemble(mean) of {len(MODELS)} model{'s' if len(MODELS) != 1 else ''}"

def compute_ensemble_outputs(feature_dict: dict):
    """
    Computes ensemble prediction and individual model probabilities.
    """
    if SCALER is None or not MODELS:
        raise RuntimeError("Model artifacts not loaded. Run train.py first.")
    
    X = pd.DataFrame([feature_dict], columns=[
        "exudates_count", "hemorrhages_count", "microaneurysms_count",
        "vessel_tortuosity", "macular_thickness", "fasting_glucose",
        "hba1c", "diabetes_duration"
    ]).copy()
    
    # ✅ Clinical scaling: 0.1
    X.loc[:, CLINICAL_FEATURES] = X.loc[:, CLINICAL_FEATURES].astype(float) * 0.1
    X = X[FEATURES]
    X_scaled = SCALER.transform(X)
    
    per_model_details = []
    probs = []
    
    for name in MODEL_ORDER:
        mdl = MODELS.get(name)
        if mdl is None:
            continue
        
        prob = float(mdl.predict_proba(X_scaled)[:, 1][0])
        
        # ✅ Store PROBABILITY
        per_model_details.append({
            "name": name,
            "probability": prob
        })
        
        probs.append(prob)
    
    if not probs:
        raise RuntimeError("No models loaded.")
    
    avg_prob = float(np.mean(probs))
    # ✅ Decision: >= 0.371
    final_pred = int(avg_prob >= ENSEMBLE_THRESHOLD_CUSTOM)
    
    return {
        "per_model": per_model_details,
        "avg_prob": avg_prob,
        "final_pred": final_pred,
    }

load_artifacts()

# ============================================
# ROUTES
# ============================================

@app.route("/")
def index():
    return render_template("index.html", model_name=ENSEMBLE_LABEL)

@app.route("/register", methods=["GET", "POST"])
def register():
    if users_col is None:
        flash("Database connection failed. Cannot register.", "danger")
        return redirect(url_for("index"))
    
    if request.method == "POST":
        role = request.form.get("role")
        name = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        
        if role not in ("patient", "doctor"):
            flash("Please select a valid role.", "danger")
            return redirect(url_for("register"))
        
        if not name or not username or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        
        if users_col.find_one({"username": username}):
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))
        
        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        res = users_col.insert_one({
            "role": role,
            "name": name,
            "username": username,
            "password_hash": pw_hash,
            "created_at": datetime.datetime.utcnow(),
        })
        
        login_user(User(res.inserted_id, username, name, role))
        flash("Registered and logged in.", "success")
        return redirect(url_for("patient_dashboard" if role == "patient" else "doctor_dashboard"))
    
    return render_template("register.html", model_name=ENSEMBLE_LABEL)

@app.route("/login", methods=["GET", "POST"])
def login():
    if users_col is None:
        flash("Database connection failed. Cannot login.", "danger")
        return redirect(url_for("index"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        
        doc = users_col.find_one({"username": username})
        if not doc or not bcrypt.check_password_hash(doc["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))
        
        login_user(User(doc["_id"], doc["username"], doc["name"], doc["role"]))
        flash("Logged in.", "success")
        
        if doc["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("patient_dashboard" if doc["role"] == "patient" else "doctor_dashboard"))
    
    return render_template("login.html", model_name=ENSEMBLE_LABEL)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

@app.route("/patient")
@login_required
def patient_dashboard():
    if current_user.role != "patient":
        abort(403)
    
    if reports_col is None or users_col is None:
        flash("Database connection error.", "danger")
        return render_template("patient_dashboard.html", user=current_user, reports=[], q="", model_name=ENSEMBLE_LABEL)

    q = request.args.get("q", "").strip().lower()
    pid = ObjectId(current_user.id)
    
    # Efficiently fetch reports
    reports = list(reports_col.find({"patient_id": pid}).sort("created_at", DESCENDING))
    
    # Cache for doctor names
    doctor_cache = {}
    
    for r in reports:
        did = r.get("doctor_id")
        r["doctor_name"] = "Not assigned"
        
        if did:
            if did not in doctor_cache:
                ddoc = users_col.find_one({"_id": did}, {"name": 1})
                doctor_cache[did] = ddoc.get("name", "Unknown") if ddoc else "Unknown"
            r["doctor_name"] = doctor_cache[did]
            
        r["id_str"] = str(r["_id"])
        r["created_str"] = to_ist_str(r.get("created_at"))
    
    if q:
        filtered = []
        for r in reports:
            hay = f"{r['doctor_name']} {r['prediction']} {r.get('model_name','')} {r['created_str']} {r.get('probability',0.0):.3f}".lower()
            if q in hay:
                filtered.append(r)
        reports = filtered
    
    return render_template("patient_dashboard.html", user=current_user, reports=reports, q=q, model_name=ENSEMBLE_LABEL)

@app.route("/assessment/new", methods=["GET", "POST"])
@login_required
def new_assessment():
    if current_user.role != "patient":
        abort(403)
    
    if reports_col is None or users_col is None:
        flash("Database connection error.", "danger")
        return redirect(url_for("patient_dashboard"))
    
    doctors = list(users_col.find({"role": "doctor"}).sort("name", ASCENDING))
    
    if request.method == "POST":
        try:
            # Type casting validation
            feature_values = {
                "exudates_count": int(request.form["exudates_count"]),
                "hemorrhages_count": int(request.form["hemorrhages_count"]),
                "microaneurysms_count": int(request.form["microaneurysms_count"]),
                "vessel_tortuosity": float(request.form["vessel_tortuosity"]),
                "macular_thickness": float(request.form["macular_thickness"]),
                "fasting_glucose": float(request.form["fasting_glucose"]),
                "hba1c": float(request.form["hba1c"]),
                "diabetes_duration": float(request.form["diabetes_duration"]),
            }
        except Exception:
            flash("Please enter valid numeric values.", "danger")
            return redirect(url_for("new_assessment"))
        
        doctor_id_str = request.form.get("doctor_id") or ""
        doctor_id = ObjectId(doctor_id_str) if doctor_id_str else None
        
        try:
            out = compute_ensemble_outputs(feature_values)
        except RuntimeError as e:
            app.logger.exception(e)
            flash(str(e), "danger") # Model artifacts missing. Train the model first.
            return redirect(url_for("patient_dashboard"))
        except Exception as e:
            app.logger.exception(e)
            flash("An unexpected error occurred during model prediction.", "danger")
            return redirect(url_for("patient_dashboard"))
        
        report_doc = {
            "patient_id": ObjectId(current_user.id),
            "doctor_id": doctor_id,
            "features": feature_values,
            "probability": out["avg_prob"],
            "prediction": out["final_pred"],
            "threshold": ENSEMBLE_THRESHOLD_CUSTOM,
            "model_name": ENSEMBLE_LABEL,
            "ensemble_balanced_accuracy": ENSEMBLE_BALACC,
            "per_model_details": out["per_model"],
            "created_at": datetime.datetime.utcnow(),
        }
        
        result = reports_col.insert_one(report_doc)
        
        # FIX: Redirect to dashboard and STOP AUTO-DOWNLOAD
        flash(f"Assessment report created (ID: {str(result.inserted_id)}). You can download the PDF from your dashboard.", "success")
        return redirect(url_for("patient_dashboard"))
    
    return render_template("report_form.html", doctors=doctors, model_name=ENSEMBLE_LABEL)

@app.route("/report/<rid>/download_pdf", methods=["GET"])
@login_required
def download_report_pdf(rid):
    if reports_col is None or users_col is None:
        flash("Database connection error.", "danger")
        return redirect(url_for("index"))

    try:
        report_id = ObjectId(rid)
    except Exception:
        abort(404)

    report_doc = reports_col.find_one({"_id": report_id})
    if not report_doc:
        abort(404)

    # Authorization check
    if current_user.role == "patient" and report_doc.get("patient_id") != ObjectId(current_user.id):
        abort(403)
    
    patient = users_col.find_one({"_id": report_doc["patient_id"]}) or {}
    doctor = users_col.find_one({"_id": report_doc.get("doctor_id")}) if report_doc.get("doctor_id") else {}

    context = {
        "patient_name": patient.get("name", "Unknown"),
        "patient_username": patient.get("username", ""),
        "doctor_name": (doctor or {}).get("name", "Not assigned"),
        "doctor_username": (doctor or {}).get("username", ""),
        "created_at": to_ist_str(report_doc["created_at"]),
        "features": report_doc["features"],
        "probability": report_doc.get("probability", 0.0),
        "prediction": report_doc.get("prediction", 0),
        "threshold": report_doc.get("threshold", ENSEMBLE_THRESHOLD_CUSTOM),
        "ensemble_balanced_accuracy": report_doc.get("ensemble_balanced_accuracy", 0.0),
        "model_name": report_doc.get("model_name", ENSEMBLE_LABEL),
        "per_model_details": report_doc.get("per_model_details", []),
    }

    # Generate PDF (using the new, better looking template)
    html = render_template("report_pdf.html", **context)
    pdf_io = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf_io)

    if pisa_status.err:
        app.logger.error("PDF generation failed using xhtml2pdf.")
        flash("PDF generation failed.", "warning")
        return redirect(url_for("patient_dashboard" if current_user.role == "patient" else "doctor_dashboard"))
    
    pdf_io.seek(0)
    return send_file(pdf_io, as_attachment=True,
                        download_name=f"DR_Assessment_{str(report_doc['_id'])}.pdf",
                        mimetype="application/pdf")

@app.route("/doctor")
@login_required
def doctor_dashboard():
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    if reports_col is None or users_col is None:
        flash("Database connection error.", "danger")
        return render_template("doctor_dashboard.html", reports=[], order="desc", q="", model_name=ENSEMBLE_LABEL)
        
    q = request.args.get("q", "").strip().lower()
    order = request.args.get("order", "desc")
    sort_dir = DESCENDING if order == "desc" else ASCENDING
    
    # FIX: Correct filtering for Doctors
    query = {}
    if current_user.role == "doctor":
        # Doctors see reports assigned to them OR reports with no doctor assigned (unassigned)
        query = {
            "$or": [
                {"doctor_id": ObjectId(current_user.id)},
                {"doctor_id": {"$exists": False}}
            ]
        }
        
    reports = list(reports_col.find(query).sort("probability", sort_dir))
    
    cache = {}
    
    for r in reports:
        pid = r["patient_id"]
        did = r.get("doctor_id")
        
        # Cache patient name
        if pid not in cache:
            pdoc = users_col.find_one({"_id": pid}, {"name": 1}) or {}
            cache[pid] = pdoc.get("name", "Unknown")
        
        # Cache doctor name
        r["doctor_name"] = "Not assigned"
        if did:
            if did not in cache:
                ddoc = users_col.find_one({"_id": did}, {"name": 1}) or {}
                cache[did] = ddoc.get("name", "Unknown")
            r["doctor_name"] = cache[did]
            
        r["patient_name"] = cache.get(pid, "Unknown")
        
        r["id_str"] = str(r["_id"])
        r["created_str"] = to_ist_str(r.get("created_at"))
    
    if q:
        filtered = []
        for r in reports:
            hay = f"{r['patient_name']} {r['doctor_name']} {r['prediction']} {r.get('model_name','')} {r['created_str']} {r.get('probability',0.0):.3f}".lower()
            if q in hay:
                filtered.append(r)
        reports = filtered
    
    return render_template("doctor_dashboard.html", reports=reports, order=order, q=q, model_name=ENSEMBLE_LABEL)

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/doctor/bulk", methods=["GET"])
@login_required
def doctor_bulk_home():
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    if batches_col is None:
        flash("Database connection error.", "danger")
        return render_template("doctor_bulk.html", batches=[], q="", model_name=ENSEMBLE_LABEL)

    q = request.args.get("q", "").strip().lower()
    
    if is_admin():
        batches = list(batches_col.find({}).sort("created_at", DESCENDING))
    else:
        batches = list(batches_col.find({"doctor_id": ObjectId(current_user.id)}).sort("created_at", DESCENDING))
    
    for b in batches:
        b["id_str"] = str(b["_id"])
        b["created_str"] = to_ist_str(b["created_at"])
    
    if q:
        batches = [b for b in batches if q in (b.get("filename", "").lower()) or q in b["created_str"].lower()]
    
    return render_template("doctor_bulk.html", batches=batches, q=q, model_name=ENSEMBLE_LABEL)

@app.route("/doctor/bulk/upload", methods=["POST"])
@login_required
def doctor_bulk_upload():
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    if batches_col is None or batch_rows_col is None:
        flash("Database connection error.", "danger")
        return redirect(url_for("doctor_bulk_home"))

    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Please choose a CSV file.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    if not allowed_file(f.filename):
        flash("Only CSV files are allowed.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    filename = secure_filename(f.filename)
    
    try:
        f.seek(0)
        df = pd.read_csv(f)
    except Exception:
        flash("Failed to read CSV. Check the file format and contents.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    # ✅ 8 FEATURES ONLY
    base_features = [
        "exudates_count", "hemorrhages_count", "microaneurysms_count",
        "vessel_tortuosity", "macular_thickness", "fasting_glucose",
        "hba1c", "diabetes_duration"
    ]
    
    missing = [col for col in base_features if col not in df.columns]
    if missing:
        flash(f"Missing required columns: {', '.join(missing)}", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    has_label = "retinal_disorder" in df.columns
    
    # Ensure all feature columns are numeric, coercing errors to NaN
    for col in base_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if has_label:
        df["retinal_disorder"] = pd.to_numeric(df["retinal_disorder"], errors="coerce")
    
    rows_before = len(df)
    # Drop rows where any of the required features are NaN after coercion
    df = df.dropna(subset=base_features)
    
    if len(df) < rows_before:
        flash(f"Dropped {rows_before - len(df)} rows with missing/invalid feature values.", "info")
    
    if len(df) == 0:
        flash("No valid rows remaining for prediction after cleaning.", "danger")
        return redirect(url_for("doctor_bulk_home"))

    batch_id = ObjectId()
    # Path for saving the results XLSX on the filesystem
    export_path = os.path.join(EXPORT_DIR, f"bulk_{batch_id}.xlsx")
    
    out_rows = []
    rows_docs = []
    
    try:
        for idx, row in df.iterrows():
            feature_values = {k: float(row[k]) for k in base_features}
            
            try:
                out = compute_ensemble_outputs(feature_values)
            except Exception as e:
                # Log the error but continue processing other rows
                app.logger.error(f"Error processing row {idx} with features {feature_values}: {e}")
                continue
            
            rec = {k: row[k] for k in base_features}
            if has_label:
                rec["retinal_disorder"] = row.get("retinal_disorder")
            
            # ✅ Store PROBABILITIES for each model
            for pm in out["per_model"]:
                rec[f"{pm['name']}_prob"] = pm["probability"]
            
            rec["ensemble_avg_prob"] = out["avg_prob"]
            rec["ensemble_pred"] = out["final_pred"]
            out_rows.append(rec)
            
            per_model_map = {pm["name"]: {"prob": pm["probability"]} for pm in out["per_model"]}
            
            true_label = None
            if has_label and not pd.isna(row["retinal_disorder"]):
                try:
                    true_label = int(row["retinal_disorder"])
                except ValueError:
                    app.logger.warning(f"True label not an integer for row {idx}")
                    
            rows_docs.append({
                "batch_id": batch_id,
                "row_index": int(idx),
                "features": feature_values,
                "true_label": true_label,
                "per_model": per_model_map,
                "ensemble_avg_prob": out["avg_prob"],
                "ensemble_pred": out["final_pred"],
            })
    except RuntimeError as e:
        flash(str(e), "danger") # Model artifacts not loaded
        return redirect(url_for("doctor_bulk_home"))
    except Exception as e:
        app.logger.exception(e)
        flash("An unexpected error occurred during bulk processing.", "danger")
        return redirect(url_for("doctor_bulk_home"))

    if not out_rows:
        flash("No rows were successfully processed.", "danger")
        return redirect(url_for("doctor_bulk_home"))

    try:
        out_df = pd.DataFrame(out_rows)
        # Use OpenPyXL for XLSX writing to disk
        with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="predictions")
    except Exception as e:
        app.logger.exception(e)
        flash("Failed to write XLSX to disk. Ensure openpyxl is installed and disk is accessible.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    batches_col.insert_one({
        "_id": batch_id,
        "doctor_id": ObjectId(current_user.id),
        "filename": filename,
        "file_path": export_path,
        "row_count": len(out_rows),
        "has_label": has_label,
        "created_at": datetime.datetime.utcnow(),
        "model_name": ENSEMBLE_LABEL,
    })
    
    if rows_docs:
        # Bulk insert batch rows
        for i in range(0, len(rows_docs), 1000):
            batch_rows_col.insert_many(rows_docs[i:i+1000])
    
    # FIX: Redirect to dashboard and STOP AUTO-DOWNLOAD
    flash(f"Bulk prediction for {len(out_rows)} rows completed. You can download the results from the list below.", "success")
    return redirect(url_for("doctor_bulk_home"))

@app.route("/doctor/bulk/<bid>", methods=["GET"])
@login_required
def doctor_bulk_view(bid):
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    if batches_col is None or batch_rows_col is None:
        flash("Database connection error.", "danger")
        return redirect(url_for("doctor_bulk_home"))

    try:
        batch_id = ObjectId(bid)
    except Exception:
        abort(404)
    
    # Authorization check
    query = {"_id": batch_id}
    if not is_admin():
        query["doctor_id"] = ObjectId(current_user.id)
        
    batch = batches_col.find_one(query)
    
    if not batch:
        abort(404)
    
    order = request.args.get("order", "desc")
    q = request.args.get("q", "").strip().lower()
    sort_dir = DESCENDING if order == "desc" else ASCENDING
    
    cursor = batch_rows_col.find({"batch_id": batch_id}).sort("ensemble_avg_prob", sort_dir)
    rows = list(cursor)
    
    base_features = [
        "exudates_count", "hemorrhages_count", "microaneurysms_count",
        "vessel_tortuosity", "macular_thickness", "fasting_glucose",
        "hba1c", "diabetes_duration"
    ]
    
    processed_rows = []
    for r in rows:
        per_model_map = r.get("per_model", {})
        pm_full = {}
        
        for name in MODEL_ORDER:
            # Safely get model probability, defaulting to 0.0
            d = per_model_map.get(name, {}).get("prob", 0.0)
            pm_full[name] = {"prob": d}
        
        row_obj = {
            "features": r.get("features", {}),
            "true_label": r.get("true_label"),
            "per_model": pm_full,
            "ensemble_avg_prob": r.get("ensemble_avg_prob", 0.0),
            "ensemble_pred": r.get("ensemble_pred", 0),
        }
        
        processed_rows.append(row_obj)
    
    if q:
        def row_matches(r):
            # Creates a single searchable string blob
            blob = " ".join([
                *(str(r["features"].get(k, "")) for k in base_features),
                str(r.get("true_label", "")),
                *(f"{m}:{r['per_model'][m]['prob']:.3f}" for m in MODEL_ORDER),
                f"{r['ensemble_pred']}", f"{r['ensemble_avg_prob']:.3f}"
            ]).lower()
            return q in blob
        
        processed_rows = [r for r in processed_rows if row_matches(r)]
    
    batch_ctx = {
        "id_str": str(batch["_id"]),
        "created_str": to_ist_str(batch["created_at"]),
        "filename": batch.get("filename", ""),
        "row_count": batch.get("row_count", 0),
        "has_label": batch.get("has_label", False),
        "model_name": batch.get("model_name", ENSEMBLE_LABEL),
    }
    
    return render_template(
        "doctor_bulk_view.html", 
        batch=batch_ctx,
        rows=processed_rows,
        model_order=MODEL_ORDER,
        order=order,
        q=q,
        model_name=ENSEMBLE_LABEL
    )

@app.route("/doctor/bulk/<bid>/download", methods=["GET"])
@login_required
def doctor_bulk_download(bid):
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    if batches_col is None:
        flash("Database connection error.", "danger")
        return redirect(url_for("doctor_bulk_home"))

    try:
        batch_id = ObjectId(bid)
    except Exception:
        abort(404)
    
    # Authorization check
    query = {"_id": batch_id}
    if not is_admin():
        query["doctor_id"] = ObjectId(current_user.id)
        
    batch = batches_col.find_one(query)
    
    if not batch:
        abort(404)
    
    path = batch.get("file_path")
    # Check if the file exists on the filesystem
    if not path or not os.path.exists(path):
        flash("Export file missing.", "danger")
        return redirect(url_for("doctor_bulk_view", bid=bid))
    
    return send_file(path, as_attachment=True, download_name=os.path.basename(path),
                      mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.route("/admin")
@login_required
def admin_dashboard():
    if not is_admin():
        abort(403)
    
    if users_col is None or batches_col is None:
        flash("Database connection error.", "danger")
        return render_template("admin_dashboard.html", patients=[], doctors=[], batches=[], model_name="Admin")
        
    patients = list(users_col.find({"role": "patient"}).sort("created_at", DESCENDING))
    doctors = list(users_col.find({"role": "doctor"}).sort("created_at", DESCENDING))
    batches = list(batches_col.find({}).sort("created_at", DESCENDING))
    
    # Cache doctor names for batches
    doctor_cache = {}
    for d in doctors:
        doctor_cache[d["_id"]] = {"name": d["name"], "username": d["username"]}
    
    for u in patients + doctors:
        u["id_str"] = str(u["_id"])
        u["created_str"] = to_ist_str(u.get("created_at", datetime.datetime.utcnow()))
    
    for b in batches:
        b["id_str"] = str(b["_id"])
        b["created_str"] = to_ist_str(b["created_at"])
        d_info = doctor_cache.get(b["doctor_id"], {})
        b["doctor_name"] = d_info.get("name", "Unknown")
        b["doctor_username"] = d_info.get("username", "unknown")
        
    return render_template("admin_dashboard.html",
                           patients=patients, doctors=doctors, batches=batches,
                           model_name="Admin")

@app.route("/admin/user/<uid>/delete", methods=["POST"])
@login_required
def admin_delete_user(uid):
    if not is_admin():
        abort(403)
    
    if users_col is None or reports_col is None or batches_col is None or batch_rows_col is None:
        flash("Database connection error.", "danger")
        return redirect(url_for("admin_dashboard"))
    
    try:
        user_id = ObjectId(uid)
    except Exception:
        flash("Invalid user ID.", "danger")
        return redirect(url_for("admin_dashboard"))
    
    user_doc = users_col.find_one({"_id": user_id})
    if not user_doc:
        flash("User not found.", "danger")
        return redirect(url_for("admin_dashboard"))
    
    role = user_doc.get("role")
    
    if role == "admin":
        if str(user_doc["_id"]) == current_user.id:
            flash("You cannot delete yourself.", "danger")
            return redirect(url_for("admin_dashboard"))
    
    if role == "patient":
        res = reports_col.delete_many({"patient_id": user_id})
        users_col.delete_one({"_id": user_id})
        flash(f"Deleted patient **{user_doc.get('name', 'Unknown')}** and **{res.deleted_count}** reports.", "success")
    
    elif role == "doctor":
        # Unassign doctor from reports
        reports_col.update_many({"doctor_id": user_id}, {"$unset": {"doctor_id": ""}})
        doctor_batches = list(batches_col.find({"doctor_id": user_id}))
        bid_list = [b["_id"] for b in doctor_batches]
        
        # Clean up files on disk
        for b in doctor_batches:
            path = b.get("file_path")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    app.logger.warning(f"Failed to remove export: {path}")
        
        # Delete batch rows and batch records
        if bid_list:
            batch_rows_col.delete_many({"batch_id": {"$in": bid_list}})
        
        batches_col.delete_many({"doctor_id": user_id})
        users_col.delete_one({"_id": user_id})
        flash(f"Deleted doctor **{user_doc.get('name', 'Unknown')}**, their batches, and rows. Their assigned reports are now unassigned.", "success")
    
    else:
        # For any other role
        users_col.delete_one({"_id": user_id})
        flash(f"User **{user_doc.get('name', 'Unknown')}** deleted.", "success")
    
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/batch/<bid>/delete", methods=["POST"])
@login_required
def admin_delete_batch(bid):
    if not is_admin():
        abort(403)
    
    if batches_col is None or batch_rows_col is None:
        flash("Database connection error.", "danger")
        return redirect(url_for("admin_dashboard"))

    try:
        batch_id = ObjectId(bid)
    except Exception:
        flash("Invalid batch ID.", "danger")
        return redirect(url_for("admin_dashboard"))
    
    batch = batches_col.find_one({"_id": batch_id})
    if not batch:
        flash("Batch not found.", "danger")
        return redirect(url_for("admin_dashboard"))
    
    path = batch.get("file_path")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            app.logger.warning(f"Failed to remove export: {path}")
    
    batch_rows_col.delete_many({"batch_id": batch_id})
    batches_col.delete_one({"_id": batch_id})
    
    flash(f"Batch **{batch.get('filename', 'Unknown File')}** and its rows deleted. Export file removed.", "success")
    return redirect(url_for("admin_dashboard"))

if __name__ == "__main__":
    # Use a dynamic port or default to 8000
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
