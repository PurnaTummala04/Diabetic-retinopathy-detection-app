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
from xhtml2pdf import pisa

# ✅ --- GridFS REWRITE ---
# Import GridFS for storing files in MongoDB
import gridfs
# --- END GridFS REWRITE ---

# ============================================
# FLASK CONFIGURATION
# ============================================
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "dr_app")

bcrypt = Bcrypt(app)

# ============================================
# MONGODB CONNECTION
# ============================================
def _uri_has_placeholders(uri: str) -> bool:
    u = (uri or "").lower()
    return any(tok in u for tok in ["<", ">", "$", "?", "&"])

# Create global variables for the db and file storage
client = None
db = None
fs = None

def connect_mongo():
    global client, db, fs
    tried = []
    primary_uri = MONGO_URI
    if not _uri_has_placeholders(primary_uri):
        try:
            client = MongoClient(primary_uri, serverSelectionTimeoutMS=8000)
            client.admin.command("ping")
            try:
                db = client.get_default_database()
                if not db or db.name in (None, "", "admin"):
                    db = client[MONGO_DB_NAME]
            except Exception:
                db = client[MONGO_DB_NAME]
            
            # ✅ --- GridFS REWRITE ---
            # Initialize GridFS with our database
            fs = gridfs.GridFS(db)
            # --- END GridFS REWRITE ---
            
            app.logger.info(f"Connected to MongoDB: {primary_uri}")
            return client, db
        except Exception as e:
            tried.append(f"{primary_uri} -> {e}")
    
    local_uri = f"mongodb://localhost:27017/{MONGO_DB_NAME}"
    try:
        client = MongoClient(local_uri, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        db = client[MONGO_DB_NAME]
        
        # ✅ --- GridFS REWRITE ---
        fs = gridfs.GridFS(db)
        # --- END GridFS REWRITE ---
        
        app.logger.warning(f"Falling back to local MongoDB: {local_uri}")
        return client, db
    except Exception as e:
        tried.append(f"{local_uri} -> {e}")
    
    try:
        import mongomock
        client = mongomock.MongoClient()
        db = client[MONGO_DB_NAME]
        
        # ✅ --- GridFS REWRITE ---
        fs = gridfs.GridFS(db)
        # --- END GridFS REWRITE ---
        
        app.logger.warning("Using in-memory MongoDB via mongomock (data will NOT persist).")
        return client, db
    except Exception as e:
        tried.append(f"mongomock -> {e}")
    
    raise RuntimeError("Could not connect to MongoDB. Please set a valid MONGO_URI or run MongoDB locally.")

connect_mongo() # Run the connection function
users_col = db["users"]
reports_col = db["reports"]
batches_col = db["batches"]
batch_rows_col = db["batch_rows"]

try:
    users_col.create_index("username", unique=True)
    reports_col.create_index([("probability", DESCENDING), ("created_at", DESCENDING)])
    batches_col.create_index([("doctor_id", ASCENDING), ("created_at", DESCENDING)])
    batch_rows_col.create_index([("batch_id", ASCENDING), ("ensemble_avg_prob", DESCENDING)])
except Exception as e:
    app.logger.warning(f"Index creation warning: {e}")

# ============================================
# LOCAL PATHS FOR MODELS
# ============================================
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"
ENSEMBLE_THRESHOLD_CUSTOM = 0.371

# ============================================
# SEED ADMIN USER
# ============================================
# ... (This section is unchanged) ...
def seed_admin():
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
# ... (This section is unchanged) ...
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
# ... (This section is unchanged) ...
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
# LOCAL ML ARTIFACTS LOADING
# ============================================
# ... (This section is unchanged) ...
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
            app.logger.info(f"Successfully loaded model '{name}' from local disk.")
        except Exception as e:
            app.logger.error(f"Failed to load model {name} at {path}: {e}")
    
    ENSEMBLE_LABEL = f"Ensemble(mean) of {len(MODELS)} model{'s' if len(MODELS) != 1 else ''}"

def compute_ensemble_outputs(feature_dict: dict):
    if SCALER is None or not MODELS:
        raise RuntimeError("Model artifacts not loaded. Check R2 connection.")
    
    X = pd.DataFrame([feature_dict], columns=[
        "exudates_count", "hemorrhages_count", "microaneurysms_count",
        "vessel_tortuosity", "macular_thickness", "fasting_glucose",
        "hba1c", "diabetes_duration"
    ]).copy()
    
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
        per_model_details.append({
            "name": name,
            "probability": prob
        })
        probs.append(prob)
    
    if not probs:
        raise RuntimeError("No models loaded.")
    
    avg_prob = float(np.mean(probs))
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
# ... (index, register, login, logout, patient_dashboard are unchanged) ...
@app.route("/")
def index():
    return render_template("index.html", model_name=ENSEMBLE_LABEL)

@app.route("/register", methods=["GET", "POST"])
def register():
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
    
    q = request.args.get("q", "").strip().lower()
    pid = ObjectId(current_user.id)
    
    reports = list(reports_col.find({"patient_id": pid}).sort("created_at", DESCENDING))
    
    for r in reports:
        did = r.get("doctor_id")
        if did:
            ddoc = users_col.find_one({"_id": did}) or {}
            r["doctor_name"] = ddoc.get("name", "Unknown")
        else:
            r["doctor_name"] = "Not assigned"
            
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

# ============================================
# ✅ --- GridFS REWRITE: new_assessment ---
# ============================================
@app.route("/assessment/new", methods=["GET", "POST"])
@login_required
def new_assessment():
    if current_user.role != "patient":
        abort(403)
    
    doctors = list(users_col.find({"role": "doctor"}).sort("name", ASCENDING))
    
    if request.method == "POST":
        try:
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
        except Exception as e:
            app.logger.exception(e)
            flash("Model artifacts missing. Check local files.", "danger")
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
            "gridfs_file_id": None # Placeholder
        }
        
        # --- PDF Generation is the same ---
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
        
        html = render_template("report_pdf.html", **context)
        pdf_io = io.BytesIO()
        pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf_io)
        
        if pisa_status.err:
            flash("Assessment created but PDF generation failed.", "warning")
            return redirect(url_for("patient_dashboard"))
        
        pdf_io.seek(0)
        
        # --- GridFS REWRITE: Upload PDF to GridFS ---
        try:
            # We save the report doc first to get its _id
            res = reports_col.insert_one(report_doc)
            report_id = res.inserted_id
            
            # Use the _id as the filename
            object_name = f"DR_Assessment_{report_id}.pdf"
            
            # Upload the PDF data to GridFS
            file_id = fs.put(
                pdf_io, 
                filename=object_name, 
                contentType="application/pdf"
            )
            
            # Add the GridFS file ID to our report doc
            reports_col.update_one(
                {"_id": report_id},
                {"$set": {"gridfs_file_id": file_id}}
            )
            
            flash("Assessment created successfully.", "success")
            # Redirect to the download function to start the download
            return redirect(url_for('download_report_pdf', report_id=str(report_id)))

        except Exception as e:
            app.logger.error(f"Failed to upload PDF to GridFS: {e}")
            flash("Assessment created, but PDF save failed.", "danger")
            return redirect(url_for("patient_dashboard"))
    
    return render_template("report_form.html", doctors=doctors, model_name=ENSEMBLE_LABEL)

# ============================================
# ✅ --- GridFS REWRITE: download_report_pdf ---
# ============================================
@app.route("/report/<report_id>/download")
@login_required
def download_report_pdf(report_id):
    # --- Check Permissions (unchanged) ---
    if not is_admin() and current_user.role != "doctor":
        try:
            report_check = reports_col.find_one(
                {"_id": ObjectId(report_id), "patient_id": ObjectId(current_user.id)}
            )
            if not report_check:
                flash("You do not have permission to view this report.", "danger")
                return redirect(url_for("patient_dashboard"))
        except Exception:
            abort(403)
    
    try:
        report_doc = reports_col.find_one({"_id": ObjectId(report_id)})
        if not report_doc:
            abort(404)
    except Exception:
        abort(404)

    # --- GridFS REWRITE: Get file from GridFS ---
    file_id = report_doc.get("gridfs_file_id")
    if not file_id:
        flash("PDF not found for this report. It may be an old record.", "danger")
        return redirect(request.referrer or url_for("patient_dashboard"))

    try:
        grid_file = fs.get(file_id)
        return send_file(
            grid_file,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=grid_file.filename
        )
    except Exception as e:
        app.logger.error(f"Failed to get file from GridFS: {e}")
        flash("Could not find or retrieve the PDF file from the database.", "danger")
        return redirect(request.referrer or url_for("patient_dashboard"))

# ============================================
# ... (doctor_dashboard is unchanged) ...
@app.route("/doctor")
@login_required
def doctor_dashboard():
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    q = request.args.get("q", "").strip().lower()
    order = request.args.get("order", "desc")
    sort_dir = DESCENDING if order == "desc" else ASCENDING
    
    reports = list(reports_col.find({}).sort("probability", sort_dir))
    
    cache = {}
    processed_reports = []
    for r in reports:
        pid = r.get("patient_id")
        if not pid:
            continue
            
        did = r.get("doctor_id")
        
        if pid not in cache:
            pdoc = users_col.find_one({"_id": pid}) or {}
            cache[pid] = pdoc.get("name", "Unknown")
        
        if did and did not in cache:
            ddoc = users_col.find_one({"_id": did}) or {}
            cache[did] = ddoc.get("name", "Unknown")
        
        r["patient_name"] = cache.get(pid, "Unknown")
        r["doctor_name"] = cache.get(did, "Not assigned")
        r["id_str"] = str(r["_id"])
        r["created_str"] = to_ist_str(r.get("created_at"))
        processed_reports.append(r)
    
    reports = processed_reports

    if q:
        filtered = []
        for r in reports:
            hay = f"{r['patient_name']} {r['doctor_name']} {r['prediction']} {r.get('model_name','')} {r['created_str']} {r.get('probability',0.0):.3f}".lower()
            if q in hay:
                filtered.append(r)
        reports = filtered
    
    return render_template("doctor_dashboard.html", reports=reports, order=order, q=q, model_name=ENSEMBLE_LABEL)

# ... (allowed_file and doctor_bulk_home are unchanged) ...
ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/doctor/bulk", methods=["GET"])
@login_required
def doctor_bulk_home():
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    q = request.args.get("q", "").strip().lower()
    
    if is_admin():
        batches = list(batches_col.find({}).sort("created_at", DESCENDING))
    else:
        batches = list(batches_col.find({"doctor_id": ObjectId(current_user.id)}).sort("created_at", DESCENDING))
    
    for b in batches:
        b["id_str"] = str(b["_id"])
        b["created_str"] = to_ist_str(b.get("created_at"))
    
    if q:
        batches = [b for b in batches if q in (b.get("filename", "").lower()) or q in b["created_str"].lower()]
    
    return render_template("doctor_bulk.html", batches=batches, q=q, model_name=ENSEMBLE_LABEL)

# ============================================
# ✅ --- GridFS REWRITE: doctor_bulk_upload ---
# ============================================
@app.route("/doctor/bulk/upload", methods=["POST"])
@login_required
def doctor_bulk_upload():
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Please choose a CSV file.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    if not allowed_file(f.filename):
        flash("Only CSV files are allowed.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    filename = secure_filename(f.filename)
    
    try:
        df = pd.read_csv(f)
    except Exception:
        flash("Failed to read CSV. Check the file format.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
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
    
    for col in base_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if has_label:
        df["retinal_disorder"] = pd.to_numeric(df["retinal_disorder"], errors="coerce")
    
    rows_before = len(df)
    df = df.dropna(subset=base_features)
    
    if len(df) < rows_before:
        flash(f"Dropped {rows_before - len(df)} rows with missing/invalid feature values.", "info")
    
    batch_id = ObjectId()
    
    out_rows = []
    rows_docs = []
    
    for idx, row in df.iterrows():
        feature_values = {k: float(row[k]) for k in base_features}
        
        try:
            out = compute_ensemble_outputs(feature_values)
        except Exception as e:
            app.logger.exception(e)
            flash(f"Error processing row {idx}: {e}", "danger")
            continue
        
        rec = {k: row[k] for k in base_features}
        if has_label:
            rec["retinal_disorder"] = row.get("retinal_disorder")
        
        for pm in out["per_model"]:
            rec[f"{pm['name']}_prob"] = pm["probability"]
        
        rec["ensemble_avg_prob"] = out["avg_prob"]
        rec["ensemble_pred"] = out["final_pred"]
        out_rows.append(rec)
        
        per_model_map = {pm["name"]: {"prob": pm["probability"]} for pm in out["per_model"]}
        rows_docs.append({
            "batch_id": batch_id,
            "row_index": int(idx),
            "features": feature_values,
            "true_label": int(row["retinal_disorder"]) if has_label and not pd.isna(row["retinal_disorder"]) else None,
            "per_model": per_model_map,
            "ensemble_avg_prob": out["avg_prob"],
            "ensemble_pred": out["final_pred"],
        })
    
    # --- GridFS REWRITE: Upload XLSX to GridFS ---
    try:
        out_df = pd.DataFrame(out_rows)
        xlsx_io = io.BytesIO()
        with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="predictions")
        xlsx_io.seek(0)
        
        # Upload the XLSX data to GridFS
        file_id = fs.put(
            xlsx_io, 
            filename=f"bulk_{batch_id}.xlsx", 
            contentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        app.logger.exception(e)
        flash("Failed to write and upload XLSX. Ensure openpyxl is installed.", "danger")
        return redirect(url_for("doctor_bulk_home"))
    
    batches_col.insert_one({
        "_id": batch_id,
        "doctor_id": ObjectId(current_user.id),
        "filename": filename,
        "gridfs_file_id": file_id,  # Store the GridFS ID
        "row_count": len(out_rows),
        "has_label": has_label,
        "created_at": datetime.datetime.utcnow(),
        "model_name": ENSEMBLE_LABEL,
    })
    
    if rows_docs:
        for i in range(0, len(rows_docs), 1000):
            batch_rows_col.insert_many(rows_docs[i:i+1000])
    
    # --- GridFS REWRITE: Redirect to the download function ---
    flash("Batch processed successfully.", "success")
    return redirect(url_for('doctor_bulk_download', bid=str(batch_id)))

# ... (doctor_bulk_view is unchanged) ...
@app.route("/doctor/bulk/<bid>", methods=["GET"])
@login_required
def doctor_bulk_view(bid):
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    try:
        batch_id = ObjectId(bid)
    except Exception:
        abort(404)
    
    if is_admin():
        batch = batches_col.find_one({"_id": batch_id})
    else:
        batch = batches_col.find_one({"_id": batch_id, "doctor_id": ObjectId(current_user.id)})
    
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
            d = per_model_map.get(name, {"prob": 0.0})
            pm_full[name] = d
        
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
        "created_str": to_ist_str(batch.get("created_at")),
        "filename": batch.get("filename", ""),
        "row_count": batch.get("row_count", 0),
        "has_label": batch.get("has_label", False),
        "model_name": batch.get("model_name", ENSEMBLE_LABEL),
    }
    
    return render_template(
        "doctor_bulk.html",
        batch=batch_ctx,
        rows=processed_rows,
        model_order=MODEL_ORDER,
        order=order,
        q=q,
        model_name=ENSEMBLE_LABEL
    )

# ============================================
# ✅ --- GridFS REWRITE: doctor_bulk_download ---
# ============================================
@app.route("/doctor/bulk/<bid>/download", methods=["GET"])
@login_required
def doctor_bulk_download(bid):
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    
    try:
        batch_id = ObjectId(bid)
    except Exception:
        abort(404)
    
    if is_admin():
        batch = batches_col.find_one({"_id": batch_id})
    else:
        batch = batches_col.find_one({"_id": batch_id, "doctor_id": ObjectId(current_user.id)})
    
    if not batch:
        abort(404)
    
    # --- GridFS REWRITE: Get file from GridFS ---
    file_id = batch.get("gridfs_file_id")
    if not file_id:
        flash("File not found for this batch.", "danger")
        return redirect(url_for("doctor_bulk_view", bid=bid))

    try:
        grid_file = fs.get(file_id)
        return send_file(
            grid_file,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=grid_file.filename
        )
    except Exception as e:
        app.logger.error(f"Failed to get file from GridFS: {e}")
        flash("Could not find or retrieve the XLSX file from the database.", "danger")
        return redirect(url_for("doctor_bulk_view", bid=bid))

# ... (admin_dashboard is unchanged) ...
@app.route("/admin")
@login_required
def admin_dashboard():
    if not is_admin():
        abort(403)
    
    patients = list(users_col.find({"role": "patient"}).sort("created_at", DESCENDING))
    doctors = list(users_col.find({"role": "doctor"}).sort("created_at", DESCENDING))
    batches = list(batches_col.find({}).sort("created_at", DESCENDING))
    
    for u in patients + doctors:
        u["id_str"] = str(u["_id"])
        u["created_str"] = to_ist_str(u.get("created_at", datetime.datetime.utcnow()))
    
    for b in batches:
        b["id_str"] = str(b["_id"])
        b["created_str"] = to_ist_str(b.get("created_at"))
        
        doctor_id = b.get("doctor_id")
        if doctor_id:
            ddoc = users_col.find_one({"_id": doctor_id}) or {}
            b["doctor_name"] = ddoc.get("name", "Unknown")
            b["doctor_username"] = ddoc.get("username", "")
        else:
            b["doctor_name"] = "N/A (No Doctor)"
            b["doctor_username"] = ""
    
    return render_template("admin_dashboard.html",
                          patients=patients, doctors=doctors, batches=batches,
                          model_name="Admin")

# ============================================
# ✅ --- GridFS REWRITE: admin_delete_user ---
# ============================================
@app.route("/admin/user/<uid>/delete", methods=["POST"])
@login_required
def admin_delete_user(uid):
    if not is_admin():
        abort(403)
    
    try:
        user_id = ObjectId(uid)
    except Exception:
        abort(404)
    
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
        # --- GridFS REWRITE: Delete reports from GridFS ---
        patient_reports = list(reports_col.find({"patient_id": user_id}))
        for r in patient_reports:
            if r.get("gridfs_file_id"):
                try:
                    fs.delete(r["gridfs_file_id"])
                except Exception as e:
                    app.logger.warning(f"Failed to delete GridFS file {r['gridfs_file_id']}: {e}")
        
        res = reports_col.delete_many({"patient_id": user_id})
        users_col.delete_one({"_id": user_id})
        flash(f"Deleted patient, {res.deleted_count} reports, and associated files.", "success")
    
    elif role == "doctor":
        reports_col.update_many({"doctor_id": user_id}, {"$unset": {"doctor_id": ""}})
        doctor_batches = list(batches_col.find({"doctor_id": user_id}))
        bid_list = [b["_id"] for b in doctor_batches]
        
        # --- GridFS REWRITE: Delete batch files from GridFS ---
        for b in doctor_batches:
            if b.get("gridfs_file_id"):
                try:
                    fs.delete(b["gridfs_file_id"])
                except Exception as e:
                    app.logger.warning(f"Failed to delete GridFS file {b['gridfs_file_id']}: {e}")

        if bid_list:
            batch_rows_col.delete_many({"batch_id": {"$in": bid_list}})
        
        batches_col.delete_many({"doctor_id": user_id})
        users_col.delete_one({"_id": user_id})
        flash(f"Deleted doctor, their batches, rows, and associated files.", "success")
    
    else:
        users_col.delete_one({"_id": user_id})
        flash("User deleted.", "success")
    
    return redirect(url_for("admin_dashboard"))

# ============================================
# ✅ --- GridFS REWRITE: admin_delete_batch ---
# ============================================
@app.route("/admin/batch/<bid>/delete", methods=["POST"])
@login_required
def admin_delete_batch(bid):
    if not is_admin():
        abort(403)
    
    try:
        batch_id = ObjectId(bid)
    except Exception:
        abort(404)
    
    batch = batches_col.find_one({"_id": batch_id})
    if not batch:
        flash("Batch not found.", "danger")
        return redirect(url_for("admin_dashboard"))
    
    # --- GridFS REWRITE: Delete file from GridFS ---
    if batch.get("gridfs_file_id"):
        try:
            fs.delete(batch["gridfs_file_id"])
        except Exception as e:
            app.logger.warning(f"Failed to delete GridFS file {batch['gridfs_file_id']}: {e}")
    
    batch_rows_col.delete_many({"batch_id": batch_id})
    batches_col.delete_one({"_id": batch_id})
    
    flash("Batch, rows, and associated file deleted.", "success")
    return redirect(url_for("admin_dashboard"))

# ============================================
# RUNNER
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)