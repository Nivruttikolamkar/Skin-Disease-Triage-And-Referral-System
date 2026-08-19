"""Flask application for the Intelligent Dermatology Triage system."""
from datetime import datetime
from functools import wraps
import os
import uuid

import tensorflow as tf
from flask import (
    Flask, request, render_template, redirect,
    url_for, flash, send_from_directory
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

from models import db, User, Case
from model import IMG_SIZE
from gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_gradcam
from triage_logic import get_risk_level, is_valid_prediction
from pdf_report import generate_pdf_report
from roboflow_api import get_cancer_risk_screening
from dermatologist_search import get_nearby_dermatologists

BASE_DIR = r"C:\Users\VICTUS\Videos\Final_year_project"
MODEL_PATH = os.path.join(BASE_DIR, "training_artifacts",
                          "final_model_consolidated.keras")
CLASS_LIST_FILE = os.path.join(BASE_DIR, "class_list_consolidated.txt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE_MB = 5

# 1. Initialize the Flask application once
app = Flask(__name__)
app.config["SECRET_KEY"] = "a11d7819b2e12c70e2aceda7eac73be4389befa54cc873e85cd48e346919b3e3"
app.secret_key = app.config["SECRET_KEY"]
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["GOOGLE_MAPS_API_KEY"] = os.environ.get("GOOGLE_MAPS_API_KEY", "")
db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    """Reload a logged-in user from the session's stored user id."""
    return db.session.get(User, int(user_id))


# ===== Load model once at startup =====
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
base_model, last_conv_layer_name = find_last_conv_layer(model)
with open(CLASS_LIST_FILE, "r", encoding="utf-8") as class_file:
    class_names = [line.strip()
                   for line in class_file.readlines() if line.strip()]
preprocess_input = tf.keras.applications.efficientnet.preprocess_input
print("Model loaded.")


def allowed_file(filename):
    """Return True if the uploaded filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def admin_required(f):
    """Restrict a route to logged-in users with the admin role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("Admin access only.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def doctor_required(f):
    """Restrict a route to logged-in users with the doctor role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "doctor":
            flash("Doctor access only.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ===== Landing Page & Auth Routes =====
@app.route("/")
def landing():
    """Render the professional skin triage landing page."""
    return render_template("landing.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle patient self-registration."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("login"))

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role="patient"
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle login for all roles (patient, doctor, admin)."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.role == "admin":
                return redirect(url_for("admin_dashboard"))
            if user.role == "doctor":
                return redirect(url_for("doctor_dashboard"))
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    """Log the current user out."""
    logout_user()
    return redirect(url_for("login"))


# ===== Patient dashboard =====
@app.route("/dashboard")
@login_required
def dashboard():
    """Show the patient's upload form and their past cases."""
    cases = Case.query.filter_by(
        patient_id=current_user.id
    ).order_by(Case.created_at.desc()).all()
    return render_template("dashboard.html", cases=cases)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """Validate image, run local model & Roboflow workflow, save case and generate report."""

    # 1. Check file upload (support both 'file' and 'image' input keys)
    file = None
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
    elif 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']

    if not file or not allowed_file(file.filename):
        flash("Please select a valid JPG or PNG image to upload.", "error")
        return redirect(url_for("dashboard"))

    # 2. Extract clinical metadata from form safely
    age = request.form.get("age", "N/A")
    gender = request.form.get("gender", "N/A")
    city = request.form.get("city", "N/A").strip()
    lesion_site = request.form.get("lesion_site", "N/A")
    duration = request.form.get("duration", "N/A")
    pain = request.form.get("pain", "No")
    itching = request.form.get("itching", "No")
    bleeding = request.form.get("bleeding", "No")
    history = request.form.get("history", "None")

    # 3. Save uploaded image
    unique_id = uuid.uuid4().hex[:8]
    safe_filename = f"{unique_id}_{file.filename}".replace(" ", "_")
    img_path = os.path.join(UPLOAD_FOLDER, safe_filename)

    try:
        file.save(img_path)
    except OSError:
        flash("Failed to save the uploaded image. Please try again.", "error")
        return redirect(url_for("dashboard"))

    # 4. Validate image decode
    try:
        img_raw = tf.io.read_file(img_path)
        img_decoded = tf.image.decode_image(
            img_raw, channels=3, expand_animations=False)
        img_decoded.set_shape([None, None, 3])
    except (tf.errors.OpError, ValueError):
        if os.path.exists(img_path):
            os.remove(img_path)
        flash("The uploaded file doesn't appear to be a valid image.", "error")
        return redirect(url_for("dashboard"))

    # 5. Local Model Preprocess & Predict
    try:
        img_resized = tf.image.resize(img_decoded, IMG_SIZE)
        img_resized = tf.cast(img_resized, tf.float32)
        img_preprocessed = preprocess_input(img_resized)
        img_array = tf.expand_dims(img_preprocessed, axis=0)

        heatmap, pred_index, all_probs = make_gradcam_heatmap(
            img_array, model, base_model, last_conv_layer_name
        )
        predicted_class = class_names[pred_index]
        confidence = float(all_probs[pred_index])
    except Exception as exc:
        print(f"Prediction error: {exc}")
        flash("Something went wrong while analyzing the image locally.", "error")
        return redirect(url_for("dashboard"))

    # 6. Secondary Screening / Fallback via Roboflow if local prediction confidence is low
    if not is_valid_prediction(confidence):
        screening = get_cancer_risk_screening(img_path)

        if screening and screening["flagged"]:
            predicted_class = screening["class_name"]
            confidence = screening["confidence"] / 100.0  
            flash(
                f"⚠️ Secondary screening flagged a possible {predicted_class} "
                f"pattern ({screening['confidence']}% confidence). URGENT dermatologist evaluation recommended.",
                "warning",
            )
        else:
            flash(
                "The uploaded image doesn't appear to clearly match any of the "
                "conditions this system is trained on. Please upload a clear, "
                "close-up photo of the affected skin area.",
                "error",
            )
            if os.path.exists(img_path):
                os.remove(img_path)
            return redirect(url_for("dashboard"))

    # 7. Grad-CAM Overlay Generation
    try:
        gradcam_filename = f"gradcam_{unique_id}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        overlay_gradcam(img_path, heatmap, save_path=gradcam_path)
    except Exception as exc:
        print(f"Grad-CAM error: {exc}")
        gradcam_filename = None
        gradcam_path = None

    # 8. Risk Level Evaluation
    risk = get_risk_level(predicted_class, confidence)

    # 9. Save Case to Database
    try:
        case = Case(
            patient_id=current_user.id,
            age=str(age), gender=str(gender), city=str(city), lesion_site=str(lesion_site), duration=str(duration),
            pain=str(pain), itching=str(itching), bleeding=str(bleeding), history=str(history),
            original_image=f"uploads/{safe_filename}",
            gradcam_image=(
                f"uploads/{gradcam_filename}" if gradcam_filename
                else f"uploads/{safe_filename}"
            ),
            predicted_class=predicted_class,
            confidence=round(confidence * 100, 2),
            risk_level=risk["risk_level"],
            urgency=risk["urgency"],
            rationale=risk["rationale"],
        )
        db.session.add(case)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        print(f"Database error: {exc}")
        flash("Something went wrong saving your case to the database.", "error")
        return redirect(url_for("dashboard"))

    # 10. Fetch Nearby Dermatologists (Google Maps API) & Generate PDF Report
    nearby_info = get_nearby_dermatologists(city, app.config.get("GOOGLE_MAPS_API_KEY"))

    patient_dict = {
        "name": current_user.name,
        "age": age, "gender": gender, "city": city,
        "email": current_user.email, "phone": "-",
        "lesion_site": lesion_site, "duration": duration,
        "pain": pain, "itching": itching, "bleeding": bleeding,
        "history": history,
    }

    try:
        pdf_filename = f"report_{unique_id}.pdf"
        pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)
        report_id, _ = generate_pdf_report(
            patient_dict, {"predicted_class": predicted_class, "confidence": round(confidence * 100, 2)}, 
            risk, img_path, gradcam_path or img_path, pdf_path, case.id, nearby_info=nearby_info
        )
        case.report_id = report_id
        case.pdf_path = pdf_filename
        db.session.commit()
    except Exception as exc:
        print(f"PDF generation error: {exc}")
        flash("Analysis complete, but PDF report generation failed.", "error")

    return redirect(url_for("view_result", case_id=case.id))


@app.route("/result/<int:case_id>")
@login_required
def view_result(case_id):
    """Show a single case's AI result to the patient who owns it."""
    case = Case.query.get_or_404(case_id)
    if case.patient_id != current_user.id:
        flash("Not authorized.", "error")
        return redirect(url_for("dashboard"))

    workflow_result = None
    try:
        full_img_path = os.path.join(BASE_DIR, "static", case.original_image)
        if os.path.exists(full_img_path):
            workflow_result = get_cancer_risk_screening(full_img_path)
    except Exception:
        pass

    nearby_info = get_nearby_dermatologists(case.city or "N/A", app.config.get("GOOGLE_MAPS_API_KEY"))

    return render_template(
        "result.html",
        case=case,
        workflow_result=workflow_result,
        nearby_info=nearby_info,
        google_maps_key=app.config.get("GOOGLE_MAPS_API_KEY")
    )


@app.route("/download/<filename>")
@login_required
def download_report(filename):
    """Serve a generated PDF report for download."""
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


# ===== Admin routes =====
@app.route("/admin/dashboard")
@login_required
@admin_required
def admin_dashboard():
    """Show admin-level stats and recent cases."""
    total_patients = User.query.filter_by(role="patient").count()
    total_doctors = User.query.filter_by(role="doctor").count()
    total_cases = Case.query.count()
    pending_cases = Case.query.filter_by(status="pending_review").count()
    recent_cases = Case.query.order_by(Case.created_at.desc()).limit(10).all()

    return render_template(
        "admin_dashboard.html",
        total_patients=total_patients,
        total_doctors=total_doctors,
        total_cases=total_cases,
        pending_cases=pending_cases,
        recent_cases=recent_cases
    )


@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    """List all registered users for the admin."""
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin_users.html", users=users)


@app.route("/admin/create-doctor", methods=["GET", "POST"])
@login_required
@admin_required
def create_doctor():
    """Let an admin create a new doctor account."""
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            flash("This email is already registered.", "error")
        else:
            doctor = User(
                name=name,
                email=email,
                password_hash=generate_password_hash(password),
                role="doctor"
            )
            db.session.add(doctor)
            db.session.commit()
            flash(f"Doctor account created: {name}", "success")
            return redirect(url_for("admin_users"))

    return render_template("create_doctor.html")


@app.route("/admin/cases")
@login_required
@admin_required
def admin_cases():
    """List every case in the system for the admin."""
    cases = Case.query.order_by(Case.created_at.desc()).all()
    return render_template("admin_cases.html", cases=cases)


# ===== Doctor routes =====
@app.route("/doctor/dashboard")
@login_required
@doctor_required
def doctor_dashboard():
    """Show a doctor their pending and recently reviewed cases."""
    pending_cases = Case.query.filter_by(
        status="pending_review"
    ).order_by(Case.created_at.desc()).all()
    reviewed_cases = Case.query.filter_by(
        status="reviewed"
    ).order_by(Case.reviewed_at.desc()).limit(10).all()
    high_risk_count = Case.query.filter_by(
        status="pending_review", risk_level="RED"
    ).count()
    return render_template(
        "doctor_dashboard.html",
        pending_cases=pending_cases,
        reviewed_cases=reviewed_cases,
        high_risk_count=high_risk_count
    )


@app.route("/doctor/case/<int:case_id>", methods=["GET", "POST"])
@login_required
@doctor_required
def doctor_case_review(case_id):
    """Let a doctor view and review a single case."""
    case = Case.query.get_or_404(case_id)

    if request.method == "POST":
        case.doctor_notes = request.form.get("doctor_notes", "")
        case.status = "reviewed"
        case.reviewed_by = current_user.id
        case.reviewed_at = datetime.utcnow()
        db.session.commit()
        flash("Review submitted successfully.", "success")
        return redirect(url_for("doctor_dashboard"))

    return render_template("doctor_case_review.html", case=case)


@app.errorhandler(404)
def not_found(_error):
    """Show a styled 404 page instead of Flask's default."""
    return render_template(
        "error.html", error_code=404, error_message="Page not found."
    ), 404


@app.errorhandler(500)
def server_error(_error):
    """Show a styled 500 page instead of Flask's default."""
    return render_template(
        "error.html", error_code=500,
        error_message="Something went wrong on our end."
    ), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)