from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="patient")  # patient / doctor / admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    cases = db.relationship("Case", backref="patient", lazy=True, foreign_keys="Case.patient_id")


class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # Patient-provided info
    age = db.Column(db.String(10))
    gender = db.Column(db.String(20))
    city = db.Column(db.String(100))
    lesion_site = db.Column(db.String(100))
    duration = db.Column(db.String(100))
    pain = db.Column(db.String(10))
    itching = db.Column(db.String(10))
    bleeding = db.Column(db.String(10))
    history = db.Column(db.String(255))

    # Image + AI results
    original_image = db.Column(db.String(255))
    gradcam_image = db.Column(db.String(255))
    predicted_class = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    risk_level = db.Column(db.String(10))
    urgency = db.Column(db.String(100))
    rationale = db.Column(db.Text)

    # Report
    report_id = db.Column(db.String(50))
    pdf_path = db.Column(db.String(255))

    # Doctor review (for later phase — nullable for now)
    status = db.Column(db.String(20), default="pending_review")  # pending_review / reviewed
    reviewed_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    doctor_notes = db.Column(db.Text, nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)