# 🩺 DermAI Triage: Intelligent Dermatology Screening & Specialist Referral System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2Fkeras-orange.svg)](https://www.tensorflow.org/)
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM-red.svg)](https://arxiv.org/abs/1610.02391)
[![Google Maps API](https://img.shields.io/badge/Integration-Google%20Maps%20Places%20API-4285F4.svg)](https://developers.google.com/maps)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, AI-assisted medical decision support system that combines **Deep Convolutional Neural Networks (EfficientNet)** with **Explainable AI (Grad-CAM visual heatmaps)**, **Clinical Metadata**, and **Google Maps Location APIs** to classify skin lesions, stratify patient risk, generate structured PDF medical reports, and route high-risk cases to nearby board-certified dermatologists.

---

## 📌 Table of Contents

- [System Overview](#-system-overview)
- [Key Features](#-key-features)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Tri-Color Urgency Triage Matrix](#-tri-color-urgency-triage-matrix)
- [Directory & File Structure](#-directory--file-structure)
- [Installation & Setup Guide](#-installation--setup-guide)
- [Environment Configuration](#-environment-configuration)
- [User Roles & Walkthrough](#-user-roles--walkthrough)
- [Explainable AI & PDF Generation](#-explainable-ai--pdf-generation)
- [Google Maps Dermatologist Referral](#-google-maps-dermatologist-referral)
- [Medical & Clinical Disclaimer](#-medical--clinical-disclaimer)

---

## 🔬 System Overview

Dermatological conditions account for a large volume of outpatient consultations, often leading to long clinical waiting lists. **DermAI Triage** bridges the gap between initial patient symptom reporting and specialist care by providing an intelligent preliminary screening platform.

### Core Objectives
1. **Accelerate Clinical Triage**: Stratify incoming skin cases into Red (Urgent), Yellow (Moderate), and Green (Low Risk) priority tiers.
2. **Explainable AI (XAI)**: Generate Gradient-weighted Class Activation Maps (Grad-CAM) to highlight exact lesion areas influencing the AI's classification.
3. **Secondary Cancer Screening**: Integrate secondary fallback screening via Roboflow API to detect potential high-risk malignant patterns.
4. **Location-Aware Referrals**: Recommend nearby dermatologist clinics using Google Maps Places API based on the patient's city.
5. **Clinical Collaboration**: Provide dedicated doctor queues where dermatologists can review visual heatmaps, clinical history, and attach official medical evaluations.

---

## ✨ Key Features

- 🧑‍🦲 **Patient Portal**: Self-registration, symptom history intake (pain, itching, duration, site, bleeding), image upload, and case tracking.
- 🧠 **Deep Learning Diagnostic Engine**: Fine-tuned EfficientNet model trained on multi-class dermatological datasets (Scabies, Psoriasis, Eczema, Tinea, Acne, Vitiligo, Melasma, Alopecia Areata, Candidal Intertrigo).
- 🔥 **Grad-CAM Visual Saliency Maps**: Generates visual heatmaps overlaying pathological regions of interest to give visual rationale for predictions.
- 🛡️ **Dual-Layer Screening**: Secondary Roboflow cloud API screening for low-confidence or borderline malignant pattern verification.
- 🚦 **Tri-Color Risk Stratification**: Automated clinical decision matrix evaluating severity ranks and confidence bounds.
- 🗺️ **Google Maps Specialist Referral**: Location-based search displaying nearby dermatologists, addresses, star ratings, review counts, and direct navigation links.
- 📄 **Automated PDF Medical Reports**: Compiles patient metadata, original & Grad-CAM visual overlays, risk rationale, precautions, nearby doctor listings, and a verification QR code.
- 🩺 **Doctor Review Portal**: Prioritized clinical queue sorted by urgency (Red priority first), allowing doctors to enter official review notes and sign off on cases.
- 📊 **Admin Dashboard**: Full system oversight, user role management, doctor account creation, and system analytics.

---

## 🏗️ System Architecture & Workflow

```
 ┌────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
 │ Patient Upload │ ───► │  EfficientNet Deep Model  │ ───► │   Grad-CAM XAI Heatmap    │
 │ (Image + City) │      │   Multi-Class Prediction  │      │  Pathological Highlights  │
 └────────────────┘      └───────────────────────────┘      └─────────────┬─────────────┘
                                                                          │
 ┌────────────────┐      ┌───────────────────────────┐                    │
 │ Google Maps    │ ◄─── │ Tri-Color Triage Matrix   │ ◄──────────────────┘
 │ Doctor Search  │      │ (RED / YELLOW / GREEN)    │
 └───────┬────────┘      └─────────────┬─────────────┘
         │                             │
         ▼                             ▼
 ┌───────────────────────────────────────────────────┐      ┌───────────────────────────┐
 │ Automated PDF Summary Report + Verification QR    │ ───► │ Specialist (Doctor) Queue │
 └───────────────────────────────────────────────────┘      │ Review, Notes & Sign-Off  │
                                                            └───────────────────────────┘
```

---

## 🚦 Tri-Color Urgency Triage Matrix

The triage engine evaluates disease severity combined with prediction confidence to assign risk levels:

| Risk Tier | Level | Action Target | Clinical Rationale & Handling |
| :--- | :--- | :--- | :--- |
| **RED** | **High Risk / Urgent** | 24 - 48 Hours | Severe/contagious conditions or low confidence predictions. Priority placement in dermatologist review queue. |
| **YELLOW** | **Moderate Risk** | 1 - 2 Weeks | Chronic or active inflammatory disorders (Psoriasis, Eczema). Scheduled outpatient consultation recommended. |
| **GREEN** | **Low Risk** | Self-Care / Follow-up | Mild or benign dermatological conditions (Acne, Melasma, Vitiligo). Educational self-monitoring guidance provided. |

---

## 📁 Directory & File Structure

```text
Final_year_project/
├── app.py                      # Core Flask Application Backend & Route Handlers
├── models.py                   # SQLAlchemy Database Models (User & Case)
├── model.py                    # Model Config & Image Size Preprocessing Definitions
├── gradcam.py                  # Grad-CAM Heatmap Generation Engine
├── triage_logic.py             # Tri-Color Risk Stratification Logic & Severity Mapping
├── pdf_report.py               # ReportLab PDF Generation & Verification QR Code Script
├── roboflow_api.py             # Roboflow Secondary Cancer Screening Integration
├── dermatologist_search.py     # Google Maps Places API Integration & Fallback Engine
├── migrate_db.py               # SQLite Schema Migration Utility
├── train.py                    # Deep Learning Model Training Script
├── evaluate.py                 # Model Evaluation & Metrics Utility
├── class_list_consolidated.txt # Supported Disease Categories
├── instance/
│   └── database.db             # SQLite Production Database
├── static/
│   ├── css/
│   │   └── style.css           # Modern Custom Medical Theme Stylesheet
│   ├── img/                    # Illustrations & SVG Assets
│   ├── uploads/                # Uploaded Lesions & Grad-CAM Overlays
│   └── reports/                # Generated Medical PDF Reports
├── templates/
│   ├── base.html               # Master Layout Shell & Navigation
│   ├── landing.html            # Public Clinical Landing Page
│   ├── dashboard.html          # Patient Dashboard & Analysis Form
│   ├── result.html             # Case Analysis Result & Nearby Doctors View
│   ├── doctor_dashboard.html   # Specialist Review Queue & Prioritized Cases
│   ├── doctor_case_review.html # Single Case Doctor Review Form
│   ├── admin_dashboard.html    # System Admin Stats & Case Overview
│   ├── admin_users.html        # Registered User Management Page
│   ├── create_doctor.html      # Admin Doctor Account Creation Page
│   ├── login.html              # Unified Auth Login Page
│   ├── register.html           # Patient Self-Registration Page
│   └── error.html              # Custom 404/500 Error Page
└── README.md                   # System Documentation
```

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.10 or higher installed.
- Pip package manager.

### 2. Clone / Open Project Directory
```bash
cd C:\Users\VICTUS\Videos\Final_year_project
```

### 3. Create & Activate Virtual Environment *(Optional but Recommended)*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install required Python libraries:
```bash
pip install flask flask-sqlalchemy flask-login werkzeug tensorflow opencv-python reportlab qrcode pillow requests
```

### 5. Run Database Migration
Ensure all database columns (including `city`) are synced:
```bash
python migrate_db.py
```

### 6. Start the Flask Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000/`**

---

## 🔑 Environment Configuration

You can configure your Google Maps Places API key to enable live local doctor searches.

### Setting API Key via Environment Variables

**Windows PowerShell:**
```powershell
$env:GOOGLE_MAPS_API_KEY="YOUR_ACTUAL_GOOGLE_MAPS_API_KEY"
python app.py
```

**Windows CMD:**
```cmd
set GOOGLE_MAPS_API_KEY=YOUR_ACTUAL_GOOGLE_MAPS_API_KEY
python app.py
```

**Linux / macOS:**
```bash
export GOOGLE_MAPS_API_KEY="YOUR_ACTUAL_GOOGLE_MAPS_API_KEY"
python app.py
```

> *Note: If no API key is set, the application uses an automatic fallback search engine generating direct Google Maps location query links for the patient's city.*

---

## 👥 User Roles & Walkthrough

| Role | Access Permissions & Responsibilities | Default Route |
| :--- | :--- | :--- |
| **Patient** | Upload skin photos, input symptom details & city, receive instant AI triage risk score, download PDF reports, and view nearby dermatologists. | `/dashboard` |
| **Doctor** | View high-risk (RED) pending cases, inspect Grad-CAM saliency heatmaps, enter clinical diagnosis notes, and approve referrals. | `/doctor/dashboard` |
| **Admin** | Full system analytics, manage registered patients/doctors, view system logs, and create verified doctor accounts. | `/admin/dashboard` |

---

## 🔬 Explainable AI & PDF Generation

### Grad-CAM Visual Heatmaps
The system computes gradients of the target class score with respect to the last convolutional layer of EfficientNet. The resulting heatmap is superimposed on the original skin photograph:
- **Red/Yellow Focus Areas**: Highlight pathological texture and color anomalies driving the model's classification.
- **Blue/Cool Regions**: Represent background healthy tissue ignored during classification.

### Verification QR Code
Every PDF report includes a embedded QR code linking directly to `http://127.0.0.1:5000/result/<case_id>` for instant clinical verification.

---

## 🗺️ Google Maps Dermatologist Referral

When a patient inputs their city (*e.g., Chicago, London, Mumbai, Delhi*):
1. **Places Text Search API** queries `dermatologist in {city}`.
2. Returns top-rated medical centers with name, address, rating, review counts, and opening status.
3. Renders interactive **🗺️ Open in Google Maps** navigation buttons on the dashboard and embeds clinic summary tables inside the PDF report.

---

## ⚠️ Medical & Clinical Disclaimer

> **IMPORTANT MEDICAL DISCLAIMER**:  
> **DermAI Triage** is an artificial intelligence-assisted decision support system developed for preliminary screening and referral triage purposes. It is **NOT** a substitute for professional medical diagnosis, physical examination, or biopsy by a licensed dermatologist. Patients must always consult a qualified medical professional for diagnosis and treatment.

---

## 🤝 Credits & Acknowledgments

- **Architecture**: EfficientNet Convolutional Neural Network & Grad-CAM Visual Explainability.
- **Secondary Screening**: Cloud Workflow Screening via Roboflow API.
- **Mapping & Location Services**: Google Maps Places API.
- **PDF Generation Engine**: ReportLab & PyQRCode.
- **Department**: Data Science & AI Engineering Capstone Project.
