import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import qrcode


def generate_pdf_report(patient, prediction, risk, original_img_path,
                         gradcam_img_path, output_path, case_id, nearby_info=None):
    """
    patient: dict with name, age, gender, city, email, phone, lesion_site,
             duration, pain, itching, bleeding, history
    prediction: dict with predicted_class, confidence
    risk: dict from triage_logic.get_risk_level()
    nearby_info: dict with city, maps_search_url, clinics list
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    elements = []

    report_id = "REP-" + datetime.now().strftime("%y%m%d-%H%M%S").upper()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Heading1"],
        fontSize=14, textColor=colors.HexColor("#1a3d6d")
    )
    elements.append(Paragraph("INTELLIGENT DERMATOLOGY TRIAGE", title_style))
    elements.append(Paragraph("AI-Assisted Skin Disease Screening & Referral", styles["Normal"]))
    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph(f"<b>Report ID:</b> {report_id} &nbsp;&nbsp;&nbsp; <b>Date:</b> {date_str}", styles["Normal"]))
    elements.append(Spacer(1, 6*mm))

    # ===== Patient Info Table =====
    patient_data = [
        ["Name:", patient.get("name", "-"), "Age / Gender:", f"{patient.get('age','-')} / {patient.get('gender','-')}"],
        ["Email:", patient.get("email", "-"), "City / Location:", patient.get("city", "N/A")],
        ["Lesion Site:", patient.get("lesion_site", "-"), "Duration:", patient.get("duration", "-")],
        ["Symptoms:", f"Pain: {patient.get('pain','No')} | Itching: {patient.get('itching','No')} | Bleeding: {patient.get('bleeding','No')}", "Previous History:", patient.get("history", "-")],
    ]
    patient_table = Table(patient_data, colWidths=[28*mm, 60*mm, 30*mm, 55*mm])
    patient_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d6e5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(Paragraph("<b>Patient Information</b>", styles["Heading3"]))
    elements.append(patient_table)
    elements.append(Spacer(1, 6*mm))

    # ===== AI Prediction Box =====
    risk_color = {
        "RED": colors.HexColor("#f8d7da"),
        "YELLOW": colors.HexColor("#fff3cd"),
        "GREEN": colors.HexColor("#d4edda"),
    }.get(risk["risk_level"], colors.HexColor("#f8f9fa"))

    risk_text_color = {
        "RED": colors.HexColor("#c0392b"),
        "YELLOW": colors.HexColor("#b8860b"),
        "GREEN": colors.HexColor("#1e7e34"),
    }.get(risk["risk_level"], colors.black)

    pred_data = [[
        Paragraph(f"<b>Predicted Classification:</b> {prediction['predicted_class']}<br/>"
                  f"<b>Prediction Confidence:</b> {prediction['confidence']:.2f}%", styles["Normal"]),
        Paragraph(f"<font color='{risk_text_color.hexval()}'><b>TRIAGE RISK LEVEL: {risk['risk_level']}</b></font><br/>"
                  f"<b>Urgency:</b> {risk['urgency']}", styles["Normal"]),
    ]]
    pred_table = Table(pred_data, colWidths=[90*mm, 83*mm])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("BOX", (0, 0), (-1, -1), 1, risk_text_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(Paragraph("<b>AI-Assisted Screening & Triage Analysis</b>", styles["Heading3"]))
    elements.append(pred_table)
    elements.append(Spacer(1, 6*mm))

    # ===== Images: Original, Grad-CAM, QR =====
    qr_path = output_path.replace(".pdf", "_qr.png")
    qr_data = f"http://127.0.0.1:5000/result/{case_id}"
    qr_img = qrcode.make(qr_data)
    qr_img.save(qr_path)

    img_row = [[
        Paragraph("<b>Original Lesion Image</b>", styles["Normal"]),
        Paragraph("<b>Grad-CAM Saliency Map</b>", styles["Normal"]),
        Paragraph("<b>Report QR Verification</b>", styles["Normal"]),
    ], [
        Image(original_img_path, width=50*mm, height=50*mm),
        Image(gradcam_img_path, width=50*mm, height=50*mm),
        Image(qr_path, width=30*mm, height=30*mm),
    ]]
    img_table = Table(img_row, colWidths=[58*mm, 58*mm, 57*mm])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d6e5")),
    ]))
    elements.append(Paragraph("<b>Clinical Visualization (Original & Grad-CAM Overlay)</b>", styles["Heading3"]))
    elements.append(img_table)
    elements.append(Spacer(1, 6*mm))

    # ===== Nearby Dermatologists (Google Maps Integration) =====
    if nearby_info and nearby_info.get("clinics"):
        elements.append(Paragraph(f"<b>Nearby Dermatologists & Referral Clinics ({nearby_info.get('city', 'Local Area')})</b>", styles["Heading3"]))
        clinic_rows = [["Clinic / Doctor Name", "Address / Location", "Rating"]]
        for c in nearby_info["clinics"][:3]:
            clinic_rows.append([
                c.get("name", "Dermatologist"),
                c.get("address", "Local Medical Center"),
                f"★ {c.get('rating', '4.5')} ({c.get('user_ratings_total', '50')}+ reviews)"
            ])
        clinic_table = Table(clinic_rows, colWidths=[65*mm, 70*mm, 38*mm])
        clinic_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d6e5")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(clinic_table)
        elements.append(Spacer(1, 4*mm))

    # ===== Recommendations =====
    elements.append(Paragraph("<b>Referral Recommendations & Precautions</b>", styles["Heading3"]))
    elements.append(Paragraph(f"<b>Triage Rationale:</b> {risk['rationale']}", styles["Normal"]))
    elements.append(Spacer(1, 3*mm))

    elements.append(Paragraph("<b>General Precautions:</b>", styles["Normal"]))
    precautions = [
        "Do NOT scratch, squeeze, pick, or attempt self-treatment of the lesion.",
        "Avoid applying non-prescription topical creams, ointments, or chemical peels.",
        "Protect the affected area from direct sunlight and irritants.",
        "Document any visible changes in size, color, or texture of the lesion.",
    ]
    for p in precautions:
        elements.append(Paragraph(f"• {p}", styles["Normal"]))
    elements.append(Spacer(1, 3*mm))

    elements.append(Paragraph("<b>Recommended Next Steps:</b>", styles["Normal"]))
    if risk["risk_level"] == "RED":
        steps = ["Schedule an urgent consultation with a dermatologist within 24-48 hours.",
                 "Avoid delaying treatment as prompt diagnosis is important."]
    elif risk["risk_level"] == "YELLOW":
        steps = ["Schedule a routine dermatologist consultation within 1-2 weeks.",
                 "Monitor for any worsening of symptoms."]
    else:
        steps = ["Monitor the area for changes.",
                 "Consult a dermatologist if symptoms persist beyond a few weeks."]
    for s in steps:
        elements.append(Paragraph(f"• {s}", styles["Normal"]))

    elements.append(Spacer(1, 8*mm))
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=7, textColor=colors.grey
    )
    elements.append(Paragraph(
        "CLINICAL DISCLAIMER: This AI clinical screening system is intended only for "
        "educational and triage screening purposes. It does not replace a qualified "
        "dermatologist's physical examination, biopsy, or diagnosis. This document does "
        "not constitute a prescription for medications or medical treatments.",
        disclaimer_style
    ))

    doc.build(elements)

    # cleanup temp qr file
    if os.path.exists(qr_path):
        os.remove(qr_path)

    return report_id, output_path