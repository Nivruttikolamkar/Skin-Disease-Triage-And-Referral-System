# Base severity mapping — adjust these based on your domain knowledge /
# your project guide's feedback if needed
DISEASE_SEVERITY = {
    "Scabies": "high",                        # contagious, spreads fast
    "Candidal Intertrigo": "medium_high",      # needs prompt diagnosis
    "Psoriasis": "medium_high",                # chronic, needs management
    "Eczema & Dermatitis": "medium",
    "Alopecia Areata": "low_medium",
    "Vitiligo": "low_medium",
    "Tinea (Fungal Infection)": "low_medium",
    "Melasma": "low",
    "Acne": "low",
}

SEVERITY_RANK = {
    "low": 0,
    "low_medium": 1,
    "medium": 2,
    "medium_high": 3,
    "high": 4,
}
# raised from 0.40, since unrelated images scored 95% confidence
MIN_VALID_CONFIDENCE = 0.50


def is_valid_prediction(confidence):
    """
    Returns False if confidence is too low to trust — likely means
    the uploaded image isn't a recognizable skin condition from our classes.
    """
    return confidence >= MIN_VALID_CONFIDENCE


def get_risk_level(predicted_class, confidence):
    """
    predicted_class: string, e.g. "Scabies"
    confidence: float between 0 and 1

    Returns: dict with risk_level, urgency, rationale
    """
    base_severity = DISEASE_SEVERITY.get(predicted_class, "medium")
    severity_score = SEVERITY_RANK[base_severity]

    # Low confidence pushes risk UP (uncertain predictions need more caution,
    # not less — we don't want to reassure someone based on a guess)
    if confidence < 0.5:
        severity_score += 2
    elif confidence < 0.7:
        severity_score += 1
    # confidence >= 0.7 -> no adjustment, trust the base severity

    # Cap at max
    severity_score = min(severity_score, 4)

    if severity_score >= 3:
        risk_level = "RED"
        urgency = "High / Urgent Referral"
        rationale = (
            f"Predicted condition ({predicted_class}) combined with the model's "
            f"confidence level suggests this case should be reviewed by a "
            f"dermatologist promptly."
        )
    elif severity_score >= 1:
        risk_level = "YELLOW"
        urgency = "Moderate / Routine Referral"
        rationale = (
            f"Predicted condition ({predicted_class}) is generally manageable, "
            f"but a dermatologist consultation is recommended to confirm "
            f"diagnosis and begin appropriate treatment."
        )
    else:
        risk_level = "GREEN"
        urgency = "Low / Self-Care or Routine Follow-up"
        rationale = (
            f"Predicted condition ({predicted_class}) is typically low-risk. "
            f"Monitor the area and consult a dermatologist if symptoms worsen "
            f"or persist."
        )

    return {
        "risk_level": risk_level,
        "urgency": urgency,
        "rationale": rationale,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
    }
