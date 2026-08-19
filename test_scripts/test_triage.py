from triage_logic import get_risk_level

# Kuch alag-alag test cases try karte hain
test_cases = [
    ("Scabies", 0.85),
    ("Scabies", 0.40),
    ("Acne", 0.90),
    ("Psoriasis", 0.55),
    ("Candidal Intertrigo", 0.30),
    ("Melasma", 0.75),
    ("Psoriasis", 0.80),
    ("Eczema & Dermatitis", 0.75),
]

for disease, confidence in test_cases:
    result = get_risk_level(disease, confidence)
    print(f"\nDisease: {disease} | Confidence: {confidence*100:.0f}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Urgency: {result['urgency']}")
    print(f"  Rationale: {result['rationale']}")
