
# Create your views here.

import json
import joblib
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.shortcuts import render
from .forms import PatientPredictionForm

BASE_DIR = Path(settings.BASE_DIR)
MODEL_PATH = BASE_DIR / "ml_model" / "glaucoma_model.joblib"
FEATURES_PATH = BASE_DIR / "ml_model" / "model_features.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    FEATURES = json.load(file)

LABELS = {
    0: "Healthy",
    1: "Glaucoma",
    2: "Glaucoma Suspect"
}

def home(request):
    result = None
    probabilities = []
    form = PatientPredictionForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data

        values = {
            "Age": data["age"],
            "Gender": int(data["gender"]),
            "Dioptre_1": data["dioptre_1"],
            "Dioptre_2": data["dioptre_2"],
            "Astigmatism": data["astigmatism"],
            "Phakic_Pseudophakic": int(data["phakic_pseudophakic"]),
            "IOP_Pneumatic": data["iop_pneumatic"],
            "IOP_Perkins": data["iop_perkins"],
            "Pachymetry": data["pachymetry"],
            "Axial_Length": data["axial_length"],
            "VF_MD": data["vf_md"],
        }

        patient = pd.DataFrame(
            [[values[name] for name in FEATURES]],
            columns=FEATURES
        )

        prediction = int(model.predict(patient)[0])
        result = LABELS.get(prediction, "Unknown")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(patient)[0]
            probabilities = [
                {
                    "label": LABELS[int(cls)],
                    "percentage": round(float(prob) * 100, 2)
                }
                for cls, prob in zip(model.classes_, probs)
            ]

    return render(request, "predictor/home.html", {
        "form": form,
        "result": result,
        "probabilities": probabilities,
    })

