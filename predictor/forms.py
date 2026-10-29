
from django import forms

class PatientPredictionForm(forms.Form):
    eye = forms.ChoiceField(
        choices=[("OD", "Right Eye"), ("OS", "Left Eye")],
        label="Eye being assessed"
    )
    age = forms.FloatField(min_value=1, max_value=120, label="Age (years)")
    gender = forms.ChoiceField(
        choices=[("0", "Female"), ("1", "Male")], label="Gender"
    )
    dioptre_1 = forms.FloatField(label="Refractive measurement 1")
    dioptre_2 = forms.FloatField(label="Refractive measurement 2")
    astigmatism = forms.FloatField(label="Astigmatism")
    phakic_pseudophakic = forms.ChoiceField(
        choices=[("0", "Phakic"), ("1", "Pseudophakic")],
        label="Lens status"
    )
    iop_pneumatic = forms.FloatField(min_value=0, label="Pneumatic IOP")
    iop_perkins = forms.FloatField(min_value=0, label="Perkins IOP")
    pachymetry = forms.FloatField(min_value=0, label="Corneal thickness")
    axial_length = forms.FloatField(min_value=0, label="Axial length")
    vf_md = forms.FloatField(
        required=False, label="Visual field MD (optional)"
    )
