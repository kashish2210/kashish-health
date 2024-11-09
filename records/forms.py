from django import forms
from .models import PatientRecord

class PatientRecordForm(forms.ModelForm):
    class Meta:
        model = PatientRecord
        fields = ['doctor_name', 'patient_name', 'description', 'date', 'file']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }
