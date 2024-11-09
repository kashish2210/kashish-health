from django.db import models
from hashlib import md5

class PatientRecord(models.Model):
    doctor_name = models.CharField(max_length=100)
    patient_name = models.CharField(max_length=100)
    description = models.TextField()
    date = models.DateField()
    file = models.FileField(upload_to='uploads/')

    file_hash = models.CharField(max_length=64, editable=False)

    def save(self, *args, **kwargs):
        if self.file:
            self.file_hash = md5(self.file.read()).hexdigest()
        super(PatientRecord, self).save(*args, **kwargs)

    def __str__(self):
        return f'{self.patient_name} ({self.doctor_name})'
