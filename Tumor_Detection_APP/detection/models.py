from django.db import models

class MRIImage(models.Model):
    image = models.ImageField(upload_to='uploads/mri_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=50, null=True, blank=True)  # e.g., "Tumor Detected"
    confidence = models.FloatField(null=True, blank=True)  # e.g., 96.5%

    def __str__(self):
        return f"MRI Image {self.id} - {self.prediction}"
