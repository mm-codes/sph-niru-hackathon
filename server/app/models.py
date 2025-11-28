from django.db import models
from django.contrib.auth.models import User
import uuid

class AudioAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    audio_file = models.FileField(upload_to='audio_uploads/')
    is_deepfake = models.BooleanField(null=True)
    confidence_score = models.FloatField(null=True)
    analysis_details = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class TextAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    text_content = models.TextField()
    is_ai_generated = models.BooleanField(null=True)
    confidence_score = models.FloatField(null=True)
    analysis_details = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
