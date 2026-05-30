from django.db import models
from django.utils import timezone

class DetectionEvent(models.Model):
    """
    Records every time the system detects waste and triggers an alert.
    """
    
    # Types of events
    EVENT_TYPES = [
        ('detection', 'Waste Detection'),
        ('alert', 'Contamination Alert'),
        ('manual', 'Manual Sort'),
    ]
    
    # Waste categories
    WASTE_CATEGORIES = [
        ('plastic', 'Plastic'),
        ('metal', 'Metal'),
        ('paper', 'Paper'),
        ('glass', 'Glass'),
        ('cardboard', 'Cardboard'),
        ('organic', 'Organic'),
        ('trash', 'Trash'),
    ]
    
    # Detection details
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES, default='detection')
    detected_class = models.CharField(max_length=20, choices=WASTE_CATEGORIES)
    confidence_score = models.FloatField()
    
    # Alert details
    alert_triggered = models.BooleanField(default=False)
    alert_reason = models.CharField(max_length=200, blank=True, null=True)
    
    # Image storage
    image_path = models.CharField(max_length=500, blank=True, null=True)
    image_base64 = models.TextField(blank=True, null=True)
    
    # Worker feedback
    worker_corrected = models.BooleanField(default=False)
    worker_corrected_class = models.CharField(max_length=20, choices=WASTE_CATEGORIES, blank=True, null=True)
    worker_notes = models.TextField(blank=True, null=True)
    
    # Timestamps
    timestamp = models.DateTimeField(default=timezone.now)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return f"{self.timestamp}: {self.detected_class} ({self.confidence_score:.2%})"
    
    class Meta:
        ordering = ['-timestamp']


class SortingSession(models.Model):
    """
    Tracks a continuous sorting session.
    """
    
    SHIFT_CHOICES = [
        ('morning', 'Morning'),
        ('afternoon', 'Afternoon'),
        ('night', 'Night'),
    ]
    
    operator_name = models.CharField(max_length=100, blank=True, null=True)
    shift = models.CharField(max_length=20, choices=SHIFT_CHOICES, blank=True, null=True)
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(blank=True, null=True)
    
    total_detections = models.IntegerField(default=0)
    total_alerts = models.IntegerField(default=0)
    worker_interventions = models.IntegerField(default=0)
    average_confidence = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"Session {self.id}: {self.start_time}"
    
    def calculate_metrics(self):
        """Calculate session metrics from detection events."""
        from django.db.models import Avg
        events = self.detectionevent_set.all()
        self.total_detections = events.count()
        self.total_alerts = events.filter(alert_triggered=True).count()
        self.worker_interventions = events.filter(worker_corrected=True).count()
        self.average_confidence = events.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
        self.save()
        return self


class ModelPerformance(models.Model):
    """
    Track model performance over time.
    """
    
    model_version = models.CharField(max_length=50)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    test_date = models.DateTimeField(default=timezone.now)
    class_performance = models.JSONField(default=dict)
    
    def __str__(self):
        return f"v{self.model_version}: {self.accuracy:.2%} on {self.test_date}"
    
    class Meta:
        ordering = ['-test_date']


class ContaminationRule(models.Model):
    """
    Configurable rules for contamination alerts.
    """
    
    WASTE_CATEGORIES = [
        ('plastic', 'Plastic'),
        ('metal', 'Metal'),
        ('paper', 'Paper'),
        ('glass', 'Glass'),
        ('cardboard', 'Cardboard'),
        ('organic', 'Organic'),
        ('trash', 'Trash'),
    ]
    
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    detected_class = models.CharField(max_length=20, choices=WASTE_CATEGORIES)
    target_class = models.CharField(max_length=20, choices=WASTE_CATEGORIES)
    alert_enabled = models.BooleanField(default=True)
    alert_message = models.CharField(max_length=200, default="Contamination detected!")
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='medium')
    
    def __str__(self):
        return f"{self.detected_class} in {self.target_class} -> Alert: {self.alert_enabled}"
    
    class Meta:
        unique_together = ['detected_class', 'target_class']