from rest_framework import serializers
from .models import DetectionEvent, SortingSession, ModelPerformance, ContaminationRule

class DetectionEventSerializer(serializers.ModelSerializer):
    """Serializer for detection events."""
    
    class Meta:
        model = DetectionEvent
        fields = '__all__'
        read_only_fields = ['timestamp']


class DetectionEventCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating detection events."""
    
    class Meta:
        model = DetectionEvent
        fields = ['detected_class', 'confidence_score', 'alert_triggered', 
                  'alert_reason', 'image_base64', 'session_id', 'event_type']


class SortingSessionSerializer(serializers.ModelSerializer):
    """Serializer for sorting sessions."""
    
    class Meta:
        model = SortingSession
        fields = '__all__'
        read_only_fields = ['start_time', 'total_detections', 'total_alerts', 
                           'worker_interventions', 'average_confidence']


class ModelPerformanceSerializer(serializers.ModelSerializer):
    """Serializer for model performance tracking."""
    
    class Meta:
        model = ModelPerformance
        fields = '__all__'


class ContaminationRuleSerializer(serializers.ModelSerializer):
    """Serializer for contamination rules."""
    
    class Meta:
        model = ContaminationRule
        fields = '__all__'