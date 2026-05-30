from django.contrib import admin
from .models import DetectionEvent, SortingSession, ModelPerformance, ContaminationRule

@admin.register(DetectionEvent)
class DetectionEventAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'detected_class', 'confidence_score', 'alert_triggered', 'worker_corrected']
    list_filter = ['alert_triggered', 'worker_corrected', 'detected_class']
    search_fields = ['detected_class', 'worker_notes']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'

@admin.register(SortingSession)
class SortingSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'operator_name', 'start_time', 'end_time', 'total_detections', 'total_alerts']
    list_filter = ['shift']
    readonly_fields = ['start_time']

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model_version', 'accuracy', 'precision', 'recall', 'f1_score', 'test_date']
    readonly_fields = ['test_date']

@admin.register(ContaminationRule)
class ContaminationRuleAdmin(admin.ModelAdmin):
    list_display = ['detected_class', 'target_class', 'alert_enabled', 'severity']
    list_filter = ['alert_enabled', 'severity']