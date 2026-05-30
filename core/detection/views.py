from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from django.db.models import Count, Avg
from .models import DetectionEvent, SortingSession, ModelPerformance, ContaminationRule
from .serializers import (
    DetectionEventSerializer, DetectionEventCreateSerializer,
    SortingSessionSerializer, ModelPerformanceSerializer,
    ContaminationRuleSerializer
)


class DetectionEventViewSet(viewsets.ModelViewSet):
    
    queryset = DetectionEvent.objects.all()
    serializer_class = DetectionEventSerializer
    
    def get_serializer_class(self):
        if self.action == 'create':
            return DetectionEventCreateSerializer
        return DetectionEventSerializer
    
    def create(self, request, *args, **kwargs):
        """Create a new detection event."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Save with timestamp
        event = serializer.save(timestamp=timezone.now())
        
        # Update session if exists
        if event.session_id:
            try:
                session = SortingSession.objects.get(id=event.session_id)
                session.total_detections += 1
                if event.alert_triggered:
                    session.total_alerts += 1
                session.save()
            except SortingSession.DoesNotExist:
                pass
        
        return Response(
            DetectionEventSerializer(event).data,
            status=status.HTTP_201_CREATED
        )
    
    @action(detail=False, methods=['get'])
    def today(self, request):

        today = timezone.now().date()
        events = DetectionEvent.objects.filter(timestamp__date=today)
        serializer = self.get_serializer(events, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def alerts(self, request):
        """Get only alert events."""
        alerts = DetectionEvent.objects.filter(alert_triggered=True)
        serializer = self.get_serializer(alerts, many=True)
        return Response(serializer.data)


class SortingSessionViewSet(viewsets.ModelViewSet):
    # API endpoints for sorting sessions.
    
    queryset = SortingSession.objects.all()
    serializer_class = SortingSessionSerializer
    
    @action(detail=True, methods=['post'])
    def end(self, request, pk=None):
        # End a sorting session.
        session = self.get_object()
        session.end_time = timezone.now()
        session.calculate_metrics()
        session.save()
        return Response(self.get_serializer(session).data)
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        # Get active (unended) sessions.
        active_sessions = SortingSession.objects.filter(end_time__isnull=True)
        serializer = self.get_serializer(active_sessions, many=True)
        return Response(serializer.data)


class ModelPerformanceViewSet(viewsets.ModelViewSet):
    # API endpoints for model performance.
    
    queryset = ModelPerformance.objects.all()
    serializer_class = ModelPerformanceSerializer


class ContaminationRuleViewSet(viewsets.ModelViewSet):
    # API endpoints for contamination rules.
    
    queryset = ContaminationRule.objects.all()
    serializer_class = ContaminationRuleSerializer


class DashboardViewSet(viewsets.ViewSet):
    # Dashboard statistics endpoint.
    
    def list(self, request):
        """Get dashboard statistics."""
        today = timezone.now().date()
        
        # Today's statistics
        total_detections_today = DetectionEvent.objects.filter(
            timestamp__date=today
        ).count()
        
        total_alerts_today = DetectionEvent.objects.filter(
            timestamp__date=today,
            alert_triggered=True
        ).count()
        
        # Top contaminants
        top_contaminants = DetectionEvent.objects.filter(
            alert_triggered=True
        ).values('detected_class').annotate(
            count=Count('id')
        ).order_by('-count')[:5]
        
        # Recent events
        recent_events = DetectionEvent.objects.all()[:100]
        
        # Active sessions
        active_sessions = SortingSession.objects.filter(end_time__isnull=True)
        
        # Latest model performance
        latest_model = ModelPerformance.objects.first()
        
        data = {
            'total_detections_today': total_detections_today,
            'total_alerts_today': total_alerts_today,
            'top_contaminants': list(top_contaminants),
            'recent_events': DetectionEventSerializer(recent_events, many=True).data,
            'active_sessions': SortingSessionSerializer(active_sessions, many=True).data,
            'model_accuracy': latest_model.accuracy if latest_model else 0
        }
        
        return Response(data)