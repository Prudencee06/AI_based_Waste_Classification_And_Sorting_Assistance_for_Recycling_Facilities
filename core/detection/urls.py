from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register(r'detections', views.DetectionEventViewSet)
router.register(r'sessions', views.SortingSessionViewSet)
router.register(r'models', views.ModelPerformanceViewSet)
router.register(r'rules', views.ContaminationRuleViewSet)

# Dashboard is a separate ViewSet
router.register(r'dashboard', views.DashboardViewSet, basename='dashboard')

urlpatterns = [
    path('', include(router.urls)),
]