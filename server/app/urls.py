from django.urls import path

from . import views

app_name = "app"

urlpatterns = [
    path("", views.landing_page, name="landing"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("profile/", views.profile, name="profile"),
    path("analysis/", views.analysis, name="analysis"),
    path("reports/", views.reports, name="reports"),
    path('/analyze/audio/', analyze_audio, name='analyze_audio'),
    path('/analyze/text/', analyze_text, name='analyze_text'),
    path('/history/', get_analysis_history, name='analysis_history'),
]
