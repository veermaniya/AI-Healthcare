# healthcare_app/urls.py
# Complete URL routing — original routes + 4 new AI endpoints

from django.urls import path
from . import views

urlpatterns = [

    # ── Main page ──────────────────────────────────────────
    path('', views.index, name='index'),

    # ── Dataset ────────────────────────────────────────────
    path('api/upload/',         views.upload_file,      name='upload_file'),
    path('api/connect-sql/',    views.connect_sql,       name='connect_sql'),
    path('api/columns/',        views.get_columns,       name='get_columns'),
    path('api/column-stats/',   views.get_column_stats,  name='column_stats'),
    path('api/clear/',          views.clear_session,     name='clear_session'),

    # ── ML Analysis ────────────────────────────────────────
    path('api/run-analysis/',   views.run_analysis,      name='run_analysis'),
    path('api/predict/',        views.predict,           name='predict'),

    # ── Chat & Imaging ─────────────────────────────────────
    path('api/chat/',           views.chat,              name='chat'),
    path('api/analyze-image/',  views.analyze_image,     name='analyze_image'),

    # ── NEW: 4 Advanced AI Endpoints ───────────────────────
    path('api/run-explainability/',     views.run_explainability,    name='run_explainability'),
    path('api/run-risk-engine/',        views.run_risk_engine,       name='run_risk_engine'),
    path('api/run-trend-analysis/',     views.run_trend_analysis,    name='run_trend_analysis'),
    path('api/run-patient-similarity/', views.run_patient_similarity, name='run_patient_similarity'),

]
