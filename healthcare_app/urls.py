from django.urls import path
from . import views

urlpatterns = [
    # ── Main app ──
    path('', views.index, name='index'),

    # ── Auth ──
    path('login/', views.login_view, name='login_view'),
    path('api/login/', views.api_login, name='api_login'),
    path('api/logout/', views.api_logout, name='api_logout'),

    # ── Data sources ──
    path('api/upload/', views.upload_file, name='upload_file'),
    path('api/connect-sql/', views.connect_sql, name='connect_sql'),
    path('api/columns/', views.get_columns, name='get_columns'),
    path('api/column-stats/', views.get_column_stats, name='get_column_stats'),

    # ── Core ML ──
    path('api/run-analysis/', views.run_analysis, name='run_analysis'),
    path('api/predict/', views.predict_input, name='predict_input'),

    # ── NEW: Advanced AI Features ──
    path('api/explainability/', views.run_explainability, name='run_explainability'),
    path('api/risk-engine/', views.run_risk_engine, name='run_risk_engine'),
    path('api/trend-analysis/', views.run_trend_analysis, name='run_trend_analysis'),
    path('api/patient-similarity/', views.run_patient_similarity, name='run_patient_similarity'),

    # ── Chat & Utilities ──
    path('api/chat/', views.chat, name='chat'),
    path('api/analyze-image/', views.analyze_image, name='analyze_image'),
    path('api/test-llm/', views.test_llm, name='test_llm'),
    path('api/clear/', views.clear_session, name='clear_session'),
    path('api/debug-key/', views.debug_key, name='debug_key'),
]
