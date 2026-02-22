from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/upload/', views.upload_file, name='upload_file'),
    path('api/connect-sql/', views.connect_sql, name='connect_sql'),
    path('api/run-analysis/', views.run_analysis, name='run_analysis'),
    path('api/predict/', views.predict_input, name='predict_input'),
    path('api/chat/', views.chat, name='chat'),
    path('api/columns/', views.get_columns, name='get_columns'),
    path('api/column-stats/', views.get_column_stats, name='get_column_stats'),
    path('api/test-llm/', views.test_llm, name='test_llm'),
    path('api/analyze-image/', views.analyze_image, name='analyze_image'),
    path('api/clear/', views.clear_session, name='clear_session'),
    path('api/debug-key/', views.debug_key, name='debug_key'),
]
