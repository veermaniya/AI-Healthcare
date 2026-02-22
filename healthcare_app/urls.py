from django.urls import path
from . import views

urlpatterns = [
    # ── Pages ──────────────────────────────────────────────
    path('',                          views.index,                name='index'),
    path('login/',                    views.login_view,           name='login'),
    path('logout/',                   views.logout_view,          name='logout'),

    # ── Data loading ────────────────────────────────────────
    path('api/upload/',               views.upload_file,          name='upload_file'),
    path('api/upload-temp/',          views.upload_temp,          name='upload_temp'),
    path('api/connect-sql/',          views.connect_sql,          name='connect_sql'),

    # ── Core ML ─────────────────────────────────────────────
    path('api/run-analysis/',         views.run_analysis,         name='run_analysis'),
    path('api/predict/',              views.predict_input,        name='predict_input'),

    # ── Feature 1: XAI ──────────────────────────────────────
    path('api/run-explainability/',   views.run_explainability,   name='run_explainability'),

    # ── Feature 2: Patient Dashboard ────────────────────────
    path('api/patient-dashboard/',    views.patient_dashboard,    name='patient_dashboard'),

    # ── Feature 3: Multi-target ─────────────────────────────
    path('api/multi-target-compare/', views.multi_target_compare, name='multi_target_compare'),

    # ── Feature 4: Survival Analysis ────────────────────────
    path('api/survival-analysis/',    views.survival_analysis,    name='survival_analysis'),

    # ── Feature 5: Alert Rules ──────────────────────────────
    path('api/evaluate-alert-rules/', views.evaluate_alert_rules, name='evaluate_alert_rules'),
    path('api/suggest-alert-rules/',  views.suggest_alert_rules,  name='suggest_alert_rules'),

    # ── Feature 6: Dataset Compare ──────────────────────────
    path('api/compare-datasets/',     views.compare_datasets,     name='compare_datasets'),

    # ── Feature 7: Patient Similarity ───────────────────────
    path('api/run-patient-similarity/',views.run_patient_similarity,name='run_patient_similarity'),

    # ── Feature 8: Risk Engine ──────────────────────────────
    path('api/run-risk-engine/',      views.run_risk_engine,      name='run_risk_engine'),

    # ── Feature 9: Trend Analysis ───────────────────────────
    path('api/run-trend-analysis/',   views.run_trend_analysis,   name='run_trend_analysis'),

    # ── Feature 10: Clinical Insights ───────────────────────
    path('api/clinical-insights/',    views.clinical_insights,    name='clinical_insights'),

    # ── Feature 11: Data Quality ────────────────────────────
    path('api/data-quality/',         views.data_quality,         name='data_quality'),

    # ── Feature 12: Privacy & PHI ───────────────────────────
    path('api/privacy-check/',        views.privacy_check,        name='privacy_check'),
    path('api/anonymise/',            views.anonymise,            name='anonymise'),

    # ── Feature 13: Clinical Coding ─────────────────────────
    path('api/clinical-coding/',      views.clinical_coding,      name='clinical_coding'),

    # ── Feature 14: Report ──────────────────────────────────
    path('api/generate-report/',      views.generate_report,      name='generate_report'),

    # ── Feature 15: Filter ──────────────────────────────────
    path('api/filter/',               views.filter_data,          name='filter_data'),

    # ── Chat & Imaging ──────────────────────────────────────
    path('api/chat/',                 views.chat,                 name='chat'),
    path('api/analyze-image/',        views.analyze_image,        name='analyze_image'),

    # ── Session helpers ─────────────────────────────────────
    path('api/columns/',              views.get_columns,          name='get_columns'),
    path('api/column-stats/',         views.get_column_stats_view,name='get_column_stats'),
    path('api/clear/',                views.clear_session,        name='clear_session'),

    # ── Debug ────────────────────────────────────────────────
    path('api/test-llm/',             views.test_llm,             name='test_llm'),
    path('api/debug-key/',            views.debug_key,            name='debug_key'),
    path('api/login/',                views.api_login,            name='api_login'),
]
