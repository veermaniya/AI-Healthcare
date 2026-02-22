# healthcare_app/views.py
# Rebuilt to match your ACTUAL models.py schema exactly.
# DataSession fields used: session_key, source_type, file_name,
#   sql_connection, sql_query, columns_json, preview_json, data_json,
#   row_count, col_count
# AnalysisResult links via session FK + result_json
# ChatMessage: session_key, role, message, context_type

import json
import logging
import pandas as pd
import numpy as np

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.conf import settings

from ml_engine import HealthcareMLEngine, GroqLLMClient, load_excel_file, load_from_sql
from .models import DataSession, AnalysisResult, ChatMessage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  SESSION HELPERS  (uses columns_json / data_json — NO pickle)
# ═══════════════════════════════════════════════════════════

def _ensure_session(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _get_session(request):
    key = _ensure_session(request)
    try:
        return DataSession.objects.get(session_key=key)
    except DataSession.DoesNotExist:
        return None


def _get_df(request):
    """Rebuild DataFrame from data_json stored in DataSession."""
    session = _get_session(request)
    if session is None:
        return None
    try:
        rows = json.loads(session.data_json or '[]')
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error("DataFrame rebuild error: %s", e)
        return None


def _save_session(request, df: pd.DataFrame, source_type: str,
                  file_name: str = '', sql_connection: str = '', sql_query: str = ''):
    """Save dataset into DataSession as JSON — no pickle, matches your real schema."""
    key = _ensure_session(request)
    engine = HealthcareMLEngine()
    engine.load_dataframe(df)
    col_info = engine.get_column_info()

    # Store full data as JSON for ML rebuild; preview = first 500 rows
    data_rows    = df.fillna('').to_dict(orient='records')
    preview_rows = df.head(500).fillna('').to_dict(orient='records')

    DataSession.objects.update_or_create(
        session_key=key,
        defaults={
            'source_type':    source_type,
            'file_name':      file_name,
            'sql_connection': sql_connection,
            'sql_query':      sql_query,
            'columns_json':   json.dumps(col_info),
            'preview_json':   json.dumps(preview_rows),
            'data_json':      json.dumps(data_rows),
            'row_count':      len(df),
            'col_count':      len(df.columns),
        }
    )
    return col_info, preview_rows


def _get_col_stats(df: pd.DataFrame) -> dict:
    """Per-column stats for sliders, smart filters, and prediction hints."""
    stats = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            rng  = float(s.max() - s.min())
            step = round(rng / 100, 4) if rng > 0 else 0.01
            stats[col] = {
                'type':   'numeric',
                'min':    round(float(s.min()),    3),
                'max':    round(float(s.max()),    3),
                'mean':   round(float(s.mean()),   3),
                'median': round(float(s.median()), 3),
                'std':    round(float(s.std()),    3),
                'step':   step,
            }
        else:
            vc = df[col].value_counts()
            stats[col] = {
                'type':          'categorical',
                'unique_values': s.astype(str).unique().tolist()[:30],
                'most_common':   str(vc.idxmax()) if len(vc) else '',
                'value_counts':  {str(k): int(v) for k, v in vc.head(20).items()},
            }
    return stats


def _get_latest_result(session):
    """Get most recent AnalysisResult for this session."""
    try:
        return session.results.latest('created_at')
    except AnalysisResult.DoesNotExist:
        return None


def _get_llm():
    return GroqLLMClient(
        api_key=getattr(settings, 'GROQ_API_KEY', ''),
        model=getattr(settings, 'GROQ_MODEL', 'llama-3.3-70b-versatile'),
    )


# ═══════════════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════════════

def index(request):
    _ensure_session(request)
    return render(request, 'healthcare_app/index.html')


# ═══════════════════════════════════════════════════════════
#  UPLOAD / SQL
# ═══════════════════════════════════════════════════════════

@require_POST
def upload_file(request):
    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file provided.'})

        df = load_excel_file(file)
        col_info, preview = _save_session(
            request, df,
            source_type='excel',
            file_name=file.name,
        )
        return JsonResponse({
            'success':     True,
            'rows':        len(df),
            'columns':     len(df.columns),
            'file_name':   file.name,
            'column_info': col_info,
            'preview':     preview,
        })
    except Exception as e:
        logger.error("Upload error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def connect_sql(request):
    try:
        data     = json.loads(request.body)
        conn_str = data.get('connection_string', '').strip()
        query    = data.get('query', '').strip()
        if not conn_str or not query:
            return JsonResponse({'success': False, 'error': 'connection_string and query are required.'})

        df = load_from_sql(conn_str, query)
        col_info, preview = _save_session(
            request, df,
            source_type='sql',
            file_name='SQL Query',
            sql_connection=conn_str,
            sql_query=query,
        )
        return JsonResponse({
            'success':     True,
            'rows':        len(df),
            'columns':     len(df.columns),
            'file_name':   'SQL Query',
            'column_info': col_info,
            'preview':     preview,
        })
    except Exception as e:
        logger.error("SQL error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


# ═══════════════════════════════════════════════════════════
#  COLUMNS & STATS  (dynamic column selection)
# ═══════════════════════════════════════════════════════════

@require_GET
def get_columns(request):
    """
    Returns all columns from current session.
    The frontend uses this to build the dynamic column selector —
    user clicks once = Feature, twice = Target.
    Also auto-detects col_type (numeric/categorical) for smart UI hints.
    """
    try:
        session = _get_session(request)
        if session is None:
            return JsonResponse({'success': False, 'columns': [], 'rows': 0})

        col_info = session.get_columns()
        preview  = session.get_preview()

        return JsonResponse({
            'success':   True,
            'columns':   col_info,       # [{name, col_type, dtype, unique_values, null_count}]
            'rows':      session.row_count,
            'file_name': session.file_name,
            'preview':   preview,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e), 'columns': [], 'rows': 0})


@require_GET
def get_column_stats(request):
    """
    Returns per-column stats for the prediction form sliders and
    smart filter dropdowns (min/max/mean/median for numeric,
    unique values + counts for categorical).
    """
    try:
        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'stats': {}})
        return JsonResponse({'success': True, 'stats': _get_col_stats(df)})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e), 'stats': {}})


@require_POST
def filter_data(request):
    """
    Dynamic column filtering — apply range/value filters and return
    filtered preview + updated stats for the UI.

    POST body:
    {
        "filters": {
            "age":     {"type": "numeric",     "min": 20, "max": 60},
            "gender":  {"type": "categorical", "values": ["Male"]},
            "glucose": {"type": "numeric",     "min": 80, "max": 200}
        },
        "page": 1,
        "page_size": 25
    }
    """
    try:
        data     = json.loads(request.body)
        filters  = data.get('filters', {})
        page     = max(1, int(data.get('page', 1)))
        pagesize = min(200, int(data.get('page_size', 25)))

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})

        filtered = df.copy()
        for col, rule in filters.items():
            if col not in filtered.columns:
                continue
            if rule.get('type') == 'numeric':
                lo, hi = rule.get('min'), rule.get('max')
                if lo is not None:
                    filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') >= float(lo)]
                if hi is not None:
                    filtered = filtered[pd.to_numeric(filtered[col], errors='coerce') <= float(hi)]
            elif rule.get('type') == 'categorical':
                vals = [str(v) for v in rule.get('values', [])]
                if vals:
                    filtered = filtered[filtered[col].astype(str).isin(vals)]

        total   = len(filtered)
        start   = (page - 1) * pagesize
        page_df = filtered.iloc[start:start + pagesize]

        return JsonResponse({
            'success':       True,
            'total_rows':    total,
            'page':          page,
            'page_size':     pagesize,
            'preview':       page_df.fillna('').to_dict(orient='records'),
            'stats':         _get_col_stats(filtered),
        })
    except Exception as e:
        logger.error("Filter error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def clear_session(request):
    try:
        session = _get_session(request)
        if session:
            session.delete()
        ChatMessage.objects.filter(session_key=request.session.session_key).delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ═══════════════════════════════════════════════════════════
#  ML ANALYSIS
#  Auto-selects task type based on target column type:
#    numeric target   → Regression
#    categorical tgt  → Classification
#    no target        → Clustering
# ═══════════════════════════════════════════════════════════

@require_POST
def run_analysis(request):
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col   = data.get('target_column', '')
        task_type    = data.get('task_type', 'auto')
        model_type   = data.get('model_type', 'random_forest')
        n_clusters   = int(data.get('n_clusters', 3))

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded. Please upload a file or connect SQL.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select at least one feature column.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)

        # ── Auto-select task ──
        if task_type == 'clustering' or not target_col:
            task_type = 'clustering'
            result    = engine.run_clustering(feature_cols, n_clusters=n_clusters)
        else:
            if task_type == 'auto':
                task_type = engine.auto_detect_task(target_col)
            if task_type == 'regression':
                result = engine.run_regression(feature_cols, target_col, model_type)
            else:
                result = engine.run_classification(feature_cols, target_col, model_type)

        # ── Normalise for frontend chart ──
        if result.get('task') in ('regression', 'classification'):
            preds   = result.pop('predictions', [])
            actuals = result.pop('actuals', [])
            result['sample_predictions'] = [
                {'actual': a, 'predicted': p}
                for a, p in zip(actuals[:50], preds[:50])
            ]
            result['total_rows'] = len(df)
            result['features']   = feature_cols
            result['target']     = target_col
            m = result['metrics']
            if result['task'] == 'regression':
                m['r2_percent'] = round(m.get('r2', 0) * 100, 1)
            else:
                m['accuracy_percent'] = round(m.get('accuracy', 0) * 100, 1)
        else:
            result['total_rows'] = len(df)
            result['features']   = feature_cols
            result['cluster_distribution'] = [
                {'cluster': f"Cluster {cs['cluster']}", 'count': cs['size']}
                for cs in result.get('cluster_stats', [])
            ]

        # ── Save result linked to session FK ──
        session = _get_session(request)
        if session:
            AnalysisResult.objects.create(
                session=session,
                task_type=result.get('task', task_type),
                feature_columns=','.join(feature_cols),
                target_column=target_col,
                model_type=model_type,
                result_json=json.dumps(result),
            )

        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Analysis error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def predict(request):
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col   = data.get('target_column', '')
        input_values = data.get('input_values', {})

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        prediction = engine.predict_new_input(feature_cols, target_col, input_values)
        return JsonResponse({'success': True, 'prediction': prediction})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ═══════════════════════════════════════════════════════════
#  LLM CHAT
#  Supports natural language queries like:
#  "What is the risk of diabetes for a 45-year-old with high BP?"
#  The LLM gets full dataset context + latest ML result.
#  Works with Groq (default), Ollama (local), Hugging Face.
# ═══════════════════════════════════════════════════════════

@require_POST
def chat(request):
    try:
        data    = json.loads(request.body)
        message = data.get('message', '').strip()
        if not message:
            return JsonResponse({'success': False, 'error': 'Empty message.'})

        session = _get_session(request)
        dataset_context = None
        ml_result       = None

        # Build dataset context from stored JSON (no need to reload file)
        if session:
            col_info = session.get_columns()
            preview  = json.loads(session.preview_json or '[]')[:5]
            dataset_context = {
                'file_name':   session.file_name,
                'rows':        session.row_count,
                'columns':     [c['name'] for c in col_info],
                'column_types': {c['name']: c['col_type'] for c in col_info},
                'preview':     preview,
            }
            # Latest ML result for context
            ar = _get_latest_result(session)
            if ar:
                ml_result = ar.get_result()

        llm    = _get_llm()
        result = llm.ask(message, dataset_context=dataset_context, ml_result=ml_result)

        # Save to chat history
        key = request.session.session_key
        ChatMessage.objects.create(session_key=key, role='user',  message=message, context_type='dataset' if dataset_context else 'general')
        if result.get('success'):
            ChatMessage.objects.create(session_key=key, role='ai', message=result['message'], context_type='llm')

        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ═══════════════════════════════════════════════════════════
#  MEDICAL IMAGING
# ═══════════════════════════════════════════════════════════

@require_POST
def analyze_image(request):
    try:
        data      = json.loads(request.body)
        image_b64 = data.get('image_base64', '')
        prompt    = data.get('prompt', '')
        mime_type = data.get('mime_type', 'image/jpeg')

        if not image_b64:
            return JsonResponse({'success': False, 'error': 'No image data.'})

        api_key = getattr(settings, 'GROQ_API_KEY', '')
        if not api_key:
            return JsonResponse({'success': False, 'error': 'GROQ_API_KEY not set in settings.py.'})

        import requests as req
        models_list = [
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'llama-3.2-11b-vision-preview',
            'llama-3.2-90b-vision-preview',
        ]
        payload = {
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_b64}'}},
                    {'type': 'text', 'text': prompt},
                ],
            }],
            'max_tokens': 1024,
            'temperature': 0.3,
        }
        last_error = 'Vision model unavailable.'
        for model in models_list:
            payload['model'] = model
            try:
                r = req.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                    json=payload, timeout=45,
                )
                if r.status_code == 200:
                    return JsonResponse({'success': True, 'result': r.json()['choices'][0]['message']['content']})
                elif r.status_code != 429:
                    last_error = r.json().get('error', {}).get('message', f'HTTP {r.status_code}')
            except Exception as e:
                last_error = str(e)
        return JsonResponse({'success': False, 'error': last_error})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ═══════════════════════════════════════════════════════════
#  4 NEW AI ENDPOINTS
# ═══════════════════════════════════════════════════════════

@require_POST
def run_explainability(request):
    """🧠 XAI — LIME-style per-feature contribution + global importance."""
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col   = data.get('target_column', '')
        sample_index = int(data.get('sample_index', 0))
        model_type   = data.get('model_type', 'random_forest')

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})
        if not feature_cols or not target_col:
            return JsonResponse({'success': False, 'error': 'Select feature columns and a target column first.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        result = engine.run_explainability(
            feature_cols=feature_cols,
            target_col=target_col,
            model_type=model_type,
            sample_index=sample_index,
        )
        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("XAI error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def run_risk_engine(request):
    """⚠️ Threshold alerts, anomaly detection, sudden change detection."""
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col   = data.get('target_column', '') or None
        thresholds   = data.get('thresholds', None)

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select feature columns first.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        result = engine.run_risk_engine(
            feature_cols=feature_cols,
            target_col=target_col,
            thresholds=thresholds,
        )
        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Risk error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def run_trend_analysis(request):
    """📈 Linear trends, moving averages, forecasting, correlation matrix."""
    try:
        data           = json.loads(request.body)
        feature_cols   = data.get('feature_columns', [])
        time_col       = data.get('time_col', None)
        target_col     = data.get('target_column', None)
        forecast_steps = int(data.get('forecast_steps', 10))

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select feature columns first.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        result = engine.run_trend_analysis(
            feature_cols=feature_cols,
            time_col=time_col,
            target_col=target_col,
            forecast_steps=forecast_steps,
        )
        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Trends error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def run_patient_similarity(request):
    """👥 K-Nearest Neighbours patient similarity with cohort comparison."""
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        query_index  = int(data.get('query_index', 0))
        query_values = data.get('query_values', None)
        n_similar    = int(data.get('n_similar', 5))

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select feature columns first.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        result = engine.run_patient_similarity(
            feature_cols=feature_cols,
            query_index=query_index,
            query_values=query_values,
            n_similar=n_similar,
        )
        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Similarity error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


# ═══════════════════════════════════════════════════════════
#  NEW FEATURE ENDPOINTS
#  1. Clinical Insights   /api/clinical-insights/
#  2. Data Quality        /api/data-quality/
#  3. Generate Report     /api/generate-report/
#  4. Privacy Check       /api/privacy-check/
#  4b. Anonymise          /api/anonymise/
# ═══════════════════════════════════════════════════════════

from ml_engine.engine_extensions import (
    ClinicalInsightsEngine,
    DataQualityEngine,
    ReportGenerator,
    PrivacyEngine,
)


@require_POST
def clinical_insights(request):
    """
    🤖 Auto Clinical Insights — LLM narrative summary + chart interpretation.
    POST body: { "feature_columns": [...], "target_column": "col" }
    Uses the latest ML result from session. Falls back to rule-based if no LLM key.
    """
    try:
        session = _get_session(request)
        if session is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})

        ar = _get_latest_result(session)
        if ar is None:
            return JsonResponse({'success': False, 'error': 'Run an ML analysis first (Analytics tab).'})

        ml_result = ar.get_result()
        col_info  = session.get_columns()

        engine  = ClinicalInsightsEngine()
        llm     = _get_llm()
        insights = engine.generate_narrative(ml_result, col_info, llm_client=llm)

        return JsonResponse({'success': True, 'insights': insights})
    except Exception as e:
        logger.error("Insights error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_GET
def data_quality(request):
    """
    🧪 Data Quality & Bias Detection.
    GET — no body needed, runs on current session dataset.
    Returns missing data report, outlier detection, class imbalance, overall score.
    """
    try:
        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})

        engine = DataQualityEngine()
        result = engine.run_quality_check(df)
        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Quality error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def generate_report(request):
    """
    📄 Generate HTML Clinical Report (printable as PDF from browser).
    POST body: {} — uses current session data + latest ML result automatically.
    Returns { success, html } — the frontend opens in a new tab for print/save.
    """
    try:
        session = _get_session(request)
        if session is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})

        ar = _get_latest_result(session)
        if ar is None:
            return JsonResponse({'success': False, 'error': 'Run an ML analysis first to generate a report.'})

        ml_result = ar.get_result()
        col_info  = session.get_columns()
        df        = _get_df(request)

        # Generate insights (rule-based fallback if no LLM)
        insights_engine = ClinicalInsightsEngine()
        llm      = _get_llm()
        insights = insights_engine.generate_narrative(ml_result, col_info, llm_client=llm)

        # Run quality check
        quality = DataQualityEngine().run_quality_check(df) if df is not None else {
            'overall_score': 0, 'grade': '?', 'issues': [], 'warnings': [],
            'recommendation': 'No data available for quality check.',
        }

        # Generate HTML report
        session_info = {
            'file_name': session.file_name,
            'row_count': session.row_count,
            'col_count': session.col_count,
        }
        reporter = ReportGenerator()
        html     = reporter.generate_html_report(session_info, ml_result, insights, quality)

        return JsonResponse({'success': True, 'html': html})
    except Exception as e:
        logger.error("Report error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_GET
def privacy_check(request):
    """
    🔒 Privacy & PHI Detection.
    GET — scans current dataset for PHI columns, returns compliance score + recommendations.
    """
    try:
        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})

        engine = PrivacyEngine()
        result = engine.run_privacy_check(df)
        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Privacy check error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def anonymise_data(request):
    """
    🔐 Anonymise PHI columns and re-save the session dataset.
    POST body: {
        "columns":  ["name", "email"],   -- columns to anonymise
        "method":   "hash"               -- hash | mask | drop | pseudonymise
    }
    After anonymisation the session dataset is updated so all subsequent
    analysis uses the clean anonymised data.
    """
    try:
        data    = json.loads(request.body)
        columns = data.get('columns', [])
        method  = data.get('method', 'hash')

        if method not in ('hash', 'mask', 'drop', 'pseudonymise'):
            return JsonResponse({'success': False, 'error': f'Unknown method: {method}. Use hash/mask/drop/pseudonymise.'})

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})
        if not columns:
            return JsonResponse({'success': False, 'error': 'Specify at least one column to anonymise.'})

        engine          = PrivacyEngine()
        anon_df, audit  = engine.anonymise(df, columns, method=method)

        # Re-save anonymised dataset back into session
        session = _get_session(request)
        _save_session(
            request, anon_df,
            source_type=session.source_type if session else 'excel',
            file_name=(session.file_name if session else 'dataset') + ' [anonymised]',
        )

        return JsonResponse({
            'success':       True,
            'audit_log':     audit,
            'columns_after': list(anon_df.columns),
            'rows':          len(anon_df),
            'message':       f"{len(audit)} column(s) anonymised using '{method}' method. Dataset updated.",
        })
    except Exception as e:
        logger.error("Anonymise error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})
