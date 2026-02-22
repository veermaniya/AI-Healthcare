"""
healthcare_app/views.py
Full views file — original endpoints + 4 new AI endpoints:
  • /api/run-explainability/
  • /api/run-risk-engine/
  • /api/run-trend-analysis/
  • /api/run-patient-similarity/
"""

import json
import pickle
import base64
import logging
import pandas as pd

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.conf import settings

from ml_engine import HealthcareMLEngine, GroqLLMClient, load_excel_file, load_from_sql, dataframe_summary
from .models import DataSession, AnalysisResult, ChatMessage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def index(request):
    """Serve the main single-page UI."""
    return render(request, 'healthcare_app/index.html')


def _get_session(request):
    """Return the current DataSession or None."""
    key = request.session.session_key
    if not key:
        request.session.create()
        key = request.session.session_key
    try:
        return DataSession.objects.filter(session_key=key).latest('created_at')
    except DataSession.DoesNotExist:
        return None


def _get_df(request) -> pd.DataFrame | None:
    """Deserialise DataFrame from session storage."""
    session = _get_session(request)
    if session is None or not session.dataframe_pickle:
        return None
    try:
        return pickle.loads(base64.b64decode(session.dataframe_pickle))
    except Exception as e:
        logger.error("DataFrame deserialise error: %s", e)
        return None


def _save_df(request, df: pd.DataFrame, file_name: str = 'dataset'):
    """Serialise and store DataFrame in session."""
    key = request.session.session_key
    if not key:
        request.session.create()
        key = request.session.session_key
    pickled = base64.b64encode(pickle.dumps(df)).decode()
    DataSession.objects.update_or_create(
        session_key=key,
        defaults={
            'dataframe_pickle': pickled,
            'file_name': file_name,
            'row_count': len(df),
            'col_count': len(df.columns),
        }
    )


def _get_llm():
    """Return a configured GroqLLMClient."""
    return GroqLLMClient(
        api_key=getattr(settings, 'GROQ_API_KEY', ''),
        model=getattr(settings, 'GROQ_MODEL', 'llama-3.3-70b-versatile'),
    )


# ═══════════════════════════════════════════════════════════
#  ORIGINAL ENDPOINTS
# ═══════════════════════════════════════════════════════════

@require_POST
def upload_file(request):
    """Upload Excel / CSV dataset."""
    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file provided.'})

        df = load_excel_file(file)
        _save_df(request, df, file_name=file.name)

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        col_info = engine.get_column_info()
        summary  = dataframe_summary(df)

        return JsonResponse({
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'file_name': file.name,
            'column_info': col_info,
            'preview': summary['preview'],
        })
    except Exception as e:
        logger.error("Upload error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def connect_sql(request):
    """Connect to SQL Server / PostgreSQL / MySQL and load query."""
    try:
        data = json.loads(request.body)
        conn_str = data.get('connection_string', '').strip()
        query    = data.get('query', '').strip()

        if not conn_str or not query:
            return JsonResponse({'success': False, 'error': 'connection_string and query required.'})

        df = load_from_sql(conn_str, query)
        _save_df(request, df, file_name='SQL Query')

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        col_info = engine.get_column_info()
        summary  = dataframe_summary(df)

        return JsonResponse({
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'file_name': 'SQL Query',
            'column_info': col_info,
            'preview': summary['preview'],
        })
    except Exception as e:
        logger.error("SQL connect error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_GET
def get_columns(request):
    """Return column info for the current session dataset."""
    try:
        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'columns': [], 'rows': 0})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        col_info = engine.get_column_info()
        summary  = dataframe_summary(df)
        session  = _get_session(request)

        return JsonResponse({
            'success': True,
            'columns': col_info,
            'rows': len(df),
            'file_name': session.file_name if session else 'dataset',
            'preview': summary['preview'],
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e), 'columns': [], 'rows': 0})


@require_GET
def get_column_stats(request):
    """Return per-column statistics for slider hints in the prediction form."""
    try:
        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'stats': {}})

        import numpy as np
        stats = {}
        for col in df.columns:
            if df[col].dtype in [float, int, 'float64', 'int64', 'float32', 'int32']:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                step = round(float((s.max() - s.min()) / 100), 4) or 0.01
                stats[col] = {
                    'type': 'numeric',
                    'min':    round(float(s.min()),    3),
                    'max':    round(float(s.max()),    3),
                    'mean':   round(float(s.mean()),   3),
                    'median': round(float(s.median()), 3),
                    'step':   step,
                }
            else:
                uvals = df[col].dropna().astype(str).unique().tolist()
                most  = df[col].value_counts().idxmax() if len(df[col].dropna()) else ''
                stats[col] = {
                    'type': 'categorical',
                    'unique_values': uvals[:20],
                    'most_common': str(most),
                }
        return JsonResponse({'success': True, 'stats': stats})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e), 'stats': {}})


@require_POST
def run_analysis(request):
    """Run ML analysis — regression, classification, or clustering."""
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col   = data.get('target_column', '')
        task_type    = data.get('task_type', 'auto')
        model_type   = data.get('model_type', 'random_forest')
        n_clusters   = int(data.get('n_clusters', 3))

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select at least one feature column.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)

        if task_type == 'clustering':
            result = engine.run_clustering(feature_cols, n_clusters=n_clusters)
        else:
            if not target_col:
                return JsonResponse({'success': False, 'error': 'Select a target column.'})
            if task_type == 'auto':
                task_type = engine.auto_detect_task(target_col)
            if task_type == 'regression':
                result = engine.run_regression(feature_cols, target_col, model_type)
            else:
                result = engine.run_classification(feature_cols, target_col, model_type)

        # Build sample_predictions for scatter chart
        if result.get('task') in ('regression', 'classification'):
            preds   = result.pop('predictions', [])
            actuals = result.pop('actuals', [])
            result['sample_predictions'] = [
                {'actual': a, 'predicted': p}
                for a, p in zip(actuals[:50], preds[:50])
            ]
            result['total_rows']  = len(df)
            result['features']    = feature_cols
            result['target']      = target_col
            if result['task'] == 'regression':
                m = result['metrics']
                m['r2_percent']  = round(m['r2'] * 100, 1)
            else:
                m = result['metrics']
                m['accuracy_percent'] = round(m['accuracy'] * 100, 1)
        elif result.get('task') == 'clustering':
            result['total_rows'] = len(df)
            result['features']   = feature_cols
            dist = []
            for cs in result.get('cluster_stats', []):
                dist.append({'cluster': f"Cluster {cs['cluster']}", 'count': cs['size']})
            result['cluster_distribution'] = dist

        # Persist result for chat context
        AnalysisResult.objects.update_or_create(
            session_key=request.session.session_key,
            defaults={'result_json': json.dumps(result)}
        )

        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Analysis error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def predict(request):
    """Single-patient prediction using the trained session model."""
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


@require_POST
def chat(request):
    """LLM chat endpoint — uses Groq LLaMA with dataset + ML context."""
    try:
        data    = json.loads(request.body)
        message = data.get('message', '').strip()
        if not message:
            return JsonResponse({'success': False, 'error': 'Empty message.'})

        # Build context
        dataset_context = None
        ml_result       = None

        df = _get_df(request)
        if df is not None:
            dataset_context = {
                'rows': len(df),
                'columns': list(df.columns),
                'dtypes': {col: str(df[col].dtype) for col in df.columns},
                'preview': df.head(5).fillna('').to_dict(orient='records'),
            }

        try:
            ar = AnalysisResult.objects.filter(
                session_key=request.session.session_key
            ).latest('created_at')
            ml_result = json.loads(ar.result_json)
        except Exception:
            pass

        llm = _get_llm()
        result = llm.ask(message, dataset_context=dataset_context, ml_result=ml_result)

        ChatMessage.objects.create(
            session_key=request.session.session_key,
            role='user',
            content=message,
        )
        if result['success']:
            ChatMessage.objects.create(
                session_key=request.session.session_key,
                role='assistant',
                content=result['message'],
            )

        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def analyze_image(request):
    """Medical imaging AI analysis via Groq vision model."""
    try:
        data        = json.loads(request.body)
        image_b64   = data.get('image_base64', '')
        prompt      = data.get('prompt', '')
        scan_type   = data.get('scan_type', 'xray')
        mime_type   = data.get('mime_type', 'image/jpeg')

        if not image_b64:
            return JsonResponse({'success': False, 'error': 'No image data provided.'})

        api_key = getattr(settings, 'GROQ_API_KEY', '')
        if not api_key:
            return JsonResponse({'success': False, 'error': 'GROQ_API_KEY not set in settings.py.'})

        import requests as req
        payload = {
            'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{mime_type};base64,{image_b64}'},
                    },
                    {'type': 'text', 'text': prompt},
                ],
            }],
            'max_tokens': 1024,
            'temperature': 0.3,
        }

        # Fallback model list
        models = [
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'llama-3.2-11b-vision-preview',
            'llama-3.2-90b-vision-preview',
        ]
        last_error = 'Vision model unavailable.'
        for model in models:
            payload['model'] = model
            try:
                r = req.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json',
                    },
                    json=payload,
                    timeout=45,
                )
                if r.status_code == 200:
                    content = r.json()['choices'][0]['message']['content']
                    return JsonResponse({'success': True, 'result': content})
                elif r.status_code == 429:
                    last_error = 'Rate limit — trying next model...'
                    continue
                else:
                    last_error = r.json().get('error', {}).get('message', f'HTTP {r.status_code}')
            except Exception as e:
                last_error = str(e)

        return JsonResponse({'success': False, 'error': last_error})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def clear_session(request):
    """Clear all session data — dataset, analysis results, chat history."""
    try:
        key = request.session.session_key
        if key:
            DataSession.objects.filter(session_key=key).delete()
            AnalysisResult.objects.filter(session_key=key).delete()
            ChatMessage.objects.filter(session_key=key).delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ═══════════════════════════════════════════════════════════
#  4 NEW AI ENDPOINTS
# ═══════════════════════════════════════════════════════════

@require_POST
def run_explainability(request):
    """
    🧠 XAI — LIME-style per-feature contribution analysis.
    POST body: {
        feature_columns: [...],
        target_column: "col_name",
        sample_index: 0,          # which row to explain (default 0)
        model_type: "random_forest"
    }
    """
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col   = data.get('target_column', '')
        sample_index = int(data.get('sample_index', 0))
        model_type   = data.get('model_type', 'random_forest')

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded. Upload data first.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select at least one feature column.'})
        if not target_col:
            return JsonResponse({'success': False, 'error': 'Select a target column (click twice on a column).'})

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
        logger.error("Explainability error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def run_risk_engine(request):
    """
    ⚠️ Clinical Risk & Alert Engine.
    POST body: {
        feature_columns: [...],
        target_column: "col_name",   # optional
        thresholds: {                # optional — auto-detected if omitted
            "glucose": {"low": 70, "high": 140, "critical_high": 200}
        }
    }
    """
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col   = data.get('target_column', '') or None
        thresholds   = data.get('thresholds', None)

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded. Upload data first.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select at least one feature column.'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        result = engine.run_risk_engine(
            feature_cols=feature_cols,
            target_col=target_col,
            thresholds=thresholds,
        )
        return JsonResponse({'success': True, 'result': result})
    except Exception as e:
        logger.error("Risk engine error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def run_trend_analysis(request):
    """
    📈 Trend & Forecast Analysis.
    POST body: {
        feature_columns: [...],
        time_col: "date_column",     # optional
        target_column: "col_name",   # optional
        forecast_steps: 10
    }
    """
    try:
        data           = json.loads(request.body)
        feature_cols   = data.get('feature_columns', [])
        time_col       = data.get('time_col', None)
        target_col     = data.get('target_column', None)
        forecast_steps = int(data.get('forecast_steps', 10))

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded. Upload data first.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select at least one feature column.'})

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
        logger.error("Trend analysis error: %s", e)
        return JsonResponse({'success': False, 'error': str(e)})


@require_POST
def run_patient_similarity(request):
    """
    👥 Patient Similarity — K-Nearest Neighbours.
    POST body: {
        feature_columns: [...],
        query_index: 0,         # row number of query patient
        query_values: {...},    # OR pass custom values dict instead of row index
        n_similar: 5
    }
    """
    try:
        data         = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        query_index  = int(data.get('query_index', 0))
        query_values = data.get('query_values', None)
        n_similar    = int(data.get('n_similar', 5))

        df = _get_df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded. Upload data first.'})
        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Select at least one feature column.'})

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
