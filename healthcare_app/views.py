import json
import uuid
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

from .models import DataSession, AnalysisResult, ChatMessage
from ml_engine import (
    HealthcareMLEngine, GroqLLMClient,
    load_excel_file, load_from_sql, dataframe_summary
)

# In-memory store for DataFrames (per session)
_dataframe_store = {}


def get_or_create_session_key(request):
    if 'healthcare_session' not in request.session:
        request.session['healthcare_session'] = str(uuid.uuid4())
    return request.session['healthcare_session']


# ─────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────
def index(request):
    session_key = get_or_create_session_key(request)
    data_session = DataSession.objects.filter(session_key=session_key).first()
    chat_messages = ChatMessage.objects.filter(session_key=session_key).order_by('created_at')[:50]
    recent_results = AnalysisResult.objects.filter(session__session_key=session_key).order_by('-created_at')[:5]

    return render(request, 'healthcare_app/index.html', {
        'data_session': data_session,
        'chat_messages': chat_messages,
        'recent_results': recent_results,
        'session_key': session_key,
    })


# ─────────────────────────────────────────────
#  FILE UPLOAD
# ─────────────────────────────────────────────
@csrf_exempt
def upload_file(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    session_key = get_or_create_session_key(request)

    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file provided'})

        df = load_excel_file(file)
        summary = dataframe_summary(df)

        # Store DataFrame in memory
        _dataframe_store[session_key] = df

        # Build column info
        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        col_info = engine.get_column_info()

        # Save or update session
        data_session, created = DataSession.objects.update_or_create(
            session_key=session_key,
            defaults={
                'source_type': 'excel',
                'file_name': file.name,
                'columns_json': json.dumps(col_info),
                'row_count': summary['rows'],
                'col_count': summary['columns'],
                'preview_json': json.dumps(summary['preview']),
            }
        )

        return JsonResponse({
            'success': True,
            'file_name': file.name,
            'rows': summary['rows'],
            'columns': summary['columns'],
            'column_info': col_info,
            'preview': summary['preview'],
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────
#  SQL CONNECTION
# ─────────────────────────────────────────────
@csrf_exempt
def connect_sql(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    session_key = get_or_create_session_key(request)

    try:
        data = json.loads(request.body)
        connection_string = data.get('connection_string', '')
        query = data.get('query', '')

        if not connection_string or not query:
            return JsonResponse({'success': False, 'error': 'Connection string and query required'})

        df = load_from_sql(connection_string, query)
        summary = dataframe_summary(df)
        _dataframe_store[session_key] = df

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        col_info = engine.get_column_info()

        data_session, _ = DataSession.objects.update_or_create(
            session_key=session_key,
            defaults={
                'source_type': 'sql',
                'sql_connection': connection_string,
                'sql_query': query,
                'columns_json': json.dumps(col_info),
                'row_count': summary['rows'],
                'col_count': summary['columns'],
                'preview_json': json.dumps(summary['preview']),
            }
        )

        return JsonResponse({
            'success': True,
            'rows': summary['rows'],
            'columns': summary['columns'],
            'column_info': col_info,
            'preview': summary['preview'],
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────
#  RUN ML ANALYSIS
# ─────────────────────────────────────────────
@csrf_exempt
def run_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    session_key = get_or_create_session_key(request)

    try:
        data = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col = data.get('target_column', '')
        task_type = data.get('task_type', 'auto')  # regression, classification, clustering, auto
        model_type = data.get('model_type', 'random_forest')
        n_clusters = int(data.get('n_clusters', 3))

        df = _dataframe_store.get(session_key)
        if df is None:
            # Try reloading from DB session
            db_session = DataSession.objects.filter(session_key=session_key).first()
            if db_session and db_session.source_type == 'sql':
                df = load_from_sql(db_session.sql_connection, db_session.sql_query)
                _dataframe_store[session_key] = df
            else:
                return JsonResponse({'success': False, 'error': 'No dataset loaded. Please upload a file first.'})

        if not feature_cols:
            return JsonResponse({'success': False, 'error': 'Please select at least one feature column'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)

        # Auto-detect or use specified task
        if task_type == 'auto' and target_col:
            task_type = engine.auto_detect_task(target_col)

        # Run the appropriate task
        if task_type == 'clustering' or not target_col:
            result = engine.run_clustering(feature_cols, n_clusters=n_clusters)
        elif task_type == 'regression':
            result = engine.run_regression(feature_cols, target_col, model_type)
        elif task_type == 'classification':
            result = engine.run_classification(feature_cols, target_col, model_type)
        else:
            result = engine.run_regression(feature_cols, target_col, model_type)

        # Save result
        db_session = DataSession.objects.filter(session_key=session_key).first()
        if db_session:
            AnalysisResult.objects.create(
                session=db_session,
                task_type=task_type,
                feature_columns=','.join(feature_cols),
                target_column=target_col or '',
                model_type=model_type,
                result_json=json.dumps(result)
            )

        return JsonResponse({'success': True, 'result': result})

    except Exception as e:
        import traceback
        return JsonResponse({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


# ─────────────────────────────────────────────
#  PREDICT SINGLE INPUT
# ─────────────────────────────────────────────
@csrf_exempt
def predict_input(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    session_key = get_or_create_session_key(request)

    try:
        data = json.loads(request.body)
        feature_cols = data.get('feature_columns', [])
        target_col = data.get('target_column', '')
        input_values = data.get('input_values', {})

        df = _dataframe_store.get(session_key)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})

        engine = HealthcareMLEngine()
        engine.load_dataframe(df)
        result = engine.predict_new_input(feature_cols, target_col, input_values)

        return JsonResponse({'success': True, 'prediction': result})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────
#  CHATBOT
# ─────────────────────────────────────────────
@csrf_exempt
def chat(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})

    session_key = get_or_create_session_key(request)

    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        include_dataset_context = data.get('include_dataset', True)
        last_result_id = data.get('last_result_id', None)

        if not user_message:
            return JsonResponse({'success': False, 'error': 'Empty message'})

        # Save user message
        ChatMessage.objects.create(
            session_key=session_key,
            role='user',
            message=user_message
        )

        # Build context
        dataset_context = None
        ml_result = None

        if include_dataset_context:
            db_session = DataSession.objects.filter(session_key=session_key).first()
            if db_session:
                dataset_context = {
                    'file': db_session.file_name or 'SQL Query',
                    'rows': db_session.row_count,
                    'columns': [c['name'] for c in db_session.get_columns()],
                    'column_types': {c['name']: c['col_type'] for c in db_session.get_columns()}
                }

        if last_result_id:
            try:
                analysis = AnalysisResult.objects.get(id=last_result_id)
                ml_result = analysis.get_result()
            except:
                pass
        else:
            # Get latest result
            latest = AnalysisResult.objects.filter(
                session__session_key=session_key
            ).order_by('-created_at').first()
            if latest:
                ml_result = latest.get_result()

        # Call LLM
        llm = GroqLLMClient(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL
        )
        response = llm.ask(user_message, dataset_context, ml_result)

        # Save AI response
        ChatMessage.objects.create(
            session_key=session_key,
            role='ai',
            message=response['message']
        )

        return JsonResponse({
            'success': True,
            'message': response['message'],
            'tokens': response.get('tokens_used', 0)
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────
#  GET COLUMNS (AJAX)
# ─────────────────────────────────────────────
@csrf_exempt
def get_columns(request):
    session_key = get_or_create_session_key(request)
    db_session = DataSession.objects.filter(session_key=session_key).first()
    if not db_session:
        return JsonResponse({'success': False, 'columns': []})
    return JsonResponse({
        'success': True,
        'columns': db_session.get_columns(),
        'preview': db_session.get_preview(),
        'rows': db_session.row_count,
    })


# ─────────────────────────────────────────────
#  GET COLUMN STATS (for smart prediction form)
# ─────────────────────────────────────────────
@csrf_exempt
def get_column_stats(request):
    """
    Returns per-column stats for building smart input widgets:
    - numeric columns: min, max, mean, median, std
    - categorical columns: all unique values (for dropdown)
    """
    session_key = get_or_create_session_key(request)
    df = _dataframe_store.get(session_key)

    if df is None:
        db_session = DataSession.objects.filter(session_key=session_key).first()
        if db_session and db_session.source_type == 'sql':
            try:
                df = load_from_sql(db_session.sql_connection, db_session.sql_query)
                _dataframe_store[session_key] = df
            except:
                pass

    if df is None:
        return JsonResponse({'success': False, 'stats': {}})

    stats = {}
    for col in df.columns:
        col_data = df[col].dropna()
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            stats[col] = {
                'type': 'numeric',
                'min': round(float(col_data.min()), 2),
                'max': round(float(col_data.max()), 2),
                'mean': round(float(col_data.mean()), 2),
                'median': round(float(col_data.median()), 2),
                'std': round(float(col_data.std()), 2),
                # Suggest a step size based on range
                'step': round(float((col_data.max() - col_data.min()) / 100), 3) or 1,
            }
        else:
            unique_vals = sorted(col_data.astype(str).unique().tolist())
            stats[col] = {
                'type': 'categorical',
                'unique_values': unique_vals,
                'most_common': col_data.mode()[0] if len(col_data) > 0 else '',
            }

    return JsonResponse({'success': True, 'stats': stats})



# ─────────────────────────────────────────────
#  MEDICAL IMAGE ANALYSIS
# ─────────────────────────────────────────────
@csrf_exempt
def analyze_image(request):
    """Analyze medical image (X-ray, CT, Sonography, MRI) using Groq vision model"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        image_base64 = data.get('image_base64', '')
        prompt = data.get('prompt', '')
        scan_type = data.get('scan_type', 'xray')
        mime_type = data.get('mime_type', 'image/jpeg')

        if not image_base64:
            return JsonResponse({'success': False, 'error': 'No image provided'})

        api_key = settings.GROQ_API_KEY
        if not api_key or api_key == 'your-groq-api-key-here':
            return JsonResponse({'success': False, 'error': 'Groq API key not configured. Please set GROQ_API_KEY in settings.py'})

        import requests as req

        # Use Groq vision model
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': 'meta-llama/llama-4-scout-17b-16e-instruct',  # Groq vision model
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:{mime_type};base64,{image_base64}'
                            }
                        },
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ],
            'max_tokens': 2048,
            'temperature': 0.3
        }

        # Try vision models in order
        vision_models = [
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'llama-3.2-11b-vision-preview',
            'llama-3.2-90b-vision-preview',
        ]

        last_error = None
        for model in vision_models:
            payload['model'] = model
            try:
                response = req.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers=headers, json=payload, timeout=60
                )
                if response.ok:
                    result = response.json()
                    text = result['choices'][0]['message']['content']
                    return JsonResponse({'success': True, 'result': text, 'model': model})
                else:
                    try:
                        err = response.json().get('error', {}).get('message', response.text)
                    except:
                        err = response.text
                    last_error = f'{model}: {err}'
                    continue
            except Exception as e:
                last_error = str(e)
                continue

        return JsonResponse({'success': False, 'error': f'Vision model error: {last_error}'})

    except Exception as e:
        import traceback
        return JsonResponse({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

# ─────────────────────────────────────────────
#  TEST LLM CONNECTION
# ─────────────────────────────────────────────
@csrf_exempt
def test_llm(request):
    """Quick endpoint to verify Groq API key works — visit /api/test-llm/ in browser"""
    key = settings.GROQ_API_KEY
    key_set = bool(key and key != 'your-groq-api-key-here')
    
    if not key_set:
        return JsonResponse({
            'api_key_set': False,
            'api_key_preview': 'NOT SET — open healthcare_ai/settings.py and replace your-groq-api-key-here',
            'llm_success': False,
            'llm_message': 'API key not configured. Open healthcare_ai/settings.py and set your key from https://console.groq.com',
            'model_used': '—',
        })
    
    from ml_engine import GroqLLMClient
    llm = GroqLLMClient(api_key=key)
    result = llm.ask("Say hello in one sentence.")
    return JsonResponse({
        'api_key_set': True,
        'api_key_preview': key[:8] + '...' + key[-4:],
        'llm_success': result['success'],
        'llm_message': result['message'],
        'model_used': result.get('model_used', '—'),
        'fix': 'If 401 error: your key is invalid or expired. Generate a new one at https://console.groq.com/keys' if not result['success'] else 'Working correctly!',
    })


# ─────────────────────────────────────────────
#  CLEAR SESSION
# ─────────────────────────────────────────────
@csrf_exempt
def clear_session(request):
    session_key = get_or_create_session_key(request)
    DataSession.objects.filter(session_key=session_key).delete()
    ChatMessage.objects.filter(session_key=session_key).delete()
    if session_key in _dataframe_store:
        del _dataframe_store[session_key]
    return JsonResponse({'success': True})

# ─────────────────────────────────────────────
#  DEBUG API KEY (visit /api/debug-key/ to check)
# ─────────────────────────────────────────────
@csrf_exempt
def debug_key(request):
    key = settings.GROQ_API_KEY
    is_placeholder = (key == 'your-groq-api-key-here' or not key)
    # Show first 8 and last 4 chars only
    if key and len(key) > 12:
        preview = key[:8] + '...' + key[-4:]
    else:
        preview = key or 'EMPTY'
    
    return JsonResponse({
        'key_preview': preview,
        'is_placeholder': is_placeholder,
        'key_length': len(key) if key else 0,
        'starts_with_gsk': key.startswith('gsk_') if key else False,
        'settings_file': str(settings.BASE_DIR / 'healthcare_ai' / 'settings.py'),
        'message': 'KEY IS PLACEHOLDER - edit settings.py' if is_placeholder else 'Key looks set - if still 401, key may be invalid/expired on Groq',
    })
