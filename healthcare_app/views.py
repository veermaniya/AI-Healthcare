import json, uuid, io
import numpy as np
import json
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.conf import settings

from ml_engine import (
    HealthcareMLEngine, GroqLLMClient,
    PatientDashboardEngine, MultiTargetComparator,
    SurvivalAnalysisEngine, AlertRulesEngine,
    DatasetComparator, ClinicalCodingAssistant,
    ClinicalInsightsEngine, DataQualityEngine,
    ReportGenerator, PrivacyEngine,
    load_excel_file, load_from_sql, dataframe_summary, get_column_stats,
)
from .models import DataSession, ChatMessage, AnalysisResult
class NumpyEncoder(json.JSONEncoder):
    """Converts numpy types so Django JsonResponse never crashes."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
_dataframe_store = {}   # session_key → DataFrame
_temp_df_store   = {}   # session_key → temp DataFrame (dataset compare)

# ── helpers ─────────────────────────────────────────────────
def _sk(request):
    if 'healthcare_session' not in request.session:
        request.session['healthcare_session'] = str(uuid.uuid4())
    return request.session['healthcare_session']

# def _df(request):
# #     return _dataframe_store.get(_sk(request))
def _df(request):
    sk = _sk(request)
    if sk in _dataframe_store:
        return _dataframe_store[sk]
    # Session rotated or server restarted — recover from DB
    db_s = DataSession.objects.filter(session_key=sk).first()
    if db_s:
        df = db_s.get_dataframe()
        if df is not None:
            _dataframe_store[sk] = df
            return df
    return None

def _engine(request):
    df = _df(request)
    if df is None:
        return None, JsonResponse({'success': False, 'error': 'No dataset loaded. Please upload a file first.'})
    e = HealthcareMLEngine()
    e.load_dataframe(df)
    return e, None

# ── AUTH ─────────────────────────────────────────────────────
def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    error = ''
    if request.method == 'POST':
        u = authenticate(request, username=request.POST.get('username',''),
                         password=request.POST.get('password',''))
        if u:
            login(request, u); return redirect('index')
        error = 'Invalid username or password'
    return render(request, 'healthcare_app/login.html', {'error': error})
@csrf_exempt
def api_login(request):
    """JSON login endpoint for the frontend fetch() call."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        u = authenticate(request,
                         username=data.get('username', ''),
                         password=data.get('password', ''))
        if u:
            login(request, u)
            return JsonResponse({'success': True})
        return JsonResponse({'success': False, 'error': 'Invalid username or password'})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})
def logout_view(request):
    logout(request); return redirect('login')

# ── MAIN ─────────────────────────────────────────────────────
@login_required(login_url='/login/')
def index(request):
    sk = _sk(request)
    return render(request, 'healthcare_app/index.html', {
        'data_session': DataSession.objects.filter(session_key=sk).first(),
        'username': request.user.username,
        'user_fullname': request.user.get_full_name() or request.user.username,
    })

# ── UPLOAD ───────────────────────────────────────────────────
@csrf_exempt

def upload_file(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file provided'})
        df = load_excel_file(file)
        summary = dataframe_summary(df)
        _dataframe_store[sk] = df
        e = HealthcareMLEngine(); e.load_dataframe(df)
        col_info = e.get_column_info()
        DataSession.objects.update_or_create(session_key=sk, defaults={
            'source_type': 'excel', 'file_name': file.name,
            'columns_json': json.dumps(col_info), 'row_count': summary['rows'],
            'col_count': summary['columns'], 'preview_json': json.dumps(summary['preview']),
            'data_json': df.to_json(orient='records'),
        })
        return JsonResponse({'success': True, 'file_name': file.name,
                             'rows': summary['rows'], 'columns': summary['columns'],
                             'column_info': col_info, 'preview': summary['preview']})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── UPLOAD TEMP (dataset compare) ────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def upload_temp(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file'})
        df = load_excel_file(file)
        summary = dataframe_summary(df)
        _temp_df_store[sk] = df
        return JsonResponse({'success': True, 'file_name': file.name,
                             'rows': summary['rows'], 'columns': summary['columns']})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── SQL ───────────────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def connect_sql(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        data = json.loads(request.body)
        df = load_from_sql(data.get('connection_string',''), data.get('query',''))
        summary = dataframe_summary(df)
        _dataframe_store[sk] = df
        e = HealthcareMLEngine(); e.load_dataframe(df)
        col_info = e.get_column_info()
        DataSession.objects.update_or_create(session_key=sk, defaults={
            'source_type': 'sql', 'file_name': 'SQL Query',
            'columns_json': json.dumps(col_info), 'row_count': summary['rows'],
            'col_count': summary['columns'], 'preview_json': json.dumps(summary['preview']),
        })
        return JsonResponse({'success': True, 'file_name': 'SQL Query',
                             'rows': summary['rows'], 'columns': summary['columns'],
                             'column_info': col_info, 'preview': summary['preview']})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── RUN ANALYSIS ─────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def run_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        data  = json.loads(request.body)
        feats = data.get('feature_columns', [])
        tgt   = data.get('target_column', '')
        task  = data.get('task_type', 'auto')
        model = data.get('model_type', 'random_forest')
        k     = int(data.get('n_clusters', 3))
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        if not feats:
            return JsonResponse({'success': False, 'error': 'Select at least one feature column'})
        e = HealthcareMLEngine(); e.load_dataframe(df)
        if task == 'auto' and tgt:
            task = e.auto_detect_task(tgt)
        if task == 'clustering' or not tgt:
            result = e.run_clustering(feats, n_clusters=k)
        elif task == 'regression':
            result = e.run_regression(feats, tgt, model)
        elif task == 'classification':
            result = e.run_classification(feats, tgt, model)
        else:
            result = e.run_regression(feats, tgt, model)
        db_s = DataSession.objects.filter(session_key=sk).first()
        if db_s:
            AnalysisResult.objects.create(session=db_s, task_type=task,
                feature_columns=','.join(feats), target_column=tgt or '',
                model_type=model, result_json=json.dumps(result))
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── PREDICT ───────────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def predict_input(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        e, err = _engine(request)
        if err: return err
        result = e.predict_new_input(data.get('feature_columns',[]),
                                     data.get('target_column',''),
                                     data.get('input_values',{}))
                                    #  data.get('model_type','random_forest'))
        return JsonResponse({'success': True, 'prediction': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── XAI EXPLAINABILITY ────────────────────────────────────────
@csrf_exempt
def run_explainability(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        e, err = _engine(request)
        if err: return err
        result = e.run_explainability(
            feature_cols=data.get('feature_columns', []),
            target_col=data.get('target_column', ''),
            model_type=data.get('model_type', 'random_forest'),
            sample_index=data.get('sample_index', 0),
            input_values=data.get('input_values', {})
        )
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── PATIENT DASHBOARD ─────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def patient_dashboard(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data  = json.loads(request.body)
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        e = HealthcareMLEngine(); e.load_dataframe(df)
        eng = PatientDashboardEngine()
        result = eng.get_patient_profile(
            df,
            row_index   = int(data.get('row_index', 0)),
            feature_cols= data.get('feature_columns', []),
            target_col  = data.get('target_column','') or None,
            ml_engine   = e,
        )
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── MULTI-TARGET COMPARE ──────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def multi_target_compare(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data  = json.loads(request.body)
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        eng = MultiTargetComparator()
        result = eng.compare_targets(df,
                                     feature_cols  = data.get('feature_columns',[]),
                                     target_cols   = data.get('target_columns',[]),
                                     model_type    = data.get('model_type','random_forest'))
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── SURVIVAL ANALYSIS ─────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def survival_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        eng = SurvivalAnalysisEngine()
        result = eng.run_kaplan_meier(df,
                                      duration_col = data.get('duration_col',''),
                                      event_col    = data.get('event_col',''),
                                      group_col    = data.get('group_col') or None)
        # ✅ Use NumpyEncoder
        return HttpResponse(
            json.dumps({'success': True, 'result': result}, cls=NumpyEncoder),
            content_type='application/json'
        )
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── ALERT RULES ───────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def evaluate_alert_rules(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        eng = AlertRulesEngine()
        result = eng.evaluate_rules(df, rules=data.get('rules',[]))
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

@csrf_exempt
@login_required(login_url='/login/')
def suggest_alert_rules(request):
    try:
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        eng = AlertRulesEngine()
        rules = eng.suggest_rules(df)
        return JsonResponse({'success': True, 'rules': rules})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── DATASET COMPARE ───────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def compare_datasets(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        data = json.loads(request.body)
        df_a = _dataframe_store.get(sk)
        df_b = _temp_df_store.get(sk)
        if df_a is None or df_b is None:
            return JsonResponse({'success': False, 'error': 'Upload both datasets first'})
        eng = DatasetComparator()
        result = eng.compare(df_a, df_b,
                             feature_cols=data.get('feature_columns',[]),
                             label='Dataset A vs Dataset B')
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── PATIENT SIMILARITY ────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def run_patient_similarity(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        e, err = _engine(request)
        if err: return err
        result = e.find_similar_patients(
            data.get('feature_columns',[]),
            data.get('query_values',{}),
            data.get('n_similar', 5))
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── RISK ENGINE ───────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def run_risk_engine(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        e, err = _engine(request)
        if err: return err
        result = e.run_risk_engine(data.get('feature_columns',[]))
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── TREND ANALYSIS ────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def run_trend_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data = json.loads(request.body)
        e, err = _engine(request)
        if err: return err
        result = e.run_trend_analysis(data.get('feature_columns',[]),
                                      forecast_steps=data.get('forecast_steps', 10))
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── CLINICAL INSIGHTS ─────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def clinical_insights(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        data = json.loads(request.body)
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        e = HealthcareMLEngine(); e.load_dataframe(df)
        col_info = e.get_column_info()
        latest = AnalysisResult.objects.filter(
            session__session_key=sk).order_by('-created_at').first()
        ml_result = latest.get_result() if latest else None
        llm = GroqLLMClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
        eng = ClinicalInsightsEngine()
        result = eng.generate_narrative(ml_result or {}, col_info, llm_client=llm)
        return JsonResponse({'success': True, 'insights': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── DATA QUALITY ──────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def data_quality(request):
    try:
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        eng = DataQualityEngine()
        result = eng.run_quality_check(df)
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── PRIVACY CHECK ─────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def privacy_check(request):
    try:
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        eng = PrivacyEngine()
        result = eng.run_privacy_check(df)
        return JsonResponse({'success': True, 'result': result})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── ANONYMISE ─────────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def anonymise(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        data = json.loads(request.body)
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        eng = PrivacyEngine()
        df_anon, summary = eng.anonymise(df,
                                         phi_columns=data.get('columns',[]),
                                         method=data.get('method','mask'))
        _dataframe_store[sk] = df_anon
        e2 = HealthcareMLEngine(); e2.load_dataframe(df_anon)
        col_info = e2.get_column_info()
        s = dataframe_summary(df_anon)
        DataSession.objects.filter(session_key=sk).update(
            columns_json=json.dumps(col_info),
            preview_json=json.dumps(s['preview']))
        return JsonResponse({'success': True, 'masked_columns': summary,
                             'rows': len(df_anon), 'column_info': col_info,
                             'preview': s['preview'][:20]})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── CLINICAL CODING ───────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def clinical_coding(request):
    try:
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        e = HealthcareMLEngine(); e.load_dataframe(df)
        col_info = e.get_column_info()
        llm = GroqLLMClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
        eng = ClinicalCodingAssistant()
        icd  = eng.detect_icd_codes(df)
        drug = eng.check_drug_interactions(df)
        advice = eng.get_llm_coding_advice(df, col_info, llm)
        return JsonResponse({'success': True, 'icd_codes': icd,
                             'drug_interactions': drug, 'llm_advice': advice})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── PDF REPORT ────────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def generate_report(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        df = _df(request)
        db_s = DataSession.objects.filter(session_key=sk).first()
        latest = AnalysisResult.objects.filter(
            session__session_key=sk).order_by('-created_at').first()
        ml_result = latest.get_result() if latest else {}
        session_info = {
            'file_name': db_s.file_name if db_s else 'Unknown',
            'rows': db_s.row_count if db_s else 0,
            'username': request.user.username,
            'generated_at': __import__('datetime').datetime.now().strftime('%d %b %Y %H:%M'),
        }
        eng = ReportGenerator()
        e = HealthcareMLEngine()
        if df is not None:
            e.load_dataframe(df)
        col_info = e.get_column_info() if df is not None else []
        html = eng.generate_html_report(session_info, ml_result, col_info, df)
        # Try PDF via reportlab
        try:
            import subprocess, sys
            from reportlab.lib.pagesizes import A4
            buf = io.BytesIO()
            # Simple fallback: return HTML as downloadable if PDF too complex
            response = HttpResponse(html, content_type='text/html')
            response['Content-Disposition'] = 'attachment; filename="HealthAI_Report.html"'
            return response
        except ImportError:
            response = HttpResponse(html, content_type='text/html')
            response['Content-Disposition'] = 'attachment; filename="HealthAI_Report.html"'
            return response
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── CHAT ─────────────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def chat(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    sk = _sk(request)
    try:
        data = json.loads(request.body)
        msg  = data.get('message','').strip()
        if not msg:
            return JsonResponse({'success': False, 'error': 'Empty message'})
        ChatMessage.objects.create(session_key=sk, role='user', message=msg)
        dataset_context = None
        ml_result = None
        if data.get('include_dataset'):
            db_s = DataSession.objects.filter(session_key=sk).first()
            if db_s:
                dataset_context = {'file': db_s.file_name, 'rows': db_s.row_count,
                                   'columns': db_s.get_columns()[:20]}
            latest = AnalysisResult.objects.filter(
                session__session_key=sk).order_by('-created_at').first()
            if latest:
                ml_result = latest.get_result()
        llm = GroqLLMClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
        resp = llm.ask(msg, dataset_context, ml_result)
        ChatMessage.objects.create(session_key=sk, role='ai', message=resp['message'])
        return JsonResponse({'success': True, 'message': resp['message'],
                             'tokens': resp.get('tokens_used', 0)})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── IMAGE ANALYSIS ────────────────────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def analyze_image(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        import requests as req
        data = json.loads(request.body)
        key  = settings.GROQ_API_KEY
        if not key or key == 'your-groq-api-key-here':
            return JsonResponse({'success': False, 'error': 'Groq API key not configured'})
        vision_models = ['meta-llama/llama-4-scout-17b-16e-instruct',
                         'llama-3.2-11b-vision-preview',
                         'llama-3.2-90b-vision-preview']
        headers   = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        last_err  = None
        for model in vision_models:
            payload = {'model': model, 'max_tokens': 1500, 'temperature': 0.3,
                       'messages': [{'role': 'user', 'content': [
                           {'type': 'image_url', 'image_url': {
                               'url': f"data:{data.get('mime_type','image/jpeg')};base64,{data.get('image_base64','')}"}},
                           {'type': 'text', 'text': data.get('prompt','Analyze this medical image.')},
                       ]}]}
            try:
                r = req.post('https://api.groq.com/openai/v1/chat/completions',
                             headers=headers, json=payload, timeout=60)
                if not r.ok:
                    try: last_err = r.json().get('error',{}).get('message', r.text)
                    except: last_err = r.text
                    continue
                return JsonResponse({'success': True,
                                     'result': r.json()['choices'][0]['message']['content'],
                                     'model_used': model})
            except Exception as ex:
                last_err = str(ex); continue
        return JsonResponse({'success': False, 'error': f'All vision models failed. {last_err}'})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── SESSION HELPERS ───────────────────────────────────────────
@csrf_exempt
def get_columns(request):
    sk = _sk(request)
    db_s = DataSession.objects.filter(session_key=sk).first()
    if not db_s:
        return JsonResponse({'success': False, 'columns': []})
    return JsonResponse({'success': True, 'columns': db_s.get_columns(),
                         'preview': db_s.get_preview(), 'rows': db_s.row_count})

@csrf_exempt
def get_column_stats_view(request):
    df = _df(request)
    if df is None:
        return JsonResponse({'success': False, 'stats': {}})
    try:
        return JsonResponse({'success': True, 'stats': get_column_stats(df)})
    except Exception as ex:
        return JsonResponse({'success': False, 'stats': {}, 'error': str(ex)})

@csrf_exempt
def clear_session(request):
    sk = _sk(request)
    _dataframe_store.pop(sk, None)
    _temp_df_store.pop(sk, None)
    DataSession.objects.filter(session_key=sk).delete()
    return JsonResponse({'success': True})

# ── FILTER DATA (basic row filter) ────────────────────────────
@csrf_exempt
@login_required(login_url='/login/')
def filter_data(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'})
    try:
        data  = json.loads(request.body)
        df = _df(request)
        if df is None:
            return JsonResponse({'success': False, 'error': 'No dataset loaded'})
        filters = data.get('filters', [])  # [{col, op, value}]
        df_f = df.copy()
        for f in filters:
            col, op, val = f.get('col'), f.get('op'), f.get('value')
            if col not in df_f.columns:
                continue
            try:
                if op == '==':   df_f = df_f[df_f[col].astype(str) == str(val)]
                elif op == '!=': df_f = df_f[df_f[col].astype(str) != str(val)]
                elif op == '>':  df_f = df_f[df_f[col] > float(val)]
                elif op == '<':  df_f = df_f[df_f[col] < float(val)]
                elif op == '>=': df_f = df_f[df_f[col] >= float(val)]
                elif op == '<=': df_f = df_f[df_f[col] <= float(val)]
                elif op == 'contains': df_f = df_f[df_f[col].astype(str).str.contains(str(val), na=False)]
            except Exception:
                pass
        preview = df_f.head(500).fillna('').to_dict(orient='records')
        return JsonResponse({'success': True, 'rows': len(df_f),
                             'preview': preview, 'filtered': True})
    except Exception as ex:
        return JsonResponse({'success': False, 'error': str(ex)})

# ── DEBUG / TEST ─────────────────────────────────────────────
@csrf_exempt
def test_llm(request):
    key = settings.GROQ_API_KEY
    if not key or key == 'your-groq-api-key-here':
        return JsonResponse({'api_key_set': False, 'message': 'KEY NOT SET'})
    llm = GroqLLMClient(api_key=key)
    r   = llm.ask("Say hello in one sentence.")
    return JsonResponse({'api_key_set': True, 'api_key_preview': key[:8]+'...'+key[-4:],
                         'llm_success': r['success'], 'llm_message': r['message'],
                         'model_used': r.get('model_used','—')})

@csrf_exempt
def debug_key(request):
    key = settings.GROQ_API_KEY
    ph  = (key == 'your-groq-api-key-here' or not key)
    return JsonResponse({'key_preview': (key[:8]+'...'+key[-4:]) if key and len(key)>12 else key or 'EMPTY',
                         'is_placeholder': ph, 'key_length': len(key) if key else 0,
                         'starts_with_gsk': key.startswith('gsk_') if key else False})
