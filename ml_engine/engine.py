"""
Healthcare AI ML Engine — Enhanced v2
Features: Regression, Classification, Clustering, XAI (SHAP-style),
          Clinical Risk Alerts, Time-Series Forecasting, Patient Similarity
"""

import pandas as pd
import numpy as np
import json
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier, IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, silhouette_score
)
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class HealthcareMLEngine:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_dataframe(self, df: pd.DataFrame):
        self.df = df.copy()
        return self

    def get_column_info(self):
        if self.df is None:
            return []
        info = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            null_count = int(self.df[col].isnull().sum())
            col_type = 'numeric' if self.df[col].dtype in [np.float64, np.int64, np.float32, np.int32] else 'categorical'
            info.append({
                'name': col,
                'dtype': dtype,
                'col_type': col_type,
                'unique_values': int(unique_count),
                'null_count': null_count,
                'sample_values': [str(v) for v in self.df[col].dropna().head(3).tolist()]
            })
        return info

    def auto_detect_task(self, target_col):
        col = self.df[target_col]
        # Float columns are always regression
        if col.dtype in [np.float64, np.float32]:
            return 'regression'
        # Text/object columns are always classification
        if col.dtype == object:
            return 'classification'
        # Integer columns: check unique count
        n_unique = col.nunique()
        # If more than 20 unique integer values → treat as regression
        if n_unique > 20:
            return 'regression'
        # If very few unique values → classification
        unique_ratio = n_unique / len(self.df)
        if unique_ratio < 0.05 or n_unique <= 10:
            return 'classification'
        return 'regression'

    def preprocess(self, feature_cols, target_col=None):
        df_work = self.df[feature_cols + ([target_col] if target_col else [])].copy()
        df_work = df_work.dropna()

        for col in feature_cols:
            if df_work[col].dtype == object:
                le = LabelEncoder()
                df_work[col] = le.fit_transform(df_work[col].astype(str))
                self.label_encoders[col] = le

        if target_col and df_work[target_col].dtype == object:
            le = LabelEncoder()
            df_work[target_col] = le.fit_transform(df_work[target_col].astype(str))
            self.label_encoders[target_col] = le

        X = df_work[feature_cols].values
        y = df_work[target_col].values if target_col else None
        return X, y, feature_cols

    def _get_model(self, task, model_type):
        if task == 'regression':
            if model_type == 'linear':
                return LinearRegression()
            elif model_type == 'gradient_boost':
                return GradientBoostingRegressor(n_estimators=100, random_state=42)
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            if model_type == 'linear':
                return LogisticRegression(max_iter=500, random_state=42)
            elif model_type == 'gradient_boost':
                return GradientBoostingClassifier(n_estimators=100, random_state=42)
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def _get_importances(self, model, cols):
        """Extract feature importances for any model type."""
        if hasattr(model, 'feature_importances_'):
            raw = model.feature_importances_
            return [{'feature': c, 'importance': round(float(v), 4)}
                    for c, v in sorted(zip(cols, raw), key=lambda x: -x[1])]
        elif hasattr(model, 'coef_'):
            raw = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            total = raw.sum() or 1.0
            return [{'feature': c, 'importance': round(float(v / total), 4)}
                    for c, v in sorted(zip(cols, raw), key=lambda x: -x[1])]
        return []

    # ─────────────────────────────────────────────
    # STANDARD: Regression
    # ─────────────────────────────────────────────
    def run_regression(self, feature_cols, target_col, model_type='random_forest'):
        X, y, cols = self.preprocess(feature_cols, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self._get_model('regression', model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = float(mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        importances = self._get_importances(model, cols)

        return {
            'task': 'regression',
            'model': model_type,
            'target': target_col,
            'features': cols,
            'total_rows': len(self.df),
            'metrics': {
                'mse': round(mse, 4),
                'rmse': round(float(np.sqrt(mse)), 4),
                'r2': round(r2, 4),
                'r2_percent': round(r2 * 100, 1),
            },
            'feature_importance': importances,
            'predictions': [round(float(v), 3) for v in y_pred[:50].tolist()],
            'actuals': [round(float(v), 3) for v in y_test[:50].tolist()],
        }

    # ─────────────────────────────────────────────
    # STANDARD: Classification
    # ─────────────────────────────────────────────
    def run_classification(self, feature_cols, target_col, model_type='random_forest'):
        X, y, cols = self.preprocess(feature_cols, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self._get_model('classification', model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        importances = self._get_importances(model, cols)

        classes = []
        if target_col in self.label_encoders:
            classes = list(self.label_encoders[target_col].classes_)

        return {
            'task': 'classification',
            'model': model_type,
            'target': target_col,
            'features': cols,
            'total_rows': len(self.df),
            'metrics': {
                'accuracy': round(accuracy, 4),
                'accuracy_percent': round(accuracy * 100, 1),
            },
            'feature_importance': importances,
            'classes': classes,
            'report': {str(k): v for k, v in report.items() if k not in ['accuracy']},
        }

    # ─────────────────────────────────────────────
    # STANDARD: Clustering
    # ─────────────────────────────────────────────
    def run_clustering(self, feature_cols, n_clusters=3):
        X, _, cols = self.preprocess(feature_cols)
        X_scaled = self.scaler.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

        cluster_stats = []
        df_temp = pd.DataFrame(X, columns=cols)
        df_temp['cluster'] = labels
        for c in range(n_clusters):
            grp = df_temp[df_temp['cluster'] == c]
            cluster_stats.append({
                'cluster': int(c),
                'size': int(len(grp)),
                'means': {col: round(float(grp[col].mean()), 3) for col in cols}
            })

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        scatter = [{'x': round(float(coords[i, 0]), 3),
                    'y': round(float(coords[i, 1]), 3),
                    'cluster': int(labels[i])} for i in range(min(len(coords), 300))]

        return {
            'task': 'clustering',
            'n_clusters': n_clusters,
            'metrics': {'silhouette_score': round(sil, 4)},
            'cluster_stats': cluster_stats,
            'scatter_data': scatter,
            'feature_cols': cols,
        }

    # ─────────────────────────────────────────────
    # ① EXPLAINABLE AI
    # ─────────────────────────────────────────────
    def run_explainability(self, feature_cols, target_col, model_type='random_forest',
                           sample_index=0, input_values=None):
        X, y, cols = self.preprocess(feature_cols, target_col)
        task = self.auto_detect_task(target_col)
        model = self._get_model(task, model_type)
        model.fit(X, y)

        # Global feature importances
        global_importance = []
        if hasattr(model, 'feature_importances_'):
            raw = model.feature_importances_
            total = raw.sum() or 1.0
            for c, v in sorted(zip(cols, raw), key=lambda x: -x[1]):
                global_importance.append({
                    'feature': c,
                    'importance': round(float(v), 4),
                    'percent': round(float(v / total * 100), 1)
                })
        elif hasattr(model, 'coef_'):
            raw = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            total = raw.sum() or 1.0
            for c, v in sorted(zip(cols, raw), key=lambda x: -x[1]):
                global_importance.append({
                    'feature': c,
                    'importance': round(float(v / total), 4),
                    'percent': round(float(v / total * 100), 1)
                })

        # Local: LIME-style perturbation for one sample
        if input_values:
            sample = []
            for col in cols:
                val = input_values.get(col, float(np.mean(X[:, cols.index(col)])))
                if col in self.label_encoders:
                    try:
                        val = self.label_encoders[col].transform([str(val)])[0]
                    except:
                        val = 0
                sample.append(float(val))
        else:
            idx = min(sample_index, len(X) - 1)
            sample = X[idx].tolist()

        sample_arr = np.array([sample])
        base_pred = float(model.predict(sample_arr)[0])
        baseline = float(model.predict(np.mean(X, axis=0, keepdims=True))[0])

        contributions = []
        for i, col in enumerate(cols):
            perturbed = sample_arr.copy()
            perturbed[0, i] = float(np.mean(X[:, i]))
            perturbed_pred = float(model.predict(perturbed)[0])
            contrib = base_pred - perturbed_pred
            contributions.append({
                'feature': col,
                'value': round(float(sample[i]), 4),
                'contribution': round(contrib, 4),
                'direction': 'increases' if contrib > 0 else 'decreases',
                'magnitude': round(abs(contrib), 4)
            })

        contributions.sort(key=lambda x: -abs(x['contribution']))

        top3 = contributions[:3]
        why_parts = []
        for c in top3:
            orig_val = c['value']
            if c['feature'] in self.label_encoders:
                try:
                    orig_val = self.label_encoders[c['feature']].inverse_transform([int(c['value'])])[0]
                except:
                    pass
            direction_text = 'pushes the prediction up' if c['direction'] == 'increases' else 'pulls the prediction down'
            why_parts.append(f"• {c['feature']} = {orig_val} ({direction_text} by {abs(c['contribution']):.3f})")

        if task == 'classification' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(sample_arr)[0]
            conf = round(float(max(proba)) * 100, 1)
            predicted_class = int(model.predict(sample_arr)[0])
            if target_col in self.label_encoders:
                try:
                    predicted_class = self.label_encoders[target_col].inverse_transform([predicted_class])[0]
                except:
                    pass
            why_narrative = f"Prediction: {predicted_class} (confidence {conf}%)\n\nKey drivers:\n" + "\n".join(why_parts)
        else:
            why_narrative = f"Predicted value: {base_pred:.3f} (baseline: {baseline:.3f})\n\nKey drivers:\n" + "\n".join(why_parts)

        return {
            'task': 'explainability',
            'global_importance': global_importance,
            'local_contributions': contributions,
            'prediction': round(base_pred, 4),
            'baseline': round(baseline, 4),
            'why_narrative': why_narrative,
            'target': target_col,
            'model': model_type,
        }

    # ─────────────────────────────────────────────
    # ② CLINICAL RISK & ALERT ENGINE
    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # ② CLINICAL RISK & ALERT ENGINE
    # ─────────────────────────────────────────────
    def run_risk_engine(self, feature_cols, target_col=None, thresholds=None):
        if self.df is None:
            return {'error': 'No data loaded'}

        numeric_cols = [c for c in feature_cols
                        if self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        if not numeric_cols:
            return {'error': 'No numeric columns selected'}

        df_work = self.df[numeric_cols].dropna().copy()
        total_rows = len(df_work)

        # Auto thresholds (mean ± 2std)
        auto_thresholds = {}
        for col in numeric_cols:
            mean_v = float(df_work[col].mean())
            std_v = float(df_work[col].std()) or 1.0
            auto_thresholds[col] = {
                'mean':          round(mean_v, 3),
                'std':           round(std_v, 3),
                'high':          round(mean_v + 2 * std_v, 3),
                'critical_high': round(mean_v + 3 * std_v, 3),
                'low':           round(mean_v - 2 * std_v, 3),
            }

        # ── Threshold alerts ──
        alerts = []
        for col in numeric_cols:
            thresh = auto_thresholds[col]
            above_critical = int((df_work[col] > thresh['critical_high']).sum())
            above_high     = int((df_work[col] > thresh['high']).sum())
            below_low      = int((df_work[col] < thresh['low']).sum())
            if above_critical > 0:
                alerts.append({
                    'column':  col,
                    'level':   'CRITICAL',
                    'message': f"{above_critical} patients have critically high {col}",
                    'count':   above_critical,
                    'percent': round(above_critical / total_rows * 100, 1),
                })
            elif above_high > 0:
                alerts.append({
                    'column':  col,
                    'level':   'WARNING',
                    'message': f"{above_high} patients have high {col}",
                    'count':   above_high,
                    'percent': round(above_high / total_rows * 100, 1),
                })
            if below_low > 0:
                alerts.append({
                    'column':  col,
                    'level':   'LOW',
                    'message': f"{below_low} patients have low {col}",
                    'count':   below_low,
                    'percent': round(below_low / total_rows * 100, 1),
                })

        # ── Anomaly detection ──
        try:
            iso = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso.fit_predict(df_work[numeric_cols].values)
            n_anomalies = int((anomaly_labels == -1).sum())
            anomaly_df = df_work[anomaly_labels == -1].head(10).copy()
            anomaly_rows = []
            for idx, row in anomaly_df.iterrows():
                row_dict = {'row_index': int(idx)}
                for k, v in row.items():
                    try:
                        row_dict[k] = round(float(v), 3)
                    except (ValueError, TypeError):
                        row_dict[k] = str(v)
                anomaly_rows.append(row_dict)
        except Exception:
            n_anomalies = 0
            anomaly_rows = []

        # ── Change alerts ──
        change_alerts = []
        for col in numeric_cols[:4]:
            series = df_work[col].values
            if len(series) > 10:
                recent  = float(np.mean(series[-10:]))
                overall = float(np.mean(series))
                if overall != 0:
                    change_pct = abs((recent - overall) / overall * 100)
                    if change_pct > 20:
                        change_alerts.append({
                            'column':         col,
                            'change_percent': round(change_pct, 1),
                            'change_ratio':   round(change_pct / 100, 2),
                            'direction':      'increasing' if recent > overall else 'decreasing',
                            'message':        f"Recent avg ({recent:.2f}) is {change_pct:.1f}% {'above' if recent > overall else 'below'} overall avg ({overall:.2f})",
                            'recent_avg':     round(recent, 3),
                            'overall_avg':    round(overall, 3),
                        })

        # ── Risk scores ──
        risk_scores = []
        for i in range(len(df_work)):
            score = 0
            for col in numeric_cols:
                val    = float(df_work[col].iloc[i])
                thresh = auto_thresholds[col]
                mean_v = thresh.get('mean', val)
                std_v  = thresh.get('std', 1) or 1
                z = abs((val - mean_v) / std_v)
                score += min(z, 3)
            normalized = min(round(score / max(len(numeric_cols), 1) / 3 * 100, 1), 100)
            risk_scores.append({'index': i, 'risk_score': normalized})

        risk_scores.sort(key=lambda x: -x['risk_score'])
        high_risk_count = int(sum(1 for r in risk_scores if r['risk_score'] > 60))

        return {
            'task': 'risk_engine',
            'alerts': alerts,
            'anomaly_detection': {
                'n_anomalies':    n_anomalies,
                'anomaly_percent': round(n_anomalies / max(total_rows, 1) * 100, 1),
                'top_anomalies':  anomaly_rows,
            },
            'change_alerts': change_alerts,
            'risk_distribution': {
                'high_risk_count':    high_risk_count,
                'total':              len(risk_scores),
                'high_risk_percent':  round(high_risk_count / max(len(risk_scores), 1) * 100, 1),
            },
            'top_risk_patients': risk_scores[:10],
            'thresholds_used': {k: v for k, v in list(auto_thresholds.items())[:8]},
            'column_stats': {col: {
                'mean':           auto_thresholds[col]['mean'],
                'std':            auto_thresholds[col]['std'],
                'high_threshold': auto_thresholds[col]['high'],
            } for col in numeric_cols},
        }
    # ─────────────────────────────────────────────
    # ③ TIME-SERIES / TREND ANALYSIS
    # ─────────────────────────────────────────────
    def run_trend_analysis(self, feature_cols, time_col=None, target_col=None, forecast_steps=10):
        if self.df is None:
            return {'error': 'No data loaded'}

        numeric_cols = [c for c in feature_cols
                        if self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        if not numeric_cols:
            return {'error': 'No numeric columns selected for trend analysis'}

        df_work = self.df[numeric_cols].dropna().reset_index(drop=True)

        trends = {}
        forecasts = {}
        for col in numeric_cols[:6]:
            series = df_work[col].values.astype(float)
            n = len(series)
            x = np.arange(n)

            coeffs = np.polyfit(x, series, 1)
            slope = float(coeffs[0])
            trend_line = (coeffs[0] * x + coeffs[1]).tolist()

            window = max(3, n // 10)
            ma = pd.Series(series).rolling(window, center=True).mean().bfill().ffill().tolist()

            x_future = np.arange(n, n + forecast_steps)
            forecast_vals = (coeffs[0] * x_future + coeffs[1]).tolist()

            r2 = float(r2_score(series, trend_line))
            if slope > 0:
                trend_dir = 'increasing'
                trend_color = '#ff4d6d'
            elif slope < 0:
                trend_dir = 'decreasing'
                trend_color = '#00e5a0'
            else:
                trend_dir = 'stable'
                trend_color = '#ffd166'

            if abs(float(series[0])) > 0.001:
                change_pct = round((float(series[-1]) - float(series[0])) / abs(float(series[0])) * 100, 2)
            else:
                change_pct = 0.0

            trends[col] = {
                'values': [round(float(v), 3) for v in series[:200].tolist()],
                'trend_line': [round(float(v), 3) for v in trend_line[:200]],
                'moving_average': [round(float(v), 3) for v in ma[:200]],
                'slope': round(slope, 5),
                'r2': round(r2, 4),
                'direction': trend_dir,
                'color': trend_color,
                'change_pct': change_pct,
                'current': round(float(series[-1]), 3),
                'start': round(float(series[0]), 3),
            }
            forecasts[col] = {
                'steps': forecast_steps,
                'values': [round(float(v), 3) for v in forecast_vals],
            }

        return {
            'task': 'trend_analysis',
            'columns': numeric_cols[:6],
            'trends': trends,
            'forecasts': forecasts,
            'total_rows': len(df_work),
        }

    # ─────────────────────────────────────────────
    # ④ PATIENT SIMILARITY
    # ─────────────────────────────────────────────
    def find_similar_patients(self, feature_cols, query_values, n_similar=5):
        if self.df is None:
            return {'error': 'No data loaded'}

        numeric_cols = [c for c in feature_cols
                        if self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        if not numeric_cols:
            return {'error': 'No numeric columns for similarity'}

        df_work = self.df[numeric_cols].dropna().reset_index(drop=True)
        X = self.scaler.fit_transform(df_work.values)

        query_row = np.array([[float(query_values.get(col, df_work[col].mean()))
                               for col in numeric_cols]])
        q_arr = self.scaler.transform(query_row)

        nn = NearestNeighbors(n_neighbors=min(n_similar + 1, len(X)))
        nn.fit(X)
        distances, indices = nn.kneighbors(q_arr)

        similar_patients = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            patient = {'rank': rank + 1, 'index': int(idx), 'distance': round(float(dist), 4)}
            for col in numeric_cols[:8]:
                patient[col] = round(float(df_work[col].iloc[idx]), 3)
            patient['similarity_pct'] = round(max(0, 100 - float(dist) * 15), 1)
            similar_patients.append(patient)

        query_patient = {col: round(float(query_values.get(col, df_work[col].mean())), 3)
                         for col in numeric_cols[:8]}

        return {
            'task': 'patient_similarity',
            'query_patient': query_patient,
            'similar_patients': similar_patients[:n_similar],
            'n_similar': len(similar_patients[:n_similar]),
            'feature_cols': numeric_cols[:8],
            'total_patients': len(X),
        }

    # ─────────────────────────────────────────────
    # PREDICT NEW INPUT
    # ─────────────────────────────────────────────
    def predict_new_input(self, feature_cols, target_col, input_values, model_type='random_forest'):
        X, y, cols = self.preprocess(feature_cols, target_col)
        task = self.auto_detect_task(target_col)
        model = self._get_model(task, model_type)
        model.fit(X, y)

        input_encoded = []
        for col in feature_cols:
            val = input_values.get(col, 0)
            if col in self.label_encoders:
                try:
                    val = self.label_encoders[col].transform([str(val)])[0]
                except:
                    val = 0
            input_encoded.append(float(val))

        prediction = model.predict([input_encoded])[0]
        result = {'prediction': float(prediction), 'task': task, 'model': model_type}

        if task == 'classification' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba([input_encoded])[0]
            result['confidence'] = round(float(max(proba)) * 100, 1)
            if target_col in self.label_encoders:
                try:
                    result['prediction_label'] = str(
                        self.label_encoders[target_col].inverse_transform([int(prediction)])[0]
                    )
                except:
                    result['prediction_label'] = str(prediction)

        return result


# ─────────────────────────────────────────────
# Groq LLM Client
# ─────────────────────────────────────────────
class GroqLLMClient:
    MODELS = [
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'mixtral-8x7b-32768',
    ]

    def __init__(self, api_key, model=None):
        self.api_key = api_key
        self.model = model if model else self.MODELS[0]
        self.base_url = 'https://api.groq.com/openai/v1/chat/completions'

    def _trim_context(self, text, max_chars=3000):
        if len(text) > max_chars:
            return text[:max_chars] + '\n...[trimmed]'
        return text

    def ask(self, user_message, dataset_context=None, ml_result=None):
        if not self.api_key or self.api_key in ('', 'your-groq-api-key-here'):
            return {
                'success': False,
                'message': '⚠️ Groq API key not set. Open healthcare_ai/settings.py and set GROQ_API_KEY.',
                'tokens_used': 0
            }

        system_prompt = (
            "You are a knowledgeable Healthcare AI Assistant helping medical professionals "
            "and researchers understand health data patterns, predictions, and insights. "
            "Speak clearly and compassionately. Always note that AI predictions support "
            "but do not replace professional medical judgment. Be concise and helpful."
        )

        context_parts = []
        if dataset_context:
            context_parts.append(f"Dataset context:\n{self._trim_context(str(dataset_context))}")
        if ml_result:
            context_parts.append(f"ML result:\n{self._trim_context(json.dumps(ml_result))}")

        full_message = user_message
        if context_parts:
            full_message = "\n\n".join(context_parts) + "\n\nUser question: " + user_message

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        for model in self.MODELS:
            payload = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_message},
                ],
                'max_tokens': 1024,
                'temperature': 0.7,
            }
            try:
                r = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                if r.ok:
                    return {
                        'success': True,
                        'message': r.json()['choices'][0]['message']['content'],
                        'model_used': model,
                        'tokens_used': r.json().get('usage', {}).get('total_tokens', 0),
                    }
            except Exception:
                continue

        return {'success': False, 'message': 'All LLM models failed.', 'tokens_used': 0}