"""
Healthcare AI ML Engine
Handles: Regression, Classification, Clustering, LLM Chat
Dynamic column selection based on user input
"""

import pandas as pd
import numpy as np
import json
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, silhouette_score
)
import warnings
warnings.filterwarnings('ignore')


class HealthcareMLEngine:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_dataframe(self, df: pd.DataFrame):
        """Load a pandas DataFrame into the engine"""
        self.df = df.copy()
        return self

    def get_column_info(self):
        """Returns metadata about each column for the UI"""
        if self.df is None:
            return []
        info = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            null_count = self.df[col].isnull().sum()
            col_type = 'numeric' if self.df[col].dtype in [np.float64, np.int64, np.float32, np.int32] else 'categorical'
            info.append({
                'name': col,
                'dtype': dtype,
                'col_type': col_type,
                'unique_values': int(unique_count),
                'null_count': int(null_count),
                'sample_values': self.df[col].dropna().head(3).tolist()
            })
        return info

    def auto_detect_task(self, target_col):
        """Auto-detect if task is regression or classification"""
        if self.df[target_col].dtype in [np.float64, np.float32]:
            return 'regression'
        unique_ratio = self.df[target_col].nunique() / len(self.df)
        if unique_ratio < 0.05 or self.df[target_col].dtype == object:
            return 'classification'
        return 'regression'

    def preprocess(self, feature_cols, target_col=None):
        """Preprocess selected columns - encode categoricals, scale numerics"""
        df_work = self.df[feature_cols + ([target_col] if target_col else [])].copy()

        # Drop rows with nulls in selected columns
        df_work = df_work.dropna()

        X = df_work[feature_cols].copy()
        y = df_work[target_col].copy() if target_col else None

        # Encode categorical feature columns
        for col in feature_cols:
            if X[col].dtype == object or str(X[col].dtype) == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        # Encode target if classification
        if y is not None and (y.dtype == object or str(y.dtype) == 'category'):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.label_encoders[target_col] = le

        return X.values, y.values if y is not None else None

    def run_regression(self, feature_cols, target_col, model_type='random_forest'):
        """Run regression prediction"""
        X, y = self.preprocess(feature_cols, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        model = models.get(model_type, models['random_forest'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            for col, imp in zip(feature_cols, model.feature_importances_):
                feature_importance.append({'feature': col, 'importance': round(float(imp), 4)})
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        # Sample predictions
        sample_preds = []
        for i in range(min(10, len(X_test))):
            sample_preds.append({
                'actual': round(float(y_test[i]), 3),
                'predicted': round(float(y_pred[i]), 3)
            })

        return {
            'task': 'regression',
            'model': model_type,
            'target': target_col,
            'features': feature_cols,
            'metrics': {
                'mse': round(float(mse), 4),
                'rmse': round(float(np.sqrt(mse)), 4),
                'r2_score': round(float(r2), 4),
                'r2_percent': round(float(r2) * 100, 2)
            },
            'feature_importance': feature_importance,
            'sample_predictions': sample_preds,
            'total_rows': len(X),
            'train_rows': len(X_train),
            'test_rows': len(X_test)
        }

    def run_classification(self, feature_cols, target_col, model_type='random_forest'):
        """Run classification prediction"""
        X, y = self.preprocess(feature_cols, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'logistic': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        model = models.get(model_type, models['random_forest'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        class_dist = [{'class': str(int(u)), 'count': int(c)} for u, c in zip(unique, counts)]

        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            for col, imp in zip(feature_cols, model.feature_importances_):
                feature_importance.append({'feature': col, 'importance': round(float(imp), 4)})
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        # Prediction probabilities sample
        sample_preds = []
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            for i in range(min(10, len(X_test))):
                sample_preds.append({
                    'actual': int(y_test[i]),
                    'predicted': int(y_pred[i]),
                    'confidence': round(float(max(proba[i])) * 100, 1)
                })

        return {
            'task': 'classification',
            'model': model_type,
            'target': target_col,
            'features': feature_cols,
            'metrics': {
                'accuracy': round(float(accuracy), 4),
                'accuracy_percent': round(float(accuracy) * 100, 2)
            },
            'class_distribution': class_dist,
            'feature_importance': feature_importance,
            'sample_predictions': sample_preds,
            'total_rows': len(X),
        }

    def run_clustering(self, feature_cols, n_clusters=3):
        """Run K-Means clustering"""
        X, _ = self.preprocess(feature_cols)
        X_scaled = self.scaler.fit_transform(X)

        n_clusters = min(n_clusters, len(X) - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        silhouette = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 else 0

        # Cluster stats
        df_temp = self.df[feature_cols].dropna().copy()
        df_temp['cluster'] = labels
        cluster_stats = []
        for c in range(n_clusters):
            cluster_df = df_temp[df_temp['cluster'] == c]
            stats = {'cluster': c, 'size': len(cluster_df)}
            for col in feature_cols:
                if df_temp[col].dtype in [np.float64, np.int64]:
                    stats[f'{col}_mean'] = round(float(cluster_df[col].mean()), 3)
            cluster_stats.append(stats)

        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        cluster_dist = [{'cluster': f'Cluster {int(u)}', 'count': int(c)} for u, c in zip(unique, counts)]

        return {
            'task': 'clustering',
            'n_clusters': n_clusters,
            'features': feature_cols,
            'metrics': {
                'silhouette_score': round(silhouette, 4),
                'inertia': round(float(kmeans.inertia_), 2)
            },
            'cluster_distribution': cluster_dist,
            'cluster_stats': cluster_stats,
            'total_rows': len(X),
        }

    def predict_new_input(self, feature_cols, target_col, input_values, model_type='random_forest'):
        """Predict for a single new input row"""
        task = self.auto_detect_task(target_col)
        X, y = self.preprocess(feature_cols, target_col)

        if task == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X, y)

        # Encode input values
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

        result = {'prediction': float(prediction), 'task': task}
        if task == 'classification' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba([input_encoded])[0]
            result['confidence'] = round(float(max(proba)) * 100, 1)
            if target_col in self.label_encoders:
                result['prediction_label'] = self.label_encoders[target_col].inverse_transform([int(prediction)])[0]

        return result


class GroqLLMClient:
    """Free Groq LLM API Client for healthcare natural language queries"""

    # Up-to-date Groq model list (as of 2026)
    MODELS = [
        'llama-3.3-70b-versatile',   # Best quality, free
        'llama-3.1-8b-instant',       # Fast, free fallback
        'mixtral-8x7b-32768',         # Good alternative
    ]

    def __init__(self, api_key, model=None):
        self.api_key = api_key
        # Use first available model by default
        self.model = model if model else self.MODELS[0]
        self.base_url = 'https://api.groq.com/openai/v1/chat/completions'

    def _trim_context(self, text, max_chars=3000):
        """Trim context to avoid token limit errors"""
        if len(text) > max_chars:
            return text[:max_chars] + '\n...[trimmed for length]'
        return text

    def ask(self, user_message, dataset_context=None, ml_result=None):
        """Send a message to Groq LLM with optional dataset and ML context"""

        if not self.api_key or self.api_key == 'your-groq-api-key-here':
            return {
                'success': False,
                'message': '⚠️ Groq API key not set. Please open healthcare_ai/settings.py and replace "your-groq-api-key-here" with your real key from https://console.groq.com (free signup).',
                'tokens_used': 0
            }

        system_prompt = (
            "You are a knowledgeable Healthcare AI Assistant helping medical professionals "
            "and researchers understand health data patterns, predictions, and insights. "
            "Speak clearly and compassionately. Always note that AI predictions support "
            "but do not replace professional medical judgment. Be concise and helpful."
        )

        # Build user content — trim large context to avoid 400 errors
        context_parts = []
        if dataset_context:
            ctx_str = json.dumps(dataset_context, indent=2)
            context_parts.append(f"Dataset Info:\n{self._trim_context(ctx_str, 1500)}")
        if ml_result:
            # Only send key metrics, not full result
            slim = {
                'task': ml_result.get('task'),
                'target': ml_result.get('target'),
                'features': ml_result.get('features', [])[:10],
                'metrics': ml_result.get('metrics', {}),
                'feature_importance': ml_result.get('feature_importance', [])[:5],
            }
            context_parts.append(f"ML Result:\n{json.dumps(slim, indent=2)}")

        if context_parts:
            full_message = "\n\n".join(context_parts) + f"\n\nQuestion: {user_message}"
        else:
            full_message = user_message

        # Trim overall message
        full_message = self._trim_context(full_message, 6000)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Try models in order until one works
        last_error = None
        for model in self.MODELS:
            payload = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': full_message}
                ],
                'max_tokens': 1024,
                'temperature': 0.7
            }
            try:
                response = requests.post(
                    self.base_url, headers=headers,
                    json=payload, timeout=30
                )

                # Capture the real Groq error message before raising
                if not response.ok:
                    try:
                        err_body = response.json()
                        err_msg = err_body.get('error', {}).get('message', response.text)
                    except Exception:
                        err_msg = response.text
                    last_error = f'Groq API error ({response.status_code}): {err_msg}'
                    continue  # try next model

                data = response.json()
                return {
                    'success': True,
                    'message': data['choices'][0]['message']['content'],
                    'tokens_used': data.get('usage', {}).get('total_tokens', 0),
                    'model_used': model
                }

            except requests.exceptions.ConnectionError:
                return {
                    'success': False,
                    'message': '⚠️ Cannot reach Groq servers. Please check your internet connection.',
                    'tokens_used': 0
                }
            except requests.exceptions.Timeout:
                last_error = f'Request timed out for model {model}'
                continue
            except Exception as e:
                last_error = str(e)
                continue

        # All models failed
        return {
            'success': False,
            'message': f'⚠️ All Groq models failed. Last error: {last_error}\n\nPlease verify your API key at https://console.groq.com',
            'tokens_used': 0
        }
