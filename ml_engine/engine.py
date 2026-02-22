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
        if self.df[target_col].dtype in [np.float64, np.float32]:
            return 'regression'
        unique_ratio = self.df[target_col].nunique() / len(self.df)
        if unique_ratio < 0.05 or self.df[target_col].dtype == object:
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

        importances = []
        if hasattr(model, 'feature_importances_'):
            raw = model.feature_importances_
            importances = [{'feature': c, 'importance': round(float(v), 4)}
                           for c, v in sorted(zip(cols, raw), key=lambda x: -x[1])]

        return {
            'task': 'regression',
            'model': model_type,
            'target': target_col,
            'metrics': {'mse': round(mse, 4), 'rmse': round(float(np.sqrt(mse)), 4), 'r2': round(r2, 4)},
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

        importances = []
        if hasattr(model, 'feature_importances_'):
            raw = model.feature_importances_
            importances = [{'feature': c, 'importance': round(float(v), 4)}
                           for c, v in sorted(zip(cols, raw), key=lambda x: -x[1])]

        classes = []
        if target_col in self.label_encoders:
            classes = list(self.label_encoders[target_col].classes_)

        return {
            'task': 'classification',
            'model': model_type,
            'target': target_col,
            'metrics': {'accuracy': round(accuracy, 4)},
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

        # PCA 2D for visualization
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
        """
        Returns: feature importance, permutation importance (SHAP-style),
                 per-feature contribution for a specific prediction,
                 'why this prediction' narrative.
        """
        X, y, cols = self.preprocess(feature_cols, target_col)
        task = self.auto_detect_task(target_col)
        model = self._get_model(task, model_type)
        model.fit(X, y)

        # --- Global: feature importances ---
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

        # --- Local: LIME-style perturbation for one sample ---
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

        # Compute contribution via perturbation
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

        # Build "Why this prediction" narrative
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
    def run_risk_engine(self, feature_cols, target_col=None, thresholds=None):
        """
        Threshold alerts, sudden risk change detection, anomaly detection.
        thresholds: dict of {col: {'low': val, 'high': val, 'critical_high': val}}
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        numeric_cols = [c for c in feature_cols if self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        if not numeric_cols:
            numeric_cols = feature_cols[:min(len(feature_cols), 8)]

        df_work = self.df[numeric_cols].dropna()

        # Auto-suggest thresholds if not provided
        auto_thresholds = {}
        for col in numeric_cols:
            q25 = float(df_work[col].quantile(0.05))
            q75 = float(df_work[col].quantile(0.95))
            mean_v = float(df_work[col].mean())
            std_v = float(df_work[col].std())
            auto_thresholds[col] = {
                'low': round(mean_v - 2 * std_v, 2),
                'high': round(mean_v + 2 * std_v, 2),
                'critical_high': round(mean_v + 3 * std_v, 2),
                'mean': round(mean_v, 2),
                'std': round(std_v, 2),
            }

        used_thresholds = thresholds or auto_thresholds

        # --- Threshold alerts ---
        alerts = []
        for col in numeric_cols:
            thresh = used_thresholds.get(col, auto_thresholds.get(col, {}))
            col_vals = df_work[col]
            n_total = len(col_vals)

            if thresh:
                if 'critical_high' in thresh:
                    n_critical = int((col_vals > thresh['critical_high']).sum())
                    if n_critical > 0:
                        alerts.append({
                            'column': col,
                            'level': 'CRITICAL',
                            'color': '#ff4d6d',
                            'count': n_critical,
                            'percent': round(n_critical / n_total * 100, 1),
                            'threshold': thresh['critical_high'],
                            'message': f"{n_critical} patients ({round(n_critical/n_total*100,1)}%) above critical threshold {thresh['critical_high']}"
                        })
                if 'high' in thresh:
                    n_high = int(((col_vals > thresh['high']) & (col_vals <= thresh.get('critical_high', float('inf')))).sum())
                    if n_high > 0:
                        alerts.append({
                            'column': col,
                            'level': 'WARNING',
                            'color': '#ffd166',
                            'count': n_high,
                            'percent': round(n_high / n_total * 100, 1),
                            'threshold': thresh['high'],
                            'message': f"{n_high} patients ({round(n_high/n_total*100,1)}%) above warning threshold {thresh['high']}"
                        })
                if 'low' in thresh:
                    n_low = int((col_vals < thresh['low']).sum())
                    if n_low > 0:
                        alerts.append({
                            'column': col,
                            'level': 'LOW',
                            'color': '#00b4ff',
                            'count': n_low,
                            'percent': round(n_low / n_total * 100, 1),
                            'threshold': thresh['low'],
                            'message': f"{n_low} patients ({round(n_low/n_total*100,1)}%) below low threshold {thresh['low']}"
                        })

        # --- Anomaly detection (IsolationForest) ---
        X_scaled = StandardScaler().fit_transform(df_work[numeric_cols])
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels = iso.fit_predict(X_scaled)
        anomaly_scores = iso.score_samples(X_scaled)

        n_anomalies = int((anomaly_labels == -1).sum())
        anomaly_indices = np.where(anomaly_labels == -1)[0].tolist()[:20]

        anomaly_rows = []
        for idx in anomaly_indices[:10]:
            row = {c: round(float(df_work[c].iloc[idx]), 3) for c in numeric_cols[:6]}
            row['anomaly_score'] = round(float(anomaly_scores[idx]), 4)
            row['row_index'] = int(idx)
            anomaly_rows.append(row)

        # --- Sudden change detection (variance over rolling windows) ---
        change_alerts = []
        for col in numeric_cols[:6]:
            col_vals = df_work[col].values
            if len(col_vals) > 20:
                window = max(10, len(col_vals) // 10)
                rolling_std = pd.Series(col_vals).rolling(window).std().dropna()
                overall_std = float(col_vals.std())
                if overall_std > 0:
                    max_local_std = float(rolling_std.max())
                    change_ratio = max_local_std / overall_std
                    if change_ratio > 1.8:
                        peak_window = int(rolling_std.idxmax())
                        change_alerts.append({
                            'column': col,
                            'change_ratio': round(change_ratio, 2),
                            'peak_window_index': peak_window,
                            'message': f"{col}: sudden volatility spike detected (×{change_ratio:.1f} normal variance)"
                        })

        # Risk score per row (composite)
        risk_scores = []
        for i in range(min(len(df_work), 200)):
            score = 0
            for col in numeric_cols:
                thresh = auto_thresholds.get(col, {})
                val = float(df_work[col].iloc[i])
                mean_v = thresh.get('mean', val)
                std_v = thresh.get('std', 1) or 1
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
                'n_anomalies': n_anomalies,
                'anomaly_percent': round(n_anomalies / len(df_work) * 100, 1),
                'top_anomalies': anomaly_rows,
            },
            'change_alerts': change_alerts,
            'risk_distribution': {
                'high_risk_count': high_risk_count,
                'total': len(risk_scores),
                'high_risk_percent': round(high_risk_count / max(len(risk_scores), 1) * 100, 1),
            },
            'top_risk_patients': risk_scores[:10],
            'thresholds_used': {k: v for k, v in list(auto_thresholds.items())[:8]},
            'column_stats': {col: {
                'mean': auto_thresholds[col]['mean'],
                'std': auto_thresholds[col]['std'],
                'high_threshold': auto_thresholds[col]['high'],
                'critical_threshold': auto_thresholds[col]['critical_high'],
            } for col in numeric_cols[:8]},
        }

    # ─────────────────────────────────────────────
    # ③ TIME-SERIES / TREND ANALYSIS
    # ─────────────────────────────────────────────
    def run_trend_analysis(self, feature_cols, time_col=None, target_col=None, forecast_steps=10):
        """
        Trend analysis, progression tracking, simple forecasting.
        If time_col provided, uses it as index. Otherwise uses row order.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        numeric_cols = [c for c in feature_cols
                        if self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        if not numeric_cols:
            return {'error': 'No numeric columns selected for trend analysis'}

        df_work = self.df[numeric_cols + ([time_col] if time_col and time_col in self.df.columns else [])].copy()
        df_work = df_work.dropna().reset_index(drop=True)

        trends = {}
        forecasts = {}
        for col in numeric_cols[:6]:
            series = df_work[col].values.astype(float)
            n = len(series)
            x = np.arange(n)

            # Linear trend
            coeffs = np.polyfit(x, series, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])
            trend_line = (coeffs[0] * x + coeffs[1]).tolist()

            # Moving average
            window = max(3, n // 10)
            ma = pd.Series(series).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill').tolist()

            # Simple linear forecast
            x_future = np.arange(n, n + forecast_steps)
            forecast_vals = (coeffs[0] * x_future + coeffs[1]).tolist()

            # Trend direction & strength
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

            # Change rate
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
                'start_index': n,
                'confidence_upper': [round(float(v) + float(series.std()), 3) for v in forecast_vals],
                'confidence_lower': [round(float(v) - float(series.std()), 3) for v in forecast_vals],
            }

        # Correlation matrix for selected columns
        corr_data = []
        if len(numeric_cols) > 1:
            corr_matrix = df_work[numeric_cols[:8]].corr()
            for r in corr_matrix.index:
                for c in corr_matrix.columns:
                    corr_data.append({'row': r, 'col': c, 'value': round(float(corr_matrix.loc[r, c]), 3)})

        return {
            'task': 'trend_analysis',
            'n_rows': len(df_work),
            'trends': trends,
            'forecasts': forecasts,
            'correlation_matrix': corr_data,
            'columns': numeric_cols[:6],
        }

    # ─────────────────────────────────────────────
    # ④ PATIENT SIMILARITY ENGINE
    # ─────────────────────────────────────────────
    def run_patient_similarity(self, feature_cols, query_index=0,
                                query_values=None, n_similar=5):
        """
        Find similar patients using KNN.
        query_values: dict of {col: val} for a new patient (optional).
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        numeric_cols = [c for c in feature_cols
                        if self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        if not numeric_cols:
            # Try to encode categorical
            numeric_cols = feature_cols[:min(len(feature_cols), 6)]

        df_work = self.df[numeric_cols].copy()
        for col in numeric_cols:
            if df_work[col].dtype == object:
                le = LabelEncoder()
                df_work[col] = le.fit_transform(df_work[col].astype(str))
                self.label_encoders[col] = le

        df_work = df_work.dropna().reset_index(drop=True)
        X = df_work.values.astype(float)
        X_scaled = StandardScaler().fit_transform(X)

        # Determine query vector
        if query_values:
            q = []
            for col in numeric_cols:
                val = query_values.get(col, float(np.mean(X[:, numeric_cols.index(col)])))
                if col in self.label_encoders:
                    try:
                        val = float(self.label_encoders[col].transform([str(val)])[0])
                    except:
                        val = 0.0
                q.append(float(val))
            q_arr = np.array([q])
            scaler2 = StandardScaler()
            scaler2.fit(X)
            q_scaled = scaler2.transform(q_arr)
        else:
            idx = min(query_index, len(X_scaled) - 1)
            q_scaled = X_scaled[idx:idx+1]
            q_arr = X[idx:idx+1]

        # KNN
        k = min(n_similar + 1, len(X_scaled))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(q_scaled)

        similar_patients = []
        indices_list = indices[0].tolist()
        distances_list = distances[0].tolist()

        # Remove query itself if present
        pairs = [(i, d) for i, d in zip(indices_list, distances_list) if i != query_index][:n_similar]

        for rank, (idx, dist) in enumerate(pairs):
            patient = {'rank': rank + 1, 'index': int(idx), 'distance': round(float(dist), 4)}
            for col in numeric_cols[:8]:
                patient[col] = round(float(X[idx, numeric_cols.index(col)]), 3)
            patient['similarity_pct'] = round(max(0, 100 - float(dist) * 15), 1)
            similar_patients.append(patient)

        # Query patient info
        query_patient = {}
        for col in numeric_cols[:8]:
            c_idx = numeric_cols.index(col)
            query_patient[col] = round(float(q_arr[0, c_idx]), 3)

        # Cohort stats: compare similar group vs full population
        sim_indices = [p['index'] for p in similar_patients]
        cohort_comparison = []
        full_df = pd.DataFrame(X, columns=numeric_cols)
        cohort_df = full_df.iloc[sim_indices]

        for col in numeric_cols[:6]:
            cohort_comparison.append({
                'feature': col,
                'cohort_mean': round(float(cohort_df[col].mean()), 3),
                'population_mean': round(float(full_df[col].mean()), 3),
                'query_value': round(float(query_patient.get(col, 0)), 3),
                'cohort_std': round(float(cohort_df[col].std()), 3),
            })

        return {
            'task': 'patient_similarity',
            'query_patient': query_patient,
            'similar_patients': similar_patients,
            'cohort_comparison': cohort_comparison,
            'n_similar': len(similar_patients),
            'feature_cols': numeric_cols[:8],
            'total_patients': len(X),
        }

    # ─────────────────────────────────────────────
    # PREDICT NEW INPUT
    # ─────────────────────────────────────────────
    def predict_new_input(self, feature_cols, target_col, input_values):
        X, y, cols = self.preprocess(feature_cols, target_col)
        task = self.auto_detect_task(target_col)
        if task == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
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
        result = {'prediction': float(prediction), 'task': task}
        if task == 'classification' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba([input_encoded])[0]
            result['confidence'] = round(float(max(proba)) * 100, 1)
            if target_col in self.label_encoders:
                result['prediction_label'] = str(self.label_encoders[target_col].inverse_transform([int(prediction)])[0])
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
            ctx = self._trim_context(json.dumps(dataset_context, indent=2))
            context_parts.append(f"Dataset context:\n{ctx}")
        if ml_result:
            ctx = self._trim_context(json.dumps(ml_result, indent=2))
            context_parts.append(f"ML result:\n{ctx}")

        messages = []
        if context_parts:
            messages.append({'role': 'system', 'content': system_prompt + '\n\nContext:\n' + '\n\n'.join(context_parts)})
        else:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_message})

        for model in self.MODELS:
            try:
                resp = requests.post(
                    self.base_url,
                    headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                    json={'model': model, 'messages': messages, 'max_tokens': 600, 'temperature': 0.7},
                    timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        'success': True,
                        'message': data['choices'][0]['message']['content'],
                        'tokens_used': data.get('usage', {}).get('total_tokens', 0),
                        'model': model
                    }
                elif resp.status_code == 429:
                    continue
            except Exception:
                continue

        return {'success': False, 'message': 'LLM service unavailable. Please try again.', 'tokens_used': 0}
