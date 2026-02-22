# ml_engine/engine_features2.py
"""
6 new high-impact healthcare features:
  1. PatientDashboardEngine    — unified patient-level profile view
  2. MultiTargetComparator     — simultaneous multi-target prediction
  3. SurvivalAnalysisEngine    — Kaplan-Meier survival curves
  4. AlertRulesEngine          — custom clinical threshold rules builder
  5. DatasetComparator         — compare 2 datasets (before/after, A/B)
  6. ClinicalCodingAssistant   — ICD-10 mapping + drug interaction flags via LLM
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# 1. PATIENT DASHBOARD ENGINE
#    Select any patient row → unified card with all values,
#    risk score, anomaly flag, predicted outcomes, similar patients
# ═══════════════════════════════════════════════════════════════════

class PatientDashboardEngine:

    def get_patient_profile(self, df: pd.DataFrame, row_index: int,
                             feature_cols: list, target_col: str = None,
                             ml_engine=None) -> dict:
        if row_index >= len(df) or row_index < 0:
            return {'error': f'Row {row_index} not found. Dataset has {len(df)} rows.'}

        row = df.iloc[row_index]
        total = len(df)

        # ── All column values with percentile rank ──
        profile_values = []
        for col in df.columns:
            val = row[col]
            entry = {
                'column': col,
                'value':  str(val) if pd.isna(val) else val,
                'is_missing': bool(pd.isna(val)),
                'is_feature': col in feature_cols,
                'is_target':  col == target_col,
            }
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.isna(val):
                col_vals = df[col].dropna()
                pct = float((col_vals < float(val)).sum() / len(col_vals) * 100)
                mean_v = float(col_vals.mean())
                std_v  = float(col_vals.std()) or 1
                z      = (float(val) - mean_v) / std_v
                entry.update({
                    'percentile':   round(pct, 1),
                    'mean':         round(mean_v, 3),
                    'z_score':      round(z, 2),
                    'is_outlier':   abs(z) > 2.5,
                    'direction':    'high' if z > 0.5 else 'low' if z < -0.5 else 'normal',
                })
            profile_values.append(entry)

        # ── Composite risk score ──
        num_cols = [c for c in feature_cols
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        risk_score = 0
        for col in num_cols:
            if pd.isna(row[col]):
                continue
            col_vals = df[col].dropna()
            if len(col_vals) < 2:
                continue
            z = abs((float(row[col]) - float(col_vals.mean())) / (float(col_vals.std()) or 1))
            risk_score += min(z, 3)
        risk_score = round(min(risk_score / max(len(num_cols), 1) / 3 * 100, 100), 1)
        risk_level = 'HIGH' if risk_score > 65 else 'MEDIUM' if risk_score > 35 else 'LOW'

        # ── Anomaly detection for this patient ──
        anomaly_flag = False
        anomaly_reason = ''
        outlier_cols = [e['column'] for e in profile_values
                        if e.get('is_outlier') and e.get('is_feature')]
        if outlier_cols:
            anomaly_flag = True
            anomaly_reason = f"Outlier values in: {', '.join(outlier_cols[:4])}"

        # ── Prediction (if ML engine and target available) ──
        prediction_result = None
        if ml_engine and target_col and feature_cols:
            try:
                input_vals = {col: row[col] for col in feature_cols if col in df.columns}
                prediction_result = ml_engine.predict_new_input(feature_cols, target_col, input_vals)
            except Exception:
                pass

        # ── Similar patients (fast KNN) ──
        similar = []
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import LabelEncoder

            df_num = df[num_cols].copy()
            for col in num_cols:
                df_num[col] = pd.to_numeric(df_num[col], errors='coerce')
            df_num = df_num.fillna(df_num.mean())

            scaler = StandardScaler()
            X = scaler.fit_transform(df_num.values)
            q = X[row_index:row_index+1]

            knn = NearestNeighbors(n_neighbors=min(6, len(X)), metric='euclidean')
            knn.fit(X)
            dists, idxs = knn.kneighbors(q)
            for i, (idx, dist) in enumerate(zip(idxs[0], dists[0])):
                if idx == row_index:
                    continue
                similar.append({
                    'row_index': int(idx),
                    'distance':  round(float(dist), 3),
                    'similarity_pct': round(max(0, 100 - float(dist) * 15), 1),
                    'values': {col: round(float(df[col].iloc[idx]), 3)
                               for col in num_cols[:5]
                               if not pd.isna(df[col].iloc[idx])},
                })
                if len(similar) >= 4:
                    break
        except Exception:
            pass

        # ── Population comparison ──
        population_stats = {}
        for col in num_cols[:8]:
            col_vals = df[col].dropna()
            val = row[col]
            if pd.isna(val):
                continue
            population_stats[col] = {
                'patient_value': round(float(val), 3),
                'population_mean': round(float(col_vals.mean()), 3),
                'population_median': round(float(col_vals.median()), 3),
                'population_std': round(float(col_vals.std()), 3),
                'percentile': round(float((col_vals < float(val)).sum() / len(col_vals) * 100), 1),
            }

        return {
            'task': 'patient_dashboard',
            'row_index': row_index,
            'total_patients': total,
            'profile_values': profile_values,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'anomaly_flag': anomaly_flag,
            'anomaly_reason': anomaly_reason,
            'outlier_columns': outlier_cols,
            'prediction': prediction_result,
            'similar_patients': similar,
            'population_stats': population_stats,
            'missing_count': sum(1 for e in profile_values if e['is_missing']),
        }


# ═══════════════════════════════════════════════════════════════════
# 2. MULTI-TARGET COMPARATOR
#    Run the same features against multiple target columns at once.
#    Returns side-by-side model performance + feature importance.
# ═══════════════════════════════════════════════════════════════════

class MultiTargetComparator:

    def compare_targets(self, df: pd.DataFrame, feature_cols: list,
                        target_cols: list, model_type: str = 'random_forest') -> dict:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
        from sklearn.preprocessing import LabelEncoder

        results = []

        for target in target_cols:
            if target not in df.columns:
                continue
            try:
                df_work = df[feature_cols + [target]].dropna()
                if len(df_work) < 20:
                    results.append({'target': target, 'error': 'Too few rows after dropping nulls'})
                    continue

                le_map = {}
                X_df = df_work[feature_cols].copy()
                for col in feature_cols:
                    if X_df[col].dtype == object:
                        le = LabelEncoder()
                        X_df[col] = le.fit_transform(X_df[col].astype(str))
                        le_map[col] = le

                X = X_df.values
                y_raw = df_work[target]

                is_numeric = pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10
                task = 'regression' if is_numeric else 'classification'

                if not is_numeric:
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y_raw.astype(str))
                else:
                    y = y_raw.values.astype(float)

                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

                if task == 'regression':
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    r2   = round(float(r2_score(y_te, y_pred)), 4)
                    rmse = round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4)
                    metrics = {'r2': r2, 'r2_percent': round(r2*100,1), 'rmse': rmse}
                    performance = r2
                else:
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    acc  = round(float(accuracy_score(y_te, y_pred)), 4)
                    metrics = {'accuracy': acc, 'accuracy_percent': round(acc*100,1)}
                    performance = acc

                fi = []
                if hasattr(model, 'feature_importances_'):
                    raw = model.feature_importances_
                    fi  = sorted(
                        [{'feature': c, 'importance': round(float(v),4)}
                         for c, v in zip(feature_cols, raw)],
                        key=lambda x: -x['importance']
                    )[:6]

                results.append({
                    'target':      target,
                    'task':        task,
                    'metrics':     metrics,
                    'performance': round(performance, 4),
                    'feature_importance': fi,
                    'n_rows':      len(df_work),
                    'status':      'success',
                })
            except Exception as e:
                results.append({'target': target, 'error': str(e), 'status': 'error'})

        # Rank by performance
        successful = [r for r in results if r.get('status') == 'success']
        successful.sort(key=lambda x: -x['performance'])
        for i, r in enumerate(successful):
            r['rank'] = i + 1

        best = successful[0] if successful else None

        return {
            'task': 'multi_target',
            'feature_cols': feature_cols,
            'results': results,
            'best_target': best['target'] if best else None,
            'best_performance': best['performance'] if best else None,
            'summary': f"Compared {len(results)} targets. Best: '{best['target']}' ({best['task']}, {list(best['metrics'].values())[0]:.3f})" if best else "No successful comparisons.",
        }


# ═══════════════════════════════════════════════════════════════════
# 3. SURVIVAL ANALYSIS ENGINE (Kaplan-Meier)
#    Estimates survival/event-free probability over time.
#    No lifelines required — pure numpy implementation.
# ═══════════════════════════════════════════════════════════════════

class SurvivalAnalysisEngine:

    def run_kaplan_meier(self, df: pd.DataFrame, duration_col: str,
                          event_col: str, group_col: str = None) -> dict:
        """
        duration_col : numeric column — time until event or censoring
        event_col    : binary column — 1=event occurred, 0=censored
        group_col    : optional categorical column to stratify curves
        """
        if duration_col not in df.columns:
            return {'error': f"Duration column '{duration_col}' not found."}
        if event_col not in df.columns:
            return {'error': f"Event column '{event_col}' not found."}

        df_work = df[[duration_col, event_col] +
                     ([group_col] if group_col and group_col in df.columns else [])].dropna()

        df_work[duration_col] = pd.to_numeric(df_work[duration_col], errors='coerce')
        df_work[event_col]    = pd.to_numeric(df_work[event_col],    errors='coerce')
        df_work = df_work.dropna()

        if len(df_work) < 5:
            return {'error': 'Not enough data for survival analysis (need ≥5 rows).'}

        groups = {}
        if group_col and group_col in df_work.columns:
            for grp in df_work[group_col].unique():
                grp_df = df_work[df_work[group_col] == grp]
                groups[str(grp)] = self._km_curve(
                    grp_df[duration_col].values,
                    grp_df[event_col].values,
                )
        else:
            groups['Overall'] = self._km_curve(
                df_work[duration_col].values,
                df_work[event_col].values,
            )

        # Median survival per group
        medians = {}
        for grp, curve in groups.items():
            below = [(t, s) for t, s in zip(curve['times'], curve['survival'])
                     if s <= 0.5]
            medians[grp] = round(below[0][0], 2) if below else None

        n_events  = int(df_work[event_col].sum())
        n_censored = int((df_work[event_col] == 0).sum())

        return {
            'task':          'survival_analysis',
            'duration_col':  duration_col,
            'event_col':     event_col,
            'group_col':     group_col,
            'n_total':       len(df_work),
            'n_events':      n_events,
            'n_censored':    n_censored,
            'event_rate':    round(n_events / len(df_work) * 100, 1),
            'groups':        groups,
            'median_survival': medians,
            'interpretation': self._interpret(medians, n_events, len(df_work), event_col),
        }

    def _km_curve(self, durations: np.ndarray, events: np.ndarray) -> dict:
        order   = np.argsort(durations)
        t_sorted = durations[order]
        e_sorted = events[order]

        unique_times = np.unique(t_sorted[e_sorted == 1])
        survival = 1.0
        times    = [0.0]
        surv_vals = [1.0]
        ci_upper  = [1.0]
        ci_lower  = [1.0]
        n_risk    = len(durations)

        for t in unique_times:
            mask   = t_sorted == t
            n_die  = int(e_sorted[mask].sum())
            n_risk_t = int((t_sorted >= t).sum())
            if n_risk_t == 0:
                continue
            survival *= (1 - n_die / n_risk_t)
            se = np.sqrt(survival * (1 - survival) / max(n_risk_t, 1))
            times.append(round(float(t), 3))
            surv_vals.append(round(survival, 4))
            ci_upper.append(round(min(survival + 1.96 * se, 1.0), 4))
            ci_lower.append(round(max(survival - 1.96 * se, 0.0), 4))

        return {
            'times':    times,
            'survival': surv_vals,
            'ci_upper': ci_upper,
            'ci_lower': ci_lower,
        }

    def _interpret(self, medians: dict, n_events: int, n_total: int, event_col: str) -> str:
        parts = []
        for grp, med in medians.items():
            if med:
                parts.append(f"{grp}: median survival = {med} time units")
            else:
                parts.append(f"{grp}: median survival not reached (>50% still event-free)")
        event_rate = round(n_events / n_total * 100, 1)
        return (f"Survival analysis of '{event_col}' across {n_total} patients. "
                f"Event rate: {event_rate}%. " + " | ".join(parts))


# ═══════════════════════════════════════════════════════════════════
# 4. ALERT RULES ENGINE
#    Define, save, and evaluate custom clinical threshold rules.
#    Rules are stored as JSON and can be re-run on new uploads.
# ═══════════════════════════════════════════════════════════════════

class AlertRulesEngine:

    def evaluate_rules(self, df: pd.DataFrame, rules: list) -> dict:
        """
        rules: list of dicts:
        {
          "name": "High Glucose",
          "column": "glucose",
          "operator": ">",        # > < >= <= == !=
          "value": 140,
          "severity": "critical", # critical | warning | info
          "action": "Flag for immediate review"
        }
        """
        triggered = []
        summary   = {}

        for rule in rules:
            col = rule.get('column', '')
            if col not in df.columns:
                continue

            op  = rule.get('operator', '>')
            val = rule.get('value')
            sev = rule.get('severity', 'warning')

            try:
                col_series = pd.to_numeric(df[col], errors='coerce').dropna()
                v = float(val)

                ops = {
                    '>':  col_series > v,
                    '<':  col_series < v,
                    '>=': col_series >= v,
                    '<=': col_series <= v,
                    '==': col_series == v,
                    '!=': col_series != v,
                }
                mask      = ops.get(op, col_series > v)
                n_trigger = int(mask.sum())
                n_total   = len(col_series)
                pct       = round(n_trigger / n_total * 100, 1) if n_total > 0 else 0

                triggered_rows = df[mask].index.tolist()[:10]

                triggered.append({
                    'rule_name':      rule.get('name', f"{col} {op} {val}"),
                    'column':         col,
                    'operator':       op,
                    'threshold':      val,
                    'severity':       sev,
                    'action':         rule.get('action', 'Review required'),
                    'n_triggered':    n_trigger,
                    'n_total':        n_total,
                    'percent':        pct,
                    'triggered_rows': triggered_rows,
                    'status':         'TRIGGERED' if n_trigger > 0 else 'CLEAR',
                    'col_mean':       round(float(col_series.mean()), 3),
                    'col_max':        round(float(col_series.max()), 3),
                })

                summary[sev] = summary.get(sev, 0) + (1 if n_trigger > 0 else 0)

            except Exception as e:
                triggered.append({
                    'rule_name': rule.get('name', col),
                    'column': col,
                    'error': str(e),
                    'status': 'ERROR',
                })

        critical_count = sum(1 for t in triggered if t.get('severity') == 'critical' and t.get('status') == 'TRIGGERED')
        warning_count  = sum(1 for t in triggered if t.get('severity') == 'warning'  and t.get('status') == 'TRIGGERED')
        all_clear      = critical_count == 0 and warning_count == 0

        return {
            'task':            'alert_rules',
            'rules_evaluated': len(rules),
            'triggered':       triggered,
            'critical_alerts': critical_count,
            'warning_alerts':  warning_count,
            'all_clear':       all_clear,
            'overall_status':  'ALL_CLEAR' if all_clear else ('CRITICAL' if critical_count > 0 else 'WARNING'),
            'summary':         summary,
        }

    def suggest_rules(self, df: pd.DataFrame) -> list:
        """Auto-suggest rules based on common clinical column name patterns."""
        suggestions = []
        patterns = {
            'glucose':       [('>', 140, 'warning', 'High glucose — check for diabetes'),
                              ('>', 200, 'critical', 'Critical hyperglycaemia — immediate review'),
                              ('<', 70,  'critical', 'Hypoglycaemia — immediate review')],
            'blood_pressure':[('>',140, 'warning', 'Elevated BP — hypertension risk'),
                              ('>',180, 'critical', 'Hypertensive crisis')],
            'systolic':      [('>',140, 'warning', 'Elevated systolic BP'),
                              ('>',180, 'critical', 'Hypertensive crisis')],
            'bmi':           [('>',30, 'warning', 'Obesity — review lifestyle factors'),
                              ('>',40, 'critical', 'Severe obesity — bariatric referral')],
            'age':           [('>',65, 'info', 'Elderly patient — enhanced care protocol')],
            'cholesterol':   [('>',200, 'warning', 'Elevated cholesterol'),
                              ('>',240, 'critical', 'High cholesterol — statin review')],
            'heart_rate':    [('>',100, 'warning', 'Tachycardia'),
                              ('<', 60, 'info', 'Bradycardia — check medications'),
                              ('>',150, 'critical', 'Critical tachycardia')],
            'temperature':   [('>',38, 'warning', 'Fever'),
                              ('>',39.5, 'critical', 'High fever — urgent review')],
            'oxygen':        [('<',95, 'warning', 'Low SpO2'),
                              ('<',90, 'critical', 'Critical hypoxia — emergency')],
            'creatinine':    [('>',1.2, 'warning', 'Elevated creatinine — renal function'),
                              ('>',2.0, 'critical', 'Severe renal impairment')],
        }
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_').replace('-', '_')
            for pattern_key, rules in patterns.items():
                if pattern_key in col_lower:
                    for op, val, sev, action in rules:
                        suggestions.append({
                            'name':     f"{col} {op} {val}",
                            'column':   col,
                            'operator': op,
                            'value':    val,
                            'severity': sev,
                            'action':   action,
                        })
        return suggestions


# ═══════════════════════════════════════════════════════════════════
# 5. DATASET COMPARATOR
#    Compare two datasets statistically — distributions, means,
#    model performance gap (before/after, control/treatment).
# ═══════════════════════════════════════════════════════════════════

class DatasetComparator:

    def compare(self, df_a: pd.DataFrame, df_b: pd.DataFrame,
                name_a: str = 'Dataset A', name_b: str = 'Dataset B') -> dict:

        common_cols = [c for c in df_a.columns if c in df_b.columns]
        if not common_cols:
            return {'error': 'No common columns between the two datasets.'}

        col_comparisons = []
        significant_diffs = []

        for col in common_cols:
            s_a = pd.to_numeric(df_a[col], errors='coerce').dropna()
            s_b = pd.to_numeric(df_b[col], errors='coerce').dropna()

            if len(s_a) < 3 or len(s_b) < 3:
                continue

            mean_a = float(s_a.mean())
            mean_b = float(s_b.mean())
            std_a  = float(s_a.std())
            std_b  = float(s_b.std())

            # Cohen's d effect size
            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2) or 1
            cohens_d   = (mean_a - mean_b) / pooled_std
            effect     = ('large' if abs(cohens_d) > 0.8
                          else 'medium' if abs(cohens_d) > 0.5
                          else 'small'  if abs(cohens_d) > 0.2 else 'negligible')

            # Welch t-test (manual, no scipy)
            se = np.sqrt(std_a**2/len(s_a) + std_b**2/len(s_b))
            t_stat = (mean_a - mean_b) / se if se > 0 else 0
            # Approximate p-value from t (two-tailed, rough)
            p_approx = 2 * (1 - min(abs(t_stat) / (abs(t_stat) + 3), 0.999))

            entry = {
                'column':   col,
                'mean_a':   round(mean_a, 3),
                'mean_b':   round(mean_b, 3),
                'std_a':    round(std_a,  3),
                'std_b':    round(std_b,  3),
                'mean_diff': round(mean_a - mean_b, 3),
                'mean_diff_pct': round((mean_a - mean_b) / abs(mean_b) * 100, 1) if mean_b != 0 else 0,
                'cohens_d': round(cohens_d, 3),
                'effect_size': effect,
                'p_approx': round(p_approx, 4),
                'significant': p_approx < 0.05,
                'direction': 'A_higher' if mean_a > mean_b else 'B_higher' if mean_b > mean_a else 'equal',
            }
            col_comparisons.append(entry)
            if entry['significant'] and effect in ('medium', 'large'):
                significant_diffs.append(col)

        # Dataset-level stats
        numeric_a = df_a.select_dtypes(include=[np.number])
        numeric_b = df_b.select_dtypes(include=[np.number])

        return {
            'task':               'dataset_comparison',
            'name_a':             name_a,
            'name_b':             name_b,
            'rows_a':             len(df_a),
            'rows_b':             len(df_b),
            'cols_a':             len(df_a.columns),
            'cols_b':             len(df_b.columns),
            'common_columns':     common_cols,
            'n_common':           len(common_cols),
            'col_comparisons':    col_comparisons,
            'significant_diffs':  significant_diffs,
            'n_significant':      len(significant_diffs),
            'summary': (
                f"Compared {name_a} ({len(df_a):,} rows) vs {name_b} ({len(df_b):,} rows). "
                f"{len(significant_diffs)} column(s) show statistically significant differences: "
                f"{', '.join(significant_diffs[:5]) or 'none'}."
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# 6. CLINICAL CODING ASSISTANT
#    LLM-powered: maps dataset column values to ICD-10 codes,
#    flags potential drug interactions, suggests clinical coding.
# ═══════════════════════════════════════════════════════════════════

class ClinicalCodingAssistant:

    # Common ICD-10 quick-lookup (no external library needed)
    ICD10_COMMON = {
        'diabetes':        ('E11', 'Type 2 diabetes mellitus'),
        'hypertension':    ('I10', 'Essential hypertension'),
        'heart_failure':   ('I50', 'Heart failure'),
        'obesity':         ('E66', 'Obesity'),
        'asthma':          ('J45', 'Asthma'),
        'copd':            ('J44', 'Chronic obstructive pulmonary disease'),
        'depression':      ('F32', 'Major depressive episode'),
        'anxiety':         ('F41', 'Anxiety disorders'),
        'stroke':          ('I63', 'Cerebral infarction'),
        'pneumonia':       ('J18', 'Pneumonia, unspecified'),
        'sepsis':          ('A41', 'Other sepsis'),
        'ckd':             ('N18', 'Chronic kidney disease'),
        'anemia':          ('D64', 'Other anaemias'),
        'fracture':        ('M84', 'Disorders of continuity of bone'),
        'cancer':          ('C80', 'Malignant neoplasm, primary site unknown'),
        'infection':       ('B99', 'Other and unspecified infectious diseases'),
        'atrial_fib':      ('I48', 'Atrial fibrillation and flutter'),
        'mi':              ('I21', 'Acute myocardial infarction'),
        'liver':           ('K76', 'Other diseases of liver'),
        'alzheimer':       ('G30', "Alzheimer's disease"),
    }

    # Common drug interactions (simplified)
    DRUG_INTERACTIONS = [
        ('warfarin', 'aspirin',    'HIGH',   'Increased bleeding risk'),
        ('metformin','alcohol',    'MEDIUM', 'Lactic acidosis risk'),
        ('ssri',     'maoi',       'HIGH',   'Serotonin syndrome — CONTRAINDICATED'),
        ('statins',  'grapefruit', 'MEDIUM', 'Increased statin levels'),
        ('ace',      'potassium',  'MEDIUM', 'Hyperkalaemia risk'),
        ('nsaid',    'warfarin',   'HIGH',   'Bleeding risk — avoid combination'),
        ('digoxin',  'amiodarone', 'HIGH',   'Digoxin toxicity risk'),
        ('lithium',  'nsaid',      'HIGH',   'Lithium toxicity risk'),
    ]

    def detect_icd_codes(self, df: pd.DataFrame) -> list:
        """Scan column names and categorical values for ICD-10 code suggestions."""
        suggestions = []
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ')
            for keyword, (code, desc) in self.ICD10_COMMON.items():
                if keyword.replace('_', ' ') in col_lower:
                    suggestions.append({
                        'column': col,
                        'keyword': keyword,
                        'icd10_code': code,
                        'description': desc,
                        'match_type': 'column_name',
                    })
                    break

            # Also scan categorical values
            if df[col].dtype == object:
                sample_vals = df[col].dropna().astype(str).str.lower().unique()[:20]
                for val in sample_vals:
                    for keyword, (code, desc) in self.ICD10_COMMON.items():
                        if keyword.replace('_', ' ') in val:
                            suggestions.append({
                                'column': col,
                                'value': val,
                                'keyword': keyword,
                                'icd10_code': code,
                                'description': desc,
                                'match_type': 'value',
                            })
                            break

        # Deduplicate
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = (s['column'], s['icd10_code'])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        return unique_suggestions

    def check_drug_interactions(self, df: pd.DataFrame) -> list:
        """Scan column names and values for potential drug interaction flags."""
        all_text = ' '.join(df.columns).lower()
        for col in df.select_dtypes(include=['object']).columns:
            all_text += ' ' + ' '.join(df[col].dropna().astype(str).str.lower().unique()[:20])

        found = []
        for drug_a, drug_b, severity, message in self.DRUG_INTERACTIONS:
            if drug_a in all_text and drug_b in all_text:
                found.append({
                    'drug_a':   drug_a,
                    'drug_b':   drug_b,
                    'severity': severity,
                    'message':  message,
                })
        return found

    def get_llm_coding_advice(self, df: pd.DataFrame, col_info: list,
                               llm_client=None, patient_row: dict = None) -> dict:
        """Use LLM to generate clinical coding suggestions for this dataset."""
        icd_suggestions = self.detect_icd_codes(df)
        drug_flags      = self.check_drug_interactions(df)

        result = {
            'icd10_suggestions': icd_suggestions,
            'drug_interaction_flags': drug_flags,
            'llm_advice': None,
        }

        if llm_client:
            context = {
                'columns': [c['name'] for c in col_info[:15]],
                'column_types': {c['name']: c['col_type'] for c in col_info[:15]},
                'detected_icd': [(s['icd10_code'], s['description']) for s in icd_suggestions[:8]],
                'drug_flags': [(f['drug_a'], f['drug_b'], f['severity']) for f in drug_flags],
                'patient_sample': patient_row or {},
            }
            prompt = f"""You are a clinical coding expert. Analyse this healthcare dataset and provide:
1. ICD-10 coding recommendations for the conditions you can detect
2. Any clinical documentation gaps
3. Suggested additional data fields that would improve coding accuracy
4. Billing/coding compliance notes

Respond ONLY with valid JSON:
{{
  "coding_recommendations": ["recommendation 1", "recommendation 2"],
  "documentation_gaps": ["gap 1", "gap 2"],
  "suggested_fields": ["field 1", "field 2"],
  "compliance_notes": ["note 1"],
  "clinical_summary": "brief overall assessment"
}}

Dataset context:
{json.dumps(context, indent=2)}"""

            try:
                resp = llm_client.ask(prompt)
                if resp.get('success'):
                    import re
                    m = re.search(r'\{[\s\S]*\}', resp['message'])
                    if m:
                        result['llm_advice'] = json.loads(m.group(0))
            except Exception:
                pass

        return result
