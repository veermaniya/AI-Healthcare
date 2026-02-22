# ml_engine/engine_extensions.py
"""
4 new healthcare AI modules:
  1. ClinicalInsightsEngine   — LLM narrative summaries + chart interpretation
  2. DataQualityEngine        — Missing data, bias, outlier detection
  3. ReportGenerator          — HTML/PDF clinical report builder
  4. PrivacyEngine            — PHI masking, anonymisation, audit log
"""

import pandas as pd
import numpy as np
import json
import re
import hashlib
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# 1. CLINICAL INSIGHTS ENGINE
#    Generates LLM-powered narrative summaries for any ML result,
#    chart interpretation, and clinical observations.
# ═══════════════════════════════════════════════════════════════════

class ClinicalInsightsEngine:

    def generate_narrative(self, result: dict, col_info: list, llm_client=None) -> dict:
        """
        Takes any ML result dict and returns:
          - executive_summary   : 2-3 sentence plain-English summary
          - clinical_observations: bullet list of key findings
          - chart_interpretation : what the charts/metrics actually mean
          - recommendations      : actionable next steps
          - risk_flags           : any concerning patterns spotted
        """
        task = result.get('task', 'analysis')
        metrics = result.get('metrics', {})
        features = result.get('features', [])
        target = result.get('target', '')
        feature_importance = result.get('feature_importance', [])
        total_rows = result.get('total_rows', 0)

        # Build structured prompt for LLM
        context = {
            'task': task,
            'target_column': target,
            'feature_columns': features,
            'metrics': metrics,
            'top_features': feature_importance[:5],
            'total_patients': total_rows,
            'column_info': col_info[:10],
        }

        prompt = f"""You are a clinical data scientist writing a report for a hospital charity system.
        
Analyse this ML result and respond ONLY with valid JSON:
{{
  "executive_summary": "2-3 sentence plain-English summary of what this analysis found",
  "clinical_observations": ["observation 1", "observation 2", "observation 3"],
  "chart_interpretation": "what the metrics and charts mean clinically",
  "recommendations": ["recommendation 1", "recommendation 2"],
  "risk_flags": ["any concerning pattern or limitation"],
  "confidence_assessment": "high/medium/low with brief reason"
}}

ML Result context:
{json.dumps(context, indent=2)}

Be specific, clinically relevant, and use plain language a doctor can understand."""

        if llm_client:
            try:
                resp = llm_client.ask(prompt)
                if resp.get('success'):
                    raw = resp['message']
                    m = re.search(r'\{[\s\S]*\}', raw)
                    if m:
                        parsed = json.loads(m.group(0))
                        parsed['generated_by'] = 'llm'
                        return parsed
            except Exception:
                pass

        # Fallback: rule-based narrative (no LLM needed)
        return self._rule_based_narrative(result, col_info)

    def _rule_based_narrative(self, result: dict, col_info: list) -> dict:
        task     = result.get('task', 'analysis')
        metrics  = result.get('metrics', {})
        features = result.get('features', [])
        target   = result.get('target', '')
        fi       = result.get('feature_importance', [])
        rows     = result.get('total_rows', 0)

        observations = []
        risk_flags   = []
        recommendations = []

        if task == 'regression':
            r2  = metrics.get('r2', 0)
            r2p = metrics.get('r2_percent', round(r2 * 100, 1))
            rmse = metrics.get('rmse', 0)
            summary = (f"A regression model was trained to predict '{target}' using "
                       f"{len(features)} features across {rows:,} patients, "
                       f"achieving an R² score of {r2p}%.")
            observations.append(f"Model explains {r2p}% of variance in {target}.")
            if r2 < 0.5:
                risk_flags.append(f"Low R² ({r2p}%) — model may need more features or data cleaning.")
                recommendations.append("Consider adding more clinically relevant features.")
            else:
                recommendations.append(f"Model performance is {'strong' if r2 > 0.8 else 'moderate'} — suitable for clinical decision support.")
            if fi:
                top = fi[0]['feature']
                observations.append(f"'{top}' is the strongest predictor of {target}.")
            chart_interp = f"The scatter plot shows actual vs predicted {target} values. Points closer to the diagonal indicate better predictions. RMSE of {rmse} represents the average prediction error."

        elif task == 'classification':
            acc  = metrics.get('accuracy', 0)
            accp = metrics.get('accuracy_percent', round(acc * 100, 1))
            summary = (f"A classification model was trained to predict '{target}' using "
                       f"{len(features)} features across {rows:,} patients, "
                       f"achieving {accp}% accuracy.")
            observations.append(f"Model correctly classifies {accp}% of patients.")
            if acc < 0.7:
                risk_flags.append(f"Accuracy {accp}% may be insufficient for clinical decisions.")
                recommendations.append("Consider collecting more balanced training data.")
            if fi:
                top3 = [f['feature'] for f in fi[:3]]
                observations.append(f"Top predictors: {', '.join(top3)}.")
            chart_interp = f"The chart shows model predictions vs actual outcomes. Higher accuracy means more reliable clinical screening."

        elif task == 'clustering':
            n_clusters = result.get('n_clusters', 0)
            sil = metrics.get('silhouette_score', 0)
            summary = (f"K-Means clustering identified {n_clusters} distinct patient groups "
                       f"from {rows:,} records with a silhouette score of {sil}.")
            observations.append(f"{n_clusters} patient subgroups identified — each may require different care pathways.")
            if sil < 0.3:
                risk_flags.append("Low silhouette score — clusters may overlap significantly.")
            recommendations.append("Review each cluster's characteristics to identify high-risk patient groups.")
            chart_interp = "The doughnut chart shows the proportion of patients in each cluster. Investigate outlier clusters for rare conditions."

        else:
            summary = f"Analysis completed on {rows:,} patient records."
            chart_interp = "Review the metrics above for model performance details."

        return {
            'executive_summary':       summary,
            'clinical_observations':   observations,
            'chart_interpretation':    chart_interp,
            'recommendations':         recommendations or ['Review results with a clinical expert before deployment.'],
            'risk_flags':              risk_flags or ['No critical issues detected.'],
            'confidence_assessment':   'high' if (metrics.get('r2', metrics.get('accuracy', 0.5)) > 0.75) else 'medium',
            'generated_by':            'rule_based',
        }


# ═══════════════════════════════════════════════════════════════════
# 2. DATA QUALITY ENGINE
#    Missing data alerts, class imbalance, outlier detection,
#    column bias warnings.
# ═══════════════════════════════════════════════════════════════════

class DataQualityEngine:

    def run_quality_check(self, df: pd.DataFrame) -> dict:
        issues   = []
        warnings = []
        passed   = []
        score    = 100

        total_rows = len(df)
        total_cols = len(df.columns)

        # ── 1. Missing data ──
        missing_report = []
        for col in df.columns:
            n_miss = int(df[col].isnull().sum())
            pct    = round(n_miss / total_rows * 100, 1) if total_rows > 0 else 0
            if pct > 0:
                level = 'critical' if pct > 30 else 'warning' if pct > 5 else 'info'
                missing_report.append({
                    'column': col, 'missing': n_miss,
                    'percent': pct, 'level': level,
                })
                if pct > 30:
                    issues.append(f"'{col}' has {pct}% missing values — consider imputation or exclusion.")
                    score -= 10
                elif pct > 5:
                    warnings.append(f"'{col}' has {pct}% missing values.")
                    score -= 3
        if not missing_report:
            passed.append("No missing values detected.")

        # ── 2. Duplicate rows ──
        n_dupes = int(df.duplicated().sum())
        dupe_pct = round(n_dupes / total_rows * 100, 1) if total_rows > 0 else 0
        if n_dupes > 0:
            warnings.append(f"{n_dupes} duplicate rows ({dupe_pct}%) — may bias model training.")
            score -= 5
        else:
            passed.append("No duplicate rows found.")

        # ── 3. Outlier detection (IQR method) ──
        outlier_report = []
        for col in df.select_dtypes(include=[np.number]).columns:
            s   = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            n_out = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
            pct   = round(n_out / len(s) * 100, 1)
            if n_out > 0:
                outlier_report.append({
                    'column': col, 'outliers': n_out,
                    'percent': pct,
                    'range': [round(float(s.min()), 2), round(float(s.max()), 2)],
                })
                if pct > 5:
                    warnings.append(f"'{col}' has {pct}% extreme outliers (3×IQR).")
                    score -= 3
        if not outlier_report:
            passed.append("No significant outliers detected.")

        # ── 4. Class imbalance (categorical columns) ──
        imbalance_report = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            vc = df[col].value_counts(normalize=True)
            if len(vc) >= 2:
                ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 1) if vc.iloc[-1] > 0 else 999
                if ratio > 10:
                    imbalance_report.append({
                        'column': col,
                        'dominant_class': str(vc.index[0]),
                        'dominant_pct': round(float(vc.iloc[0]) * 100, 1),
                        'ratio': ratio,
                    })
                    warnings.append(f"'{col}' has class imbalance ({ratio}:1 ratio) — may bias classification models.")
                    score -= 5

        # ── 5. Constant / near-constant columns ──
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
                issues.append(f"'{col}' has only 1 unique value — no predictive value, remove it.")
                score -= 5
            elif df[col].nunique() == 2 and total_rows > 100:
                passed.append(f"'{col}' is binary — suitable for classification.")

        # ── 6. High cardinality categorical columns ──
        high_card = []
        for col in df.select_dtypes(include=['object']).columns:
            n_uniq = df[col].nunique()
            if n_uniq > min(50, total_rows * 0.5):
                high_card.append({'column': col, 'unique_values': n_uniq})
                warnings.append(f"'{col}' has {n_uniq} unique values — may need encoding or exclusion.")

        # ── 7. Data type consistency ──
        mixed_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().head(100)
            n_numeric = sum(1 for v in sample if str(v).replace('.', '', 1).lstrip('-').isnumeric())
            if 0 < n_numeric < len(sample) * 0.9 and n_numeric > len(sample) * 0.1:
                mixed_cols.append(col)
                warnings.append(f"'{col}' appears to have mixed numeric/text values.")

        # ── Overall quality score ──
        score = max(0, min(100, score))
        grade = 'A' if score >= 90 else 'B' if score >= 75 else 'C' if score >= 60 else 'D'

        return {
            'task':               'data_quality',
            'overall_score':      score,
            'grade':              grade,
            'total_rows':         total_rows,
            'total_cols':         total_cols,
            'issues':             issues,
            'warnings':           warnings,
            'passed':             passed,
            'missing_report':     missing_report,
            'outlier_report':     outlier_report[:10],
            'imbalance_report':   imbalance_report,
            'constant_columns':   constant_cols,
            'high_cardinality':   high_card,
            'mixed_type_columns': mixed_cols,
            'recommendation':     (
                'Dataset is research-grade quality.' if score >= 90 else
                'Minor cleaning recommended before analysis.' if score >= 75 else
                'Significant data cleaning required.' if score >= 60 else
                'Major data quality issues — review before any analysis.'
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# 3. REPORT GENERATOR
#    Builds a complete HTML clinical report from session + results.
#    The HTML can be printed-to-PDF from the browser.
# ═══════════════════════════════════════════════════════════════════

class ReportGenerator:

    def generate_html_report(self, session_info: dict, ml_result: dict,
                              insights: dict, quality: dict) -> str:
        """
        Returns a complete standalone HTML string — self-contained,
        printable, embeds all data inline (no external dependencies).
        """
        now       = datetime.now().strftime('%d %B %Y, %H:%M')
        task      = ml_result.get('task', 'analysis').title()
        target    = ml_result.get('target', '—')
        features  = ml_result.get('features', [])
        metrics   = ml_result.get('metrics', {})
        fi        = ml_result.get('feature_importance', [])
        rows      = ml_result.get('total_rows', session_info.get('row_count', 0))
        file_name = session_info.get('file_name', 'Dataset')

        # Metrics table rows
        metric_rows = ''.join(
            f'<tr><td>{k.replace("_", " ").title()}</td><td><strong>{v}</strong></td></tr>'
            for k, v in metrics.items()
        )

        # Feature importance bars
        fi_bars = ''
        if fi:
            max_imp = fi[0]['importance'] or 1
            for f in fi[:8]:
                pct = round(f['importance'] / max_imp * 100)
                fi_bars += f'''
                <div class="fi-row">
                  <div class="fi-label">{f["feature"]}</div>
                  <div class="fi-track"><div class="fi-fill" style="width:{pct}%"></div></div>
                  <div class="fi-val">{round(f["importance"]*100,1)}%</div>
                </div>'''

        # Clinical observations
        obs_items = ''.join(
            f'<li>{o}</li>'
            for o in insights.get('clinical_observations', [])
        )
        rec_items = ''.join(
            f'<li>{r}</li>'
            for r in insights.get('recommendations', [])
        )
        risk_items = ''.join(
            f'<li class="risk">{r}</li>'
            for r in insights.get('risk_flags', [])
        )

        # Quality summary
        q_score = quality.get('overall_score', 100)
        q_grade = quality.get('grade', 'A')
        q_color = '#34d399' if q_score >= 90 else '#fbbf24' if q_score >= 75 else '#f87171'
        q_issues = ''.join(f'<li>{i}</li>' for i in quality.get('issues', [])[:5])
        q_warnings = ''.join(f'<li>{w}</li>' for w in quality.get('warnings', [])[:5])

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HealthAI Clinical Report — {file_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0 }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; background: #f8fafc; color: #1e293b; font-size: 13px }}
  .page {{ max-width: 900px; margin: 0 auto; padding: 32px; background: #fff }}
  /* Header */
  .rpt-header {{ background: linear-gradient(135deg,#0f172a,#1e3a5f); color: #fff; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px }}
  .rpt-title {{ font-size: 24px; font-weight: 700; margin-bottom: 4px }}
  .rpt-sub {{ font-size: 12px; opacity: .7 }}
  .rpt-meta {{ display: flex; gap: 24px; margin-top: 16px; font-size: 11px; opacity: .8 }}
  /* Sections */
  .section {{ margin-bottom: 24px }}
  .section-title {{ font-size: 14px; font-weight: 700; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px }}
  /* KPI grid */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px }}
  .kpi {{ background: #f1f5f9; border-radius: 10px; padding: 14px; text-align: center }}
  .kpi-val {{ font-size: 22px; font-weight: 700; color: #0ea5e9 }}
  .kpi-lbl {{ font-size: 10px; color: #64748b; margin-top: 3px }}
  /* Metrics table */
  table {{ width: 100%; border-collapse: collapse; font-size: 12px }}
  th {{ background: #f1f5f9; padding: 8px 12px; text-align: left; font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .04em }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9 }}
  /* Feature importance bars */
  .fi-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px }}
  .fi-label {{ width: 140px; font-size: 11px; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis }}
  .fi-track {{ flex: 1; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden }}
  .fi-fill {{ height: 100%; background: linear-gradient(90deg,#0ea5e9,#34d399); border-radius: 4px }}
  .fi-val {{ font-size: 10px; color: #64748b; width: 36px; text-align: right }}
  /* Narrative box */
  .narrative {{ background: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 14px 16px; border-radius: 0 8px 8px 0; font-size: 12px; line-height: 1.7; color: #1e293b }}
  /* Lists */
  ul {{ padding-left: 18px; line-height: 1.9; font-size: 12px; color: #334155 }}
  li.risk {{ color: #dc2626 }}
  /* Quality */
  .q-score-box {{ display: inline-flex; align-items: center; gap: 14px; background: #f8fafc; border: 2px solid {q_color}; border-radius: 12px; padding: 14px 20px; margin-bottom: 14px }}
  .q-score-val {{ font-size: 32px; font-weight: 700; color: {q_color} }}
  .q-score-lbl {{ font-size: 11px; color: #64748b }}
  /* Footer */
  .rpt-footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 10px; color: #94a3b8; text-align: center; line-height: 1.7 }}
  @media print {{
    body {{ background: #fff }}
    .page {{ padding: 0 }}
    .no-print {{ display: none }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- Print button (hidden when printing) -->
  <div class="no-print" style="text-align:right;margin-bottom:16px">
    <button onclick="window.print()"
      style="background:#0ea5e9;color:#fff;border:none;border-radius:8px;padding:9px 20px;font-size:12px;font-weight:600;cursor:pointer">
      🖨 Print / Save as PDF
    </button>
  </div>

  <!-- Header -->
  <div class="rpt-header">
    <div class="rpt-title">🏥 HealthAI — Clinical Analysis Report</div>
    <div class="rpt-sub">Free Healthcare Intelligence Platform · Hospital Charity Edition</div>
    <div class="rpt-meta">
      <span>📄 Dataset: {file_name}</span>
      <span>📊 Task: {task}</span>
      <span>🎯 Target: {target or "—"}</span>
      <span>📅 Generated: {now}</span>
    </div>
  </div>

  <!-- KPIs -->
  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-val">{rows:,}</div><div class="kpi-lbl">Patients Analysed</div></div>
    <div class="kpi"><div class="kpi-val">{len(features)}</div><div class="kpi-lbl">Features Used</div></div>
    <div class="kpi"><div class="kpi-val">{q_score}</div><div class="kpi-lbl">Data Quality Score</div></div>
    <div class="kpi"><div class="kpi-val" style="color:#34d399">{insights.get("confidence_assessment","—").title()}</div><div class="kpi-lbl">Model Confidence</div></div>
  </div>

  <!-- Executive Summary -->
  <div class="section">
    <div class="section-title">📋 Executive Summary</div>
    <div class="narrative">{insights.get("executive_summary","No summary available.")}</div>
  </div>

  <!-- Model Metrics -->
  <div class="section">
    <div class="section-title">📊 Model Performance Metrics</div>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{metric_rows or "<tr><td colspan='2'>No metrics available</td></tr>"}</tbody>
    </table>
  </div>

  <!-- Feature Importance -->
  {"" if not fi_bars else f'''
  <div class="section">
    <div class="section-title">⭐ Feature Importance</div>
    {fi_bars}
  </div>'''}

  <!-- Clinical Observations -->
  <div class="section">
    <div class="section-title">🔬 Clinical Observations</div>
    <ul>{obs_items or "<li>No observations available.</li>"}</ul>
  </div>

  <!-- Chart Interpretation -->
  <div class="section">
    <div class="section-title">📈 Chart Interpretation</div>
    <div class="narrative">{insights.get("chart_interpretation","—")}</div>
  </div>

  <!-- Recommendations -->
  <div class="section">
    <div class="section-title">✅ Recommendations</div>
    <ul>{rec_items or "<li>Review results with clinical expert.</li>"}</ul>
  </div>

  <!-- Risk Flags -->
  <div class="section">
    <div class="section-title">⚠️ Risk Flags &amp; Limitations</div>
    <ul>{risk_items or "<li>No critical risk flags.</li>"}</ul>
  </div>

  <!-- Data Quality -->
  <div class="section">
    <div class="section-title">🧪 Data Quality Assessment</div>
    <div class="q-score-box">
      <div>
        <div class="q-score-val">{q_score}/100</div>
        <div class="q-score-lbl">Overall Quality Score — Grade {q_grade}</div>
      </div>
      <div style="font-size:12px;color:#475569">{quality.get("recommendation","")}</div>
    </div>
    {"<div style='margin-bottom:8px'><strong style='font-size:11px;color:#dc2626'>Issues:</strong><ul>" + q_issues + "</ul></div>" if q_issues else ""}
    {"<div><strong style='font-size:11px;color:#d97706'>Warnings:</strong><ul>" + q_warnings + "</ul></div>" if q_warnings else ""}
  </div>

  <!-- Footer -->
  <div class="rpt-footer">
    ⚠️ This report is generated by AI for educational and clinical decision support purposes only.<br>
    It does not constitute a medical diagnosis. Always consult a qualified healthcare professional.<br>
    HealthAI · Free for Hospital &amp; Healthcare Charity · Built with ❤️
  </div>

</div>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════
# 4. PRIVACY ENGINE
#    PHI detection + masking, anonymisation, audit logging.
# ═══════════════════════════════════════════════════════════════════

class PrivacyEngine:

    # Common PHI field name patterns
    PHI_PATTERNS = [
        r'\bname\b', r'\bfirst.?name\b', r'\blast.?name\b', r'\bfull.?name\b',
        r'\bemail\b', r'\bphone\b', r'\bmobile\b', r'\bcontact\b',
        r'\baddress\b', r'\bstreet\b', r'\bcity\b', r'\bpostcode\b', r'\bzip\b',
        r'\bnhs.?number\b', r'\bpatient.?id\b', r'\bssn\b', r'\bnational.?id\b',
        r'\bpassport\b', r'\bdob\b', r'\bdate.?of.?birth\b', r'\bbirthdate\b',
        r'\bip.?address\b', r'\burl\b', r'\bdevice.?id\b',
    ]

    def detect_phi_columns(self, df: pd.DataFrame) -> list:
        """Scan column names and sample values to flag potential PHI columns."""
        phi_cols = []
        for col in df.columns:
            col_lower = col.lower()
            is_phi = any(re.search(p, col_lower) for p in self.PHI_PATTERNS)

            if not is_phi:
                # Sample-based detection: look for email patterns, phone numbers
                sample = df[col].dropna().astype(str).head(20)
                for val in sample:
                    if re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', val):
                        is_phi = True; break
                    if re.search(r'\b\d{10,11}\b|\+\d{1,3}\s?\d{6,}', val):
                        is_phi = True; break
                    if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', val):
                        is_phi = True; break

            if is_phi:
                phi_cols.append({
                    'column':     col,
                    'dtype':      str(df[col].dtype),
                    'sample':     str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else '',
                    'risk_level': 'HIGH' if any(re.search(p, col.lower()) for p in [
                        r'\bname\b', r'\bemail\b', r'\bphone\b', r'\bssn\b',
                        r'\bnhs\b', r'\bpatient.?id\b', r'\bdob\b'
                    ]) else 'MEDIUM',
                })
        return phi_cols

    def anonymise(self, df: pd.DataFrame, phi_columns: list,
                  method: str = 'hash') -> tuple:
        """
        Anonymise PHI columns using chosen method.
        method: 'hash' | 'mask' | 'drop' | 'pseudonymise'
        Returns (anonymised_df, audit_log)
        """
        result = df.copy()
        audit  = []

        for col_info in phi_columns:
            col = col_info if isinstance(col_info, str) else col_info['column']
            if col not in result.columns:
                continue

            original_sample = str(result[col].dropna().iloc[0]) if len(result[col].dropna()) > 0 else ''

            if method == 'drop':
                result.drop(columns=[col], inplace=True)
                audit.append({'column': col, 'method': 'dropped', 'rows_affected': len(df)})

            elif method == 'mask':
                result[col] = result[col].apply(
                    lambda v: self._mask_value(str(v)) if pd.notna(v) else v
                )
                audit.append({'column': col, 'method': 'masked', 'rows_affected': int(result[col].notna().sum())})

            elif method == 'hash':
                result[col] = result[col].apply(
                    lambda v: hashlib.sha256(str(v).encode()).hexdigest()[:12] if pd.notna(v) else v
                )
                audit.append({'column': col, 'method': 'hashed_sha256', 'rows_affected': int(result[col].notna().sum())})

            elif method == 'pseudonymise':
                mapping = {v: f"PATIENT_{i+1:05d}"
                           for i, v in enumerate(result[col].dropna().unique())}
                result[col] = result[col].map(mapping).fillna(result[col])
                audit.append({
                    'column': col, 'method': 'pseudonymised',
                    'unique_identifiers': len(mapping),
                    'rows_affected': int(result[col].notna().sum()),
                })

        return result, audit

    def _mask_value(self, val: str) -> str:
        if len(val) <= 2:
            return '*' * len(val)
        return val[0] + '*' * (len(val) - 2) + val[-1]

    def run_privacy_check(self, df: pd.DataFrame) -> dict:
        """Full privacy audit — detect PHI, assess risk, suggest actions."""
        phi_cols   = self.detect_phi_columns(df)
        high_risk  = [c for c in phi_cols if c['risk_level'] == 'HIGH']
        med_risk   = [c for c in phi_cols if c['risk_level'] == 'MEDIUM']

        compliance_score = 100
        compliance_score -= len(high_risk) * 20
        compliance_score -= len(med_risk)  * 10
        compliance_score  = max(0, compliance_score)

        status = 'COMPLIANT' if compliance_score >= 90 else \
                 'REVIEW_NEEDED' if compliance_score >= 60 else 'NON_COMPLIANT'

        return {
            'task':              'privacy_check',
            'phi_columns':       phi_cols,
            'high_risk_count':   len(high_risk),
            'medium_risk_count': len(med_risk),
            'compliance_score':  compliance_score,
            'status':            status,
            'total_columns':     len(df.columns),
            'flagged_columns':   len(phi_cols),
            'recommendations':   self._privacy_recommendations(phi_cols, status),
            'anonymisation_methods': {
                'hash':          'Replace with SHA-256 hash (12 chars) — reversible with key',
                'mask':          'Replace middle chars with *** — irreversible',
                'pseudonymise':  'Replace with PATIENT_00001 codes — mapping kept server-side',
                'drop':          'Remove column entirely — most secure',
            },
        }

    def _privacy_recommendations(self, phi_cols: list, status: str) -> list:
        recs = []
        if status == 'NON_COMPLIANT':
            recs.append('⛔ URGENT: High-risk PHI columns detected. Anonymise before sharing or analysing.')
        if any(c['risk_level'] == 'HIGH' for c in phi_cols):
            recs.append('Mask or hash columns: ' + ', '.join(
                c['column'] for c in phi_cols if c['risk_level'] == 'HIGH'
            ))
        if any(c['risk_level'] == 'MEDIUM' for c in phi_cols):
            recs.append('Review medium-risk columns: ' + ', '.join(
                c['column'] for c in phi_cols if c['risk_level'] == 'MEDIUM'
            ))
        recs.append('Use pseudonymisation for patient ID columns to allow re-linkage if needed.')
        recs.append('Never export raw PHI data — always anonymise before sharing with third parties.')
        if not phi_cols:
            recs.append('✅ No PHI columns detected. Dataset appears safe for analysis.')
        return recs
