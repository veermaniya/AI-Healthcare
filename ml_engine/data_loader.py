"""
Healthcare AI - Data Loaders
Supports: Excel (.xlsx, .xls, .csv), SQL Server, PostgreSQL
"""

import pandas as pd
import io


def load_excel_file(file_obj) -> pd.DataFrame:
    """Load Excel or CSV file into DataFrame"""
    filename = file_obj.name.lower()
    if filename.endswith('.csv'):
        df = pd.read_csv(file_obj)
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_obj)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return df


def load_from_sql(connection_string: str, query: str) -> pd.DataFrame:
    """
    Load data from SQL Server or other databases
    
    Connection string formats:
    SQL Server: mssql+pyodbc://user:pass@server/database?driver=ODBC+Driver+17+for+SQL+Server
    PostgreSQL: postgresql://user:pass@host:5432/database
    MySQL:      mysql+pymysql://user:pass@host/database
    SQLite:     sqlite:///path/to/db.sqlite3
    """
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df
    except ImportError:
        raise ImportError("SQLAlchemy not installed. Run: pip install sqlalchemy pyodbc")
    except Exception as e:
        raise ConnectionError(f"Database connection failed: {str(e)}")


def get_table_names(connection_string: str) -> list:
    """Get all table names from a database"""
    try:
        from sqlalchemy import create_engine, inspect
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        raise ConnectionError(f"Could not fetch tables: {str(e)}")


def get_column_names(connection_string: str, table_name: str) -> list:
    """Get column names for a specific table"""
    try:
        from sqlalchemy import create_engine, inspect
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        cols = inspector.get_columns(table_name)
        return [col['name'] for col in cols]
    except Exception as e:
        raise ConnectionError(f"Could not fetch columns: {str(e)}")


def dataframe_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of a DataFrame for display"""
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'null_counts': {col: int(df[col].isnull().sum()) for col in df.columns},
        'preview': df.head(500).fillna('').to_dict(orient='records')
    }


def get_column_stats(df) -> dict:
    """
    Returns per-column stats for the smart prediction form sliders.
    Numeric cols: min, max, mean, median, step
    Categorical cols: unique_values, most_common
    """
    import numpy as np
    stats = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]:
            col_min  = round(float(series.min()), 3)
            col_max  = round(float(series.max()), 3)
            col_mean = round(float(series.mean()), 3)
            col_med  = round(float(series.median()), 3)
            col_range = col_max - col_min
            # sensible step size
            if col_range == 0:
                step = 1
            elif col_range < 1:
                step = round(col_range / 100, 6)
            elif col_range < 10:
                step = 0.1
            elif col_range < 100:
                step = 1
            else:
                step = round(col_range / 100)
            stats[col] = {
                'type': 'numeric',
                'min': col_min, 'max': col_max,
                'mean': col_mean, 'median': col_med,
                'step': step,
            }
        else:
            unique_vals = series.astype(str).unique().tolist()[:50]
            most_common = series.astype(str).mode().iloc[0] if len(series) > 0 else ''
            stats[col] = {
                'type': 'categorical',
                'unique_values': unique_vals,
                'most_common': most_common,
            }
    return stats
