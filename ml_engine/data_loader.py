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
