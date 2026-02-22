# healthcare_app/models.py

from django.db import models


class DataSession(models.Model):
    """
    Stores the uploaded dataset for a user's browser session.
    The DataFrame is serialised as base64-encoded pickle so it
    survives server restarts without writing temp files.
    """
    session_key       = models.CharField(max_length=64, db_index=True)
    file_name         = models.CharField(max_length=255, default='dataset')
    row_count         = models.IntegerField(default=0)
    col_count         = models.IntegerField(default=0)
    # Base64-encoded pickle of the pandas DataFrame
    dataframe_pickle  = models.TextField(blank=True, default='')
    created_at        = models.DateTimeField(auto_now_add=True)
    updated_at        = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"DataSession({self.session_key[:8]}… | {self.file_name} | {self.row_count}×{self.col_count})"


class AnalysisResult(models.Model):
    """
    Stores the latest ML analysis result for a session so the
    chat assistant can reference it.
    """
    session_key  = models.CharField(max_length=64, db_index=True)
    result_json  = models.TextField(default='{}')
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"AnalysisResult({self.session_key[:8]}…)"


class ChatMessage(models.Model):
    """
    Persists chat history per session (optional — used for audit/context).
    """
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant')]

    session_key = models.CharField(max_length=64, db_index=True)
    role        = models.CharField(max_length=12, choices=ROLE_CHOICES)
    content     = models.TextField()
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"
