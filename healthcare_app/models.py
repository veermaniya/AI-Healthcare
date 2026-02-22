from django.db import models
import json


class DataSession(models.Model):
    """Stores uploaded dataset sessions"""
    session_key = models.CharField(max_length=100, unique=True)
    source_type = models.CharField(max_length=20, choices=[('excel', 'Excel/CSV'), ('sql', 'SQL Server')])
    file_name = models.CharField(max_length=255, blank=True)
    sql_connection = models.TextField(blank=True)
    sql_query = models.TextField(blank=True)
    columns_json = models.TextField(default='[]')  # JSON list of column info
    row_count = models.IntegerField(default=0)
    col_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    preview_json = models.TextField(default='[]')

    def get_columns(self):
        return json.loads(self.columns_json)

    def get_preview(self):
        return json.loads(self.preview_json)

    def __str__(self):
        return f"Session {self.session_key} - {self.file_name or 'SQL'}"


class AnalysisResult(models.Model):
    """Stores ML analysis results"""
    session = models.ForeignKey(DataSession, on_delete=models.CASCADE, related_name='results')
    task_type = models.CharField(max_length=30)  # regression, classification, clustering
    feature_columns = models.TextField()  # comma-separated
    target_column = models.CharField(max_length=100, blank=True)
    model_type = models.CharField(max_length=50)
    result_json = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def get_result(self):
        return json.loads(self.result_json)

    def __str__(self):
        return f"{self.task_type} - {self.target_column}"


class ChatMessage(models.Model):
    """Stores chatbot conversation history"""
    session_key = models.CharField(max_length=100)
    role = models.CharField(max_length=10, choices=[('user', 'User'), ('ai', 'AI')])
    message = models.TextField()
    context_type = models.CharField(max_length=30, blank=True)  # dataset, ml_result, general
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.message[:50]}"
