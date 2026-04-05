# healthcare_app/migrations/0002_datasession_data_json.py
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('healthcare_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='datasession',
            name='data_json',
            field=models.TextField(default='[]'),
        ),
    ]
