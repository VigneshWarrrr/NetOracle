from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("alerts", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="alert",
            name="severity",
            field=models.CharField(default="MEDIUM", max_length=10),
        ),
    ]
