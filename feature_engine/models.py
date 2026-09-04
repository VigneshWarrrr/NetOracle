from django.db import models


class FeatureWindow(models.Model):
	"""A live output window produced by the network feature engine."""

	stream_key = models.CharField(max_length=255, default='global', db_index=True)
	features = models.JSONField()
	created_at = models.DateTimeField(auto_now_add=True, db_index=True)

	class Meta:
		ordering = ['created_at']

	def __str__(self):
		return f'{self.stream_key} @ {self.created_at}'
