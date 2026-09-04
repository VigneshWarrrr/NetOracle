from django.db import models


class ForecastSettings(models.Model):
	"""Global runtime control for NetOracle forecast evaluation."""

	enabled = models.BooleanField(default=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		verbose_name = 'Forecast settings'
		verbose_name_plural = 'Forecast settings'

	def __str__(self):
		return 'Enabled' if self.enabled else 'Disabled'

	@classmethod
	def get_current(cls):
		settings, _ = cls.objects.get_or_create(pk=1)
		return settings

# Create 

