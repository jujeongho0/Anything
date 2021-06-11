from django.db import models

# Create your models here.
class Photo(models.Model):
	photo = models.ImageField(upload_to='', blank=True, null=True)