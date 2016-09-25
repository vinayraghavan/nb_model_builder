from __future__ import unicode_literals

from django.db import models
from django.conf import settings
from django.conf import settings




class ClassificationModel(models.Model):
    """
    Details about a classification model trained using the user's 
    uploaded training data
    """
    
    model_name = models.CharField(max_length=30,null=True)
    selected_column = models.CharField(max_length=30,null=True)
    #column_states = models.CharField(max_length=1000,null=True)
    features = models.CharField(max_length=1000,null=True)
    date_created = models.DateTimeField(auto_now_add=True)
    #model_filepath = models.CharField(max_length=200)
    #training_file = models.FileField(upload_to=settings.BASE_DIR + "/media/training_files")
    
    def __unicode__(self):
        return str(self.id)
