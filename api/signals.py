"""
This file contains definitions of all functions called when a cretain
signal occurs
"""

from django.db.models.signals import pre_delete
from django.dispatch import receiver
from .models import ClassificationModel
import os




@receiver(pre_delete, sender=ClassificationModel)
def delete_classification_model(sender, **kwargs):
    """
    Deletes the classification model which is stored as a pickled file
    (A signal was required since pickled files cannot be save by Django ORM
    and needs to be done using pickle module itself)
    """
    
    filepath = sender.filepath
    os.remove(filepath)

