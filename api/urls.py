from django.conf.urls import url
import views


urlpatterns = [
    url(r'^train_model$', views.TrainModel.as_view(), name='trainmodel'),
    url(r'^predict$', views.CalculatePrediction.as_view(), name='predict'),
]
