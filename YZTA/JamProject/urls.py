from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate-text/', views.generate_text, name='generate_text'),

]
