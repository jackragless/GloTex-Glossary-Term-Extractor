from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.baseRedirect, name="baseRedirect"),
    path('document', views.document, name="document"),
    path('text', views.text, name="text"),
    path('ack', views.ack, name="ack"),
    path('download', views.download_file),
]
