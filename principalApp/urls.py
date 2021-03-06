"""crepes_bretonnes URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/dev/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
[...]
"""
from django.contrib import admin
from django.urls import path
from principalApp import views

urlpatterns = [
    path('', views.index, name='index'),
    path('form', views.form, name='form'),
    path('download/<str:file_name>', views.download, name='download'),
]