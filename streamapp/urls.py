from django.urls import path, include
from streamapp import views
from django.conf.urls import url, include

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('update', views.update, name="update")
    ]
