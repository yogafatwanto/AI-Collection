from django.contrib import admin
from django.urls import path, include
from AIFirstPaymentDefaultPrevention import views



urlpatterns = [
    path('', include('AIFirstPaymentDefaultPrevention.urls')),
    path("admin/", admin.site.urls),
    
]
