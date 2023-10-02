from django.urls import path
from . import views

urlpatterns = [
    path('first_payment_default/kredit_pinjaman/', views.predict_pinjaman),
    path('first_payment_default/kredit_benda/', views.predict_benda),
    
]