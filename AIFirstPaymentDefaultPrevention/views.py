from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework import viewsets
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
# Create your views here.

import pandas as pd
import numpy as np
import joblib
import json  

from preprocessing.first_payment_default import data_transformation_pipeline, kredit_preprocessing, \
    solution_preprocessing, PotentialDefaultCalculator, CategoricalEncoder


@api_view(["POST"])
def predict_pinjaman(request):
    try:
        mydata = request.data
        unit = list(mydata.values())
        columns = ['debtor_name','debtor_age', 'debtor_education_level','employment_year',
                   'monthly_income', 'monthly_expenses', 'asset_value','loan_amount',
                   'interest_rate', 'tenor', 'monthly_payment',
                    'loan_purpose']
       
        df = pd.DataFrame([unit], columns=columns)
        nama = unit[0]
        data_transformation_pipeline.fit_transform(df)
        df = kredit_preprocessing.fit_transform(df)
        df.drop(['debtor_name','debtor_education_level'], axis=1, inplace=True)
        new_column_order = ['employment_year', 'monthly_income', 'monthly_expenses', 'asset_value',
                            'loan_amount', 'interest_rate', 'tenor', 'monthly_payment',
                            'loan_income_expenses_ratio', 'default_risk', 'loan_purpose']

        kredit_df = df[new_column_order]
        
        scaler = joblib.load('savedmodel/kredit_pinjaman_scaler.joblib')
        scaled_df = scaler.transform(kredit_df)
        model = joblib.load('savedmodel/kredit_pinjaman.joblib')
        default_score = model.predict(scaled_df)
        df['default_score'] = default_score
        default_potential_cal = PotentialDefaultCalculator()
        default_potential_cal.fit_transform(df)
        solution_df  = df[['debtor_age', 'asset_value', 'loan_amount', 'loan_purpose', 'ses','default_risk', 'default_score', 'default_potential']]
        solution_df = solution_preprocessing.fit_transform(solution_df)
        cat_encoder = CategoricalEncoder()
        solution_df = cat_encoder.fit_transform(solution_df)
        solution_scaler = joblib.load('savedmodel/kredit_pinjaman_solution_scaler.joblib')
        solution_df = solution_scaler.transform(solution_df)
        solution_model = joblib.load('savedmodel/kredit_pinjaman_solution.joblib')
        solution_given = solution_model.predict(solution_df)


        default_score = float(df['default_score'].values[0])

        response_data = {
            "nama": nama,
            "default_score": round(default_score),
            "default_potential": df['default_potential'].values[0],
            "solution": solution_given[0],
            "status" : status.HTTP_200_OK
        }

        # Serialize the response data to JSON


        return Response(response_data, status = status.HTTP_200_OK)
    except ValueError as e:
        error_message = {
            "error" : str(e),
            "status": status.HTTP_400_BAD_REQUEST
        }
        return Response(error_message, status=status.HTTP_400_BAD_REQUEST)




@api_view(["POST"])
def predict_benda(request):
    try:
        mydata = request.data
        unit = list(mydata.values())
        columns = ['debtor_name','debtor_age', 'debtor_education_level','employment_year',
                   'monthly_income', 'monthly_expenses', 'asset_value','loan_amount',
                   'interest_rate', 'tenor', 'monthly_payment',
                    'loan_purpose']
       
        df = pd.DataFrame([unit], columns=columns)
        nama = unit[0]
        data_transformation_pipeline.fit_transform(df)
        df = kredit_preprocessing.fit_transform(df)
        df.drop(['debtor_name','debtor_education_level'], axis=1, inplace=True)
        new_column_order = ['employment_year', 'monthly_income', 'monthly_expenses', 'asset_value',
                            'loan_amount', 'interest_rate', 'tenor', 'monthly_payment',
                            'loan_income_expenses_ratio', 'default_risk', 'ses']

        kredit_df = df[new_column_order]
        
        scaler = joblib.load('savedmodel/kredit_benda_scaler.joblib')
        scaled_df = scaler.transform(kredit_df)
        model = joblib.load('savedmodel/kredit_benda.joblib')
        default_score = model.predict(scaled_df)
        df['default_score'] = default_score
        default_potential_cal = PotentialDefaultCalculator()
        default_potential_cal.fit_transform(df)
        solution_df  = df[['debtor_age', 'asset_value', 'loan_amount', 'loan_purpose', 'ses','default_risk', 'default_score', 'default_potential']]
        solution_df = solution_preprocessing.fit_transform(solution_df)
        cat_encoder = CategoricalEncoder()
        solution_df = cat_encoder.fit_transform(solution_df)
        solution_scaler = joblib.load('savedmodel/kredit_benda_solution_scaler.joblib')
        solution_df = solution_scaler.transform(solution_df)
        solution_model = joblib.load('savedmodel/kredit_benda_solution.joblib')
        solution_given = solution_model.predict(solution_df)


        default_score = float(df['default_score'].values[0])

        response_data = {
            "nama": nama,
            "default_score": round(default_score),
            "default_potential": df['default_potential'].values[0],
            "solution": solution_given[0],
            "status" : status.HTTP_200_OK
        }

        # Serialize the response data to JSON


        return Response(response_data, status = status.HTTP_200_OK)
    except ValueError as e:
        error_message = {
            "error" : str(e),
            "status": status.HTTP_400_BAD_REQUEST
        }
        return Response(error_message, status=status.HTTP_400_BAD_REQUEST)
