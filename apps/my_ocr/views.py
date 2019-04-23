import requests
import re
import base64
import json

from django.shortcuts import render
from django.http import HttpResponse
from main import Sign, Post, json2text

from django.shortcuts import render
from django.http import HttpResponse
from apps.my_ocr.load_model.load_model import rec_pic


# Create your views here.
def index(request):
    data = {'answer': 'no'}
    return render(request, 'ocr.html',data)


def upload(request):
    data = {}
    if request.method == 'GET':
        return HttpResponse('ERROR')
    elif request.method == 'POST':
        url_signal = int(request.POST['type'])
        obj = request.FILES.get('image')
        s = rec_pic(url_signal, obj)
        data['res'] = s
        data['answer'] = 'yes'
        return render(request, 'ocr.html',data)
