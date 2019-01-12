from django.shortcuts import render
import os
import pickle
import pandas as pd
import numpy as np
from nltk.stem.snowball import ArabicStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import html

from django.http import HttpResponse


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

two_plus_letters_RE = re.compile(r"(\w)\1{1,}", re.DOTALL)
three_plus_letters_RE = re.compile(r"(\w)\1{2,}", re.DOTALL)
two_plus_words_RE = re.compile(r"(\w+\s+)\1{1,}", re.DOTALL)

with open('model.pkl', 'rb') as file:
    nb_model, bow_model_char = pickle.load(file)

def index(request):
    return render(request, 'index.html')

def form(request):
    results = []
    if request.method == 'POST':
        if request.FILES["file"] is not None:
            with open(os.path.join(BASE_DIR, 'test.txt'), 'wb+') as destination:
                for chunk in request.FILES["file"].chunks():
                    destination.write(chunk)
            with open(os.path.join(BASE_DIR, 'test.txt'), 'r', encoding="utf-8") as destination:
                for line in destination:
                    results.append({"line": line, "lang": nb_model.predict(bow_model_char.transform([cleanup_text(line)]))[0]})
    return render(request, 'result.html',{"results":results})

def cleanup_text(text):
    cpt = 0
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub('@[^\s]+', '', text)
    if re.search("&#",text):
        text = html.unescape(text)
    text = re.sub(r'_[xX]000[dD]_', '', text)
    text = re.sub('[\W_]', ' ', text)
    text = text.strip()
    text = re.sub('[\s]+', ' ', text)
    text = two_plus_letters_RE.sub(r"\1\1", text)
    text = two_plus_words_RE.sub(r"\1", text)
    for c in text:
        if c >= '0' and c <= '9':
            cpt = cpt + 1
    if cpt > 0.4 * len(text):
        text = ''
    if len(text) < 10:
        text = ''
    return text

def test(request):
    result = []
    if request.method == 'POST':
        if request.FILES["file"] is not None:
            with open(os.path.join(BASE_DIR, 'test.txt'), 'wb+') as destination:
                for chunk in request.FILES["file"].chunks():
                    destination.write(chunk)
            with open(os.path.join(BASE_DIR, 'test.txt'), 'r', encoding="utf-8") as destination:
                for line in destination:
                    result.append({"line":line,"lang":nb_model.predict(bow_model_char.transform([cleanup_text(line)]))})