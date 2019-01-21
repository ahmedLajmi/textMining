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
from django.utils.encoding import smart_str
from wsgiref.util import FileWrapper



from django.http import HttpResponse


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

two_plus_letters_RE = re.compile(r"(\w)\1{1,}", re.DOTALL)
three_plus_letters_RE = re.compile(r"(\w)\1{2,}", re.DOTALL)
two_plus_words_RE = re.compile(r"(\w+\s+)\1{1,}", re.DOTALL)

with open('model.pkl', 'rb') as file:
    nb_model, bow_model_char = pickle.load(file)

with open('model2.pkl', 'rb') as file2:
    NB_model, bow_model_ara, NB_model_tun, bow_model_tun = pickle.load(file2)

def index(request):
    return render(request, 'index.html')

def form(request):
    results = []
    if request.method == 'POST':
        if "file" in request.FILES:
            with open(os.path.join(BASE_DIR, 'test.txt'), 'wb+') as destination:
                for chunk in request.FILES["file"].chunks():
                    destination.write(chunk)
        else:
            with open(os.path.join(BASE_DIR, 'test.txt'), 'wb+') as destination:
                destination.write(request.POST["manuel"].encode('utf-8'))
        with open(os.path.join(BASE_DIR, 'result.txt'), 'w',encoding="utf-8") as result_file:
            cpt = 1
            with open(os.path.join(BASE_DIR, 'test.txt'), 'r', encoding="utf-8") as destination:
                for line in destination:
                    if (line != ''):
                        print(line)
                        clean_text = cleanup_text(line)
                        norma_text = norm_text(clean_text)
                        stemmer = ArabicStemmer()
                        stem_text = stemmer.stem(norma_text)

                        if ((clean_text == 'barcha nwamer') or (clean_text == 'barcha latin')):
                            results.append({"line": line, "lang": "Other", "proba": "Unknown"})
                        else:
                            text = bow_model_char.transform([stem_text])
                            proba = nb_model.predict_proba(text)[0]
                            if (proba[0] <= 0.80 and (proba[1] >= 0.15 and proba[1] <= 0.22)):
                                lang = "Other"
                            else:
                                if nb_model.predict(text)[0] == "ARA":
                                    lang = "Arabic"
                                else:
                                    lang = "Tunisian"
                            sentiment = sentimentAnalyse(lang,stem_text)
                            results.append({"line": line, "lang": lang, "proba":sentiment })
                    result_file.write("Line n : " + str(cpt) + " : \n " + line + "\n" )
                    result_file.write("\t Language : " + lang + "\n")
                    result_file.write("\t Sentiment : " + sentiment + "\n")
                    cpt += 1

    return render(request, 'result.html',{"results":results})

def cleanup_text(text):
    cpt = 0
    cpt1 = 0
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
        if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'):
            cpt1 = cpt1 + 1
    if (cpt > 0.4 * len(text)):
        text = 'barcha nwamer'
    if (cpt1 > 0.5 * len(text)):
        text = 'barcha latin'


    return text

def norm_text(text):
    text = re.sub(r'[أءؤاٍ]','ا',text)
    text = re.sub('ظ', 'ض', text)
    text = re.sub('ة', 'ت', text)
    return text

def sentimentAnalyse(language,text):
    if (language == "Arabic"):
        emotion = NB_model.predict_proba(bow_model_ara.transform([text]))[0]
        if (emotion[0] < 0.6 and emotion[0] > 0.4):
            return "Neutre"
        elif (emotion[0] >= 0.6):
            return "Negative"
        else:
            return "Positive"
    else:
        emotion = NB_model_tun.predict_proba(bow_model_tun.transform([text]))[0]
        if (emotion[0] < 0.6 and emotion[0] > 0.4):
            return "Neutre"
        elif (emotion[0] >= 0.6):
            return "Negative"
        else:
            return "Positive"

def download(request):
    file_path = os.path.join(BASE_DIR, 'result.txt',)

    if os.path.exists(file_path):
        with open(os.path.join(BASE_DIR, 'result.txt'), 'rb') as destination:
            response = HttpResponse(
                destination.read(),
                content_type='application/force-download')  # mimetype is replaced by content_type for django 1.7
            response['Content-Disposition'] = 'attachment; filename=%s' % smart_str("result.txt")

            response['X-Sendfile'] = smart_str(file_path)
            # It's usually a good idea to set the 'Content-Length' header too.
            # You can also set any other required headers: Cache-Control, etc.
        return response
