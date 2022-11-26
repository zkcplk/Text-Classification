#!/usr/bin/env python
# coding: utf-8

# T.C. Marmara Üniversitesi
# Teknoloji Fakültesi | Bilgisayar Mühendisliği
# Zeki ÇIPLAK | 523622981

import numpy as np
import pandas as pd

# Mail mesajları ve o mesajların spam olup-olmadığını belirten 
# etiketlerin olduğu verisetini yüklüyoruz.
df = pd.read_csv("dataset.csv")

# Veri setine ilk bakış...
print(df.head())

# Veri setinde eksik değer olup-olmadığını kontrol ediyoruz.
print(df.isna().sum())

# Spam ve Spam olmayan (Ham) kaç mesaj var?
print(df['Category'].value_counts())

# Spam ve Spam olmayan mesajların sayısının eşit olması, 
# yapılan çalışmanın doğruluğu açısından önemli..
spam = df[df['Category'] == 'spam']
spamdegil = df[df['Category'] == 'ham']
spamdegil = spamdegil.sample(spam.shape[0])

# Artık spam ve spam olmayan mesajların sayısı eşitlendi.
print(spamdegil.shape, spam.shape)

# Yukarıda ayrı ayrı olan verileri tek bir dataframe'de birleştirelim.
# Eski dataframe'lerin sahip oldukları index numaralarını görmezden geliyoruz.
# Yeni oluşan dataframe'in kendine özel olarak, otomatik index numaraları oluşuyor.
data = pd.concat([spam, spamdegil], ignore_index = True)
print(data.head())

# Şimdi NLTK, doğal dil işleme kütüphanesine geçelim.
import nltk
import re
nltk.download('stopwords')
##########################
# Bu kısımdakiler, 
# çıkan error mesajlarından sonra eklendiler.
nltk.download('wordnet')
nltk.download('omw-1.4')
##########################

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer # kelime köküne indirme kütüphanesi

# One hot encoding ile spam ve spam olmayan kategorileri sayısal değerlere dönüştürelim.
# Makine öğrenmesi algoritmaları, sayısal olmayan verilerle çalışamazlar.
# spam=1, spamdegil=0
from sklearn.preprocessing import LabelEncoder
data['Category']=LabelEncoder().fit_transform(data['Category'])
print(data.head())

corpus=[]
datalen = len(data)
wordnet=WordNetLemmatizer()

for i in range(0,datalen):
    # mesajlardaki linklerin kaldırılması
    review = re.sub(r'https?://\S+|www.\S+', '', data["Message"][i])
    # mesajlardaki < ve > şeklinde html tarzı kodların kaldırılması
    review = review.replace('<', '')
    review = review.replace('>', '')
    # a-z ve A-Z arasındaki harflerin dışındaki harflerin " " ile değiştirilmesi
    review = re.sub(r'[^a-zA-Z]+', ' ', review)
    # Tüm sayısal ifadelerin tamamen kaldırılması
    review = re.sub(r'[0-9]', '', review)
    # Tüm karakterlerin küçük harfe çevrilmesi
    review = review.lower()
    # Mesajın boşluk karakterine (" ") parçalanması ve dizi haline getirilmesi
    review = review.split()
    # Dizi içerisinde mesajın parçalanmış kelimeleri var.
    # Her birinin öncelikle stopwords'ta olup olmadığına bakılıyor.
    # Ardından, her bir kelime lematizasyona tabi tutuluyor.
    review = [str(wordnet.lemmatize(word)) for word in review if not word in stopwords.words('english')]
    # Parça parça olan kelimeler, boşluk karakteri ile yeniden birleştiriliyor.
    review = ' '.join(review)
    # Temizlenen mesaj, tüm temizlenmiş mesajların bulunduğu listenin içine ekleniyor.
    corpus.append(review)

# Önişlemeden geçirilmiş tüm mesajlar
print(corpus)

# Bağımlı ve bağımsız değişken değerleri oluşturuluyor.
y=data["Category"] # bağımlı hedef değişken
X=pd.DataFrame(corpus,columns=['text']) # bağımsız değişken

print(X)
print(y)

# Tüm veriyi eğitim-test verisi olarak ikiye ayırıyoruz.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state = 42)

# Bir mesajda hangi kelimeden kaç adet olduğunu gösteren kelime vekörünü oluşturuyoruz.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000) # maksimum 3000 elemanlı bir vektör oluşturulacak.

train_X = cv.fit_transform(X_train['text']).toarray()
test_X  = cv.transform(X_test['text']).toarray()

print("Test ve Eğitim verilerinin boyutları: ")
print(train_X.shape)
print(test_X.shape)
print(y_train.shape)
print(y_test.shape)

# test_X ve train_X birer matristir.
# her biri, her kelimeden, hangi mesajda, kaç tane olduğunu gösterir.
# satırlar mesaj sayısıdır. sütunlar kelime sayısıdır.
# her kelime bir numara ile numaralandırılmıştır.
ornek = pd.DataFrame(test_X)
print(ornek)

# Şimdi makine öğrenmesi algoritmalarına geçelim.
# 1) RandomForest, 2) SVM, 3) Naive Bayes (GaussianNB ve MultinomialNB)

####################
# 1) RANDOM FOREST
####################
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=0).fit(train_X,y_train)
y_pred_rfc=rfc.predict(test_X)

# Accuracy (Doğruluk) hesabı
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print("Random Forest | Doğruluk Oranı (Accuracy): %", (accuracy_score(y_test,y_pred_rfc) * 100), sep="")

# Sadece doğruluk oranına bakmak, modelin doğruluğunu anlamamıza yetmez.
# Karmaşıklık matrisini de incelemek gerekir.

import matplotlib.pyplot as plt
import seaborn as sns

# Karmaşıklık Matrisinin oluşturulması
cm = confusion_matrix(y_test, y_pred_rfc)
sns.heatmap(cm, cmap='BuPu', annot=True,fmt='d')
print(cm)

# Precision, Recall ve F1-Score Hesaplamaları
from sklearn.metrics import precision_score, recall_score, f1_score

print("----RANDOM FOREST----")
print("Precision (macro): ", precision_score(y_test, y_pred_rfc, average='macro'))
print("Recall (macro): ", recall_score(y_test, y_pred_rfc, average='macro'))
print("F1 Score (macro): ", f1_score(y_test, y_pred_rfc, average='macro'))
print("----------------")
print("Precision (micro): ", precision_score(y_test, y_pred_rfc, average='micro'))
print("Recall (micro): ", recall_score(y_test, y_pred_rfc, average='micro'))
print("F1 Score (micro): ", f1_score(y_test, y_pred_rfc, average='micro'))
print("----------------")
print("Precision (weighted): ", precision_score(y_test, y_pred_rfc, average='weighted'))
print("Recall (weighted): ", recall_score(y_test, y_pred_rfc, average='weighted'))
print("F1 Score (weighted): ", f1_score(y_test, y_pred_rfc, average='weighted'))
print("----------------")
print("RandomForest, En iyi sonuçları vermiştir.")

###################################
# 2) SVM (Support Vector Machines)
###################################
from sklearn.svm import SVC
classifier = SVC(kernel='linear')
classifier.fit(train_X,y_train)
y_pred_svm=classifier.predict(test_X)

# Accuracy (Doğruluk) hesabı
print("SVM | Doğruluk Oranı (Accuracy): %", (accuracy_score(y_test,y_pred_svm) * 100), sep="")

# Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, cmap='BuPu', annot=True,fmt='d')
print(cm)

print("----SVM----")
print("Precision (macro): ", precision_score(y_test, y_pred_svm, average='macro'))
print("Recall (macro): ", recall_score(y_test, y_pred_svm, average='macro'))
print("F1 Score (macro): ", f1_score(y_test, y_pred_svm, average='macro'))
print("----------------")
print("Precision (micro): ", precision_score(y_test, y_pred_svm, average='micro'))
print("Recall (micro): ", recall_score(y_test, y_pred_svm, average='micro'))
print("F1 Score (micro): ", f1_score(y_test, y_pred_svm, average='micro'))
print("----------------")
print("Precision (weighted): ", precision_score(y_test, y_pred_svm, average='weighted'))
print("Recall (weighted): ", recall_score(y_test, y_pred_svm, average='weighted'))
print("F1 Score (weighted): ", f1_score(y_test, y_pred_svm, average='weighted'))
print("----------------")
print("SVM: Tüm sonuçlar, gayet başarılı")

###################################
# 3) NAIVE BAYES SINIFLANDIRMALAR
###################################
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train_X,y_train)
y_pred_gnb=gnb.predict(test_X)

mnb = MultinomialNB()
mnb.fit(train_X,y_train)
y_pred_mnb=mnb.predict(test_X)

# Accuracy 
print("GaussianNB | Doğruluk Oranı (Accuracy): %", (accuracy_score(y_test,y_pred_gnb) * 100), sep="")
print("MultinomialNB | Doğruluk Oranı (Accuracy): %", (accuracy_score(y_test,y_pred_mnb) * 100), sep="")

# Karmaşıklık Matrisi GaussianNB
cm = confusion_matrix(y_test, y_pred_gnb)
sns.heatmap(cm, cmap='BuPu', annot=True, fmt='d')
print(cm)

# Karmaşıklık Matrisi MultinomialNB
cm = confusion_matrix(y_test, y_pred_mnb)
sns.heatmap(cm, cmap='BuPu', annot=True, fmt='d')
print(cm)

print("----GaussianNB----")
print("Precision (macro): ", precision_score(y_test, y_pred_gnb, average='macro'))
print("Recall (macro): ", recall_score(y_test, y_pred_gnb, average='macro'))
print("F1 Score (macro): ", f1_score(y_test, y_pred_gnb, average='macro'))
print("----------------")
print("Precision (micro): ", precision_score(y_test, y_pred_gnb, average='micro'))
print("Recall (micro): ", recall_score(y_test, y_pred_gnb, average='micro'))
print("F1 Score (micro): ", f1_score(y_test, y_pred_gnb, average='micro'))
print("----------------")
print("Precision (weighted): ", precision_score(y_test, y_pred_gnb, average='weighted'))
print("Recall (weighted): ", recall_score(y_test, y_pred_gnb, average='weighted'))
print("F1 Score (weighted): ", f1_score(y_test, y_pred_gnb, average='weighted'))
print("----------------")
print("GaussianNB: En kötü sonuçlar!")

print("----MultinomialNB----")
print("Precision (macro): ", precision_score(y_test, y_pred_mnb, average='macro'))
print("Recall (macro): ", recall_score(y_test, y_pred_mnb, average='macro'))
print("F1 Score (macro): ", f1_score(y_test, y_pred_mnb, average='macro'))
print("----------------")
print("Precision (micro): ", precision_score(y_test, y_pred_mnb, average='micro'))
print("Recall (micro): ", recall_score(y_test, y_pred_mnb, average='micro'))
print("F1 Score (micro): ", f1_score(y_test, y_pred_mnb, average='micro'))
print("----------------")
print("Precision (weighted): ", precision_score(y_test, y_pred_mnb, average='weighted'))
print("Recall (weighted): ", recall_score(y_test, y_pred_mnb, average='weighted'))
print("F1 Score (weighted): ", f1_score(y_test, y_pred_mnb, average='weighted'))
print("----------------")
print("MultinomialNB: Sonuçlar gayet başarılı!")

print("*************")
print("En iyi sonucu veren algoritma RandomForest olmuştur.")
print("En kötü sonucu da GaussianNB vermiştir.")
print("*************")
print("Zeki ÇIPLAK")
