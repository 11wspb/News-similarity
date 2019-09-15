import re
import numpy as np
import pandas as pd
import string, csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #ImportLibrary untuk Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#Inialisasi Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

pd.set_option('display.max_colwidth', -2)
text = pd.read_csv("testing.csv")

#Tahap Preprosessing
def clean_text(text):
    #Tokenize
    words = word_tokenize(text.lower())
    temp = str(words)
    #RemoveNumber
    stripped = re.sub(r'\d+', '', temp)
    #RemoveKata(.com dan tanda -)
    stripped = re.sub(r'.com', '', stripped)
    stripped = re.sub(r'www', '', stripped)
    #RemovePunctuation
    #stripped = temp.translate(str.maketrans('', '', string.punctuation))
    #Stemming
    temp = [stemmer.stem(stripped)]
    #print(temp)
    #StopwordsRemoval
    stop_words = set(stopwords.words('indonesian'))
    temp = [j for i in temp for j in i.split() if j not in stop_words] #loop setiap kata displit dgn space, dan jika tdk termasuk stopword, maka tidak masuk divariabel temp
    temp = ' '.join(temp)
    return temp

#Tahap proses judul
def proses_judul(__judul):
    __judul = text['Judul'].apply(clean_text)
    hasil = str(__judul)
    #print(hasil)
    remove = hasil.replace(':', '')
    kalimat = remove.replace('dtype', '')
    kalimat = kalimat.replace('Name', '')
    kalimat = kalimat.replace('Judul', '')
    kalimat = kalimat.replace('object', '')
    remove = kalimat.replace(',', '')
    remove = re.sub(r'\d+', '', remove)
    words = word_tokenize(remove)
    stop_words = set(stopwords.words('indonesian'))
    
    tampung_judul = []
    for x in words:
        if x not in stop_words:
            tampung_judul.append(x)
    return tampung_judul
    #print(tampung_judul)

#def prosesJudul(str):
    #return str.split()

#Tahap proses isi
def prosesIsi(isi):
    hasil = str(isi)
    #print(hasil)
    remove = hasil.replace(':', '')
    kalimat = remove.replace('dtype', '')
    kalimat = kalimat.replace('Name', '')
    kalimat = kalimat.replace('Isi', '')
    kalimat = kalimat.replace('object', '')
    remove = kalimat.replace(',', '')
    remove = re.sub(r'\d+', '', remove)
    #print(remove)
    words = word_tokenize(remove)
    stop_words = set(stopwords.words('indonesian'))
    
    tampung_isi = []

    for y in words:
        if y not in stop_words:
            tampung_isi.append(y)
        
    return tampung_isi
    #print(tampung_isi)
    
#Proses mencari makna kata
def _mencari_makna(__judul, _isi):
    _judul = proses_judul(__judul)
    
    #print(_judul)
    _isi_berita = prosesIsi(_isi)
    
    #print(_isi_berita)
    
    synonyms = []
    result = []
    #hasil = []
    list_sinonim = []
    
    for i in range(0, len(_judul)):
        kata = _judul[i]
        for syn in wn.synsets(kata, lang="ind"):
            for l in syn.lemmas(lang="ind"):
                hasil1 = str(l.name())
                stem = [stemmer.stem(hasil1)]
                stop_words = set(stopwords.words('indonesian'))
                temp = [j for i in stem for j in i.split() if j not in stop_words] #loop setiap kata displit dgn space, dan jika tdk termasuk stopword, maka tidak masuk divariabel temp
                temp = ' '.join(temp)
                pisah_kata = word_tokenize(temp)
                for z in range(len(pisah_kata)):
                    synonyms.append(pisah_kata[z])
                
        for word in synonyms:
            if word not in result:
                result.append(word)
            
        list_sinonim.append([])
        list_sinonim[i].append(_judul[i])
        for j in range(len(result)):
            list_sinonim[i].append(result[j])
            
        synonyms = []
        result = []

    isi_bersih = ''
    for a in range(len(_isi_berita)):
        for b in range(len(list_sinonim)):
            for j in range(len(list_sinonim[b])):
                if list_sinonim[b][j] == _isi_berita[a]:
                    _isi_berita[a] = list_sinonim[b][0]

    for i in range(len(_isi_berita)):
        if (i == 0):
            isi_bersih = isi_bersih + str(_isi_berita[i])
        else:
            isi_bersih = isi_bersih +' '+ str(_isi_berita[i])

    return isi_bersih
    #print(isi_bersih)
    #return isi_bersih
    #print(isi_berita)
    #return isi_bersih
    #return list_berita[a]
    #print(isi_berita)
    #return hasil
    #return hasil
    #print(list_sinonim)
        
#Tahap hitung Cosine
def cosine_sim(text1, text2):
    #proses_judul(judul)
    vectorizer = TfidfVectorizer(analyzer='word')
    #TrainVectors
    train_vectors = vectorizer.fit([text1, text2])
    #TestVectors
    test_vectors = vectorizer.transform([text1, text2])
    #print(test_vectors)
    return ((test_vectors * test_vectors.T).A)[0,1] #[0,1] adalah posisi dalam matriks u/ kesamaan karena dua input akan membuat
    #matriks simetris 2x2

text['judul'] = text['Judul'].apply(clean_text)
#print(text['judul'])
isi = text['Isi'].apply(clean_text)
#print(isi)

passing_attr_judul = text['judul']
passing_attr_isi = isi
#print(passing_attr_isi)

isi_berita_final = ''
for i in range(len(passing_attr_isi)):
    text['news_clean'] = _mencari_makna(passing_attr_judul[i], passing_attr_isi[i])
    #text['news_clean'] = str(isi_berita_final)
    
print(text['news_clean'])
#text['isi_berita'] = _mencari_makna(judul, isi)
#print(passing_attr_judul)
#print(passing_attr_isi)

#text['isi_berita'] = _mencari_makna(prosesIsi(_isi))
#print(text['isi_berita'])
#text['isi_berita'] = mencari_makna(text['judul'], text['isi'])
#print(text['isi_berita'])
#print(judul_kata)
#text['isi_kata'] = mencari_makna(judul)
#isi_kata = mencari_makna(proses_isi(isi))
#print(judul_kata)

hasil = []

#for i in range(0, len(text)):
    #Loc untuk mengakses baris dan kolom 
    #hasil_cosine = cosine_sim(text['judul'].loc[i], text['isi_berita_final'].loc[i])
    #print(text['isi_berita_final'].loc[i])
    #print(text['isi_berita'].loc[i])
    #print(hasil_cosine)
    #hasil.append(hasil_cosine)
