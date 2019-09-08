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

#definisi variabel
tampung_judul = []
list_makna = []

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
    #StopwordsRemoval
    stop_words = set(stopwords.words('indonesian'))
    temp = [j for i in temp for j in i.split() if j not in stop_words] #loop setiap kata displit dgn space, dan jika tdk termasuk stopword, maka tidak masuk divariabel temp
    temp = ' '.join(temp)
    return temp

#Tahap proses judul
def proses_judul(judul):
    judul = text['Judul'].apply(clean_text)
    hasil = str(judul)
    remove = hasil.replace(':', '')
    kalimat = remove.replace('dtype', '')
    kalimat = kalimat.replace('Name', '')
    kalimat = kalimat.replace('Judul', '')
    kalimat = kalimat.replace('object', '')
    remove = kalimat.replace(',', '')
    remove = re.sub(r'\d+', '', remove)
    words = word_tokenize(remove)
    stop_words = set(stopwords.words('indonesian'))
    for x in words:
        if x not in stop_words:
            tampung_judul.append(x)
    return tampung_judul

#Proses mencari makna kata
def mencari_makna(judulx):
    judul = judulx

    synonyms = []
    result = []
    for i in range(0, len(judul)):
        kata = judul[i]
        for syn in wordnet.synsets(kata, lang="ind"):
            for l in syn.lemmas(lang="ind"):
                synonyms.append(l.name())
    
    for word in synonyms:
        if word not in result:
            result.append(word)
    
    hasil = result + judul
    print(result)
        
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

hasil = []
judul_kata = mencari_makna(proses_judul(judul))
for i in range(0, len(text)):
    #Loc untuk mengakses baris dan kolom 
    hasil_cosine = cosine_sim(text['Judul'].loc[i], text['Isi'].loc[i])
    #print(hasil_cosine)
    #hasil.append(hasil_cosine)
