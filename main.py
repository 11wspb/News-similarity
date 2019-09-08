import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #ImportLibrary untuk Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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
    stripped = re.sub(r'\\n', '', stripped)
    #RemoveTags
    stripped = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", stripped)
    #Stemming
    temp = [stemmer.stem(stripped)]
    #StopwordsRemoval
    stop_words = set(stopwords.words('indonesian'))
    temp = [j for i in temp for j in i.split() if j not in stop_words] #looping setiap kata, dengan displit dgn spasi
    temp = ' '.join(temp)
    return temp

#Tahap hitung Cosine
def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(analyzer='word')
    train_vectors = vectorizer.fit_transform([text1, text2])
    #print(train_vectors)
    test_vectors = vectorizer.transform([text1, text2])
    return ((train_vectors * train_vectors.T).A)[0,1]
    
judul = text['Judul'].apply(clean_text)
isi = text['Isi'].apply(clean_text)
    

hasil = []
for i in range(0, len(text)):
    hasil_cosine = cosine_sim(text['Judul'].loc[i], text['Isi'].loc[i])
    hasil.append(hasil_cosine)
    
#y_pred = []
for data in hasil:
    if data > 0.2 :
        print(data, '- Non-clickbait')
        #temp = 0
    else :
        print(data, '- Clickbait')
        #temp = 1
    #y_pred.append(temp)
    
#print(y_pred)
