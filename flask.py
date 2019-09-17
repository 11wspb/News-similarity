import re
import numpy as np
import pandas as pd
import string, csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # ImportLibrary untuk Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import math

class Sistem:
	factory = None
	stemmer = None

	def __init__(self):
		#Inialisasi Stemming
		self.factory = StemmerFactory()
		self.stemmer = self.factory.create_stemmer()		

	#Tahap Preprosessing
	def clean_text(self, text):
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
	    temp = [self.stemmer.stem(stripped)]
	    #StopwordsRemoval
	    stop_words = set(stopwords.words('indonesian'))
	    temp = [j for i in temp for j in i.split() if j not in stop_words]
	    temp = ' '.join(temp)
	    return temp

	# Tahap proses judul
	def proses_judul(self, judul):
		judul = clean_text(judul)
		hasil = str(judul)
		# print(hasil)
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
		# print(tampung_judul)

	# Tahap proses isi
	def proses_isi(self, isi):
		isi = clean_text(isi)
		hasil = str(isi)
		remove = hasil.replace(':', '')
		remove = re.sub(r'\d+', '', remove)
		words = word_tokenize(remove)

		stop_words = set(stopwords.words('indonesian'))

		tampung_isi = []
		for y in words:
			if y not in stop_words:
				tampung_isi.append(y)
		return tampung_isi
		# print(tampung_isi)

	# Proses mencari makna kata
	def mencari_makna(judul, isi):
		judul = proses_judul(judul)
		# print(judul)
		isi_berita = proses_isi(isi)
		# print(isi_berita)
		synonyms = []
		result = []
		# hasil = []
		list_sinonim = []
		for i in range(0, len(judul)):
			kata = judul[i]
			for syn in wn.synsets(kata, lang="ind"):
				for l in syn.lemmas(lang="ind"):
					hasil1 = str(l.name())
					stem = [stemmer.stem(hasil1)]
					stop_words = set(stopwords.words('indonesian'))
					temp = [j for i in stem for j in i.split() if
							j not in stop_words]  # loop setiap kata displit dgn space, dan jika tdk termasuk stopword, maka tidak masuk divariabel temp
					temp = ' '.join(temp)
					pisah_kata = word_tokenize(temp)
					for z in range(len(pisah_kata)):
						synonyms.append(pisah_kata[z])

			for word in synonyms:
				if word not in result:
					result.append(word)

			list_sinonim.append([])
			list_sinonim[i].append(judul[i])
			for j in range(len(result)):
				list_sinonim[i].append(result[j])

			synonyms = []
			result = []

		for a in range(len(isi_berita)):
			for b in range(len(list_sinonim)):
				for j in range(len(list_sinonim[b])):
					if list_sinonim[b][j] == isi_berita[a]:
						isi_berita[a] = list_sinonim[b][0]

		isi_bersih = ''
		for i in range(len(isi_berita)):
			if (i == 0):
				isi_bersih = isi_bersih + str(isi_berita[i])
			else:
				isi_bersih = isi_bersih + ' ' + str(isi_berita[i])

		return isi_bersih

	#Tahap hitung Cosine
	def cosine_sim(self, text1, text2):
	    vectorizer = TfidfVectorizer(analyzer='word')
	    train_vectors = vectorizer.fit_transform([text1, text2])
	    #print(train_vectors)
	    test_vectors = vectorizer.transform([text1, text2])
	    return ((train_vectors * train_vectors.T).A)[0,1]

	def checkup_single(self, params):
		simpan = []
		for index, row in text.iterrows():
			judul = row["Judul"]
			isi = row["Isi"]
			# print(f"{index:>4} - {judul[0:40]} - {isi[0:50]}")
			judul_clean = clean_text(judul)
			judul_berita = proses_judul(judul)
			# print(f"token judul berita:\n {judul_berita}")
			isi_berita = proses_isi(isi)
			# print(f"hasil proses_isi: \n {isi_berita} !!")
			makna = mencari_makna(judul_clean, isi)

			simpan.append(makna)

		judul = text['Judul'].apply(clean_text)
		isi = pd.DataFrame(simpan)

		judul = self.clean_text(params.get('judul'))
		isi = self.clean_text(params.get('isi'))

		# format response
		fmt_response = {}

		# empty output
		hasil = []
		hasil_cosine = self.cosine_sim(params.get('judul'), params.get('isi'))
		#hasil_cosine = self.cosine_sim(judul, isi)
		hasil.append(hasil_cosine)

		#y_pred = []
		for data in hasil:
		    if data > 0.4 :
		        fmt_response['status'] = 'Non-clickbait'
		        #temp = 0
		    else :
		        fmt_response['status'] = 'Clickbait'
				#temp = 1
		fmt_response['procentage'] = math.trunc(data*100)
		
		return fmt_response
