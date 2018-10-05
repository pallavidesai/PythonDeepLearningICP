import urllib.request
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk,pos_tag, wordpunct_tokenize
from nltk.util import ngrams
# Parsing HTML Page using BeautifulSoup librry
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
source_code = urllib.request.urlopen(url)
soup = BeautifulSoup(source_code, "html.parser")
# Open File
f = open('input.txt','w',encoding='utf-8')
# Write to it
f.write(soup.text)
f.close()

pStemmer = PorterStemmer();
lemmatizer = WordNetLemmatizer();
with open("input.txt",encoding='utf-8') as f:
    for line in f:
        #Named Entuty Recognition
        print(ne_chunk(pos_tag(wordpunct_tokenize(line))))
        #Tokenization
        wtokens = nltk.word_tokenize(line)
        #Trigrams
        trigrams = ngrams(wtokens, 3)
        print([' '.join(grams) for grams in trigrams])
        print(wtokens)
        #POS
        print(nltk.pos_tag(wtokens))
        for word in wtokens:
            # Stemming
            print(pStemmer.stem(word))
            #Lemmatization
            print(lemmatizer.lemmatize(word))
f.close()
