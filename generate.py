# Install gensim - pip install gensim
import nltk
import matplotlib.pyplot as plt
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec
from wordcloud import WordCloud
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


#function def
def remove_stopwords(sentences): # remove all stopwords from the texts
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i]) 
        words = [t for t in words if t not in stopwords.words('english')]
        sentences[i] = ' '.join(words)
    return sentences
  
punct = ", . ? ! ( ) @ # $ % ^ & * [ ] { } \ | : ; < > / ` ~ - _ + =" #all punctuation symbols
punct = punct.split()
def remove_punctuation(sentences): # remove all punctuation of the texts
    for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i]) 
        words = [t for t in words if t not in punct]
        sentences[i] = ' '.join(words)
    return sentences
  

url = 'https://en.wikipedia.org/wiki/Global_warming' # given input of the passage (link rn for tester)
source = urllib.request.urlopen(url).read()
soup = bs.BeautifulSoup(source,"html.parser") 
text = ""
for paragraph in soup.find_all('p'): #The <p> tag defines a paragraph in the webpages
    text += paragraph.text

text = re.sub(r'\[[0-9]*\]',' ',text) # [0-9]* --> Matches zero or more repetitions of any digit from 0 to 9
text = text.lower() #everything to lowercase
text = re.sub(r'\W^.?!',' ',text) # \W --> Matches any character which is not a word character except (.?!)
text = re.sub(r'\d',' ',text) # \d --> Matches any decimal digit
text = re.sub(r'\s+',' ',text) # \s --> Matches any characters that are considered whitespace (Ex: [\t\n\r\f\v].)

all_sentences = nltk.sent_tokenize(text)
all_sentences = remove_punctuation(all_sentences)
all_sentences = remove_stopwords(all_sentences)
for i in range(len(all_sentences)):
    all_sentences[i]  = [t for t in nltk.word_tokenize(all_sentences[i])]
    
    
model = Word2Vec(sentences=all_sentences, min_count=1)
common_words = model.wv.index_to_key
model.save("word2vec.model")

wordcloud = WordCloud(  background_color='white',
                        max_words=100,
                        max_font_size=50, 
                        random_state=42
                        ).generate(str(all_sentences))
fig = plt.figure(1)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
