import re, io, json, numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from keras.models import load_model

# Load models and tokenizers outside the function to avoid reloading them on each call
hate_model = load_model('hate_model.h5')
spam_model = load_model('spam_model.h5')

with open('tokenizer_hate.json') as f:
    tokenizer_hate = tokenizer_from_json(json.load(f))
with open('tokenizer_spam.json') as f:
    tokenizer_spam = tokenizer_from_json(json.load(f))

stopwords_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def processed(text) -> list:
    cont_patterns = [
        ('(W|w)on\'t', 'will not'),
        ('(C|c)an\'t', 'can not'),
        ('(I|i)\'m', 'i am'),
        ('(A|a)in\'t', 'is not'),
        ('(\w+)\'ll', '\g<1> will'),
        ('(\w+)n\'t', '\g<1> not'),
        ('(\w+)\'ve', '\g<1> have'),
        ('(\w+)\'s', '\g<1> is'),
        ('(\w+)\'re', '\g<1> are'),
        ('(\w+)\'d', '\g<1> would'),
    ]
    patterns = [(re.compile(regex), repl) for regex, repl in cont_patterns]
    text = text.lower()
    for pattern, repl in patterns: text = re.sub(pattern, repl, text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = ' '.join(text.split())
    text = [word for word in word_tokenize(text) if word not in stopwords_set]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [stemmer.stem(word) for word in text]
    return [text]

def get_ratings(comment_text):
    vec = processed(comment_text)
    vec_hate = np.array(tokenizer_hate.texts_to_sequences(vec))
    vec_spam = np.array(tokenizer_spam.texts_to_sequences(vec))
    vec_hate = pad_sequences(vec_hate, maxlen=2000)
    vec_spam = pad_sequences(vec_spam, maxlen=2000)
    hate_rating = 100 * hate_model.predict(vec_hate)[0,0]
    spam_rating = 100 * spam_model.predict(vec_spam)[0,0]
    print(f'{hate_rating:.2f}% hate')
    print(f'{spam_rating:.2f}% spam')
    return hate_rating, spam_rating



