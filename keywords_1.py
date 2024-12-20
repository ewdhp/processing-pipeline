import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
vectorizer = TfidfVectorizer()

# Preprocessing Functions
async def tokenize(text, **kwargs):
    return text.split()

async def normalize(tokens, **kwargs):
    return [token.lower() for token in tokens]

async def remove_stopwords(tokens, **kwargs):
    return [token for token in tokens if token not in stop_words]

async def stem(tokens, **kwargs):
    return [stemmer.stem(token) for token in tokens]

async def lemmatize(tokens, **kwargs):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

# Analysis Functions
async def keyword_extraction(tokens, method="tfidf", top_n=10):
    if method == "tfidf":
        tfidf_matrix = vectorizer.fit_transform([' '.join(tokens)])
        feature_names = vectorizer.get_feature_names_out()
        top_keywords = feature_names[tfidf_matrix.toarray().argsort()[0, -top_n:]]
        return {"keywords": top_keywords}
