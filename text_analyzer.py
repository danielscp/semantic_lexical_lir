
from flashtext import KeywordProcessor
import pandas as pd
import nltk
import string
import unidecode

def load_thesaurus(thesaurus_file):
  df = pd.read_csv(thesaurus_file)
  df.fillna('', inplace=True)
  thesaurus = KeywordProcessor()
  thesaurus.add_keywords_from_list(list(df['name'].values))
  
  def use(term):
    u = df[df.name == term]['USE']
    if len(u) == 0 or u.values[0] == '':
        return term
    else:
        return u.values[0]
  
  def transform(txt):
    terms = thesaurus.extract_keywords(txt)
    terms = [use(t) for t in terms]
    return terms
  
  thesaurus.transform = transform
  return thesaurus


thesaurus = load_thesaurus()

def load_stopwords_processor(stopwords_file):
  pt_chars = set(list('áãâéêíóõôúç'))
  kp = KeywordProcessor()
  kp.non_word_boundaries = kp.non_word_boundaries | pt_chars
  stopwords = [n.strip() for n in open(stopwords_file)]
  for s in stopwords: kp.add_keyword(s, ' ')
  for s in nltk.corpus.stopwords.words('portuguese'): kp.add_keyword(s, ' ')
    
  def transform(txt):
    return " ".join(kp.replace_keywords(txt).split())

  kp.transform = transform
  return kp

stopwords = load_stopwords_processor()

def remove_digits_punct(s):
    """
    Removes all digits, punctuation and others characters
    :param s: The string to remove punctuation
    :return: The string without punctuation
    """

    remove_digits = str.maketrans('', '', string.digits)
    s = s.translate(remove_digits)
    punctuations = string.punctuation + '”“\'ªº–§˚'
    remove_punct = str.maketrans('', '', punctuations)
    s = s.translate(remove_punct)
    return s

stemmer = nltk.stem.RSLPStemmer()
def stem(word):
  return unidecode.unidecode(stemmer.stem(word))
[stem(w) for w in "habeas corpus homicídio arma de fogo impedimento homicídio culposo".split()]

def analyze(text):
  text = stopwords.transform(text)
  terms = thesaurus.transform(text)
  terms = [t for t in terms if len(t.split()) > 1]
  terms = [" ".join(stem(w) for w in t.split()) for t in terms]
  text = remove_digits_punct(text.lower())
  text = ' '.join([stem(w) for w in text.split()])
  text = [w for w in text.split() if len(w) > 2]
  text += terms
  return text


