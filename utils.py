
from sentence_transformers import SentenceTransformer
import re
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score

def replaceAtUser(text):
    """ Replaces "@user" with "" """
    text = re.sub('@[^\s]+|RT @[^\s]+','',text)
    return text

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '!', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '?', text)
    return text

def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

def removeNewLines(text):
    text = re.sub('\n', '', text)
    return text

def preprocess_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(removeUnicode(replaceURL(s)))))))

def preprocess_french_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(replaceURL(s))))))

def SBERT_embed(s_list, language = 'English'):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    output: the embeddings of the sentences/ tokens.
    '''
    if language == 'English':
        model = SentenceTransformer('all-MiniLM-L6-v2') # for English
    elif language == 'French':
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1') # for French
    embeddings = model.encode(s_list, convert_to_tensor = True, normalize_embeddings = True)
    return embeddings.cpu()

def evaluate(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari

def decode(division):
    if type(division) is dict:
        prediction_dict = {m: event for event, messages in division.items() for m in messages}
    elif type(division) is list:
        prediction_dict = {m: event for event, messages in enumerate(division) for m in messages}
    prediction_dict_sorted = dict(sorted(prediction_dict.items()))
    return list(prediction_dict_sorted.values())
