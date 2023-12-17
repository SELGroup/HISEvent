import pandas as pd
import numpy as np
import os
from os.path import exists
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import re
import pickle
import torch
from textblob import TextBlob

def get_offline_messages_embeddings():
     # load binary test mask
    mask_path = './data/event2012_offline/masks/'
    # load binary test mask
    test_mask = torch.load(mask_path + 'test_mask.pt').cpu().detach().numpy()
    # convert binary mask to indices
    test_mask = list(np.where(test_mask==True)[0])
    #print('test_mask: ', test_mask)
    #print('len(test_mask): ', len(test_mask)) # len(test_mask):  13769

    # ground_truth = [labels[i] for i in test_mask]
    # n_clusters = len(list(set(ground_truth)))
    # print('n_clusters: ', n_clusters) # n_clusters:  488

    # sample_ids = [tweet_ids[i] for i in test_mask]
    # print('len(sample_ids): ', len(sample_ids)) # len(sample_ids):  13769


    df_path_2 = './data/preprocessed_df_event2012.npy'
    df_np_2 = np.load(df_path_2, allow_pickle=True)
    df_2 = pd.DataFrame(data=df_np_2, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases'])
    # tweet_ids_2 = df_2['tweet_id'].values
    # labels_2 = df_2['event_id'].values
    # sample_ids_2 = [tweet_ids_2[i] for i in test_mask]
    # ground_truth_2 = [labels_2[i] for i in test_mask]
    sentences = df_2['processed_text'].values
    #print('sentences[:10]: ', sentences[:10])
    test_sentences = [sentences[i] for i in test_mask]

    '''
    # get embeddings of the test set messages
    embedding_path = './data/event2012_offline/test_set_embeddings/SBERT_embeddings.pkl'
    if not exists(embedding_path):
        SBERT_embeddings = SBERT_embed(test_sentences)
        with open(embedding_path, 'wb') as fp:
            pickle.dump(SBERT_embeddings, fp)
    else:
        with open(embedding_path, 'rb') as f:
            SBERT_embeddings = pickle.load(f)
    print('SBERT_embeddings.size(): ', SBERT_embeddings.size()) #torch.Size([13769, 384])
    '''

    # get embeddings of the test set messages
    embedding_path = './data/event2012_offline/test_set_embeddings/BERT_embeddings.pkl'
    if not exists(embedding_path):
        BERT_embeddings = BERT_embed(test_sentences)
        with open(embedding_path, 'wb') as fp:
            pickle.dump(BERT_embeddings, fp)
    else:
        with open(embedding_path, 'rb') as f:
            BERT_embeddings = pickle.load(f)
    print('BERT_embeddings.size(): ', BERT_embeddings.size()) # torch.Size([13769, 1024])
    return

def test_construct_graph():
    # load preprocessed df
    df_path = './data/preprocessed_df_event2012.npy'
    df_np = np.load(df_path, allow_pickle=True)
    df = pd.DataFrame(data=df_np, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases'])
    df = df.head(20)

    all_node_features = [[str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e + \
            n
            for u, um, hs, e, n in \
            zip(df['user_id'], df['user_mentions'], df['hashtags'], df['entities'], df['noun_phrases'])]
    print('all_node_features: ', all_node_features)

    src_id, dst_id = [], []
    for i in range(len(all_node_features)):
        for j in range(i + 1, len(all_node_features)):
            if set(all_node_features[i])&set(all_node_features[j]):
                src_id.append(i + 1) # note that the node indices starts from 1 other than 0
                dst_id.append(j + 1)

    df_graph = pd.DataFrame(list(zip(src_id, dst_id)),
        columns =['src_id', 'dst_id'])
    print('df_graph: ', df_graph)
    return

def construct_graph():
    '''
    Split the dataframe by dates (following KPGNN).
    For each date, construct a graph by connecting tweets that has common:
    1. user + user mentions
    2. hashtags (lower-cased)
    3. named entities
    4. noun phrases
    '''
    cos = torch.nn.CosineSimilarity(dim=0)
    root_path = './data/df_split/'
    embedding_root_path = './data/df_split/'

    # load preprocessed df
    df_path = './data/preprocessed_df_event2012.npy'
    df_np = np.load(df_path, allow_pickle=True)
    df = pd.DataFrame(data=df_np, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases'])

    # sort data by time
    df = df.sort_values(by='created_at').reset_index()
    # append date
    df['date'] = [d.date() for d in df['created_at']]

    # split the df by dates
    distinct_dates = df.date.unique()
    #print("Distinct dates: ", distinct_dates)

    # first week -> initial graph (20254 tweets)
    folder = root_path + '0/'
    if not exists(folder):
        os.mkdir(folder)

    # extract and save df slice
    df_np_path = folder + '0.npy'
    if not exists(df_np_path):
        ini_df = df.loc[df['date'].isin(distinct_dates[:7])]  # find top 7 dates
        ini_df_np = ini_df.to_numpy()
        np.save(df_np_path, ini_df_np)

    # constuct and save graph
    df_graph_path = folder + '0.txt'
    if not exists(df_graph_path):
        embedding_path = embedding_root_path + '0/embeddings_0.pkl'
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f) 
        assert len(embeddings) == ini_df.shape[0]

        all_node_features = [[str(u)] + \
            [str(each) for each in um] + \
            [h.lower() for h in hs] + \
            e + \
            n
            for u, um, hs, e, n in \
            zip(ini_df['user_id'], ini_df['user_mentions'], ini_df['hashtags'], ini_df['entities'], ini_df['noun_phrases'])]

        src_id, dst_id, weight = [], [], []
        for i in range(len(all_node_features)):
            for j in range(i + 1, len(all_node_features)):
                if set(all_node_features[i])&set(all_node_features[j]):
                    src_id.append(i + 1) # note that the node indices starts from 1 other than 0
                    dst_id.append(j + 1)
                    weight.append(cos(embeddings[i], embeddings[j]).item())

        df_graph = pd.DataFrame(list(zip(src_id, dst_id, weight)),
               columns =['src_id', 'dst_id', 'weight'])
        df_graph.to_csv(df_graph_path, sep='\t', index=False)
    
    #================================
    for i in range(7, len(distinct_dates) - 1):
        folder = root_path + str(i - 6) + '/'
        if not exists(folder):
            os.mkdir(folder)
        
        # extract and save df slice
        df_np_path = folder + str(i - 6) + '.npy'
        if not exists(df_np_path):
            incr_df = df.loc[df['date'] == distinct_dates[i]]
            incr_df_np = incr_df.to_numpy()
            np.save(df_np_path, incr_df_np)
        
        # constuct and save graph
        df_graph_path = folder + str(i - 6) + '.txt'
        if not exists(df_graph_path):
            embedding_path = embedding_root_path + str(i - 6) + '/embeddings_' + str(i - 6) + '.pkl'
            with open(embedding_path, 'rb') as f:
                embeddings = pickle.load(f) 
            assert len(embeddings) == incr_df.shape[0]

            all_node_features = [[str(u)] + \
                [str(each) for each in um] + \
                [h.lower() for h in hs] + \
                e + \
                n
                for u, um, hs, e, n in \
                zip(incr_df['user_id'], incr_df['user_mentions'], incr_df['hashtags'], incr_df['entities'], incr_df['noun_phrases'])]

            src_id, dst_id, weight = [], [], []
            for i in range(len(all_node_features)):
                for j in range(i + 1, len(all_node_features)):
                    if set(all_node_features[i])&set(all_node_features[j]):
                        src_id.append(i + 1) # note that the node indices starts from 1 other than 0
                        dst_id.append(j + 1)
                        weight.append(cos(embeddings[i], embeddings[j]).item())

            df_graph = pd.DataFrame(list(zip(src_id, dst_id, weight)),
                columns =['src_id', 'dst_id', 'weight'])
            df_graph.to_csv(df_graph_path, sep='\t', index=False)

    return

def preprocess_df():
    '''
    Extract useful fields from the raw dataframe. Append new feilds that are needed.
    '''
    p_part1 = './data/68841_tweets_multiclasses_filtered_0722_part1.npy'
    p_part2 = './data/68841_tweets_multiclasses_filtered_0722_part2.npy'
    store_path = './data/preprocessed_df_event2012.npy'

    df_np_part1 = np.load(p_part1, allow_pickle=True)
    df_np_part2 = np.load(p_part2, allow_pickle=True)
    df_np = np.concatenate((df_np_part1, df_np_part2), axis = 0)
    print("Loaded data.")
    df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
        "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
        "words", "filtered_words", "sampled_words"])
    print("Data converted to dataframe.")

    #df = df.head(20)
    df = df[['event_id', 'tweet_id', 'text', 'user_id', 'created_at', 'hashtags', 'user_mentions', 'entities']]
    df['processed_text'] = [preprocess_sentence(s) for s in df['text']] # hastags are kept (with '#' removed). RTs are removed. 
    df['noun_phrases'] = [extract_noun_phrases(s) for s in df['processed_text']]
    
    df_np = df.to_numpy()
    np.save(store_path, df_np)
    df_np = np.load(store_path, allow_pickle=True)
    df = pd.DataFrame(data=df_np, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', \
        'hashtags', 'user_mentions', 'entities', 'processed_text', 'noun_phrases'])
    print('df.head(10): ', df.head(10))
    #print('df: ', df)
    #print('df[text]: ', df['text'])
    #print('df[processed_text]: ', df['processed_text'])
    #print('df[noun_phrases]: ', df['noun_phrases'])
    #print('df[hashtags]: ', df['hashtags'])

def extract_noun_phrases(s):
    '''
    Take a cleaned sentence (string) and return a list of noun phrases contained in this sentence.
    '''
    blob = TextBlob(s)
    #print(blob.noun_phrases)
    noun_phrases = blob.noun_phrases
    return noun_phrases

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

def SBERT_embed(s_list):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    output: the embeddings of the sentences/ tokens.
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(s_list, convert_to_tensor = True, normalize_embeddings = True)
    return embeddings.cpu()

def BERT_embed(sentences):
    '''
    Use BERT to embed sentences.
    sentences: a list of sentences/ tokens to be embedded.
    output: [CLS] (pooler_output) of the embedded sentences/ tokens.
    '''
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    BERT_model = AutoModel.from_pretrained("bert-large-cased")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    #BERT_model = AutoModel.from_pretrained("bert-base-cased")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #BERT_model = AutoModel.from_pretrained("bert-base-uncased")
    output = []
    for each in sentences:
        #print('each: ', each)
        inputs_ids = tokenizer(each, padding=True, truncation=True, return_tensors="pt")
        output.append(torch.squeeze(BERT_model(**inputs_ids)[1]).tolist()) # [CLS] (pooler_output)
    #print('---------- In BERT_embed ---------')
    result = torch.Tensor(output)
    #print(result)
    #print(result.size())
    return result

def test_BERT_embed():
    sentences = ['Do not go gentle into that good night.', 'Old age should burn and rave at close of day;', 'Rage, rage against the dying of the light.', \
        'Though wise men at their end know dark is right, Because their words had forked no lightning they do not go gentle into that good night.']
    result = BERT_embed(sentences)
    print('result: ', result)
    print('result.shape: ', result.shape) # torch.Size([4, 1024])
    return

if __name__ == '__main__':
    #preprocess_df()
    #test_construct_graph()
    construct_graph()
    get_offline_messages_embeddings()
    #test_BERT_embed()