import re
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Bidirectional, GRU, Conv1D, GlobalMaxPooling1D, Dropout, TimeDistributed
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def toxiccomment_load():
    import keras
    model1 = keras.models.load_model("C:\\Users\MaSsS\Downloads\Try_model1.h5")

    train_df = pd.read_csv("C:\\Users\\MaSsS\\Downloads\\train.csv")
    test_df = pd.read_csv("C:\\Users\\MaSsS\\Downloads\\test.csv")

    MAX_SEQUENCE_LENGTH = 100
    MAX_NB_WORDS = 100000
    EMBEDDING_DIM = 50

    train_df = pd.read_csv("C:\\Users\\MaSsS\\Downloads\\train.csv")
    test_df = pd.read_csv("C:\\Users\\MaSsS\\Downloads\\test.csv")
    def get_pos_ratio(data):
        return data.sum() / len(data)

    from collections import defaultdict

    def clean_text(text, stem_words=False):
        special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
        replace_numbers=re.compile(r'\d+',re.IGNORECASE)

        # Clean the text, with the option to remove stopwords and to stem words.
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"i’m", "i am", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = replace_numbers.sub('', text)
        text = special_character_removal.sub('',text)
        
        return text


    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

    def index_encoding(sentences, raw_sent):
        word2idx = {}
        idx2word = {}
        ctr = 1
        for sentence in sentences:
            for word in sentence:
                if word not in word2idx.keys():
                    word2idx[word] = ctr
                    idx2word[ctr] = word
                    ctr += 1
        results = []
        for sent in raw_sent:
            results.append([word2idx[word] for word in sent])
        return results

def comment_toxic_predict(comment):
  CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
  ad=[comment]
  cla=[clean_text(text) for text in ad]
  cla = tokenizer.texts_to_sequences(cla)
  ada = pad_sequences(cla, maxlen=MAX_SEQUENCE_LENGTH)
  pred=model1.predict(ada, batch_size=256, verbose=1)
  df_pred=pd.DataFrame(data=pred, columns=CLASSES)
  sams = df_pred.iloc[0]
  if (sams[0]>0.5 or sams[1]>0.5 or sams[2]>0.5 or sams[3]>0.5 or sams[4]>0.5 or sams[5]>0.5):
    
    return print("\n Your comment is found Toxic\n ")
    #print("Given below gives your level of toxicity in u=your comment. Please be resposible while posting comment.")
    #print(df_pred)
  else:
    
    return print("\nposting comment\n")

    
    






