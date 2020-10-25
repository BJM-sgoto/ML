import numpy as np	
import pandas as pd 
import re		
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('Reviews.csv', nrows=100000)
data.drop_duplicates(subset=['Text'],inplace=True)
data.dropna(axis=0,inplace=True)

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not","he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is","I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",	"i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam","mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def text_cleaner(text):
	newString = text.lower()
	newString = re.sub(r'\([^)]*\)', '', newString)
	newString = re.sub('"','', newString)
	newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])		
	newString = re.sub(r"'s\b","",newString)
	newString = re.sub("[^a-zA-Z]", " ", newString) 
	tokens = newString.split()
	long_words=[]
	for i in tokens:
			if len(i)>=3:									#removing short word
					long_words.append(i)	 
	return (" ".join(long_words)).strip()

def summary_cleaner(text):
	newString = re.sub('"','', text)
	newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])		
	newString = re.sub(r"'s\b","",newString)
	newString = re.sub("[^a-zA-Z]", " ", newString)
	newString = newString.lower()
	tokens=newString.split()
	newString=''
	for i in tokens:
		if len(i)>1:																 
			newString=newString+i+' '	
	return newString

max_len_text=80 
max_len_summary=10

cleaned_text = []
for t in data['Text']:
	new_t = text_cleaner(t)
	if len(new_t.split())>max_len_text:
		new_t = np.nan
	cleaned_text.append(new_t)

cleaned_summary = []
for t in data['Summary']:
	new_t = summary_cleaner(t)
	if len(new_t.split())>max_len_summary:
		new_t = np.nan
	cleaned_summary.append(new_t)
		
data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)
data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x : 'START_ '+ x + ' _END')

x_tr,x_val,y_tr,y_val=train_test_split(data['cleaned_text'],data['cleaned_summary'],test_size=0.1,random_state=0, shuffle=True) 
del data

#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

x_tr = x_tokenizer.texts_to_sequences(x_tr) 
x_val = x_tokenizer.texts_to_sequences(x_val)

x_voc_size = len(x_tokenizer.word_index) +1

#prepare a tokenizer for summary on training data 
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

y_tr = y_tokenizer.texts_to_sequences(y_tr) 
y_val =  y_tokenizer.texts_to_sequences(y_val) 

y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_val = pad_sequences(y_val, maxlen=max_len_summary, padding='post')

y_voc_size = len(y_tokenizer.word_index) +1

latent_dim = 128

# encoder
encoder_input = Input(shape=(None,))
enc_emb = Embedding(x_voc_size, latent_dim)(encoder_input)

encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output, state_h, state_c = encoder_lstm3(encoder_output2)
print(encoder_output)
# decoder
decoder_input = Input(shape=(None,))
dec_emb = Embedding(y_voc_size, latent_dim)(decoder_input)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_output,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])
print(decoder_output)
