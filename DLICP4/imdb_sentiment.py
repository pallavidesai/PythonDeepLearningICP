import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, SimpleRNN, LSTM
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer

imdb_df = pd.read_csv('./labeledTrainData.tsv', sep = '\t')
pd.set_option('display.max_colwidth', 500)

num_words = 10000
tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts( imdb_df.review )
sequences = tokenizer.texts_to_sequences(imdb_df.review)
y = np.array(imdb_df.sentiment)
print(y)
from keras.preprocessing.sequence import pad_sequences

max_review_length = 552

pad = 'pre'

X = pad_sequences(sequences,
                  max_review_length,
                  padding=pad,
                  truncating=pad)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2)
input_shape = X_train.shape
K.clear_session()

rnn_model = Sequential()
# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs

rnn_model.add(Embedding(num_words,
                        20,
                        input_length=max_review_length))

rnn_model.add(LSTM(32))
rnn_model.add(Dense(1))
rnn_model.add(Activation('sigmoid'))
rnn_model.summary()

rnn_model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs/{}",histogram_freq=0, write_graph=True, write_images=True)
rnn_history = rnn_model.fit(X_train,
                            y_train,
                            epochs=2,
                            batch_size=32,
                            validation_split=0.3,callbacks=[tensorboard])

twt = ['With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, dtype='int32', value=0)
print(twt)
sentiment = rnn_model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")



