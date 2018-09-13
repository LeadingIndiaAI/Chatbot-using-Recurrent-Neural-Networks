#importing necessary libraries and classes
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import os
import sys
import zipfile

np.random.seed(42)

#Hyperparameter tuning 
BATCH_SIZE = 16
NUM_EPOCHS = 250
GLOVE_EMBEDDING_SIZE = 100
HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 30
MAX_TARGET_SEQ_LENGTH = 30
MAX_VOCAB_SIZE = 10000
DATA_SET_NAME = 'my_data'
DATA_DIR_PATH = 'E:/chatbot/ChatCrazie/my_data'
WEIGHT_FILE_PATH = 'E:/chatbot/ChatCrazie/support files/model-weights.h5'
GLOVE_MODEL = "E:/chatbot/ChatCrazie/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'

#defines the valid characters for the chatbot
def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True

    return False


#it is done to load the glove embeddings file     
def load_glove_embeddings():
    _word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em

word2em = load_glove_embeddings()

target_counter = Counter()

input_texts = []
target_texts = []

for file in os.listdir(DATA_DIR_PATH):
    filepath = os.path.join(DATA_DIR_PATH, file)
    if os.path.isfile(filepath):
        print('processing file: ', file)
        lines = open(filepath, 'rt', encoding='utf8').read().split('\n')
        prev_words = []
        for line in lines:

            if line.startswith('- - '):
                prev_words = []

            if line.startswith('- - ') or line.startswith('  - '):
                line = line.replace('- - ', '')
                line = line.replace('  - ', '')
                next_words = [w.lower() for w in nltk.word_tokenize(line)]
                next_words = [w for w in next_words if in_white_list(w)]
                if len(next_words) > MAX_TARGET_SEQ_LENGTH:
                    next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

                if len(prev_words) > 0:
                    input_texts.append(prev_words)

                    target_words = next_words[:]
                    target_words.insert(0, 'start')
                    target_words.append('end')
                    for w in target_words:
                        target_counter[w] += 1
                    target_texts.append(target_words)

                prev_words = next_words

for idx, (input_words, target_words) in enumerate(zip(input_texts, target_texts)):
    if idx > 10:
        break
    print([input_words, target_words])

target_word2idx = dict()
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 1

if 'unknown' not in target_word2idx:
    target_word2idx['unknown'] = 0

target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

num_decoder_tokens = len(target_idx2word)

np.save('E:/chatbot/ChatCrazie/support files/target-word2idx.npy', target_word2idx)
np.save('E:/chatbot/ChatCrazie/support files/target-idx2word.npy', target_idx2word)

input_texts_word2em = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(input_texts, target_texts):
    encoder_input_wids = []
    for w in input_words:
        emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
        if w in word2em:
            emb = word2em[w]
        encoder_input_wids.append(emb)

    input_texts_word2em.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

print(context)
np.save('E:/chatbot/ChatCrazie/support files/word-glove-context.npy', context)


def generate_batch(input_word2em_data, output_text_data):
    num_batches = len(input_word2em_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, GLOVE_EMBEDDING_SIZE))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = target_word2idx['unknown']  # default unknown
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    if w in word2em:
                        decoder_input_data_batch[lineIdx, idx, :] = word2em[w]
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

#Encoder layers,inputs,outputs
encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

#Decoder layers - input,output,LSTM,Dense
decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)
#model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
json = model.to_json()
open('E:/chatbot/ChatCrazie/support files/word-architecture.json', 'w').write(json)

#train_test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_texts_word2em, target_texts, test_size=0.2, random_state=42)

#execution and updation of wt in batches
train_gen = generate_batch(Xtrain, Ytrain)
test_gen = generate_batch(Xtest, Ytest)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

#saving of model at checks through checkpoint
checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

#fitting of chatbot moel
model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])

#final weights saved at desired location
model.save_weights(WEIGHT_FILE_PATH)
