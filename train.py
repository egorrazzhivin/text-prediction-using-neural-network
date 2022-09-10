import tensorflow as tf
from tensorflow import keras
import keras_nlp
import numpy as np
import re
import random
from string import ascii_letters, whitespace
from tensorflow.keras.layers import TextVectorization

text_url_1 = 'https://www.gutenberg.org/files/4280/4280-h/4280-h.htm'
filepath_1 = keras.utils.get_file(f'The Critique of Pure Reason.txt', origin=text_url_1)
text = ''
with open(filepath_1, encoding='utf-8') as f:
    text = f.read()
    text = text[18000:-22000] #skip the description and conclusion
#text preprocessing
text = re.sub('<.*?>', '', text)
text = text.replace('\n', '')
text = text.replace('§', '')
text_list = text.split('.')
random.shuffle(text_list)
#len(max(text_list).split(' ')) #38 in this case

maxlen = 40
def custom_standardization(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, "\n", " ")
    return sentence
vectorize_layer = TextVectorization(
    standardize = custom_standardization,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_list)
vocab = vectorize_layer.get_vocabulary()
index = dict(zip(range(len(vocab)), vocab)) 

length = len(text_list)
train = text_list[:int(0.7*(length))]
test = text_list[int(0.7*(length)):]

batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices(train)
train_dataset = train_dataset.shuffle(buffer_size=256)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(test)
test_dataset = test_dataset.shuffle(buffer_size=256)
test_dataset = test_dataset.batch(batch_size)

def preprocess_text(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

train_dataset = train_dataset.map(preprocess_text)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_text)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


embed_dim = 128
num_heads = 4

def create_model():
    inputs = keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)(inputs)
    decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=embed_dim, 
                                                            num_heads=num_heads, 
                                                            dropout=0.5)(embedding_layer)
    
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer="adam", 
        loss='sparse_categorical_crossentropy',
        metrics=[keras_nlp.metrics.Perplexity(), 'accuracy']
    )
    return model
class text_generator(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens
        
    def sample_token(self, logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def on_epoch_end(self, epoch, logs=None):
        decoded_sample = self.start_prompt
        
        for i in range(self.max_tokens-1):
            tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
            predictions = self.model.predict([tokenized_prompt], verbose=0)
            sample_index = len(decoded_sample.strip().split())-1
            sampled_token = self.sample_token(predictions[0][sample_index])
            sampled_token = index[sampled_token]
            decoded_sample += " " + sampled_token
            
        print(f"\nSample text:\n{decoded_sample}...\n")
random_sentence = ' '.join(random.choice(test).split(' ')[:4]) 
sampler = text_generator(random_sentence, 30)
reducelr = keras.callbacks.ReduceLROnPlateau(patience=10, monitor='val_loss')

model =  create_model()
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=[sampler, reducelr])
