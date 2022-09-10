def sample_token(logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

def generate_text(prefix='', length=20):
    decoded_sample = prefix
    for i in range(length-1):
        tokenized_prefix = vectorize_layer([decoded_sample])[:, :-1]
        predictions = model.predict([tokenized_prefix], verbose=0)
        sample_index = len(decoded_sample.strip().split())-1

        sampled_token = sample_token(predictions[0][sample_index])
        sampled_token = index[sampled_token]
        decoded_sample += " " + sampled_token
    return decoded_sample
#generate_text('The meaning of life is', 20)
#'The meaning of life is the nature and the nature it question in may sense is mixed of all world'
#'The meaning of life is the understanding may make a cognition of pure conceptions which are taken by means of an object in itself'
