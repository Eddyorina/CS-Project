# pylint: disable=invalid-name
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from tensorflow import keras as tf
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def create_model(ratings):
    '''Returns a model from the ratings given by users on architects'''
    n_architects = ratings.architect_id.nunique()
    n_users = ratings.user_id.nunique()

    input_architects = tf.layers.Input(shape=[1])
    embed_architects = tf.layers.Embedding(n_architects + 1, 15)(input_architects)
    architects_out = tf.layers.Flatten()(embed_architects)

    input_users = tf.layers.Input(shape=[1])
    embed_users = tf.layers.Embedding(n_users + 1, 15)(input_users)
    users_out = tf.layers.Flatten()(embed_users)

    conc_layer = tf.layers.Concatenate()([architects_out, users_out])
    x = tf.layers.Dense(128, activation='relu')(conc_layer)
    x_out = x = tf.layers.Dense(1, activation='relu')(x)

    # Create model
    model = tf.Model([input_architects, input_users], x_out)
    opt = tf.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model

def create_model_checkpoints(checkpoint_path):
    '''Create checkpoints so that model is not trained again'''
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create callback to save model weights
    return tf.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                        save_weights_only=True,
                                        verbose=1)

def train_model(model, Xtrain, Xtest, cp_callback):
    '''Train given model against test data'''
    return model.fit([Xtrain.architect_id, Xtrain.user_id], Xtrain.rating,
                     batch_size=64,
                     epochs=5,
                     verbose=1,
                     validation_data=([Xtest.architect_id, Xtest.user_id], Xtest.rating),
                     callbacks=[cp_callback])

def evaluate_model(hist):
    '''Evaluate model and reurn train and validation loss'''
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    return train_loss, val_loss

def plot_model(train_loss, val_loss):
    '''Plot metrics on model performance'''
    plot.plot(train_loss, color='r', label='Train Loss')
    plot.plot(val_loss, color='b', label='Validation Loss')
    plot.title('Train and Validation Loss Curve')
    plot.legend()
    plot.show()

def save_unique_architect_ids(ratings, architects, architect_em_weights):
    '''Save unique architect ids to file'''
    # Get architect titles
    architects_copy = architects.copy()
    architects_copy = architects_copy.set_index('architect_id')

    # Save all the unique architect ids to file
    architect_ids = list(ratings.architect_id.unique())
    architect_ids.remove(10000)
    dict_map = {}
    for i in architect_ids:
        dict_map[i] = architects_copy.iloc[i]['name']

    out_v = open('model/vsecs.tsv', 'w')
    out_m = open('model/meta.tsv', 'w')
    for i in architect_ids:
        architect = dict_map[i]
        embeddings = architect_em_weights[i]
        out_m.write(architect+ '\n')
        out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')

    out_v.close()
    out_m.close()

    return architect_ids

def create_recommendations(model, user_id, architect_ids):
    '''Create recommendations for given user'''
    architects_array = np.array(architect_ids)
    architect_ids_length = len(architect_ids)
    user = np.array([user_id for i in range(architect_ids_length)])
    return model.predict([architects_array, user])

def get_top_x_recommendations(predictions, top_x):
    '''Return the top `x` recommendations'''
    # Reshape predictions to a single dimension
    prediction_ids = predictions.reshape(-1)
    # Sort and return top x recommendations
    return (-prediction_ids).argsort()[0:top_x]

# Import datasets
architect_ratings = pd.read_csv('data/architect_ratings.csv')
architect_data = pd.read_csv('data/architects.csv')

# Split data into train and test sets
Xtrain, Xtest = train_test_split(architect_ratings, test_size=0.2, random_state=1)

# Create and train model if it does not exist
if os.path.exists('model'):
    model = tf.models.load_model('model')
else:
    model = create_model(architect_ratings)
    # Train the model
    train_model(model, Xtrain, Xtest, create_model_checkpoints('models/checkpoint.ckpt'))

# Display model architecture
model.summary()

# Save model
model.save('model')

# Extract embeddings
architect_embeddings = model.get_layer('embedding')
architect_embedding_weights = architect_embeddings.get_weights()[0]
# print(architect_embedding_weights.shape)

# Save unique architect ids to file
architect_ids = save_unique_architect_ids(architect_ratings, architect_data, architect_embedding_weights)

# Get the top 5 recommedations for user 100
predictions = create_recommendations(model, 100, architect_ids)
prediction_ids = get_top_x_recommendations(predictions, 5)
print(architect_data.iloc[prediction_ids])

# Save the  recommendations to json
json_architect_data = architect_data[['architect_id', 'Name', 'WorkPhone', 'Country']]
json_architect_data = json_architect_data.sort_values('architect_id')
json_architect_data.to_json(r'json_architect_data.json', orient='records')
