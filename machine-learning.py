import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

rating = pd.read_csv('data/ratings.csv')
architect = pd.read_csv('data/architects.csv')
user = pd.read_csv('data/users.csv');
architect_rating = pd.merge(rating, architect, on='architect_id')
cols = ['Registration', 'Country', 'Address 2', 'Address 3', 'Company', 'WorkPhone', 'City', 'State', 'Postcode', 'Member Type']
architect_rating.drop(cols, axis=1, inplace=True)
architect_rating.head()

rating_count = (architect_rating.
                groupby(by = ['architect_id'])['rating'].
                count().
                reset_index().
                rename(columns = {'rating': 'rating_count'})
               )
rating_count.head()

threshold = 5
rating_count = rating_count.query('rating_count >= @threshold')
user_rating = pd.merge(rating_count, architect_rating, left_on='architect_id', right_on='architect_id', how='left')
user_count = (user_rating.
              groupby(by = ['user_id'])['rating'].
              count().
              reset_index().
              rename(columns = {'rating': 'rating_count'})
              [['user_id', 'rating_count']]
             )

threshold = 5
user_count = user_count.query('rating_count >= @threshold')
combined = user_rating.merge(user_count, left_on='user_id', right_on='user_id', how='inner')
print('Number of unique architects: ', combined['architect_id'].nunique())
print('Number of unique users: ', combined['user_id'].nunique())

scaler = MinMaxScaler()
combined['rating'] = combined['rating'].values.astype(float)
rating_scaled = pd.DataFrame(scaler.fit_transform(combined['rating'].values.reshape(-1,1)))
combined['rating'] = rating_scaled
combined.head()

combined = combined.drop_duplicates(['user_id', 'architect_id'])
user_architect_matrix = combined.pivot(index='user_id', columns='architect_id', values='rating')
user_architect_matrix.fillna(0, inplace=True)
users = user_architect_matrix.index.tolist()
architects = user_architect_matrix.columns.tolist()
#df.as_matrix() deprecated as of v0.23.0 using df.values
user_architect_matrix = user_architect_matrix.values

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

num_input = combined['architect_id'].nunique()
num_hidden_1 = 10
num_hidden_2 = 5

X = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op

y_true = X

loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()
print(pred_data)

with tf.Session() as session:
    epochs = 50
    batch_size = 10

    session.run(init)
    session.run(local_init)

    num_batches = int(user_architect_matrix.shape[0] / batch_size)
    print(num_batches)
    user_architect_matrix = np.array_split(user_architect_matrix, num_batches)

    for i in range (epochs):
        avg_cost = 0
        for batch in user_architect_matrix:
            _, l = session.run([optimizer, loss], feed_dict = {X: batch})
            avg_cost += 1

        avg_cost /= num_batches

        print("epoch: {} Loss: {}".format(i+1, avg_cost))

    user_architect_matrix = np.concatenate(user_architect_matrix, axis=0)

    preds = session.run(decoder_op, feed_dict = {X: user_architect_matrix})

    pred_data = pred_data.append(pd.DataFrame(preds))

    pred_data = pred_data.stack().reset_index(name='rating')
    pred_data.rename(columns = {'level_0': 'user_id', 'level_1': 'architect_id'}, inplace=True)
    pred_data['user_id'] = pred_data['user_id'].map(lambda value: users[value])
    pred_data['architect_id'] = pred_data['architect_id'].map(lambda value: architects[value])

    keys = ['user_id', 'architect_id']
    index_1 = pred_data.set_index(keys).index
    index_2 = combined.set_index(keys).index

    top_ten_ranked = pred_data[~index_1.isin(index_2)]
    top_ten_ranked = top_ten_ranked.sort_values(['user_id', 'rating'], ascending=[True, False])
    top_ten_ranked = top_ten_ranked.groupby('user_id').head(10)

print(top_ten_ranked.loc[top_ten_ranked['user_id'] == 5])

print(rating.loc[rating['user_id'] == 5].sort_values(by=['rating'], ascending=False))
