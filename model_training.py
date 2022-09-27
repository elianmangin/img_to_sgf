import tensorflow as tf
from preprocess import vectorisation_img, vectorisation_sgf
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X = vectorisation_img('img_train/')
    Y = vectorisation_sgf('sgf_train/')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(500, input_shape=(651468,)),
        tf.keras.layers.Dense(430, activation='relu'),
        tf.keras.layers.Dense(361, activation='sigmoid')
    ])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, input_shape=(150*150*3,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(361, activation='sigmoid')
    ])

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

    model.fit(X, Y, epochs=50)
