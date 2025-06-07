"""
Time-series Generative Adversarial Networks (TimeGAN) - TensorFlow 2.x Refactored Version
"""

import tensorflow as tf
import numpy as np
from utils import extract_time, rnn_layer, random_generator, batch_generator


def timegan(ori_data, parameters):
    def min_max_scaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    ori_data, min_val, max_val = min_max_scaler(ori_data)
    ori_time, max_seq_len = extract_time(ori_data)

    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = ori_data[0].shape[1]
    gamma = 1

    class Embedder(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.rnns = [rnn_layer(module_name, hidden_dim) for _ in range(num_layers)]
            self.dense = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

        def call(self, x):
            for rnn in self.rnns:
                x = rnn(x)
            return self.dense(x)

    class Recovery(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.rnns = [rnn_layer(module_name, hidden_dim) for _ in range(num_layers)]
            self.dense = tf.keras.layers.Dense(z_dim, activation='sigmoid')

        def call(self, h):
            for rnn in self.rnns:
                h = rnn(h)
            return self.dense(h)

    class Generator(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.rnns = [rnn_layer(module_name, hidden_dim) for _ in range(num_layers)]
            self.dense = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

        def call(self, z):
            for rnn in self.rnns:
                z = rnn(z)
            return self.dense(z)

    class Supervisor(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.rnns = [rnn_layer(module_name, hidden_dim) for _ in range(num_layers - 1)]
            self.dense = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')

        def call(self, h):
            for rnn in self.rnns:
                h = rnn(h)
            return self.dense(h)

    class Discriminator(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.rnns = [rnn_layer(module_name, hidden_dim) for _ in range(num_layers)]
            self.dense = tf.keras.layers.Dense(1)

        def call(self, h):
            for rnn in self.rnns:
                h = rnn(h)
            return self.dense(h)

    embedder = Embedder()
    recovery = Recovery()
    generator = Generator()
    supervisor = Supervisor()
    discriminator = Discriminator()

    embedder_optimizer = tf.keras.optimizers.Adam()
    recovery_optimizer = tf.keras.optimizers.Adam()
    generator_optimizer = tf.keras.optimizers.Adam()
    supervisor_optimizer = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()

    print('Start Embedding Network Training')
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)

        with tf.GradientTape() as tape:
            H = embedder(X_mb)
            X_tilde = recovery(H)
            E_loss_T0 = tf.reduce_mean(tf.keras.losses.mse(X_mb, X_tilde))
            E_loss0 = 10 * tf.sqrt(E_loss_T0)

        e_vars = embedder.trainable_variables + recovery.trainable_variables
        grads = tape.gradient(E_loss0, e_vars)
        embedder_optimizer.apply_gradients(zip(grads, e_vars))

        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, e_loss: {np.round(E_loss0.numpy(), 4)}')

    print('Finish Embedding Network Training')

    print('Start Training with Supervised Loss Only')
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Z_mb = tf.convert_to_tensor(random_generator(batch_size, z_dim, T_mb, max_seq_len), dtype=tf.float32)

        with tf.GradientTape() as tape:
            E_hat = generator(Z_mb)
            H_hat = supervisor(E_hat)
            H = embedder(X_mb)
            H_supervise = supervisor(H)
            G_loss_S = tf.reduce_mean(tf.square(H[:, 1:, :] - H_supervise[:, :-1, :]))

        g_vars = generator.trainable_variables + supervisor.trainable_variables
        grads = tape.gradient(G_loss_S, g_vars)
        supervisor_optimizer.apply_gradients(zip(grads, g_vars))

        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(G_loss_S.numpy()), 4)}')

    print('Finish Training with Supervised Loss Only')

    print('Start Joint Training')
    for itt in range(iterations):
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            Z_mb = tf.convert_to_tensor(random_generator(batch_size, z_dim, T_mb, max_seq_len), dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                H = embedder(X_mb)
                X_tilde = recovery(H)

                E_hat = generator(Z_mb)
                H_hat = supervisor(E_hat)
                X_hat = recovery(H_hat)
                Y_fake = discriminator(H_hat)
                Y_real = discriminator(H)
                Y_fake_e = discriminator(E_hat)
                H_supervise = supervisor(H)

                G_loss_U = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake), Y_fake, from_logits=True))
                G_loss_U_e = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True))
                G_loss_S = tf.reduce_mean(tf.square(H[:, 1:, :] - H_supervise[:, :-1, :]))
                G_loss_V1 = tf.reduce_mean(tf.abs(tf.math.reduce_std(X_hat, axis=0) - tf.math.reduce_std(X_mb, axis=0)))
                G_loss_V2 = tf.reduce_mean(tf.abs(tf.reduce_mean(X_hat, axis=0) - tf.reduce_mean(X_mb, axis=0)))
                G_loss_V = G_loss_V1 + G_loss_V2
                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

                D_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(Y_real), Y_real, from_logits=True))
                D_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(Y_fake), Y_fake, from_logits=True))
                D_loss_fake_e = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(Y_fake_e), Y_fake_e, from_logits=True))
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

                E_loss_T0 = tf.reduce_mean(tf.keras.losses.mse(X_mb, X_tilde))
                E_loss0 = 10 * tf.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1 * G_loss_S

            g_vars = generator.trainable_variables + supervisor.trainable_variables
            e_vars = embedder.trainable_variables + recovery.trainable_variables
            d_vars = discriminator.trainable_variables

            generator_grads = tape.gradient(G_loss, g_vars)
            embedder_grads = tape.gradient(E_loss, e_vars)
            discriminator_grads = tape.gradient(D_loss, d_vars)

            generator_optimizer.apply_gradients(zip(generator_grads, g_vars))
            embedder_optimizer.apply_gradients(zip(embedder_grads, e_vars))
            if D_loss > 0.15:
                discriminator_optimizer.apply_gradients(zip(discriminator_grads, d_vars))

            if itt % 1000 == 0:
                print(f'step: {itt}/{iterations}, d_loss: {np.round(D_loss.numpy(), 4)}, g_loss_u: {np.round(G_loss_U.numpy(), 4)}, g_loss_s: {np.round(np.sqrt(G_loss_S.numpy()), 4)}, g_loss_v: {np.round(G_loss_V.numpy(), 4)}, e_loss_t0: {np.round(np.sqrt(E_loss_T0.numpy()), 4)}')

    print('Finish Joint Training')

    print('Generating synthetic data')
    Z_mb = tf.convert_to_tensor(random_generator(len(ori_data), z_dim, ori_time, max_seq_len), dtype=tf.float32)
    E_hat = generator(Z_mb)
    H_hat = supervisor(E_hat)
    X_hat = recovery(H_hat)
    X_hat = X_hat.numpy()

    generated_data = [X_hat[i, :ori_time[i], :] for i in range(len(ori_data))]
    generated_data = [g * max_val + min_val for g in generated_data]

    return generated_data


if __name__ == '__main__':
    print("[Test] Running TimeGAN with random sine data...")
    sample = np.sin(np.linspace(0, 2 * np.pi, 24))
    ori_data = [np.stack([sample + np.random.normal(0, 0.1, 24) for _ in range(5)], axis=1) for _ in range(100)]
    parameters = {
        'module': 'gru',
        'hidden_dim': 24,
        'num_layer': 3,
        'iterations': 100,
        'batch_size': 16
    }
    generated = timegan(ori_data, parameters)
    print(f"Generated {len(generated)} samples with shape {generated[0].shape}")
