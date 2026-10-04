# Alternative gesture classifiers used by compare_models.py (the CNNs live in train_gestures.py).
# Import this module before keras.models.load_model() on a VIT .keras file so the custom layer is registered.
import keras
from keras import layers, ops

NUM_LANDMARKS = 21


def DefineMLP(num_classes, num_features=2 * NUM_LANDMARKS, dropout=0.3):
    # Input: flat vector x0,y0,x1,y1,... (42,)
    return keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax'),
    ], name='MLP')


@keras.saving.register_keras_serializable(package='HAR_mediapipe')
class CLSAndPosition(layers.Layer):
    # Antepone un token CLS aprendido y suma un embedding posicional aprendido (num_tokens + 1 posiciones)
    def __init__(self, num_tokens, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_token = self.add_weight(name='cls_token', shape=(1, 1, self.d_model),
                                         initializer='random_normal', trainable=True)
        self.position_embedding = self.add_weight(name='position_embedding',
                                                  shape=(1, self.num_tokens + 1, self.d_model),
                                                  initializer='random_normal', trainable=True)

    def call(self, x):
        cls = ops.broadcast_to(self.cls_token, (ops.shape(x)[0], 1, self.d_model))
        return ops.concatenate([cls, x], axis=1) + self.position_embedding

    def get_config(self):
        config = super().get_config()
        config.update({'num_tokens': self.num_tokens, 'd_model': self.d_model})
        return config


def DefineVIT(num_classes, num_landmarks=NUM_LANDMARKS, d_model=32, num_blocks=2, num_heads=4,
              key_dim=8, mlp_units=(64, 32), dropout=0.1):
    # ViT sobre landmarks: cada landmark (x, y) es un "patch" -> token de dimension d_model.
    # Input: (21, 2)
    inputs = keras.Input(shape=(num_landmarks, 2))
    x = layers.Dense(d_model)(inputs)                      # point embedding
    x = CLSAndPosition(num_landmarks, d_model)(x)          # (22, d_model)
    for _ in range(num_blocks):
        # pre-norm attention + residual
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(h, h)
        x = layers.Add()([x, h])
        # pre-norm MLP + residual (the last layer projects back to d_model)
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        for units in mlp_units:
            h = layers.Dense(units, activation='gelu')(h)
            h = layers.Dropout(dropout)(h)
        x = layers.Add()([x, h])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls = x[:, 0, :]                                       # CLS token
    outputs = layers.Dense(num_classes, activation='softmax')(cls)
    return keras.Model(inputs, outputs, name='VIT')
