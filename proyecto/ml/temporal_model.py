"""Modelo temporal liviano para clasificacion de senas dinamicas."""

from __future__ import annotations

from proyecto.ml.sequence_dataset import SEQUENCE_FEATURES, SEQUENCE_LENGTH


def build_temporal_model(tf, num_classes: int):
    """Construye un clasificador GRU para secuencias de landmarks."""
    if num_classes < 2:
        raise ValueError("El modelo temporal necesita al menos dos clases.")

    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, SEQUENCE_FEATURES), name="landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.GRU(96, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.GRU(48)(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="sign_class")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dynamic_sign_gru")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
