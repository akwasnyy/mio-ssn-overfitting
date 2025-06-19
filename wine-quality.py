import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2

def przygotuj_dane():
    wine = fetch_ucirepo(id=186)
    X = wine.data.features
    y = wine.data.targets.values.ravel()
    y_shifted = y - y.min()
    num_classes = len(np.unique(y_shifted))
    y_cat = to_categorical(y_shifted, num_classes=num_classes)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_shifted
    )


    y_train_labels = y_train_full.argmax(axis=1)
    X_train, _, y_train_cat, _ = train_test_split(
        X_train_full, y_train_full, test_size=0.8, random_state=42, stratify=y_train_labels
    )

    return X_train, X_test, y_train_cat, y_test, num_classes, y_cat, y_shifted

def przeuczony_model(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def dropout_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def l2_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def l1_model(input_dim, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l1(0.001)),
        Dense(32, activation='relu', kernel_regularizer=l1(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def uproszczony_model(input_dim, num_classes):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def trening(model, X_train, y_train, X_test, y_test, use_early_stopping=False):
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True)] if use_early_stopping else []
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=callbacks
    )
    elapsed = time.time() - start_time
    return model, history, elapsed

def ewaluacja(model, history, X_test, y_test, title='Model',elapsed_time=None):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    print(f"\n=== {title} ===")
    if elapsed_time is not None:
        print(f"Czas treningu {elapsed_time:.2f} sekundy")
    print(classification_report(y_true, y_pred))
    print("Precyzja:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1:", f1_score(y_true, y_pred, average='macro'))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Macierz pomyłek - {title}")
    plt.xlabel("Przewidywane")
    plt.ylabel("Prawdziwe")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Trening')
    plt.plot(history.history['val_loss'], label='Walidacja')
    plt.title(f"Strata - {title}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Trening')
    plt.plot(history.history['val_accuracy'], label='Walidacja')
    plt.title(f"Dokładność - {title}")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    X_train, X_test, y_train, y_test, num_classes, _, _ = przygotuj_dane()
    input_dim = X_train.shape[1]

    configs = [
        ("Przeuczony model", przeuczony_model, False),
        ("Dropout", dropout_model, False),
        ("L2", l2_model, True),
        ("L1",l1_model, True),
        ("Uproszczony Model", uproszczony_model, False)
    ]

    for name, builder, use_es in configs:
        model = builder(input_dim, num_classes)
        model, history, elapsed = trening(model, X_train, y_train, X_test, y_test, use_early_stopping=use_es)
        ewaluacja(model, history, X_test, y_test, title=name, elapsed_time=elapsed)


if __name__ == "__main__":
    main()