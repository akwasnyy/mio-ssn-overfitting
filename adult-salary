import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from ucimlrepo import fetch_ucirepo

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2

def prepare_data():
    adult = fetch_ucirepo(name='Adult')
    X = adult.data.features
    y = adult.data.targets
    y = (y.values.ravel() == '>50K').astype(int)

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    y_cat = to_categorical(y, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def overfitted_model(input_dim, num_classes):
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
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
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
        Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l1(0.005)),
        Dense(32, activation='relu', kernel_regularizer=l1(0.005)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def simplified_model(input_dim, num_classes):
    model = Sequential([
        Dense(48, activation='relu', input_shape=(input_dim,)),
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
        batch_size=64,
        verbose=0,
        callbacks=callbacks
    )
    elapsed = time.time() - start_time
    return model, history, elapsed


def ewaluacja(model, history, X_test, y_test, title='Model', elapsed_time=None):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    print(f"\n=== {title} ===")
    if elapsed_time is not None:
        print(f"Czas treningu: {elapsed_time:.2f} sekundy")
    print(classification_report(y_true, y_pred))
    print("Precyzja:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))

    train_acc = history.history['accuracy'][-1] * 100
    val_acc = history.history['val_accuracy'][-1] * 100
    print(f"Dokładność treningowa: {train_acc:.2f}%")
    print(f"Dokładność testowa: {val_acc:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Macierz pomyłek - {title}")
    plt.xlabel("Przewidywane")
    plt.ylabel("Prawdziwe")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Trening')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title(f"Strata - {title}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Trening')
    plt.plot(history.history['val_accuracy'], label='Test')
    plt.title(f"Dokładność - {title}")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    input_dim = X_train.shape[1]
    num_classes = 2

    configs = [
        ("Przeuczony model", overfitted_model, False),
        ("Dropout", dropout_model, False),
        ("L2", l2_model, True),
        ("L1", l1_model, True),
        ("Uproszczony Model", simplified_model, False)
    ]
    results = []

    for name, builder, use_es in configs:
        model = builder(input_dim, num_classes)
        model, history, elapsed = trening(model, X_train, y_train, X_test, y_test, use_early_stopping=use_es)

        y_pred = model.predict(X_test).argmax(axis=1)
        y_true = y_test.argmax(axis=1)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results.append([name, precision, recall, f1, elapsed])
        ewaluacja(model, history, X_test, y_test, title=name, elapsed_time=elapsed)

    df_results = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1", "Czas (s)"])
    print("\nPodsumowanie wyników:")
    print(df_results.sort_values(by="F1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
