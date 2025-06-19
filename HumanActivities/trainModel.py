from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix

def trenuj_i_podaj_dane(
        model,
        name,
        x_train,
        x_test,
        y_train,
        y_test,
        epochs=100,
        batch_size=32,
        use_early_stopping=False,
        patience=10,
        plot=True
):
    callbacks = []
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stop)

    print(f"\n🧠 Trening modelu: {name}")
    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    duration = time.time() - start

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    epochs_trained = len(history.history['loss'])
    overfitting_gap = train_acc - test_acc

    print(f"⏱️ Czas treningu: {duration:.2f} s")
    print(f"📊 Dokładność na zbiorze treningowym: {train_acc:.4f}")
    print(f"🎯 Dokładność na zbiorze testowym:     {test_acc:.4f}")
    print(f"⚠️ Luka przeuczenia (Trening - Test): {overfitting_gap:.4f}")
    print(f"🔁 Liczba przeprowadzonych epok: {epochs_trained}")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    print("\n📋 Raport klasyfikacji:\n", classification_report(y_test, y_pred))
    print("\n🔢 Macierz pomyłek:\n", confusion_matrix(y_test, y_pred))

    if plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Błąd trenowania')
        plt.plot(history.history['val_loss'], label='Błąd walidacyjny')
        plt.title(f"{name} - Loss")
        plt.xlabel('Liczba epok')
        plt.ylabel('Wartość funkcji straty')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Dokładność trenowania')
        plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
        plt.title(f"{name} - Dokładność")
        plt.xlabel('Liczba epok')
        plt.ylabel('Dokładność')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    return test_acc, train_acc, duration