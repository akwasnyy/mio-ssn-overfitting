import numpy as np
import pandas as pd
from keras import Input
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from HumanActivities.trainModel import trenuj_i_podaj_dane

X_train = pd.read_csv("./X_train.txt", sep='\s+', header=None)
X_test = pd.read_csv("./X_test.txt", sep='\s+', header=None)
y_train = pd.read_csv("./y_train.txt", sep='\s+', header=None).values.ravel()
y_test = pd.read_csv("./y_test.txt", sep='\s+', header=None).values.ravel()
activity_labels = pd.read_csv("./activity_labels.txt", sep='\s+', header=None, index_col=0)
y_train -= 1
y_test -= 1

activity_labels_dict = activity_labels[1].to_dict()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_full_scaled = X_train_scaled.copy()
y_train_full = y_train.copy()

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.05, random_state=42)
for small_idx, _ in sss.split(X_train_full_scaled, y_train_full):
    X_train_scaled = X_train_full_scaled[small_idx]
    y_train = y_train_full[small_idx]

print("Rozkład klas w treningu:", np.bincount(y_train))
print("Rozkład klas w teście:  ", np.bincount(y_test))
Adam(learning_rate=0.005)
baseline_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(6, activation='softmax')
])
baseline_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size=16

# Trening
trenuj_i_podaj_dane(
    baseline_model,
    name="Model z przeuczeniem",
    x_train=X_train_scaled,
    x_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    epochs=300,
    batch_size=len(X_train_scaled),
    use_early_stopping=False,
    plot=True
)


# DODATKOWE MODELE
dropout_early_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(6, activation='softmax')
])

dropout_early_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

trenuj_i_podaj_dane(
    dropout_early_model,
    name="Dropout (identyczna struktura) + EarlyStopping",
    x_train=X_train_scaled,
    x_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    epochs=150,
    batch_size=16,  # jak w modelu bazowym
    use_early_stopping=True,
    patience=30
)

l1l2_strong_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
    Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
    Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
    Dense(6, activation='softmax')
])

l1l2_strong_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

trenuj_i_podaj_dane(
    l1l2_strong_model,
    name="L1/L2 (mocniejsze) + Bez Dropoutu + EarlyStopping",
    x_train=X_train_scaled,
    x_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    epochs=150,
    batch_size=16,
    use_early_stopping=True,
    patience=30
)


sss = StratifiedShuffleSplit(n_splits=1, train_size=0.25, random_state=42)
for small_idx, _ in sss.split(X_train_full_scaled, y_train_full):
    X_train_more_data = X_train_full_scaled[small_idx]
    y_train_more_data = y_train_full[small_idx]

l1l2_strong_model_more_data = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
    Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
    Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
    Dense(6, activation='softmax')
])

l1l2_strong_model_more_data.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

trenuj_i_podaj_dane(
    l1l2_strong_model_more_data,
    name="L1/L2 (mocniejsze) + Bez Dropoutu + EarlyStopping + 2-krotnie więcej danych treningowych",
    x_train=X_train_more_data,
    x_test=X_test_scaled,
    y_train=y_train_more_data,
    y_test=y_test,
    epochs=150,
    batch_size=16,
    use_early_stopping=True,
    patience=30
)




# Augmentacja danych
noise_factor = 0.01
X_train_noisy = X_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_scaled.shape)
X_train_noisy = np.clip(X_train_noisy, -3, 3)

# Model
augment_dropout_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.2),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(6, activation='softmax')
])

augment_dropout_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trening
trenuj_i_podaj_dane(
    augment_dropout_model,
    name="Augmentacja (0.025) + Dropout (2 warstwy) + L2 + EarlyStopping",
    x_train=X_train_noisy,
    x_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    epochs=250,
    batch_size=32,
    use_early_stopping=True,
    patience=50
)
