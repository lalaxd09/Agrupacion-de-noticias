"""""#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#21/04/2023
En este codigo es para realizar la capa de embending y poder "agrupar " con la capa de salida

En este código lo necesario para aplicar el modelo  con 6000 observaciones cada clase con 2000 datos
Funciones
1.Dummy_loss->devuelve un valor constante igual a cero. Se usará para compilar los modelos  esto debido a que se ocupara en loss cuando se entrena
el modelo
2.valores_s(n)->Funcion para obtener valores aleatorios para probar en las capas de salida
3.modelo->para crear un modelo de redes neuronales que genera una matriz de embedding a partir de una matriz de representación one-hot de palabras.
y poder representar las salidas que se tienen por medio de PCA
"""
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
from preprocesamiento import normalizar
from tensorflow.keras.models import Sequential
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras.backend as K
import numpy as np


from scipy.sparse import csr_matrix
# Leer noticias
df = pd.read_csv('train.csv')
# Normalización del texto
df['Documento_Normalizado'] = df['Contenido'].apply(normalizar)

# Crear un objeto CountVectorizer
vectorizer = CountVectorizer(binary=True, max_features=1000)

# Ajustar el vectorizador a tus datos
vectorizer.fit(df['Documento_Normalizado'])

# Obtener la matriz de representaciones de palabras
X = vectorizer.transform(df['Documento_Normalizado'])

# Crear el objeto OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=True)

# Convertir la matriz X en una matriz one-hot encoding dispersa
X_dense = X.toarray()
X_onehot_sparse = onehot_encoder.fit_transform(X_dense)

print(f"Forma de la matriz X_onehot: {X_onehot_sparse.shape}")

#devuelve un valor constante igual a cero. Se usará para compilar los modelos
def dummy_loss(y_true, y_pred):
    return K.constant(0.0)

#Funcion para obtener una lista con valores aleatorias
def valores_s(n):
    valores_salida = []
    for i in range(n):
        valor = random.randint(3, 10)
        valores_salida.append(valor)

    return valores_salida


def modelo(data):
    # Dimensión del espacio de embedding
    embedding_dim = 10

    # Longitud máxima de las secuencias
    max_length = 50

    # Convertir la matriz dispersa en una matriz densa y redimensionarla
    data = data.toarray().reshape(data.shape[0], max_length, -1)

    # Crear la capa de embedding
    embedding_layer = Dense(units=embedding_dim, activation='linear', input_shape=(max_length, data.shape[2]))
    salida_embeding = valores_s(5)
    for valor in salida_embeding:
        # Crear un modelo de redes neuronales
        model = Sequential([
            embedding_layer,  # Capa de embedding
            Flatten(),  # Aplanar la matriz de salida del embedding
            Dense(16, activation='relu'),
            Dense(valor, activation='softmax')  # Capa de salida con valores de salidas aleatorias y función de activación softmax
        ])

        # Compilar el modelo
        model.compile(optimizer='adam', loss=dummy_loss)

        # Entrenar el modelo
        y = np.random.uniform(size=(data.shape[0], 1))
        model.fit(data, y, epochs=10)

        # Obtener la matriz de embedding
        embedding_matrix = embedding_layer.get_weights()[0]

        pca = PCA(n_components=2)
        embedding_matrix_pca = pca.fit_transform(embedding_matrix)

        plt.scatter(embedding_matrix_pca[:, 0], embedding_matrix_pca[:, 1])
        plt.show()

        print('Matriz de embedding:', embedding_matrix)

        return embedding_matrix

modelo(X_onehot_sparse)





