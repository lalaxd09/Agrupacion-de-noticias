"""""#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#21/24/2023
En este código lo que se hará es aplicar el modelo de K-means con 6000 observaciones cada clase con 2000 datos
tambien se obtendra el k-optimo
Las funciones que se encuentran son:
normalizar->que se crea en el preprocesamiento de los datos
valores_s->Crea una lista aleatoria para probar diferentes valores de k
k_meas->donde se buscar el valor optimo y se prueba con distintos valores de cluesters
ajuste->En esta funcion se busca utilizar el valor k_optimo en la funcion de k_meas"""
import pandas as pd
from preprocesamiento import normalizar,frequency_words
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer




#Leer archivo
df = pd.read_csv('train.csv')
# Agrupar los datos por categoría y seleccionar los primeros 2000 de cada grupo
df['label'] = df['label'].astype(str)

#2000 observaciones para cada clase
world = df[df['label'] == '1'].head(2000)
Sports = df[df['label'] == '2'].head(2000)
Business = df[df['label'] == '3'].head(2000)

#Se junta las selecciones de cadaD
df=pd.concat([world, Sports, Business], ignore_index=True)
print(df)
#Aplicamos normalización de texto
df['Documento_Normalizado'] = df['text'].apply(normalizar)

#Verificar frecuencia de palabras
frequency_words(df['Documento_Normalizado'])


# crear el objeto vectorizador
vectorizer = TfidfVectorizer()
# ajustar y transformar los documentos del DataFrame
X = vectorizer.fit_transform(df['Documento_Normalizado'])

#Funcion para hacer una lista con valores aleatorios de un rango de 3,15
def valores_s(n):
    valores_salida = []
    for i in range(n):
        valor = random.randint(2, 8)
        if valor not in valores_salida:
            valores_salida.append(valor)

    valores_salida.sort(reverse=True)

    return valores_salida

def k_meas(X,n):
    #Creación de lista al azar
    k_list=valores_s(n)
    # Inicializar listas vacías para almacenar los resultados
    inertia_scores = []

    # Iterar sobre el número de clusters
    for k in k_list:
        # Inicializar modelo KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)

        # Ajustar el modelo a los datos
        kmeans.fit(X)

        # Calcular inercia
        inertia_scores.append(kmeans.inertia_)

    #Valors de los cluster e inertiria
    print(k_list,inertia_scores)
    # Encontrar el valor óptimo de K
    #calcula la diferencia entre cada par de valores consecutivos de la lista y devuelve una nueva lista llamada deltas con esas diferencias. Esto se hace para calcular las pendientes de la curva de inercia.
    deltas = np.diff(inertia_scores)
    # para cada valor en la lista deltas se calcula la pendiente dividiendo el valor actual por el valor anterior
    slopes = [deltas[i] / deltas[i - 1] for i in range(1, len(deltas))]
    #es la lista de números de clusters para los que se calculó la inercia y slopes es la lista de pendientes. max(slopes) devuelve el valor máximo de las pendientes
    k_optimo = k_list[slopes.index(max(slopes))]




    return k_optimo,k_list,inertia_scores



#Funcion para ajustar de acuerdo al
def ajuste(data):
    #Retomamos la función del k_means para retorna el valor optimo
    kopt,list,inertia=k_meas(data,6)
    #Funcion para el algoritmo
    kmeans = KMeans(n_clusters=kopt, random_state=42)
    #Ajustamos datos
    kmeans.fit(data)

    # Asignar etiquetas a cada muestra
    labels = kmeans.predict(data)

    # Imprimir el número de muestras en cada cluster
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # Calcular el score de silueta
    silhouette_avg = silhouette_score(data, labels)
    print("El score de silueta es:", silhouette_avg)

    # Reducir dimensionalidad a 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # Graficar resultados
    plt.plot(list, inertia, 'bx-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del codo para encontrar k óptimo')
    plt.show()

    # Graficar clusters
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title('Clusters con K Optimo = %d' % kopt)
    plt.show()





ajuste(X)