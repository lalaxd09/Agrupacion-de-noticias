"""""#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#17/24/2023

#Código para hacer la limpieza de los datos:stopwords,signos de puntuación,lematización y stemming
#En este caso los datos requeridos son un csv con contenido de noticias y etiquitado
#Funciones requeridas
#1.Normalizar->Ayuda al preprocesamiento del texto como quitar stop words,signos de puntuacíón,stemming
2.clen_html->Ayuda a la limpieza de las etiquetas de html
3.frequency_words->Ayuda a verificar la frecuencia de las palabras"""

import re
import unicodedata
import spacy
import pandas as pd
import nltk
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords')
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = nltk.corpus.stopwords.words('english')

# Cargar modelo de Spacy
nlp = spacy.load('en_core_web_sm')

# Agregar stop words personalizados al modelo de Spacy
custom_stop_words = set(stop_words)
for word in custom_stop_words:
    nlp.vocab[word].is_stop = True



"""Función que se utiliza para normalizar el texto desde quitar signos de puntuación,stopw words
 lematizar,stemming"""
def normalizar(n_text):

    clean_text = n_text
    #Eliminar numeros
    clean_text = re.sub('\d', '', clean_text)
    # Eliminar signos de puntuación
    clean_text = re.sub('[^\w\s]', '', clean_text)

    # Acentos
    #clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8')

    # Convertir texto en  mínisculas
    clean_text = clean_text.lower()

    # Crear lista de tokens que deseas incluir
    #include_tags = ['ADJ', 'NOUN', 'VERB']

# Analizar texto completo con Spacy
    clean_text = nlp(clean_text)
    # Lematizar y Stemming
    tokens = [stemmer.stem(token.lemma_) if token.lemma_ != '-PRON-' else stemmer.stem(token.text) for token in clean_text if not token.is_stop and not token.is_punct and not token.is_space]
    print(tokens)
    documento = ' '.join(tokens)

    return documento



#Verificar Frecuencia de palabras
def frequency_words(column):
    # Lista de tokens de la columna 'Documento_Normalizado'
    tokens = [token for doc in column for token in doc.split()]

    # Contar la frecuencia de cada palabra
    frecuencia_palabras = Counter(tokens)

    # Visualizar los resultados
    frecuencia_palabras_grafico = frecuencia_palabras.most_common(20)
    plt.bar(*zip(*frecuencia_palabras_grafico))
    plt.xticks(rotation=90)
    plt.show()

    return frecuencia_palabras





