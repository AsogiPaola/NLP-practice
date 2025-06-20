import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split

archivos = [
    "benedetti.txt",
    "neruda.txt"
]

texts = []
labels = []

for label, l in enumerate(archivos):
    print(f"{l} Corresponde a {label}")
    
    with open(l, "r", encoding="utf-8") as archivo:
        for line in archivo: #recorrer cada una de las lineas de los textos con line
            #print(line)
            line = line.rstrip().lower() #todo a minuscula
            #print(line)
            if line:
                #eliminar puntuacion
                line = line.translate(str.maketrans("", "", string.punctuation))
                texts.append(line)
                labels.append(label)
            #print(line)
            
train_text, test_text, Ytrain, Ytest = train_test_split(texts, labels, test_size=0.1, random_state=42)        
len(Ytrain), len(Ytest)  
#<unk> es una convencion que se utiliza a menudo en NLP para representar palabras desconocidas o fuera del vocabulario. En este caso, se esta
#asignando el indice 0 a esta palabra especial
indice = 1
indice_palabras = {"<unk>": 0}

#construccion deun diccionario de codificacion de palabras a indices recorrer el conjunto de entranmiento 
# a cada linea se le asigna una tokenizacion, es decir separar en palabras las lineas y a cada palabra se le asigna un indice
for text in train_text:  
    tokens = text.split()  
    for token in tokens: 
        if token not in indice_palabras:
            indice_palabras[token] = indice  
            indice += 1
            
#convertir datos a enteros porque necesitamos entrenar
train_text_int =[]
test_text_int = []

#acorde a los indices que transformamos construimos el conjunto de entramiento en enteros
for text in train_text:
    tokens = text.split()
    linea_entero = [indice_palabras[token] for token in tokens]
    train_text_int.append(linea_entero)

len(indice_palabras)   
    
for text in test_text:
    tokens = text.split()
    line_as_int = [indice_palabras.get(token, 0)for token in tokens]
    test_text_int.append(line_as_int)
    
    
v = len(indice_palabras)
#A va a representar las maatriz de transicion para los textos y pi las probabilidades iniciales de la palabras plicando la MARKOV
A0 = np.ones((v,v))
pi0 = np.ones(v)

A1 = np.ones((v,v))
pi1 = np.ones(v)


def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx  = None
        for idx in tokens:
            #primera palabra de la secuencia
            if last_idx is None:
                pi[idx] +=1
            else:
                A[last_idx, idx] +=1
            last_idx = idx
            
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)
compute_counts([t for t, y in zip(test_text_int, Ytest) if y == 0], A1, pi1)

A0 /=  A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()

A1 /=  A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()

logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)


#cuenta dde etiquetas de calase 0 en Ytrain
count0 = sum( y == 0 for y in Ytrain)
#cuenta  dde etiquetas de calase 1 en Ytrain
count1 = sum( y == 1 for y in Ytrain)

#cantidad total de ejemplos de entrenamiento
total = len(Ytrain)

#probabilidad a priori de clase 0
p0 = count0 / total
#probabilidad a priori de clase 1
p1 = count1 / total

#logaritmo de la probabilidad a priori de la clase 0
logp0 = np.log(p0)
#logaritmo de la probabilidad a priori de la clase 1
logp1 = np.log(p1)

#ver las probabilidades de ambas clases
p0, p1

"""
The classifier receives three pre-trained probability matrices:

- logAs: Transition probabilities between words (log scale) - "given the previous word was X, what's the probability the next word is Y?"
- logpis: Initial word probabilities (log scale) - "what's the probability a sequence starts with word X?"
- logpriors: Class probabilities (log scale) - "how likely is each class overall?"
- self.K: Number of classes to classify into

Core Calculation (_compute_log_likehood)
This method calculates how likely a text sequence belongs to a specific class:
1. Gets the transition matrix (logA) and initial probabilities (logpi) for the given class
2. Iterates through each word index in the input sequence
3. For the first word: adds logpi[idx] (probability of starting with this word)
4. For subsequent words: adds logA[last_idx, idx] (probability of transitioning from previous word to current word)
Returns the total log probability

Prediction (predict)
1. Calculates posteriors: For each possible class, computes likelihood + prior
2. self._compute_log_likehood(input_, c): How well the sequence fits class c
3. +self.logpriors[c]: Adds the overall probability of class c
4. np.argmax(posteriors): Picks the class with highest probability
"""

#construccion de clasificador 
class Clasifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors) #numero de clases
        
    #funcion de probabilidad de que pertenezca a una clase que se le proporciona
    def _compute_log_likehood(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]
            
        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                #es el primer token de la secuencia
                logprob += logpi[idx]
            else:
                #calcula la probabilidad de traansicion de la palabra anterior a la actual
                logprob += logA[last_idx, idx]
                    
            #actualiza las_idx para la proxima iteracion
            last_idx = idx
        return logprob
    
    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            #calcula los logaritmos de las probabilidades posteriores para cada clase
            posteriors = [self._compute_log_likehood(input_, c) + self.logpriors[c]  
                          for c in range(self.K)]
            
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions
    
clf = Clasifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

Ptrain = clf.predict(train_text_int)
print(f"Train accuracy: {np.mean(Ptrain == Ytrain)}")

Ptest = clf.predict(test_text_int)
print(f"Test accuracy: {np.mean(Ptest == Ytest)}")