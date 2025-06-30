import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Ejemplo 1: Implementación desde cero (educativa)
class MultinomialNaiveBayesFromScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
        self.vocab_size = 0
    
    def fit(self, X, y):
        """
        X: matriz de conteos (n_samples, n_features)
        y: etiquetas de clase
        """
        self.classes = np.unique(y)
        n_samples, self.vocab_size = X.shape
        
        # Calcular probabilidades a priori
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
        
        # Calcular probabilidades condicionales
        for c in self.classes:
            # Documentos de la clase c
            X_c = X[y == c]
            
            # Suma de todas las palabras en la clase c
            total_words_in_class = np.sum(X_c) + self.alpha * self.vocab_size
            
            # Probabilidad de cada palabra dada la clase
            word_counts = np.sum(X_c, axis=0) + self.alpha
            self.feature_probs[c] = word_counts / total_words_in_class
    
    def predict_proba(self, X):
        """Predecir probabilidades para cada clase"""
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # Log-probabilidad para evitar underflow
            log_prior = np.log(self.class_priors[c])
            log_likelihood = np.sum(X * np.log(self.feature_probs[c]), axis=1)
            proba[:, i] = log_prior + log_likelihood
        
        # Convertir de log-space a probabilidades normalizadas
        proba = np.exp(proba)
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        return proba
    
    def predict(self, X):
        """Predecir clases"""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

# Ejemplo 2: Uso con scikit-learn (recomendado para producción)
def ejemplo_clasificacion_texto():
    # Datos de ejemplo
    textos = [
        "Me encanta este producto, es excelente y funciona perfectamente",
        "Terrible calidad, no funciona para nada, muy decepcionante",
        "Buen producto, recomendado, funciona bien",
        "Pésimo servicio, no lo recomiendo, muy malo",
        "Excelente calidad, muy satisfecho con la compra",
        "No me gustó, tiene muchos problemas",
        "Fantástico producto, superó mis expectativas",
        "Muy mal, se rompió al primer uso"
    ]
    
    etiquetas = ['positivo', 'negativo', 'positivo', 'negativo', 
                'positivo', 'negativo', 'positivo', 'negativo']
    
    # Pipeline completo con preprocesamiento
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(
            stop_words=None,  # En español usarías stop_words personalizadas
            ngram_range=(1, 2),  # Unigrams y bigrams
            max_features=1000
        )),
        ('classifier', MultinomialNB(alpha=1.0))
    ])
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=0.3, random_state=42
    )
    
    # Entrenamiento
    pipeline.fit(X_train, y_train)
    
    # Predicciones
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    print("=== RESULTADOS DE CLASIFICACIÓN ===")
    print(f"Textos de prueba: {X_test}")
    print(f"Predicciones: {y_pred}")
    print(f"Probabilidades: {y_pred_proba}")
    
    return pipeline

# Ejemplo 3: Análisis completo con métricas
def analisis_completo():
    # Simular dataset más grande
    np.random.seed(42)
    
    # Generar datos sintéticos para demostración
    palabras_positivas = ['excelente', 'genial', 'fantástico', 'bueno', 'recomiendo']
    palabras_negativas = ['terrible', 'malo', 'pésimo', 'horrible', 'decepcionante']
    
    textos = []
    etiquetas = []
    
    # Generar textos positivos
    for _ in range(100):
        n_palabras = np.random.randint(3, 8)
        palabras = np.random.choice(palabras_positivas + ['producto', 'servicio'], n_palabras)
        textos.append(' '.join(palabras))
        etiquetas.append('positivo')
    
    # Generar textos negativos
    for _ in range(100):
        n_palabras = np.random.randint(3, 8)
        palabras = np.random.choice(palabras_negativas + ['producto', 'servicio'], n_palabras)
        textos.append(' '.join(palabras))
        etiquetas.append('negativo')
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=0.3, random_state=42, stratify=etiquetas
    )
    
    # Diferentes configuraciones para comparar
    configuraciones = [
        ('Count + Alpha=1.0', CountVectorizer(), MultinomialNB(alpha=1.0)),
        ('Count + Alpha=0.1', CountVectorizer(), MultinomialNB(alpha=0.1)),
        ('TF-IDF + Alpha=1.0', TfidfVectorizer(), MultinomialNB(alpha=1.0)),
    ]
    
    resultados = {}
    
    for nombre, vectorizer, classifier in configuraciones:
        # Pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        # Entrenar en todo el conjunto de entrenamiento
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        resultados[nombre] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': np.mean(y_pred == y_test),
            'y_pred': y_pred
        }
    
    # Mostrar resultados
    print("=== COMPARACIÓN DE CONFIGURACIONES ===")
    for nombre, res in resultados.items():
        print(f"{nombre}:")
        print(f"  CV Accuracy: {res['cv_mean']:.3f} (+/- {res['cv_std']*2:.3f})")
        print(f"  Test Accuracy: {res['test_accuracy']:.3f}")
        print()
    
    return resultados, X_test, y_test

# Ejemplo 4: Análisis de features importantes
def analizar_features_importantes():
    # Crear dataset de ejemplo
    textos = [
        "excelente producto recomiendo calidad",
        "terrible servicio malo experiencia",
        "bueno funciona bien satisfecho",
        "pésimo no funciona problemas",
        "fantástico supera expectativas",
        "horrible decepcionante no recomiendo"
    ]
    
    etiquetas = ['positivo', 'negativo', 'positivo', 'negativo', 'positivo', 'negativo']
    
    # Entrenar modelo
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(textos)
    
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X, etiquetas)
    
    # Obtener nombres de features
    feature_names = vectorizer.get_feature_names_out()
    
    # Log-probabilidades por clase
    print("=== FEATURES MÁS IMPORTANTES POR CLASE ===")
    for i, clase in enumerate(classifier.classes_):
        log_probs = classifier.feature_log_prob_[i]
        # Ordenar features por log-probabilidad
        indices_ordenados = np.argsort(log_probs)[::-1]
        
        print(f"\nClase '{clase}' - Top 5 palabras:")
        for j in range(min(5, len(indices_ordenados))):
            idx = indices_ordenados[j]
            print(f"  {feature_names[idx]}: {np.exp(log_probs[idx]):.4f}")

# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== EJEMPLO 1: CLASIFICACIÓN BÁSICA ===")
    pipeline = ejemplo_clasificacion_texto()
    
    print("\n=== EJEMPLO 2: ANÁLISIS COMPLETO ===")
    resultados, X_test, y_test = analisis_completo()
    
    print("\n=== EJEMPLO 3: ANÁLISIS DE FEATURES ===")
    analizar_features_importantes()
    
    # Ejemplo de nueva predicción
    print("\n=== EJEMPLO 4: PREDICCIÓN DE NUEVOS TEXTOS ===")
    nuevos_textos = [
        "Este producto es increíblemente bueno",
        "No me gustó nada, muy decepcionante"
    ]
    
    predicciones = pipeline.predict(nuevos_textos)
    probabilidades = pipeline.predict_proba(nuevos_textos)
    
    for texto, pred, prob in zip(nuevos_textos, predicciones, probabilidades):
        print(f"Texto: '{texto}'")
        print(f"Predicción: {pred}")
        print(f"Probabilidades: {dict(zip(pipeline.classes_, prob))}")
        print()