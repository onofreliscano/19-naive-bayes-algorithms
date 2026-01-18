# STUDY_NOTES — Naive Bayes (Sentiment Analysis)

## 1) Objetivo del proyecto
- Construir un clasificador de sentimiento (0 = negativo, 1 = positivo) usando reseñas de Google Play.
- Convertir texto a variables numéricas y entrenar modelos de clasificación.
- Comparar Naive Bayes vs alternativas y guardar el mejor pipeline.

## 2) Dataset
- Fuente: playstore_reviews.csv (repo 4Geeks).
- Variables originales:
  - package_name (categórica)
  - review (texto)
  - polarity (label 0/1)
- Decisión: eliminar `package_name` porque no aporta al sentimiento.

## 3) Preprocesamiento de texto
- Limpieza básica:
  - `strip()` + `lower()` para normalizar.
  - Manejo de nulos y coerción de `polarity` a int.
- Split:
  - train/test 80/20 con `random_state=42` y `stratify` para estabilidad.

## 4) Vectorización
- CountVectorizer:
  - Convierte texto a matriz de conteos de palabras.
  - Ideal para Naive Bayes (especialmente MultinomialNB).
- TF-IDF:
  - Pondera palabras por importancia (frecuencia vs rareza).
  - Suele ayudar a modelos lineales en NLP.

---

## 5) Modelado con Naive Bayes (comparación)
Se entrenaron las 3 variantes principales:

### MultinomialNB (conteos / frecuencias discretas)
- Mejor ajuste teórico para word counts.
- Resultado test (accuracy): **0.8547**

### BernoulliNB (presencia/ausencia)
- Útil cuando importa “si aparece” más que “cuántas veces”.
- Resultado test (accuracy): **0.7821**

### GaussianNB (continuo con normalidad)
- No es la mejor opción para conteos dispersos.
- Resultado test (accuracy): **0.8156**

**NB ganador por accuracy:** `MultinomialNB`

---

## 6) Optimización del Naive Bayes ganador
- Se optimizó con GridSearchCV (CV=5) usando F1:
  - `ngram_range` (1,1) vs (1,2)
  - `min_df` para filtrar ruido
  - `alpha` (suavizado) para evitar probabilidades cero
- Mejor configuración:
  - **{'clf__alpha': 0.5, 'vec__min_df': 2, 'vec__ngram_range': (1, 1)}**
- F1 promedio (CV): **0.7176**
- Accuracy final en test: **0.8268**

---

## 7) Random Forest (requerimiento del proyecto)
- Se probó RandomForest con TF-IDF.
- Accuracy test: **0.8101**
- Nota práctica: en texto, RandomForest suele rendir peor que NB o modelos lineales.

---

## 8) Alternativas para superar Naive Bayes
Modelos típicamente fuertes en NLP (TF-IDF + lineales):

- Logistic Regression (TF-IDF):
  - Accuracy test: **0.7821**
- LinearSVC (TF-IDF):
  - Accuracy test: **0.8268**

**Mejor modelo global por accuracy:** `NaiveBayes_Optimized`

---

## 9) Artefacto final (modelo guardado)
- Se guardó el pipeline completo (vectorizador + modelo) en:
  - `models/NaiveBayes_Optimized.pkl`
- Beneficio: el modelo es “plug & play” para inferencia futura.

---

## 10) Conclusiones (ML/Negocio)
- Naive Bayes es una solución rápida y fuerte para clasificación de texto.
- La elección Multinomial/Bernoulli se valida por resultados + teoría del dato.
- Si el objetivo es exprimir performance en NLP, modelos lineales con TF-IDF suelen ser top.
- Guardar el pipeline completo evita errores de despliegue y mejora mantenibilidad.
