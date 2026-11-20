# Guion: Predicción del abandono de clientes con deep learning

Artículo: Agrawal, Das, Gaikwad y Dhage (2018), *Customer Churn Prediction Modelling Based on Behavioural Patterns Analysis using Deep Learning*, ICSCEE 2018 (IEEE).

Duración estimada: **≈ 9:30 min** (12 diapositivas). El mismo texto está en las notas del orador del PowerPoint (vista Moderador).
Lo que va entre paréntesis son aclaraciones para ti, no hace falta decirlas.

| # | Diapositiva | Tiempo | Empieza en |
|---|---|---|---|
| 1 | Predicción del abandono de clientes con deep learning | 20 s | 0:00 |
| 2 | ¿Qué es el churn y por qué importa? | 50 s | 0:20 |
| 3 | Estado del arte: de boosting a deep learning | 55 s | 1:10 |
| 4 | Metodología propuesta | 45 s | 2:05 |
| 5 | Dataset: IBM Watson Telco Customer Churn | 45 s | 2:50 |
| 6 | Preprocesamiento: de texto a números | 60 s | 3:35 |
| 7 | Modelo: red neuronal feed-forward | 55 s | 4:35 |
| 8 | Resultados: 80,03 % de accuracy | 55 s | 5:30 |
| 9 | ¿Qué factores explican el abandono? | 55 s | 6:25 |
| 10 | Análisis crítico | 70 s | 7:20 |
| 11 | Conclusiones e implicaciones para el CRM | 50 s | 8:30 |
| 12 | ¿Preguntas? | 10 s | 9:20 |

---

## 1. Predicción del abandono de clientes con deep learning  ·  ~20 s

Buenos días. Voy a presentar el artículo «Customer Churn Prediction Modelling Based on Behavioural Patterns Analysis using Deep Learning», de Agrawal, Das, Gaikwad y Dhage, presentado en la conferencia ICSCEE de 2018 del IEEE. Trata de cómo predecir qué clientes van a abandonar una compañía usando redes neuronales y, sobre todo, de entender por qué se van.

## 2. ¿Qué es el churn y por qué importa?  ·  ~50 s

Empecemos por el concepto. El churn, o abandono, es el conjunto de clientes que dejan de usar un servicio durante un periodo concreto. En el dataset del artículo, más o menos uno de cada cuatro clientes se marchó en el último mes (este 26,5 % sale del dataset público, no del artículo).

¿Por qué importa? Primero, porque afecta directamente a la rentabilidad: para crecer hay que captar clientes más rápido de lo que se pierden. Segundo, porque predecirlo es difícil: los que se van son minoría, así que los datos están desbalanceados. Y tercero, porque con tanto volumen de datos diseñar buenas variables a mano es costoso. Ahí ven los autores la ventaja del deep learning: la red aprende sus propias representaciones y da poco peso a lo irrelevante.

## 3. Estado del arte: de boosting a deep learning  ·  ~55 s

Los autores revisan cuatro trabajos previos. En gris, dos de machine learning clásico: Lu y colaboradores usan boosting sobre regresión logística con unos 7.200 clientes, de los que solo 678 abandonan; y Wu y Meng atacan el desbalanceo en comercio electrónico con SMOTE mejorado —que genera ejemplos sintéticos de la clase minoritaria— más AdaBoost.

En verde, dos de deep learning: Spanoudes y Nguyen evitan el feature engineering con una red de cuatro capas y obtienen entre un 70 y un 80 % de acierto; y Castanedo y colaboradores trabajan con 1,2 millones de clientes de telefonía, aunque su modelo no escala bien a interacciones largas.

La aportación de este artículo es juntar las dos cosas: una red neuronal que predice el churn y, además, señala qué factores lo provocan.

## 4. Metodología propuesta  ·  ~45 s

La metodología sigue un pipeline de seis pasos. Se recogen los registros de clientes, se limpian y codifican, se seleccionan las variables y con ellas se entrena una red neuronal multicapa con el 80 % de los datos. Después el modelo clasifica a cada cliente como churn o no churn y se valida con el 20 % restante.

El último paso, en naranja, es lo que diferencia el trabajo: un análisis de correlación para ver qué variables pesan más en el abandono. Así el sistema da dos salidas: una predicción, que es la probabilidad de que el cliente se vaya, y un diagnóstico de qué factores empujan o frenan ese abandono.

## 5. Dataset: IBM Watson Telco Customer Churn  ·  ~45 s

Los datos son el dataset público IBM Watson Telco Customer Churn: 7.043 clientes de una operadora, cada uno etiquetado según si abandonó o no en el último mes.

Las variables se agrupan en cuatro bloques: datos demográficos, como el género o si tiene pareja o personas a cargo; los servicios contratados, desde teléfono e internet hasta soporte técnico o streaming; el perfil de facturación, con el tipo de contrato, el método de pago y los cargos mensual y total; y la antigüedad, es decir, cuántos meses lleva con la compañía. Solo hay tres variables numéricas y las tres están muy concentradas en valores bajos, lo que obliga a normalizarlas.

## 6. Preprocesamiento: de texto a números  ·  ~60 s

Para que la red pueda trabajar con estos datos hay que pasarlos a números. Primero se limpian: se quitan columnas sin valor predictivo, como el identificador del cliente, y los registros incompletos. Las variables binarias, tipo sí o no, se convierten en ceros y unos con label encoding. Pero esa técnica impone un orden, y en variables con varias categorías ese orden no tiene sentido.

Por eso, para ellas se usa one-hot encoding: una columna binaria por categoría. A la derecha tenéis el ejemplo del propio artículo con el tipo de contrato: una sola columna de texto se convierte en tres columnas —mes a mes, un año y dos años— con un uno en la que corresponde.

Por último, la antigüedad, que va de 0 a 72 meses, se discretiza en tramos por años, del 1 al 6.

## 7. Modelo: red neuronal feed-forward  ·  ~55 s

El modelo es una red neuronal feed-forward secuencial implementada en Keras. La entrada son las variables ya preprocesadas. La primera capa densa tiene 16 neuronas con activación ReLU; después hay otra capa densa con ReLU, y entre ellas capas de dropout del 10 %: durante el entrenamiento se apaga al azar una de cada diez neuronas, para que la red no dependa de neuronas concretas y no sobreajuste. La salida es una única neurona con activación sigmoide, que devuelve un valor entre 0 y 1: la probabilidad de churn.

En cuanto al entrenamiento: optimizador Nadam —que es Adam con momento de Nesterov y fue el que mejor funcionó—, 100 épocas, batch de 10, pesos inicializados de forma uniforme y una división 80/20 entre entrenamiento y validación.

(Si preguntan por la segunda capa densa: el artículo no indica cuántas neuronas tiene.)

## 8. Resultados: 80,03 % de accuracy  ·  ~55 s

¿Y qué resultados obtiene? Un 80,03 % de accuracy sobre el conjunto de validación, que según los autores mejora lo que encontraron en los trabajos revisados. Pero lo más interesante es lo que dicen los datos sobre el comportamiento de los clientes.

En el gráfico central, la antigüedad agrupada en años: en el primer año prácticamente se van tantos clientes como se quedan, y a partir de ahí el abandono cae en picado; la mayoría de los que se van llevan menos de 18 meses.

En el de la derecha, cargo mensual frente a antigüedad, con 500 clientes al azar: los puntos naranjas, los que se van, se concentran abajo a la derecha, es decir, clientes recientes con facturas altas. Y aunque un cliente lleve años, si su factura es alta también acaba marchándose.

## 9. ¿Qué factores explican el abandono?  ·  ~55 s

Para identificar los factores, los autores calculan la correlación de cada variable con la etiqueta de churn. Un valor positivo significa que la variable favorece el abandono; uno negativo, que lo frena. Aquí he resumido los más claros de su figura 7 (son valores aproximados leídos del gráfico).

Lo que más empuja a irse es estar en el primer año, tener un contrato mes a mes y tener internet de fibra óptica. Lo que más retiene es un contrato de dos años y llevar más de cinco años en la compañía. Y hay variables que apenas influyen, como la factura en papel o no tener servicio telefónico.

La lectura de negocio es clara: el compromiso, tanto por contrato como por tiempo, retiene.

## 10. Análisis crítico  ·  ~70 s

Ahora, una lectura crítica.

Primero, el 80 % no es tanto como parece. En este dataset alrededor del 73,5 % de los clientes no abandona, así que un clasificador que diga siempre «no churn» ya acierta ese 73,5 %. La red solo mejora unos seis puntos y medio.

Segundo, solo se reporta accuracy: faltan la matriz de confusión, precision, recall, F1 o AUC, que son las métricas importantes con clases desbalanceadas y que el propio artículo cita en el estado del arte.

Tercero, se evalúa sobre el conjunto de validación, sin test independiente y sin comparar con modelos más sencillos, como una regresión logística o un boosting.

Y cuarto, los factores no salen de la red, sino de una correlación univariante que se podría calcular sin ella. Además hay una errata: el texto dice que el contrato mensual y la fibra «previenen» el churn, y su propia figura muestra lo contrario.

## 11. Conclusiones e implicaciones para el CRM  ·  ~50 s

Para cerrar. El artículo aporta un pipeline completo, de los datos a la predicción y a los factores, sobre un dataset público, y deja un perfil de riesgo muy claro: cliente nuevo, con contrato mensual, fibra y factura alta.

En términos de CRM, eso se traduce en acciones concretas —que son mi propuesta, no del artículo—: reforzar el onboarding durante el primer año, incentivar el paso a contratos anuales o bianuales, revisar precio y calidad de la fibra y lanzar alertas cuando la factura de un cliente sube.

Como trabajo futuro, los autores proponen relacionar el churn con el valor de vida del cliente, diseñar modelos de retención y comparar el coste de captar frente al de retener.

Si me quedo con una idea: el compromiso, por contrato y por tiempo, es la mejor palanca de retención.

## 12. ¿Preguntas?  ·  ~10 s

Y con esto termino. Muchas gracias por vuestra atención; si tenéis alguna pregunta, encantado de responderla.

---

## Datos clave para recordar

- **7.043** clientes (IBM Watson Telco Customer Churn); etiqueta = abandonó en el último mes.
- Red: Dense 16 (ReLU) → Dropout 10 % → Dense (ReLU) → Dropout 10 % → Dense 1 (sigmoide). Nadam, 100 épocas, batch 10, 80/20.
- Resultado: **80,03 %** de accuracy en validación.
- Impulsan el churn: 1.er año (≈ +0,48), contrato mes a mes (≈ +0,34), fibra óptica (≈ +0,28).
- Retienen: contrato de 2 años (≈ −0,36), más de 5 años de antigüedad (≈ −0,29).
- Línea base trivial (siempre «no churn»): **≈ 73,5 %** (dato del dataset público, no del artículo).

## Posibles preguntas (respuestas sugeridas)

**¿Tiene sentido usar deep learning con 7.000 filas de datos tabulares?**
Es discutible: con datos tabulares pequeños, modelos como gradient boosting o una regresión logística suelen rendir igual o mejor y son más interpretables. El artículo no hace esa comparación, y esa es una de sus debilidades.

**¿Cómo tratarías el desbalanceo de clases?**
Con sobremuestreo (SMOTE, como Wu y Meng), pesos de clase en la función de pérdida o ajustando el umbral de decisión, y evaluando con recall, F1 o AUC en lugar de solo accuracy.

**¿Qué métrica es la más útil para un equipo de CRM?**
Sobre todo el recall de la clase churn (no dejar escapar a clientes en riesgo), equilibrado con la precision, porque cada acción de retención tiene un coste. También el lift en el top de clientes con más riesgo.

**¿Por qué la fibra óptica aumenta el abandono?**
El artículo no lo explica. Posibles hipótesis: precio más alto, más competencia en ese segmento o problemas de calidad. Además, es una correlación, no una relación causal.

**¿Cuántas neuronas tiene la segunda capa densa?**
El artículo no lo indica; solo especifica las 16 neuronas de la primera capa y la neurona de salida.
