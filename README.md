# Estrategias Decodificacion

El enlace al repositorio de la tarea es el siguiente: [GitHub Repository](https://github.com/MiguelGG03/EstrategiasDecodificacion.git)

## Descripción Detallada de LLM

### Explicación de cómo LLMs, no producen texto directamente, sino logits.

LLM (Large Language Model) es un modelo de lenguaje NPL (Natural Processing Language), que se entrena con conjuntos de datos muy grandes de texto en lenguaje natural utilizando Deep Learning, generalmente reddes neuronales artificiales.
Es utilizado para entender, resumir, generar y predecir contenido.


## Proceso de Generación de texto

### Descripción del proceso

El LLM es un modelo de lenguaje basado en Transformers, generan texto de una manera autoregresiva.Esto significa que se generan token uno a uno, basandose siempre enlos tokens previamente generados.
Este proceso sigue unos pasos:
1. `Codificación de entrada:` La entrada (frase o contenido inicial) se codificaen una representación numérica utilizando un "tokenizador", donde cada palabra o caracter pasa a ser un token específico. Esta representación son los `input_ids`.
2. `Generación de Tokens:` El modelo de lenguaje toma los `input_ids` y comienza a generar tokens con ellos. En cada paso que da, el modelo predice la probabilidad de la siguiente palabra o token en función de las palabras anteriores de la secuencia. 
3. `Muestreo de Tokens:` Para generar el siguiente token, el modelo puede usar varias estrategias, como "greedy search" (elegir la palabra más probable en cada paso), "beam search" (mantener varias secuencias candidatas y elegir la mejor), o "sampling" (muestreo aleatorio basado en las probabilidades de los tokens).

4. `Decodificación de Tokens:` Los tokens generados se decodifican en texto legible para los humanos utilizando el tokenizador inverso. Esto puede incluir la eliminación de tokens especiales y la reconversión de tokens numéricos a palabras.

### Representación ilustrativa

![Tokenización](imgs/tokenización.png)