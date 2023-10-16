# Estrategias Decodificacion

El enlace al repositorio de la tarea es el siguiente: [GitHub Repository](https://github.com/MiguelGG03/EstrategiasDecodificacion.git)

## Descripción Detallada de LLM

### Explicación de cómo LLMs, no producen texto directamente, sino logits.
Los Modelos de Lenguaje con Transformers (como GPT-2) no producen texto directamente, sino que generan `"logits"`.

Los `"logits"` son valores numéricos generados por un modelo de lenguaje LLM, o cualquier otro modelo basado en redes neuronales, como parte del proceso de generación de texto.

Estos valores representan una puntuación o una medida de la probabilidad asociada a cada token en el vocabulario del modelo. Los logits se utilizan para determinar cuán probable es que un token en particular sea el siguiente en una secuencia de texto generada.
#### Algunas características importantes de los logits
- Ditribución de probabilidad
- Mapeo de posibilidades
- Selección del siguiente token
- Impacto en la generación de texto

Se generan logits en vez de directamente texto por las siguientes razones fundamentales:

- `Flexibilidad y Control:` Al generar logits en lugar de texto directamente, los modelos tienen más flexibilidad y control en el proceso de generación. Esto permite ajustar la generación para satisfacer diferentes objetivos, como controlar la creatividad del modelo o influir en la dirección de la generación.

- `Probabilidades Asociadas:` Los logits reflejan las probabilidades asociadas a cada token en el vocabulario. Esto proporciona información detallada sobre cuán probable es que cada token sea el siguiente en la secuencia. Al tomar decisiones basadas en estas probabilidades, se puede influir en la coherencia y la calidad del texto generado.

- `Estrategias de Decodificación:` Al generar logits, los modelos pueden utilizar diversas estrategias de decodificación, como muestreo de softmax, muestreo de núcleo superior (top-k sampling), muestreo de núcleo de diversidad, entre otras. Cada una de estas estrategias afecta la forma en que se selecciona el siguiente token y, por lo tanto, el texto resultante.

- `Temperatura:` La temperatura es un hiperparámetro que se puede ajustar al calcular las probabilidades de los logits. Un valor alto de temperatura hace que las distribuciones de probabilidad sean más uniformes y, por lo tanto, genera texto más diverso pero potencialmente incoherente. Un valor bajo de temperatura, en cambio, favorece las opciones más probables, generando texto más coherente pero menos diverso.

- `Control Creativo:` Al trabajar con logits, se pueden diseñar estrategias específicas para controlar la creatividad del modelo. Por ejemplo, se pueden penalizar ciertos tokens o categorías de tokens para evitar respuestas inapropiadas o no deseadas.

- `Escalabilidad:` Los modelos generan texto de manera autoregresiva, lo que significa que toman decisiones secuenciales sobre los tokens siguientes en función del contexto actual. Esta arquitectura es escalable y puede generar secuencias de texto de longitud variable sin necesidad de cambiar la estructura del modelo.

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