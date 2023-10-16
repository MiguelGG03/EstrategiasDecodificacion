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

LLM (Large Language Model) es un modelo de lenguaje NPL (Natural Processing Language), que se entrena con conjuntos de datos muy grandes de texto en lenguaje natural utilizando Deep Learning, generalmente redes neuronales artificiales.
Es utilizado para entender, resumir, generar y predecir contenido.

### Análisis de cómo los logits se traducen en texto

Los logits se traducen en texto a través de un proceso conocido como decodificación.

La decodificación es la etapa en la que se selecciona un token específico a partir de los valores de los logits para construir una secuencia de texto coherente.

####  Análisis detallado de cómo los logits se traducen en texto

1. `Generación de Logits:` Primero, un modelo de lenguaje generativo, como GPT-2, produce una secuencia de logits. Cada logit corresponde a un token en el vocabulario del modelo y representa cuán probable es que ese token sea el siguiente en la secuencia de texto.

2. `Normalización:` Los logits no son directamente interpretables como probabilidades, ya que no suman 1. Para hacer que los logits sean interpretables como probabilidades, se aplica una función de activación llamada "softmax" a los valores. El softmax transforma los logits en una distribución de probabilidad, donde cada valor representa la probabilidad de que un token en el vocabulario sea el siguiente en la secuencia.

3. `Selección de Token:` Una vez que los logits se han convertido en una distribución de probabilidad, se puede seleccionar el próximo token de varias maneras. Las estrategias comunes incluyen:

    - `Argmax:` Se elige el token con la probabilidad más alta. Esto genera una secuencia de texto determinista y puede llevar a una falta de diversidad en el texto generado.
    - `Muestreo Aleatorio:` Se elige un token aleatoriamente de acuerdo con las probabilidades estimadas por el softmax. Esto introduce variabilidad en el texto generado y puede hacer que sea más diverso.
4. `Consideración del Contexto:` La elección del token siguiente suele basarse en el contexto de la secuencia de texto anterior. El modelo utiliza la secuencia generada hasta ese punto para influir en la selección del próximo token, lo que le permite generar texto coherente y relevante.

5. `Generación Continua:` El proceso de selección y generación de tokens se repite para generar una secuencia más larga de texto. Cada nuevo token generado se agrega a la secuencia anterior, y los logits se recalculan teniendo en cuenta la secuencia ampliada.

6. `Fin de la Generación:` La generación de texto puede finalizar según ciertos criterios, como alcanzar una longitud deseada, encontrar un token especial de finalización o cuando el modelo considera que se ha completado el texto de manera coherente.

En resumen, los logits se traducen en texto a través del proceso de decodificación, que implica la selección de tokens basados en las probabilidades calculadas por el modelo. La elección del token siguiente se basa en el contexto de la secuencia de texto generada hasta ese punto, lo que permite generar texto coherente y significativo. La forma en que se eligen los tokens y se utiliza el contexto influye en la calidad y la fluidez del texto generado por un modelo de lenguaje.

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

## Estrategias de Decodificación

### Descripción y análisis de la búsqueda codiciosa y la búsqueda de haz

#### `- La búsqueda codiciosa:`

La búsqueda codiciosa (también conocida como greedy search) es un algoritmo de búsqueda que se utiliza comúnmente en procesos de decodificación en modelos de lenguaje generativo, como los Transformers utilizados en los LLM (Modelos de Lenguaje con Atención, por sus siglas en inglés) como GPT-2 y GPT-3.

##### Descripción

La búsqueda codiciosa es un enfoque de decodificación que se utiliza para generar secuencias de texto de manera iterativa. Funciona de la siguiente manera:

1. Comienza con una secuencia de texto inicial (por lo general, una o varias palabras iniciales) y un modelo de lenguaje entrenado.

2. Se pasa la secuencia inicial al modelo de lenguaje, que produce una distribución de probabilidad sobre el vocabulario para el siguiente token en la secuencia.

3. El token con la probabilidad más alta (el token argmax) se selecciona como el siguiente token en la secuencia.

4. El token seleccionado se agrega a la secuencia y se repite el proceso. La secuencia se amplía en un token a la vez.

5. Este proceso continúa hasta que se alcanza una longitud deseada, se encuentra un token de finalización específico o se utiliza algún otro criterio de finalización.

##### Análisis

La búsqueda codiciosa es un enfoque simple y eficiente para generar texto, pero tiene algunas limitaciones notables:

1. `Determinismo:` Dado que la búsqueda codiciosa siempre selecciona el token más probable en cada paso, es un proceso determinista. Esto significa que si se inicia con la misma secuencia inicial, el modelo siempre generará la misma secuencia de salida. Esto puede llevar a una falta de diversidad en el texto generado y hacer que sea predecible.

2. `Falta de Exploración:` La búsqueda codiciosa no explora diferentes caminos de generación. Se adhiere a la ruta más probable en cada paso. Esto puede llevar a que el modelo ignore opciones menos probables que podrían resultar en un texto más interesante o creativo.

3. `Coherencia Limitada:` Aunque la búsqueda codiciosa genera texto coherente a nivel gramatical y semántico, no siempre produce texto con coherencia global. Puede haber problemas de coherencia a medida que la secuencia se alarga, ya que cada elección se basa únicamente en el token anterior.

4. `Incapacidad para Retroceder:` Una vez que se ha seleccionado un token en la búsqueda codiciosa, no se puede retroceder y cambiar esa decisión. Esto puede llevar a problemas si el modelo se da cuenta de un error o una elección incorrecta en pasos anteriores.

##### Conclusión

A pesar de estas limitaciones, la búsqueda codiciosa sigue siendo un enfoque ampliamente utilizado debido a su simplicidad y eficiencia. Sin embargo, en aplicaciones donde la diversidad y la creatividad son fundamentales, suelen emplearse enfoques de generación más avanzados, como el muestreo aleatorio (sampling) o la búsqueda estocástica. Estos enfoques permiten una generación más variada y exploratoria, pero a menudo son más costosos en cuanto a tiempo de cómputo.

#### `- La búsqueda de haz:`


La búsqueda de haz (beam search) es un algoritmo de búsqueda utilizado en procesos de decodificación en modelos de lenguaje generativo, como los Modelos de Lenguaje con Atención (LLM).

##### Descripción
La búsqueda de haz es un enfoque de decodificación que se utiliza para generar secuencias de texto de manera más sofisticada que la búsqueda codiciosa. Funciona de la siguiente manera:

1. Comienza con una secuencia de texto inicial (por lo general, una o varias palabras iniciales) y un modelo de lenguaje entrenado.

2. Se pasa la secuencia inicial al modelo de lenguaje, que produce una distribución de probabilidad sobre el vocabulario para el siguiente token en la secuencia.

3. En lugar de seleccionar solo el token más probable, la búsqueda de haz mantiene un conjunto (o "haz") de las N secuencias más probables en cada paso. N se conoce como el ancho del haz y es un parámetro ajustable.

4. Se calcula la probabilidad conjunta de cada secuencia en el haz tomando en cuenta las probabilidades de los tokens anteriores y las probabilidades actuales del modelo.

5. Se seleccionan las N secuencias con las probabilidades conjuntas más altas para continuar generando. Estas N secuencias se convierten en el nuevo haz.

6. Este proceso se repite hasta que se alcanza una longitud deseada, se encuentra un token de finalización específico o se utiliza algún otro criterio de finalización.

##### Análisis

La búsqueda de haz aborda algunas de las limitaciones de la búsqueda codiciosa y tiene varias ventajas:

1. `Diversidad en la Generación:` Al mantener múltiples secuencias en el haz, la búsqueda de haz puede generar texto más diverso y variado. Esto evita la repetición y la previsibilidad de la búsqueda codiciosa.

2. `Coherencia Mejorada:` Las secuencias en el haz suelen ser más coherentes en términos de contenido y contexto, ya que se consideran múltiples opciones en cada paso de generación.

3. `Mayor Flexibilidad:` El ancho del haz (el número de secuencias en el haz) es un parámetro ajustable, lo que permite a los usuarios controlar el equilibrio entre diversidad y calidad en la generación.

Sin embargo, la búsqueda de haz también presenta algunas `limitaciones`:

1. `Mayor Uso de Recursos:` Aumentar el ancho del haz requiere más recursos computacionales, ya que se deben mantener y calcular las probabilidades de múltiples secuencias. Esto puede hacer que la generación sea más lenta.

2. `Riesgo de Bloqueo en Mínimos Locales:` La búsqueda de haz tiende a converger hacia secuencias comunes y seguras, lo que puede llevar a que el texto generado sea conservador y carezca de originalidad.

3. `Complejidad en la Implementación:` Implementar una búsqueda de haz eficiente puede ser más complejo que la búsqueda codiciosa debido a la necesidad de rastrear múltiples secuencias y sus probabilidades conjuntas.

##### Conclusión

 La búsqueda de haz es un enfoque intermedio entre la búsqueda codiciosa y la generación completamente aleatoria. Ofrece un equilibrio entre diversidad y coherencia en el texto generado y se utiliza comúnmente en aplicaciones de generación de lenguaje donde se requiere una salida de alta calidad y variedad.


### Discusión sobre muestreo con top-k y muestreo de núcleo.

El muestreo con top-k (top-k sampling) y el muestreo de núcleo (nucleus sampling) son dos técnicas de muestreo utilizadas en modelos de lenguaje generativos, como los Modelos de Lenguaje con Atención (LLM), para controlar la generación de texto. A continuación, se presenta una discusión sobre ambas técnicas y sus diferencias.

#### Muestreo con Top-k:

El muestreo con top-k es una técnica que se utiliza para limitar las opciones de generación a un subconjunto de tokens con las probabilidades más altas en cada paso. El proceso se realiza de la siguiente manera:

1. En cada paso de generación, se calculan las probabilidades de todos los tokens en el vocabulario en función del contexto actual.

2. Luego, se seleccionan los k tokens con las probabilidades más altas. El valor de k es un parámetro ajustable y controla cuántas opciones se consideran.

3. Finalmente, se elige un token aleatoriamente de entre esos k tokens con una probabilidad uniforme o ponderada por las probabilidades.

#### Muestreo de Núcleo (Nucleus Sampling):

El muestreo de núcleo es una técnica que se utiliza para mantener un "núcleo" de tokens cuyas probabilidades acumuladas superan un cierto umbral. El proceso se realiza de la siguiente manera:

1. Se calculan las probabilidades de todos los tokens en el vocabulario en función del contexto actual.

2. Se ordenan los tokens en función de sus probabilidades de manera descendente.

3. Se seleccionan los tokens hasta que la suma de sus probabilidades acumuladas alcance un cierto umbral, definido por un parámetro ajustable, a menudo denominado "núcleo" (por ejemplo, 0.9 significa que se considerarán tokens cuyas probabilidades acumuladas sumen al menos el 90% de la masa total de probabilidad).

4. Luego, se elige un token aleatoriamente de entre este conjunto seleccionado con una probabilidad uniforme o ponderada por las probabilidades.

#### Comparación:

- `Diversidad:` El muestreo con top-k tiende a ser más restrictivo en términos de diversidad, ya que limita las opciones a un número fijo de tokens con las probabilidades más altas. En contraste, el muestreo de núcleo permite una mayor diversidad, ya que selecciona tokens en función de un umbral acumulado.

- `Control de Calidad:` El muestreo con top-k puede garantizar una cierta calidad en la generación, ya que se eligen las opciones más probables. El muestreo de núcleo ofrece un mayor equilibrio entre calidad y diversidad.

- `Parámetros Ajustables:` Ambas técnicas son parametrizables (k en top-k y el umbral en núcleo). Los valores de estos parámetros influyen en la generación.

- `Uso de Recursos:` El muestreo de núcleo puede ser más eficiente en términos de recursos que el top-k, ya que no requiere ordenar y seleccionar explícitamente los k tokens.

#### Conclusión
En resumen, tanto el muestreo con top-k como el muestreo de núcleo son útiles para controlar la generación de texto en modelos de lenguaje generativos. La elección entre ellos depende de los objetivos específicos de generación y de la cantidad de diversidad y calidad deseada en el texto generado.

## Hiperparámetros y su Manipulación

### Discusión sobre temperatura, num_beams, top_k y top_p.

- `Temperatura (Temperature):` La temperatura es un hiperparámetro que controla la aleatoriedad en la generación de texto. Un valor alto de temperatura, como 2, hace que la generación sea más aleatoria, lo que puede dar lugar a resultados creativos pero a veces incoherentes. Un valor bajo de temperatura, como 0.5, hace que la generación sea más determinista y centrada en las palabras más probables.

- `num_beams:` Num_beams controla la cantidad de "rayos" o "beams" que el modelo considera al generar texto. Un valor mayor aumenta la coherencia y calidad, pero puede reducir la diversidad. Es útil para la generación de texto coherente y bien estructurado.

- `top_k:` Top_k especifica la cantidad de tokens con las probabilidades más altas que se consideran en cada paso de generación. Limita las opciones a un conjunto fijo de tokens, lo que puede ser útil para evitar generaciones incoherentes y raras.

- `top_p (núcleo):` Top_p, a menudo llamado "núcleo," establece un umbral acumulado para la suma de las probabilidades de los tokens en cada paso. Solo se consideran tokens cuyas probabilidades acumuladas superan ese umbral. Es útil para equilibrar calidad y diversidad en la generación.

La elección de estos hiperparámetros depende de los objetivos específicos de generación. Una temperatura alta con un valor bajo de num_beams puede generar texto creativo pero potencialmente incoherente. Por otro lado, un num_beams alto con un top_k o top_p ajustado puede producir texto altamente coherente. Los usuarios ajustan estos hiperparámetros para lograr el equilibrio deseado entre coherencia, calidad y diversidad en los resultados generados.

### Ejemplos de cómo manipular estos hiperparámetros puede afectar la salida.

1. `Temperatura (Temperature):`

    - `Alta temperatura (por ejemplo, 2.0):` Generará texto muy aleatorio, donde las predicciones son menos predecibles. Por ejemplo, "El cielo azul es como un río de ensueño que baila con los elefantes morados."
    - `Baja temperatura (por ejemplo, 0.5):` Generará texto más determinista y coherente, donde las predicciones se centran en las opciones más probables. Por ejemplo, "El cielo azul es sereno y claro."
2. `num_beams:`

    - `Num_beams bajo (por ejemplo, 1):` Generará una sola secuencia de texto y puede carecer de diversidad. Por ejemplo, "El gato está en la alfombra."
    - `Num_beams alto (por ejemplo, 5):` Generará múltiples secuencias candidatas y seleccionará la mejor. Esto puede dar lugar a una mayor coherencia y calidad. Por ejemplo, "El gato duerme placenteramente en la suave alfombra mientras el perro juega cerca."
3. `top_k:`

    - `Top_k bajo (por ejemplo, 10):` Limitará las opciones a un pequeño número de palabras, lo que puede llevar a resultados más predecibles y repetitivos. Por ejemplo, "El clima es soleado y el cielo está despejado."
    - `Top_k alto (por ejemplo, 50):` Considerará una gama más amplia de palabras, lo que puede aumentar la diversidad en la generación. Por ejemplo, "El clima es soleado y maravilloso, con un cielo azul y claro."
4. `top_p (núcleo):`

    - `Top_p bajo (por ejemplo, 0.2):` Establece un umbral estricto y limitará las opciones a las palabras más probables, generando textos muy centrados. Por ejemplo, "El clima es soleado."
    - `Top_p alto (por ejemplo, 0.8):` Permite una gama más amplia de opciones y puede llevar a textos más largos y diversos. Por ejemplo, "El clima es increíblemente hermoso, con un cielo despejado y un sol radiante que calienta el día."

## Reflexión y Conclusiones

### Reflexiones

Las estrategias de decodificación desempeñan un papel fundamental en la generación de texto mediante Modelos de Lenguaje con Transformers (LLMs) como GPT-2. Su importancia radica en que permiten ajustar y controlar el proceso de generación de texto de manera que se adecúe a los objetivos y requisitos específicos. A continuación, se destacan algunas reflexiones sobre su impacto y relevancia:

- `Control de la Creatividad:` Las estrategias de decodificación ofrecen un control significativo sobre la creatividad del modelo. Por ejemplo, al ajustar la temperatura, es posible equilibrar entre generación altamente creativa y coherencia. Esto es crucial en aplicaciones como la escritura creativa, donde se busca contenido original pero no caótico.

- `Coherencia y Calidad del Texto:` Estrategias como la búsqueda de haz (beam search) tienden a generar texto más coherente y de alta calidad al considerar múltiples secuencias candidatas y elegir la mejor. Esto es beneficioso en aplicaciones donde la precisión y la fluidez son fundamentales.

- `Diversidad y Exploración:` Para tareas que requieren diversidad en la generación, como la recomendación de productos o la creación de variantes de texto, estrategias como el muestreo con top-k y el muestreo de núcleo permiten explorar diferentes opciones. Esto evita la generación repetitiva y monótona.

- `Alineación con el Contexto:` La elección de la estrategia de decodificación puede influir en la coherencia contextual. Por ejemplo, el muestreo de núcleo es útil cuando se desea que las palabras generadas estén más alineadas con el contexto previo.

- `Eficiencia y Recursos Computacionales:` La elección de la estrategia puede afectar los recursos computacionales necesarios. Por ejemplo, la búsqueda de haz es más costosa en términos de tiempo de cómputo que la búsqueda codiciosa. Esto es relevante en entornos donde se necesita una generación rápida.

- `Personalización y Adaptación:` La elección de la estrategia de decodificación puede ser personalizada según el caso de uso específico. Diferentes aplicaciones pueden requerir un enfoque único para lograr los mejores resultados.

- `Interacción con el Usuario:` En aplicaciones de chatbots o asistentes virtuales, la elección de la estrategia de decodificación puede influir en la naturaleza de la conversación. Estrategias que favorecen la coherencia pueden mejorar la interacción con el usuario.

En resumen, las estrategias de decodificación son esenciales en la generación de texto y permiten adaptar los modelos de lenguaje a diversas aplicaciones y requisitos. La selección adecuada de estas estrategias es clave para lograr un equilibrio entre coherencia, creatividad, calidad y eficiencia en la generación de texto.

### Conclusiones

- Las estrategias de decodificación desempeñan un papel fundamental en la generación de texto y permiten ajustar la salida de los modelos de lenguaje generativos a los objetivos específicos.

- La elección de hiperparámetros como la temperatura, num_beams, top_k y top_p tiene un impacto significativo en la coherencia, creatividad y calidad del texto generado.

- Estrategias como la búsqueda de haz y el muestreo de núcleo ofrecen un equilibrio entre coherencia y diversidad en la generación de texto.

- Las aplicaciones de los LLMs son variadas, desde escritura creativa hasta chatbots y traducción automática. La elección de estrategias de decodificación debe ser específica para cada caso de uso.