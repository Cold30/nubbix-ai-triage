# Nubbix AI Triage

Sistema de clasificación inteligente de tickets de soporte para Nubbix, desarrollado como proyecto integrador del Módulo 1 de AI Engineering de SoyHenry.

## 1. Descripción

Nubbix AI Triage es una aplicación CLI desarrollada en Python que utiliza modelos de lenguaje (LLMs) para analizar consultas de soporte, clasificarlas y generar sugerencias útiles para el equipo de atención al cliente.

El sistema permite intercambiar proveedores LLM sin modificar la lógica de negocio y utiliza Pydantic para validar la estructura de las respuestas generadas.

Gemini y OpenRouter siguen siendo los dos proveedores principales. OpenAI nativo es un tercer proveedor opcional mediante Responses API; no se necesita una clave OpenAI para usar los otros dos.

### Objetivos

- Clasificar consultas de soporte en cuatro categorías.
- Generar respuestas y acciones sugeridas para agentes humanos.
- Validar las respuestas mediante un contrato estructurado.
- Manejar respuestas inválidas mediante reintentos.
- Registrar consumo de tokens, latencia y costos estimados.
- Comparar estrategias de Prompt Engineering mediante evaluaciones reproducibles.

## 2. Tecnologías

- Python 3.12
- Pydantic
- Google Gemini API
- OpenRouter API
- OpenAI Python SDK (Responses API nativa y la integración compatible de OpenRouter)
- PyYAML
- python-dotenv
- pytest

## 3. Arquitectura

El sistema separa las siguientes responsabilidades:

| Componente | Responsabilidad |
|---|---|
| `src/main.py` | Interfaz CLI |
| `src/provider_factory.py` | Selección del proveedor LLM |
| `src/configuration.py` | Validación de credenciales y parámetros de generación |
| `src/gemini_provider.py` | Integración con Gemini |
| `src/openrouter_provider.py` | Integración con OpenRouter |
| `src/openai_provider.py` | Integración opcional con OpenAI Responses API |
| `src/llm_client.py` | Contrato de respuesta normalizada del proveedor |
| `src/prompting.py` | Carga y construcción de prompts YAML |
| `src/triage.py` | Orquestación, validación y reintentos |
| `src/models.py` | Contrato de datos Pydantic |
| `src/guardrails.py` | Validaciones de entrada y salida |
| `src/metrics.py` | Persistencia de métricas JSONL |
| `src/pricing.py` | Estimación de costos |
| `evals/scoring.py` | Cálculo de métricas de clasificación |
| `evals/runner.py` | Ejecución independiente de evaluaciones |
| `evals/adversarial.py` | Evaluación offline de detección y falsos positivos |

### Flujo de clasificación

1. La CLI recibe una consulta y el proveedor seleccionado.
2. Se crea el proveedor mediante la factory.
3. `TriageService` valida la entrada y construye el prompt.
4. El proveedor genera una respuesta.
5. Se registran tokens, latencia y costo estimado.
6. La respuesta se normaliza y valida mediante Pydantic.
7. Se aplican las validaciones de salida.
8. Se devuelve el resultado estructurado.

Ante errores de validación del contrato, el servicio puede realizar nuevos intentos hasta alcanzar el límite configurado.

## 4. Contrato JSON

Cada clasificación devuelve los siguientes campos:

```json
{
  "answer": "Un agente puede revisar los cargos de tu suscripción.",
  "confidence": 0.95,
  "category": "billing",
  "actions": [
    "Verificar el historial de pagos",
    "Comprobar si existen cargos duplicados"
  ]
}
```

El contrato establece:

- `answer`: respuesta sugerida al cliente.
- `confidence`: valor numérico entre 0 y 1.
- `category`: `billing`, `technical`, `account` u `other`.
- `actions`: lista de acciones sugeridas al agente humano.

El valor de `confidence` es una estimación generada por el modelo; no debe interpretarse automáticamente como una probabilidad calibrada.

## 5. Instalación

### Requisitos

- Python 3.12
- Conexión a Internet
- API key del proveedor que se elija: Gemini, OpenRouter u OpenAI opcional

Clonar el repositorio y acceder al directorio del proyecto.

Crear y activar un entorno virtual en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

Copiar `.env.example` a `.env` y configurar las credenciales correspondientes.

Nunca incluir claves privadas en commits o documentación pública.

### Configuración efectiva

Ejecutá los comandos desde la raíz del repositorio. Solo se exige la clave y el modelo del proveedor seleccionado; los valores ausentes o vacíos se rechazan antes de llamar a la API.

| Variable | Valor predeterminado | Uso |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Proveedor de la CLI y del runner; `--provider` tiene prioridad |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Sin valor | Obligatorios al seleccionar Gemini |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Sin valor | Obligatorios al seleccionar OpenRouter |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Sin valor en el código | Obligatorios solo al seleccionar `openai`; `.env.example` propone `gpt-6-luna` |
| `OPENAI_REASONING_EFFORT` | Obligatorio para `openai` | `none`, `low`, `medium`, `high`, `xhigh` o `max`; el ejemplo selecciona `low` |
| `LLM_TEMPERATURE` | `0.1` | Número finito entre 0 y 2; Gemini/OpenRouter lo reciben siempre, OpenAI solo con esfuerzo `none` |
| `LLM_MAX_OUTPUT_TOKENS` | `512` | Entero positivo; Gemini y OpenAI reciben `max_output_tokens`, OpenRouter `max_tokens` |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout por solicitud en segundos, finito y al menos `0.001`; Gemini lo recibe en milisegundos |
| `LLM_MAX_RETRIES` | `2` | Reintentos adicionales por JSON inválido; el ejemplo configura `3` |
| `PROMPT_VERSION` | `triage_v1` | Versión predeterminada de la CLI; el runner exige `--prompt-version` |
| `METRICS_PATH` | `data/metrics.jsonl` | Archivo append-only de métricas |

Completá los modelos con identificadores habilitados para tu cuenta. El identificador Gemini de la evidencia histórica es `gemini-3.5-flash-lite`; no se confirmó su disponibilidad actual. La integración nativa OpenAI valida únicamente `gpt-6-luna`, cuya documentación pública se verificó; otros nombres se rechazan hasta comprobar sus parámetros. Esto no confirma acceso a ese modelo en tu cuenta.

Los límites admitidos por cada modelo pueden ser más estrechos que la validación compartida. El timeout es por solicitud y no una fecha límite global: los reintentos y esperas pueden extender el tiempo total.

Los SDK no agregan reintentos: OpenAI/OpenRouter usan `max_retries=0` y Gemini `HttpRetryOptions(attempts=1)`. `retry_transient()` permite dos reintentos adicionales con esperas de 1 y 2 segundos. HTTP 408, 429, 5xx y fallos temporales de transporte se consideran transitorios; otros errores HTTP son permanentes. En OpenAI, un HTTP 429 por créditos agotados, límites de gasto o cuota se clasifica como permanente, no como rate limit temporal. Un error permanente no se reintenta. Con `LLM_MAX_RETRIES=3`, el límite combinado es de hasta 12 solicitudes de generación si se combinan fallos transitorios y respuestas JSON inválidas.

## 6. Uso de la CLI

### Clasificar con Gemini

```powershell
python -m src.main classify "Me cobraron dos veces la suscripción" --provider gemini
```

### Clasificar con OpenRouter

```powershell
python -m src.main classify "No puedo iniciar sesión" --provider openrouter
```

### Seleccionar una versión del prompt

```powershell
python -m src.main classify "No puedo descargar un reporte" --provider gemini --prompt-version triage_v2
```

### OpenAI opcional: GPT-6 Luna

Configurá `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-6-luna` y `OPENAI_REASONING_EFFORT=low` en tu `.env`. El código no crea credenciales ni comprueba el saldo de tu cuenta. Una suscripción a ChatGPT no implica créditos de API; tampoco se afirma que Luna sea gratuito ni esté habilitado para tu proyecto.

La integración usa `client.responses.create`, el endpoint oficial `https://api.openai.com/v1`, timeout explícito, `service_tier=default` y `store=false`. No agrega herramientas ni cambia los templates de prompts. Pydantic y los guardrails existentes siguen validando el resultado.

Según la [documentación oficial de GPT-6](https://developers.openai.com/api/docs/guides/latest-model), al activar razonamiento se deben omitir controles de muestreo incompatibles. Con esfuerzo distinto de `none` no se envían `temperature`, `top_p` ni logprobs; con `none` se envía la temperatura configurada. La identidad de la evaluación guarda los parámetros efectivos: cambiar una temperatura omitida no invalida el checkpoint, pero cambiar el esfuerzo sí.

El límite de salida incluye razonamiento y texto visible. Con 512 tokens podría agotarse antes de formar el JSON. Para una prueba real se puede configurar `LLM_MAX_OUTPUT_TOKENS=2048`, sin garantía de que alcance para cualquier solicitud y con posible aumento de costo. El máximo documentado del modelo es 128000. Si la respuesta queda incompleta se devuelve un error permanente que indica revisar presupuesto y esfuerzo; no se devuelve texto parcial ni se reintenta ampliando el presupuesto automáticamente.

Estos comandos son **reales, opcionales y potencialmente pagos**; no forman parte de los tests offline:

```powershell
python -m src.main classify "No puedo ingresar a mi cuenta" --provider openai
python -m evals.runner --provider openai --prompt-version triage_v1 --fresh --output evals/results/openai_50_v1.json
python -m evals.runner --provider openai --prompt-version triage_v1 --resume --output evals/results/openai_50_v1.json
```

Las pruebas nativas simuladas se ejecutan sin claves ni conexión:

```powershell
python -m pytest tests/test_openai_provider.py tests/test_openai_configuration.py tests/test_openai_cli.py -q
```

Se verificaron el [modelo y sus límites](https://developers.openai.com/api/docs/models/gpt-6-luna) y los [errores de API](https://developers.openai.com/api/docs/guides/error-codes) el 7 de noviembre de 2026. No se realizaron solicitudes al modelo ni se verificaron permisos, facturación, créditos o restricciones de nivel de servicio de la cuenta. `store=false` no sustituye las políticas de retención de datos del proveedor.

## 7. Estrategias de Prompt Engineering

Se implementaron dos versiones de prompts en español.

### Versión 1 — Zero-shot

Utiliza instrucciones de clasificación, definición de categorías y especificación del contrato JSON, sin ejemplos de tickets.

### Versión 2 — Few-shot

Mantiene las instrucciones principales e incorpora ejemplos etiquetados para las cuatro categorías.

El objetivo experimental es comprobar si los ejemplos mejoran el comportamiento del clasificador frente a la versión base.

## 8. Evaluaciones

El dataset de clasificación `evals/dataset.jsonl` contiene **50 consultas etiquetadas**: los 30 tickets originales (`T001`–`T030`) se conservaron sin cambiar textos, etiquetas ni IDs y se agregaron exactamente 20 (`T031`–`T050`). Son ejemplos sintéticos redactados para el proyecto, no tickets reales ni predicciones de un LLM.

Distribución:

| Categoría | Originales | Nuevos | Total |
|---|---:|---:|---:|
| billing | 8 | 5 | 13 |
| technical | 8 | 5 | 13 |
| account | 7 | 5 | 12 |
| other | 7 | 5 | 12 |
| **Total** | **30** | **20** | **50** |

La ampliación incluye solicitudes formales e informales, errores de escritura, mensajes breves y descripciones detalladas de pymes de LATAM. Los casos mixtos mantienen una sola etiqueta según la resolución principal solicitada, sin inferir causas no expresadas ni clasificar solo por palabras clave. La [política de etiquetado](docs/dataset_labeling.md) documenta los criterios y las razones de cada etiqueta nueva; algunos casos requieren revisión humana antes de congelar el conjunto para la comparación final.

Los ejemplos se revisaron para evitar duplicados y reformulaciones cercanas; las comprobaciones automáticas de unicidad no garantizan diversidad semántica. El conjunto sigue siendo pequeño y de autoría única. Los tickets adversariales permanecen separados en `evals/adversarial.jsonl`.

La **comparación final Gemini sobre 50 tickets** está documentada en [docs/evidence/gemini_50_comparison.md](docs/evidence/gemini_50_comparison.md), con fuentes, hashes, metodología y límites. Las tablas históricas de 30 casos se conservan por separado más abajo. Para repetir la evaluación, usá `--fresh` con archivos de salida nuevos para ambas versiones; no reanudes checkpoints del dataset de 30, porque cambió el hash. Conservá el mismo modelo y configuración y no ajustes las etiquetas según qué prompt gane.

### Ejecutar evaluaciones

Zero-shot:

```powershell
python -m evals.runner --provider gemini --prompt-version triage_v1 --fresh --output evals/results/submission_50_v1.json
```

Few-shot:

```powershell
python -m evals.runner --provider gemini --prompt-version triage_v2 --fresh --output evals/results/submission_50_v2.json
```

Estos comandos hacen llamadas reales y consumen cuota. No se ejecutan como parte de las pruebas offline. Los reportes históricos `evals/results_v1.json` y `evals/results_v2.json` se conservan sin cambios.

`--fresh` exige que la salida no exista: nunca sobrescribe una evaluación. `--resume` exige un checkpoint existente y compatible, y omite los tickets ya registrados:

```powershell
python -m evals.runner --provider gemini --prompt-version triage_v1 --resume --output evals/results/submission_50_v1.json
```

El runner valida campos, categorías e IDs únicos antes de generar. La identidad incluye hashes del dataset y del archivo de prompt, proveedor, modelo, temperatura, límite de tokens, timeout y políticas de reintento. No incluye credenciales. Un cambio de identidad rechaza la reanudación. Los reportes antiguos sin esta identidad no pueden reanudarse: elegí una salida nueva.

Cada checkpoint y el reporte final se escriben mediante un archivo temporal en el mismo directorio y reemplazo atómico. Un fallo del proveedor interrumpe la ejecución conservando los tickets anteriores; el ticket pendiente se intenta al reanudar. Los errores de validación y guardrail guardados cuentan como completados: para repetirlos, iniciá una corrida nueva. No ejecutes dos procesos sobre la misma salida.

La pausa predeterminada es de seis segundos; `--delay-seconds 0` la desactiva. El tiempo de espera entre tickets no modifica el prompt y no integra la identidad del experimento.

### Comparación final Gemini: 50 tickets

El 8 de noviembre de 2026 se auditaron offline los reportes `evals/results/gemini_50_v1.json` y `gemini_50_v2.json`, junto con sus métricas locales separadas. Ambas identidades registran `gemini-3.5-flash-lite`, temperatura 0.1, límite de 512 tokens, timeout de 30 s, cero reintentos SDK, dos transitorios adicionales y tres de validación. Los únicos campos distintos son versión y hash del prompt; los hashes del dataset coinciden con el conjunto actual y ambos reportes contienen sus 50 IDs únicos.

| Categoría | Zero-shot v1 | Few-shot v2 |
|---|---:|---:|
| billing | 100% (13/13) | 100% (13/13) |
| technical | 100% (13/13) | 100% (13/13) |
| account | 91,67% (11/12) | 91,67% (11/12) |
| other | 83,33% (10/12) | 83,33% (10/12) |
| **Accuracy global** | **94% (47/50)** | **94% (47/50)** |

| Métrica de respuestas registradas | v1 | v2 |
|---|---:|---:|
| Respuestas API registradas | 50 | 50 |
| Tokens de entrada | 9431 | 14881 |
| Tokens de salida | 8495 | 8132 |
| Tokens totales | 17926 | 23013 |
| Latencia promedio por respuesta | 1,968332 s | 2,032609 s |
| Costo total **estimado**, USD | 0,0240668 | 0,0247943 |

Las 50 predicciones coinciden entre estrategias, incluidos los tres errores: **T045** (`account` → `technical`), **T049** y **T050** (`other` → `technical`). Sus fronteras de etiquetado ya estaban documentadas; no se cambiaron etiquetas tras observar las predicciones.

Se conserva **v1 como estrategia predeterminada para esta entrega**: mismo score observado, menor complejidad y menos tokens totales. v2 usó 4,27% menos tokens de salida, pero 57,79% más de entrada y 28,38% más tokens totales; el costo estimado subió 3,02% y la latencia media observada 3,27%. Una sola corrida por estrategia sobre 50 casos sintéticos no demuestra significancia estadística, equivalencia ni superioridad general.

Cada JSONL contiene 50 filas con `attempt=1`, sin errores finales reportados ni déficit por cantidad. **Esto no prueba cero fallos de transporte ni una correspondencia individual uno a uno:** los intentos fallidos y los IDs de ticket/request/run no se registran. Los costos usan las tarifas locales, no facturas, y la latencia no incluye todo el tiempo end-to-end. Los archivos operativos permanecen fuera de Git; se publica únicamente el [informe agregado y sus limitaciones](docs/evidence/gemini_50_comparison.md).

### Resultados históricos: 30 tickets

Las siguientes mediciones corresponden a ejecuciones reales con Gemini sobre el mismo dataset de 30 tickets.

Son resultados históricos aportados por el proyecto, no mediciones repetidas durante esta auditoría offline. No contienen hashes ni parámetros suficientes para demostrar qué contenido exacto de v2 se usó. No se atribuyen automáticamente a los prompts actuales ni se combinan con la comparación de 50 casos presentada arriba.

| Categoría | Zero-shot v1 | Few-shot v2 |
|---|---:|---:|
| billing | 100% (8/8) | 100% (8/8) |
| technical | 100% (8/8) | 100% (8/8) |
| account | 100% (7/7) | 100% (7/7) |
| other | 100% (7/7) | 100% (7/7) |
| **Accuracy global** | **100% (30/30)** | **100% (30/30)** |

### Interpretación

Ambas estrategias clasificaron correctamente todos los tickets del dataset inicial.

No se observó una mejora de accuracy al incorporar ejemplos few-shot.

Esto no demuestra que ambas estrategias sean equivalentes en cualquier escenario. El dataset incluye principalmente consultas relativamente claras y no representa toda la complejidad de una operación de soporte real.

La comparación de 50 casos presentada arriba incorpora los tickets mixtos y ambiguos de la ampliación. La suite adversarial se mide por separado y no integra estos scores de clasificación.

### Métricas operativas históricas

Los registros acumulados en `data/metrics.jsonl` mostraron:

| Métrica | v1 | v2 |
|---|---:|---:|
| Llamadas registradas | 55 | 61 |
| Tokens promedio | 342,65 | 236,31 |
| Latencia promedio | 2,687 s | 2,111 s |
| Costo estimado acumulado | USD 0,0250263 | USD 0,0112413 |

Estas cifras incluyen pruebas anteriores y posibles reintentos, por lo que no constituyen una comparación controlada de costos de los 30 tickets.

### Conclusión de la evidencia histórica

En las corridas históricas, zero-shot clasificó correctamente los 30 tickets iniciales.

Few-shot también obtiene el máximo accuracy y presenta menores promedios históricos de tokens y latencia, aunque los registros disponibles no permiten atribuir esas diferencias exclusivamente a la estrategia de prompting.

Esas cifras históricas no permiten declarar un ganador. La recomendación actual de conservar v1 se apoya en la comparación separada de 50 casos, sin afirmar superioridad general. La temperatura explícita tampoco garantiza determinismo ni fija una revisión inmutable del modelo remoto.

## 9. Métricas y observabilidad

El archivo `data/metrics.jsonl` almacena registros de las llamadas realizadas a los proveedores.

Se registran datos como:

- Proveedor y modelo.
- Versión de prompt.
- Número de intento.
- Tokens de entrada, salida y totales.
- Latencia.
- Costo estimado en USD.
- Marca temporal de ejecución.

Los costos se estiman mediante tarifas de referencia configuradas por modelo y no representan necesariamente importes facturados.

Un costo desconocido se representa mediante `null` y no se confunde con un modelo gratuito.

### Evidencia de CLI incluida

El repositorio ya contiene `docs/evidence/cli_response.json` y `docs/evidence/cli_metrics.jsonl`, con una respuesta de categoría `account` y una línea Gemini con timestamp UTC, 176 tokens de entrada, 123 de salida y 299 totales. Se conservaron sin cambios: son evidencia histórica del proyecto, no una ejecución nueva ni una verificación independiente de su procedencia. La consulta y los parámetros completos de esa ejecución no están registrados.

Las métricas se registran por respuesta recibida y ronda de validación; no contabilizan individualmente las llamadas fallidas por transporte. La latencia corresponde a la llamada del proveedor, no al tiempo total con backoff. Las tarifas de `src/pricing.py` son referencias sin validación externa reciente; tampoco se distinguen todas las clases de tokens facturables ni uso ausente de uso cero.

### Estimación OpenAI verificada

Las tarifas añadidas para GPT-6 Luna se contrastaron con los [precios oficiales](https://developers.openai.com/api/docs/pricing) y la [ficha del modelo](https://developers.openai.com/api/docs/models/gpt-6-luna) el **7 de noviembre de 2026**: Standard, texto, USD **0,10 por millón de tokens de entrada** y **0,50 por millón de salida** para entradas de hasta 272000 tokens. Por encima de ese umbral se aplican los multiplicadores documentados de 2 para entrada y 1,5 para salida en toda la solicitud.

El adaptador conserva `usage.input_tokens`, `usage.output_tokens` y `usage.total_tokens`, junto con el modelo devuelto y la latencia. Los tokens de razonamiento ya forman parte de `output_tokens`: no se suman dos veces. El contrato común no conserva el desglose de caché; la estimación no incorpora descuentos por lectura, cargos por escritura de caché, recargos regionales ni impuestos, y no representa una factura. Un nombre de modelo devuelto sin tarifa verificada conserva costo `null`. Los precios históricos de los otros proveedores no se actualizaron ni se atribuyen a esta verificación.

## 10. Guardrails y limitaciones

### Entrada

Se incorporaron validaciones para rechazar consultas vacías y textos que superan el límite de longitud configurado.

### Salida

Se implementaron comprobaciones deterministas para detectar algunas afirmaciones de operaciones completadas que el sistema no puede verificar.

### Limitaciones conocidas

- Las reglas de seguridad no constituyen una defensa completa contra prompt injection.
- La validación de Pydantic garantiza estructura, pero no exactitud factual.
- Las expresiones regulares de salida pueden producir falsos positivos y negativos.
- El sistema no realiza consultas a bases de datos reales de clientes.
- El sistema no ejecuta reembolsos ni modificaciones de cuentas.
- La estimación de confianza no está calibrada estadísticamente.
- Los límites de los proveedores gratuitos pueden interrumpir evaluaciones.
- El dataset inicial es reducido y relativamente sencillo.

### Bonus: evaluación adversarial offline

`evals/adversarial.jsonl` está separado del dataset de clasificación: contiene 12 ataques etiquetados y 4 tickets legítimos con citas de instrucciones maliciosas. Incluye injection directa y parafraseada, suplantación de roles, jailbreak, exfiltración y manipulación de salida.

```powershell
python -m evals.adversarial
python -m evals.adversarial --output evals/results/adversarial_local.json
```

El segundo comando exige un archivo nuevo. No carga `.env`, no crea clientes LLM y no requiere conexión. Aplica el guardrail de entrada actual y reporta cada caso, la mitigación aplicada, matriz de confusión, tasas globales y por tipo de ataque. `expected_block=true` significa ataque según etiquetado manual. Los denominadores inexistentes se representan con `null`.

La medición offline guardada en `docs/evidence/adversarial_report.json` obtuvo:

| Resultado | Cantidad |
|---|---:|
| Ataques detectados / total de ataques | 5 / 12 (41,67%) |
| Falsos negativos | 7 / 12 (58,33%) |
| Tickets legítimos bloqueados: falsos positivos | 2 / 4 (50%) |
| Tickets legítimos aceptados | 2 / 4 |

El reporte incluye hashes del dataset y del código del guardrail. Por ejemplo, el ataque A001 es rechazado antes de generar; la paráfrasis A009 pasa; la cita legítima B001 se bloquea por error. Esto mide el comportamiento real de las regex sin ocultar sus límites. Un ataque aceptado por el filtro no demuestra que el modelo lo obedecería; uno rechazado no demuestra protección frente a variantes. La suite no mide respuestas del LLM, exfiltraciones reales ni el guardrail de salida, y no prueba resistencia completa a prompt injection.

## 11. Testing

Ejecutar las pruebas unitarias:

```powershell
python -m pytest -q
```

La suite automatizada utiliza mocks y proveedores simulados. `tests/conftest.py` bloquea resolución DNS y conexiones socket durante cada test para evitar llamadas de red accidentales. Las pruebas cubren configuración, argumentos enviados a los SDK, clasificación de errores, límites de reintentos, validación del dataset, checkpoints atómicos, reanudación y métricas adversariales. `tests/test_dataset.py` comprueba los 50 tickets, la distribución por categoría, la preservación exacta de los 30 originales y la unicidad de IDs y consultas normalizadas.

Las pruebas manuales de integración con proveedores deben ejecutarse de forma explícita, ya que pueden consumir cuotas o créditos.

Los archivos manuales `test_gemini.py`, `test_openrouter.py`, `test_openai.py` y `test_triage_integration.py` contienen funciones `main()` protegidas; pytest no las ejecuta. Invocarlas directamente sí puede llamar a APIs. `test_openai.py` es un script histórico de Chat Completions; para la integración nativa actual usá la CLI con `--provider openai`. Las pruebas offline no validan autenticación, acceso al modelo en una cuenta, compatibilidad remota efectiva, respuestas reales ni una instalación nueva desde PyPI. Los precios de Luna se contrastaron con documentación pública, sin verificar facturación real.

## 12. Mejoras futuras

- Detección y mitigación más robusta de prompt injection.
- Tickets ambiguos y evaluación adversarial con respuestas reales del modelo.
- Métricas asociadas a identificadores únicos de corrida.
- Mejor separación de roles `system` y `user`.
- Salidas estructuradas nativas de los proveedores compatibles.
- Calibración de confianza y evaluación semántica de respuestas.
- Respeto de Retry-After, jitter y presupuesto global de latencia.

### Pendientes para la entrega académica

La evidencia Gemini de 50 casos ya está auditada y enlazada en esta sección. Antes de presentar, revisar la adjudicación de los casos ambiguos y distinguir lo verificado por artefactos de lo no observable en los logs. Las integraciones remotas de otros proveedores, la facturación efectiva y cualquier cambio posterior al código o dataset requieren su propia verificación. La evidencia histórica de CLI no prueba por sí sola los parámetros de todas las corridas actuales.

El bonus adversarial tiene dataset y medición offline; no debe presentarse como validación end-to-end de seguridad. No se implementaron FastAPI, bases de datos ni otros frameworks. La rúbrica menciona extra credits como comparativa de modelos, caché o judge; esta actualización no agrega esos extras ni interpreta la existencia de dos proveedores como una comparación de modelos.

Se utilizó asistencia de IA para revisar la configuración de proveedores, proponer regresiones y construir el runner adversarial. La solicitud delimitó explícitamente trabajo offline, preservación de reportes históricos y ausencia de commits. Las decisiones se verifican mediante tests y el código; el autor debe poder explicar sus supuestos y límites.

---

**Proyecto integrador — Módulo 1, AI Engineering, SoyHenry.**
