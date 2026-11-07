# Nubbix AI Triage

Sistema de clasificación inteligente de tickets de soporte para Nubbix, desarrollado como proyecto integrador del Módulo 1 de AI Engineering de SoyHenry.

## 1. Descripción

Nubbix AI Triage es una aplicación CLI desarrollada en Python que utiliza modelos de lenguaje (LLMs) para analizar consultas de soporte, clasificarlas y generar sugerencias útiles para el equipo de atención al cliente.

El sistema permite intercambiar proveedores LLM sin modificar la lógica de negocio y utiliza Pydantic para validar la estructura de las respuestas generadas.

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
- OpenAI Python SDK (utilizado para la integración compatible de OpenRouter)
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
- API key de Gemini y/o OpenRouter

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
| `LLM_TEMPERATURE` | `0.1` | Número finito entre 0 y 2, enviado a ambos proveedores |
| `LLM_MAX_OUTPUT_TOKENS` | `512` | Entero positivo; Gemini recibe `max_output_tokens`, OpenRouter `max_tokens` |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout por solicitud en segundos, finito y al menos `0.001`; Gemini lo recibe en milisegundos |
| `LLM_MAX_RETRIES` | `2` | Reintentos adicionales por JSON inválido; el ejemplo configura `3` |
| `PROMPT_VERSION` | `triage_v1` | Versión predeterminada de la CLI; el runner exige `--prompt-version` |
| `METRICS_PATH` | `data/metrics.jsonl` | Archivo append-only de métricas |

Completá los modelos con identificadores habilitados para tu cuenta. El identificador Gemini de la evidencia histórica es `gemini-3.5-flash-lite`; no se confirmó su disponibilidad actual. Las variables `OPENAI_API_KEY` y `OPENAI_MODEL` solo corresponden al script manual opcional `tests/test_openai.py`, no a un tercer proveedor de la aplicación.

Los límites admitidos por cada modelo pueden ser más estrechos que la validación compartida. El timeout es por solicitud y no una fecha límite global: los reintentos y esperas pueden extender el tiempo total.

Los SDK no agregan reintentos: OpenRouter usa `max_retries=0` y Gemini `HttpRetryOptions(attempts=1)`. `retry_transient()` permite dos reintentos adicionales con esperas de 1 y 2 segundos. HTTP 408, 429, 5xx y fallos temporales de transporte se consideran transitorios; otros errores HTTP son permanentes. Un error permanente no se reintenta. Con `LLM_MAX_RETRIES=3`, el límite combinado es de hasta 12 solicitudes de generación si se combinan fallos transitorios y respuestas JSON inválidas.

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

## 7. Estrategias de Prompt Engineering

Se implementaron dos versiones de prompts en español.

### Versión 1 — Zero-shot

Utiliza instrucciones de clasificación, definición de categorías y especificación del contrato JSON, sin ejemplos de tickets.

### Versión 2 — Few-shot

Mantiene las instrucciones principales e incorpora ejemplos etiquetados para las cuatro categorías.

El objetivo experimental es comprobar si los ejemplos mejoran el comportamiento del clasificador frente a la versión base.

## 8. Evaluaciones

Se construyó un dataset de 30 consultas etiquetadas en `evals/dataset.jsonl`.

Distribución:

| Categoría | Tickets |
|---|---:|
| billing | 8 |
| technical | 8 |
| account | 7 |
| other | 7 |
| **Total** | **30** |

### Ejecutar evaluaciones

Zero-shot:

```powershell
python -m evals.runner --provider gemini --prompt-version triage_v1 --fresh --output evals/results/submission_v1.json
```

Few-shot:

```powershell
python -m evals.runner --provider gemini --prompt-version triage_v2 --fresh --output evals/results/submission_v2.json
```

Estos comandos hacen llamadas reales y consumen cuota. No se ejecutan como parte de las pruebas offline. Los reportes históricos `evals/results_v1.json` y `evals/results_v2.json` se conservan sin cambios.

`--fresh` exige que la salida no exista: nunca sobrescribe una evaluación. `--resume` exige un checkpoint existente y compatible, y omite los tickets ya registrados:

```powershell
python -m evals.runner --provider gemini --prompt-version triage_v1 --resume --output evals/results/submission_v1.json
```

El runner valida campos, categorías e IDs únicos antes de generar. La identidad incluye hashes del dataset y del archivo de prompt, proveedor, modelo, temperatura, límite de tokens, timeout y políticas de reintento. No incluye credenciales. Un cambio de identidad rechaza la reanudación. Los reportes antiguos sin esta identidad no pueden reanudarse: elegí una salida nueva.

Cada checkpoint y el reporte final se escriben mediante un archivo temporal en el mismo directorio y reemplazo atómico. Un fallo del proveedor interrumpe la ejecución conservando los tickets anteriores; el ticket pendiente se intenta al reanudar. Los errores de validación y guardrail guardados cuentan como completados: para repetirlos, iniciá una corrida nueva. No ejecutes dos procesos sobre la misma salida.

La pausa predeterminada es de seis segundos; `--delay-seconds 0` la desactiva. El tiempo de espera entre tickets no modifica el prompt y no integra la identidad del experimento.

### Resultados obtenidos

Las siguientes mediciones corresponden a ejecuciones reales con Gemini sobre el mismo dataset de 30 tickets.

Son resultados históricos aportados por el proyecto, no mediciones repetidas durante esta actualización offline. No contienen hashes ni parámetros suficientes para demostrar qué contenido exacto de v2 se usó. La versión actual de v2 conserva las instrucciones de v1; los reportes anteriores no permiten atribuirle automáticamente el mismo resultado. Se requiere una comparación nueva, con el mismo modelo y configuración, para evaluar los prompts actuales.

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

Para una evaluación más rigurosa será necesario incorporar casos ambiguos y adversariales.

### Métricas operativas históricas

Los registros acumulados en `data/metrics.jsonl` mostraron:

| Métrica | v1 | v2 |
|---|---:|---:|
| Llamadas registradas | 55 | 61 |
| Tokens promedio | 342,65 | 236,31 |
| Latencia promedio | 2,687 s | 2,111 s |
| Costo estimado acumulado | USD 0,0250263 | USD 0,0112413 |

Estas cifras incluyen pruebas anteriores y posibles reintentos, por lo que no constituyen una comparación controlada de costos de los 30 tickets.

### Conclusión provisional

Zero-shot constituye una alternativa simple y suficiente para clasificar correctamente el dataset actual.

Few-shot también obtiene el máximo accuracy y presenta menores promedios históricos de tokens y latencia, aunque los registros disponibles no permiten atribuir esas diferencias exclusivamente a la estrategia de prompting.

No se declara un ganador definitivo. Se mantiene v1 como opción predeterminada por simplicidad, con respaldo limitado al dataset histórico; no se afirma que sea superior a v2. La temperatura explícita tampoco garantiza determinismo ni fija una revisión inmutable del modelo remoto.

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

La suite automatizada utiliza mocks y proveedores simulados. `tests/conftest.py` bloquea resolución DNS y conexiones socket durante cada test para evitar llamadas de red accidentales. Las pruebas cubren configuración, argumentos enviados a los SDK, clasificación de errores, límites de reintentos, validación del dataset, checkpoints atómicos, reanudación y métricas adversariales.

Las pruebas manuales de integración con proveedores deben ejecutarse de forma explícita, ya que pueden consumir cuotas o créditos.

Los archivos manuales `test_gemini.py`, `test_openrouter.py`, `test_openai.py` y `test_triage_integration.py` contienen funciones `main()` protegidas; pytest no las ejecuta. Invocarlas directamente sí puede llamar a APIs. Las pruebas offline no validan autenticación, existencia de modelos, compatibilidad remota de parámetros, precios vigentes, respuestas reales ni una instalación nueva desde PyPI.

## 12. Mejoras futuras

- Detección y mitigación más robusta de prompt injection.
- Tickets ambiguos y evaluación adversarial con respuestas reales del modelo.
- Métricas asociadas a identificadores únicos de corrida.
- Mejor separación de roles `system` y `user`.
- Salidas estructuradas nativas de los proveedores compatibles.
- Calibración de confianza y evaluación semántica de respuestas.
- Respeto de Retry-After, jitter y presupuesto global de latencia.

### Pendientes para la entrega académica

Antes de presentar, ejecutar voluntariamente una CLI real con cada proveedor y nuevas evals v1/v2 en archivos nuevos, revisar sus métricas y actualizar la interpretación con esos resultados. Confirmar instalación limpia en Python 3.12 y acceso público al repositorio. La evidencia de CLI existente no prueba los parámetros incorporados en esta actualización.

El bonus adversarial tiene dataset y medición offline; no debe presentarse como validación end-to-end de seguridad. No se implementaron FastAPI, bases de datos ni otros frameworks. La rúbrica menciona extra credits como comparativa de modelos, caché o judge; esta actualización no agrega esos extras ni interpreta la existencia de dos proveedores como una comparación de modelos.

Se utilizó asistencia de IA para revisar la configuración de proveedores, proponer regresiones y construir el runner adversarial. La solicitud delimitó explícitamente trabajo offline, preservación de reportes históricos y ausencia de commits. Las decisiones se verifican mediante tests y el código; el autor debe poder explicar sus supuestos y límites.

---

**Proyecto integrador — Módulo 1, AI Engineering, SoyHenry.**
