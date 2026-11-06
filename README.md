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
python -m evals.runner --provider gemini --prompt-version triage_v1 --output evals/results_v1.json
```

Few-shot:

```powershell
python -m evals.runner --provider gemini --prompt-version triage_v2 --output evals/results_v2.json
```

El runner permite guardar checkpoints y reanudar los tickets pendientes. También utiliza una pausa configurable entre tickets para reducir el riesgo de exceder límites de solicitudes.

### Resultados obtenidos

Las siguientes mediciones corresponden a ejecuciones reales con Gemini sobre el mismo dataset de 30 tickets.

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

No se declara un ganador definitivo.

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

## 11. Testing

Ejecutar las pruebas unitarias:

```powershell
python -m pytest tests/test_models.py tests/test_pricing.py tests/test_metrics.py tests/test_guardrails.py tests/test_triage_service.py tests/test_scoring.py tests/test_runner.py -v
```

Las pruebas unitarias utilizan proveedores simulados cuando corresponde, evitando llamadas reales a servicios externos.

Las pruebas manuales de integración con proveedores deben ejecutarse de forma explícita, ya que pueden consumir cuotas o créditos.

## 12. Mejoras futuras

- Detección y mitigación más robusta de prompt injection.
- Evaluaciones adversariales y tickets ambiguos.
- Métricas asociadas a identificadores únicos de corrida.
- Mejor separación de roles `system` y `user`.
- Salidas estructuradas nativas de los proveedores compatibles.
- Calibración de confianza y evaluación semántica de respuestas.
- Manejo avanzado de rate limits y errores transitorios.

---

**Proyecto integrador — Módulo 1, AI Engineering, SoyHenry.**
