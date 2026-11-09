# Evidencia pública sanitizada de clasificación — versión 1

## Alcance y procedencia

El archivo [classification_runs_v1.json](classification_runs_v1.json) es una proyección de los resultados locales existentes, no una nueva ejecución. Publica exclusivamente IDs sintéticos, categorías esperadas y predichas, configuración del experimento, conteos y SHA-256 de los archivos fuente. No contiene consultas, respuestas completas, credenciales, logs operativos ni datos de clientes. Los tests verifican sus etiquetas contra el dataset público y recalculan los conteos sin requerir archivos privados.

Los hashes de fuente corresponden a los bytes originales y sirven para cotejar una copia local exacta; no son hashes canónicos ni prueban por sí solos la autenticidad de una llamada remota. La evidencia sanitizada permite auditar las clasificaciones publicadas, pero no reproduce la ejecución del proveedor ni su facturación.

| Corrida | Estado | Tickets registrados / previstos | Correctos registrados | Accuracy final |
|---|---|---:|---:|---:|
| Gemini v1 | Completa | 50/50 | 47 | 94% |
| Gemini v2 | Completa | 50/50 | 47 | 94% |
| OpenRouter v1 | Completa | 50/50 | 47 | 94% |
| OpenRouter v2 | **Incompleta** | **29/50** | **29** | **No disponible** |

Los tres resultados completos fallan en T045, T049 y T050. Gemini predice `technical` en los tres; OpenRouter v1 predice `technical`, `billing` y `technical`, respectivamente. Las fronteras de las etiquetas se mantienen según [la política existente](../dataset_labeling.md). No se cambiaron etiquetas después de observar las predicciones.

## OpenRouter v2: interrupción y reanudación

El checkpoint contiene T001–T029, sin errores finales guardados en esas filas. Faltan 21 tickets; el siguiente ID pendiente es T030. Los 29 aciertos corresponden únicamente a ese prefijo y **no constituyen un score final ni una estimación del rendimiento en 50 tickets**.

El HTTP 429 en T030 es un **incidente reportado por el usuario**, no un error registrado en el checkpoint. Los archivos disponibles no permiten confirmar su detalle HTTP ni contar todos los intentos fallidos de transporte. No se completó ni se reanudó esa corrida durante esta remediación.

La identidad guardada de OpenRouter usa `nvidia/nemotron-3.5-lightning:free`, temperatura 0.1, 4096 tokens máximos, timeout 30 s, cero retries del SDK, dos retries transitorios adicionales y dos de validación. `--resume` exige mantener esos valores, el dataset y el prompt v2 originales, además de seleccionar el archivo de métricas correspondiente. Copiar los valores actuales de otra corrida puede producir un rechazo de identidad. No se publica un comando que cargue credenciales ni se afirma compatibilidad remota actual.

## Comparabilidad y límites

Gemini v1/v2 comparten configuración y dataset; solo difieren versión y hash del prompt. El [informe Gemini existente](gemini_50_comparison.md) conserva tokens, latencia y costos **estimados**, que no representan cargos facturados. OpenRouter v1/v2 también comparten su configuración guardada, pero v2 está incompleta. Gemini y OpenRouter usan límites de salida y de reintentos de validación diferentes: no es una comparación controlada de modelos ni demuestra superioridad de ninguno. Una corrida pequeña por estrategia no establece significancia estadística.

Las clasificaciones son evidencia histórica anterior a la corrección del guardrail de salida. No se conservan las respuestas completas en estos reportes; por eso **se detuvo su validación retrospectiva contra el nuevo guardrail**. Los scores originales no cambian, pero no se certifica que una nueva corrida produzca los mismos resultados o acepte las mismas respuestas. No hubo llamadas API nuevas.

## Remediación del hash y seguridad

El reporte [adversarial_report.json](adversarial_report.json) mide exclusivamente el filtro de **entrada**. Su hash del archivo completo `src/guardrails.py` ahora usa `sha256-utf8-lf-v1`: decodificación UTF-8, normalización CRLF y CR a LF, y SHA-256. El hash cambia por esa normalización y por la corrección de la salida en el mismo archivo. El hash previo de bytes Windows era `7183631cdd0e016b8492841573717d7ac225e83b5060af86dfe27ce4af3f4229`.

Se volvió a medir offline la entrada sin modificar sus reglas ni casos. Solo se actualizaron el hash y su esquema: resultados individuales, etiquetas, métricas y mitigaciones permanecen iguales (5 TP, 7 FN, 2 FP, 2 TN). Las regresiones verifican tanto la equivalencia LF/CRLF del reporte completo como la detección de cambios reales de contenido.

La salida ahora revisa `answer` y cada elemento de `actions`, contempla negación directa y sigue rechazando una afirmación positiva posterior. También contempla el caso observado «El reembolso ya fue emitido». Son reglas acotadas: no interpretan todas las negaciones indirectas, citas, condicionales, dobles negaciones o paráfrasis. No verifican operaciones reales y pueden producir falsos positivos y negativos. Esta corrección no aumenta el score adversarial de entrada ni demuestra resistencia completa a prompt injection.

## Hallazgos de auditoría y estado

| Hallazgo recuperado de la auditoría | Estado tras esta remediación | Evidencia |
|---|---|---|
| Hash dependiente de CRLF/LF | Resuelto | Hash canónico, metadatos y regresión de reporte completo |
| Predicciones de 50 casos ausentes de la evidencia pública | Resuelto localmente; pendiente publicar por el autor | JSON versionado, tres corridas completas y checkpoint parcial sanitizados |
| Negación rechazada y `actions` sin inspeccionar | Resueltos los casos reproducidos | Tests de ambas propiedades, negación directa, afirmación posterior y sugerencias legítimas |
| Filtro de entrada débil y citas benignas bloqueadas | Pendiente | 7 FN y 2 FP conservados y visibles |
| Reanudación OpenRouter requiere configuración original | Documentado; corrida sin completar | Identidad preservada, 29/50 y próximo ID T030 |
| Métricas sin IDs de request/ticket/run ni fallos de transporte | Pendiente | No se modificó el logger; el HTTP 429 sigue siendo reportado, no registrado |
| Pydantic permite contenido vacío; acceso/facturación OpenAI no verificados | Pendiente, fuera del alcance | No se modificaron contrato ni proveedores |
| Una sola corrida y etiquetas ambiguas | Limitación conservada | Sin inferencia estadística ni modificación del dataset |

La consigna P1 completa se leyó localmente. El adjunto disponible contiene la solicitud de auditoría, no su informe final completo; esta trazabilidad usa los hallazgos recuperados del contexto anterior y los defectos reproducidos, sin recalcular la nota no oficial de 78/100.

## Verificación offline

La suite completa pasó con Python 3.12.1: **270 tests**, tanto con el archivo del guardrail en LF como en CRLF. La suite bloquea DNS y conexiones socket mediante `tests/conftest.py`; no se ejecutaron llamadas reales a proveedores ni se instalaron paquetes. Se cotejaron las 179 predicciones publicadas con los originales locales antes de preparar esta evidencia. El reporte adversarial anterior se comparó campo por campo: solo cambiaron el hash y su esquema.

Desde la raíz del repositorio, con el entorno del proyecto activado:

```powershell
python -m pytest -q
python -m evals.adversarial
```

El segundo comando imprime una medición offline; no escribe archivos ni completa evaluaciones de proveedores. La validación con respuestas reales del nuevo guardrail y la finalización de OpenRouter v2 permanecen pendientes.
