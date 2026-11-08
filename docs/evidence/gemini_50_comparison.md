# Gemini: comparación final de zero-shot y few-shot sobre 50 tickets

Evidencia preparada el **8 de noviembre de 2026** mediante lectura y recálculo offline. No se realizaron llamadas a APIs ni se modificaron las fuentes. Los resultados proceden de las ejecuciones locales aportadas por el autor; esta auditoría comprueba su consistencia interna, no certifica de forma independiente la procedencia de las respuestas remotas.

## Metodología y fuentes

Se usaron los dos reportes y sus JSONL operativos separados, el dataset actual de 50 casos, los archivos YAML actuales y `docs/dataset_labeling.md`. Se recalcularon las métricas a partir de las predicciones y los registros, sin confiar únicamente en los scores guardados. `evals/comparison.py` contiene los cálculos y `tests/test_comparison.py` los verifica con datos sintéticos; los tests no necesitan las fuentes operativas locales.

Los JSON/JSONL operativos permanecen en sus ubicaciones locales ignoradas por Git. Este documento contiene únicamente agregados, IDs/etiquetas de errores, configuración no secreta y hashes; no incorpora claves, respuestas completas ni registros operativos individuales.

| Fuente local | SHA-256 del archivo original |
|---|---|
| `evals/results/gemini_50_v1.json` | `ffe1deec247ce326103f61c8d7346c5b2914bb539d1c17738990e0dee1500a72` |
| `evals/results/gemini_50_v2.json` | `69e44670e5262f3b78ec23790984a45044703896ce82fc624e8a96277975e949` |
| `data/metrics_gemini_50_v1.jsonl` | `2fc3e75d36a7d726b0bd5ea446ab5304c971909402dbc77611521d97962353c8` |
| `data/metrics_gemini_50_v2.jsonl` | `f9b50bea980c0de453fe446e63209b1216efa061d2f8526f4f1db10e29efa73b` |

Estos hashes corresponden a los bytes originales inspeccionados; una conversión de saltos de línea cambia el hash de archivo. El hash canónico de dataset registrado en la identidad se calcula sobre los registros JSON y es independiente de sus saltos de línea.

## Integridad y control del experimento

Ambas evaluaciones contienen exactamente **50 IDs únicos**, `T001`–`T050`, y coinciden con los IDs y etiquetas de `evals/dataset.jsonl`. Su hash de dataset es idéntico y coincide con el recálculo actual: `20f2cc2f1097e9843155e92551d87a399766236033265daae38f209c90a57c44`.

Distribución congelada: billing 13, technical 13, account 12 y other 12. No se cambiaron etiquetas después de ver las predicciones.

| Configuración registrada | v1 zero-shot | v2 few-shot |
|---|---|---|
| Proveedor | `gemini` | `gemini` |
| Modelo | `gemini-3.5-flash-lite` | `gemini-3.5-flash-lite` |
| Temperatura | 0.1 | 0.1 |
| Límite de salida | 512 tokens | 512 tokens |
| Timeout por solicitud | 30 s | 30 s |
| Reintentos SDK | 0 | 0 |
| Reintentos transitorios adicionales | 2 | 2 |
| Reintentos de validación adicionales | 3 | 3 |
| Versión de identidad | 2 | 2 |
| Versión de prompt | `triage_v1` | `triage_v2` |
| Hash del prompt | `bdc7df1462bc7b70bac749facf61f49a2dd934e481909b835ad49138f33c393c` | `061fb777a8740d7b87f58450eac465a38cf9e7cac4fb69c8a1160ee7403e01f1` |

Al comparar todos los campos de `experiment`, las únicas diferencias son **`prompt_version` y `prompt_hash`**. Ambos hashes de prompt coinciden con los YAML actuales. La configuración es la registrada por el runner; los JSONL no repiten temperatura, límites o política de reintentos y no permiten verificar independientemente el cuerpo de cada solicitud enviada.

## Respuestas registradas, reintentos y errores

| Comprobación | v1 | v2 |
|---|---:|---:|
| Resultados de tickets | 50 | 50 |
| Respuestas API registradas en JSONL | 50 | 50 |
| Respuestas con `attempt=1` | 50 | 50 |
| Respuestas con `attempt>1` | 0 | 0 |
| Errores finales de evaluación | 0 | 0 |
| Predicciones ausentes | 0 | 0 |
| Costos ausentes | 0 | 0 |
| Diferencia entre cantidad de métricas y tickets | 0 | 0 |
| Inconsistencias input + output frente a total | 0 | 0 |
| Inconsistencias de costo con la tabla local | 0 | 0 |

Todos los registros corresponden al proveedor, modelo y versión esperados. Las marcas temporales tienen zona horaria, están ordenadas y no se repiten dentro de cada JSONL. No se observan respuestas adicionales por reintentos de validación ni un déficit de métricas por cantidad.

**Límite de observabilidad:** `attempt` identifica la ronda de validación, no los intentos de transporte de `retry_transient()`. Las llamadas que fallan antes de devolver una respuesta no se registran. Por tanto, no se puede afirmar que hubo exactamente 50 intentos HTTP, cero fallos de transporte o cero reintentos transitorios. Tampoco hay IDs de ticket/request/run en las métricas: 50 filas compatibles apoyan la correspondencia esperada, pero no prueban una relación individual uno a uno ni excluyen omisiones compensadas por duplicados. No se emparejaron filas por posición como si esa asociación estuviera demostrada.

| Ventana de registros UTC | Primera respuesta | Última respuesta |
|---|---|---|
| v1 | `2026-11-08T02:51:09.887663+00:00` | `2026-11-08T02:57:40.797875+00:00` |
| v2 | `2026-11-08T02:58:17.209615+00:00` | `2026-11-08T03:04:50.701961+00:00` |

En Argentina (UTC−03), la primera corrida fue el 7 de noviembre cerca de las 23:51–23:57; la segunda comenzó el 7 de noviembre a las 23:58 y terminó el 8 de noviembre cerca de las 00:04. Son corridas secuenciales, no intercaladas. La ventana entre registros no equivale a la duración completa de ejecución.

## Exactitud global y por categoría

| Categoría | v1: correctos / total | v1 accuracy | v2: correctos / total | v2 accuracy |
|---|---:|---:|---:|---:|
| billing | 13/13 | 100.00% | 13/13 | 100.00% |
| technical | 13/13 | 100.00% | 13/13 | 100.00% |
| account | 11/12 | 91.67% | 11/12 | 91.67% |
| other | 10/12 | 83.33% | 10/12 | 83.33% |
| **Global** | **47/50** | **94.00%** | **47/50** | **94.00%** |

Accuracy global = aciertos / 50. La métrica por categoría usa como denominador los tickets cuya etiqueta esperada pertenece a esa categoría; corresponde al recall de esa clase, no a precision ni a una evaluación de la calidad de respuesta. Todos los scores guardados coinciden con el recálculo.

## Errores y coincidencia entre estrategias

| Ticket | Etiqueta esperada | v1 | v2 | Criterio documentado para revisar el error |
|---|---|---|---|---|
| T045 | account | technical | technical | Rotar/revocar credenciales cuando la integración funciona es gestión de acceso: account, no una avería de integración. |
| T049 | other | technical | technical | Consultar capacidades multimoneda antes de contratar es información de producto: other, no incidente técnico ni cobro del abono. |
| T050 | other | technical | technical | La consulta no identifica operación ni síntoma suficiente: other para pedir aclaración, sin inferir una causa técnica. |

Errores compartidos: **T045, T049, T050**. Errores exclusivos de v1: **0**; exclusivos de v2: **0**. Diferencias de predicción entre estrategias: **0/50**. No solo empatan en accuracy: las 50 categorías predichas coinciden.

Los tres casos ya estaban marcados como fronteras de revisión humana en [la política de etiquetado](../dataset_labeling.md). Se mantienen sus etiquetas actuales. Las razones anteriores interpretan la política, no revelan el razonamiento interno del modelo. Si una adjudicación humana cambia etiquetas, debe documentarse como una nueva versión, no mezclarse silenciosamente con este resultado.

## Consumo, latencia y costo estimado

Los tokens y costos se suman sobre las 50 respuestas registradas. Cada promedio usa 50 como denominador. Las variaciones se calculan como `(v2 / v1 − 1) × 100`, antes de redondear.

| Métrica | v1 zero-shot | v2 few-shot | Variación v2 vs v1 |
|---|---:|---:|---:|
| Tokens de entrada totales | 9431 | 14881 | +57.79% |
| Tokens de salida totales | 8495 | 8132 | -4.27% |
| Tokens totales | 17926 | 23013 | +28.38% |
| Tokens de entrada promedio | 188.62 | 297.62 | +57.79% |
| Tokens de salida promedio | 169.90 | 162.64 | -4.27% |
| Tokens totales promedio | 358.52 | 460.26 | +28.38% |
| Latencia promedio por respuesta (s) | 1.968332 | 2.032609 | +3.27% |
| Suma de latencias de proveedor (s) | 98.416589 | 101.630463 | +3.27% |
| Costo total estimado (USD) | 0.0240668 | 0.0247943 | +3.02% |
| Costo promedio estimado (USD) | 0.000481336 | 0.000495886 | +3.02% |

v2 agregó **5450 tokens de entrada**, redujo **363 tokens de salida**, sumó **5087 tokens totales** y aumentó el costo estimado en **USD 0.0007275** en estos 50 casos. La diferencia de latencia media fue de unos **64 ms** por respuesta; no se atribuye causalidad al prompt a partir de dos corridas secuenciales.

Los costos se recalcularon con las tarifas locales registradas para `gemini-3.5-flash-lite`: USD 0,30/M tokens de entrada y USD 2,50/M de salida. La fórmula es `(input × 0,30 + output × 2,50) / 1 000 000`. Los 100 valores guardados coinciden con esta fórmula dentro de tolerancia numérica. No se consultaron precios externos, facturas, nivel gratuito, caché, impuestos ni facturación de la cuenta. **Son estimaciones de generación, no importes efectivamente cobrados.**

La latencia mide cada generación del proveedor e incluye la gestión del cliente según el adaptador actual. No incluye la pausa entre tickets, ni el backoff de llamadas fallidas previas, ni todo el tiempo de validación/orquestación. No equivale a latencia end-to-end ni a throughput del servicio.

## Conclusión y decisión de prompting

En este dataset congelado, zero-shot y few-shot obtuvieron la misma clasificación en todos los casos: 94% de accuracy y tres errores compartidos. Los ejemplos de v2 no resolvieron las fronteras de gestión de credenciales, información comercial y falta de contexto que explican esas discrepancias frente a las etiquetas.

Se recomienda **conservar v1 como estrategia predeterminada para esta entrega**: mantiene el score observado, usa un prompt más simple y consumió menos tokens totales y menor costo estimado. v2 produjo menos tokens de salida, pero ese ahorro no compensó el incremento de entrada. No se declara superioridad general, equivalencia estadística ni significancia en calidad, latencia o costos; el criterio es descriptivo y limitado a estas corridas.

Antes de modificar prompts, conviene adjudicar las etiquetas ambiguas con una segunda persona. Para investigar mejoras futuras, usar más casos independientes y corridas repetidas/intercaladas con la misma configuración. No optimizar y volver a medir indefinidamente sobre estos 50 como si siguieran siendo un conjunto de prueba no visto.

## Limitaciones y relación con la evidencia histórica

- Una sola ejecución por estrategia, 50 ejemplos sintéticos de autoría única y sin replicaciones. No se infiere representatividad del tráfico real.
- Mismo nombre de modelo no garantiza idéntica revisión interna del servicio; no se registraron snapshot, seed, run ID ni cuerpos de solicitud. Temperatura baja no garantiza determinismo.
- La igualdad de parámetros proviene de las identidades guardadas, no de una inspección de tráfico remoto. No se verificaron el orden exacto de ejecución, pausas o ausencia de reanudaciones por una bitácora externa.
- No hay prueba independiente de cero llamadas HTTP fallidas, métricas exhaustivas ni asociación individual de cada respuesta con un ticket.
- Los reportes retienen categorías, no respuestas, acciones ni confianza. No evalúan utilidad, factualidad, calibración ni seguridad del contenido generado.
- El bonus adversarial tiene su dataset y evidencia propios; no se mezcla con esta accuracy ni se actualiza aquí.
- Los reportes históricos `evals/results_v1.json` y `evals/results_v2.json` corresponden a 30 casos con 100% reportado y permanecen intactos. No se combinan sus llamadas, promedios o costos con los de esta comparación.

**Inconsistencias encontradas:** ninguna en los IDs, etiquetas, hashes actuales, configuración registrada, scores recalculados, sumas de tokens o fórmula de costo de estas fuentes. Sí existe una limitación de trazabilidad para fallos y correspondencia individual; el README anterior todavía decía que no había resultados sobre 50 tickets y se actualiza con esta evidencia.

## Validación offline de los cálculos

```powershell
python -m pytest tests/test_comparison.py -q
python -m pytest -q
```

Las pruebas usan fixtures sintéticos y verifican scores, distribución, errores compartidos/exclusivos, cambios de configuración, IDs repetidos, diferencias de hashes, totales/promedios, costos desconocidos, reintentos observados y conteos de métricas incompletos. No ejecutan modelos ni necesitan copiar los JSONL operativos al repositorio público.
