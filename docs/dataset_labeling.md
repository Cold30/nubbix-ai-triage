# Política de etiquetado del dataset de clasificación

## Alcance y procedencia

`evals/dataset.jsonl` tiene 50 ejemplos sintéticos: 30 originales preservados y 20 nuevos, redactados para ampliar situaciones y estilos. No son registros de clientes reales. No se usaron llamadas a proveedores ni resultados de predicción para decidir las etiquetas. El esquema del runner permanece igual: `id`, `query`, `expected_category`.

La distribución es billing 13, technical 13, account 12 y other 12. Se agregaron cinco casos por categoría. La consigna P1 exige al menos 30 casos; la ampliación agrega diversidad, sin convertir el conjunto en una muestra representativa de toda la operación.

## Criterios comunes

1. Etiquetar por la **resolución principal solicitada**, considerando el ticket completo. No decidir por la presencia aislada de palabras como factura, error, integración o acceso.
2. `billing`: importes, cobros, comprobantes del servicio Nubbix, condiciones de pago y ajustes comerciales del abono.
3. `technical`: funcionamiento observado distinto del esperado, pérdida o inconsistencia de datos, errores reproducibles y fallos de flujos o integraciones. La mera mención de una integración no basta.
4. `account`: identidad, titularidad, sesiones, habilitación de usuarios, alcance de permisos y ciclo de vida de credenciales. No implica que el asistente pueda ejecutar cambios por sí mismo.
5. `other`: consultas comerciales generales, capacidades del producto, políticas, accesibilidad, horarios de atención y pedidos sin información suficiente para decidir entre las otras tres categorías.
6. En casos mixtos, priorizar lo que el usuario pide resolver ahora y la evidencia explícita. No suponer una causa técnica cuando pide administrar permisos ni una causa de pagos cuando el servicio ya fue restablecido.
7. Si dos necesidades tienen la misma prioridad y no se puede elegir responsablemente, usar `other` como solicitud de aclaración y registrar la ambigüedad. No inventar una quinta categoría ni etiquetas múltiples.
8. Conservar los 30 originales aunque sean más sencillos. Congelar las etiquetas antes de comparar prompts; no corregirlas para favorecer resultados. Si la revisión humana cambia etiquetas nuevas, generar una evaluación nueva con la identidad correspondiente.

La consigna ofrece ejemplos orientativos, no una taxonomía exhaustiva. En particular, una consulta sobre funciones de facturación que ofrece el producto es `other`; un comprobante por el abono pagado a Nubbix es `billing`. Esta distinción debe revisarse antes de la comparación final.

## Justificación de los 20 casos nuevos

“Revisar” identifica fronteras que conviene adjudicar humanamente antes de ejecutar las evals, no resultados de un modelo.

| ID | Etiqueta | Motivo | Revisión humana |
|---|---|---|---|
| T031 | billing | Las bajas de usuarios ya se hicieron; se solicita calcular el importe proporcional, no cambiar accesos. | Revisar frontera account/billing. |
| T032 | billing | Pide explicar conversión e impuestos del cargo, sin afirmar un error del software. | No prioritaria. |
| T033 | billing | Solicita acordar vencimientos de pago para una contratación anual. No pregunta por capacidades del producto. | No prioritaria. |
| T034 | billing | El perfil ya se actualizó; queda corregir el comprobante del abono emitido con datos fiscales anteriores. | Revisar frontera account/billing. |
| T035 | billing | Pide dividir la documentación del abono entre administraciones sin modificar el espacio de trabajo. | No prioritaria. |
| T036 | technical | Describe una transformación reproducible del separador decimal que altera el valor guardado; el contenido monetario no lo convierte en un cobro de Nubbix. | No prioritaria. |
| T037 | technical | Hay discrepancia entre registros y cierre según ubicación; se pide revisar el tratamiento de horarios, no cambiar perfil o pagar. | No prioritaria. |
| T038 | technical | El adjunto falla solo en la ruta de captura de cámara; aporta contraste con galería y varios usuarios. | No prioritaria. |
| T039 | technical | La invitación pertenece a un flujo de acceso, pero el problema reportado es una página en blanco reproducible también por una administradora autenticada. | Revisar frontera account/technical. |
| T040 | technical | Se describen versiones inconsistentes de una cotización tras guardar desde dos pestañas; no se solicita emitir ni cobrar un comprobante del servicio. | No prioritaria. |
| T041 | account | Solicita transferir titularidad y autoridad administrativa con el equipo aún operativo. | No prioritaria. |
| T042 | account | La petición principal es restringir el alcance de un usuario a su sucursal; no se afirma un fallo de persistencia de permisos ya configurados. | Revisar frontera account/technical. |
| T043 | account | Solicita revocar una sesión propia en un equipo compartido, sin afirmar que el inicio de sesión esté averiado. | No prioritaria. |
| T044 | account | La deuda ya no suspende el servicio; queda una desactivación manual de usuario que debe revertirse. | Revisar frontera billing/account. |
| T045 | account | La integración funciona; solicita rotar y revocar una credencial por cambio de contratista. Se trata como administración de acceso, no avería. | Revisar frontera technical/account. |
| T046 | other | Es una revisión legal previa a contratar sobre ubicación y retención de datos, sin incidente ni petición de modificar una cuenta. | No prioritaria. |
| T047 | other | Pide información de accesibilidad para decidir una compra; no reporta un defecto observado en una instalación. | Revisar frontera other/technical. |
| T048 | other | Pregunta por cobertura horaria de atención durante feriados, sin incidencia concreta. | No prioritaria. |
| T049 | other | Consulta una capacidad funcional del producto antes de contratar; no cuestiona cobros ni comprobantes del abono de Nubbix. | Revisar frontera other/billing. |
| T050 | other | No identifica pantalla, operación ni síntoma. “No me deja” no basta para inferir una avería o un problema de permisos: corresponde pedir aclaración. | Revisar frontera other/technical/account. |

## Integridad y límites

Las pruebas comprueban 50 IDs únicos, la distribución acordada y un hash canónico de todos los campos de T001–T030, independiente de saltos de línea. También evitan consultas exactamente iguales después de normalizar mayúsculas, acentos y puntuación. La diversidad semántica requiere revisión humana; una comprobación de cadenas no la garantiza.

El conjunto adversarial sigue en `evals/adversarial.jsonl` y no se mezcla con clasificación. No se modificaron prompts, proveedores ni reportes históricos. Los resultados históricos de 30 tickets no son resultados del conjunto de 50. Los checkpoints anteriores no deben reutilizarse con el nuevo hash.
