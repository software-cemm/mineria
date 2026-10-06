# U2-T3 · Análisis: limpieza de datos sobre código existente

**Estudiante:** Skay Gisell Alvarado Rodriguez · **Tabla propia:** `customer_credit_clean_alvarado`

## 1. Parte A: comparación de mi pipeline contra el del repo

Datos de partida: `customer_credit_transactions` con 2017 filas (9 registros base + 2008 sintéticos con `seed_synthetic.py --n 2000`, que genera nulos, duplicados, outliers y regiones con diferentes formatos). Ejecuté el pipeline del repo (`src/cleaning.py` → `customer_credit_clean`) y mi versión (`scripts/cleaning_alvarado.py` → `customer_credit_clean_alvarado`).

| Métrica (annual_income) | Repo | Mío |
|---|---:|---:|
| Filas | 2009 | 2017 |
| Media | 33,609.95 | 33,667.89 |
| Desviación estándar | 16,780.03 | 16,867.41 |
| Varianza | 281,569,322.23 | 284,509,593.78 |
| Máximo | 74,926.56 | 75,437.75 |

- Filas que están solo en mi tabla: **8**.
- Al comparar por `transaction_id` las filas que están en ambas tablas, encontré diferencias en `annual_income` (172), `credit_score` (4) y `loan_amount` (110). En `age` y `region` no encontré diferencias.
- En mi pipeline, la varianza de `annual_income` bajó de 711,013,895.22 a 284,509,593.78 después de realizar la imputación y el tratamiento de outliers, lo que representa una reducción aproximada del 60 %.

### ¿Coinciden las filas? ¿Por qué difieren?

No coinciden completamente: el repo termina con 2009 filas y mi versión con 2017. La diferencia de 8 filas se debe a que el repo elimina duplicados exactos y mi versión los conserva.

En las filas que sí coinciden, `age` y `region` quedan iguales, pero existen diferencias en `annual_income`, `credit_score` y `loan_amount`. Esto ocurre porque los dos pipelines realizan la imputación y el tratamiento de outliers, pero no trabajan exactamente con los mismos datos. El repo primero elimina los duplicados y también valida algunos rangos de las variables numéricas, por lo que la mediana y los límites utilizados para el IQR pueden cambiar.

Por esta razón, aunque los dos pipelines utilizan técnicas similares, algunos valores finales no son iguales.

### ¿Qué hace el repo que mi versión no hace?

- **Elimina duplicados** de las filas. En esta ejecución eliminó 8 registros.
- **Valida los rangos de las variables numéricas** (`age`, `annual_income`, `credit_score` y `loan_amount`). Yo solamente valido `age`.
- En el repo se encontraron 7 valores fuera de rango: 2 en `age`, 2 en `annual_income` y 3 en `credit_score`.
- **Genera un reporte por cada paso** con información antes y después de la limpieza. Mi versión muestra principalmente los resultados generales.
- Tiene funciones adicionales como `audit_frame` y `detect_outlier_mask`, además de las opciones `drop` y `log` para el tratamiento de outliers.
- En el caso de `knn`, el repo utiliza las 4 columnas numéricas al mismo tiempo, mientras que mi versión trabaja con la columna indicada.

### ¿Qué pasaría en producción si mi versión hubiera reemplazado a la del repo?

1. **Podrían fallar partes que ya dependen del código existente.** Las funciones del repo devuelven `(DataFrame, reporte)`, mientras que mis funciones devuelven solamente un DataFrame. Los tests y algunas partes de la API esperan la estructura original.

2. **Se conservarían los 8 registros duplicados**, por lo que podrían existir registros repetidos en la tabla limpia.

3. **Algunos datos inválidos podrían pasar a las siguientes etapas**, ya que mi versión solamente valida `age` y no las otras variables numéricas.

4. **Sería más difícil revisar los cambios realizados durante la limpieza**, porque mi versión no tiene el mismo nivel de detalle en el reporte de cada etapa.

5. **Los resultados de la tabla `customer_credit_clean` podrían cambiar** con respecto a los valores que ya esperan las partes que utilizan esta tabla.

## 2. Parte B2: bug hunting en `impute_column`

### Bugs encontrados

1. `impute_column(df, col, "mode")` con una columna que tiene todos sus valores como nulos produce un **IndexError**, porque `series.mode()` no devuelve ningún valor y luego se intenta acceder a `iloc[0]`.

2. `impute_column(df, col, "knn")` con alguna columna numérica completamente nula produce un **ValueError: Columns must be same length as key**. Esto ocurre porque `KNNImputer` puede descartar una columna que no tiene ningún dato disponible y después el resultado tiene menos columnas de las que se intentan asignar.

Impacto: el error ocurre antes del `to_sql` final, por lo que el pipeline se detiene y `customer_credit_clean` puede quedar con los datos anteriores.

### Test antes del arreglo

Escribí `tests/test_cleaning_alvarado.py` con 3 tests para comprobar estos casos. Con el código original los 3 tests fallaron, debido a los errores `IndexError` y `ValueError`.

### Arreglo

En `src/cleaning.py` hice un cambio pequeño para manejar estos casos. En KNN se utilizan solamente las columnas numéricas que tienen al menos un dato disponible. Para `mode`, se comprueba primero si existe una moda antes de intentar utilizarla. Si no existe, la columna se mantiene sin cambios.

También mantuve la firma y el retorno de las funciones para no afectar las partes del proyecto que ya utilizan este código.

### Verificación

Después de realizar los cambios, la suite pasó de **26 a 29 tests**, y los 5 tests originales de `test_cleaning.py` continuaron funcionando correctamente.

### Observación (no corregida)

Con `knn`, el repo imputa las 4 columnas numéricas al mismo tiempo, pero el reporte cuenta solamente los nulos de la columna que se indicó en la función. Lo dejé de esta manera para no modificar el comportamiento que ya tiene el repositorio. Se podría revisar posteriormente si se necesita mejorar el reporte de esta estrategia.
## 3. Auditoría de calidad (salida por consola de `cleaning_alvarado.py`)

- `Tabla 'customer_credit_clean_alvarado' guardada: 2017 -> 2017 filas`
- `Nulos tras normalizar: 146 | nulos finales: 0`
- `Varianza annual_income: 711,013,895.22 -> 284,509,593.78` (reducción de ≈ 60 %)

## 4. Reflexión: ¿por qué capping y no eliminar los registros atípicos?

En riesgo crediticio los valores extremos de `annual_income` y `loan_amount` suelen ser clientes reales con alto ingreso o préstamos grandes, que son justamente los de mayor exposición. Eliminarlos dejaría un dataset sesgado hacia el cliente promedio y subestimaría el riesgo de la cartera. Además, borrar la fila descarta también datos válidos de ese cliente (edad, puntaje de crédito) y reduce la muestra. El capping conserva todos los registros (2017 en mi tabla) y limita la influencia de los extremos: la varianza de `annual_income` bajó cerca de 60 % sin perder una sola fila, lo que protege las medias y los modelos lineales de la distorsión. También deja trazabilidad: cada cliente sigue contabilizado, algo importante cuando el proceso puede auditarse. Tiene un costo: se pierde la magnitud real del extremo (un ingreso de 500K queda tratado como uno de unos 75K). Por eso el criterio correcto es separar los casos: los valores imposibles (fuera de rango de negocio) se tratan como error y se imputan, y los extremos plausibles se acotan en lugar de eliminarse.
