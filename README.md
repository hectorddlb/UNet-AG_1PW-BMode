# UNet-AG — B-mode desde una sola onda plana

Reconstrucción de imagen ecográfica modo B a partir de **una única onda plana**,
mediante una U-Net con *attention gates* condicionadas por F-number, montada
sobre el adjunto de un operador físico de conformación en el dominio de Fourier.

El modelo se entrena **solo con simulaciones de Field II** y se valida en
**PICMUS**, sin ver un solo dato del benchmark durante el entrenamiento: es una
transferencia limpia de simulación a adquisición real.

Este repositorio contiene lo mínimo para replicar ese entrenamiento y esa
validación. Nada más.

---

## Idea

Un conformador clásico de retardo y suma sobre una sola onda plana da una imagen
pobre; la referencia de calidad se obtiene comprimiendo coherentemente 75 ondas
planas, a 75× el coste de adquisición y de cómputo.

UNet-AG parte de la imagen adjunta de **una** onda plana y aprende a acercarla a
la de 75, con tres decisiones de diseño que importan:

- **Conexión residual con `out_conv` inicializada a cero.** En la época 0 el
  modelo *es* exactamente el conformador clásico. Aprende una corrección, no la
  imagen desde cero.
- **Anclaje L1 a la imagen clásica**, con peso β. Impide que la red se aleje de
  la solución físicamente consistente. β = 0.5 es la configuración de la tesis.
- **Attention gates condicionadas por el mapa de F-number**, que le dan a la red
  la geometría de apertura de cada píxel en vez de obligarla a inferirla.

Coste: **372 GFLOP y 3.5 M de parámetros**, frente a 23,417 GFLOP de la
compresión de 75 ondas y 115,875 GFLOP / 552.8 M de parámetros de una línea base
de difusión. El 84 % de ese coste es el operador físico; la red es el 16 %.

---

## Qué hay aquí

```
unet_ag/
  models/           B1V3Residual — la arquitectura propuesta, sobre la U-Net
                    con attention gates condicionadas por F-number
  physics/          operator_fdf.py (OpHFdf: operador de medida en Fourier y
                    su adjunto), su envoltorio y el mapa de F-number
  data/             cargadores de Field II y PICMUS, augmentación de pulso
  train/            common.py (piezas compartidas), b1_v4_anchored.py (pérdida
                    con anclaje) y b1_v5_pulse_aug.py — el punto de entrada
  losses/ metrics/  MSE+L1+SSIM+anclaje; FWHM, CNR, gCNR
  eval/             evaluación en Field II (dentro de dominio) y PICMUS (fuera)
schiffner_das/      F-number dependiente de frecuencia. Genera los targets con
                    das_pw, y el adjunto lo usa en tiempo de ejecución
dataset_fieldii/    scripts MATLAB que generan el dataset de entrenamiento
scripts/            precómputo y alineado de targets
slurm/              los cuatro pasos del pipeline
```

Aquí está solo lo que interviene en el modelo de la tesis. Se han eliminado las
arquitecturas de las propuestas anteriores (A2 y B1-v2), que solo eran
alcanzables por una bandera de línea de comandos, y los módulos de carga y
dibujo del port de Schiffner, que este repositorio no usa.

**No se incluyen** los datasets (~30 GB), los pesos entrenados ni los resultados.
Todo se regenera con el pipeline de abajo.

---

## Instalación

```bash
git clone https://github.com/hectorddlb/UNet-AG_1PW-BMode.git
cd UNet-AG_1PW-BMode

conda create -n unet_ag python=3.11
conda activate unet_ag
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Sin instalación del paquete: los comandos se lanzan con `python -m` desde la raíz
del repositorio, que es lo que hacen los scripts de `slurm/`.

Verificado con Python 3.11.6, PyTorch 2.5.1+cu121, sobre una NVIDIA TITAN RTX
(24 GB). El entrenamiento cabe holgadamente; el paso de targets es el más pesado.

---

## Datos

### Field II — entrenamiento (hay que generarlo)

Requiere MATLAB y el toolbox [Field II](https://field-ii.dk/). Desde
`dataset_fieldii/`:

```matlab
>> run_generation_multiangle
```

Genera 300 fantomas de dispersores puntuales y 300 de quistes, cada uno con
75 ángulos entre −16° y +16°:

```
field_ii_data_multiangle/
  points/points_0001.mat ... points_0300.mat
  cysts/cysts_0001.mat   ... cysts_0300.mat
```

Cada `.mat` contiene `rf_data (max_D, 128, 75)`, `angles_deg`, `t_start` por
ángulo, posiciones y amplitudes de los dispersores, y la geometría de la sonda
(fs = 20.832 MHz, fc = 5.208 MHz, c = 1540 m/s, 128 elementos, paso 300 µm).

Es reproducible: `rng(i)` por fantoma, con semillas disjuntas por tipo. Coste:
de 2 a 7 días de CPU, dominado por los quistes (50.000 dispersores cada uno).

### PICMUS — validación (descarga)

Benchmark público: <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/>

Se usan los cuatro fantomas: `{simulation, experiments} × {resolution_distorsion,
contrast_speckle}`. Basta con conservar la estructura de directorios original.

---

## Pipeline

```bash
export DATA_ROOT="/ruta/a/field_ii_data_multiangle"
export PICMUS_ROOT="/ruta/a/PICMUS"
```

### 1. Targets — Schiffner Fd-F 75 PW

El supervisor del entrenamiento: compresión coherente de 75 ondas planas con
apodización de F-number dependiente de frecuencia.

```bash
sbatch slurm/1_targets.sbatch          # array de 6 × ~2 h en GPU
```

### 2. Alineado de la rejilla — **no es opcional**

```bash
sbatch slurm/2_align.sbatch            # CPU, ~1 h
```

El generador de targets conforma en z ∈ [5, 50] mm con 612 filas; el entrenador
construye la suya como z ∈ [5, (ns−1)·c/(2·fs)] = [5, 54.22] mm, también con 612
filas. Mismo tamaño, distinta escala. Como el cargador solo reinterpola si cambia
el tamaño, el target quedaría **desplazado un 9.4 % en profundidad**: un eco a
44 mm caería 3.7 mm fuera de sitio, muy por encima de la FWHM axial (~1.6 mm), y
la red aprendería a emborronar para minimizar la pérdida.

Se interpola la envolvente compleja (sobremuestreada ~13×) y no la señal de
radiofrecuencia (4 muestras por ciclo), para que el remuestreo sea exacto.

### 3. Entrenamiento

```bash
sbatch slurm/3_train.sbatch            # 20 épocas, ~9 h en GPU
```

Configuración de la tesis: `phantom_type=points`, β = 0.5, 20 épocas, seed 42,
`pulse_apply_prob=0.7`, rejilla 612×388, `base_ch=32`. Salen 3,529,366
parámetros entrenables.

Antes de quemar 9 h conviene comprobar la instalación con un smoke de 1 minuto:

```bash
sbatch --export=ALL,EPOCHS=2,MAX_PHANTOMS_TRAIN=4,MAX_PHANTOMS_VAL=2,SMOKE_STEPS=4 \
       slurm/3_train.sbatch
```

### 4. Validación

```bash
sbatch --export=ALL,RUN=results/unet_ag_beta05 slurm/4_eval.sbatch
```

Produce `eval_fieldii.csv` (dentro de dominio: FWHM lateral/axial por alambre,
CNR y gCNR, comparado en el mismo fichero contra DAS 1 PW y Schiffner Fd-F
75 PW) y `eval_picmus.csv` (fuera de dominio: los cuatro fantomas del
benchmark).

Sin SLURM, cada `.sbatch` es un script de bash normal: la línea `python -m ...`
se puede copiar tal cual.

---

## Resultados

PICMUS, media ± desviación entre 3 seeds. FWHM lateral en mm (↓ mejor);
gCNR (↑ mejor).

| Método | entrada | sim FWHM_lat | exp FWHM_lat | sim gCNR | exp gCNR | GFLOP | parámetros |
|---|---|---|---|---|---|---|---|
| DAS 1 PW | 1 PW | **1.236** | 3.728 | 0.558 | 0.528 | 312 | — |
| Schiffner Fd-F 1 PW | 1 PW | 1.489 | 2.997 | 0.593 | 0.608 | 312 | — |
| DAS 75 PW | 75 PW | 0.770 | 1.734 | 0.964 | **0.798** | 23,417 | — |
| Schiffner Fd-F 75 PW | 75 PW | 0.896 | **1.305** | 0.966 | 0.664 | 23,417 | — |
| DRUS (difusión) | 1 PW | 0.869 | 1.619 | **0.976** | 0.786 | 115,875 | 552.8 M |
| **UNet-AG** | 1 PW | 1.487±0.157 | 2.765±0.245 | 0.564±0.006 | 0.721±0.058 | **372** | **3.5 M** |

**Lectura honesta.** UNet-AG no bate a los métodos de 75 ondas planas en
resolución. Frente a su línea base correcta de una sola onda plana —Schiffner
Fd-F 1 PW, que cuesta lo mismo (312 GFLOP) y no tiene parámetros— el balance
está repartido:

- **Gana en el dominio experimental**: +7.7 % en FWHM lateral y +18.6 % en gCNR,
  con 1.19× su coste en operaciones.
- **No gana en simulación**: empata en resolución (1.487 frente a 1.489) y
  pierde en contraste (gCNR 0.564 frente a 0.593).
- **En contraste experimental** alcanza a una línea base de difusión con **311×
  menos operaciones y 157× menos parámetros**.

Comparar contra DAS 1 PW da cifras más vistosas (−26 % y +37 %), pero es la más
débil de las tres líneas base de una onda plana, así que no se citan aquí.

Dos advertencias que conviene tener presentes al comparar:

1. **El coste es la física, no la red.** De los 372 GFLOP, 312 son el operador
   adjunto y solo 60 la red. En reloj de pared, 12 ms de 5.585 s — el 0.2 %.
   Optimizar el modelo no compra nada; el margen está en el operador.
2. **El «×1.19» es relativo a este pipeline.** El conformador clásico en el
   dominio del tiempo, que es lo que hace un ecógrafo real, son 0.699 GFLOP:
   447× más barato que el kernel de Fourier que usa todo este código. Frente a
   él, UNet-AG sería ~87×.

### Limitación conocida

Buena parte de lo que UNet-AG aporta sobre DAS lo aporta ya la **apodización
Fd-F**, que es clásica, no tiene parámetros y cuesta lo mismo. La contribución
neta del aprendizaje se limita al dominio experimental.

Existe además un **salto de dominio Field II → PICMUS** que no se ha conseguido
cerrar.
Se probaron tres intervenciones ortogonales —el peso del anclaje β, entrenar con
quistes además de dispersores puntuales, y desactivar la augmentación de forma
de pulso— y **las tres mejoran dentro de dominio y degradan en PICMUS**.

El CNR fuera de dominio crece de forma monótona con β, y β → 1 es exactamente el
conformador clásico: cuanto más se parece el modelo a DAS, mejor puntúa fuera de
dominio. Es sobreajuste de dominio, no un hiperparámetro mal elegido.

Queda sin explorar la adaptación de dominio explícita y el cambio de fuente de
datos de entrenamiento.

---

## Créditos

- Conformación con F-number dependiente de frecuencia: M. F. Schiffner
  (implementación aquí portada a PyTorch en `schiffner_das/`).
- Operador de medida en el dominio de Fourier: adaptado del trabajo de Zhang
  et al. sobre reconstrucción autosupervisada con ondas planas.
- [Field II](https://field-ii.dk/) (J. A. Jensen) para la simulación.
- [PICMUS](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/) para la
  validación.

Desarrollado como parte de una tesis de maestría.
