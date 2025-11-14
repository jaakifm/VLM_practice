# VLM_practice

Proyecto de práctica para ejecutar un VLM (Vision-Language Model) local que genera captions para imágenes y calcula sus métricas.

## Descripción

Se ejecuta un modelo de image-captioning (por defecto `Salesforce/blip-image-captioning-base`) sobre un pequeño dataset localizado en la carpeta `datasets/`. Para cada imagen se genera un caption y, si existe una referencia en `labels_captions.json`, se pueden calcular métricas (BLEU, ROUGE, METEOR).

Archivos clave:
- `run_vlm.py` - Script principal que recorre `datasets/`, genera captions con un modelo y calcula métricas.
- `labels_captions.json` - JSON con captions de referencia para algunas imágenes.
- `datasets/` - Carpeta que contine las imagenes utilizades divididas por clases.
- `Dockerfile`, `Makefile` - Para crear una imagen Docker y ejecutar el script en contenedores.

## Requisitos

El proyecto usa Python 3.10 (el `Dockerfile` usa `python:3.10-slim`):

Dependencias principales:
- transformers (>=4.40.0)
- torch (se instala en Docker con la rueda CPU por defecto)
- pillow
- tqdm
- evaluate
- nltk
- rouge_score


## Guía:

El repo incluye un `Dockerfile` y un `Makefile` para facilitar la ejecución en contenedor.
```bat
# Construir imagen (Makefile):
make build

# Ejecutar el contenedor (sin evaluación):
make run

# Ejecutar con evaluación:
make run-eval

# Ejecutar con modelo/prompt personalizado:
make run-custom MODEL=Salesforce/blip-image-captioning-large PROMPT="A photo of"
```


Comportamiento en `run_vlm.py`:
- Para cada imagen procesada se busca la referencia en `labels_captions.json` primero por nombre (sin extensión).
- Si no se encuentra por nombre, busca por coincidencia en el campo `url` que termine con `categoria/filename`.

Si no hay referencia para una imagen, el script solo imprime el caption generado.

## Estructura esperada de `datasets/`

La carpeta `datasets/` debe contener subcarpetas por clase, p. ej:

```
datasets/
  casas/
    cabana.png
    casa_moderna.jpg
  perros/
    bulldog.png
  pajaros/
    canario.png
```

El script recorre recursivamente cada subcarpeta (no sub-subcarpetas) y procesa archivos con extensiones `.png`, `.jpg`, `.jpeg`.

