IMAGE_NAME=practica-vlm:latest
CURRENT_DIR := $(shell cd)
DATA_DIR := $(CURRENT_DIR)/datasets
JSON_FILE := $(CURRENT_DIR)/labels_captions.json


# PAra probar sistintos prompts y modelso
MODEL ?= Salesforce/blip-image-captioning-base
PROMPT ?=

.PHONY: build run run-eval clean help run-custom

build:
	docker build -t $(IMAGE_NAME) .

# Ejecuta el contenedor SIN evaluación 
run: build
	docker run --rm -it -v "$(DATA_DIR):/app/datasets" -v "$(JSON_FILE):/app/labels_captions.json" $(IMAGE_NAME)

# Ejecuta el contenedor CON evaluación 
run-eval: build
	docker run --rm -it -v "$(DATA_DIR):/app/datasets" -v "$(JSON_FILE):/app/labels_captions.json" $(IMAGE_NAME) --evaluate



# Ejemplo: make run-custom MODEL=microsoft/git-base-coco
# Ejemplo: make run-custom MODEL=Salesforce/blip-image-captioning-large PROMPT="A photo of..."
run-custom: build
	@echo Ejecutando con modelo: $(MODEL)
	@if not "$(PROMPT)"=="" ( \
		echo Usando prompt: $(PROMPT) & \
		docker run --rm -it -v "$(DATA_DIR):/app/datasets" -v "$(JSON_FILE):/app/labels_captions.json" $(IMAGE_NAME) --model "$(MODEL)" --prompt-template "$(PROMPT)" --evaluate \
	) else ( \
		docker run --rm -it -v "$(DATA_DIR):/app/datasets" -v "$(JSON_FILE):/app/labels_captions.json" $(IMAGE_NAME) --model "$(MODEL)" --evaluate \
	)



clean:
	docker image rm -f $(IMAGE_NAME) || true

help:
	@echo "Comandos disponibles:"
	@echo "  make build      - Construir la imagen Docker"
	@echo "  make run        - Ejecutar SIN evaluación (modo original)"
	@echo "  make run-eval   - Ejecutar CON evaluación y métricas"
	@echo "  make clean      - Eliminar la imagen Docker"
	@echo "  make help       - Mostrar esta ayuda"
	@echo "  make run-custom - Ejecutar con modelo y prompt personalizados"