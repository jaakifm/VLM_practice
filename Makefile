IMAGE_NAME=practica-vlm:latest
CURRENT_DIR := $(shell cd)
DATA_DIR := $(CURRENT_DIR)/datasets
JSON_FILE := $(CURRENT_DIR)/labels_captions.json

.PHONY: build run run-eval clean help

build:
	docker build -t $(IMAGE_NAME) .

# Ejecuta el contenedor SIN evaluación 
run: build
	docker run --rm -it -v "$(DATA_DIR):/app/datasets" -v "$(JSON_FILE):/app/labels_captions.json" $(IMAGE_NAME)

# Ejecuta el contenedor CON evaluación 
run-eval: build
	docker run --rm -it -v "$(DATA_DIR):/app/datasets" -v "$(JSON_FILE):/app/labels_captions.json" $(IMAGE_NAME) --evaluate

clean:
	docker image rm -f $(IMAGE_NAME) || true

help:
	@echo "Comandos disponibles:"
	@echo "  make build      - Construir la imagen Docker"
	@echo "  make run        - Ejecutar SIN evaluación (modo original)"
	@echo "  make run-eval   - Ejecutar CON evaluación y métricas"
	@echo "  make clean      - Eliminar la imagen Docker"
	@echo "  make help       - Mostrar esta ayuda"