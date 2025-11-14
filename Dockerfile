#usar3.10 para compatibilidad con transformers
FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y git && apt-get clean


COPY requirements.txt /app/requirements.txt
COPY run_vlm.py /app/run_vlm.py
COPY labels_captions.json /app/labels_captions.json

# Instalamos dependencias
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir transformers>=4.40.0 pillow tqdm

ENTRYPOINT ["python3", "run_vlm.py"]
