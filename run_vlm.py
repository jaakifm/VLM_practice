# run_vlm.py
import os
import json
from pathlib import Path
from platform import processor
from PIL import Image
from tqdm import tqdm
import argparse

from transformers import pipeline

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Imports para evaluación
import evaluate
from collections import defaultdict


def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_images(datasets_dir):
    images = []
    for cls in os.listdir(datasets_dir):
        cls_path = os.path.join(datasets_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                images.append({
                    "class": cls,
                    "file": os.path.join(cls_path, fname),
                    "name": fname
                })
    return images

def main():
    parser = argparse.ArgumentParser(description="Run a local VLM captioner on a small dataset")
    parser.add_argument("--datasets-dir", default="datasets", help="Path to datasets folder")
    parser.add_argument("--labels-json", default="labels_captions.json", help="JSON with labels/captions")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base", help="Hugging Face image-captioning model")
    parser.add_argument("--prompt-template", default="", help="Optional prompt/template to prepend (prompt-tuning)")
    parser.add_argument("--evaluate", action="store_true", help="Calcular métricas de evaluación")
    args = parser.parse_args()

    labels = load_labels(args.labels_json)
    images = find_images(args.datasets_dir)
    if not images:
        print("No se encontraron imágenes en", args.datasets_dir)
        return

    print("Cargando pipeline de image-to-text:", args.model)




    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model)

    def generate_caption(image):
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs, max_length=64)
        return processor.decode(out[0], skip_special_tokens=True)

    
    predictions = []
    references = []
    results_by_class = defaultdict(lambda: {"predictions": [], "references": []})

    for img_meta in tqdm(images, desc="Imágenes"):
        img_path = img_meta["file"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("ERROR al abrir", img_path, e)
            continue

    
        prompt = args.prompt_template.strip()
        if prompt:
            
            generated = generate_caption(img)

        else:
            generated = generate_caption(img)


        generated = generate_caption(img)

        print("\n------------------------------------")
        print("Fichero:", img_path)
        print("Clase/Carpeta:", img_meta["class"])
        print("Caption generado:", generated)


        gt_text = None
        cls = img_meta["class"]
        name = Path(img_meta["name"]).stem  # sin extension
        if cls in labels and isinstance(labels[cls], dict):
            
            if name in labels[cls]:
                gt_text = labels[cls][name].get("caption")
            else:
                
                for k,v in labels[cls].items():
                    if isinstance(v, dict) and "url" in v and v["url"].endswith(img_meta["class"] + "/" + Path(img_meta["name"]).name):
                        gt_text = v.get("caption")
                        break

        if gt_text:
            print("Caption referencia (JSON):", gt_text)
            # Guardar para evaluación
            predictions.append(generated)
            references.append(gt_text)
            results_by_class[cls]["predictions"].append(generated)
            results_by_class[cls]["references"].append(gt_text)
        else:
            print("Caption referencia (JSON): NO encontrada para este fichero.")

    print("\nFIN. Se han procesado", len(images), "imágenes.")

    # EVALUACIÓN CON MÉTRICAS
    if args.evaluate and predictions and references:
        print("\n" + "="*60)
        print("EVALUACIÓN CON MÉTRICAS OBJETIVAS")
        print("="*60)
        
        # Cargar métricas
        print("\nCargando métricas...")
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load("meteor")
        
        # Calcular métricas globales
        print(f"\nResultados globales ({len(predictions)} imágenes con referencia):")
        print("-" * 60)
        
        # BLEU
        bleu_results = bleu_metric.compute(predictions=predictions, references=references)
        print(f"\n BLEU Score: {bleu_results['bleu']:.4f}")
        print(f"   - BLEU-1: {bleu_results['precisions'][0]:.4f}")
        print(f"   - BLEU-2: {bleu_results['precisions'][1]:.4f}")
        print(f"   - BLEU-3: {bleu_results['precisions'][2]:.4f}")
        print(f"   - BLEU-4: {bleu_results['precisions'][3]:.4f}")
        
        # ROUGE
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
        print(f"\n ROUGE Scores:")
        print(f"   - ROUGE-1: {rouge_results['rouge1']:.4f}")
        print(f"   - ROUGE-2: {rouge_results['rouge2']:.4f}")
        print(f"   - ROUGE-L: {rouge_results['rougeL']:.4f}")
        
        # METEOR
        meteor_results = meteor_metric.compute(predictions=predictions, references=references)
        print(f"\n METEOR Score: {meteor_results['meteor']:.4f}")
        
        # Evaluación por clase
        print("\n" + "="*60)
        print("RESULTADOS POR CLASE/CATEGORÍA")
        print("="*60)
        
        for cls_name, cls_data in sorted(results_by_class.items()):
            if cls_data["predictions"]:
                print(f"\n Clase: {cls_name.upper()} ({len(cls_data['predictions'])} imágenes)")
                print("-" * 60)
                
                bleu_cls = bleu_metric.compute(
                    predictions=cls_data["predictions"], 
                    references=cls_data["references"]
                )
                rouge_cls = rouge_metric.compute(
                    predictions=cls_data["predictions"], 
                    references=cls_data["references"]
                )
                meteor_cls = meteor_metric.compute(
                    predictions=cls_data["predictions"], 
                    references=cls_data["references"]
                )
                
                print(f"   BLEU: {bleu_cls['bleu']:.4f}")
                print(f"   ROUGE-L: {rouge_cls['rougeL']:.4f}")
                print(f"   METEOR: {meteor_cls['meteor']:.4f}")
        
        # Análisis cualitativo
        print("\n" + "="*60)
        print("ANÁLISIS CUALITATIVO")
        print("="*60)
        print("\nEjemplos de predicciones vs referencias:")
        print("-" * 60)
        
        # Mostrar algunos ejemplos (máximo 5)
        for i in range(min(5, len(predictions))):
            print(f"\nEjemplo {i+1}:")
            print(f"  Predicción: {predictions[i]}")
            print(f"  Referencia: {references[i]}")
        
        print("\n" + "="*60)
        print("INTERPRETACIÓN DE LAS MÉTRICAS:")
        print("="*60)
        print("""
- BLEU (0-1): Mide la precisión de n-gramas. Valores >0.3 son buenos.
- ROUGE-L (0-1): Mide la subsecuencia común más larga. Valores >0.4 son buenos.
- METEOR (0-1): Considera sinónimos y stemming. Valores >0.3 son buenos.

Valores más altos indican mayor similitud con las referencias.
        """)

    elif args.evaluate:
        print("\n  No se encontraron suficientes imágenes con referencias para evaluar.")
        print("Asegúrate de que el archivo labels_captions.json contiene captions para las imágenes procesadas.")

if __name__ == "__main__":
    main()