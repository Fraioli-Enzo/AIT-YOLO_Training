from ultralytics import YOLO
import torch
import os

def train_model():
    model = YOLO("yolov10n.pt")  # Load YOLOv8/v10/v11 model
    model.train(data="D:\Enzo\Dataset\RoboflowDataset\data.yaml", epochs=1, batch=1, imgsz=640, project="YOLO_Training")
    # CHANGE THE PATH TO YOUR DATASET YAML FILE /!\ /!\

def evaluate_model():
    # Le chemin correct vers le modèle entraîné
    model = YOLO("YOLO_Training/train/weights/best.pt")
    metrics = model.val(project="YOLO_Training")
    print(metrics)

def export_model():
    model = YOLO("YOLO_Training/train/weights/best.pt")
    model.export(format="torchscript")  


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("####################################")
        print("CUDA is available. Training on GPU.")
        print("####################################")
    else:
        print("----------------------/!\CPU/!\----------------------")

    if not os.path.exists("YOLO_Training"):
        os.makedirs("YOLO_Training")

    train_model()
    print("Training completed.")
    
    # Vérifiez si le fichier de modèle existe avant de continuer
    if os.path.exists("YOLO_Training/train/weights/best.pt"):
        evaluate_model()
        print("Evaluation completed.")
        
        export_model()
        print("Model exported.")
    else:
        print("ERROR: Le fichier de modèle entraîné n'a pas été trouvé.")
        print("Chemin attendu: YOLO_Training/train/weights/best.pt")
        print("Vérifiez le dossier de sortie de l'entraînement et ajustez les chemins.")


### Uncomment the following lines to check CUDA version and GPU details (and comment the lines above)###
# print("CUDA version de PyTorch :", torch.version.cuda)  
# print("CUDA disponible :", torch.cuda.is_available())  
# print("Nombre de GPU détectés :", torch.cuda.device_count())  

# if torch.cuda.is_available():
#     print("Nom du GPU :", torch.cuda.get_device_name(0))  
# else:
#     print("❌ PyTorch ne détecte pas CUDA")


