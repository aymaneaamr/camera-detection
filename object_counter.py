import cv2
import numpy as np
from collections import deque

class ObjectCounter:
    """Classe pour gérer le comptage d'objets"""
    
    def __init__(self, history_length=50):
        self.history = deque(maxlen=history_length)
        self.total_count = 0
        self.current_count = 0
        
    def update(self, count):
        """Met à jour le compteur avec le nouveau nombre d'objets"""
        self.history.append(count)
        self.current_count = count
        self.total_count += count
        
    def get_average(self):
        """Retourne la moyenne des détections récentes"""
        if len(self.history) == 0:
            return 0
        return sum(self.history) / len(self.history)
    
    def reset(self):
        """Réinitialise le compteur"""
        self.history.clear()
        self.total_count = 0
        self.current_count = 0

class YOLODetector:
    """Détecteur YOLO pour les objets avancés"""
    
    def __init__(self, config_path=None, weights_path=None, classes_path=None):
        self.net = None
        self.classes = []
        self.load_model(config_path, weights_path, classes_path)
    
    def load_model(self, config_path, weights_path, classes_path):
        """Charge le modèle YOLO"""
        if config_path and weights_path:
            self.net = cv2.dnn.readNet(weights_path, config_path)
        
        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
    
    def detect(self, frame, confidence_threshold=0.5):
        """Détecte les objets dans l'image"""
        if self.net is None:
            return []
        
        height, width = frame.shape[:2]
        
        # Préparation de l'image pour YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Inférence
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        # Traitement des résultats
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                results.append({
                    'box': (x, y, w, h),
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': class_name
                })
        
        return results
