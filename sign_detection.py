import cv2
import numpy as np
import os

import torch
import torch.nn as nn
import torchvision.models as models

# --- Initialisation de la caméra ---
cap = cv2.VideoCapture(0)  # 0 = webcam par défaut



# --- CHARGEMENT DU MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"

class ResNet1Chan(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapter à 1 canal
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



model = ResNet1Chan()
model.load_state_dict(torch.load("model_weights/resnet_model.pth",  map_location=torch.device('cpu')))
model.to(device)
model.eval()

last_idx = 0
old_probs = None
cutoff_freq = 0.8
classes = ["0", "1", "2", "3", "4", "5", "metal", "tel"]
# -----------------------



if not cap.isOpened():
    print("Erreur : impossible d'accéder à la caméra.")
    exit()

print("Appuyez sur 'Echap' pour quitter.")

# --- Boucle de capture ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de lecture de la frame.")
        break

    frame = cv2.resize(frame, (640, 480))


    # Lisse la donnee avec un filtre bilateral
    blurred = cv2.bilateralFilter(frame, d=15, sigmaColor=75, sigmaSpace=75)

    
    # # segmentation de contenu en fonction de la couleur de peau en utilisant HSV #
    # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
    # upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
    # mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
    # #----------------------------------------------------------------------------#

    # # segmentation de contenu en fonction de la couleur de peau en utilisant HSV #
    # ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    
    # lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    # upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    # mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
    # #----------------------------------------------------------------------------#



    # SEGMENTATION RGB
    img_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    cond1 = (R > 95) & (G > 40) & (B > 20)
    cond2 = (np.max(img_rgb, axis=2) - np.min(img_rgb, axis=2)) > 15
    cond3 = (np.abs(R - G) > 15) & (R > G) & (R > B)

    mask_rgb = cond1 & cond2 & cond3
    mask_rgb = mask_rgb.astype(np.uint8) * 255




    # Fusion les deux resultats pour supprimer les donnees aberrantes et garder seulement ce qui est similaire entre les deux images
    #mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    mask = mask_rgb




    # Suppression de contenu residuel (grenailles) et bouche les trous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))           # creation du kernel avec une forme elliptique
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


    # Lissage du resultat pour boucher des petit trous
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # utilisation d'openCV pour determiner les different contour des elements sur la scene (surement la main et un visage ou autre)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()
    result_only_hand = frame.copy()
    mask_hand = np.zeros_like(frame, dtype=np.uint8)

    if len(contours) > 0:
        # On garde le plus grand contour (probablement la main)
        # si la main est au premier plan alors son contour est le plus grand
        max_contour = max(contours, key=cv2.contourArea)                        
        if cv2.contourArea(max_contour) > 5000:             # on evite de recuperer d'utiliser le contour si est trop petit (ce n'est surement pas une main)

            # initalisation d'un masque sur les dimension de la frame
            mask_hand = np.zeros_like(frame, dtype=np.uint8)


            # # Optionnel : dessine les contours
            #cv2.drawContours(result, [max_contour], -1, (0, 255, 0), 2)

            # utilisation d'openCV pour dessiner les contours remplit sur le masque initalise plus tot
            cv2.drawContours(mask_hand, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            
            # Appliquer le masque : tout ce qui n’est pas dans la main devient noir
            # La main est gardé en couleur
            #result_only_hand = cv2.bitwise_and(frame, mask_hand)
            # result_only_hand = cv2.cvtColor(result_only_hand, cv2.COLOR_BGR2)

            # # Optionnel : dessiner un rectangle englobant
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)


    # Affichage du resultat
    #cv2.imshow("Flux camera", frame)
    #cv2.imshow("Toutes les surfaces de peau", mask)
    cv2.imshow("Main isolee en couleur", result)

    mask_hand = cv2.cvtColor(mask_hand, cv2.COLOR_BGR2GRAY)
    mask_hand = cv2.resize(mask_hand, (224, 224))
    cv2.imshow("Main isolee en noir et blanc", mask_hand)
    mask_hand = torch.from_numpy(mask_hand).to(dtype=torch.float32) / 255.0
    mask_hand = mask_hand.unsqueeze(0).unsqueeze(0) 

    with torch.no_grad():

        x = mask_hand.to(device)
        outputs = model(x)                # shape (1, num_classes)

        probs = torch.softmax(outputs, dim=1)   # (1, C)

        # Exponential Moving Average sur TOUTES les classes
        if old_probs is None:
            smooth_probs = probs
        else:
            smooth_probs = cutoff_freq * probs + (1 - cutoff_freq) * old_probs

        # Classe finale
        final_prob, final_idx = smooth_probs.max(dim=1)

        # Sauvegarde pour la prochaine itération
        old_probs = smooth_probs.detach()


        if(last_idx != final_idx.item()):
            last_idx = final_idx.item()
            print(classes[last_idx])

    # Action pour quitter
    if cv2.waitKey(1) & 0xFF == 27:
        break

# libere la memoire et ferme les fenetres
cap.release()
cv2.destroyAllWindows()

