import cv2
import numpy as np


import model_definition

import torch
import torch.nn as nn
import torchvision.models as models

classes = ["0", "1", "2", "3", "4", "5", "metal", "tel"]

# --- Initialisation de la caméra ---
cap = cv2.VideoCapture(0)  # 0 = webcam par défaut


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

    # Action pour quitter
    if cv2.waitKey(1) & 0xFF == 27:
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
    mask_hand = np.zeros_like(frame, dtype=np.uint8)

    if len(contours) > 0:
        # On garde le plus grand contour (probablement la main)
        # si la main est au premier plan alors son contour est le plus grand
        max_contour = max(contours, key=cv2.contourArea)                        
        if cv2.contourArea(max_contour) > 5000:             # on evite de recuperer d'utiliser le contour si est trop petit (ce n'est surement pas une main)

            x, y, w, h = cv2.boundingRect(max_contour)

            # Calcul des centres
            cx_frame, cy_frame = frame.shape[1] // 2, frame.shape[0] // 2
            cx_hand,  cy_hand  = x + w // 2,       y + h // 2

            # valeurs de translation pour centrer le resultats
            dx, dy = cx_frame - cx_hand, cy_frame - cy_hand

            # translation
            translated_contour = max_contour + np.array([dx, dy], dtype=np.int32)


            # utilisation d'openCV pour dessiner les contours remplit sur le masque initalise plus tot
            cv2.drawContours(mask_hand, [translated_contour], -1, (255, 255, 255), thickness=cv2.FILLED)




    for_model = cv2.cvtColor(mask_hand, cv2.COLOR_BGR2GRAY)
    for_model = cv2.resize(for_model, (224, 224))
    for_model = torch.from_numpy(for_model).to(dtype=torch.float32) / 255.0
    for_model = for_model.unsqueeze(0).unsqueeze(0) 

    predicted_class_idx = model_definition.predict_class(for_model)
    predicted_class = "class detected : " + classes[predicted_class_idx]

    cv2.putText(
        mask_hand,                   
        predicted_class,             
        (50, 50),                  
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,                     
        (0, 255, 0),         
        2,                       
        cv2.LINE_AA              
    )


    #PRESENTATION DES RESULTATS
    cv2.imshow("Main isolee en couleur", result)
    cv2.imshow("Main isolee en noir et blanc", mask_hand)


# libere la memoire et ferme les fenetres
cap.release()
cv2.destroyAllWindows()

