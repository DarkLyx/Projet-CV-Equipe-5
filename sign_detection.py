from enum import IntEnum

import cv2
import numpy as np

import skin_segmentation
import model_definition
from game_controls import SubwayController 

import torch

class Mode(IntEnum):
    HSV = 0
    YCRCB = 1
    HY = 2
    RGB = 3
    RY = 4


def hand_detection(mode, frame):
    # Lisse la donnee avec un filtre bilateral
    blurred = cv2.bilateralFilter(frame, d=15, sigmaColor=75, sigmaSpace=75)
    mask = None

    match mode :
        case Mode.HSV: 
            mask = skin_segmentation.hsv_seg(blurred)

        case Mode.YCRCB :
            mask = skin_segmentation.ycrcb_seg(blurred)

        case Mode.HY :
            mask_hsv = skin_segmentation.hsv_seg(blurred)
            mask_ycrcb = skin_segmentation.ycrcb_seg(blurred)
            mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        case Mode.RY :
            mask_rgb = skin_segmentation.rgb_seg(blurred)
            mask_ycrcb = skin_segmentation.ycrcb_seg(blurred)
            mask = cv2.bitwise_and(mask_rgb, mask_ycrcb)

        case _ :
            mask = skin_segmentation.rgb_seg(blurred)


    # Suppression de contenu residuel (grenailles) et bouche les trous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))           # creation du kernel avec une forme elliptique
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


    # Lissage du resultat pour boucher des petit trous
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # utilisation d'openCV pour determiner les different contour des elements sur la scene (surement la main et un visage ou autre)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    return mask_hand





def detect_sign(mode):
    controller = SubwayController()
    model = model_definition.GesturePredictor()

    # --- Initialisation de la caméra ---
    cap = cv2.VideoCapture(0)  # 0 = webcam par défaut
    if not cap.isOpened():
        print("Erreur : impossible d'accéder à la caméra.")
        exit()

    print("Appuyez sur 'q' pour quitter.")



    # --- Boucle principale ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de la frame.")
            break

        # Action pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        frame = cv2.resize(frame, (640, 480))
        mask_hand = hand_detection(mode, frame)


        for_model = cv2.cvtColor(mask_hand, cv2.COLOR_BGR2GRAY)
        for_model = cv2.resize(for_model, (224, 224))
        for_model = torch.from_numpy(for_model).to(dtype=torch.float32) / 255.0
        for_model = for_model.unsqueeze(0).unsqueeze(0) 

        predicted_class_idx = model.predict(for_model)
        predicted_class = "class detected : " + controller.execute(predicted_class_idx)

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
        cv2.imshow("Main isolee en couleur", frame)
        cv2.imshow("Main isolee en noir et blanc", mask_hand)


    # libere la memoire et ferme les fenetres
    cap.release()
    cv2.destroyAllWindows()




def test_luminosity():

    w = 300
    h = 300
    padding = 50

    # --- Initialisation de la caméra ---
    cap = cv2.VideoCapture(0)  # 0 = webcam par défaut
    if not cap.isOpened():
        print("Erreur : impossible d'accéder à la caméra.")
        exit()

    print("Appuyez sur 'q' pour quitter.")
    

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture de la frame.")
            break

        # Action pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = cv2.resize(frame, (w, h))

        
        mask_RGB = hand_detection(Mode.RGB, frame)
        mask_HSV = hand_detection(Mode.HSV, frame)
        mask_YCRCB = hand_detection(Mode.YCRCB, frame)
        mask_HY = hand_detection(Mode.HY, frame)
        mask_RY = hand_detection(Mode.RY, frame)


        cv2.imshow("Main RGB", mask_RGB)
        cv2.moveWindow("Main RGB", 0, 0)

        cv2.imshow("Main HSV", mask_HSV)
        cv2.moveWindow("Main HSV", w + padding, 0)
        
        cv2.imshow("Main YCRCB", mask_YCRCB)
        cv2.moveWindow("Main YCRCB", 0, h + padding)
        
        cv2.imshow("Main HSV + YCRCB", mask_HY)
        cv2.moveWindow("Main HSV + YCRCB", w + padding, h + padding)


        cv2.imshow("Main RGB + YCRCB", mask_RY)
        cv2.moveWindow("Main RGB + YCRCB", 2*w + padding, 0)


    cap.release()
    cv2.destroyAllWindows()