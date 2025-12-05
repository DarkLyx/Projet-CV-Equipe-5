from pynput.keyboard import Key, Controller
import time

class SubwayController:
    def __init__(self):
        self.keyboard = Controller()
        self.last_action_idx = None
        self.last_hoverboard_time = 0
        
        self.MAPPING = {
            0: None,        # Classe "0" -> Neutre (rien ne se passe)
            1: Key.left,      # Classe "1" -> gauche
            2: Key.right,    # Classe "2" -> droite
            3: Key.down,    # Classe "3" -> roulade
            4: None,   # Classe "4" -> Neutre
            5: Key.up,   # Classe "5" -> sauter
            6: Key.esc,        # Classe "metal" -> pause
            7: Key.space      # Classe "tel" -> hoverboard
        }
        
        self.CLASS_NAMES = ["Neutre", "gauche", "droite", "bas", "Neutre", "haut", "pause", "hoverboard"]

    def execute(self, class_idx):
        """
        Reçoit l'index prédit par le modèle et gère les touches.
        """
        if class_idx not in self.MAPPING:
            return

        action_key = self.MAPPING[class_idx]
        action_name = self.CLASS_NAMES[class_idx] if class_idx < len(self.CLASS_NAMES) else str(class_idx)

        if action_key is None:
            self.last_action_idx = class_idx
            return action_name

        
        # if action_key == Key.space:
        #     current_time = time.time()
        #     if class_idx != self.last_action_idx and (current_time - self.last_hoverboard_time > 1.0):
        #         self.keyboard.press(action_key)
        #         self.keyboard.release(action_key)
        #         self.last_hoverboard_time = current_time
        if class_idx != self.last_action_idx:
            self.keyboard.press(action_key)
            self.keyboard.release(action_key)

        self.last_action_idx = class_idx
        
        return action_name