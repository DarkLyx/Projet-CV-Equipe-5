import torch
import torch.nn as nn
import torchvision.models as models



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


def init_model():    
    model = ResNet1Chan()
    model.load_state_dict(torch.load("model_weights/resnet_model.pth",  map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    return model




# last_idx = 0
# old_probs = None
# cutoff_freq = 0.8


# def predict_class(mask_hand) :
#     with torch.no_grad():
#         global old_probs
#         global last_idx

#         x = mask_hand.to(device)
#         outputs = model(x)                # shape (1, num_classes)

#         probs = torch.softmax(outputs, dim=1)   # (1, C)

#         # Exponential Moving Average sur TOUTES les classes
#         if old_probs is None:
#             smooth_probs = probs
#         else:
#             smooth_probs = cutoff_freq * probs + (1 - cutoff_freq) * old_probs

#         # Classe finale
#         _, final_idx = smooth_probs.max(dim=1)

#         # Sauvegarde pour la prochaine itération
#         old_probs = smooth_probs.detach()


#         if(last_idx != final_idx.item()):
#             last_idx = final_idx.item()

#         return last_idx




class GesturePredictor:
    """Gère les prédictions + le smoothing"""

    def __init__(self, cutoff_freq=0.8):
        self.model = init_model().eval().to(device)
        self.cutoff = cutoff_freq
        self.old_probs = None
        self.last_idx = 0

    @torch.no_grad()
    def predict(self, img_tensor):
        img_tensor = img_tensor.to(device)
        outputs = self.model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

        if self.old_probs is None:
            smooth = probs
        else:
            smooth = self.cutoff * probs + (1 - self.cutoff) * self.old_probs

        final_idx = smooth.argmax(dim=1).item()
        self.old_probs = smooth.detach()

        if final_idx != self.last_idx:
            self.last_idx = final_idx

        return self.last_idx
