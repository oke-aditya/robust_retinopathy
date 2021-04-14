import torch
import config
import timm
from PIL import Image
import torchvision.transforms as T


if __name__ == "__main__":
    model = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=5)
    model.load_state_dict(torch.load((config.MODEL_SAVE + "_{epoch}" + ".pt"), map_location="CPU"))
    model = model.eval()

    img = Image.open("file_name")

    img = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    out = model(img)
    print(out)
