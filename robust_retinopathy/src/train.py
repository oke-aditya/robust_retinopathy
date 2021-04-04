import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import dataset
import config
import model

from quickvision.models.classification import cnn
from quickvision.utils import seed_everything


if __name__ == "__main__":

    train_transfomrs = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.CenterCrop((224, 224)),
    ])

    train_dataset = dataset.RetinopathyDataset("csv_file_path", transforms=train_transfomrs)
    # model = model.create_model("")
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # if config.USE_AMP:
    #     from torch.cuda import amp
    #     scaler = amp.GradScaler()

    for epoch in tqdm(range(config.EPOCHS)):
        metrics = cnn.train_step(model, train_loader, criterion, device, optimizer)

