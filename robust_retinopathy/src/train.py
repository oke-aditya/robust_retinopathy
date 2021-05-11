import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import dataset
import config
import engine
import utils
import timm


if __name__ == "__main__":

    utils.seed_everything(42)
    utils.set_debug_apis(False)

    train_trasforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = dataset.RetinopathyDataset(config.TRAIN_DIR, config.CSV_PATH, transform=train_trasforms)
    # model = model.create_model("")
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)

    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(batch_idx)
        # print(inputs)
        print(target)
        break

    model = timm.create_model(config.MODEL_NAME, pretrained=True, num_classes=5)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if config.USE_AMP:
        from torch.cuda import amp
        scaler = amp.GradScaler()

    for epoch in tqdm(range(config.EPOCHS)):
        train_metrics = engine.train_step(model, train_loader, criterion, device, optimizer, scaler=scaler)

        val_metrics = engine.val_step(model, val_loader, criterion, device)

        # Save model every epoch
        torch.save(model.state_dict(), config.MODEL_SAVE + f"_{epoch}" + ".pt")
