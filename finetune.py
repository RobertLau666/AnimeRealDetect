import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import wandb

# ------------------------
# 参数设置
# ------------------------
DATA_DIR = './data'  # 包含 train/ 和 val/ 文件夹
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
INPUT_SIZE = 224
MODEL_SAVE_PATH = 'model/finetuned_model.pth'
ONNX_EXPORT_PATH = 'model/caformer_s36_v1.3_fixed/model.onnx'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 初始化 wandb
# ------------------------
def init_wandb():
    wandb.init(
        project="binary-image-classification",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "input_size": INPUT_SIZE,
            "model": "resnet18",
            "num_classes": NUM_CLASSES,
        }
    )

# ------------------------
# 数据加载和预处理
# ------------------------
# data/
# ├── train/
# │   ├── anime/
# │   └── real/
# └── val/
#     ├── anime/
#     └── real/
def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print("Classes:", class_names)
    return dataloaders, dataset_sizes, class_names

# ------------------------
# 模型定义与微调
# ------------------------
def build_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model.to(DEVICE)

# ------------------------
# 训练函数
# ------------------------
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # wandb logging
            wandb.log({
                f"{phase}_loss": epoch_loss,
                f"{phase}_acc": epoch_acc,
                "epoch": epoch + 1
            })

            # 保存最优模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'\nBest val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# ------------------------
# 导出 ONNX 模型
# ------------------------
def export_onnx(model, export_path):
    model.eval()
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    torch.onnx.export(
        model, dummy_input, export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"ONNX model exported to {export_path}")

# ------------------------
# 主函数
# ------------------------
def main():
    init_wandb()
    dataloaders, dataset_sizes, class_names = load_data(DATA_DIR)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer)

    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 导出为 ONNX
    export_onnx(model, ONNX_EXPORT_PATH)

    wandb.finish()

# ------------------------
# 入口
# ------------------------
if __name__ == '__main__':
    main()
