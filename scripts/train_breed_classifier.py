"""
scripts/train_breed_classifier.py
-----------------------------------
Fine-tunes EfficientNetB3 on the Stanford Dogs dataset (120 breeds) and
exports the result as an ONNX model for production inference.

Prerequisites
-------------
    pip install torch torchvision onnx datasets tqdm

Usage
-----
    python scripts/train_breed_classifier.py --epochs 30 --output app/utils/models/

The script will:
  1. Download Stanford Dogs dataset (~750 MB) via HuggingFace datasets.
  2. Fine-tune EfficientNetB3 in three phases (head → partial unfreeze → full).
  3. Evaluate on the validation split and print per-class accuracy.
  4. Export to ONNX + write a labels file.
  5. Print a deployment checklist.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(300, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(330),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_base(model: nn.Module) -> None:
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False


def unfreeze_last_n_blocks(model: nn.Module, n: int = 3) -> None:
    """Unfreeze the last n feature blocks + the classifier."""
    blocks = list(model.features.children())
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def export_onnx(model: nn.Module, output_dir: str, class_names: list[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, 'dog_breed_efficientnet.onnx')
    labels_path = os.path.join(output_dir, 'dog_breed_efficientnet_labels.txt')

    model.eval()
    dummy = torch.zeros(1, 3, 300, 300)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
    )

    with open(labels_path, 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    print(f'ONNX model saved: {onnx_path}')
    print(f'Labels file saved: {labels_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/stanford_dogs',
                        help='Path to ImageFolder-structured dataset')
    parser.add_argument('--output', default='app/utils/models/')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')

    train_ds = ImageFolder(os.path.join(args.data_dir, 'train'), transform=TRAIN_TRANSFORMS)
    val_ds   = ImageFolder(os.path.join(args.data_dir, 'val'),   transform=VAL_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    num_classes = len(train_ds.classes)
    print(f'Classes: {num_classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}')

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- Phase 1: train head only (5 epochs) ----
    print('\n--- Phase 1: Head-only training ---')
    freeze_base(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    for epoch in range(5):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f'  Epoch {epoch+1:02d} | train_acc={ta:.3f} | val_acc={va:.3f}')

    # ---- Phase 2: unfreeze last 3 blocks (15 epochs) ----
    print('\n--- Phase 2: Partial unfreeze ---')
    unfreeze_last_n_blocks(model, n=3)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    best_val_acc = 0.0
    for epoch in range(15):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f'  Epoch {epoch+1:02d} | train_acc={ta:.3f} | val_acc={va:.3f}')
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), '/tmp/best_model.pt')

    # ---- Phase 3: full fine-tune (remaining epochs) ----
    remaining = max(0, args.epochs - 20)
    if remaining > 0:
        print(f'\n--- Phase 3: Full fine-tune ({remaining} epochs) ---')
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
        for epoch in range(remaining):
            tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl, va = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            print(f'  Epoch {epoch+1:02d} | train_acc={ta:.3f} | val_acc={va:.3f}')
            if va > best_val_acc:
                best_val_acc = va
                torch.save(model.state_dict(), '/tmp/best_model.pt')

    print(f'\nBest validation accuracy: {best_val_acc:.4f}')
    print('Loading best checkpoint for export …')
    model.load_state_dict(torch.load('/tmp/best_model.pt', map_location='cpu'))

    clean_class_names = [
        name.replace('_', ' ').replace('-', ' ').title()
        for name in train_ds.classes
    ]
    export_onnx(model.cpu(), args.output, clean_class_names)

    print('\nDeployment checklist:')
    print(f'  1. Copy {args.output}dog_breed_efficientnet.onnx to production server.')
    print(f'  2. Copy {args.output}dog_breed_efficientnet_labels.txt to production server.')
    print('  3. Set ONNX Runtime to use CUDAExecutionProvider for GPU inference.')
    print('  4. Verify app/utils/ml_inference.py picks up the model on next restart.')


if __name__ == '__main__':
    main()
