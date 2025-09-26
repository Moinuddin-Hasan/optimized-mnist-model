import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
from torchsummary import summary
from tqdm import tqdm
import argparse

# Import models from the 'models' folder
from models.model_step1 import Model_1
from models.model_step2 import Model_2
from models.model_step3 import Model_3

def get_data_loaders(use_augmentation=False):
    # Define transforms
    if use_augmentation:
        # For Step 3, we add data augmentation
        train_transforms = transforms.Compose([
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        # For Steps 1 and 2, no augmentation
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = nn.functional.nll_loss(y_pred, target)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

def main():
    parser = argparse.ArgumentParser(description='MNIST CNN Training')
    parser.add_argument('--model', type=str, required=True, choices=['model_step1', 'model_step2', 'model_step3'], help='Which model to train')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if args.model == 'model_step1':
        model, epochs, use_augmentation, use_scheduler = Model_1().to(device), 20, False, False
    elif args.model == 'model_step2':
        model, epochs, use_augmentation, use_scheduler = Model_2().to(device), 20, False, False
    else: # model_step3
        model, epochs, use_augmentation, use_scheduler = Model_3().to(device), 15, True, True

    summary(model, input_size=(1, 28, 28))
    train_loader, test_loader = get_data_loaders(use_augmentation=use_augmentation)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = None
    if use_scheduler:
        scheduler = OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=5/epochs, div_factor=100, anneal_strategy='linear')

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        train(model, device, train_loader, optimizer, epoch, scheduler)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()