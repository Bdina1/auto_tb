from model.models import *
from utils import *
import argparse

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision.datasets import Cityscapes
import torch.utils.data as data
from torchvision.datasets import SBDataset

def generate_random_input(batch_size, num_channels, height, width):
    return torch.rand(batch_size, num_channels, height, width)


def setup(rank, world_size, config):
    os.environ['MASTER_ADDR'] = config['MASTER_ADDR']
    os.environ['MASTER_PORT'] = config['MASTER_PORT']
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()


def train_ddp(rank, world_size, model, dataloader, validation_dataloader, criterion, optimizer, config, writer):
    setup(rank, world_size, config)
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        running_loss = 0.0
        dataloader.sampler = DistributedSampler(dataloader, num_replicas=world_size, rank=rank)
        
        # Create tqdm progress bar
        pbar = tqdm(dataloader, total=len(dataloader))

        for i, data in enumerate(pbar):
            inputs, labels = data[0].to(rank), data[1].to(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm progress bar
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / (i + 1)}")
        
        # Record loss into Tensorboard
        if config.get('log_training_loss', False):
            writer.add_scalar('training loss', running_loss / len(dataloader), epoch)
        if config.get('log_validation_loss', False) and validation_dataloader is not None:
            validation_loss = validate(model, validation_dataloader, criterion, config)
            writer.add_scalar('validation loss', validation_loss, epoch)
    cleanup()


def train_single(model, dataloader, validation_dataloader, criterion, optimizer, config, writer):
    device = torch.device(config['device'])
    model.to(device)

    num_epochs = config.get('num_epochs', 1)
    label = torch.tensor(0).unsqueeze(0)
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, total=len(dataloader))

        for i, data in enumerate(pbar):
            inputs, labels = data[0].to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / (i + 1)}")
            pbar.refresh()
            break

        # Record loss into Tensorboard
        if config.get('log_training_loss', False):
            writer.add_scalar('training loss', running_loss / len(dataloader), epoch)
        if config.get('log_validation_loss', False) and validation_dataloader is not None:
            validation_loss = validate(model, validation_dataloader, criterion, config)
            writer.add_scalar('validation loss', validation_loss, epoch)

    return running_loss / len(dataloader)

def run_training(model, dataloader, validation_dataloader, criterion, optimizer, config, writer):
    # Check if distributed key is in the config and if it's True
    if config.get('distributed', False):
        devices = config.get('device', [])
        if isinstance(devices, list) and len(devices) > 1:  # Multiple GPUs
            world_size = len(devices)
            mp.spawn(train_ddp, args=(world_size, model, dataloader, criterion, optimizer, config, writer), nprocs=world_size, join=True)
        else:  # Single GPU
            train_single(model, dataloader, validation_dataloader, criterion, optimizer, config, writer)
    else:  # Non-distributed training, either single GPU or CPU
        devices = config.get('device', [])
        if isinstance(devices, list) and len(devices) > 0:  # Use the first GPU specified in the list
            config['device'] = devices[0]
        else:  # If no device was specified, use CPU
            config['device'] = 'cpu'
        running_loss = train_single(model, dataloader, validation_dataloader, criterion, optimizer, config, writer)
        
        # Record final loss into Tensorboard
        if config.get('log_training_loss', False):
            writer.add_scalar('training loss', running_loss, config.get('num_epochs', 1) - 1)
        
        if config.get('log_validation_loss', False) and validation_dataloader is not None:
            for epoch in range(config.get('num_epochs', 1)):
                validation_loss = validate(model, validation_dataloader, criterion, config)
                writer.add_scalar('validation loss', validation_loss, epoch)



def validate(model, validation_dataloader, criterion, config):
    device = torch.device(config['device'])
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(inputs)
            num_samples += len(inputs)
    
    model.train()
    
    if num_samples > 0:
        average_loss = total_loss / num_samples
        return average_loss
    else:
        return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="")
    parser.add_argument("--epochs", default=10)
    
    args = parser.parse_args()
    config = read_config_file(args.cfg)
    
    writer = SummaryWriter(config['TB_path'])
    model = CustomModel(config, writer)

    criterion = getattr(nn, config['loss_function'])(reduction='mean')
    
    # Define the optimizer dynamically
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    optimizer_args = {k: v for k, v in optimizer_config.items() if k != 'name'}
    optimizer = getattr(optim, optimizer_name)(model.parameters(), **optimizer_args)
    


    train_dataset = generate_random_input(1, 3, 256, 256)
    val_dataset = generate_random_input(1, 3, 256, 256)

    # train_dataset = SBDataset(root='./', mode='segmentation', download=False)
    # val_dataset = SBDataset(root='./', mode='segmentation', download=True)


    dataloader = data.DataLoader([train_dataset, 0],batch_size=1)
    validation_dataloader = data.DataLoader(val_dataset,batch_size=1)
    
    # Start the training
    run_training(model, dataloader, None, criterion, optimizer, config, writer)
    
    # Close the TensorBoard writer
    writer.close()
    