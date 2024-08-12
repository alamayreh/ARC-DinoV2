import os
import time
import yaml

import torch
import logging

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss

from dataset import ArcDatasetFullTrain
from model import DinoArc
from utils import DictDefault, image_to_rgb, set_seed

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

LOG = logging.getLogger(__name__)


def get_dataloader(data_path: str, batch_size:int, split: str):
    LOG.info(f"Creating dataloader for: {split}")

    arcdataset=ArcDatasetFullTrain(
        path=data_path,
        shuffle= split == "train",
    )

    dataloader = DataLoader(  
            arcdataset,  
            batch_size=batch_size, 
            pin_memory=True,  
            drop_last=True,  
        )  

    return dataloader


def get_dataloader_full(data_path_train: str, data_path_valid: str, batch_size:int):
    LOG.info(f"Creating full dataloader")

    train_arc=ArcDatasetFullTrain(
        path=data_path_train,
        shuffle=True,
    )

    valid_arch=ArcDatasetFullTrain(
        path=data_path_valid,
        shuffle=True,
    )

    # Concat valid and train data for more examples in the initial training
    combined_dataset = ConcatDataset([train_arc, valid_arch])

    dataloader = DataLoader(  
            combined_dataset,  
            batch_size=batch_size, 
            pin_memory=True,  
            drop_last=False,  
        )  

    return dataloader


def get_model(checkpoint):

    model = DinoArc()

    if checkpoint:
        model= model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)

    return model


def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                model: DinoArc, criterion: CrossEntropyLoss, optimizer: Adam, scheduler: StepLR, num_epochs: int, output_dir: str):
  
    start_time = time.time()
    tensorboard_writer = SummaryWriter(log_dir=output_dir)

    min_val_loss = float('inf') 

    for epoch in range(0, num_epochs):

        model.train()

        tqdm_bar = tqdm(enumerate(train_loader, 1), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        total_loss_batch, count_batches = 0., 0
    
        for step_num, (inputs, targets) in tqdm_bar:

            inputs = inputs.to(device)  
            targets = targets.to(device)  
        
            outputs = model(inputs)  

            loss = criterion(outputs, targets) 
          
            loss.backward()
            optimizer.step()  
            optimizer.zero_grad()  

            step_loss = loss.item()
            total_loss_batch += step_loss 
            
            tqdm_bar.set_postfix(step_loss=step_loss)
   
            count_batches += 1

        scheduler.step()

        tensorboard_writer.add_scalar('Training/Loss_epoch', total_loss_batch / count_batches, epoch)
            
        val_loss = evaluate_model(val_loader, model, criterion)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            LOG.info(f"\nBest validation loss: {val_loss:.2f}. Saving best model for epoch: {epoch + 1}\n")
            os.makedirs(os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}"), exist_ok=True)           
            model_save_dict = {
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'min_val_loss': min_val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(model_save_dict, os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}", 'model_state.pt'))
        else:
            LOG.info(f"\nValidation loss: {val_loss:.2f}\n")
        
        tensorboard_writer.add_scalar('Validation/Loss', val_loss, epoch) 
                   
    elapsed_time = time.time() - start_time
    LOG.info(f'Training complete in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
    LOG.info(f'Minimum Validation loss: {min_val_loss:4f}')
    tensorboard_writer.close()

    return model


@torch.no_grad()
def evaluate_model(val_loader: DataLoader, model: DinoArc, criterion: CrossEntropyLoss):
    
    model.eval()

    total_loss_batch, count_batches= 0., 0
    tqdm_bar = tqdm(enumerate(val_loader, 1), desc='Evaluation Progress', leave=True)

    for _, (inputs, targets) in tqdm_bar:
 
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss_batch += loss.item()

            count_batches += 1

    avg_loss_batch = total_loss_batch / count_batches

    return avg_loss_batch
   

def main(argv = None):

    args = parse_args(argv)

    with open(args.config, encoding="utf-8") as file:
        config: DictDefault = DictDefault(yaml.safe_load(file)) 
    
    if not config.seed:
        LOG.info("No seed provided, using default seed of 42")
        config.seed = 42

    set_seed(config.seed)

    train_loader = get_dataloader_full(data_path_train=config.training_challenges, data_path_valid=config.evaluation_challenges, batch_size=config.batch_size)
    val_loader = get_dataloader(data_path=config.evaluation_challenges, batch_size=config.batch_size, split='valid')

    if config.show_sample_path:
            batch_tensor, _ = next(iter(train_loader))
            grid_img = make_grid(batch_tensor, nrow=5)
            plt.imsave(f'{config.show_sample_path}/batch_sample.png', image_to_rgb(grid_img.permute(1, 2, 0).numpy()))

    if config.reload_from_checkpoint:
        checkpoint = torch.load(config.model_params.reload_from_checkpoint)
        LOG.info(f"Reloading model from checkpoint: {config.reload_from_checkpoint}")
    else:
        checkpoint = None

    model = get_model(checkpoint)

    criterion = CrossEntropyLoss() 
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.99)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)    
    
    LOG.info(f'Start full training ... ')
    model = train_model(train_loader=train_loader,
                val_loader=val_loader, 
                model=model, 
                criterion=criterion, 
                optimizer=optimizer,
                scheduler=scheduler, 
                num_epochs=config.max_epochs, 
                output_dir=config.out_dir,
                )
    
    LOG.info(f'Training finished ... ')
    

def parse_args(argv):
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, required=True, help="Path to the yaml config file")
    return args.parse_args(argv)


if __name__ == "__main__":
    main()