import os
import json
import torch
import logging

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss

from dataset import ArcDatasetTest, ArcDatasetTrainANDLearn
from model import DinoArcLearn
from utils import load_json, remove_padding, set_seed

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

LOG = logging.getLogger(__name__)


def get_train_learn_data(task_data: dict):

    train_data = task_data['train']
    test_data = task_data['test']

    train_task_arc = ArcDatasetTrainANDLearn(
        train_data=train_data,
        shuffle=False,
    )

    dataloader_train = DataLoader(  
        train_task_arc,  
        batch_size=1, 
        pin_memory=True,  
        drop_last=False,
    )
    
    test_task_arc = ArcDatasetTest(
        test_data=test_data,
        shuffle=True,
    )

    dataloader_test = DataLoader(  
        test_task_arc,  
        batch_size=1, 
        pin_memory=True,  
        drop_last=False,
    )
    return dataloader_train, dataloader_test


def get_model(checkpoint):

    model = DinoArcLearn()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model
    

def train_and_learn(data_path: str, 
                    model: DinoArcLearn, criterion: CrossEntropyLoss, optimizer: Adam, scheduler: StepLR, num_epochs: int):
                
    challenges = load_json(data_path)
    submission_dict = {} 

    for task_id, task_data in challenges.items():

        
        train_loader, dataloader_test = get_train_learn_data(task_data)
        attempts = [] 

        LOG.info(f'Test on task_id attempt 1 : {task_id}')        
        model.eval()
        tqdm_bar_test = tqdm(enumerate(dataloader_test, 1), desc='Test Progress', leave=True)

        attempt_results_1 = []

        for _, (inputs) in tqdm_bar_test:

            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1) 
            trimmed_grid = remove_padding(predictions.squeeze(0).cpu(), pad_value=10)
            attempt_results_1.append(trimmed_grid.tolist())
        
        LOG.info(f'Learn on task_id : {task_id}')

        for epoch in range(0, num_epochs):

            model.train()

            tqdm_bar = tqdm(enumerate(train_loader, 1), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
            total_loss_batch, count_batches = 0., 0
        
            for _, (inputs, targets) in tqdm_bar:

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

        LOG.info(f'Test on task_id : {task_id}')        
        model.eval()
        tqdm_bar_test = tqdm(enumerate(dataloader_test, 1), desc='Test Progress', leave=True)

        attempt_results_2 = []

        for _, (inputs) in tqdm_bar_test:

            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1) 
            trimmed_grid = remove_padding(predictions.squeeze(0).cpu(), pad_value=10)
            attempt_results_2.append(trimmed_grid.tolist())

        # Use the same results for both attempts
        attempts.append({
            'attempt_1': attempt_results_1,
            'attempt_2': attempt_results_2  # Duplicate the same results
        })
        submission_dict[task_id] = attempts

    return submission_dict


def main(test_path):
    
    LOG.info("Using default seed of 42")
    set_seed(42)

    reload_from_checkpoint = "ai_model/model_state_epoch_1352.pt"
    LOG.info(f"Reloading model from checkpoint: {reload_from_checkpoint}")
    checkpoint = torch.load(reload_from_checkpoint, map_location=device)

    model = get_model(checkpoint)

    criterion = CrossEntropyLoss() 
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.99)
    
    LOG.info(f'Start train and Learn ... ')

    sub_dict = train_and_learn(data_path=test_path,
                    model=model,  
                    criterion=criterion, 
                    optimizer=optimizer,
                    scheduler=scheduler, 
                    num_epochs=100)

    with open('/data/submission.json', 'w') as file:
        json.dump(sub_dict, file, indent=4)

    return sub_dict


if __name__ == "__main__":
    LOG.info("Script started")
    test_path = "/data/arc-agi_test_challenges.json"
    main(test_path)
    LOG.info("Script finished")

