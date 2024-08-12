
import torch
import random

from matplotlib import colors
from utils import load_json, pad_to_30x30, scale_and_pad


class ArcDatasetFullTrain(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        shuffle=True,
        test=False,
        scale_factor=7,
        target_size=224,
        padding_color=10,
    ):
        super(ArcDatasetFullTrain, self).__init__()
        self.path = path
        self.shuffle = shuffle
        self.test = test
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.padding_color = padding_color
        self.cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])

        self.samples = self.__load_samples(self.path)
        self.length = len(self.samples)

    def __load_samples(self, path):

        training_challenges = load_json(path)
        
        samples = []

        # Iterate over all tasks in the training challenges
        for _, task_data in training_challenges.items():

            # Iterate over each example in the task
            for example in task_data['train']:

                input_grid = example['input']
                output_grid = example['output']

                # Scale and pad the input grid to 224x224
                input_pair = scale_and_pad(input_grid, self.target_size, self.scale_factor, self.cmap, self.padding_color)

                # Pad the output grid to 30x30
                output_pair = pad_to_30x30(output_grid)
                samples.append((input_pair, output_pair))
        
        if self.shuffle:
            random.shuffle(samples)
        
        return samples

    def __getitem__(self, index):
        
        input_pair, output_pair = self.samples[index]
    
        return input_pair, output_pair
        
    def __len__(self):
        return self.length


class ArcDatasetTrainANDLearn(torch.utils.data.Dataset):
    def __init__(
        self,
        train_data: list,
        shuffle=True,
        test=False,
        scale_factor=7,
        target_size=224,
        padding_color=10,
    ):
        super(ArcDatasetTrainANDLearn, self).__init__()
        self.train_data = train_data
        self.shuffle = shuffle
        self.test = test
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.padding_color = padding_color
        self.cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])

        self.samples = self.__load_samples(self.train_data)
        self.length = len(self.samples)

    def __load_samples(self, train_data):

        samples = []

        for example in train_data:

            input_grid = example['input']
            output_grid = example['output']
        
            input_pair = scale_and_pad(input_grid, self.target_size, self.scale_factor, self.cmap, self.padding_color)
            output_pair = pad_to_30x30(output_grid)
            
            samples.append((input_pair, output_pair))
        
        if self.shuffle:
            random.shuffle(samples)
        
        return samples

    def __getitem__(self, index):

        input_pair, output_pair = self.samples[index]
    
        return input_pair, output_pair
        
    def __len__(self):
        return self.length


class ArcDatasetTest(torch.utils.data.Dataset):
    def __init__(
        self,
        test_data: list,
        shuffle=True,
        test=False,
        scale_factor=7,
        target_size=224,
        padding_color=10,
    ):
        super(ArcDatasetTest, self).__init__()
        self.test_data = test_data
        self.shuffle = shuffle
        self.test = test
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.padding_color = padding_color
        self.cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])

        self.samples = self.__load_samples(self.test_data)
        self.length = len(self.samples)

    def __load_samples(self, test_data):

        samples = []

        for example in test_data:

            input_grid = example['input']
                    
            input_pair = scale_and_pad(input_grid, self.target_size, self.scale_factor, self.cmap, self.padding_color)

            samples.append((input_pair))
        
        if self.shuffle:
            random.shuffle(samples)
        
        return samples

    def __getitem__(self, index):
        input_pair = self.samples[index]
    
        return input_pair
        
    def __len__(self):
        return self.length
