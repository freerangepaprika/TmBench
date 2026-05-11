from .finetune import Finetune
from .utils import *
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .training import create_dataset

class ModelTester:
    def __init__(self, finetuner: Finetune | str):
        if isinstance(finetuner, Finetune):
            self.finetuner = finetuner
        else:
            self.finetuner = Finetune.load(finetuner)

    def plot_history(self):
        history = self.finetuner.history

        plot_history(history)

    def predict(self, test_data: pd.DataFrame, batch_size=8, return_attentions = False, remove_border_tokens=True, device=None, return_labels = False):
        assert "sequence" in test_data.columns and "label" in test_data.columns
        model = self.finetuner.trained_model
        tokenizer = self.finetuner.tokenizer
        # Set the device to use
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        test_set = create_dataset(tokenizer, list(test_data['sequence']),list(test_data['label']))
        test_set = test_set.with_format("torch", device=device)

        # Create a dataloader for the test dataset

        # recent edit: changed 'batch' param to 'batch_size'

        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # Put the model in evaluation mode
        model.eval()

        # Make predictions on the test dataset
        predictions = []
        attentions = []
        all_labs = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to(device)
                input_labels = [l.item() for l in batch['labels']]
                attention_mask = batch['attention_mask'].to(device)
                output = model.float()(input_ids, attention_mask=attention_mask, output_attentions=return_attentions)
                predictions += output.logits.tolist()
                all_labs += input_labels

        if return_labels:
            return predictions, all_labs
        
        return predictions