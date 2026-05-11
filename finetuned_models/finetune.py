from utils import *
# from training import *
from model_loading import *

from torch.utils.data import DataLoader
from tqdm import tqdm

### Code is largely inspired from Schmirler et al. (https://www.nature.com/articles/s41467-024-51844-2)
### We thank them for making everything easily accessible

def load_model(checkpoint, filepath, num_labels=1, mixed = True, full = False, deepspeed = False, device = torch.device('cpu')):
# Creates a new PT5 model and loads the finetuned weights from a file
    # load model
    if "esm" in checkpoint:
        model, tokenizer = load_esm_model(checkpoint, num_labels, mixed, full, deepspeed)
    else:
        model, tokenizer = load_T5_model(checkpoint, num_labels, mixed, full, deepspeed)
        
    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath, map_location=device)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model



class Finetune:
    def __init__(self, base_model: PretrainedModel):
        self.base_model = base_model.value.full
        self.save_name = base_model.value.short

        # train / test: check if there is set column, else do manually
        self.trained_model = None
        self.tokenizer = None
        self.history = None

    @staticmethod
    def from_file(trained_model_file: str, base_model: PretrainedModel):
        checkpoint = base_model.value.full
        tokenizer, model = load_model(checkpoint, trained_model_file)

        f = Finetune(base_model)
        f.trained_model = model
        f.tokenizer = tokenizer
        return f


class ModelTester:
    def __init__(self, finetuner: Finetune | str):
        if isinstance(finetuner, Finetune):
            self.finetuner = finetuner
        else:
            self.finetuner = Finetune.load(finetuner)

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
        all_labs = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to(device)
                input_labels = [l for l in batch['labels']]
                all_labs += input_labels
                
                attention_mask = batch['attention_mask'].to(device)
                output = model.float()(input_ids, attention_mask=attention_mask, output_attentions=return_attentions)
                predictions += output.logits.tolist()
                

        if return_labels:
            return predictions, all_labs
        
        return predictions

