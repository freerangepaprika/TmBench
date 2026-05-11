from finetune import *
import os, torch
import numpy as np
import pandas as pd
from Bio import SeqIO



### Code is largely inspired from Schmirler et al. (https://www.nature.com/articles/s41467-024-51844-2)
### We thank them for making everything easily accessible


model_paths = {
    'FINE_650M_NO_INT': './models/ESM_650M_NO_INT__epochs=20_batch=4_accum=2.pth',
    'FINE_650M_FULL_MELTOME': './models/ESM_650M_FULL_MELTOME__epochs=20_batch=4_accum=2.pth',
    'ESM_650M_FLIP': './models/ESM_650M_FULL_MELTOME__epochs=20_batch=4_accum=2.pth',
    'FINE_3B_NO_INT': './models/ESM_3B_NO_INT__epochs=20_batch=2_accum=4.pth',
}

base_models = {
    'FINE_650M_NO_INT': PretrainedModel.ESM_650M,
    'FINE_650M_FULL_MELTOME': PretrainedModel.ESM_650M,
    'ESM_650M_FLIP': PretrainedModel.ESM_650M,
    'FINE_3B_NO_INT': PretrainedModel.ESM_3B,
}


def run(finetuned_model_id: str, input_fasta_path: str, outfile: str, dev = None):
    path_to_model = model_paths[finetuned_model_id]
    print(f'Running {finetuned_model_id} on {input_fasta_path}')

    if dev is None:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(os.path.dirname(outfile)):
        msg = f"Destination directory {os.path.dirname(outfile)} not found.."
        raise FileNotFoundError(msg)

    
    ft = Finetune.from_file(path_to_model, base_model=base_models[finetuned_model_id])
    runner = ModelTester(ft)

    fasta = list(SeqIO.parse(input_fasta_path, "fasta"))
    sequences = [str(rec._seq) for rec in fasta]
    fasta_names = [rec.id for rec in fasta]
    frame = pd.DataFrame({'sequence': sequences, 'label': fasta_names})
    
    tms, names = runner.predict(frame, device = dev, return_labels=True)
    tms = np.array(tms).flatten()
    res_frame = pd.DataFrame({'name': names, 'pred': tms})

    res_frame.to_csv(outfile, index=False)


if __name__ == '__main__':
    model_id = 'FINE_650M_FULL_MELTOME'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fasta_path = './example.fasta'
    outfile = './example.csv'

    run(finetuned_model_id=model_id, input_fasta_path=fasta_path, outfile=outfile, dev=device)
