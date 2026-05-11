from finetune import *

import os, torch,time, datetime
import numpy as np
import pandas as pd
from Bio import SeqIO
import argparse

### Code is largely inspired from Schmirler et al. (https://www.nature.com/articles/s41467-024-51844-2)
### We thank them for making everything easily accessible !!


all_models = ['FINE_650M_NO_INT', 'FINE_650M_FULL_MELTOME', 'ESM_650M_FLIP', 'FINE_3B_NO_INT']


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
    
    ft = Finetune.from_file(path_to_model, base_model=base_models[finetuned_model_id])
    runner = ModelTester(ft)

    # to run mutliple fastas on a same model, add your loop here (after 'runner = ModelTester(ft)') to avoid loading the model multiple times

    fasta = list(SeqIO.parse(input_fasta_path, "fasta"))
    sequences = [str(rec._seq) for rec in fasta]
    fasta_names = [rec.id for rec in fasta]
    frame = pd.DataFrame({'sequence': sequences, 'label': fasta_names})
    
    tms, names = runner.predict(frame, device = dev, return_labels=True)
    tms = np.array(tms).flatten()
    res_frame = pd.DataFrame({'name': names, 'pred': tms})

    res_frame.to_csv(outfile, index=False)


def main():
    parser = argparse.ArgumentParser(description= \
        "Runs one of our finetuned models on an input FASTA file of your choice. It will try to run on a cuda device by default if there's one on your system; if you do not want this use '-d cpu'. Running with no arguments will launch the FINE_650M_FULL_MELTOME model on 'example.fasta' and save as 'example.csv', using a GPU is available."
    )
    parser.add_argument("-m", "--model", type=str, default = 'FINE_650M_FULL_MELTOME', help=f"Finetuned model ID. Choose among: {', '.join(all_models)}")
    parser.add_argument("-f", "--input_fasta", type=str, default = './example.fasta', help="Path to the input FASTA file (e.g., './example.fasta')")
    parser.add_argument("-o", "--outfile", type=str, default = './out.csv', help="Path to the output CSV file (e.g., './example.csv')")
    parser.add_argument("-d", "--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu'. If 'auto', checks if CUDA available and uses it, otherwise cpu.")

    # parse args
    args = parser.parse_args()

    # checking model name
    if not args.model in model_paths:
        msg = f"Model should be one of the following:{', '.joint(all_models)}"
        raise ValueError(msg)
    
    # checking input fasta
    if not os.path.exists(args.input_fasta):
        msg = f"Input fasta '{args.input_fasta}' not found :/"
        raise FileNotFoundError(msg)
    
    nb_seqs = sum(1 for _ in SeqIO.parse(args.input_fasta, "fasta"))

    # checking destination (propose to make dir if not exists)
    dest_dirname = os.path.dirname(args.outfile)
    if dest_dirname != '' and not os.path.exists(dest_dirname):
        msg = f"Destination directory {dest_dirname} not found"
        resp = input(f'Create {dest_dirname} ? y / n: ')
        if resp == 'y':
            os.makedirs(dest_dirname, exist_ok=True)
        else:
            print('Exiting..')
            exit(0)
    
    # determine device
    if args.device == "auto":
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Run the model
    t = time.time_ns()
    run(
        finetuned_model_id=args.model,
        input_fasta_path=args.input_fasta,
        outfile=args.outfile,
        dev=device
    )
    e = time.time_ns()
    s = ns_to_pretty_time(e-t)
    print(f"Time for {nb_seqs} sequences on device='{device}':", s)


if __name__ == '__main__':
    # To run manually use the following (and remove 'main()' below)
    # model_id = 'FINE_650M_FULL_MELTOME'
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # fasta_path = './example.fasta'
    # outfile = './example.csv'
    # run(finetuned_model_id=model_id, input_fasta_path=fasta_path, outfile=outfile, dev=device)
    
    main()