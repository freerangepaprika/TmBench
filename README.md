# Welcome to TmBench !

This repo accompanies a paper introducing TmBench, a new benchmark study designed to rigorously evaluate protein thermal **stability** prediction models based on protein language models (pLMs). The benchmark integrates diverse experimental data sources, such as thermal proteome profiling and differential scanning calorimetry, to assess model performance across different conditions.


- Results of the models are in `/results`, one subdirectory per model
- $T_m$ data is in `/data` under `/data/DSC` and `/data/TPP`
- To easily check results use the notebook, see [Quick start](#quick-start) below
- To run our finetuned model locally see [Running prediction models](#running-prediction-models)
- We also present the [Datasets](#datasets) below

Be sure to check out the correlation plots, included at the end of the `check_results.ipynb` notebook !
  
# Quick start

The results and datasets are provided in clear tabular and FASTA formats, making them easy to inspect with any appropriate software.
However, to have an easy look at the results we suggest you use the `check_results.ipynb` notebook (from this directory):

- Make a python environment: `python -m venv tmbench`
- Activate: `source tmbench/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Launch notebook and follow suggestions


# Running prediction models

Here we explain how to reproduce the results (CSV files in the `/results` directory) by running the models yourself.

- For the [models from litterature](#models-from-litterature), we provide the corresponding GitHub repositories.
- For our [homemade fine-tuned models](#homemade-fine-tuned-models), we include code that makes it easy to run them on your own machine.

## Models from litterature

We curated 7 pLM-based $T_m$ predictors from literature, following instructions using the links below:
- PRIME: https://github.com/ai4protein/Pro-Prime
- TemStaPro: https://github.com/ievapudz/TemStaPro
- PPTStab: https://github.com/raghavagps/pptstab/
- DeepSTABp: https://github.com/CSBiology/deepStabP (webserver https://csb-deepstabp.bio.rptu.de/)
- TemBERTure: https://github.com/ibmm-unibe-ch/TemBERTure
- ThermoFormer: https://github.com/mingchen-li/ThermoFormer
- TMProt: https://huggingface.co/spaces/loschmidt/tmprot


## Homemade fine-tuned models

We finetuned a number of pLMs ourselves. You can run the ones from the paper directly from this repository, using another conda environment (we seperated both to avoid heavy downloading if you just want to check out the results).

:exclamation: from the `finetuned_models` directory :exclamation: execute the following commands in you terminal:

- Create the conda environment: `conda env create -f environment.yml` (can take a few minutes) and activate it with `conda activate tmbench_run`
- Run the example: `python3 run.py` from the `/finetuned_models` directory.

Running with no arguments will launch the FINE_650M_FULL_MELTOME model on 'example.fasta' and save as 'out.csv', using a GPU is available.

Usage: 

`python3 run.py [-h] [-m MODEL] [-i INPUT_FASTA] [-o OUTFILE] [-d DEVICE]`

Runs one of our finetuned models on an input FASTA file of your choice. It will try to run on a cuda device by default if there's one on your system; if you do not want this use '-d cpu'. 
options:

  `-h, --help`            show this help message and exit

  `-m, --model MODEL`     Finetuned model ID. Choose among: FINE_650M_NO_INT, FINE_650M_FULL_MELTOME, ESM_650M_FLIP, FINE_3B_NO_INT. Default is FINE_650M_FULL_MELTOME.

  `-i, --input_fasta INPUT_FASTA`
                        Path to the input FASTA file. Default is './example.fasta'.

  `-o, --outfile OUTFILE`
                        Path to the output CSV file. Default is './out.csv'.

  `-d, --device DEVICE`   Device to use: 'cuda', 'cpu'. If 'auto', checks if CUDA available and uses it, otherwise cpu. Default is 'auto'.

  `-w, --overwrite OVERWRITE`
                        Set to 'True' to ignore if destination file already exists and to overwrite, else set to 'False' to keep checking. Default is 'False'.


# Datasets

Two types of experimental $T_m$ data are used in our benchmark :
- High-throughput Thermal Proteome Profiling (TPP), located in `/data/TPP/tpp.fasta`:
  - 31998 proteins from the **Meltome Atlas** [Jarzab et al. 2020].
  - 1567 proteins from the **Toxoplasmagondii** organism [Herneisen et al. 2022]
  - 1279 proteins from the **Trypanosomacruzi** organism [Coutinho et al. 2021]
  - 1064 proteins from the **Geobacillusthermoleovorans** organism [Oztug et al. 2020]
- Low-throughput data, mostly produced by Differential Scanning Calorimetry (DSC):
  - **SCooP_DB**: (`/data/DSC/scoop.fasta`) 246 proteins annotated with $T_m$ from DSC results curated from litterature [Pucci et al., 2017].
  - **NanoMelt_DB**: (`/data/DSC/nanomelt.fasta`) 723 antibodies [Ramon et al., 2025] 
  - **CSP_DB**: (`/data/DSC/csp.fasta`) 98 Cold Shock Proteins (CSP) with $T_m$ determined using various DSC-type [Perl et al. 2000]. These are highly similar proteins with very different $T_m$ values, making them particularly useful for assessing predictors' ability to capture subtle information. For a more complete annotations check data/DSC/csp_full.fasta. 
  - **PETases_DB**: (`/data/DSC/petases.fasta`) 232 PET-degrading proteins annotated with $T_m$ using differential scanning techniques [Norton-Baker et al. 2025].


We also include *synthetic* sequences to assess the models' abilities to distinguish between random or modified sequences and their real counterparts. We propose the following three sets derived from our test set (TEST):
- `FIRST_100`: the sequence is truncated to its first 100 amino acids
- `REVERSED`: the chain of amino acids is reversed
- `SHUFFLED`: the chain of amino acids is shuffled

The latter two sets are particularly interesting for assessing a model's understanding of amino acid frequency: reversing or shuffling an amino acid sequence does not modify its composition, though both function and melting temperature are significantly altered.