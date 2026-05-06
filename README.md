# Welcome to TmBench !

This repo accompanies a paper introducing TmBench, a new benchmark study designed to rigorously evaluate protein thermal stability prediction models based on protein language models (pLMs). The benchmark integrates diverse experimental data sources, such as thermal proteome profiling and differential scanning calorimetry, to assess model performance across different conditions.


- Results of the models are in `/results`, one subdirectory per model
- $T_m$ data is in `/data` under `/data/DSC` and `/data/TPP`
  
## Quick start

The results and datasets are in clear tabular and fasta format so the files should be easy to inspect using any appropriate software.
However, to have an easy look at the results we suggest you use the `check_results.ipynb` notebook (from this directory):

- Make a python environment: `python -m venv tmbench`
- Activate: `source tmbench/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Launch notebook and follow suggestions


# Datasets

Two types of experimental $T_m$ data are used in our benchmark :
- High-throughput Thermal Proteome Profiling (TPP), located in `/data/TPP/tpp.fasta`:
  - 31998 proteins from the **Meltome Atlas** [Jarzab et al. 2020].
  - 1567 proteins from the **Toxoplasmagondii** organism [Herneisen et al. 2022]
  - 1279 proteins from the **Trypanosomacruzi** organism [Coutinho et al. 2021]
  - 1064 proteins from the **Geobacillusthermoleovorans** organism [Oztug et al. 2020]
- Low-throughput data, mostly produced by Differential Scanning Calorimetry (DSC):
  - **SCooP_DB**: (`data/DSC/scoop.fasta`) 246 proteins annotated with $T_m$ from DSC results currated from litterature [Pucci et al., 2017]
  - **NanoMelt_DB**: (`data/DSC/nanomelt.fasta`)723 antibodies [Ramon et al., 2025] 
  - **CSP_DB**: (`data/DSC/csp.fasta`) 98 Cold Shock Proteins (CSP) with $T_m$ dtermined using various differential scanning techniques [Perl et al. 2000]. They are highly similar proteins with (very) different $T_m$, therfore very useful in assessing predictors' ability to capture subtle information. Note, for more complete annotations check data/DSC/csp_full.fasta.
  - **PETases_DB**: (`data/DSC/petases.fasta`) 232 PET-degrading proteins annotated with $T_m$ using differential scanning techniques [Norton-Baker et al. 2025]


We also insert *synthetic* sequences we designed to asses the models' abilities to differentiate a random or modified variant from its real counterpart sequence. We propose the three following variants derived from our test set (TEST):
- `FIRST_100`: the sequence is truncated to its first 100 amino acids
- `REVERSED`: the chain of amino acids is reversed 
- `SHUFFLED`: the chain of amino acids is shuffled

The latter two sets are particularly interesting to asses a model's understanding of amino acid frequency: reversing or shuffling an amino acid sequence does not modify its composition, although function and melting temperature are obviously highly altered.



# Running prediction models

We curate 7 pLM-based $T_m$ predictors from litterature:
- PRIME: https://github.com/ai4protein/Pro-Prime
- TemStaPro: https://github.com/ievapudz/TemStaPro
- PPTStab: https://github.com/raghavagps/pptstab/
- DeepSTABp: https://github.com/CSBiology/deepStabP (webserver https://csb-deepstabp.bio.rptu.de/)
- TemBERTure: https://github.com/ibmm-unibe-ch/TemBERTure
- ThermoFormer: https://github.com/mingchen-li/ThermoFormer
- TMProt: https://huggingface.co/spaces/loschmidt/tmprot


[Note: 6/05/2026] Our home-made finetuned models will be available shortly through a HuggingFace repository.