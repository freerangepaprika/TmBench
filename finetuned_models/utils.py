from enum import Enum
from collections import namedtuple
import pandas as pd
import numpy as np

### Code is largely inspired from Schmirler et al. (https://www.nature.com/articles/s41467-024-51844-2)
### We thank them for making everything easily accessible


ModelName = namedtuple('ModelName', ['full', 'short'])


class PretrainedModel(Enum):
    ESM_8M = ModelName("facebook/esm2_t6_8M_UR50D", "ESM_8M")
    ESM_35M = ModelName("facebook/esm2_t12_35M_UR50D", "ESM_35M")
    ESM_150M = ModelName("facebook/esm2_t30_150M_UR50D", "ESM_150M")
    ESM_650M = ModelName("facebook/esm2_t33_650M_UR50D", "ESM_650M")
    ESM_3B = ModelName("facebook/esm2_t36_3B_UR50D", "ESM_3B")

    PROT_T5_XL = ModelName("Rostlab/prot_t5_xl_uniref50", "PROT_T5_XL")
    PROST_T5 = ModelName("Rostlab/ProstT5", "PROST_T5")
    ANKH_BASE = ModelName("ElnaggarLab/ankh-base", "ANKH_BASE")
    ANKH_LARGE = ModelName("ElnaggarLab/ankh-large", "ANKH_LARGE")

    @staticmethod
    def all():
        return (
            PretrainedModel.ESM_8M,
            PretrainedModel.ESM_35M,
            PretrainedModel.ESM_150M,
            PretrainedModel.ESM_650M,
            PretrainedModel.ESM_3B,
            PretrainedModel.ANKH_BASE,
            PretrainedModel.ANKH_LARGE,
            PretrainedModel.PROST_T5,
            PretrainedModel.PROT_T5_XL,
        )


def pre_process(base_model: "PretrainedModel", frame: pd.DataFrame, in_place=False) -> pd.DataFrame:
    if not in_place:
        frame = frame.copy(deep=True)
    
    frame = frame.dropna()
    # frame["sequence"]=frame["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)
    repl = {non_valid: 'X' for non_valid in ["O","B","U","Z","J"]}
    frame['sequence'] = [sequence.translate(repl) for sequence in frame["sequence"]]

    checkpoint_full_name = base_model.value.full

    if "Rostlab" in checkpoint_full_name:
        frame['sequence']=frame.apply(lambda row : " ".join(row["sequence"]), axis = 1)
        
    # Add <AA2fold> for ProstT5 to inform the model of the input type (amino acid sequence here)
    if "ProstT5" in checkpoint_full_name:    
        frame['sequence']=frame.apply(lambda row : "<AA2fold> " + row["sequence"], axis = 1)  


    return frame