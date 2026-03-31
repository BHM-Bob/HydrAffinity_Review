<!--
 * @Date: 2026-03-23 22:39:40
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2026-03-31 09:42:09
 * @Description: 
-->
This repository contains the code and essential files to reproduce all experiments described in the work **A Multimodal Mixture-of-Experts Framework for Protein–Ligand Affinity Prediction**.
To reproduce this project, please download the PDBBind dataset and various molecular pre-trained models, and use the code in this repository to generate all types of molecular embeddings.


## File Description:
1. config: configuration utils.
2. data: contains the dataset partition (EHIGN standard partition and GEMS clean-split) files.
3. log: log utils.
4. models: data generation and loading, model definition, training, and evaluation.
   1. _blocks: basic building blocks for the model.
      1. attn.py: attention module.
      2. flash_attn.py: fast attention module.
      3. hydraformer.py: Hydraformer module.
      4. mlp.py: MLP module.
      5. moe.py: Mixture of Experts module.
      6. transformer.py: Transformer module.
   2. _utils: utility.
      1. geminimol: fixed GeminiMol code for model loading and inference.
      2. prosst: fixed ProSST code for model loading and inference.
      3. args.py: argument parser.
      4. img_pymol_constructor.py: constructor for PyMOL images.
      5. img_rdkit_constructor.py: constructor for RDKit images.
      6. meter.py: meter for training.
      7. scheduler.py: scheduler.
   3. m1: deprecated model architecture. Now only used for data generation.
   4. m3: Multimodal version of HydrAffinity.
   5. s1: Dual-modal version of HydrAffinity.
5. environment_review.yml: conda environment file. Note that `fair-esm` has naming conflict with ESM3, please install `fair-esm==2.0.0` in a separate environment.
    The original environment file is too big (950 lines), so we pick the essential packages in this file manually.


## Data Availability:
All protein–ligand complex structures and corresponding binding affinity data used in this study were sourced from the publicly available PDBbind database (http://www.pdbbind.org.cn).
The dataset is freely accessible to the research community without restrictions.
You can also download the assembled dataset from https://zenodo.org/records/19336925.

The molecular pretraining models employed in this work are publicly available. Specifically, the model weights can be accessed at the following repositories: 
- ChemBERTa (https://huggingface.co/DeepChem/models); 
- PepDoRA (https://huggingface.co/ChatterjeeLab/PepDoRA);
- MolFormer (https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct); 
- MolAI (https://github.com/AnyoLabs/MolAI-Publication);
- SELFormer (https://huggingface.co/HUBioDataLab/SELFormer);
- UniMol (https://huggingface.co/dptech/Uni-Mol2);
- GeminiMol (https://huggingface.co/AlphaMWang/GeminiMol);
- ImageMol (https://github.com/ChengF-Lab/ImageMol);
- MaskMol (https://github.com/ZhixiangCheng/MaskMol).

General pretrained image models are available through the `torchvision` and `timm` Python packages.

Protein language models include:
- ESM-2 (`esm-fair` Python package);
- ESM-3 (https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1);
- SaProt (https://huggingface.co/westlake-repl/SaProt_650M_AF2);
- ProSST-2048 (https://huggingface.co/AI4Protein/ProSST-2048).

Due to the size limit, we only provide the code for embedding generation.


## Evaluation:
We put our SOTA model weights on Zenodo: https://zenodo.org/records/19336925.