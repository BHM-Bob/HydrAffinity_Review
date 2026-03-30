copy from https://github.com/ai4protein/ProSST

modified:
1. models/_utils/prosst/structure/get_sst_seq.py: L293: remove multiprocess logic, use single process
2. models/_utils/prosst/structure/get_sst_seq.py: L228: use 'EveryThingOK' as pdb name, to fit StringIO
3. models/_utils/prosst/structure/get_sst_seq.py: disable some print and set leave=False to avoid tqdm print multiple times