from fastai.distributed import *
from fastai.text.all import *
import pandas as pd

path= Path('/scratch/08791/hsingh24/datasets')

dls_lm = torch.load(path/'dls_lm_stack_128_small.pkl')

learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()

learn.path = path
learn = learn.load('lm_stack_exc')
with learn.distrib_ctx():
    learn.fit_one_cycle(1, 2e-2)
learn.save('lm_stack_exc_epoch2')