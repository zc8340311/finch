# -*- coding: utf-8 -*-
from self_attn_lm import LM
from io import open


if __name__ == '__main__':
    with open('./temp/beijing.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = LM(text, seq_len=100)
    log = model.fit()