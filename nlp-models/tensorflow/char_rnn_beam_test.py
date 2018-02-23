from char_rnn_beam import RNNTextGen


if __name__ == '__main__':
    with open('./temp/anna.txt') as f:
        text = f.read()
    
    model = RNNTextGen(text, seq_len=200)
    log = model.fit()
