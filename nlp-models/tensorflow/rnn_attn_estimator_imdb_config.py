import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=20000)
parser.add_argument('--max_len', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--embedding_dims', type=int, default=128)
parser.add_argument('--rnn_size', type=int, default=128)
parser.add_argument('--attn_size', type=int, default=50)
parser.add_argument('--dropout_rate', type=float, default=0.2)
parser.add_argument('--clip_norm', type=float, default=5.0)
args = parser.parse_args()