# -*- coding: utf-8 -*-

import argparse
import io


EOS = 0
UNK = 1


def parse_args():
    parser = argparse.ArgumentParser(description='simple machine translator')
    parser.add_argument('source_text', type=str, help='a text file.')
    parser.add_argument('vocabulary_file', type=str, help='an output vocabulrary file.')
    return parser.parse_args()


def make_vocabulary(file_name):
    vocab = []
    vocab.append('<EOS>')
    vocab.append('<UNK>')
    with open(file_name) as f:
        for l in f:
            for w in l.strip().split():
                if w not in vocab:
                    vocab.append(w)
    return vocab


def main():
    args = parse_args()
    vocab = make_vocabulary(args.source_text)
    with io.open(args.vocabulary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))


if __name__ == '__main__':
    main()
