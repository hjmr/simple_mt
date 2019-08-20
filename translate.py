# -*- coding: utf-8 -*-

import argparse
import io

import numpy
import chainer
import progressbar

import mt

EOS = 0
UNK = 1


def parse_args():
    parser = argparse.ArgumentParser(description='simple machine translator')
    parser.add_argument('MODEL', type=str, help='specify model file.')
    parser.add_argument('SOURCE', type=str, help='a source text file.')
    parser.add_argument('SOURCE_VOCAB', type=str, help='a source vocabulary file.')
    parser.add_argument('TARGET_VOCAB', type=str, help='a target vocabulary file.')
    parser.add_argument('-w', '--w2v_dim', type=int, default=100,
                        help='the dimension of embedded vector (Word2Vec).')
    parser.add_argument('-u', '--num_units', type=int, default=100,
                        help='the number of neurons in each of LSTM layers.')
    return parser.parse_args()


def load_vocabulary(path):
    with io.open(path, encoding='utf-8') as f:
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<EOS>'] = EOS
    word_ids['<UNK>'] = UNK
    return word_ids


def count_lines(path):
    with io.open(path, encoding='utf-8') as f:
        return sum([1 for _ in f])


def load_text(path, vocabulary):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: {}'.format(path))
    with io.open(path, encoding='utf-8') as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK) for w in words], numpy.int32)
            data.append(array)
    return data


def translate():
    args = parse_args()
    source_vocab = load_vocabulary(args.SOURCE_VOCAB)
    target_vocab = load_vocabulary(args.TARGET_VOCAB)
    source_data = load_text(args.SOURCE, source_vocab)

    source_words = {i: w for w, i in source_vocab.items()}
    target_words = {i: w for w, i in target_vocab.items()}

    model = mt.SimpleMT(source_vocab, target_vocab, args.w2v_dim, args.num_units)
    chainer.serializers.load_npz(args.MODEL, model)
    for s in source_data:
        r = model.translate([model.xp.array(s)])[0]
        source_sentence = ''.join([source_words[x] for x in s])
        result_sentence = ''.join([target_words[y] for y in r])
        print("# source : {}".format(source_sentence))
        print("# result : {}".format(result_sentence))


if __name__ == '__main__':
    translate()
