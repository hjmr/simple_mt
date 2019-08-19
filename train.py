# -*- coding: utf-8 -*-

import argparse

import six
import numpy
import chainer
from chainer import training
from chainer.training import extensions

import mt

EOS = 0
UNK = 1


def parse_args():
    parser = argparse.ArgumentParser(description='simple machine translator')
    parser.add_argument('SOURCE', type=str, help='source file.')
    parser.add_argument('TARGET', type=str, help='target file.')
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='the number of sentence paris in each mini-patch.')
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='number of sweeps over the dataset to train.')
    parser.add_argument('-s', '--save', type=str,
                        help='save a snapshot of the training.')
    parser.add_argument('-d', '--device', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('-w', '--w2v_dim', type=int, default=100,
                        help='the dimension of embedded vector (Word2Vec).')
    parser.add_argument('-o', '--out_dir', default='result',
                        help='directory to output the result')
    parser.add_argument('-u', '--num_units', type=int, default=100,
                        help='the number of neurons in each of LSTM layers.')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='the number of iteration to show log.')
    parser.add_argument('--snapshot-interval', type=int, default=4000,
                        help='the number of iteration to save snapshot of the model.')
    return parser.parse_args()


def load_text(file_name):
    lines = []
    vocab = {}
    words = {}
    vocab['<eos>'] = EOS
    words[EOS] = '<eos>'
    vocab['<unk>'] = UNK
    words[UNK] = '<eos>'
    with open(file_name) as f:
        for l in f:
            line = []
            for w in l.strip().split():
                if w not in vocab:
                    vocab[w] = len(vocab)
                    words[vocab[w]] = w
                line.append(vocab[w])
            lines.append(line)
    return lines, vocab, words


@chainer.dataset.converter()
def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        src_xp = chainer.backend.get_array_module(*batch)
        xp = device.xp
        concat = src_xp.concatenate(batch, axis=0)
        sections = list(numpy.cumsum(
            [len(x) for x in batch[:-1]], dtype=numpy.int64))
        concat_dst = device.send(concat)
        batch_dst = xp.split(concat_dst, sections)
        return batch_dst

    return {'src_seq': to_device_batch([x for x, _ in batch]),
            'dst_seq': to_device_batch([y for _, y in batch])}


def train():
    args = parse_args()
    src_data, src_vocab, src_words = load_text(args.SOURCE)
    dst_data, dst_vocab, dst_words = load_text(args.TARGET)
    train_data = [(s, t) for s, t in six.moves.zip(src_data, dst_data)]

    model = mt.SimpleMT(src_vocab, dst_vocab, args.num_units)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    device = chainer.get_device(args.device)
    model.to_device(device)
    device.use()

    # Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batch_size)

    # Setup updater and trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=convert, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out_dir)
    trainer.extend(extensions.LogReport(trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/perp', 'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))

    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.iteration}'),
        trigger=(args.snapshot_interval, 'iteration'))

    trainer.run()

    if args.save is not None:
        # chainer.serializers.save_npz(args.save, trainer)
        chainer.serializers.save_npz(args.save, model)


if __name__ == '__main__':
    train()
