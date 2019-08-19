# -*- coding: utf-8 -*-

import numpy
import chainer
import chainer.functions as F
import chainer.links as L

EOS = 0
UNK = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class SimpleMT(chainer.Chain):
    def __init__(self, src_vocab, dst_vocab, w2v_dim, n_units):
        super(SimpleMT, self).__init__()
        with self.init_scope():
            self.embed_src = L.EmbedID(len(src_vocab), w2v_dim)
            self.embed_dst = L.EmbedID(len(dst_vocab), w2v_dim)
            self.lstm = L.NStepLSTM(n_layers=1, in_size=w2v_dim, out_size=n_units, dropout=0.1)
            self.lin = L.Linear(in_size=n_units, out_size=len(dst_vocab))
        self.src_vocab = src_vocab
        self.dst_vocab = dst_vocab

    def forward(self, src_seq, dst_seq):
        eos = self.xp.array([EOS], numpy.int64)
        dst_in = [F.concat([eos, dst], axis=0) for dst in dst_seq]
        dst_out = [F.concat([dst, eos], axis=0) for dst in dst_seq]

        src_seq_emb = sequence_embed(self.embed_src, src_seq)
        dst_seq_emb = sequence_embed(self.embed_dst, dst_in)

        batch = len(src_seq)

        hx, cx, _ = self.lstm(None, None, src_seq_emb)
        _, _, out_seq = self.lstm(hx, cx, dst_seq_emb)

        concat_out_seq = F.concat(out_seq, axis=0)
        concat_dst_out = F.concat(dst_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(self.lin(concat_out_seq), concat_dst_out, reduce='no')) / batch

        chainer.report({'loss': loss}, self)
        n_words = concat_dst_out.shape[0]
        perp = self.xp.exp(loss.array * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, src_seq, max_length=100):
        batch = len(src_seq)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            src_seq_emb = sequence_embed(self.embed_src, src_seq)
            h, c, _ = self.lstm(None, None, src_seq_emb)
            dst_seq = self.xp.full(batch, EOS, numpy.int64)
            result = []
            for i in range(max_length):
                dst_seq_emb = self.embed_dst(dst_seq)
                dst_seq_emb = F.split_axis(dst_seq_emb, batch, 0)
                h, c, dst_seq = self.lstm(h, c, dst_seq_emb)
                hid_seq = F.concat(dst_seq, axis=0)
                tmp = self.lin(hid_seq)
                out_seq = self.xp.argmax(tmp.array, axis=1).astype(numpy.int64)
                result.append(out_seq)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = chainer.get_device('@numpy').send(
            self.xp.concatenate([x[None, :] for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs
