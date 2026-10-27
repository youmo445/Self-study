"""GPT-2 的核心模型定义（加入中文学习注释，计算逻辑保持官方原样）。

建议不要机械地从第一行读到最后一行，而按下面的数据流阅读：

    model -> block -> attn / mlp -> norm / conv1d 等基础函数

读完时应该能回答四个问题：
1. GPT-2 为什么是 decoder-only Transformer？
2. 一个 block 中，LayerNorm、Attention、MLP 和残差分别放在哪里？
3. past/present 如何组成 KV cache？
4. token embedding 为什么还能充当最终输出层的权重？

这份代码使用 TensorFlow 1.x。常见写法与 PyTorch 的大致对应关系：

    tf.get_variable(...)       ~= torch.nn.Parameter / nn.Linear 中的参数
    tf.variable_scope(name)    ~= 给参数划分有层次的名字，并控制参数复用
    tf.gather(weight, ids)     ~= F.embedding(ids, weight)
    tf.concat                  ~= torch.cat
    tf.stack / tf.unstack      ~= torch.stack / torch.unbind
    tf.transpose               ~= tensor.permute
    tf.reshape                 ~= tensor.reshape

这里只定义前向传播和 KV cache，没有训练数据、交叉熵和优化器代码。
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    """GPT-2 small 的默认结构参数。

    n_vocab 在这里是占位值；实际运行时会由 models/<型号>/hparams.json 覆盖。
    n_ctx   : 最大上下文长度（位置向量最多有多少个位置）。
    n_embd  : 隐藏维度 D，也是 token/position embedding 的维度。
    n_head  : 注意力头数 H，因此每头维度 Dh = D/H = 64。
    n_layer : Transformer block 数量 L。
    """
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """同时处理 TensorFlow 1.x 的静态 shape 和运行时动态 shape。

    x.shape.as_list() 是建图时已知的形状；未知维度会显示为 None。
    tf.shape(x) 是运行计算图时才得到的真实形状。
    这个函数优先使用静态整数，遇到 None 才使用动态 Tensor。

    PyTorch 中一般直接写 list(x.shape)，不需要做这层区分。
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    # 先减最大值不会改变 softmax 的数学结果，因为分子、分母同时乘了
    # exp(-max(x))；但它能避免 exp(很大的数) 溢出。这正是数值稳定版 softmax。
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    # GELU 的 tanh 近似式。GPT-2 的 MLP 使用 GELU，而原始 Transformer 使用 ReLU。
    # PyTorch 可对应 F.gelu(x, approximate='tanh')。
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """手写 LayerNorm：在最后一个维度 D 上标准化，再做可学习仿射变换。

    若 x 为 [B, T, D]，每个 token 的 D 个特征独立做一次 LayerNorm。
    这里不是 BatchNorm，不会混合不同样本或不同 token 的统计量。
    """
    with tf.variable_scope(scope):
        # .value 取 TensorFlow 1.x 的静态维度；g、b 都是长度为 D 的可训练参数。
        # 分别对应 PyTorch nn.LayerNorm(D) 中的 weight 和 bias。
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)  # 均值，shape [B,T,1]
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)  # 方差
        # tf.rsqrt(z) = 1/sqrt(z)。epsilon 防止方差为 0 时除零。
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """把最后一维 D 拆成 [n, D/n]，用于把隐藏维拆成多个头。

    例如 [B,T,768] -> [B,T,12,64]。这里只 reshape，尚未交换维度。
    """
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """split_states 的逆操作：把最后两维重新合并。"""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    """名字叫 conv1d，实际就是逐 token 的全连接层。

    输入 x 的形状是 [..., nx]，输出是 [..., nf]。代码先把所有前导维展平，
    做一次矩阵乘法，再恢复原来的前导维。因此它等价于：

        nn.Linear(nx, nf)

    TensorFlow 权重实际参与计算时为 [nx,nf]；PyTorch Linear.weight 保存为
    [nf,nx]，迁移权重时需要转置。前面的长度 1 是旧版 Conv1D 实现遗留的布局。
    """
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.

    nd = destination/query sequence length，ns = source/key sequence length。

    无 cache 时 nd=ns=T，得到普通下三角 causal mask：
        1 0 0
        1 1 0
        1 1 1

    有 cache 且本次只输入一个新 token 时，nd=1、ns=past_len+1，得到全 1：
        1 1 1 ... 1
    因为这个最新 token 可以看见所有历史 token 和自己。这里必须从右下角对齐，
    不能简单截取普通下三角矩阵的第一行。
    """
    # i 的 shape 是 [nd,1]，j 的 shape 是 [ns]；比较时利用广播得到 [nd,ns]。
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    """一个 GPT-2 causal multi-head self-attention 层。

    x       : [B,T,D]，当前这次调用输入的 token 隐状态。
    past    : 单层历史 cache，[B,2,H,Tpast,Dh]，其中维度 2 分别是 K、V。
    返回值 a: [B,T,D]；present 是本次输入新产生的 K、V。

    重点：Q 不需要缓存，因为未来不会再次查询旧 token；K、V 会被未来 token
    反复使用，所以缓存 K、V。
    """
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        # [B,T,D] -> [B,T,H,Dh] -> [B,H,T,Dh]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        # [B,H,T,Dh] -> [B,T,H,Dh] -> [B,T,D]
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        # w 的最后两维是 [Tq,Tk]。mask reshape 成 [1,1,Tq,Tk] 后，
        # 会自动广播到所有 batch 和所有 head。
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        # 允许的位置保留分数；禁止的位置减去 1e10，softmax 后概率近似为 0。
        # 等价于 PyTorch 的 scores.masked_fill(~mask, -inf)。
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        # q:[B,H,Tq,Dh]，k:[B,H,Tk,Dh]
        # q @ k^T -> 注意力分数 [B,H,Tq,Tk]
        w = tf.matmul(q, k, transpose_b=True)
        # 除以 sqrt(Dh)，防止维度增大时点积绝对值过大、softmax 过度饱和。
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)          # 在 Tk 维归一化，每行概率和为 1
        a = tf.matmul(w, v)     # [B,H,Tq,Tk] @ [B,H,Tk,Dh] -> [B,H,Tq,Dh]
        return a

    with tf.variable_scope(scope):
        # 用一个大 Linear(D,3D) 同时生成 QKV，再沿最后一维等分成三份。
        # 数学上等价于分别使用 q_proj、k_proj、v_proj，融合后通常更高效。
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        # present 只保存“本次输入”产生的新 K/V，不包含传入的历史 past。
        # shape: [B,2,H,T,Dh]。
        present = tf.stack([k, v], axis=1)
        if past is not None:
            # pk、pv: [B,H,Tpast,Dh]。沿 token/sequence 维拼接历史与当前 K/V。
            # 拼接后 k、v 为 [B,H,Tpast+T,Dh]，而 q 仍只对应当前 T 个 token。
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        # 多头结果合并后再经过输出投影 W_O。
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    """GPT-2 block 中逐 token、彼此独立的前馈网络。

    block() 传入 n_state=4*D，所以形状变化是：
        [B,T,D] -> Linear(D,4D) -> GELU -> Linear(4D,D) -> [B,T,D]

    它不会在 token 维度交换信息；不同 token 的信息交换只发生在 attention 中。
    """
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    """一个完整 GPT-2 Transformer block，采用 Pre-LayerNorm。

        x -> LN -> causal self-attention -> +x
          -> LN -> MLP                   -> +x

    GPT-2 是 decoder-only，因此这里没有 encoder-decoder cross-attention。
    """
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        # 注意力子层：先 LN，再 attention，最后残差相加。
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        # MLP 子层同样是 Pre-LN；隐藏层宽度固定扩张到 4D。
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    # 全模型 cache 比单层 cache 多一个 layer 维：
    # [B,L,2,H,Tpast,Dh]。None 表示 TensorFlow 建图时允许该维动态变化。
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """在最前面增加 batch 维，并把同一份位置序列复制给每个样本。"""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    # 无 cache 时位置是 [0,1,...,T-1]；有 cache 时从 past_length 开始。
    # 例如已有 5 个 token，新 token 的位置编号就是 5，而不是重新从 0 开始。
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=False):
    """完整 GPT-2 前向传播。

    X    : token id，shape [B,T]。
    past : 所有层的 KV cache，[B,L,2,H,Tpast,Dh]，首次调用为 None。

    返回：
        logits  [B,T,V]：每个位置对下一个 token 的未归一化分数。
        present [B,L,2,H,T,Dh]：本次输入在所有层中新产生的 K/V。

    GPT-2 没有 encoder，也没有 cross-attention。token 经过 embedding 后直接进入
    多层 causal Transformer block，所以称为 decoder-only language model。
    """
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        # wpe: learned position embedding，[最大上下文长度,D]。
        # wte: word/token embedding，[词表大小,D]。
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        # cache 的倒数第二维就是历史 token 数量。
        past_length = 0 if past is None else tf.shape(past)[-2]
        # tf.gather 相当于查 embedding 表。GPT-2 直接把 token embedding 与对应的
        # learned position embedding 相加，没有原始 Transformer 的正弦位置编码。
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        # 在 layer 维拆开总 cache，让第 l 层只接收第 l 层自己的历史 K/V。
        # 不同层的 cache 绝不相同，因为每层输入的隐藏状态不同。
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        # 把各层本次产生的 cache 沿 layer 维堆叠起来。
        results['present'] = tf.stack(presents, axis=1)
        # 所有 block 结束后还有一次最终 LayerNorm，这是 GPT-2 结构的一部分。
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        # 把 [B,T,D] 展平只是为了做一次矩阵乘法；随后再恢复 [B,T,V]。
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        # 重点：这里没有创建新的 lm_head 权重，而是复用输入 embedding wte 的转置。
        # 即 logits = h @ wte^T，这叫 weight tying，能减少 V*D 个参数。
        # logits 还是 raw scores；softmax/cross_entropy 应在训练或采样代码中完成。
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
