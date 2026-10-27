"""GPT-2 自回归采样（加入中文学习注释，计算逻辑保持官方原样）。

读这个文件时抓住一条主线：

    prompt 整体前向一次，建立各层 KV cache
        -> 采样一个新 token
        -> 以后每步只输入刚生成的一个 token，并复用历史 cache

这里实现的是“推理/生成”，不是训练。训练时可以一次并行输入整段序列；生成时
不知道未来 token，只能循环逐个产生。

TensorFlow 1.x 的 tf.while_loop 是在计算图内部循环，不是普通 Python while。
循环状态包含 past（cache）、prev（本步输入 token）和 output（累计生成结果）。
"""

import tensorflow as tf

import model

def top_k_logits(logits, k):
    """每个样本只保留分数最高的 k 个 token，其余位置设为近似负无穷。

    logits: [B,V]，是未经过 softmax 的分数。
    k=1 等价于只允许当前最高分 token；k=0 表示不做截断。
    """
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        # tf.nn.top_k 返回每行最大的 k 个值（以及索引，这里不需要索引）。
        values, _ = tf.nn.top_k(logits, k=k)
        # 排在第 k 位的分数就是保留阈值，shape 从 [B] 扩为 [B,1] 以便广播。
        min_values = values[:, -1, tf.newaxis]
        # tf.where(condition, a, b) 对每个位置选择 a 或 b。
        # 小于阈值的位置变成 -1e10，softmax 后概率近似为 0。
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    # tf.cond 是计算图中的条件分支，大致对应运行时的 if/else。
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling：保留累计概率约为 p 的高概率 token 集合。

    top-k 固定保留 k 个；top-p 保留多少个取决于当前概率分布。
    分布很集中时可能只保留少数 token，分布平坦时会保留更多 token。
    """
    batch, _ = logits.shape.as_list()
    # 先按分数从高到低排列。排序前后 softmax 概率的相对顺序不变。
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    # softmax 得到概率，再从最高概率开始累加。
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        # 对每个 batch 找到核采样集合边界在排序后序列中的位置。
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    # 取边界位置的 logit 作为阈值，再回到原始词表顺序进行过滤。
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    """自回归生成 length 个新 token，并返回“上下文 + 新 token”。

    两种启动方式必须二选一：
      start_token: 无条件生成，从一个统一的特殊 token 开始。
      context    : 条件生成，输入用户 prompt 对应的 token ids，[B,Tprompt]。

    temperature、top_k、top_p 只影响如何从下一 token 的 logits 中采样，
    不会改变 GPT-2 模型本身。
    """
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        # tf.fill([B,1], id) 创建 B 份相同的单 token 起始上下文。
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        """执行一次模型前向。

        第一次 tokens 是完整 prompt；之后 tokens 只有刚生成的一个 token。
        past 是此前累计的所有层 KV cache。
        """
        # reuse=tf.AUTO_REUSE 表示后续循环继续复用第一次创建的模型参数，
        # 不能每生成一个 token 就创建一套新权重。
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        # logits: [B,T,V]。切到 n_vocab 是为了只保留有效词表范围。
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        # presents 只包含本次 tokens 新产生的 K/V，而不是完整历史 cache。
        presents = lm_output['present']
        # set_shape 不计算新数据，只向 TensorFlow 补充静态 shape 信息，
        # 让后面的 tf.while_loop 知道 cache 各维的结构；sequence 维保持动态。
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            """生成一个 token；它也是 tf.while_loop 每轮执行的循环体。

            past  : 进入本轮前累计的 KV cache。
            prev  : 本轮真正送入模型的 token；首轮是 prompt，之后长度恒为 1。
            output: prompt 与此前所有生成 token 的完整记录，仅用于最终返回。
            """
            next_outputs = step(hparams, prev, past=past)
            # 模型可能一次返回多个位置，但预测下一个 token 只需要最后位置 logits。
            # temperature 的含义：
            #   <1 放大分数差距，分布更尖锐、更确定；
            #   >1 缩小分数差距，分布更平坦、更随机。
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            # tf.multinomial 直接把 logits 当作类别分布的未归一化对数权重来采样，
            # 内部会完成归一化，所以这里不需要先手动 softmax。
            # samples shape 为 [B,1]，不是 argmax，因此输出可以有随机性。
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                # model.py 返回的 present 只有本轮新 KV：
                # 首轮没有 past，直接用 prompt 的 presents；后续沿 token 维（-2）
                # 把旧 cache 与本轮一个 token 的 presents 拼起来。
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                # 下一轮 prev 只保留刚生成的一个 token。这是 KV cache 能省计算的关键。
                samples,
                # output 保存完整 token 序列，但不会作为下一轮模型输入重新计算。
                tf.concat([output, samples], axis=1)
            ]

        # 重点：循环外先执行一次 body。
        # 这一次把完整 context/prompt 送入模型，建立 prompt 的 cache，并生成第一个新 token。
        past, prev, output = body(None, context, context)

        def cond(*args):
            # 终止不是由 cond 控制，而是由下面的 maximum_iterations 控制。
            # 这个官方版本不会遇到 EOS 就提前停止。
            return True

        # 还需生成 length-1 个 token，因为循环前已经生成了第一个。
        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                # cache 的 token 长度、prev 长度、output 长度会随循环变化，
                # 因此相应维度写成 None，其他维度保持固定。
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            # 这是推理采样，不保留循环的反向传播图，可降低内存开销。
            back_prop=False,
        )

        # 返回 [B, Tprompt+length]。调用方如果只想要新生成部分，
        # 会再切掉前面的 context（interactive_conditional_samples.py 就这样做）。
        return tokens
