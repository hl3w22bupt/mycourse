"""P5 - 玩具语料。

刻意写成高度重复、句式固定的迷你英文语料：小模型在几百步内就能把「拼写 +
词序」学出明显的 loss 下降，适合快速验证训练闭环。真实练习时可换成
tiny shakespeare 等更大的语料，接口不变。
"""

TOY_CORPUS = (
    "the little engine could run. the little engine could think. "
    "the little engine could read. the little engine could write. "
    "a tiny model can learn. a tiny model can grow. "
    "a tiny model can run. a tiny model can dream. "
    "we build the engine step by step. we build the model day by day. "
    "attention is all you need. attention is what we study. "
    "tokens flow into embeddings. embeddings flow into attention. "
    "attention flows into the cache. the cache flows into sampling. "
    "sampling gives us the next token. the next token builds the sentence. "
) * 12
