import regex as re
from typing import List, Tuple

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BasicTokenizer:

    def __init__(self):
        self.merges = {}        # (0, 1) => 256
        self.vocab = {          # 1 => \x00, 2 => \x01
            i: bytes([i])       # 用列表也可以，但主流是用字典
            for i in range(256)
        }

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size > 256

        idxs = list(text.encode('utf-8'))
        for new_idx in range(256, vocab_size):
            # find top pair
            pairs = {}          # (0, 1) => count
            for idx1, idx2 in zip(idxs, idxs[1:]):
                if (idx1, idx2) not in pairs:
                    pairs[(idx1, idx2)] = 0
                pairs[(idx1, idx2)] += 1
            top_pair = max(pairs, key=pairs.get)
            self.merges[top_pair] = new_idx
            tk1 = self.vocab[top_pair[0]]  # `idx` 是 token 的索引
            tk2 = self.vocab[top_pair[1]]  # `tk` 是 token 的实际字节
            tk3 = tk1 + tk2
            self.vocab[new_idx] = tk3
            if verbose:
                print(
                    f'Tokens {tk1} and {tk2} {top_pair} are merged into token {tk3} ({new_idx})'
                )

            # merge
            idxs = self._merge(idxs, top_pair, new_idx)

    def encode(self, text: str):
        idxs = list(text.encode('utf-8'))
        for pair, target in self.merges.items():
            idxs = self._merge(idxs, pair, target)
        return idxs

    def decode(self, idxs: List[int]):
        byte_string = b''.join([self.vocab[i] for i in idxs])
        return byte_string.decode('utf-8', errors='replace')

    def save(self, prefix: str = 'tokenizer'):
        model_file = prefix + '.model'  # .model 文件记录合并，用于后续加载
        vocab_file = prefix + '.vocab'  # .vocab 文件记录词汇表，仅用于用户参考

        with open(model_file, 'w') as f:
            for pair, target in self.merges.items():
                f.write(f'{pair[0]}+{pair[1]}={target}\n')

        with open(vocab_file, 'w') as f:
            for idx, tk in self.vocab.items():
                tk_unicode = tk.decode('utf-8', errors='replace')
                f.write(f'[{idx}][{tk_unicode}]\n')

    def load(self, prefix: str = 'tokenizer'):
        model_file = prefix + '.model'

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        with open(model_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                pair, target = line.split('=')
                idx1, idx2 = pair.split('+')
                idx1 = int(idx1)
                idx2 = int(idx2)
                target = int(target)
                self.merges[(idx1, idx2)] = target  # 加载合并
                tk1 = self.vocab[idx1]
                tk2 = self.vocab[idx2]
                tk3 = tk1 + tk2
                self.vocab[target] = tk3            # 并重构词汇表

    def _merge(self, idxs: List[int], pair: Tuple[int], target: int):
        idx1, idx2 = pair
        i = 0
        l = len(idxs)
        new_idxs = []
        while i < l:
            if idxs[i] == idx1 and i < l - 1 and idxs[i + 1] == idx2:
                new_idxs.append(target)
                i += 2
            else:
                new_idxs.append(idxs[i])
                i += 1
        return new_idxs


class RegexTokenizer(BasicTokenizer):

    def __init__(self):
        super().__init__()
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size > 256

        parts = self._split_by_pattern(text)
        parts_idxs = [list(p.encode('utf-8')) for p in parts]
        for new_idx in range(256, vocab_size):
            # find top pair
            pairs = {}  # (0, 1) => count
            for idxs in parts_idxs:
                for idx1, idx2 in zip(idxs, idxs[1:]):
                    if (idx1, idx2) not in pairs:
                        pairs[(idx1, idx2)] = 0
                    pairs[(idx1, idx2)] += 1

            top_pair = max(pairs, key=pairs.get)
            self.merges[top_pair] = new_idx
            tk1 = self.vocab[top_pair[0]]
            tk2 = self.vocab[top_pair[1]]
            tk3 = tk1 + tk2
            self.vocab[new_idx] = tk3
            if verbose:
                print(
                    f'Tokens {tk1} and {tk2} {top_pair} are merged into token {tk3} ({new_idx})'
                )

            # merge
            new_parts_idxs = []
            for idxs in parts_idxs:
                new_parts_idxs.append(self._merge(idxs, top_pair, new_idx))
            parts_idxs = new_parts_idxs

    def encode(self, text: str):
        parts = self._split_by_pattern(text)  # 将原文本按照模式切分，分别编码再拼接
        results = [super(RegexTokenizer, self).encode(p) for p in parts]
        return [idx for result in results for idx in result]

    def _split_by_pattern(self, text: str):
        res = re.findall(self.compiled_pattern, text)  # 将原文本按照模式划分
        return res


# 增加对特殊 token 的处理请参阅 https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
# 简单地说，就是向词汇表手动添加特殊 token，预处理文本时率先正则匹配特殊 token 并划分

# 进一步地，可以在 RegexTokenizer 的基础上实现 GPT4Tokenizer，但需要
# 1. 根据 tiktoken 模块提供的 gpt4 tokenizer 词汇表重建合并（逆向工程）
# 2. 重新排列 256 个单字节 token
# 这些操作是为了还原 gpt4 tokenizer 而引入的特殊操作，比较 tricky
# 请参阅 https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py


with open('./taylorswift.txt') as file:
    text = file.read()
tokenizer = RegexTokenizer()
tokenizer.train(text, 500, True)

tokenizer.save()
tokenizer.load()
print(tokenizer.encode('Hello, Taylor Swift!'))
print(tokenizer.decode(tokenizer.encode('Hello, Taylor Swift!')))
