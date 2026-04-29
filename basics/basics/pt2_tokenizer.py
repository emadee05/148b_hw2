# from collections import defaultdict

# import regex as re

# from eecs148b_hw1.bpe_constants import PAT

# class Tokenizer:
#     def __init__(self, vocab, merges, special_tokens=None):
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens = special_tokens or []
#         self.token_to_id = {token_byte: id for id, token_byte in self.vocab.items()}
#         self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    
#     @classmethod
#     def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
#         import pickle
        
#         with open(vocab_filepath, "rb") as f:
#             vocab = pickle.load(f)
        
#         with open(merges_filepath, "rb") as f:
#             merges = pickle.load(f)
        
#         return cls(vocab, merges, special_tokens)

#     def encode(self, text):
#         '''
#         given a string of text, return a list of token IDs
#         needs bytes -> id
#         '''
#         ids = []
#         if self.special_tokens:
#             split_pattern = "(" + "|".join(
#                 re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
#             ) + ")"
#             chunks = re.split(split_pattern, text)
#         else:
#             chunks = [text]
#         for chunk in chunks:
#             if chunk in self.special_tokens:
#                 ids.append(self.token_to_id[chunk.encode("utf-8")])
#                 continue
            
#             for match in re.finditer(PAT, chunk):
#                 pretok = match.group(0)
#                 seq = [bytes([b]) for b in pretok.encode("utf-8")]
                
#                 # Build a lookup for fast pair checking
                
#                 while True:
#                     # Find the lowest-rank merge present in current seq
#                     best = None
#                     best_rank = float('inf')
#                     for i in range(len(seq) - 1):
#                         pair = (seq[i], seq[i+1])
#                         rank = self.merge_rank.get(pair)
#                         if rank is not None and rank < best_rank:
#                             best_rank = rank
#                             best = (i, pair)
                    
#                     if best is None:
#                         break  # no more applicable merges
                    
#                     # Apply just that one merge
#                     i, pair = best
#                     new_token = pair[0] + pair[1]
#                     seq = seq[:i] + [new_token] + seq[i+2:]
                
#                 ids.extend(self.token_to_id[token] for token in seq)
#                     # convert to ids
#         return ids


#     def encode_iterable(self, iterable):
#         '''
#         given an iterable of strings (python file handle) return a generator that lazily yields token IDs
#         supports tokenizing large files without loading them into memory all at once
#         '''
#         for line in iterable: 
#             for token_id in self.encode(line):
#                 yield token_id


#     def decode(self, ids):
#         '''
#         decode a sequence of token IDs into the text 
#         '''
#         all_bytes = b"".join(self.vocab[token_id] for token_id in ids)
#         return all_bytes.decode("utf-8", errors="replace")

from collections import defaultdict
import regex as re

from eecs148b_hw1.bpe_constants import PAT


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token_to_id = {token_byte: i for i, token_byte in self.vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

        self.pat = re.compile(PAT)

        if self.special_tokens:
            split_pattern = "(" + "|".join(
                re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
            self.special_split_re = re.compile(split_pattern)
        else:
            self.special_split_re = None

        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import pickle
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _encode_pretok(self, pretok: str) -> list[int]:
        if pretok in self.cache:
            return self.cache[pretok]

        seq = [bytes([b]) for b in pretok.encode("utf-8")]

        while True:
            best = None
            best_rank = float("inf")

            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best = (i, pair)

            if best is None:
                break

            i, pair = best
            seq = seq[:i] + [pair[0] + pair[1]] + seq[i + 2:]

        ids = [self.token_to_id[token] for token in seq]
        self.cache[pretok] = ids
        return ids

    def encode(self, text):
        ids = []

        if self.special_split_re is not None:
            chunks = self.special_split_re.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
                continue

            for match in self.pat.finditer(chunk):
                pretok = match.group(0)
                ids.extend(self._encode_pretok(pretok))

        return ids

    def encode_iterable(self, iterable):
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids):
        all_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return all_bytes.decode("utf-8", errors="replace")