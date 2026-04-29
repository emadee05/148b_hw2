from typing import Any


from itertools import count
import regex as re 
from collections import Counter

def get_pair_counts(seq_counts):
    pair_counts = Counter()
    for seq,count in seq_counts.items():
        for i in range(len(seq)-1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair]+=count
    return pair_counts

def merge_sequence(seq, pair):
    merged = []
    i=0
    new_token = pair[0] + pair[1]
    while i<len(seq):
        if i<len(seq)-1 and (seq[i], seq[i+1]) == pair:
            merged.append(new_token)
            i+=2
        else:
            merged.append(seq[i])
            i+=1
    return tuple(merged)
    

def train_bpe(input_path, vocab_size, special_tokens):
    '''
    given path to input text file, trains byte-level BPE tokenizer 

    return vocabulary and merges 
    vocab: dict[int, bytes] (tokenizer vocabulary, mapping from int (token ID 
    in the vocabulary) to bytes (token bytes)
    merges: list[tuple[bytes, bytes]] (BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
    representing that <token1> was merged with <token2>.
    Merges are ordered by order of creation.)
    '''
    with open(input_path, 'r') as f:
        text = f.read()
    
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        chunks = re.split(split_pattern, text) 
    else:
        chunks = [text]
    
    pretoken_ct = Counter[Any]()
    for chunk in chunks: 
        # loops over every substring in chunk that matches regex PAT 
        # scans text, finds all matches of the pattern, returns as match objects
        for match in re.finditer(PAT, chunk):
            # gets the string and encodes it to bytes 
            pretok = match.group(0)
            # adds to counter
            pretoken_ct[pretok.encode('utf-8')] += 1
    
    # converts each pre-token to a sequence of single-byte tokens 
    seq_counts = Counter()
    for pretok, count in pretoken_ct.items():
        # turns into a sequence of single-byte tokens (tuple of bytes) and adds to counter
        seq = tuple(bytes([b]) for b in pretok)
        seq_counts[seq]+=count

    # solves for merges until vocab size is reached
    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        pair_counts = get_pair_counts(seq_counts)
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        newtok = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = newtok
        
        new_seq_ct = Counter()
        for seq, count in seq_counts.items():
            new_seq = merge_sequence(seq, best_pair)
            new_seq_ct[new_seq]+=count
        seq_counts = new_seq_ct
    
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    return vocab, merges

            
