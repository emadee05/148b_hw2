from pt2_tokenizer import Tokenizer
from pt3_linear import TransformerLM, softmax
import torch
from pt6_decoding import generate
# load tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath="tinystories_vocab.pkl",
    merges_filepath="tinystories_merges.pkl",
    special_tokens=["<|endoftext|>"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# rebuild model with same training hyperparameters
model = TransformerLM(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    vocab_size=10000,
    context_length=256,
    num_layers=4,
    device=device,
    dtype=torch.float32,
).to(device)

# load trained checkpoint
ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Loaded checkpoint from step:", ckpt["step"])
print("Best val loss:", ckpt["best_val_loss"])

# eos id
eos_id = tokenizer.token_to_id["<|endoftext|>".encode("utf-8")]

# prompt
prompt = "Once upon a time"
prompt_ids = tokenizer.encode(prompt)
prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

# generate
generated_tokens = generate(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    eos_id=eos_id,
)

# decode
generated_text = tokenizer.decode(generated_tokens[0].tolist())
print(generated_text)