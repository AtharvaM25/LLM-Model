import os
import kagglehub
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
block_size = 128
batch_size = 32
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 50
dropout = .2
n_embd = 384
n_head = 4
n_layer = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Download latest version
path = kagglehub.dataset_download("moxxis/harry-potter-lstm")

print("Path to dataset files:", path)

# Construct the full path to the text file
file_path = os.path.join(path, 'Harry_Potter_all_books_preprocessed.txt')

# Open the file using the full path
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
# print(chars)
vocab_size = len(chars)
# print(vocab_size)


# tokeninzing the characters
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [string_to_int[ch] for ch in s]
def decode(i): return "".join(int_to_string[x] for x in i)


data = torch.tensor(encode(text), dtype=torch.long)
# print(data[:100])


n = int(.8*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


x, y = get_batch("train")
# print("inputs:")
# print(x.shape)
# print(x)
# print("targets:")
# print(y)


x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print("when input is ",context, " target is ",target)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd,)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()  # Moved super().__init__() here
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd)  # Corrected typo here
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):  # Corrected nn.linear to nn.Linear
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape  # Define T here
        # logits=self.token_embedding_table(index)
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(
                probs, num_samples=1)  # Changed num_samples to 1
            index = torch.cat((index, index_next), dim=-1)
        return index


model = GPTLanguageModel(vocab_size)  # Added n_head here
# context=torch.zeros((1,1), dtype=torch.long)
# generated_chars = decode(model.generate(context,max_new_tokens=500)[0].tolist())
# print(generated_chars)
m = model.to(device)


# with open("model-01.pkl", 'rb') as f:
# model = pickle.load(f)
# print("loaded sucessfully")


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"steps: {iter}, train loss: {losses['train']:.3f}, val loss:{losses['val']:.3f}")

    xb, yb = get_batch("train")
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open("model-01.pkl", 'wb') as f:
    pickle.dump(model, f)
print("saved")
