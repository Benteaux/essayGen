import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


WEIGHTS_PATH = "homemade_moe_weights"
load_model = False
load_iter = None

# hyperparams
max_steps = 25000
eval_interval = 2000
eval_iters = 1000
batch_size = 16
block_size = 8
dropout = 0.2
n_embd = 32
n_hidden = 2 * n_embd
window_size = 3
n_heads = 4
n_groups = 2
n_experts = 2
top_k_experts = 1
n_layers = 2
b1 = 0.9
b2 = 0.98
epsilon = 1e-9
lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('essays.txt', 'r') as file:
  essays = file.read()

essays = essays[1:]

chars = sorted(list(set(essays)))
n_vocab = len(chars)
vocab_size = n_vocab # b/c i forget sometimes which name to use

itos = {i:s for i,s in enumerate(chars)}
stoi = {s:i for i,s in enumerate(chars)}
encode = lambda x: [stoi[s] for s in x]
decode = lambda x: ''.join([itos[c] for c in x])

import re

pattern = r'\n\d+\n'

essays = re.split(pattern, essays)
essays = [essay.strip() for essay in essays]
essays = '\n'.join(essays)
essays = essays[2:]

essays = torch.tensor(encode(essays), dtype = torch.long, device = device)

train_size = int(0.7 * len(essays))
train_data = essays[:train_size]
val_data = essays[train_size:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ixs = torch.randint(len(data) - block_size, (batch_size,))
  xs = torch.stack([data[ix : ix + block_size] for ix in ixs])
  ys = torch.stack([data[ix + 1 : ix + block_size + 1] for ix in ixs])
  return xs, ys

xb, yb = get_batch('train')

class SwiGLU(nn.Module):

  def __init__(self, n_hidden = n_hidden):
    super().__init__()
    self.swishLinear = nn.Linear(n_hidden, n_hidden, device = device)
    self.swishBeta = nn.Parameter(torch.randn(1), requires_grad = True).to(device)
    self.gluLinear = nn.Linear(n_hidden, n_hidden, device = device)
  def forward(self, x):
    swish_x = self.swishLinear(x)
    swish = swish_x * F.sigmoid(self.swishBeta * swish_x)
    glu = self.gluLinear(x)
    return swish * glu

class Expert(nn.Module):

  def __init__(self, n_embd = n_embd, n_hidden = 4*n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_hidden, device = device),
        SwiGLU(n_hidden),
        nn.Linear(n_hidden, n_embd, device = device),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)
  
# not used, just meant to show implementation of SWA w/o GQA
class SWA(nn.Module):

  def __init__(self, d_head, window_size = window_size):
    super().__init__()
    self.k = nn.Linear(n_embd, d_head, bias = False, device = device)
    self.q = nn.Linear(n_embd, d_head, bias = False, device = device)
    self.v = nn.Linear(n_embd, d_head, bias = False, device = device)
    self.register_buffer('future_mask', torch.tril(torch.ones(block_size, block_size, device = device)))
    self.register_buffer('past_mask', torch.tril(torch.ones(block_size, block_size, device = device), -window_size))
    self.dropout = nn.Dropout(dropout)
    self.d_head = d_head


  def forward(self, x):
    keys = self.k(x)
    B, T, C = keys.shape
    queries = self.q(x)
    values = self.v(x)
    kq = (queries @ keys.view(B, C, T)) * self.d_head ** -0.5
    kq.masked_fill(self.future_mask[:T, :t] == 0, float('-inf'))
    kq.masked_fill(self.past_mask[:T, :T] == 1, float('-inf'))
    weights = F.softmax(kq, dim = -1)
    weights = self.dropout(weights)
    out = weights @ values
    return out

# sliding window attention, but modified to fit the group query attention scheme
class GQA_SWA_Head(nn.Module):

  def __init__(self, d_head, n_q_heads, window_size = window_size):
    super().__init__()
    self.k = nn.Linear(n_embd, d_head, bias = False, device = device)
    self.q = [nn.Linear(n_embd, d_head, bias = False, device = device) for _ in range(n_q_heads)]
    self.v = nn.Linear(n_embd, d_head, bias = False, device = device)
    self.register_buffer('future_mask', torch.tril(torch.ones(block_size, block_size, device = device)))
    # create a mask to block out tokens too far in the past
    self.register_buffer('past_mask', torch.tril(torch.ones(block_size, block_size, device = device), -window_size))
    self.dropout = nn.Dropout(dropout)
    self.d_head = d_head


  def forward(self, x):
    global count
    keys = self.k(x)
    B, T, C = keys.shape
    queries = [q_head(x) for q_head in self.q]
    values = self.v(x)
    # paper didn't provide details on how to combine multiple query heads w/ 1 key head,
    # so i just added the products together
    product = torch.zeros(B, T, T, device = device)
    for query in queries:
      product += query @ keys.view(B,C,T)
    scaled = product * self.d_head ** -0.5
    scaled.masked_fill(self.future_mask[:T, :T] == 0, float('-inf'))
    # block out tokens too far in the past
    scaled = scaled.masked_fill(self.past_mask[:T, :T] == 1, float('-inf'))
    weights = F.softmax(scaled, dim = -1)
    weights = self.dropout(weights)
    out = weights @ values
    return out

class GQA(nn.Module):

  def __init__(self, n_heads, n_kv_heads = n_heads // n_groups):
    super().__init__()
    d_head = n_embd // n_heads
    n_q_heads = n_heads // n_kv_heads
    self.heads = [GQA_SWA_Head(d_head, n_q_heads) for head in range(n_heads)]
    self.projection = nn.Linear(n_embd, n_embd, device = device)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.concatenate([head(x) for head in self.heads], dim = -1)
    return self.dropout(self.projection(out))

class LayerNorm(nn.Module):

  def __init__(self, n_embd = n_embd, eps = 1e-5):
    super().__init__()
    self.gamma = torch.ones(n_embd, device = device)
    self.beta = torch.zeros(n_embd, device = device)
    self.eps = eps

  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True)
    var = x.var(dim = -1, keepdim = True)
    xhat = (x - mean) / torch.sqrt((var + self.eps))
    return (xhat * self.gamma) + self.beta

  def parameters(self):
    return [self.gamma, self.beta]
  
class Router(nn.Module):

  def __init__(self, experts, top_k_experts, n_embd = n_embd):
    super().__init__()
    self.top_k_experts = top_k_experts
    self.experts = experts
    self.gate = nn.Linear(n_embd, len(experts), device = device)

  def forward(self, tokens):
    routing_table = self.gate(tokens)
    weights, expert_ixs = torch.topk(routing_table, self.top_k_experts)
    weights = F.softmax(weights, 1)
    results = torch.zeros_like(tokens)
    for i, expert in enumerate(self.experts):
      batch_ix, tok_ix, expert_ix = torch.where(expert_ixs == i)
      if batch_ix.shape[0] > 0:
        selected_weights = weights[batch_ix, tok_ix, expert_ix].unsqueeze(-1)
        weighted_logits = selected_weights * expert(tokens[batch_ix, tok_ix])
        results[batch_ix, tok_ix] += weighted_logits
    return results

class MoeBlock(nn.Module):

  def __init__(self, n_vocab = n_vocab, n_embd = n_embd, n_heads = n_heads, n_experts = n_experts, top_k_experts = top_k_experts):
    super().__init__()
    self.GQA = GQA(n_heads)
    self.experts = nn.ModuleList([Expert(n_embd) for _ in range(n_experts)])
    self.router = Router(self.experts, top_k_experts)
    self.ln1 = LayerNorm(n_embd)
    self.ln2 = LayerNorm(n_embd)


  def forward(self, x):
    x = x + self.GQA(self.ln1(x))
    x = x + self.router(self.ln2(x))
    return x

class MixtureOfExperts(nn.Module):

  def __init__(self, block_size = block_size, n_layers = n_layers, n_vocab = n_vocab, n_embd = n_embd, n_heads = n_heads, n_experts = n_experts, top_k_experts = top_k_experts):
    super().__init__()
    self.tok_embed = nn.Embedding(n_vocab, n_embd, device = device)
    self.pos_embed = nn.Embedding(block_size, n_embd, device = device)
    self.lm_head = nn.Linear(n_embd, n_vocab, device = device)
    self.dropout = nn.Dropout(dropout)
    self.blocks = nn.Sequential(*[MoeBlock(n_vocab, n_embd, n_heads, n_experts, top_k_experts) for layer in range(n_layers)])
    self.layernorm = LayerNorm(n_embd)



  def forward(self, x, targets = None):
    B, T = x.shape
    tok_embed = self.tok_embed(x)
    pos_embed = self.pos_embed(torch.arange(T, device = device)) # not quite
    embed = tok_embed + pos_embed
    embed = self.dropout(embed)
    out = self.blocks(embed)
    out = self.layernorm(out)
    logits = self.lm_head(out)

    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    idx = idx.to(device)
    for _ in range(max_new_tokens):
      idx_crop = idx[:, -block_size:]
      logits, loss = self(idx_crop)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = -1)
      new_idx = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, new_idx), dim = 1)
    return idx

model = MixtureOfExperts()
optimizer = torch.optim.Adam(model.parameters(), betas = (b1, b2), eps = epsilon, lr = lr)
print("Model created successfully")

# save and loading weights functions
def save_weights(iteration, folder = WEIGHTS_PATH):
  if not os.path.exists(folder):
    os.makedirs(folder)
  savePath = os.path.join(folder, f'weights_{iteration}')
  torch.save(model, savePath)

def load_weights(iteration, folder = WEIGHTS_PATH):
  loadPath = os.path.join(folder, f'weights_{iteration}')
  model.load_state_dict(torch.load(loadPath))


if load_model:
  load_weights(load_iter)
  print(f"Successfully loaded model from iteration {load_iter}")



lossList = []



@torch.no_grad
def estimate_loss(model):
  out = {}
  model.eval()
  splits = ['train', 'val']
  for split in splits:
    losses = torch.zeros(eval_iters)
    for iter in range(eval_iters):
      xb, yb = get_batch(split)
      logits, loss = model(xb, yb)
      losses[iter] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out



for step in range(max_steps):
  if step % eval_interval == 0 and step != 0:
    losses = estimate_loss(model)
    print(f"Step {step}: train loss {losses['train']}, val_loss {losses['val']}")
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad
  loss.backward()
  optimizer.step()
  lossList.append(loss.item())



print(loss.item())



