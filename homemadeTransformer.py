import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

WEIGHTS_PATH = "homemade_model_weights"
load_model = False
load_iter = None
# hyperparameters
train_size = 0.7 # percentage of data used for training
batch_size = 4
block_size = 8
n_embd = 32 # 192
n_heads = 4
n_layers = 2
dropout = 0.1
b1 = 0.9
b2 = 0.98
epsilon = 10e-9
warmup_steps = 4000
max_iters = 10000
eval_iters = 100
eval_interval = max_iters // 5
lossList = []








# data processing - access the essay data
with open('essays.txt', 'r', encoding = 'utf-8') as file:
  text = file.read()

# sort vocab
chars = sorted(list(set(text)))
n_vocab = len(chars)
vocab_size = n_vocab # alias b/c sometimes i mix up the names

# create encoding/decoding functions
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# clean up essays

pattern = r'\n\d+\n'

essays = re.split(pattern, text)

essays = [essay.strip() for essay in essays]

essays = '\n'.join(essays)
essays = essays[2:] # minor fix

essays = torch.tensor(encode(essays), dtype = torch.long)

# train / val split
size = int(train_size*len(essays))
train_data = essays[:size]
val_data = essays[size:]

# get random minibatch function
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
  return x, y


# function to estimate total model loss on dataset
@torch.no_grad
def estimate_loss(model):

  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x, y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# ----- ACTUAL TRANSFORMER HERE -----

# MLP
class FeedForward(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

# Attention Head
class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.k = nn.Linear(n_embd, head_size, bias = False)
    self.q = nn.Linear(n_embd, head_size, bias = False)
    self.v = nn.Linear(n_embd, head_size, bias = False)
    self.head_size = head_size
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    key = self.k(x) 
    query = self.q(x)
    value = self.v(x)
    product = query @ key.transpose(-2, -1)
    scaled = product * self.head_size ** -0.5
    masked = scaled.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weights = F.softmax(masked, dim = -1)
    weights = self.dropout(weights)
    output = weights @ value
    return output


# Multihead Attention Mechanism
class MultiHeadAttention(nn.Module):

  def __init__(self, n_heads):
    super().__init__()
    self.heads = [Head(n_embd // n_heads) for h in range(n_heads)]
    self.out = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = torch.cat([head(x) for head in self.heads], -1)
    return self.dropout(self.out(x))
  
# Layer Normalization implementation
class LayerNorm(nn.Module):

  def __init__(self, dim = n_embd, eps = 1e-5):
    super().__init__()
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def forward(self, x):
    xmean = x.mean(dim = -1, keepdim = True)
    y = x + self.eps
    xvar = y.var(dim = -1, keepdim = True)
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    out = self.gamma * xhat + self.beta
    return out

  def parameters(self):
    return [self.gamma, self.beta]

# Transformer Block
class TransformerBlock(nn.Module):

  def __init__(self, n_heads, n_embd):
    super().__init__()
    self.multihead = MultiHeadAttention(n_heads)
    self.ln1 = LayerNorm(n_embd) # apparently it's more common nowadays to do layernorm first, but i'm just gonna replicate the paper
    self.ffwd = FeedForward(n_embd)
    self.ln2 = LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.ln1(self.multihead(x))
    x = x + self.ln2(self.ffwd(x))
    return x

# Transformer
class Transformer(nn.Module):

  def __init__(self, n_heads, n_embd, n_layers, block_size):
    super().__init__()
    self.embedding = nn.Embedding(n_vocab, n_embd) # in paper, same weight matrix shared btwn both embedding layers & projection layer,
    # but in embedding layer, weights are multiplied by d_model ** 0.5. not sure how to implement that b/c shapes are swapped
    self.posembedding = nn.Embedding(block_size, n_embd)
    self.dropout = nn.Dropout(dropout)
    self.blocks = nn.Sequential(*[TransformerBlock(n_heads, n_embd) for l in range(n_layers)])
    self.ln = LayerNorm(n_embd)
    self.projection = nn.Linear(n_embd, n_vocab)



  def forward(self, x, targets = None):
    B, T = x.shape
    tok_embed = self.embedding(x)
    posembed = self.posembedding(torch.arange(T))
    x = tok_embed + posembed
    x = self.dropout(x)
    x = self.blocks(x)
    # karpathy includes a layer normalization here, but i didn't see it in the paper
    x = self.ln(x)
    logits = self.projection(x)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # pytorch's dogma dictates that logits & targets must be shaped in a specific way for loss evaluation
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_crop = idx[:, -block_size:]
      logits, loss = self(idx_crop)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim = -1)
      new_idx = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, new_idx), dim = 1)
    return idx

# implementation of learning schedule in original paper
class Scheduler():

  def __init__(self, optimizer, warmup_steps, n_embd):
    self.optimizer = optimizer
    self.warmup_steps = warmup_steps
    self.n_embd = n_embd
    self.steps = 0

  def step(self):
    self.steps += 1
    lrate = (self.n_embd ** -0.5) * min(self.steps ** -0.5, self.steps * (self.warmup_steps ** -1.5))
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lrate
    self.optimizer.step()
  
# model and optimizer setup
model = Transformer(n_heads, n_embd, n_layers, block_size)
optimizer = torch.optim.Adam(model.parameters(), betas = (b1, b2), eps = epsilon)
scheduler = Scheduler(optimizer, warmup_steps, n_embd)
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


# training
print("Training now starting")
for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses = estimate_loss(model)
    print(f"iter {iter}: train loss {losses['train']}, val_loss {losses['val']}")
    try:
      save_weights(iter)
      print(f'Iteration {iter} saved successfully')
    except:
      print(f'Iteration {iter} failed to save')
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  scheduler.optimizer.zero_grad()
  loss.backward()
  scheduler.step()
  lossList.append(loss.item())

print(loss.item())
save_weights(max_iters)
print(f'Iteration {max_iters} saved successfully.\nTraining has terminated')

plt.plot(lossList)