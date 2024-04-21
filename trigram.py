import torch
import torch.nn.functional as F

NUM_TRAINING_ITERATIONS = 100
LEARNING_RATE = 1
NUM_SAMPLES = 5

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# create the training set
xs, ys, zs = [], [], []
num = 0

for w in words:
    chs = ['.', '.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append(ix1)
        ys.append(ix2)
        zs.append(ix3)
        num += 1

xs = torch.tensor(xs)
ys = torch.tensor(ys)
zs = torch.tensor(zs)

# initialize the network
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27**2, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(NUM_TRAINING_ITERATIONS):

    # forward pass
    xenc = F.one_hot(xs * 27 + ys, num_classes=27**2).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), zs].log().mean()
    print(loss.item())

    # backward pass
    W.grad = None
    loss.backward()
    W.data += -LEARNING_RATE * W.grad # type: ignore

# sample from the model
for i in range(NUM_SAMPLES):
    out = []
    ix = 0

    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27**2).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix]) # type: ignore
        if ix == 0:
            break
    
    print(''.join(out))
