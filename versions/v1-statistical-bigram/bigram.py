import torch
import matplotlib.pyplot as plt

with open('/Users/mduraimurugan/Documents/Dev/make-more/dataset/names.txt') as f:
    names = f.read().splitlines() # .splitlines() -> splits the multiline string.

chars = ['.'] + sorted(list(set(''.join(names))))
stoi = {c:i for i,c in enumerate(chars)}  # char -> int
itos = {i:c for c,i in stoi.items()}      # int -> char  

N = torch.zeros((27,27), dtype=torch.int32) # type -> int (coz, it's just count)

for name in names:
    name = '.' + name + '.'
    for c1, c2 in zip(name, name[1:]):
        x1 = stoi[c1]
        x2 = stoi[c2]

        N[x1][x2] += 1

# Visualization
plt.figure(figsize=(18,18))
plt.xticks(torch.arange(27))
plt.yticks(torch.arange(27))
for i in range(27):
    for j in range(27):
        plt.text(j,i,N[i,j].item(), ha='center', va='center')
plt.imshow(N)   # imshow() -> image show
# plt.show()

g = torch.Generator().manual_seed(34152562) # This makes the output deterministic 
result = []
xi = 0

N += 1 # Laplace smoothing to avoid divide by zero.
probs = N / N.sum(1, keepdims=True)

for _ in range(5):
    cur = []
    while True:
        xi = torch.multinomial(probs[xi].float(), num_samples=1, replacement=True, generator=g).item()
        cur.append(itos[xi])
        if xi == 0:
            break
    result.append(''.join(cur))

# loss calculation
loss = 0
num = 0
for name in names:
    name = '.' + name + '.'
    for c1, c2 in zip(name, name[1:]):
        x1 = stoi[c1]
        x2 = stoi[c2]
        num += 1
        loss += probs[x1][x2].log()
loss = -loss / num
print(f"loss : {loss.item()}")