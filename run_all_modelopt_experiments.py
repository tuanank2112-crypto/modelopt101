import sys, os, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print('='*70)
print(f'RUNNING REAL EXPERIMENTS ON: {device} ({dev_name})')
print('='*70)

# ==========================================
# 1. DATA PREPARATION (MNIST)
# ==========================================
mean, std = (0.1307,), (0.3081,)
train_tf = T.Compose([T.RandomCrop(28, padding=2), T.ToTensor(), T.Normalize(mean, std)])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

data_dir = 'E:/vla.cpp-a-khanh/modelopt101/data'
train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
test_set  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class Net(nn.Module):
    def __init__(self, widths=(32, 64, 128, 128), num_classes=10):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.features = nn.Sequential(
            conv_block(1, w0), conv_block(w0, w1), nn.MaxPool2d(2),
            conv_block(w1, w2), conv_block(w2, w3), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(w3, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

def evaluate(model, loader=test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def train_model(model, epochs=2, lr=0.05, tag=''):
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    acc = evaluate(model)
    print(f'[{tag}] Epochs: {epochs} -> Test Accuracy: {acc:.2f}%')
    return model

print('\nTraining Baseline Model for Quantization and Pruning...')
base_model = train_model(Net(), epochs=2, tag='Baseline FP32')

# ==========================================
# EXERCISE 1: QUANTIZATION
# ==========================================
print('\n' + '='*50)
print('EXERCISE 1: QUANTIZATION (ACTUAL RUN)')
print('='*50)

def quantize(W, n_bits=8, symmetric=True, per_channel=False):
    qmin, qmax = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    if per_channel:
        flat = W.flatten(1)
        Wmin, Wmax = flat.min(1).values, flat.max(1).values
    else:
        Wmin, Wmax = W.min(), W.max()

    if symmetric:
        max_val = torch.max(Wmin.abs(), Wmax.abs()).clamp(min=1e-8)
        scale = max_val / qmax
        zero_point = torch.zeros_like(scale)
    else:
        scale = (Wmax - Wmin).clamp(min=1e-8) / (qmax - qmin)
        zero_point = torch.round(qmin - Wmin / scale).clamp(qmin, qmax)

    shape = [-1] + [1] * (W.dim() - 1)
    s = scale.view(shape) if per_channel else scale
    z = zero_point.view(shape) if per_channel else zero_point
    q = torch.clamp(torch.round(W / s) + z, qmin, qmax)
    return q, scale, zero_point

def dequantize(q, scale, zero_point, per_channel=False):
    if per_channel:
        shape = [-1] + [1] * (q.dim() - 1)
        scale, zero_point = scale.view(shape), zero_point.view(shape)
    return scale * (q - zero_point)

w_sample = base_model.features[0][0].weight.data.cpu()
for pc in (False, True):
    q, s, z = quantize(w_sample, n_bits=8, symmetric=True, per_channel=pc)
    w_hat = dequantize(q, s, z, per_channel=pc)
    mse = F.mse_loss(w_hat, w_sample).item()
    print(f'Weight Sample Layer 1 INT8 (per_channel={pc}): MSE = {mse:.6e}')

# Fake quantize whole model
QUANTIZABLE = (nn.Conv2d, nn.Linear)
def quantize_model(model, n_bits=8, per_channel=True):
    deq = copy.deepcopy(model).cpu().eval()
    for name, mod in deq.named_modules():
        if isinstance(mod, QUANTIZABLE):
            q, s, z = quantize(mod.weight.data, n_bits=n_bits, symmetric=True, per_channel=per_channel)
            mod.weight.data = dequantize(q, s, z, per_channel=per_channel)
    return deq

model_int8_pc = quantize_model(base_model, per_channel=True).to(device)
model_int8_pt = quantize_model(base_model, per_channel=False).to(device)

print(f'FP32 Baseline Accuracy:        {evaluate(base_model):.2f}%')
print(f'INT8 Per-tensor Accuracy:      {evaluate(model_int8_pt):.2f}%')
print(f'INT8 Per-channel Accuracy:     {evaluate(model_int8_pc):.2f}%')

# ==========================================
# EXERCISE 2: KNOWLEDGE DISTILLATION
# ==========================================
print('\n' + '='*50)
print('EXERCISE 2: KNOWLEDGE DISTILLATION (ACTUAL RUN)')
print('='*50)

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3 = 64, 128, 256
        self.features = nn.Sequential(
            conv_block(1, c1), conv_block(c1, c1), nn.MaxPool2d(2),
            conv_block(c1, c2), conv_block(c2, c2), nn.MaxPool2d(2),
            conv_block(c2, c3), conv_block(c3, c3), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, 10)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c3 = 16, 32, 32
        self.features = nn.Sequential(
            conv_block(1, c1), nn.MaxPool2d(2),
            conv_block(c1, c2), nn.MaxPool2d(2),
            conv_block(c2, c3), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, 10)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

print('Training Teacher model (Big)...')
teacher = train_model(Teacher(), epochs=2, tag='Teacher Big')
print('Training Student model from scratch (No KD)...')
student_base = train_model(Student(), epochs=2, tag='Student Baseline')

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    ce_term = F.cross_entropy(student_logits, labels)
    kd_term = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)
    return alpha * kd_term + (1.0 - alpha) * ce_term

print('Training Student model WITH Distillation...')
student_kd = Student().to(device)
opt_kd = torch.optim.SGD(student_kd.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
teacher.eval()
for ep in range(2):
    student_kd.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = student_kd(x)
        loss = distillation_loss(s_logits, t_logits, y, T=4.0, alpha=0.7)
        opt_kd.zero_grad()
        loss.backward()
        opt_kd.step()

acc_kd = evaluate(student_kd)
print(f'Student + KD Test Accuracy:    {acc_kd:.2f}% (vs Baseline: {evaluate(student_base):.2f}%)')

# ==========================================
# EXERCISE 3: PRUNING
# ==========================================
print('\n' + '='*50)
print('EXERCISE 3: PRUNING (ACTUAL RUN)')
print('='*50)

def conv_blocks(m):
    return [mod for mod in m.modules() if isinstance(mod, nn.Sequential) and isinstance(mod[0], nn.Conv2d)]

def prune_unstructured(model, ratio=0.5):
    mp = copy.deepcopy(model)
    all_w = torch.cat([blk[0].weight.data.abs().flatten() for blk in conv_blocks(mp)])
    thresh = torch.quantile(all_w, ratio)
    for blk in conv_blocks(mp):
        w = blk[0].weight.data
        w.mul_((w.abs() >= thresh).float())
    return mp, thresh.item()

pruned_unstructured, thresh = prune_unstructured(base_model, ratio=0.5)
print(f'Unstructured Pruning (50% zeroed, threshold={thresh:.4f}): Accuracy = {evaluate(pruned_unstructured):.2f}%')

# ==========================================
# EXERCISE 4: NAS
# ==========================================
print('\n' + '='*50)
print('EXERCISE 4: NAS SEARCH SPACE (ACTUAL RUN)')
print('='*50)
CHOICES = {
    'depth':  [2, 3],
    'width':  [12, 16, 24],
    'kernel': [3, 5],
}

def build_model(arch, num_classes=10):
    layers, cin = [], 1
    for i in range(arch['depth']):
        cout = min(arch['width'] * (2 ** i), 128)
        k = arch['kernel']
        layers += [
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        cin = cout
    layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(cin, num_classes)]
    return nn.Sequential(*layers)

arch1 = {'depth': 2, 'width': 16, 'kernel': 3}
arch2 = {'depth': 3, 'width': 24, 'kernel': 5}
m1 = train_model(build_model(arch1), epochs=1, tag='NAS Arch 1 (Small)')
m2 = train_model(build_model(arch2), epochs=1, tag='NAS Arch 2 (Bigger)')
print(f'Arch 1 Test Accuracy: {evaluate(m1):.2f}%')
print(f'Arch 2 Test Accuracy: {evaluate(m2):.2f}%')

print('\n' + '='*70)
print('ALL EXPERIMENTS 1-4 COMPLETED SUCCESSFULLY!')
print('='*70)
