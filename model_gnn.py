import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)

        self.conv1 = SAGEConv((-1, -1), hidden_channels, normalize=True)

        self.conv2 = SAGEConv((-1, -1), hidden_channels, normalize=True)

        self.conv3 = SAGEConv((-1, -1), hidden_channels, normalize=True)
        self.conv4 = SAGEConv((-1, -1), out_channels,    normalize=False)

    def forward(self, x, edge_index):
        # Lapis 1: aggregate tetangga hop-1 → ReLU → Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.drop(x)

        # Lapis 2: aggregate tetangga hop-2 → ReLU → Dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.drop(x)

        # Lapis 3: aggregate tetangga hop-3 → ReLU → Dropout
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.drop(x)

        # Lapis 4: proyeksi ke ruang kelas (logits), tanpa aktivasi
        x = self.conv4(x, edge_index)
        return x

def latih_model(graf, hidden_channels=64, out_channels=3, dropout=0.3,
                lr=0.005, weight_decay=5e-4, epochs=200):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graf = graf.to(device)

    model = GNN(hidden_channels=hidden_channels, out_channels=out_channels, dropout=dropout)
    model = to_hetero(model, graf.metadata(), aggr='mean')

    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train():
        model.train()
        optimizer.zero_grad()
        out_dict = model(graf.x_dict, graf.edge_index_dict)
        out = out_dict['siswa']
        mask = graf['siswa'].train_mask
        loss = criterion(out[mask], graf['siswa'].y[mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate_step():
        model.eval()
        out_dict = model(graf.x_dict, graf.edge_index_dict)
        out = out_dict['siswa']

        tr_mask = graf['siswa'].train_mask
        train_acc = out[tr_mask].argmax(dim=-1).eq(graf['siswa'].y[tr_mask]).sum().item() / tr_mask.sum().item()

        v_mask = graf['siswa'].val_mask
        val_loss = criterion(out[v_mask], graf['siswa'].y[v_mask]).item()
        val_acc = out[v_mask].argmax(dim=-1).eq(graf['siswa'].y[v_mask]).sum().item() / v_mask.sum().item()

        te_mask = graf['siswa'].test_mask
        test_acc = out[te_mask].argmax(dim=-1).eq(graf['siswa'].y[te_mask]).sum().item() / te_mask.sum().item()

        return train_acc, val_loss, val_acc, test_acc

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        out_dict = model(graf.x_dict, graf.edge_index_dict)
        out = out_dict['siswa']
        loss = criterion(out[mask], graf['siswa'].y[mask]).item()
        pred = out[mask].argmax(dim=-1)
        correct = pred.eq(graf['siswa'].y[mask]).sum().item()
        acc = correct / mask.sum().item()
        return loss, acc, pred, graf['siswa'].y[mask]

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    print(f"Mulai pelatihan di device: {device}\n" + "=" * 40)
    for epoch in range(1, epochs + 1):
        train_loss = train()
        train_acc, val_loss, val_acc, test_acc = evaluate_step()

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("=" * 40)
        print(f"Model terbaik dimuat dari Epoch {best_epoch:03d} (Val Loss: {best_val_loss:.4f})")

    return model, history, evaluate


def evaluasi_akhir(model, graf, evaluate, history):
    nama_kelas = ['Rendah', 'Cukup', 'Tinggi']

    test_loss, test_acc, test_pred, test_true = evaluate(graf['siswa'].test_mask)

    y_true = test_true.cpu().numpy()
    y_pred = test_pred.cpu().numpy()

    # Hitung metrik keseluruhan
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 55)
    print("HASIL EVALUASI MODEL GNN TEACHER (DATA TEST)")
    print("=" * 55)
    print(f"Loss (Focal Loss)   : {test_loss:.4f}")
    print(f"Accuracy            : {acc:.4f} ({acc * 100:.2f}%)")
    print("-" * 55)
    print(f"Precision (Macro)   : {prec_macro:.4f} ({prec_macro * 100:.2f}%)")
    print(f"Recall (Macro)      : {rec_macro:.4f} ({rec_macro * 100:.2f}%)")
    print(f"F1-Score (Macro)    : {f1_macro:.4f} ({f1_macro * 100:.2f}%)")
    print("-" * 55)
    print(f"Precision (Weighted): {prec_weighted:.4f} ({prec_weighted * 100:.2f}%)")
    print(f"Recall (Weighted)   : {rec_weighted:.4f} ({rec_weighted * 100:.2f}%)")
    print(f"F1-Score (Weighted) : {f1_weighted:.4f} ({f1_weighted * 100:.2f}%)")
    print("=" * 55)

    print("\nDETAIL PER KELAS (CLASSIFICATION REPORT)")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=nama_kelas, digits=4, zero_division=0))

    # Kurva loss & akurasi
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Dinamika Penurunan Loss pada Model GNN')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (Focal Loss)')
    ax1.legend()

    ax2.plot(history['epoch'], history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['epoch'], history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Peningkatan Akurasi Prediksi Kelas Siswa')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Akurasi')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('kurva_loss_akurasi.png', dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nama_kelas, yticklabels=nama_kelas)
    plt.title(f'Confusion Matrix (Test)\nAccuracy: {acc * 100:.2f}% | F1-Macro: {f1_macro * 100:.2f}%')
    plt.xlabel('Prediksi'); plt.ylabel('Aktual')
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()

    print("\nGrafik disimpan: kurva_loss_akurasi.png, confusion_matrix.png")

@torch.no_grad()
def ekstrak_soft_targets(model, graf, temperature: float = 4.0, simpan_ke: str = "soft_targets_teacher.pt"):
    model.eval()
    device = next(model.parameters()).device
    graf = graf.to(device)

    out_dict = model(graf.x_dict, graf.edge_index_dict)
    logits = out_dict['siswa'].cpu()

    soft_targets = F.softmax(logits / temperature, dim=1) 

    hard_pred = logits.argmax(dim=-1)
    y_true    = graf['siswa'].y.cpu()

    payload = {
        'logits'       : logits,
        'soft_targets' : soft_targets,
        'hard_pred'    : hard_pred,
        'y_true'       : y_true,
        'temperature'  : temperature,
        'train_mask'   : graf['siswa'].train_mask.cpu(),
        'val_mask'     : graf['siswa'].val_mask.cpu(),
        'test_mask'    : graf['siswa'].test_mask.cpu(),
    }

    torch.save(payload, simpan_ke)

    nama_kelas = ['Rendah', 'Cukup', 'Tinggi']
    print("EKSTRAKSI SOFT TARGETS TEACHER — KNOWLEDGE DISTILLATION\n")
    print(f"Temperature (T)    : {temperature}")
    print(f"Ukuran Logits      : {list(logits.shape)}  (N_siswa x N_kelas)")
    print(f"Ukuran Soft Targets: {list(soft_targets.shape)}")
    print(f"\nContoh 3 baris pertama (node siswa 0–2):")
    print(f"{'Node':<6} {'y_true':<8} {'hard_pred':<11} {'Logits':^30} {'Soft Targets (T='+str(temperature)+')':^36}\n")
    for i in range(3):
        logit_str  = "[" + ", ".join(f"{v:.3f}" for v in logits[i].tolist()) + "]"
        soft_str   = "[" + ", ".join(f"{v:.4f}" for v in soft_targets[i].tolist()) + "]"
        print(f"{i:<6} {nama_kelas[y_true[i]]:<8} {nama_kelas[hard_pred[i]]:<11} {logit_str:^30} {soft_str:^36}")
    print(f"\nSoft Targets disimpan ke: '{simpan_ke}'")
    print("Siap digunakan sebagai target pelatihan model Student.")

    return payload

if __name__ == "__main__":
    graf = torch.load("graf_timur.pt", weights_only=False)

    model, history, evaluate = latih_model(graf)
    evaluasi_akhir(model, graf, evaluate, history)

    soft_data = ekstrak_soft_targets(model, graf, temperature=4.0, simpan_ke="soft_targets_teacher.pt")
