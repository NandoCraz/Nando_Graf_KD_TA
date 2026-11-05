"""
Model GNN Teacher berbasis GraphSAGE — Skripsi
Diadaptasi dari kode kating (Michael_Modeling_GNN_SHAP.ipynb) dan
diimplementasikan menggunakan metode GraphSAGE dari folder `graphsage/`.

Memuat graf hasil persiapan_data.py (graf_timur.pt), melatih model GNN
Teacher dengan arsitektur GraphSAGE, lalu mengevaluasinya.

ARSITEKTUR GraphSAGE (sesuai folder graphsage/):
- graphsage/aggregators.py  → MeanAggregator : merata-rata embedding tetangga
  yang disampling → dipadankan dengan SAGEConv(aggr='mean') di PyG
- graphsage/encoders.py     → Encoder        : concat(h_self, h_neigh) lalu
  Linear → ReLU (2 lapis), persis perilaku SAGEConv
- graphsage/model.py        → SupervisedGraphSage : stack 2 Encoder (2-hop),
  classifier Linear di atas embedding akhir

Implementasi di sini memakai SAGEConv dari PyTorch Geometric yang secara
matematis identik dengan MeanAggregator + Encoder di atas, namun sudah
dioptimasi untuk sparse graph dan mendukung heterogeneous graph via to_hetero.

Langkah Knowledge Distillation ke model Student BELUM ada di sini --
itu bagian yang perlu dikembangkan sendiri sebagai kontribusi skripsi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.metrics import confusion_matrix, classification_report
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
    """
    Model GraphSAGE Teacher — arsitektur 4-lapis dengan SAGEConv.

    Korespondensi dengan folder graphsage/:
    - Setiap SAGEConv(aggr='mean') ≡ MeanAggregator + Encoder satu lapis.
      SAGEConv menghitung:  h = W · concat(h_self, mean(h_tetangga)) + b
      yang persis sama dengan forward() di encoders.py (concat → weight.mm).
    - Stack 4 lapis SAGEConv ≡ 4-hop neighborhood aggregation, mirip
      enc2 yang di-stack di atas enc1 pada graphsage/model.py run_cora().
    - Dropout antar lapis mencegah overfitting, analog dengan sampling
      tetangga (num_sample) di MeanAggregator yang juga berfungsi sebagai
      regularisasi implisit.

    Parameter:
        hidden_channels : dimensi embedding tiap lapis tersembunyi
        out_channels    : jumlah kelas output (3: Rendah/Cukup/Tinggi)
        dropout         : probabilitas dropout antar lapis (default 0.3)
    """
    def __init__(self, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Lapis 1 — setara agg1 + enc1 di graphsage/model.py
        # normalize=True → L2-normalisasi embedding output (menstabilkan training)
        self.conv1 = SAGEConv((-1, -1), hidden_channels, normalize=True)

        # Lapis 2 — setara agg2 + enc2 di graphsage/model.py (2-hop)
        self.conv2 = SAGEConv((-1, -1), hidden_channels, normalize=True)

        # Lapis 3 & 4 — perluasan ke 4-hop untuk graf yang lebih dalam
        self.conv3 = SAGEConv((-1, -1), hidden_channels, normalize=True)
        self.conv4 = SAGEConv((-1, -1), out_channels,    normalize=False)

    def forward(self, x, edge_index):
        # Lapis 1: aggregate tetangga hop-1 → ReLU → Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Lapis 2: aggregate tetangga hop-2 → ReLU → Dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Lapis 3: aggregate tetangga hop-3 → ReLU → Dropout
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Lapis 4: proyeksi ke ruang kelas (logits), tanpa aktivasi
        x = self.conv4(x, edge_index)
        return x

def latih_model(graf, hidden_channels=64, out_channels=3, dropout=0.3,
                lr=0.005, weight_decay=5e-4, epochs=500):

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

    print(f"Mulai pelatihan di device: {device}\n" + "=" * 40)
    for epoch in range(1, epochs + 1):
        train_loss = train()
        _, train_acc, _, _ = evaluate(graf['siswa'].train_mask)
        val_loss, val_acc, _, _ = evaluate(graf['siswa'].val_mask)
        _, test_acc, _, _ = evaluate(graf['siswa'].test_mask)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}')

    return model, history, evaluate


def evaluasi_akhir(model, graf, evaluate, history):
    nama_kelas = ['Rendah', 'Cukup', 'Tinggi']

    test_loss, test_acc, test_pred, test_true = evaluate(graf['siswa'].test_mask)
    print("=" * 50)
    print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print("=" * 50)

    y_true = test_true.cpu().numpy()
    y_pred = test_pred.cpu().numpy()

    print("\nCLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=nama_kelas))

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
    plt.title(f'Confusion Matrix (Test)\nAccuracy: {test_acc * 100:.2f}%')
    plt.xlabel('Prediksi'); plt.ylabel('Aktual')
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()

    print("\nGrafik disimpan: kurva_loss_akurasi.png, confusion_matrix.png")


if __name__ == "__main__":
    graf = torch.load("graf_timur.pt", weights_only=False)

    model, history, evaluate = latih_model(graf)
    evaluasi_akhir(model, graf, evaluate, history)

    # --- Selanjutnya (belum ditulis di sini): Knowledge Distillation ke model Student ---
