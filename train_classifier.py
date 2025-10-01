import argparse, os, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

CLASS_NAMES = ["not_tshirt", "tshirt"]

def build_model(num_classes=2):
    # ResNet18 adapté à 1 canal (images Fashion-MNIST en N&B)
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_loaders(batch_size=128, val_split=0.1, data_dir="./data"):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tfm)

    # train/val split
    n = len(train_full)
    idx = np.random.permutation(n)
    n_val = int(val_split * n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # wrapper dataset binaire
    class MapBin(torch.utils.data.Dataset):
        def __init__(self, base, indices=None):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices) if self.indices is not None else len(self.base)
        def __getitem__(self, i):
            j = self.indices[i] if self.indices is not None else i
            x, y = self.base[j]
            yb = 1 if self.base.targets[j].item() == 0 else 0  # classe 0 (tshirt/top) -> 1
            return x, torch.tensor(yb, dtype=torch.long)

    train_bin = MapBin(train_full, train_idx)
    val_bin = MapBin(train_full, val_idx)
    test_bin = MapBin(test, None)

    train_loader = DataLoader(train_bin, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_bin, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_bin, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, opt, device):
    model.train(); losses=[]; preds=[]; gts=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds += out.argmax(1).detach().cpu().tolist()
        gts += y.cpu().tolist()
    return np.mean(losses), accuracy_score(gts, preds), f1_score(gts, preds)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); losses=[]; preds=[]; gts=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())
        preds += out.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()
    return np.mean(losses), accuracy_score(gts, preds), f1_score(gts, preds)

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.colorbar(im)
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints_tshirt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = make_loaders(batch_size=args.batch_size)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, opt, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | Train loss {tr_loss:.3f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"Val loss {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict()}, os.path.join(args.out, "tshirt_resnet18_best.pth"))

    # Test final
    model.load_state_dict(torch.load(os.path.join(args.out, "tshirt_resnet18_best.pth"))["model"])
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Final Test | loss {test_loss:.3f} acc {test_acc:.3f} f1 {test_f1:.3f}")

    # Rapport + confusion matrix
    gts, preds = [], []
    model.eval()
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        preds += out.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()
    print(classification_report(gts, preds, target_names=CLASS_NAMES))
    plot_confusion(gts, preds, os.path.join(args.out, "confusion_matrix.png"))

if __name__ == "__main__":
    main()
