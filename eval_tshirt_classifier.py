import argparse, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

CLASS_NAMES = ["not_tshirt", "tshirt"]

def build_model():
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)

    # Data
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=tfm)

    # Binary targets (class 0 = T-shirt/top → label=1, else=0)
    test_bin = [(img, 1 if y == 0 else 0) for (img, y) in test]
    loader = DataLoader(test_bin, batch_size=128, shuffle=False)

    # Model
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
    model.eval()

    preds, gts = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds += out.argmax(1).cpu().tolist()
        gts += y

    print(classification_report(gts, preds, target_names=CLASS_NAMES))

if __name__ == '__main__':
    main()
