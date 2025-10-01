import argparse, os, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.utils import save_image

def build_model():
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--out', default='pred_samples.png')
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

    # Batch d’images à prédire
    batch = [test[i][0] for i in range(min(args.n, len(test)))]
    x = torch.stack(batch).to(device)

    # Model
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
    model.eval()

    # Prédictions
    yhat = model(x).argmax(1).cpu()

    # De-normalize pour sauvegarder en PNG
    imgs = x.cpu() * 0.5 + 0.5
    save_image(imgs, args.out, nrow=int(args.n ** 0.5))

    print(f"✅ Saved {args.out} with {args.n} images")
    print("Predictions (0=not_tshirt, 1=tshirt):")
    print(yhat.tolist())

if __name__ == '__main__':
    main()
