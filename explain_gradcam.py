import argparse, os, cv2, numpy as np, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def build_model():
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--outdir', default='cam_reports')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    # Data
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=tfm)

    # Model
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
    model.eval()

    target_layer = model.layer4[-1]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for i in range(args.k):
            img, _ = test[i]                         # (1,224,224) normalized
            inp = img.unsqueeze(0).to(device)        # (1,1,224,224)

            model.zero_grad()                        # important: clear grads
            grayscale_cam = cam(input_tensor=inp)[0] # (H,W) in [0,1]

            # de-normalize to [0,1] and make RGB
            img_np = img.squeeze().numpy() * 0.5 + 0.5
            rgb = np.repeat(img_np[..., None], 3, axis=2).astype(np.float32)

            vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            cv2.imwrite(os.path.join(args.outdir, f'cam_{i:03d}.png'),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"Saved {args.k} Grad-CAM images to {args.outdir}")

if __name__ == '__main__':
    main()


