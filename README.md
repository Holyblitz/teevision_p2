# teevision_p2

This project is the second part of the TeeVision portfolio series.
It trains and evaluates a deep learning model to classify T-shirts vs. other clothing items from the Fashion-MNIST dataset.
We also use Grad-CAM to visualize which parts of the image the model focuses on.

# Features

Binary classification: T-shirt (1) vs Not T-shirt (0)

Backbone: ResNet18 adapted to grayscale inputs

Evaluation: accuracy, F1-score, confusion matrix

Explainability: Grad-CAM visualizations

# Use

Train:
python classifiers/train_classifier.py --epochs 20 --batch-size 128 --out checkpoints_tshirt

Evaluate
python classifiers/eval_tshirt.py --ckpt checkpoints_tshirt/tshirt_resnet18_best.pth

Inference (save prediction grid)
python classifiers/infer_tshirt_classifier.py --ckpt checkpoints_tshirt/tshirt_resnet18_best.pth --n 64 --out preds.png

Explainability
python classifiers/explain_gradcam.py --ckpt checkpoints_tshirt/tshirt_resnet18_best.pth --k 8 --outdir cam_reports

# Results
Results

Accuracy: ~XX%

F1-score: ~XX

Confusion matrix and Grad-CAM samples are available in the repo.

Requirements:
torch
torchvision
scikit-learn
matplotlib
opencv-python
grad-cam
