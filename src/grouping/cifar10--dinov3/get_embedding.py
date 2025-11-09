#!/usr/bin/env python3
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from transformers import AutoImageProcessor, AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="HuggingFace model name for DINOv3")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_pth", type=str, default="cifar10_dinov3_embeddings.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load processor + model
    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    # CIFAR-10: train set = 50,000 samples
    transform = lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    dataset = datasets.CIFAR10(root="datasets/", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_embeddings = []

    print("Embedding CIFAR-10 (train set)...")
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader):
            inputs = processor(batch_imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)

            # CLS token embedding: outputs.last_hidden_state[:, 0]
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]
            all_embeddings.append(cls_embeddings.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)  # [50000, 768]

    torch.save(embeddings_tensor, args.output_pth)
    print(f"\n✅ Saved embeddings to {args.output_pth}")
    print(f"Shape: {embeddings_tensor.shape} (expected [50000, 768])")


if __name__ == "__main__":
    main()
