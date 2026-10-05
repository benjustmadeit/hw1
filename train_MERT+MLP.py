import os
import argparse
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from data_loader import get_dataloader, load_manifest
from model import MERTFineTune

SAMPLE_RATE = 24000

ENCODERS = {
    "mert_v1_95m": "m-a-p/MERT-v1-95M",
    "culturemert_95m": "ntua-slp/CultureMERT-95M",
}

DATASETS = {
    "A": {
        "dataset_dir": "/home/benjaminwu/mir_course_dataset/hw1/dataset_A",
        "classes": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    },
    "B": {
        "dataset_dir": "/home/benjaminwu/mir_course_dataset/hw1/dataset_B",
        "classes": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
parser.add_argument("--encoder", choices=ENCODERS.keys(), required=True)
args = parser.parse_args()

dataset_config = DATASETS[args.dataset]
classes = dataset_config["classes"]
manifest_path = os.path.join(dataset_config["dataset_dir"], "manifest.csv")

model_name = ENCODERS[args.encoder]

project_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(project_dir, "checkpoints", f"finetune_{args.dataset}_{args.encoder}.pt")

data = load_manifest(manifest_path, classes=classes)

train_loader = get_dataloader(data, split="train", batch_size=1, num_workers=0) # batch size: A=4, b=1
validation_loader = get_dataloader(data, split="validation", batch_size=1, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
model = MERTFineTune(model_name=model_name, n_classes=len(classes),).to(device)

loss_fn = nn.CrossEntropyLoss()
best_val_top1 = -1.0

# Small learning rate for the encoder, larger learning rate for the classifier
optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-5}, # 1e-5
    {"params": model.classifier.parameters(), "lr": 1e-3}, # 1e-3
])

def prepare_batch(chunks):
    batch_size, num_chunks, num_samples = chunks.shape

    waveforms = chunks.reshape(batch_size * num_chunks, num_samples).cpu().numpy()

    inputs = processor(list(waveforms), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,)
    input_values = inputs["input_values"].to(device)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    return input_values, attention_mask, batch_size, num_chunks

num_epochs = 10

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False,):
        chunks, labels, sample_ids = batch
        labels = labels.to(device)

        input_values, attention_mask, batch_size, num_chunks = prepare_batch(chunks)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_values=input_values,
            batch_size=batch_size,
            num_chunks=num_chunks,
            attention_mask=attention_mask,
        )
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    print(f"Epoch {epoch}/{num_epochs} | train loss: {total_loss / total_samples:.4f}")

    model.eval()
    all_true = []
    all_pred = []
    top3_correct = 0
    total = 0

    with torch.inference_mode():
        for batch in tqdm(validation_loader, desc=f"Epoch {epoch}/{num_epochs} [Validation]", leave=False,):
            chunks, labels, sample_ids = batch
            labels = labels.to(device)

            input_values, attention_mask, batch_size, num_chunks = prepare_batch(chunks)
            logits = model(
                input_values=input_values,
                batch_size=batch_size,
                num_chunks=num_chunks,
                attention_mask=attention_mask,
            )

            top1 = logits.argmax(dim=1)
            top3 = logits.topk(k=3, dim=1).indices

            top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(top1.cpu().tolist())
    
    val_top1 = sum(true == pred for true, pred in zip(all_true, all_pred)) / total
    val_top3 = top3_correct / total

    print(f"Validation Top1: {val_top1:.4f} | Top3: {val_top3:.4f}")

    matrix = confusion_matrix(
            all_true,
            all_pred,
            labels=list(range(len(classes))),
        )
    print("Class order:", classes)
    print("Confusion matrix:")
    print(matrix)
    
    if val_top1 > best_val_top1:
        best_val_top1 = val_top1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dataset": args.dataset,
                "encoder_tag": args.encoder,
                "model_name": model_name,
                "classes": classes,
                "epoch": epoch,
                "val_top1": val_top1,
                "val_top3": val_top3,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")