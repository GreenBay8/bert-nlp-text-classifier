import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode labels
encoder = LabelEncoder()
df["label_encoded"] = encoder.fit_transform(df["label"])

texts = df["text"].tolist()
labels = df["label_encoded"].tolist()

# Load BERT
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)

# Tokenize
encodings = tokenizer(texts, truncation=True, padding=True)

# Dataset class
class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = Dataset(encodings, labels)

# Train
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
