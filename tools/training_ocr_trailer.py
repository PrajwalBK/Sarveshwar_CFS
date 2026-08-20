"""
OCR CRNN Training Script for Trailer ID Recognition

This trains a CRNN (CNN + BiLSTM + CTC) model for text recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import os

# Configuration
class Config:
    # Model
    img_height = 32
    img_width = 320
    num_channels = 1  # Grayscale
    
    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # Paths
    train_data_dir = "training_data/train"
    train_label_file = "training_data/train_label.txt"
    valid_data_dir = "training_data/valid"
    valid_label_file = "training_data/valid_label.txt"
    alphabet_file = "training_data/alphabet.txt"
    output_dir = "models"

# Load alphabet
def load_alphabet(path):
    with open(path, 'r') as f:
        alphabet = f.read().strip()
    return alphabet

# OCR Dataset
class OCRDataset(Dataset):
    def __init__(self, data_dir, label_file, alphabet, img_height=32, img_width=320):
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.alphabet = alphabet
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 is blank
        self.samples = []
        
        # Load labels
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) >= 1:
                    filename = parts[0]
                    label = parts[1] if len(parts) > 1 else ""
                    if (self.data_dir / filename).exists():
                        self.samples.append((filename, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        
        # Load and preprocess image
        img_path = self.data_dir / filename
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add channel dim
        
        # Encode label
        encoded_label = [self.char_to_idx.get(c.upper(), 0) for c in label if c.upper() in self.char_to_idx]
        
        return torch.FloatTensor(img), torch.LongTensor(encoded_label), len(encoded_label)

# CRNN Model
class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
        )
        
        # RNN (BiLSTM)
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        
        # Classifier
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # [B, 512, 1, W']
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)  # [B, W', 512]
        
        # RNN
        rnn_out, _ = self.rnn(conv)  # [B, W', hidden*2]
        
        # Classifier
        output = self.classifier(rnn_out)  # [B, W', num_classes]
        output = output.permute(1, 0, 2)  # [W', B, num_classes] for CTC
        
        return output

# Training function
def train_ocr():
    config = Config()
    
    # Load alphabet
    alphabet = load_alphabet(config.alphabet_file)
    num_classes = len(alphabet) + 1  # +1 for blank
    
    print(f"Alphabet: {alphabet} ({len(alphabet)} chars)")
    print(f"Num classes: {num_classes}")
    
    # Create datasets
    train_dataset = OCRDataset(
        config.train_data_dir, config.train_label_file, 
        alphabet, config.img_height, config.img_width
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # DataLoader with collate function for variable length labels
    def collate_fn(batch):
        images, labels, lengths = zip(*batch)
        images = torch.stack(images)
        # Pad labels to max length
        max_len = max(lengths)
        padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        return images, padded_labels, torch.LongTensor(lengths)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(config.img_height, config.num_channels, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print(f"\nStarting training on {device}...")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for images, labels, lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # [T, B, C]
            
            # CTC loss
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{config.output_dir}/ocr_crnn_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"{config.output_dir}/ocr_crnn_final.pth")
    print(f"\nTraining complete! Model saved to {config.output_dir}/ocr_crnn_final.pth")
    
    return model

if __name__ == "__main__":
    train_ocr()