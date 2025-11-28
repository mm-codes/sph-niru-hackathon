import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import BertTokenizer, BertModel

class AudioDeepfakeDetector(nn.Module):
    """
    CNN-LSTM hybrid model for audio deepfake detection
    Analyzes spectral features to identify synthetic speech patterns
    """
    def __init__(self, input_features=128, hidden_size=256, num_layers=2):
        super(AudioDeepfakeDetector, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_features, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(256, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary: real vs fake
        
    def forward(self, x):
        # x shape: (batch, features, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classification
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output, attn_weights


class TextAIDetector(nn.Module):
    """
    BERT-based model for detecting AI-generated text
    Analyzes linguistic patterns and coherence
    """
    def __init__(self, bert_model='bert-base-uncased', hidden_size=256):
        super(TextAIDetector, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.3)
        
        # Additional feature extraction layers
        self.lstm = nn.LSTM(768, hidden_size, 2, 
                           batch_first=True, bidirectional=True)
        
        # Statistical feature analyzer
        self.stat_fc = nn.Linear(10, 64)  # For statistical features
        
        # Combined classifier
        self.fc1 = nn.Linear(hidden_size * 2 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Binary: human vs AI
        
    def forward(self, input_ids, attention_mask, stat_features):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # LSTM for additional context
        lstm_out, _ = self.lstm(sequence_output)
        pooled = torch.mean(lstm_out, dim=1)
        
        # Statistical features
        stat_out = F.relu(self.stat_fc(stat_features))
        
        # Combine features
        combined = torch.cat([pooled, stat_out], dim=1)
        
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        
        return output