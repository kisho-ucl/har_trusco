import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# CNN Encoder
# -----------------------------
class CNN_Encoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, feature_dim, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 10, 4, 3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 6, 2, 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = nn.Conv1d(hidden_dim, feature_dim, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(feature_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        return torch.mean(x, dim=-1)


# -----------------------------
# SimCLR Projection
# -----------------------------
class SimCLR_new(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, feature_dim, feedforward_dim,
                 num_heads, num_layers, projection_dim, dropout=0.3):
        super().__init__()
        self.encoder = CNN_Encoder(input_dim, seq_len, hidden_dim, feature_dim, dropout)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projection(feat)
        return F.normalize(proj, dim=-1)


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# Simple MLP Classifier
# -----------------------------
class TaskClassifier_simple(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(torch.mean(x, dim=1))
    
# -----------------------------
# Pretrained CNN + Transformer 
# -----------------------------
class TaskClassifier_Transformer(nn.Module):
    def __init__(self, encoder, hidden_dim, window_len,
                 overlap, num_heads, num_layers, num_classes,
                 freeze_encoder=False):
        super().__init__()

        self.encoder = encoder
        self.window_len = window_len
        self.stride = int(window_len * (1 - overlap))

        # === Encoder の freeze 設定 ===
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos = PositionalEncoding(hidden_dim, 500)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def extract_windows(self, x):
        x = x.unfold(2, self.window_len, self.stride)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, signals):
        B, C, T = signals.shape

        # Step1: windowing
        x = self.extract_windows(signals)     # (B,W,C,Tw)
        B, W, C, Tw = x.shape

        # Step2: CNN encoder
        feats = self.encoder(x.reshape(B * W, C, Tw))     # (B*W,F)
        feats = feats.reshape(B, W, -1)                   # (B,W,F)

        # Step3: positional encoding + CLS
        x = self.pos(feats)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), 1)

        # Step4: transformer
        h = self.transformer(x)

        # Step5: classifier
        return self.classifier(h[:, 0])




# -----------------------------
# Pretrained CNN + Transformer (Fine-tuning)
# -----------------------------
class TaskClassifier_Transformer_Tuning(nn.Module):
    def __init__(self, encoder, hidden_dim, window_len,
                 overlap, num_heads, num_layers, num_classes):
        super().__init__()
        self.encoder = encoder
        self.window_len = window_len
        self.stride = int(window_len * (1 - overlap))

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos = PositionalEncoding(hidden_dim, 500)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def extract_windows(self, x):
        x = x.unfold(2, self.window_len, self.stride)
        return x.permute(0, 2, 1, 3)

    def forward(self, signals):
        B, C, T = signals.shape
        x = self.extract_windows(signals)
        B, W, C, Tw = x.shape

        feats = self.encoder(x.view(B * W, C, Tw)).view(B, W, -1)

        x = self.pos(feats)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), 1)

        h = self.transformer(x)
        return self.classifier(h[:, 0])
