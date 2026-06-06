"""
Olist Review Score Analyzer — Streamlined Gradio App
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import tiktoken
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import joblib
from statsmodels.miscmodels.ordinal_model import OrderedModel

# --- ADD THESE IMPORTS ---
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- ADD THIS DATA LOADING BLOCK ---
# Path to your local PBI processed dataset
PBI_DATA_PATH = "/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/interim/olist_PBI_processed_dataset.csv"

try:
    df_pbi = pd.read_csv(PBI_DATA_PATH)
    # Convert timestamps for time-series charts
    df_pbi['order_purchase_timestamp'] = pd.to_datetime(df_pbi['order_purchase_timestamp'])
except Exception as e:
    print(f"[App] Error loading PBI dataset: {e}")
    df_pbi = pd.DataFrame()


from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def generate_dashboard_plots():
    # If data fails to load, return 12 empty figures to prevent crashes
    if df_pbi.empty:
        return [go.Figure()] * 12

    # --- Existing Plots 1 to 9 (Summarized for space, keep your existing logic here) ---
    
    # 1. Data Distributions
    fig_dist = make_subplots(rows=2, cols=2, subplot_titles=('Price', 'Freight', 'Actual Days', 'Delay Days'))
    fig_dist.add_trace(go.Histogram(x=df_pbi['price'], marker_color='blue'), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=df_pbi['freight_value'], marker_color='orange'), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=df_pbi['actual_delivery_days'], marker_color='green'), row=2, col=1)
    fig_dist.add_trace(go.Histogram(x=df_pbi['delivery_delay_days'], marker_color='red'), row=2, col=2)
    fig_dist.update_layout(height=500, showlegend=False)

    # 2. Pearson Correlation
    num_cols = ['price', 'freight_value', 'product_weight_g', 'product_photos_qty', 'actual_delivery_days', 'review_score']
    fig_corr = px.imshow(df_pbi[num_cols].corr().round(2), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Feature Correlations")

    # 3. RFM Segmentation
    max_date = df_pbi['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df_pbi.groupby('customer_unique_id').agg({'order_purchase_timestamp': lambda x: (max_date - x.max()).days, 'order_id': 'count', 'price': 'sum'}).reset_index()
    rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[1, 2, 3, 4])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[4, 3, 2, 1])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[4, 3, 2, 1])
    rfm['RFM_Group'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm_top10 = rfm['RFM_Group'].value_counts().head(10).reset_index()
    rfm_top10.columns = ['RFM Score', 'Number of Customers']
    fig_rfm = px.bar(rfm_top10, x='RFM Score', y='Number of Customers', title='Top 10 Customer Segments by RFM', text_auto=True, color='Number of Customers')

    # 4. Delivery Duration vs Review Score
    fig_del_rev = px.box(df_pbi, x='review_score', y='actual_delivery_days', title='Delivery Duration Impact on Reviews', color='review_score')
    fig_del_rev.update_traces(boxpoints=False)

    # 5. Top Categories Revenue
    top_cat = df_pbi.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(15).reset_index()
    fig_cat_rev = px.bar(top_cat, x='price', y='product_category_name', orientation='h', title='Gross Revenue by Top Categories')
    fig_cat_rev.update_layout(yaxis={'categoryorder':'total ascending'})

    # 6. Average Freight by State
    state_freight = df_pbi.groupby('customer_state')['freight_value'].mean().sort_values(ascending=False).reset_index()
    fig_freight = px.bar(state_freight, x='customer_state', y='freight_value', title='Average Freight Cost by State')

    # 7. Sales by Hour
    if 'order_hour' not in df_pbi.columns: df_pbi['order_hour'] = df_pbi['order_purchase_timestamp'].dt.hour
    hour_counts = df_pbi['order_hour'].value_counts().sort_index().reset_index()
    hour_counts.columns = ['Hour', 'Total Orders']
    fig_hour = px.bar(hour_counts, x='Hour', y='Total Orders', title='Sales Volume by Hour', color='Total Orders')

    # 8. Avg Delay by State
    state_delay = df_pbi.groupby('customer_state')['delivery_delay_days'].mean().sort_values(ascending=False).reset_index()
    fig_delay = px.bar(state_delay, x='customer_state', y='delivery_delay_days', title='Logistics Bottlenecks: Avg Delay by State', color='delivery_delay_days', color_continuous_scale='Reds')

    # 9. Review Word Count vs Score
    if 'review_length_words' not in df_pbi.columns: df_pbi['review_length_words'] = df_pbi['clean_review'].astype(str).apply(lambda x: len(x.split()))
    fig_rev_len = px.box(df_pbi, x='review_score', y='review_length_words', title='NLP Insight: Review Word Count vs. Score', color='review_score')
    fig_rev_len.update_traces(boxpoints=False)

    # ---------------------------------------------------------
    # 10. NEW 1: Freight-to-Price Ratio (Shipping Penalty)
    # ---------------------------------------------------------
    # Calculate ratio, replacing 0 price with a small number to avoid division by zero
    if 'freight_ratio' not in df_pbi.columns:
        df_pbi['freight_ratio'] = df_pbi['freight_value'] / df_pbi['price'].replace(0, 0.01)
    
    fig_freight_ratio = px.box(df_pbi, x='review_score', y='freight_ratio',
                               title='The Shipping Penalty: Freight-to-Price Ratio vs. Reviews',
                               labels={'freight_ratio': 'Freight / Price Ratio'}, color='review_score')
    fig_freight_ratio.update_traces(boxpoints=False)
    fig_freight_ratio.update_yaxes(range=[0, 1.5]) # Cap the y-axis to hide extreme outliers and keep it readable

    # ---------------------------------------------------------
    # 11. NEW 2: Time-Series Sentiment & Verbosity
    # ---------------------------------------------------------
    df_time_nlp = df_pbi.set_index('order_purchase_timestamp').resample('ME').agg({
        'review_score': 'mean', 
        'review_length_words': 'mean'
    }).reset_index()
    
    fig_nlp_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_nlp_time.add_trace(go.Scatter(x=df_time_nlp['order_purchase_timestamp'], y=df_time_nlp['review_score'], name="Avg Score", line=dict(color='blue')), secondary_y=False)
    fig_nlp_time.add_trace(go.Scatter(x=df_time_nlp['order_purchase_timestamp'], y=df_time_nlp['review_length_words'], name="Avg Word Count", line=dict(color='red', dash='dot')), secondary_y=True)
    fig_nlp_time.update_layout(title_text="Brand Health: Customer Satisfaction & Review Verbosity Over Time", margin=dict(l=20, r=20, t=40, b=20))
    fig_nlp_time.update_yaxes(title_text="Average Review Score", secondary_y=False)
    fig_nlp_time.update_yaxes(title_text="Average Word Count", secondary_y=True)

    # ---------------------------------------------------------
    # 12. NEW 3: Listing Quality (Photos vs Volume)
    # ---------------------------------------------------------
    photo_agg = df_pbi.groupby('product_photos_qty').agg({'order_id': 'count', 'review_score': 'mean'}).reset_index()
    photo_agg = photo_agg[photo_agg['product_photos_qty'] <= 10] # Filter out extremes for better visualization
    
    fig_photos = px.bar(photo_agg, x='product_photos_qty', y='order_id', color='review_score',
                        title='Listing Quality: Number of Photos vs. Sales Volume',
                        labels={'product_photos_qty': 'Number of Photos', 'order_id': 'Total Orders', 'review_score': 'Avg Score'},
                        color_continuous_scale='RdYlGn')
    fig_photos.update_layout(xaxis=dict(tickmode='linear', dtick=1))

    # CRITICAL: Return exactly 12 figures
    return (fig_dist, fig_corr, fig_rfm, fig_del_rev, fig_cat_rev, fig_freight, 
            fig_hour, fig_delay, fig_rev_len, fig_freight_ratio, fig_nlp_time, fig_photos)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP & REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[App] Device: {device}")

_ROOT = "/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/notebooks/nlp_evaluation"

MODEL_REGISTRY = {
    "GPT-2 124M | Cross-Entropy": {"family": "gpt2", "size": "124M", "ckpt": os.path.join(_ROOT, "gpt2", "gpt2_124M_weights.npz")},
    "GPT-2 124M | Huber Loss":    {"family": "gpt2", "size": "124M", "ckpt": os.path.join(_ROOT, "gpt2", "gpt2_124M_huber_weights.pt")},
    "GPT-2 355M | Cross-Entropy": {"family": "gpt2", "size": "355M", "ckpt": os.path.join(_ROOT, "gpt2", "gpt2_355M_weights.npz")},
    "GPT-2 355M | Huber Loss":    {"family": "gpt2", "size": "355M", "ckpt": os.path.join(_ROOT, "gpt2", "gpt2_355M_huber_weights.pt")},
    "RoBERTa-base | Cross-Entropy":  {"family": "roberta", "hf_name": "roberta-base", "ckpt": os.path.join(_ROOT, "bert", "roberta_base_review_best.pt")},
    "RoBERTa-large | Cross-Entropy": {"family": "roberta", "hf_name": "roberta-large", "ckpt": os.path.join(_ROOT, "bert", "roberta_large_review_best.pt")},
    "Text-JEPA | Predictor":         {"family": "jepa", "ckpt": os.path.join(_ROOT, "vljepa", "textjepa_predictor_best.pt")}
}

NUM_CLASSES = 5
MAX_LENGTH  = 256


# ─────────────────────────────────────────────────────────────────────────────
# 2. DELIVERY PREDICTION MODELS & SETUP
# ─────────────────────────────────────────────────────────────────────────────

# --- Regression Model Architecture ---
class DeliveryDelayNN(nn.Module):
    def __init__(self, input_dim, y_min, y_max, dropout_rate=0.2):
        super(DeliveryDelayNN, self).__init__()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8),
        ])
        self.y_min = y_min
        self.y_max = y_max
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.gelu(x)
            x = self.dropout(x)
        raw_out = self.output_layer(x)
        final_out = self.y_min + (self.y_max - self.y_min) * torch.sigmoid(raw_out)
        return final_out

# --- Classification Model Architecture ---
class RegularizedDeepClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(RegularizedDeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 6)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.gelu(self.fc2(x)))
        x = self.dropout(self.gelu(self.fc3(x)))
        x = self.dropout(self.gelu(self.fc4(x)))
        x = self.dropout(self.gelu(self.fc5(x)))
        x = self.dropout(self.gelu(self.fc6(x)))
        return self.out(x)


# IMPORTANT: Replace these with the actual min/max of df['actual_delivery_days'] from your training data!
Y_MIN_TRAIN = 0.0  
Y_MAX_TRAIN = 28.0

# Load Regression Model
reg_model_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/Neural_Network_Model.pkl'
reg_model = DeliveryDelayNN(input_dim=7, y_min=Y_MIN_TRAIN, y_max=Y_MAX_TRAIN).to(device)
reg_model.load_state_dict(torch.load(reg_model_path, map_location=device, weights_only=True))
reg_model.eval()

# Load Classification Model
cls_model_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/Neural_Network_Model_classification.pkl'
cls_model = RegularizedDeepClassifier(input_dim=7).to(device)
cls_model.load_state_dict(torch.load(cls_model_path, map_location=device, weights_only=True))
cls_model.eval()

# Classification Labels
STATUS_MAPPING = {
    0: 'Very Early',
    1: 'Somewhat Early',
    2: 'Early',
    3: 'On Time',
    4: 'Somewhat Late',
    5: 'Very Late'
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD REVIEW ML MODELS
# ─────────────────────────────────────────────────────────────────────────────

# Load Ordinal Logistic Regression (OLR) Review Model
rev_olr_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/Reviews_ordinal_logistic_regression.pth'
rev_olr_model = joblib.load(rev_olr_path)

# Load Random Forest Review Model
rev_rf_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/Reviews_Random_Forest_Model.pkl'
rev_rf_model = joblib.load(rev_rf_path)

# Load SVM Review Model
rev_svm_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/Reviews_SVM_Model.pkl'
rev_svm_model = joblib.load(rev_svm_path)

# Load XGBoost Review Model
rev_xgb_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/Reviews_XGBoost_Model.pkl'
rev_xgb_model = joblib.load(rev_xgb_path)

# Mapping indices (0 to 4) back to 1-5 Star Ratings
REVIEW_SCORE_MAPPING = {
    0: '1 Star ⭐',
    1: '2 Stars ⭐⭐',
    2: '3 Stars ⭐⭐⭐',
    3: '4 Stars ⭐⭐⭐⭐',
    4: '5 Stars ⭐⭐⭐⭐⭐'
}



# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD MACHINE LEARNING MODELS
# ─────────────────────────────────────────────────────────────────────────────

# Load Ordinal Logistic Regression (OLR) Model
olr_model_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/ordinal_logistic_regression.pth'
olr_model = joblib.load(olr_model_path)

# Load Random Forest Model
rf_model_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/Random_Forest_Model.pkl'
rf_model = joblib.load(rf_model_path)

# Load SVM Model
svm_model_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/SVM_Model.pkl'
svm_model = joblib.load(svm_model_path)

# Load XGBoost Model
xgb_model_path = '/home/labib/Desktop/labib_codes/uiu_codes/7th_trimester/da_project/da_project_local/data/model/XGBoost_Model.pkl'
xgb_model = joblib.load(xgb_model_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────
# Compacted GPT-2 Architecture
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale, self.shift = nn.Parameter(torch.ones(emb_dim)), nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        return self.scale * (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt() + self.shift

class GPTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, n_heads = cfg["emb_dim"], cfg["n_heads"]
        self.att = nn.MultiheadAttention(d, n_heads, dropout=cfg["drop_rate"], batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1, self.n2 = LayerNorm(d), LayerNorm(d)
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.register_buffer("mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool())
        
    def forward(self, x):
        t = x.size(1)
        attn_out, _ = self.att(self.n1(x), self.n1(x), self.n1(x), attn_mask=self.mask[:t, :t])
        x = x + self.drop(attn_out)
        return x + self.drop(self.ff(self.n2(x)))

class GPTClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.blocks  = nn.Sequential(*[GPTBlock(cfg) for _ in range(cfg["n_layers"])])
        self.out     = nn.Linear(cfg["emb_dim"], cfg["num_classes"], bias=False)
    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.size(1), device=idx.device))
        return self.out(LayerNorm(x.size(-1)).to(idx.device)(self.blocks(x)))

GPT_CFG = {
    "124M": {"vocab_size":50257, "context_length":1024, "emb_dim":768, "n_heads":12, "n_layers":12, "drop_rate":0.1, "num_classes":5},
    "355M": {"vocab_size":50257, "context_length":1024, "emb_dim":1024, "n_heads":16, "n_layers":24, "drop_rate":0.1, "num_classes":5},
}

class RoBERTaClassifier(nn.Module):
    def __init__(self, hf_name):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(hf_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, NUM_CLASSES)
    def forward(self, input_ids, attention_mask):
        return self.classifier(self.roberta(input_ids, attention_mask).last_hidden_state[:, 0, :])

# ─────────────────────────────────────────────────────────────────────────────
# EXACT TEXT-JEPA REPLACEMENT BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class FrozenEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size
        for param in self.bert.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = out.last_hidden_state                     
        mask_exp   = attention_mask.unsqueeze(-1).float()      
        sum_emb    = (token_embs * mask_exp).sum(dim=1)        
        count      = mask_exp.sum(dim=1).clamp(min=1e-9)       
        return sum_emb / count                                 

class TextJEPAPredictor(nn.Module):
    def __init__(self, vocab_size=30522, encoder_dim=768, predictor_dim=512, out_dim=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.context_proj = nn.Sequential(nn.Linear(encoder_dim, predictor_dim), nn.LayerNorm(predictor_dim))
        self.query_embed = nn.Embedding(vocab_size, predictor_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=predictor_dim, nhead=n_heads, dim_feedforward=predictor_dim * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(nn.Linear(predictor_dim, predictor_dim), nn.GELU(), nn.Linear(predictor_dim, out_dim))

    def forward(self, context_emb, query_ids, query_mask):
        B = context_emb.size(0)
        ctx_tok = self.context_proj(context_emb).unsqueeze(1)
        ctx_mask = torch.ones(B, 1, device=context_emb.device)
        q_tok = self.query_embed(query_ids)
        combined = torch.cat([ctx_tok, q_tok], dim=1)
        combined_mask = torch.cat([ctx_mask, query_mask], dim=1)
        pad_mask = combined_mask == 0
        tf_out = self.transformer(combined, src_key_padding_mask=pad_mask)
        real_mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (tf_out * real_mask).sum(1) / real_mask.sum(1).clamp(min=1e-9)
        s_hat_y = self.out_proj(pooled)
        return torch.nn.functional.normalize(s_hat_y, p=2, dim=-1)

class TextJEPAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.shared_encoder = FrozenEncoder("distilbert-base-uncased")
        self.predictor = TextJEPAPredictor(vocab_size=self.tokenizer.vocab_size)
        self.y_proj = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 256))
        
        self.label_strings = [
            "1 star — very negative review",
            "2 stars — negative review",
            "3 stars — neutral review",
            "4 stars — positive review",
            "5 stars — very positive review",
        ]
        
    def load_state_dict(self, state_dict, strict=False):
        # Gracefully intercepts your custom nested notebook dictionary
        if "predictor" in state_dict and "y_proj" in state_dict:
            self.predictor.load_state_dict(state_dict["predictor"], strict=strict)
            self.y_proj.load_state_dict(state_dict["y_proj"], strict=strict)
        else:
            super().load_state_dict(state_dict, strict=strict)

    def encode_target(self, label_ids, label_mask):
        s_y_raw = self.shared_encoder(label_ids, label_mask)
        s_y = self.y_proj(s_y_raw)
        return torch.nn.functional.normalize(s_y, p=2, dim=-1)

    def forward(self, idx):
        device = idx.device
        
        # 1. Reverse the app's GPT-2 tokenization back to raw text
        token_list = [t for t in idx[0].cpu().tolist() if t != 50256]
        text = gpt2_tok.decode(token_list)
        
        # 2. Tokenize context and query properly with DistilBERT
        enc = self.tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(device)
        q_enc = self.tokenizer("What is the sentiment score of this review?", max_length=16, padding="max_length", truncation=True, return_tensors="pt").to(device)
        
        # 3. Pass through Predictor
        context_emb = self.shared_encoder(enc["input_ids"], enc["attention_mask"])
        s_hat_y = self.predictor(context_emb, q_enc["input_ids"], q_enc["attention_mask"])
        
        # 4. Generate Target Embeddings (Candidates)
        l_enc = self.tokenizer(self.label_strings, max_length=8, padding="max_length", truncation=True, return_tensors="pt").to(device)
        cand_embs = self.encode_target(l_enc["input_ids"], l_enc["attention_mask"])
        
        # 5. Cosine Similarity -> Scaled to act as UI Probabilities
        similarity_scores = torch.matmul(s_hat_y, cand_embs.T)
        return similarity_scores * 20.0

# ─────────────────────────────────────────────────────────────────────────────
# 3. ROBUST MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
_cache = {"name": None, "model": None}
gpt2_tok = tiktoken.get_encoding("gpt2")
_bert_toks = {}

def load_model(name: str):
    if _cache["name"] == name: return _cache["model"]
    reg, ckpt_path = MODEL_REGISTRY[name], MODEL_REGISTRY[name]["ckpt"]
    if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Missing file: {ckpt_path}")

    # Init Architecture
    if reg["family"] == "gpt2": model = GPTClassifier(GPT_CFG[reg["size"]])
    elif reg["family"] == "roberta": model = RoBERTaClassifier(reg["hf_name"])
    else: model = TextJEPAClassifier()

    # Safe Weight Loading
    if ckpt_path.endswith('.npz'):
        weights, state_dict = np.load(ckpt_path), model.state_dict()
        for k, v in weights.items():
            if k in state_dict: state_dict[k].copy_(torch.from_numpy(v))
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        # Fixes the exact KeyError: 'model' shown in the image
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    _cache.update({"name": name, "model": model})
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE & UI
# ─────────────────────────────────────────────────────────────────────────────
def analyze_review(text: str, model_name: str):
    if not text.strip(): return "⚠️ Please enter a review.", None

    try:
        model, reg = load_model(model_name), MODEL_REGISTRY[model_name]
        
        with torch.no_grad():
            if reg["family"] == "gpt2":
                ids = gpt2_tok.encode(text)[:MAX_LENGTH]
                ids += [50256] * (MAX_LENGTH - len(ids))
                logits = model(torch.tensor([ids], device=device))[:, -1, :]
            elif reg["family"] == "roberta":
                if reg["hf_name"] not in _bert_toks:
                    _bert_toks[reg["hf_name"]] = AutoTokenizer.from_pretrained(reg["hf_name"])
                enc = _bert_toks[reg["hf_name"]](text, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(device)
                logits = model(enc["input_ids"], enc["attention_mask"])
            else:
                logits = model(torch.tensor([gpt2_tok.encode(text)[:MAX_LENGTH]], device=device))

            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_score = int(np.argmax(probs)) + 1
        
        # UI Outputs
        badge_color = ["#c62828", "#e65100", "#f9a825", "#558b2f", "#1b5e20"][pred_score-1]
        stars = "⭐" * pred_score
        html = f"<div style='text-align:center;padding:20px;background:{badge_color};border-radius:8px;color:white;font-size:2em;font-weight:bold;'>{stars}<br>Score: {pred_score} / 5</div>"
        
        # Format for Gradio's native gr.Label
        conf_dict = {f"Score {i+1}": p for i, p in enumerate(probs)}
        return html, conf_dict

    except Exception as e:
        import traceback
        traceback.print_exc() # Prints full error to terminal
        return f"<div style='color:red;'><b>Error:</b> {str(e)}</div>", None

def predict_delivery(f1, f2, f3, f4, f5, f6, f7):
    """Takes 7 input features, runs them through all 6 models."""
    try:
        inputs = [f1, f2, f3, f4, f5, f6, f7]
        
        # 1. PyTorch Neural Networks
        input_tensor = torch.tensor([inputs], dtype=torch.float32).to(device)
        with torch.no_grad():
            reg_output = reg_model(input_tensor)
            predicted_days = reg_output.item()
            
            cls_output = cls_model(input_tensor)
            nn_class_idx = torch.argmax(cls_output, dim=1).item()
            nn_status = STATUS_MAPPING[nn_class_idx]
            
        # 2. Machine Learning Models (Numpy/joblib)
        input_array = np.array([inputs], dtype=np.float32)
        
        # Ordinal Logistic Regression
        olr_probs = olr_model.model.predict(olr_model.params, exog=input_array)
        olr_class_idx = int(np.argmax(olr_probs, axis=1)[0])
        olr_status = STATUS_MAPPING[olr_class_idx]
        
        # Random Forest
        rf_class_idx = int(rf_model.predict(input_array)[0])
        rf_status = STATUS_MAPPING[rf_class_idx]
        
        # SVM
        svm_class_idx = int(svm_model.predict(input_array)[0])
        svm_status = STATUS_MAPPING[svm_class_idx]
        
        # XGBoost
        xgb_class_idx = int(xgb_model.predict(input_array)[0])
        xgb_status = STATUS_MAPPING[xgb_class_idx]
        
        return (
            f"{predicted_days:.2f} Days", 
            nn_status,
            olr_status,
            rf_status,
            svm_status,
            xgb_status
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "Error", "Error", "Error", "Error", "Error", "Error"


def predict_review_ml(f1, f2, f3, f4, f5, f6, f7, f8):
    """Takes 8 numeric features and predicts the review score using ML models."""
    try:
        inputs = [f1, f2, f3, f4, f5, f6, f7, f8] 
        input_array = np.array([inputs], dtype=np.float32)
        
        # Ordinal Logistic Regression
        olr_probs = rev_olr_model.model.predict(rev_olr_model.params, exog=input_array)
        olr_class_idx = int(np.argmax(olr_probs, axis=1)[0])
        olr_score = REVIEW_SCORE_MAPPING[olr_class_idx]
        
        # Random Forest
        rf_class_idx = int(rev_rf_model.predict(input_array)[0])
        rf_score = REVIEW_SCORE_MAPPING[rf_class_idx]
        
        # SVM
        svm_class_idx = int(rev_svm_model.predict(input_array)[0])
        svm_score = REVIEW_SCORE_MAPPING[svm_class_idx]
        
        # XGBoost
        xgb_class_idx = int(rev_xgb_model.predict(input_array)[0])
        xgb_score = REVIEW_SCORE_MAPPING[xgb_class_idx]
        
        return olr_score, rf_score, svm_score, xgb_score
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}", "Error", "Error", "Error"



with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("## 📦 Olist Multi-Task Operations")
    
    # Wrap ALL tabs inside a single gr.Tabs() block
    with gr.Tabs():
        
        # ==============================================================
        # TAB 1: Analysis Dashboard
        # ==============================================================
        with gr.Tab("📊 Analysis Dashboard"):
            gr.Markdown("### 📈 Interactive E-Commerce Performance Metrics")
            
            # KPI Row
            with gr.Row():
                total_orders = len(df_pbi) if not df_pbi.empty else 0
                total_rev = df_pbi['price'].sum() if not df_pbi.empty else 0
                avg_score = df_pbi['review_score'].mean() if not df_pbi.empty else 0
                
                gr.Textbox(label="Total Orders", value=f"{total_orders:,}", interactive=False)
                gr.Textbox(label="Total Revenue (R$)", value=f"R$ {total_rev:,.2f}", interactive=False)
                gr.Textbox(label="Average Review Score", value=f"{avg_score:.2f} ⭐", interactive=False)
            
            with gr.Accordion("💰 Sales, Revenue & Customer Segments", open=True):
                with gr.Row():
                    plot_cat_rev = gr.Plot()
                    plot_hour = gr.Plot()
                with gr.Row():
                    plot_rfm = gr.Plot()
                    plot_photos = gr.Plot() # NEW
            
            with gr.Accordion("🚚 Logistics & Geographic Performance", open=True):
                with gr.Row():
                    plot_freight = gr.Plot()
                    plot_delay = gr.Plot() 
                with gr.Row():
                    plot_freight_ratio = gr.Plot() # NEW
                    plot_del_rev = gr.Plot() 
            
            with gr.Accordion("🧠 NLP, Reviews & Impact Analysis", open=True):
                with gr.Row():
                    plot_rev_len = gr.Plot()
                    plot_nlp_time = gr.Plot() # NEW
            
            with gr.Accordion("📊 Underlying Data Distributions & Correlations", open=False):
                with gr.Row():
                    plot_dist = gr.Plot()
                    plot_corr = gr.Plot()
                
            # CRITICAL: Map all 12 outputs in the exact order they are returned from the function
            demo.load(fn=generate_dashboard_plots, inputs=[], 
                      outputs=[
                          plot_dist, plot_corr, plot_rfm, plot_del_rev, 
                          plot_cat_rev, plot_freight, plot_hour, plot_delay, 
                          plot_rev_len, plot_freight_ratio, plot_nlp_time, plot_photos
                      ])
        
        # ==============================================================
        # TAB 2: Existing Review Analyzer
        # ==============================================================
        with gr.Tab("Review Analyzer"):
            gr.Markdown("### 📝 Simplified Olist Review Analyzer")
            
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(choices=list(MODEL_REGISTRY.keys()), value=list(MODEL_REGISTRY.keys())[0], label="Select Model")
                    review_input   = gr.Textbox(lines=5, placeholder="Paste a customer review here...", label="Customer Review")
                    analyze_btn    = gr.Button("Analyze Review", variant="primary")
                
                with gr.Column():
                    score_output = gr.HTML(label="Prediction")
                    conf_output  = gr.Label(num_top_classes=5, label="Confidence Breakdown")

            # Click event for the review analyzer
            analyze_btn.click(
                fn=analyze_review, 
                inputs=[review_input, model_dropdown], 
                outputs=[score_output, conf_output]
            )

        # ==============================================================
        # TAB 3: Delivery Prediction (Regression & Classification)
        # ==============================================================
        with gr.Tab("Delivery Prediction"):
            gr.Markdown("### 🚚 Delivery Days & Status Estimator")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Enter the scaled/numeric features from your processed dataset:**")
                    
                    # The exact 7 features remaining after dropping targets
                    f1 = gr.Number(label="Price", value=0.0)
                    f2 = gr.Number(label="Freight Value", value=0.0)
                    f3 = gr.Number(label="Product Weight (g)", value=0.0)
                    f4 = gr.Number(label="Product Length (cm)", value=0.0)
                    f5 = gr.Number(label="Product Height (cm)", value=0.0)
                    f6 = gr.Number(label="Product Width (cm)", value=0.0)
                    f7 = gr.Number(label="Delivery Delay Days", value=0.0)
                    
                    predict_delivery_btn = gr.Button("Predict Delivery", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 📈 Model Predictions")
                    
                    # All 6 Model Outputs
                    days_output = gr.Textbox(label="NN Regression (Predicted Days)", text_align="center")
                    nn_status_output = gr.Textbox(label="Neural Network Classification", text_align="center")
                    olr_status_output = gr.Textbox(label="Ordinal Logistic Regression", text_align="center")
                    rf_status_output = gr.Textbox(label="Random Forest", text_align="center")
                    svm_status_output = gr.Textbox(label="Support Vector Machine (SVM)", text_align="center")
                    xgb_status_output = gr.Textbox(label="XGBoost", text_align="center")
            
            # Connect the button to the prediction function and outputs
            predict_delivery_btn.click(
                fn=predict_delivery,
                inputs=[f1, f2, f3, f4, f5, f6, f7],
                outputs=[
                    days_output, 
                    nn_status_output, 
                    olr_status_output, 
                    rf_status_output, 
                    svm_status_output, 
                    xgb_status_output
                ]
            )

        # ==============================================================
        # TAB 4: Review Prediction (Numeric ML Models)
        # ==============================================================
        with gr.Tab("Review Prediction (ML)"):
            gr.Markdown("### ⭐ Predict Customer Review Score using Numeric Features")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Enter the scaled/numeric features from your review dataset:**")
                    
                    # The exact 8 features from olist_ml_processed_review_dataset.csv
                    r_f1 = gr.Number(label="Price", value=0.0)
                    r_f2 = gr.Number(label="Freight Value", value=0.0)
                    r_f3 = gr.Number(label="Product Weight (g)", value=0.0)
                    r_f4 = gr.Number(label="Product Length (cm)", value=0.0)
                    r_f5 = gr.Number(label="Product Height (cm)", value=0.0)
                    r_f6 = gr.Number(label="Product Width (cm)", value=0.0)
                    r_f7 = gr.Number(label="Actual Delivery Days", value=0.0)
                    r_f8 = gr.Number(label="Delivery Delay Days", value=0.0)
                    
                    predict_review_ml_btn = gr.Button("Predict Review Score", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 📈 ML Model Predictions")
                    
                    rev_olr_output = gr.Textbox(label="Ordinal Logistic Regression", text_align="center")
                    rev_rf_output = gr.Textbox(label="Random Forest", text_align="center")
                    rev_svm_output = gr.Textbox(label="Support Vector Machine (SVM)", text_align="center")
                    rev_xgb_output = gr.Textbox(label="XGBoost", text_align="center")
            
            # Connect the button to the prediction function with all 8 inputs
            predict_review_ml_btn.click(
                fn=predict_review_ml,
                inputs=[r_f1, r_f2, r_f3, r_f4, r_f5, r_f6, r_f7, r_f8],
                outputs=[rev_olr_output, rev_rf_output, rev_svm_output, rev_xgb_output]
            )


if __name__ == "__main__":
    demo.launch()