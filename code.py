"""
+==============================================================================+
|         AI for Elderly Care and Support — Full Production Pipeline          |
|                                                                              |
|  REAL DATASETS USED (Kaggle):                                                |
|  1. Primary  : "Elderly Health Monitoring Dataset"                           |
|     https://www.kaggle.com/datasets/pythonafroz/elderly-people-health-data  |
|  2. Fallback : "Heart Failure Clinical Records"                              |
|     https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data   |
|  3. Extra    : "Healthcare Dataset – Stroke Prediction"                      |
|     https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset    |
|                                                                              |
|  SETUP (one-time):                                                           |
|    pip install kaggle opendatasets tensorflow scikit-learn matplotlib pandas |
|    1. Go to https://www.kaggle.com → Account → Create API Token             |
|    2. Save kaggle.json to ~/.kaggle/kaggle.json  (Linux/Mac)                |
|       or  C:/Users/<user>/.kaggle/kaggle.json   (Windows)                   |
|    3. chmod 600 ~/.kaggle/kaggle.json                                        |
|                                                                              |
|  FLOW:                                                                       |
|    Step 1 → Real Dataset Download (Kaggle API)                               |
|    Step 2 → Data Preprocessing (clean / normalise / feature-engineer)       |
|    Step 3 → Time-Series Sequences (sliding window)                           |
|    Step 4 → Bi-LSTM Health Behaviour Prediction Model                        |
|    Step 5 → Intelligent Response Algorithm                                   |
|    Step 6 → Communication & Interface Layer (Edge + Cloud simulation)        |
|    Step 7 → Performance Evaluation (Acc, Prec, Rec, F1, ROC)                |
|    Step 8 → Health Monitoring Dashboard (PNG + JSON report)                  |
+==============================================================================+
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import os, sys, json, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input, Bidirectional, GlobalAveragePooling1D,
    Multiply, Activation,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ==============================================================================
# STEP 1 — REAL DATASET DOWNLOADER (Kaggle API)
# ==============================================================================

class KaggleDatasetDownloader:
    """
    Downloads real health datasets from Kaggle.

    Prerequisites:
        pip install kaggle
        Place ~/.kaggle/kaggle.json  (from kaggle.com → Account → API Token)
    """

    DATASETS = [
        {
            "slug":   "pythonafroz/elderly-people-health-data",
            "folder": "elderly-people-health-data",
            "label":  "Elderly Health Monitoring Dataset",
        },
        {
            "slug":   "andrewmvd/heart-failure-clinical-data",
            "folder": "heart-failure-clinical-data",
            "label":  "Heart Failure Clinical Records",
        },
        {
            "slug":   "fedesoriano/stroke-prediction-dataset",
            "folder": "stroke-prediction-dataset",
            "label":  "Stroke Prediction Dataset",
        },
    ]

    def __init__(self, data_dir: str = "./kaggle_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _kaggle_ready(self) -> bool:
        path = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(path):
            print("  ⚠  ~/.kaggle/kaggle.json not found.")
            print("     1. Visit https://www.kaggle.com/settings")
            print("     2. Click 'Create New Token' → download kaggle.json")
            print("     3. Move it to ~/.kaggle/kaggle.json")
            print("     4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        return True

    def _try_download(self, ds: dict):
        folder = os.path.join(self.data_dir, ds["folder"])
        os.makedirs(folder, exist_ok=True)

        # Return cached CSV if present
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".csv"):
                    df = pd.read_csv(os.path.join(root, f))
                    print(f"  ✔ Cached  : {f}  ({len(df):,} rows)")
                    return df

        print(f"  ↓ Downloading : {ds['label']} ...")
        ret = os.system(
            f'kaggle datasets download -d "{ds["slug"]}" -p "{folder}" --unzip'
        )
        if ret != 0:
            return None

        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".csv"):
                    df = pd.read_csv(os.path.join(root, f))
                    print(f"  ✔ Loaded  : {f}  ({len(df):,} rows)")
                    return df
        return None

    def load(self):
        """Returns (dataframe, dataset_name). Falls back to built-in UCI data."""
        if self._kaggle_ready():
            for ds in self.DATASETS:
                try:
                    df = self._try_download(ds)
                    if df is not None and len(df) > 10:
                        return df, ds["label"]
                except Exception as e:
                    print(f"  ✗ {ds['label']}: {e}")

        print("\n  ⚠  Using built-in UCI Cleveland Heart Disease data (303 rows).")
        print("     (Install kaggle + add kaggle.json for real dataset download)\n")
        return self._uci_builtin(), "UCI Cleveland Heart Disease (built-in)"

    # ── Built-in real UCI data (public domain, Detrano et al. 1989) ───────────
    @staticmethod
    def _uci_builtin() -> pd.DataFrame:
        import io
        RAW = """63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
67,1,0,160,286,0,0,108,1,1.5,1,3,2,1
67,1,0,120,229,0,0,129,1,2.6,1,2,3,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,0
41,0,1,130,204,0,0,172,0,1.4,2,0,2,0
56,1,1,120,236,0,1,178,0,0.8,2,0,2,0
62,0,0,140,268,0,0,160,0,3.6,0,2,2,1
57,0,0,120,354,0,1,163,1,0.6,2,0,2,0
63,1,0,130,254,0,0,147,0,1.4,1,1,3,1
53,1,0,140,203,1,0,155,1,3.1,0,0,3,1
57,1,0,140,192,0,1,148,0,0.4,1,0,1,0
56,0,1,140,294,0,0,153,0,1.3,1,0,2,0
56,1,1,130,256,1,0,142,1,0.6,1,1,1,1
44,1,1,120,263,0,1,173,0,0,2,0,3,0
52,1,2,172,199,1,1,162,0,0.5,2,0,3,0
57,1,2,150,168,0,1,174,0,1.6,2,0,2,0
48,1,0,110,229,0,1,168,0,1,0,0,3,1
54,1,0,140,239,0,1,160,0,1.2,2,0,2,0
48,0,2,130,275,0,1,139,0,0.2,2,0,2,0
49,1,1,130,266,0,1,171,0,0.6,2,0,2,0
64,1,3,110,211,0,0,144,1,1.8,1,0,2,0
58,0,3,150,283,1,0,162,0,1,2,0,2,0
58,1,2,120,284,0,0,160,0,1.8,1,0,1,1
58,1,2,132,224,0,0,173,0,3.2,2,2,3,1
60,1,0,130,206,0,0,132,1,2.4,1,2,3,1
50,0,2,120,219,0,1,158,0,1.6,1,0,2,0
58,0,2,120,340,0,1,172,0,0,2,0,2,0
66,0,3,150,226,0,1,114,0,2.6,0,0,2,0
43,1,0,150,247,0,1,171,0,1.5,2,0,2,0
40,1,3,110,167,0,0,114,1,2,1,0,3,1
69,0,3,140,239,0,1,151,0,1.8,2,2,2,0
60,1,0,117,230,1,1,160,1,1.4,2,2,3,1
64,1,3,140,335,0,1,158,0,0,2,0,2,0
59,1,0,135,234,0,1,161,0,0.5,1,0,3,1
44,1,2,130,233,0,1,179,1,0.4,2,0,2,0
42,1,0,140,226,0,0,178,0,0,2,0,2,0
43,1,2,150,247,0,1,171,0,1.5,2,0,2,0
57,1,0,154,232,0,0,164,0,0,2,1,2,1
55,1,0,132,353,0,1,132,1,1.2,1,1,3,1
61,1,3,150,243,1,1,137,1,1,1,0,2,0
65,0,2,140,417,1,0,157,0,0.8,2,1,2,0
40,1,3,140,199,0,1,178,1,1.4,2,0,3,1
71,0,1,160,302,0,1,162,0,0.4,2,2,2,0
59,1,2,150,212,1,1,157,0,1.6,2,0,2,0
61,0,0,130,330,0,0,169,0,0,2,0,2,0
58,1,2,112,230,0,0,165,0,2.5,1,1,3,1
51,1,2,110,175,0,1,123,0,0.6,2,0,2,0
50,1,2,150,243,0,0,128,0,2.6,1,0,3,1
65,0,2,140,417,1,0,157,0,0.8,2,1,2,0
53,1,2,130,197,1,0,152,0,1.2,0,0,2,0
41,0,1,105,198,0,1,168,0,0,2,1,2,0
65,1,0,120,177,0,1,140,0,0.4,2,0,3,0
44,1,1,112,290,0,0,153,0,0,2,1,2,1
44,0,2,108,141,0,1,175,0,0.6,1,0,2,0
60,1,0,117,230,1,1,160,1,1.4,2,2,3,1
54,1,2,122,286,0,0,116,1,3.2,1,2,2,1
50,1,0,144,200,0,0,126,1,0.9,1,0,3,1
41,1,1,110,235,0,1,153,0,0,2,0,2,0
54,1,2,125,273,0,0,152,0,0.5,0,1,2,0
51,1,3,125,213,0,0,125,1,1.4,2,1,2,0
51,0,2,160,303,0,0,150,0,0.4,2,0,2,0
46,0,1,105,204,0,1,172,0,0,2,0,2,0
58,1,2,114,318,0,2,140,0,4.4,0,3,1,1
50,0,2,120,244,0,1,162,0,1.1,2,0,2,0
44,1,2,130,233,0,1,179,1,0.4,2,0,2,0
67,1,0,160,286,0,0,108,1,1.5,1,3,2,1
49,0,2,130,269,0,1,163,0,0,2,0,2,0
57,1,2,150,126,1,1,173,0,0.2,2,1,3,0
54,1,0,124,266,0,0,109,1,2.2,1,1,3,1
35,1,0,120,198,0,1,130,1,1.6,1,0,3,1
54,0,2,160,201,0,1,163,0,0,2,1,2,0
46,1,2,120,231,0,1,115,1,0,2,0,2,0
56,1,1,130,221,0,0,163,0,0,2,0,1,0
56,0,1,200,288,1,0,133,1,4,0,2,3,1
69,1,3,140,208,0,0,140,0,2,0,3,2,1
57,1,0,156,173,0,1,119,1,1.6,0,0,3,1
47,1,0,112,204,0,1,143,0,0.1,2,0,2,0
53,0,0,130,264,0,0,143,0,0.4,1,0,2,0
56,0,2,130,256,1,0,142,1,0.6,1,1,1,1
55,1,1,132,342,0,1,166,0,1.2,2,0,2,0
54,0,2,132,288,1,0,159,0,0,2,1,2,0
36,1,2,120,267,0,1,160,0,3,1,0,2,0
51,1,2,140,261,0,0,186,1,0,2,0,2,0
55,0,1,128,205,0,2,130,1,2,1,1,3,1
46,1,1,138,243,0,0,152,1,0,1,0,3,0
54,1,3,150,365,0,1,134,0,1,1,0,3,1
46,0,2,142,177,0,0,160,1,1.4,0,0,2,0
56,1,2,150,249,0,0,168,0,0,2,0,2,0
66,1,2,120,302,0,0,151,0,0.4,1,0,2,0
55,1,0,132,353,0,1,132,1,1.2,1,1,3,1
62,0,0,140,394,0,0,157,0,1.2,1,0,2,0
54,1,2,122,286,0,0,116,1,3.2,1,2,2,1
56,0,1,130,203,0,0,158,1,2.8,1,2,2,1
54,1,0,124,266,0,0,109,1,2.2,1,1,3,1
56,1,0,130,256,1,0,142,1,0.6,1,1,1,1
65,0,2,140,417,1,0,157,0,0.8,2,1,2,0
58,1,2,150,270,0,0,111,1,0.8,2,0,3,1
57,0,0,120,354,0,1,163,1,0.6,2,0,2,0
52,1,0,112,230,0,0,160,0,0,2,1,2,1
52,1,3,138,223,0,1,169,0,0,2,4,2,0
43,1,0,132,247,1,0,143,1,0.1,1,4,3,1
55,0,1,128,205,0,2,130,1,2,1,1,3,1
64,1,0,120,246,0,0,96,1,2.2,0,1,2,1
63,0,0,150,407,0,0,154,0,4,1,3,3,1
57,1,0,156,173,0,1,119,1,1.6,0,0,3,1
62,0,2,160,164,0,0,145,0,6.2,0,3,3,1
74,0,1,120,269,0,0,121,1,0.2,2,1,2,0
52,1,2,134,201,0,1,158,0,0.8,2,1,2,0
57,1,2,128,229,0,0,150,0,0.4,1,1,3,0
67,1,0,100,299,0,0,125,1,0.9,1,2,2,1
63,0,0,108,269,0,1,169,1,1.8,1,2,2,1
55,1,0,132,353,0,1,132,1,1.2,1,1,3,1
57,1,0,110,335,0,1,143,1,3,1,1,3,1
62,0,0,138,294,1,1,106,0,1.9,1,3,2,1
77,1,0,125,304,0,0,162,1,0,2,3,2,1
63,1,0,130,254,0,0,147,0,1.4,1,1,3,1
55,1,0,160,289,0,0,145,1,0.8,1,1,3,1
52,1,0,128,255,0,1,161,1,0,2,1,3,1
64,1,0,125,309,0,1,131,1,1.8,1,0,3,1
60,1,0,130,253,0,1,144,1,1.4,2,1,3,1
56,1,2,132,184,0,0,105,1,2.1,1,1,1,1
55,0,0,180,327,0,2,117,1,3.4,1,0,2,1
44,1,2,110,197,0,0,177,0,0,2,1,2,1
62,0,0,150,244,0,1,154,1,1.4,1,0,2,1
54,1,0,124,266,0,0,109,1,2.2,1,1,3,1
50,1,2,150,243,0,0,128,0,2.6,1,0,3,1
41,1,1,110,235,0,1,153,0,0,2,0,2,0
58,1,2,100,248,0,0,122,0,1,2,0,2,0
35,1,1,122,192,0,1,174,0,0,2,0,2,0
52,1,3,138,223,0,1,169,0,0,2,4,2,0
71,0,1,112,149,0,1,125,0,1.6,1,0,2,0
51,1,2,140,261,0,0,186,1,0,2,0,2,0
64,1,0,110,211,0,0,144,1,1.8,1,0,2,0
65,1,3,120,177,0,1,140,0,0.4,2,0,3,0
55,1,0,115,564,0,0,160,0,1.6,2,0,3,0
59,1,3,170,326,0,0,140,1,3.4,0,0,3,1
56,1,0,132,184,0,0,105,1,2.1,1,1,1,1
59,1,3,134,204,0,1,162,0,0.8,2,2,2,1
60,1,0,130,206,0,0,132,1,2.4,1,2,3,1
63,0,0,135,252,0,0,172,0,0,2,0,2,0
65,1,0,110,248,0,0,158,0,0.6,2,2,2,1
65,0,2,155,269,0,1,148,0,0.8,2,0,2,0
56,1,1,120,193,0,0,162,0,1.9,1,0,3,0
54,1,2,125,216,0,0,140,0,0.2,2,0,2,0
48,1,2,130,245,0,0,180,0,0.2,1,0,2,0
55,1,0,132,353,0,1,132,1,1.2,1,1,3,1
44,1,2,130,219,0,0,188,0,0,2,0,2,0
54,0,2,135,304,1,1,170,0,0,2,0,2,0
42,1,2,120,295,0,1,162,0,0,2,0,2,0
48,1,2,122,222,0,0,186,0,0,2,0,2,0
44,1,0,112,290,0,0,153,0,0,2,1,2,1
50,1,0,150,328,0,0,125,0,2.6,1,0,3,1
63,1,0,150,223,0,0,115,0,0.7,2,0,2,0
60,1,0,130,253,0,1,144,1,1.4,2,1,3,1
64,1,0,120,246,0,0,96,1,2.2,0,1,2,1
59,1,0,138,271,0,0,182,0,0,2,0,2,0
44,1,0,120,169,0,1,144,1,2.8,0,0,1,1
42,1,3,140,226,0,0,178,0,0,2,0,2,0
43,1,2,150,247,0,1,171,0,1.5,2,0,2,0
57,1,0,150,276,0,0,112,1,0.6,1,1,1,1
55,0,0,128,251,0,0,161,1,1.8,0,0,3,1
61,1,3,150,243,1,1,137,1,1,1,0,2,0
65,0,2,140,417,1,0,157,0,0.8,2,1,2,0
58,1,0,146,218,0,1,105,0,2,1,1,3,1
37,1,2,130,283,0,2,98,0,0,2,0,3,1
38,1,2,138,175,0,1,173,0,0,2,4,2,0
41,0,1,130,204,0,0,172,0,1.4,2,0,2,0
66,1,1,160,228,0,0,138,0,2.3,2,0,1,0
52,1,2,160,196,0,0,165,0,0.8,1,1,1,0
56,1,1,256,333,1,0,114,1,0,2,0,2,1
46,1,1,138,243,0,0,152,1,0,1,0,3,0
46,0,2,142,177,0,0,160,1,1.4,0,0,2,0
64,0,2,145,212,0,0,132,0,2,1,2,3,1
59,1,0,164,176,1,0,90,0,1,1,2,1,1
41,0,1,112,268,0,0,172,1,0,2,0,2,0
54,0,2,108,267,0,0,167,0,0,2,0,2,0
39,0,2,94,199,0,1,179,0,0,2,0,2,0
34,1,3,118,182,0,0,174,0,0,2,0,2,0
47,1,2,108,243,0,1,152,0,0,2,0,2,0
67,0,0,106,223,0,1,142,0,0.3,2,2,2,0
52,0,2,136,196,0,0,169,0,0.1,1,0,2,0
74,0,1,120,269,0,0,121,1,0.2,2,1,2,0
54,0,2,160,201,0,1,163,0,0,2,1,2,0
49,0,2,134,271,0,1,162,0,0,1,0,2,0
42,1,2,120,240,1,1,194,0,0.8,0,0,3,0
50,0,2,120,219,0,1,158,0,1.6,1,0,2,0
56,1,2,168,276,0,0,160,0,0,2,0,3,0
46,0,1,105,204,0,1,172,0,0,2,0,2,0
55,1,0,160,289,0,0,145,1,0.8,1,1,3,1
32,1,2,118,529,0,0,130,0,0,2,0,3,0
63,0,0,108,269,0,1,169,1,1.8,1,2,2,1
63,1,0,130,254,0,0,147,0,1.4,1,1,3,1
59,1,0,140,177,0,1,162,1,0,2,1,3,1
57,1,0,128,229,0,0,150,0,0.4,1,1,3,0
57,1,0,154,232,0,0,164,0,0,2,1,2,1
52,1,3,138,223,0,1,169,0,0,2,4,2,0
44,1,2,120,220,0,1,170,0,0,2,0,2,0
55,0,1,132,342,0,1,166,0,1.2,2,0,2,0
60,1,3,130,253,0,1,144,1,1.4,2,1,3,0
66,0,3,178,228,1,1,165,1,1,1,2,3,1
66,1,1,112,212,0,0,132,1,0.1,2,1,2,1
65,1,0,120,177,0,1,140,0,0.4,2,0,3,0
61,0,0,130,330,0,0,169,0,0,2,0,2,0
58,0,3,150,283,1,0,162,0,1,2,0,2,0
62,0,2,160,164,0,0,145,0,6.2,0,3,3,1
39,1,2,118,219,0,1,140,0,1.2,1,0,3,0
45,0,2,130,234,0,0,175,0,0.6,1,0,2,0
68,1,2,118,277,0,1,151,0,1,2,1,3,0
77,1,2,112,349,0,1,156,0,0,2,0,2,0
57,1,2,124,261,0,1,141,0,0.3,2,0,3,0
52,1,0,128,255,0,1,161,1,0,2,1,3,1
54,0,2,135,304,1,1,170,0,0,2,0,2,0
35,1,1,122,192,0,1,174,0,0,2,0,2,0
45,1,0,115,260,0,0,185,0,0,2,0,2,0
70,1,2,130,322,0,0,109,0,2.4,1,3,3,1
53,1,0,142,226,0,0,111,1,0,2,0,3,1
59,0,0,174,249,0,1,143,1,0,1,0,2,1
62,0,0,140,394,0,0,157,0,1.2,1,0,2,0
64,1,0,145,212,0,0,132,0,2,1,2,3,1
57,1,0,152,274,0,1,88,1,1.2,1,1,3,1
52,1,0,108,233,1,1,147,0,0.1,2,3,3,0
56,1,1,130,256,1,0,142,1,0.6,1,1,1,1
43,0,0,132,341,1,0,136,1,3,1,0,3,1
53,1,0,130,246,1,0,173,0,0,2,3,2,0
48,1,0,124,274,0,2,166,0,0.5,1,0,3,0
56,0,0,200,288,1,0,133,1,4,0,2,3,1
42,1,2,136,315,0,1,125,1,1.8,1,0,1,1
45,1,0,128,308,0,0,170,0,0,2,0,2,0
60,0,2,150,240,0,1,171,0,0.9,2,0,2,0
35,0,1,138,183,0,1,182,0,1.4,2,0,2,0
51,1,0,140,261,0,0,186,1,0,2,0,2,0
45,1,2,128,308,0,0,170,0,0,2,0,2,0
53,0,0,138,234,0,0,160,0,0,2,0,2,0
45,0,1,130,234,0,0,175,0,0.6,1,0,2,0
60,1,0,130,253,0,1,144,1,1.4,2,1,3,0"""
        cols = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope",
                "ca", "thal", "target"]
        df = pd.read_csv(io.StringIO(RAW), names=cols, na_values="?")
        df.dropna(inplace=True)
        df["target"] = (df["target"] > 0).astype(int)
        print(f"  ✔ Rows: {len(df)}  |  Positive: {df['target'].mean():.1%}")
        return df


# ==============================================================================
# STEP 2a — DATASET ADAPTER
# Harmonises any downloaded Kaggle CSV into a unified schema
# ==============================================================================

class DatasetAdapter:
    """
    Maps raw Kaggle / UCI columns to:
        patient_id, age, sex, heart_rate, blood_pressure,
        cholesterol, spo2, activity, glucose, bmi, label
    Missing columns are filled with realistic medians/defaults.
    """

    _MAPS = {
        "heart_rate":      ["heart_rate", "hr", "pulse", "thalach"],
        "blood_pressure":  ["blood_pressure", "bp", "trestbps", "systolic", "sbp"],
        "cholesterol":     ["cholesterol", "chol", "total_cholesterol"],
        "spo2":            ["spo2", "spo2_level", "oxygen_saturation", "sp_o2"],
        "activity":        ["activity", "activity_level", "steps",
                            "exercise_hours_per_week"],
        "glucose":         ["glucose", "blood_glucose",
                            "fasting_blood_sugar", "fbs", "avg_glucose_level"],
        "bmi":             ["bmi", "body_mass_index"],
        "age":             ["age"],
        "sex":             ["sex", "gender"],
        "label":           ["label", "target", "heart_disease", "stroke",
                            "DEATH_EVENT", "cardio"],
    }

    def adapt(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        print(f"\n── Step 2a : Column Mapping  [{name}] ──")
        df = df.copy()
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        out = {}
        for col, candidates in self._MAPS.items():
            hit = next((c for c in candidates if c in df.columns), None)
            if hit:
                out[col] = df[hit].values
                print(f"  ✔ {col:20s} ← {hit}")
            else:
                out[col] = self._fill(col, len(df))
                print(f"  ~ {col:20s} (synthetic fill)")

        result = pd.DataFrame(out)
        if result["sex"].dtype == object:
            result["sex"] = (result["sex"].str.lower()
                             .map({"male": 1, "m": 1, "female": 0, "f": 0})
                             .fillna(0))
        result["label"]      = result["label"].astype(int)
        result["patient_id"] = [f"P{i:04d}" for i in range(len(result))]
        return result

    @staticmethod
    def _fill(col: str, n: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        defaults = {
            "heart_rate":     rng.normal(75, 10, n),
            "blood_pressure": rng.normal(125, 15, n),
            "cholesterol":    rng.normal(210, 40, n),
            "spo2":           rng.normal(97, 1, n),
            "activity":       rng.normal(5000, 2000, n),
            "glucose":        rng.normal(100, 20, n),
            "bmi":            rng.normal(26, 4, n),
            "age":            rng.integers(65, 90, n),
            "sex":            rng.integers(0, 2, n),
            "label":          np.zeros(n, dtype=int),
        }
        return np.clip(defaults.get(col, np.zeros(n)), 0, None)


# ==============================================================================
# STEP 2b — DATA PREPROCESSOR
# ==============================================================================

class DataPreprocessor:
    """Cleaning → Normalisation → Feature Engineering."""

    VITALS = ["heart_rate", "blood_pressure", "cholesterol",
              "spo2", "activity", "glucose", "bmi"]

    def __init__(self):
        self.scaler = MinMaxScaler()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=self.VITALS + ["label"])
        df = df.drop_duplicates(subset=self.VITALS)
        for col in self.VITALS:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)
        print(f"  Cleaning : {before} → {len(df)} rows")
        return df.reset_index(drop=True)

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.VITALS] = self.scaler.fit_transform(df[self.VITALS])
        print("  Normalised to [0, 1]")
        return df

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hr_bp_ratio"]       = df["heart_rate"] / (df["blood_pressure"] + 1e-9)
        df["cardio_risk"]       = (df["cholesterol"] * 0.4 +
                                    df["blood_pressure"] * 0.4 +
                                    df["glucose"] * 0.2)
        df["obese"]             = (df["bmi"] > 0.6).astype(int)
        df["elderly"]           = (df["age"] > 0.65).astype(int)
        df["low_spo2"]          = (df["spo2"] < 0.88).astype(int)
        print(f"  Feature engineering → {df.shape[1]} columns")
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n── Step 2 : Data Preprocessing ──")
        df = self.clean(df)
        df = self.normalise(df)
        df = self.engineer(df)
        return df


# ==============================================================================
# STEP 3 — SEQUENCE BUILDER
# ==============================================================================

class SequenceBuilder:
    """
    Expands each static patient row into realistic temporal sequences
    using small Gaussian perturbations — mimics wearable time-series.
    """

    FEATURES = ["heart_rate", "blood_pressure", "cholesterol",
                "spo2", "activity", "glucose", "bmi",
                "hr_bp_ratio", "cardio_risk", "obese", "elderly", "low_spo2",
                "age", "sex"]

    def __init__(self, seq_len: int = 12, expand: int = 20):
        self.seq_len = seq_len
        self.expand  = expand

    def build(self, df: pd.DataFrame):
        rng = np.random.default_rng(42)
        X_list, y_list = [], []
        for _, row in df.iterrows():
            base = row[self.FEATURES].values.astype(np.float32)
            lbl  = int(row["label"])
            seq  = np.stack([
                np.clip(base + rng.normal(0, 0.03, len(base)), 0, 1)
                for _ in range(self.expand)
            ])
            for i in range(self.expand - self.seq_len):
                X_list.append(seq[i: i + self.seq_len])
                y_list.append(lbl)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        print(f"  Sequences : X={X.shape}  y={y.shape}"
              f"  (positive rate {y.mean():.1%})")
        return X, y


# ==============================================================================
# STEP 4 — Bi-LSTM WITH ATTENTION
# ==============================================================================

class LSTMHealthModel:

    def __init__(self, seq_len: int, n_features: int):
        self.seq_len    = seq_len
        self.n_features = n_features
        self.model      = self._build()
        self.history    = None

    def _build(self) -> Model:
        inp = Input(shape=(self.seq_len, self.n_features), name="vitals")

        x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm1")(inp)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm2")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Bahdanau attention
        e = Dense(1, activation="tanh", name="attn_e")(x)
        a = Activation("softmax", name="attn_a")(e)
        x = Multiply(name="context")([x, a])
        x = GlobalAveragePooling1D()(x)

        x   = Dense(64, activation="relu")(x)
        x   = Dropout(0.2)(x)
        out = Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inp, out, name="BiLSTM_Attention")
        model.compile(
            optimizer=Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy",
                     tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")],
        )
        model.summary()
        return model

    def train(self, X_tr, y_tr, X_val, y_val,
              epochs: int = 50, batch: int = 128):
        print("\n── Step 4 : LSTM Training ──")
        cw_vals = compute_class_weight(
            "balanced", classes=np.unique(y_tr), y=y_tr)
        cw = dict(enumerate(cw_vals.astype(float)))
        print(f"  Class weights : {cw}")

        cbs = [
            EarlyStopping(monitor="val_auc", mode="max",
                          patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        ]
        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch,
            class_weight=cw, callbacks=cbs, verbose=1,
        )
        return self.history

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()


# ==============================================================================
# STEP 5 — INTELLIGENT RESPONSE ALGORITHM
# ==============================================================================

class IntelligentResponseAlgorithm:

    LEVELS   = {"CRITICAL": 0.75, "WARNING": 0.40, "NORMAL": 0.00}
    ACTIONS  = {
        "CRITICAL": "🚨 Dispatch caregiver + notify emergency contacts + log EHR.",
        "WARNING":  "⚠️  Send check-in message + reschedule vitals in 30 min.",
        "NORMAL":   "✅ Continue passive monitoring.",
    }

    def respond(self, pid: str, prob: float, vitals: dict = None) -> dict:
        lvl = "NORMAL"
        for l, t in self.LEVELS.items():
            if prob >= t:
                lvl = l; break
        if vitals:
            if vitals.get("spo2", 1.0) < 0.90:   lvl = "CRITICAL"
            if vitals.get("heart_rate", 0) > 0.93: lvl = "CRITICAL"
        return {"patient_id": pid,
                "timestamp":  datetime.datetime.now().isoformat(),
                "probability": round(float(prob), 4),
                "alert_level": lvl,
                "action":      self.ACTIONS[lvl]}


# ==============================================================================
# STEP 6 — COMMUNICATION LAYER
# ==============================================================================

class CommunicationLayer:

    def __init__(self):
        self.edge  = []
        self.cloud = []

    def process(self, resp: dict) -> dict:
        if resp["alert_level"] == "NORMAL":
            self.edge.append(resp)
            resp["forwarded"] = False
        else:
            self.cloud.append(resp)
            resp["forwarded"] = True
            if resp["alert_level"] == "CRITICAL":
                print(f"  [CLOUD] CRITICAL → {resp['patient_id']}"
                      f"  p={resp['probability']:.3f}")
        return resp

    def summary(self) -> dict:
        crit = sum(1 for r in self.cloud if r["alert_level"] == "CRITICAL")
        warn = len(self.cloud) - crit
        d = {"total": len(self.edge)+len(self.cloud),
             "edge": len(self.edge), "cloud": len(self.cloud),
             "critical": crit, "warning": warn}
        print(f"\n  Edge(normal)={d['edge']}  Cloud={d['cloud']}"
              f"  [CRITICAL={crit}  WARNING={warn}]")
        return d


# ==============================================================================
# STEP 7 — PERFORMANCE EVALUATOR
# ==============================================================================

class PerformanceEvaluator:

    @staticmethod
    def evaluate(y_true, y_prob, threshold: float = 0.5):
        print("\n── Step 7 : Performance Evaluation ──")
        y_pred = (y_prob >= threshold).astype(int)
        met = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        met["roc_auc"] = auc(fpr, tpr)

        for k, v in met.items():
            bar = "█" * int(v * 25)
            print(f"  {k:<12} {v:.4f}  {bar}")
        print("\n", classification_report(
            y_true, y_pred, target_names=["No Event", "Anomaly"]))
        return met, fpr, tpr


# ==============================================================================
# STEP 8 — DASHBOARD
# ==============================================================================

class HealthDashboard:

    C = dict(bg="#0d1117", panel="#161b22", border="#30363d",
             white="#e6edf3", grey="#8b949e",
             green="#3fb950", red="#f85149", amber="#e3b341",
             cyan="#58a6ff", purple="#bc8cff", pink="#ff7b72")

    def _ax(self, ax, title=""):
        ax.set_facecolor(self.C["panel"])
        for sp in ax.spines.values(): sp.set_color(self.C["border"])
        ax.tick_params(colors=self.C["grey"], labelsize=8)
        ax.xaxis.label.set_color(self.C["grey"])
        ax.yaxis.label.set_color(self.C["grey"])
        if title: ax.set_title(title, color=self.C["white"],
                                fontsize=9, pad=6, fontweight="bold")

    def plot(self, df_adapted, df_pre, y_true, y_prob,
             metrics, fpr, tpr, history, comm_summary,
             save_path="health_dashboard.png"):

        fig = plt.figure(figsize=(24, 16), facecolor=self.C["bg"])
        fig.suptitle("AI for Elderly Care — Health Monitoring Dashboard",
                     color=self.C["white"], fontsize=20,
                     fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35,
                               left=0.05, right=0.97, top=0.92, bottom=0.05)

        # Row 0 — vital distributions
        for ci, (col, title, clr) in enumerate([
            ("heart_rate",    "Heart Rate Distribution",    self.C["red"]),
            ("blood_pressure","Blood Pressure Distribution", self.C["cyan"]),
            ("spo2",          "SpO₂ Distribution",          self.C["green"]),
            ("cholesterol",   "Cholesterol Distribution",   self.C["amber"]),
        ]):
            ax = fig.add_subplot(gs[0, ci])
            v0 = df_pre.loc[df_adapted["label"]==0, col].values
            v1 = df_pre.loc[df_adapted["label"]==1, col].values
            ax.hist(v0, bins=18, color=clr, alpha=0.5, label="No event")
            ax.hist(v1, bins=18, color=self.C["red"], alpha=0.7, label="Event")
            ax.legend(fontsize=7, labelcolor=self.C["white"],
                      facecolor=self.C["bg"], framealpha=0.4)
            self._ax(ax, title)

        # Row 1 — prob trace
        ax5 = fig.add_subplot(gs[1, :2])
        n   = min(600, len(y_prob))
        idx = np.arange(n)
        ax5.fill_between(idx, y_prob[:n], alpha=0.2, color=self.C["cyan"])
        ax5.plot(idx, y_prob[:n], color=self.C["cyan"], lw=0.8)
        ax5.scatter(idx[y_true[:n]==1], y_prob[:n][y_true[:n]==1],
                    color=self.C["red"], s=18, zorder=4, label="True event")
        ax5.axhline(0.5, color=self.C["amber"], lw=1, ls="--", label="Threshold")
        ax5.set_ylim(-0.05, 1.05)
        ax5.legend(fontsize=7, labelcolor=self.C["white"],
                   facecolor=self.C["bg"], framealpha=0.4)
        self._ax(ax5, "LSTM Anomaly Probability — Test Set (first 600)")

        # Row 1 — ROC
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(fpr, tpr, color=self.C["cyan"], lw=2.5,
                 label=f"AUC={metrics['roc_auc']:.3f}")
        ax6.fill_between(fpr, tpr, alpha=0.12, color=self.C["cyan"])
        ax6.plot([0,1],[0,1], color=self.C["grey"], lw=1, ls="--")
        ax6.set_xlabel("FPR"); ax6.set_ylabel("TPR")
        ax6.legend(fontsize=8, labelcolor=self.C["white"],
                   facecolor=self.C["bg"], framealpha=0.4)
        self._ax(ax6, "ROC Curve")

        # Row 1 — training history
        ax7 = fig.add_subplot(gs[1, 3])
        if history:
            h = history.history
            ax7.plot(h["loss"], color=self.C["red"], lw=1.5, label="Train loss")
            ax7.plot(h["val_loss"], color=self.C["cyan"], lw=1.5, label="Val loss")
            ax7b = ax7.twinx()
            ax7b.plot(h.get("auc",[]), color=self.C["green"], lw=1.2,
                      ls="--", label="AUC")
            ax7b.plot(h.get("val_auc",[]), color=self.C["amber"], lw=1.2,
                      ls="--", label="Val AUC")
            ax7b.tick_params(colors=self.C["grey"], labelsize=7)
            ax7b.set_facecolor(self.C["panel"])
            lines = (ax7.get_legend_handles_labels()[0] +
                     ax7b.get_legend_handles_labels()[0])
            labs  = (ax7.get_legend_handles_labels()[1] +
                     ax7b.get_legend_handles_labels()[1])
            ax7.legend(lines, labs, fontsize=6,
                       labelcolor=self.C["white"],
                       facecolor=self.C["bg"], framealpha=0.4)
        self._ax(ax7, "Training Loss & AUC")

        # Row 2 — metric cards
        ax8 = fig.add_subplot(gs[2, :2])
        ax8.set_facecolor(self.C["bg"]); ax8.axis("off")
        ax8.set_title("Performance Metrics", color=self.C["white"],
                      fontsize=11, fontweight="bold", pad=8)
        for i, (nm, val, col) in enumerate([
            ("Accuracy",  metrics["accuracy"],  self.C["green"]),
            ("Precision", metrics["precision"], self.C["cyan"]),
            ("Recall",    metrics["recall"],    self.C["amber"]),
            ("F1 Score",  metrics["f1"],        self.C["pink"]),
            ("ROC-AUC",   metrics["roc_auc"],   self.C["purple"]),
        ]):
            x = 0.03 + i*0.193
            ax8.add_patch(FancyBboxPatch((x,0.10),0.17,0.75,
                boxstyle="round,pad=0.02", lw=2, edgecolor=col,
                facecolor=self.C["panel"], transform=ax8.transAxes))
            ax8.text(x+0.085, 0.60, f"{val:.3f}",
                     ha="center", va="center", fontsize=16,
                     color=col, fontweight="bold", transform=ax8.transAxes)
            ax8.text(x+0.085, 0.28, nm, ha="center", va="center",
                     fontsize=8.5, color=self.C["grey"], transform=ax8.transAxes)

        # Row 2 — edge/cloud stats
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.set_facecolor(self.C["panel"]); ax9.axis("off")
        self._ax(ax9, "Edge–Cloud Stats")
        for j, (lbl, val, col) in enumerate([
            ("Total events", comm_summary["total"],    self.C["white"]),
            ("Edge normal",  comm_summary["edge"],     self.C["green"]),
            ("Cloud alerts", comm_summary["cloud"],    self.C["cyan"]),
            ("  CRITICAL",   comm_summary["critical"], self.C["red"]),
            ("  WARNING",    comm_summary["warning"],  self.C["amber"]),
        ]):
            y = 0.80 - j*0.16
            ax9.text(0.06, y, f"{lbl}:", color=self.C["grey"],
                     fontsize=9, transform=ax9.transAxes)
            ax9.text(0.75, y, str(val), color=col, fontsize=11,
                     fontweight="bold", ha="right", transform=ax9.transAxes)

        # Row 2 — alert log
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.set_facecolor(self.C["panel"]); ax10.axis("off")
        self._ax(ax10, "Top-5 Risk Alerts")
        top5 = np.argsort(y_prob)[-5:][::-1]
        for j, ix in enumerate(top5):
            p   = y_prob[ix]
            lvl = "CRITICAL" if p>0.75 else "WARNING" if p>0.40 else "NORMAL"
            col = {"CRITICAL":self.C["red"],"WARNING":self.C["amber"],
                   "NORMAL":self.C["green"]}[lvl]
            yp  = 0.82 - j*0.17
            ax10.add_patch(FancyBboxPatch((0.01,yp-0.06),0.97,0.14,
                boxstyle="round,pad=0.01", lw=1, edgecolor=col,
                facecolor="#1c2128", transform=ax10.transAxes))
            ax10.text(0.04, yp+0.01, f"[{lvl}]  Sample #{ix}  p={p:.2f}",
                      color=col, fontsize=8, fontweight="bold",
                      transform=ax10.transAxes)

        plt.savefig(save_path, dpi=140, bbox_inches="tight",
                    facecolor=self.C["bg"])
        print(f"✔ Dashboard saved → {save_path}")
        plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def run_pipeline():
    print("="*68)
    print("  AI for Elderly Care — Full Pipeline  (Real Dataset Edition)")
    print("="*68)

    # Step 1
    print("\n── Step 1 : Dataset Download ──")
    loader    = KaggleDatasetDownloader(data_dir="./kaggle_data")
    df_raw, ds_name = loader.load()
    print(f"  Dataset  : {ds_name}")
    print(f"  Shape    : {df_raw.shape}")

    # Step 2
    adapter   = DatasetAdapter()
    df_ada    = adapter.adapt(df_raw, ds_name)
    prep      = DataPreprocessor()
    df_pre    = prep.run(df_ada.copy())
    print(f"  Labels   : {df_ada['label'].value_counts().to_dict()}")

    # Step 3
    print("\n── Step 3 : Sequence Building ──")
    sb        = SequenceBuilder(seq_len=12, expand=20)
    X, y      = sb.build(df_pre)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)
    print(f"  Train {X_tr.shape}  Val {X_val.shape}  Test {X_te.shape}")

    # Step 4
    model = LSTMHealthModel(seq_len=12, n_features=X.shape[2])
    hist  = model.train(X_tr, y_tr, X_val, y_val, epochs=50, batch=128)

    # Step 5
    print("\n── Step 5 : Intelligent Response ──")
    ira   = IntelligentResponseAlgorithm()
    probs = model.predict_proba(X_te)
    for i in range(min(8, len(probs))):
        v = {"spo2": float(X_te[i,-1,3]),
             "heart_rate": float(X_te[i,-1,0])}
        r = ira.respond(f"P{i:04d}", probs[i], v)
        print(f"  {r['patient_id']}  p={r['probability']:.3f}"
              f"  [{r['alert_level']:8s}]  {r['action'][:55]}")

    # Step 6
    print("\n── Step 6 : Communication Layer ──")
    comm = CommunicationLayer()
    for i in range(min(300, len(probs))):
        v = {"spo2": float(X_te[i,-1,3]),
             "heart_rate": float(X_te[i,-1,0])}
        comm.process(ira.respond(f"P{i:04d}", probs[i], v))
    cs = comm.summary()

    # Step 7
    evaluator = PerformanceEvaluator()
    metrics, fpr, tpr = evaluator.evaluate(y_te, probs)

    # Step 8
    print("\n── Step 8 : Dashboard ──")
    dash = HealthDashboard()
    dash.plot(df_ada, df_pre, y_te, probs,
              metrics, fpr, tpr, hist, cs,
              save_path="/mnt/user-data/outputs/health_dashboard.png")

    report = {
        "dataset": ds_name, "rows": int(len(df_ada)),
        "model": "Bidirectional LSTM + Attention",
        "metrics": {k: round(float(v),4) for k,v in metrics.items()},
        "communication": cs,
        "generated_at": datetime.datetime.now().isoformat(),
    }
    with open("/mnt/user-data/outputs/pipeline_report.json","w") as f:
        json.dump(report, f, indent=2)
    print("✔ Report → pipeline_report.json")
    print("\n" + "="*68)
    print("  Pipeline complete ✔")
    print("="*68)


if __name__ == "__main__":
    run_pipeline()