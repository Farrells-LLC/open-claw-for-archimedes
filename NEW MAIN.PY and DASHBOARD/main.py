# ============================================================
# ArchimedesMD.com — Real Training Pipeline
# Actual data ingestion, real model training, real inference
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import stripe
import os, json, uuid, shutil, re, io, math, warnings, time
from pathlib import Path
from datetime import datetime

# ── Supabase storage ──
import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BUCKET = "Models"

# ── Contact form email config (Resend) ──
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
CONTACT_FROM_EMAIL = os.environ.get("CONTACT_FROM_EMAIL", "onboarding@resend.dev")  # must be a verified sender/domain in Resend
CONTACT_RECIPIENT = os.environ.get("CONTACT_RECIPIENT", "archimedesmodeldesign@gmail.com")  # where messages land

# ── Report sampling config ──
# Large datasets are sampled before stat computation to stay within LLM
# prompt limits. Set REPORT_SAMPLE_ROWS in Railway to override the default.
REPORT_SAMPLE_ROWS: int = int(os.environ.get("REPORT_SAMPLE_ROWS", "10000"))

# Pure insurance, not a length restriction: the live stress test showed
# generation can stall for 90-120+ seconds while only ~4,000-6,000 tokens in
# (well under max_tokens=20000) — so max_tokens alone doesn't protect against
# a slow-but-short-so-far generation just sitting there. This is a generous
# wall-clock ceiling that only ever fires if something has genuinely gone
# wrong; it does not cap report length, shorten the prompt, or change normal
# report depth in any way. If crossed, whatever was generated gets force-
# closed into valid HTML (see _finalize_truncated_report_html) instead of
# the connection hanging with nothing returned.
REPORT_MAX_GENERATION_SECONDS: float = float(os.environ.get("REPORT_MAX_GENERATION_SECONDS", "300"))

MAX_UPLOAD_BYTES: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_ZIP_MEMBERS: int = int(os.environ.get("MAX_ZIP_MEMBERS", "250"))
MAX_ZIP_UNCOMPRESSED_BYTES: int = int(os.environ.get("MAX_ZIP_UNCOMPRESSED_BYTES", str(250 * 1024 * 1024)))
MAX_ZIP_MEMBER_BYTES: int = int(os.environ.get("MAX_ZIP_MEMBER_BYTES", str(100 * 1024 * 1024)))
MAX_PARSED_ROWS: int = int(os.environ.get("MAX_PARSED_ROWS", "250000"))
MAX_PARSED_COLUMNS: int = int(os.environ.get("MAX_PARSED_COLUMNS", "500"))

# ── Stripe config ──
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://archimedesmd.com")

# Stripe Price IDs — set these after creating products in Stripe dashboard
STRIPE_PRICES = {
    "pro":       os.environ.get("STRIPE_PRICE_PRO", ""),
    "team":      os.environ.get("STRIPE_PRICE_TEAM", ""),
    "unlimited": os.environ.get("STRIPE_PRICE_UNLIMITED", ""),
}

PLAN_NAMES = {
    "pro":       "Pro Plan",
    "team":      "Team Plan",
    "unlimited":  "Unlimited Plan",
}

# In-memory model cache — avoids re-downloading from Supabase on every request
_model_cache: dict = {}

def upload_model_to_supabase(local_path: str, model_id: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{model_id}.pkl"
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/octet-stream",
            "x-upsert": "true"
        }
        resp = httpx.put(url, content=data, headers=headers, timeout=60)
        print(f"Supabase upload status: {resp.status_code}")
        print(f"Supabase upload response: {resp.text[:200]}")
        return resp.status_code in [200, 201]
    except Exception as e:
        print(f"Supabase upload error: {e}")
        return False

def download_model_from_supabase(model_id: str, local_path: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{model_id}.pkl"
        headers = {"Authorization": f"Bearer {SUPABASE_KEY}"}
        resp = httpx.get(url, headers=headers, timeout=60)
        if resp.status_code == 200:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(resp.content)
            return True
        return False
    except Exception as e:
        print(f"Supabase download error: {e}")
        return False

REGISTRY_OBJECT = "_registry/model_registry.json"

def _supabase_put(object_path: str, data: bytes, content_type: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{object_path}"
        headers = {"Authorization": f"Bearer {SUPABASE_KEY}",
                   "Content-Type": content_type, "x-upsert": "true"}
        return httpx.put(url, content=data, headers=headers, timeout=60).status_code in (200, 201)
    except Exception as e:
        print(f"Supabase put error ({object_path}): {e}")
        return False

def _supabase_get(object_path: str):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{object_path}"
        r = httpx.get(url, headers={"Authorization": f"Bearer {SUPABASE_KEY}"}, timeout=60)
        return r.content if r.status_code == 200 else None
    except Exception as e:
        print(f"Supabase get error ({object_path}): {e}")
        return None

# ── ML imports ──
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             mean_squared_error, r2_score, classification_report,
                             confusion_matrix)
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import joblib

app = FastAPI(title="Archimedes MD — Real Training Pipeline", version="2.0.0")

def clean(obj):
    import numpy as np
    import math
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    elif isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Storage ──
MODELS_DIR  = Path("./saved_models");  MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = Path("./uploads");       UPLOADS_DIR.mkdir(exist_ok=True)
BUFFERS_DIR = Path("./data_buffers");  BUFFERS_DIR.mkdir(exist_ok=True)
DATA_DIR    = Path("./training_data"); DATA_DIR.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

REGISTRY_PATH = Path("./model_registry.json")

def load_registry():
    if not REGISTRY_PATH.exists():
        data = _supabase_get(REGISTRY_OBJECT)
        if data:
            REGISTRY_PATH.write_bytes(data)
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_registry(reg):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    _supabase_put(REGISTRY_OBJECT, json.dumps(reg).encode("utf-8"), "application/json")

def buffer_path(model_id): return BUFFERS_DIR / f"{model_id}_buffer.json"
def load_buffer(model_id):
    p = buffer_path(model_id)
    return json.loads(p.read_text()) if p.exists() else []
def save_buffer(model_id, buf): buffer_path(model_id).write_text(json.dumps(buf, indent=2))
def append_to_buffer(model_id, entry):
    buf = load_buffer(model_id); buf.append(entry); save_buffer(model_id, buf)
def clear_buffer(model_id):
    p = buffer_path(model_id)
    if p.exists(): p.unlink()

# ============================================================
# SMART DATA ANALYSIS
# ============================================================

def analyze_dataframe(df: pd.DataFrame, target_col: str = None):
    info = {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "numeric_cols": list(df.select_dtypes(include=np.number).columns),
        "categorical_cols": list(df.select_dtypes(include='object').columns),
    }

    if not target_col:
        target_candidates = [
            'outcome','target','label','class','result',
            'diagnosis','fraud','churn','survived',
            'disease','condition','status','y','subscribed',
            'purchased','converted','clicked','hired','attrition',
            'leaveornot','left','exited','response','success'
        ]
        for c in target_candidates:
            exact = [col for col in df.columns if col.lower() == c.lower()]
            if exact:
                target_col = exact[0]
                break
        if not target_col:
            # Short keywords (3 chars or fewer, e.g. 'y') are too prone to
            # accidental substring matches inside unrelated column names
            # (e.g. 'y' matching inside 'employee_id'). Only allow substring
            # matching for longer, more specific keywords.
            substring_candidates = [c for c in target_candidates if len(c) > 3]
            for c in substring_candidates:
                matches = [col for col in df.columns if c.lower() in col.lower()]
                if matches:
                    target_col = matches[0]
                    break
        if not target_col:
            target_col = df.columns[-1]

    info["target_col"] = target_col

    if target_col in df.columns:
        n_unique = df[target_col].nunique()
        if n_unique <= 20:
            info["task"] = "classification"
            info["n_classes"] = n_unique
            info["class_labels"] = list(df[target_col].unique())
        else:
            info["task"] = "regression"
    else:
        info["task"] = "clustering"

    n_rows = len(df)
    if info["task"] == "classification":
        info["algorithm"] = "GradientBoosting" if n_rows > 10000 else "RandomForest"
    elif info["task"] == "regression":
        info["algorithm"] = "GradientBoostingRegressor"
    else:
        info["algorithm"] = "KMeans"

    return clean(info)


def preprocess_dataframe(df: pd.DataFrame, target_col: str, task: str):
    df = df.copy()
    thresh = len(df) * 0.4
    df = df.dropna(thresh=thresh, axis=1)

    if target_col in df.columns:
        y_raw = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found")

    cat_cols = X.select_dtypes(include='object').columns
    le_map = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    target_encoder = None
    if task == "classification" and y_raw.dtype == object:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, imputer, le_map, target_encoder, list(X.columns)


def _detect_leakage_risks(df: pd.DataFrame, target_col: str, task: str) -> list[str]:
    """Return warnings for columns that may leak the answer into training.

    This is intentionally conservative and warning-only. It catches common
    cases like post-outcome fields, exact target duplicates, near-perfect
    numeric correlation with the target, and categorical columns that almost
    perfectly determine the target.
    """
    if target_col not in df.columns:
        return []

    warnings_out: list[str] = []
    target = df[target_col]
    target_name = target_col.lower()
    suspicious_name_terms = (
        "actual", "true", "outcome", "result", "label", "target",
        "post", "future", "final", "prediction", "predicted",
    )

    y_numeric = None
    try:
        if task == "classification":
            y_numeric = pd.Series(LabelEncoder().fit_transform(target.astype(str)), index=df.index)
        else:
            y_numeric = pd.to_numeric(target, errors="coerce")
    except Exception:
        y_numeric = None

    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        series = df[col]

        if target_name in col_lower or any(term in col_lower for term in suspicious_name_terms):
            warnings_out.append(
                f"Potential leakage column by name: {col!r}. Confirm it is known before the prediction moment."
            )

        try:
            if series.astype(str).fillna("__missing__").equals(target.astype(str).fillna("__missing__")):
                warnings_out.append(
                    f"Potential leakage column: {col!r} exactly duplicates the target column."
                )
        except Exception:
            pass

        if y_numeric is not None and pd.api.types.is_numeric_dtype(series):
            try:
                pair = pd.concat([pd.to_numeric(series, errors="coerce"), y_numeric], axis=1).dropna()
                if len(pair) > 20:
                    corr = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))
                    if corr >= 0.98:
                        warnings_out.append(
                            f"Potential leakage column: {col!r} has near-perfect correlation with the target (r={corr:.3f})."
                        )
            except Exception:
                pass

        try:
            if series.nunique(dropna=True) <= 100 and target.nunique(dropna=True) <= 50:
                tmp = pd.DataFrame({"feature": series.astype(str), "target": target.astype(str)}).dropna()
                if len(tmp) > 20 and tmp["feature"].nunique() > 1:
                    purity = (
                        tmp.groupby("feature")["target"]
                        .value_counts()
                        .groupby(level=0)
                        .max()
                        .sum()
                        / len(tmp)
                    )
                    if float(purity) >= 0.98:
                        warnings_out.append(
                            f"Potential leakage column: {col!r} almost perfectly determines the target ({purity * 100:.1f}% purity)."
                        )
        except Exception:
            pass

    # De-duplicate while preserving order and keep the payload readable.
    deduped = []
    for warning in warnings_out:
        if warning not in deduped:
            deduped.append(warning)
    return deduped[:8]


def _build_feature_input_profile(
    df: pd.DataFrame,
    feature_names: list[str],
    le_map: dict,
) -> dict:
    """Build lightweight training-time feature metadata for prediction QA."""
    profile = {"numeric": {}, "categorical": {}}

    for feature in feature_names:
        if feature not in df.columns:
            continue

        if feature in le_map:
            classes = [str(c) for c in getattr(le_map[feature], "classes_", [])]
            profile["categorical"][feature] = {
                "known_values": classes[:50],
                "known_value_count": len(classes),
            }
            continue

        series = pd.to_numeric(df[feature], errors="coerce").dropna()
        if len(series) < 2:
            continue
        profile["numeric"][feature] = {
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
            "q1": round(float(series.quantile(0.25)), 6),
            "q3": round(float(series.quantile(0.75)), 6),
            "median": round(float(series.median()), 6),
        }

    return profile


# ============================================================
# REAL TRAINING ENGINE
# ============================================================

def train_real_model(df: pd.DataFrame, target_col: str, task: str, algorithm: str, model_id: str):
    leakage_warnings = _detect_leakage_risks(df, target_col, task)
    X, y, scaler, imputer, le_map, target_encoder, feature_names = preprocess_dataframe(df, target_col, task)
    feature_input_profile = _build_feature_input_profile(df, feature_names, le_map)

    stratify = None
    if task == "classification":
        try:
            _, counts = np.unique(y, return_counts=True)
            if len(counts) > 1 and counts.min() >= 2:
                stratify = y
        except Exception:
            stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    if task == "classification":
        candidates = {
            "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        }
        best_model = None
        best_score = -1
        best_name = algorithm
        all_scores = {}

        for name, candidate in candidates.items():
            try:
                candidate.fit(X_train, y_train)
                candidate_preds = candidate.predict(X_test)
                score = float(balanced_accuracy_score(y_test, candidate_preds))
                all_scores[name] = {
                    "accuracy": round(float(accuracy_score(y_test, candidate_preds)) * 100, 2),
                    "balanced_accuracy": round(score * 100, 2),
                    "macro_f1": round(float(f1_score(y_test, candidate_preds, average="macro", zero_division=0)), 4),
                }
                if score > best_score:
                    best_score = score
                    best_model = candidate
                    best_name = name
            except Exception as e:
                print(f"Candidate {name} failed: {e}")

        model = best_model
        algorithm = best_name
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        acc = float(accuracy_score(y_test, preds))
        f1  = float(f1_score(y_test, preds, average='weighted', zero_division=0))
        macro_f1 = float(f1_score(y_test, preds, average='macro', zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_test, preds))
        precision_macro = float(precision_score(y_test, preds, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_test, preds, average="macro", zero_division=0))
        labels = list(np.unique(y))
        cm = confusion_matrix(y_test, preds, labels=labels)
        report = classification_report(y_test, preds, labels=labels, output_dict=True, zero_division=0)

        if target_encoder:
            label_names = [str(target_encoder.inverse_transform([label])[0]) for label in labels]
        else:
            label_names = [str(label) for label in labels]
        class_counts = {label_names[i]: int((y == labels[i]).sum()) for i in range(len(labels))}
        test_class_counts = {label_names[i]: int((y_test == labels[i]).sum()) for i in range(len(labels))}
        pred_class_counts = {label_names[i]: int((preds == labels[i]).sum()) for i in range(len(labels))}
        per_class_metrics = {}
        for i, label in enumerate(labels):
            src = report.get(str(label), report.get(label_names[i], {}))
            per_class_metrics[label_names[i]] = {
                "precision": round(float(src.get("precision", 0)), 4),
                "recall": round(float(src.get("recall", 0)), 4),
                "f1_score": round(float(src.get("f1-score", 0)), 4),
                "support": int(src.get("support", test_class_counts[label_names[i]])),
            }

        reliability_warnings = leakage_warnings.copy()
        majority_pct = max(class_counts.values()) / max(sum(class_counts.values()), 1) * 100 if class_counts else 0
        if majority_pct > 85:
            reliability_warnings.append(
                f"Severe class imbalance: the largest class is {majority_pct:.1f}% of the training data. Accuracy can look high while minority outcomes are missed."
            )
        if len(labels) == 2:
            minority_idx = min(range(len(labels)), key=lambda i: class_counts[label_names[i]])
            minority_label = label_names[minority_idx]
            minority_recall = per_class_metrics[minority_label]["recall"]
            minority_precision = per_class_metrics[minority_label]["precision"]
            if pred_class_counts[minority_label] == 0:
                reliability_warnings.append(
                    f"The model predicted zero examples of minority class {minority_label!r} on the test set. Do not rely on it to catch that outcome yet."
                )
            elif minority_recall < 0.5:
                reliability_warnings.append(
                    f"Minority-class recall is low for {minority_label!r} ({minority_recall:.2f}). The model may miss many important cases."
                )
            if minority_precision < 0.25 and pred_class_counts[minority_label] > 0:
                reliability_warnings.append(
                    f"Minority-class precision is low for {minority_label!r} ({minority_precision:.2f}). Positive predictions may need human review."
                )
        if acc - bal_acc > 0.15:
            reliability_warnings.append(
                "Accuracy is much higher than balanced accuracy, which usually means class imbalance is making the headline score look better than real-world performance."
            )

        metrics = {
            "accuracy": round(acc * 100, 2),
            "f1_score": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
            "balanced_accuracy": round(bal_acc * 100, 2),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "algorithm_selected": best_name,
            "all_scores": all_scores,
            "class_balance": class_counts,
            "test_class_balance": test_class_counts,
            "predicted_class_balance": pred_class_counts,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": {
                "labels": label_names,
                "matrix": cm.tolist(),
            },
            "reliability_warnings": reliability_warnings,
            "leakage_warnings": leakage_warnings,
            "feature_names": feature_names,
            "confidence_note": "Classification confidence is an uncalibrated model probability estimate, not a guarantee.",
        }
        if proba is not None and len(np.unique(y)) == 2:
            try:
                auc = float(roc_auc_score(y_test, proba[:, 1]))
                metrics["auc_roc"] = round(auc, 4)
                if auc < 0.7:
                    metrics["reliability_warnings"].append(
                        f"AUC is only {auc:.2f}; the model separates the classes weakly and should be treated as directional."
                    )
            except:
                pass

    elif task == "regression":
        candidates = {
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        }
        best_model = None
        best_score = -999
        best_name = algorithm
        all_scores = {}

        for name, candidate in candidates.items():
            try:
                candidate.fit(X_train, y_train)
                score = float(r2_score(y_test, candidate.predict(X_test)))
                all_scores[name] = round(score, 4)
                if score > best_score:
                    best_score = score
                    best_model = candidate
                    best_name = name
            except Exception as e:
                print(f"Candidate {name} failed: {e}")

        model = best_model
        algorithm = best_name
        preds = model.predict(X_test)
        mse = float(mean_squared_error(y_test, preds))
        r2  = float(r2_score(y_test, preds))
        metrics = {
            "r2_score": round(r2, 4),
            "rmse": round(np.sqrt(mse), 4),
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "algorithm_selected": best_name,
            "all_scores": all_scores,
            "feature_names": feature_names,
            "reliability_warnings": leakage_warnings + (
                ["R² is below 0.50, so predictions may be too noisy for automated decisions."]
                if r2 < 0.5 else []
            ),
            "leakage_warnings": leakage_warnings,
        }

    else:  # clustering
        # min(8, len(df)//50) becomes 0 for any dataset under 50 rows, and
        # KMeans(n_clusters=0) crashes outright. Forcing n_clusters=1 instead
        # would "succeed" but produce a meaningless single group — not
        # actually clustering anything, just silently returning a useless
        # result dressed up as a real one. A clear refusal is more honest.
        if len(df) < 100:
            raise ValueError(
                f"Clustering needs at least 100 rows to produce meaningful groups "
                f"(this dataset has {len(df)}). Try a classification or regression "
                f"target instead, or upload more data."
            )
        n_clusters = max(2, min(8, len(df) // 50))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X)
        metrics = {
            "n_clusters": n_clusters,
            "inertia": round(float(model.inertia_), 2),
            "feature_names": feature_names,
            "reliability_warnings": leakage_warnings + [
                "Clustering is exploratory. Cluster numbers are not predictions and should be validated with business context before action."
            ],
            "leakage_warnings": leakage_warnings,
        }

    # ── Feature importance ──
    feature_importance = {}
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = {
                name: round(float(imp), 4)
                for name, imp in zip(feature_names, importances)
            }
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = np.abs(coefs).mean(axis=0)
            else:
                coefs = np.abs(coefs)
            total = coefs.sum() or 1
            feature_importance = {
                name: round(float(c / total), 4)
                for name, c in zip(feature_names, coefs)
            }
        if feature_importance:
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            )
            metrics["feature_importance"] = feature_importance
    except Exception as e:
        print(f"Feature importance extraction failed: {e}")

    model_path = MODELS_DIR / f"{model_id}_model.pkl"

    model_package = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "le_map": le_map,
        "target_encoder": target_encoder,
        "feature_names": feature_names,
        "feature_input_profile": feature_input_profile,
        "target_col": target_col,
        "task": task,
        "algorithm": algorithm,
        "metrics": metrics,
    }
    joblib.dump(model_package, model_path)

    # Cache immediately so first prediction is instant
    _model_cache[model_id] = model_package

    print(f"Attempting Supabase upload for model {model_id}")
    success = upload_model_to_supabase(str(model_path), model_id)
    print(f"Supabase upload result: {success}")

    return clean(metrics), str(model_path)


def predict_with_model(model_id: str, input_data: dict):
    registry = load_registry()
    if model_id not in registry:
        raise ValueError("Model not found")

    meta = registry[model_id]
    model_path = Path(meta.get("model_path", ""))

    # Check in-memory cache first — avoids disk read and Supabase download
    if model_id in _model_cache:
        pkg = _model_cache[model_id]
    else:
        if not model_path.exists():
            print(f"Local model {model_id} missing — restoring from Supabase...")
            if not download_model_from_supabase(model_id, str(model_path)):
                raise ValueError("Trained model file not found — model may need to be retrained")
        pkg = joblib.load(model_path)
        _model_cache[model_id] = pkg  # cache for next request
    model      = pkg["model"]
    scaler     = pkg["scaler"]
    imputer    = pkg["imputer"]
    le_map     = pkg["le_map"]
    features   = pkg["feature_names"]
    task       = pkg["task"]
    target_enc = pkg.get("target_encoder")
    feature_input_profile = pkg.get("feature_input_profile", {})

    # Build a case-insensitive lookup from feature name → original feature name.
    # This prevents silent 0-fill when input sends "age" but model was trained
    # on "Age" — the model would silently receive 0 for every mismatched feature.
    input_lower = {k.lower(): v for k, v in input_data.items()}
    input_warnings: list[str] = []
    missing_features: list[str] = []
    unknown_categories: dict[str, str] = {}
    invalid_numeric: dict[str, str] = {}
    out_of_range_values: dict[str, dict] = {}
    provided_features: list[str] = []

    row = {}
    for f in features:
        if f.lower() in input_lower:
            original_val = input_lower[f.lower()]
            provided_features.append(f)
        else:
            original_val = 0
            missing_features.append(f)
        row[f] = original_val
    df_input = pd.DataFrame([row])

    for col, le in le_map.items():
        if col in df_input.columns:
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except:
                if col not in missing_features:
                    unknown_categories[col] = str(df_input[col].iloc[0])
                df_input[col] = 0

    numeric_profile = feature_input_profile.get("numeric", {})
    for col, stats in numeric_profile.items():
        if col not in df_input.columns:
            continue
        raw_val = df_input[col].iloc[0]
        numeric_val = pd.to_numeric(pd.Series([raw_val]), errors="coerce").iloc[0]
        if pd.isna(numeric_val):
            if col not in missing_features:
                invalid_numeric[col] = str(raw_val)
            continue
        numeric_val = float(numeric_val)
        if not math.isfinite(numeric_val):
            invalid_numeric[col] = str(raw_val)
            continue
        df_input[col] = numeric_val

        if col in missing_features:
            continue
        train_min = stats.get("min")
        train_max = stats.get("max")
        if train_min is not None and train_max is not None and (
            numeric_val < float(train_min) or numeric_val > float(train_max)
        ):
            out_of_range_values[col] = {
                "value": numeric_val,
                "training_min": train_min,
                "training_max": train_max,
            }

    if invalid_numeric:
        shown = ", ".join(f"{col}={val!r}" for col, val in list(invalid_numeric.items())[:8])
        raise ValueError(f"Invalid numeric input for: {shown}. Use numbers for numeric model features.")

    if missing_features:
        shown = ", ".join(missing_features[:8])
        extra = f" and {len(missing_features) - 8} more" if len(missing_features) > 8 else ""
        input_warnings.append(
            f"Missing feature values were filled with 0 for: {shown}{extra}. Prediction reliability may be reduced."
        )
    input_completeness_pct = round(len(provided_features) / max(len(features), 1) * 100, 1)
    if input_completeness_pct < 50:
        input_warnings.append(
            f"Only {input_completeness_pct}% of expected features were provided. This prediction may be unreliable."
        )
    if unknown_categories:
        shown = ", ".join(f"{col}={val!r}" for col, val in list(unknown_categories.items())[:5])
        input_warnings.append(
            f"Input contains category values not seen during training ({shown}). They were encoded with a fallback value, so treat this prediction cautiously."
        )
    if out_of_range_values:
        shown = ", ".join(
            f"{col}={vals['value']} outside training range [{vals['training_min']}, {vals['training_max']}]"
            for col, vals in list(out_of_range_values.items())[:5]
        )
        input_warnings.append(
            f"Input values fall outside the training data range ({shown}). Prediction reliability may be reduced."
        )

    X_imputed = imputer.transform(df_input[features])
    X_imputed = pd.DataFrame(X_imputed, columns=features)
    X = scaler.transform(X_imputed)

    result = {}
    if task == "classification":
        pred = model.predict(X)[0]
        pred_label = target_enc.inverse_transform([pred])[0] if target_enc else str(pred)
        result["prediction"] = pred_label
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            result["confidence"] = round(float(max(proba)) * 100, 1)
            result["confidence_note"] = pkg.get(
                "metrics",
                {},
            ).get("confidence_note", "Classification confidence is a model probability estimate, not a guarantee.")
            if target_enc:
                result["probabilities"] = {target_enc.inverse_transform([i])[0]: round(float(p)*100,1) for i, p in enumerate(proba)}
            else:
                result["probabilities"] = {str(i): round(float(p)*100,1) for i, p in enumerate(proba)}
    elif task == "regression":
        result["prediction"] = round(float(model.predict(X)[0]), 4)
    else:
        result["cluster"] = int(model.predict(X)[0])

    result["task"] = task
    result["model_name"] = meta["name"]
    result["input_completeness_pct"] = input_completeness_pct
    result["provided_features"] = provided_features
    if input_warnings:
        result["input_warnings"] = input_warnings
    if missing_features:
        result["missing_features"] = missing_features
    if unknown_categories:
        result["unknown_categories"] = unknown_categories
    if out_of_range_values:
        result["out_of_range_values"] = out_of_range_values
    reliability_warnings = pkg.get("metrics", {}).get("reliability_warnings") or []
    if reliability_warnings:
        result["model_reliability_warnings"] = reliability_warnings
    return result


# ============================================================
# SUPPORT CHATBOT
# ============================================================

CHAT_SYSTEM_PROMPT = """You are the support assistant for Archimedes MD (ArchimedesMD.com) — a platform that lets anyone organize messy data, generate professional data intelligence reports, and train real machine learning models in minutes — then call them forever on new inputs.

Core features you should know well:

DATA ORGANIZATION: The first and most important step for users who have messy or disorganized data. Users upload multiple files from different sources (CSVs, Excel sheets, JSONs) and Archimedes MD automatically analyzes them, proposes a merge plan for user review, then executes it — producing one clean coherent dataset. A two-pass verification process ensures data integrity. The clean dataset can instantly be used to train a model or generate a report without downloading. Free users get 5 Data Organization sessions, Pro and above get unlimited.

DATA INTELLIGENCE REPORT: After organizing or uploading clean data, users can generate a full professional-grade data analysis report — executive summary, statistical deep dive, correlation analysis, segment breakdowns, outlier profiling, effect sizes, and 8+ prioritized business recommendations with specific action steps. Takes just a few minutes. Useful on its own even if the user never trains a model.

MODEL TRAINING: The system automatically tests multiple algorithms and picks the best one. Training takes under a minute on most datasets. Algorithms tested: Random Forest, Gradient Boosting, Logistic Regression (classification); Gradient Boosting and Random Forest (regression); KMeans (clustering).

EVALUATE DATA: Run predictions on any new input using a saved model, with a plain English summary of the result and feature importance chart.

MODEL LIBRARY: All trained models are saved and accessible forever via Supabase.

PYTHON DOWNLOAD: Every trained model ships as a .pkl file with a clean inference script.

RETRAINING: After making predictions, users can submit the actual real-world outcome via the "Submit actual outcome" feature in the Evaluate Data panel. Once 10+ confirmed ground-truth outcomes are saved, users can retrain the model to absorb real-world feedback. The model learns from actual outcomes, not from its own predictions.

Pricing tiers:
- Free ($0): 1 model, 5 Data Organization sessions total (lifetime, does not reset), 1 Data Intelligence Report total (lifetime, does not reset), up to 10MB, CSV/JSON/Excel only
- Pro ($49.99/mo): 5 models, 100 Data Organization sessions, 50 Data Intelligence Reports, HTML download, 2GB uploads, all file types
- Team ($149/mo): 15 models, 400 Data Organization sessions, 200 Data Intelligence Reports + PDF export, 10GB uploads, full API access
- Unlimited ($399/mo): unlimited everything, priority support, custom integrations

When someone asks if the platform is worth it or what makes it valuable, start with Data Organization — most small businesses and teams have messy data and this solves their biggest problem before they even get to analysis or modeling. Then mention the Data Intelligence Report and model training.

Rules:
- Be concise and friendly. 2-4 sentences per reply unless the user explicitly asks for detail.
- Never make up features that aren't described above.
- Reply in plain text only — no markdown formatting (no **bold**, no bullet points with -, no headers). The chat window displays raw text, so markdown symbols would show up literally instead of being styled.
- If asked about billing issues or specific account problems, direct them to use the contact form on the homepage.
- Do not discuss competitors or make claims about being better than other tools.
- Always stay on topic: Archimedes MD features, usage, and troubleshooting only."""


class ChatRequest(BaseModel):
    message: str
    history: list = []


def _send_contact_email(subject: str, body_text: str):
    """Sends a contact-form message via the Resend API. Best-effort —
    failures are logged but never raise, since this runs in a background task."""
    if not RESEND_API_KEY or not CONTACT_RECIPIENT:
        print("Contact email skipped — RESEND_API_KEY/CONTACT_RECIPIENT not configured")
        return

    try:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": CONTACT_FROM_EMAIL,
                "to": [CONTACT_RECIPIENT],
                "subject": subject,
                "text": body_text,
            },
            timeout=10.0,
        )
        if resp.status_code >= 400:
            print(f"Resend API error ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"Contact email failed to send: {e}")


@app.post("/contact")
async def contact(request: Request, background_tasks: BackgroundTasks):
    """
    Public contact form endpoint — no auth required since this is meant for
    visitors on the marketing site before they've signed up. Accepts a name,
    email, and message, and emails it to the business inbox via Resend.
    """
    body = await request.json()
    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip()
    message = (body.get("message") or "").strip()

    if not message:
        raise HTTPException(status_code=400, detail="Please write a message before sending.")
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please provide a valid email so we can reply.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body_text = (
        f"Time: {timestamp}\n"
        f"Name: {name or 'not provided'}\n"
        f"Email: {email}\n\n"
        f"Message:\n{message}"
    )
    subject = f"[ArchimedesMD Contact] {name or email}"

    background_tasks.add_task(_send_contact_email, subject, body_text)

    return {"success": True, "message": "Thanks — we'll be in touch soon."}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        history = req.history[-8:] if len(req.history) > 8 else req.history
        messages = history + [{"role": "user", "content": req.message}]
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            system=CHAT_SYSTEM_PROMPT,
            messages=messages
        )
        reply = response.content[0].text
        reply = reply.replace("**", "").replace("__", "")
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DATA INTELLIGENCE REPORT ENGINE
# ============================================================

from scipy import stats as scipy_stats

def sample_for_report(
    df: pd.DataFrame,
    target_col: str | None,
    n: int,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, bool, str]:
    """Return a representative sample of df for report stat computation.

    Attempts stratified sampling when a classification-like target column is
    provided.  Falls back to random sampling with a fixed seed whenever
    stratification is not possible (continuous target, too-small classes,
    nulls in target, or any sklearn error).

    Args:
        df: The full source dataframe.
        target_col: Optional name of the prediction target column.
        n: Maximum number of rows to keep.  If len(df) <= n the original
           dataframe is returned unchanged.
        random_seed: Seed for the random number generator so sampling is
                     reproducible across runs.

    Returns:
        A 3-tuple of:
            - sampled dataframe (or original if no sampling needed)
            - bool indicating whether sampling actually occurred
            - str describing the sampling method used (for internal logging)
    """
    if len(df) <= n:
        return df, False, "no sampling needed"

    # Decide whether to attempt stratified sampling.
    use_stratified = False
    if target_col and target_col in df.columns:
        n_unique = df[target_col].dropna().nunique()
        # Treat as classification if cardinality is low relative to row count.
        if 2 <= n_unique <= 50:
            use_stratified = True

    if use_stratified:
        try:
            from sklearn.model_selection import train_test_split
            target_series = df[target_col].fillna("__missing__")
            # Ensure every class has at least 2 samples so train_test_split
            # can maintain proportions without error.
            class_counts = target_series.value_counts()
            if class_counts.min() < 2:
                raise ValueError(
                    f"class '{class_counts.idxmin()}' has only "
                    f"{class_counts.min()} sample(s) — too small to stratify"
                )
            fraction = n / len(df)
            _, sampled = train_test_split(
                df,
                test_size=fraction,
                stratify=target_series,
                random_state=random_seed,
            )
            return sampled.reset_index(drop=True), True, "stratified"
        except Exception as exc:
            print(
                f"[sample_for_report] stratified sampling failed "
                f"({exc}); falling back to random sampling"
            )

    # Random fallback.
    sampled = df.sample(n=n, random_state=random_seed)
    return sampled.reset_index(drop=True), True, "random"



def _positive_mask(series: "pd.Series") -> "pd.Series":
    """Return a boolean mask identifying the positive class in a binary series.

    Handles integer (0/1), boolean (True/False), and string-encoded binary
    values ("yes"/"no", "true"/"false", "1"/"0") so that auto segment rate
    computation works for all binary column encodings detected by
    _find_binary_outcome_cols, not just 0/1 integer columns.

    Args:
        series: A pandas Series containing a binary outcome column.

    Returns:
        A boolean Series where True indicates the positive class.
    """
    normalized = series.astype(str).str.strip().str.lower()
    return series.eq(1) | series.eq(True) | normalized.isin({"1", "true", "yes"})

def _find_binary_outcome_cols(df: pd.DataFrame) -> list[str]:
    """Detect columns that look like binary business outcome flags.

    Scans for columns with exactly 2 unique non-null values matching common
    binary encodings (0/1, True/False, yes/no) and whose name suggests a
    business outcome (e.g. stockout_flag, churned, converted, late_payment).
    Columns with names that suggest identifiers or categories are excluded.

    Args:
        df: The dataframe to scan.

    Returns:
        A list of column names that appear to be binary outcome flags,
        ordered by how strongly the name suggests an outcome column.
    """
    outcome_keywords = (
        "flag", "churn", "convert", "stockout", "default", "fraud",
        "late", "return", "cancel", "fail", "dropout", "readmit",
        "attrition", "delinquent", "bounce", "lapse", "event",
    )
    # Exclusion uses structural checks rather than broad substring matching.
    # Broad matching (e.g. "id" in col_lower) incorrectly excludes legitimate
    # binary outcome columns like "is_paid", "is_active", or "invalid".
    # Instead we check for exact match, common suffix/prefix patterns, and a
    # small set of unambiguous non-outcome keywords.
    non_outcome_suffixes = ("_id", "_key", "_code", "_index", "_num", "_no")
    non_outcome_prefixes = ("id_",)
    non_outcome_exact = {"id", "key", "index", "zip", "postal", "sku", "code"}
    non_outcome_contains = ("zipcode", "postcode", "sku_", "_sku")

    def _is_identifier_col(col: str) -> bool:
        """Return True if the column name looks like an identifier, not an outcome."""
        col_lower = col.lower()
        if col_lower in non_outcome_exact:
            return True
        if any(col_lower.endswith(s) for s in non_outcome_suffixes):
            return True
        if any(col_lower.startswith(p) for p in non_outcome_prefixes):
            return True
        if any(s in col_lower for s in non_outcome_contains):
            return True
        return False

    result: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        if _is_identifier_col(col):
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        unique_vals = set(series.unique())
        if len(unique_vals) != 2:
            continue
        # Accept 0/1 integers, True/False booleans, or yes/no strings
        is_binary = (
            unique_vals <= {0, 1}
            or unique_vals <= {True, False}
            or {str(v).strip().lower() for v in unique_vals} <= {"yes", "no"}
            or {str(v).strip().lower() for v in unique_vals} <= {"true", "false"}
        )
        if not is_binary:
            continue
        # Prefer columns whose name suggests an outcome
        has_outcome_keyword = any(kw in col_lower for kw in outcome_keywords)
        result.append((col, has_outcome_keyword))

    # Return outcome-named columns first, then generic binary columns
    result.sort(key=lambda x: (not x[1], x[0]))
    return [col for col, _ in result[:5]]  # cap at 5 to avoid prompt bloat


def compute_dataset_stats(df: pd.DataFrame, target_col: str = None) -> dict:
    result = {}

    total_cells = len(df) * len(df.columns)
    missing_cells = int(df.isnull().sum().sum())
    result["shape"] = {"rows": len(df), "cols": len(df.columns)}
    result["completeness_pct"] = round((1 - missing_cells / total_cells) * 100, 2) if total_cells else 100
    result["duplicates"] = int(df.duplicated().sum())
    result["columns"] = list(df.columns)
    result["dtypes"] = {c: str(df[c].dtype) for c in df.columns}

    missing = df.isnull().sum()
    result["missing"] = {
        c: {"count": int(missing[c]), "pct": round(missing[c] / len(df) * 100, 2)}
        for c in df.columns if missing[c] > 0
    }

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    result["numeric_analysis"] = {}
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        try:
            _, normality_p = scipy_stats.shapiro(s.sample(min(len(s), 500), random_state=42))
        except Exception:
            normality_p = None
        try:
            skew = float(s.skew())
            kurt = float(s.kurtosis())
        except Exception:
            skew, kurt = 0.0, 0.0
        # Real histogram bins so the report can render an actual distribution
        # chart instead of approximating from summary stats.
        try:
            counts, edges = np.histogram(s, bins=10)
            histogram = {
                "bin_edges": [round(float(e), 4) for e in edges],
                "bin_counts": [int(c) for c in counts],
            }
        except Exception:
            histogram = None

        # Exact value distribution for low-cardinality numeric columns.
        # This prevents the report from inventing categories (e.g. discount tiers)
        # that don't actually appear in the data. Only computed when n_unique <= 20
        # to avoid bloating the stats payload with continuous distributions.
        n_unique = int(s.nunique())  # defined here for numeric cols; redefined later for cat cols
        value_counts_exact: dict | None = None
        if n_unique <= 20:
            vc_exact = s.value_counts().sort_index()
            # Use string keys — JSON object keys are always strings anyway,
            # and float keys cause inconsistent key handling downstream.
            value_counts_exact = {
                str(round(float(k), 6)): int(v)
                for k, v in vc_exact.items()
            }

        result["numeric_analysis"][col] = {
            "mean":          round(float(s.mean()), 4),
            "median":        round(float(s.median()), 4),
            "std":           round(float(s.std()), 4),
            "min":           round(float(s.min()), 4),
            "max":           round(float(s.max()), 4),
            "q1":            round(q1, 4),
            "q3":            round(q3, 4),
            "iqr":           round(iqr, 4),
            "skewness":      round(skew, 4),
            "kurtosis":      round(kurt, 4),
            "skew_label":    "highly right-skewed" if skew > 1 else "right-skewed" if skew > 0.5 else "highly left-skewed" if skew < -1 else "left-skewed" if skew < -0.5 else "approximately symmetric",
            "outlier_count": len(outliers),
            "outlier_pct":   round(len(outliers) / len(s) * 100, 2),
            "outlier_mean":  round(float(outliers.mean()), 4) if len(outliers) else None,
            "is_normal":     bool(normality_p > 0.05) if normality_p is not None else None,
            "normality_p":   round(float(normality_p), 4) if normality_p is not None else None,
            "zero_count":    int((s == 0).sum()),
            "zero_pct":      round(float((s == 0).sum()) / len(s) * 100, 2),
            "cv":            round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else None,
            "histogram":     histogram,
            "value_counts":  value_counts_exact,
        }

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    result["categorical_analysis"] = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        total = len(df[col].dropna())
        n_unique = int(df[col].nunique())
        probs = vc / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        # Cap top_values more aggressively for high-cardinality columns —
        # showing 10 entries of a 500-unique-value column wastes prompt tokens
        # and rarely adds analytical value beyond the top 5.
        top_n = 5 if n_unique > 50 else 10
        top_vc = vc.head(top_n)
        # Truncate long string keys so a single category name can't consume
        # disproportionate tokens (e.g. product description fields).
        def _trunc(s: str, max_len: int = 40) -> str:
            """Truncate a string to max_len characters."""
            return s[:max_len] + "…" if len(s) > max_len else s

        result["categorical_analysis"][col] = {
            "n_unique":     n_unique,
            "entropy":      round(entropy, 4),
            "top_values":   {_trunc(str(k)): int(v) for k, v in top_vc.items()},
            "top_pcts":     {_trunc(str(k)): round(v / total * 100, 2) for k, v in top_vc.items()},
            "dominant_pct": round(float(vc.iloc[0]) / total * 100, 2) if len(vc) else 0,
            "cardinality":  "high" if n_unique > 50 else "medium" if n_unique > 10 else "low",
        }

    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = float(corr.iloc[i, j])
                if not np.isnan(val):
                    pairs.append({
                        "col1": num_cols[i],
                        "col2": num_cols[j],
                        "r": round(val, 4),
                        "strength": "very strong" if abs(val) > 0.8 else "strong" if abs(val) > 0.6 else "moderate" if abs(val) > 0.4 else "weak",
                        "direction": "positive" if val > 0 else "negative",
                    })
        pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        result["top_correlations"] = pairs[:20]

    if target_col and target_col in df.columns:
        result["target_col"] = target_col
        tgt = df[target_col].dropna()
        n_unique = int(tgt.nunique())

        if n_unique <= 20:
            vc = tgt.value_counts()
            result["target_distribution"] = {str(k): int(v) for k, v in vc.items()}
            result["target_pcts"] = {str(k): round(v / len(tgt) * 100, 2) for k, v in vc.items()}
            imbalance = round(float(vc.iloc[0] / vc.sum() * 100), 2)
            result["class_imbalance_pct"] = imbalance
            result["imbalance_severity"] = "severe" if imbalance > 85 else "moderate" if imbalance > 70 else "mild" if imbalance > 60 else "balanced"

            if len(vc) == 2:
                classes = list(vc.index)
                group_a = df[df[target_col] == classes[0]]
                group_b = df[df[target_col] == classes[1]]
                effect_sizes = {}
                for col in num_cols:
                    if col == target_col:
                        continue
                    a, b = group_a[col].dropna(), group_b[col].dropna()
                    if len(a) < 2 or len(b) < 2:
                        continue
                    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
                    if pooled_std == 0:
                        continue
                    d = float((a.mean() - b.mean()) / pooled_std)
                    effect_sizes[col] = {
                        "cohens_d": round(d, 4),
                        "magnitude": "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small",
                        "mean_a": round(float(a.mean()), 4),
                        "mean_b": round(float(b.mean()), 4),
                        "class_a": str(classes[0]),
                        "class_b": str(classes[1]),
                    }
                effect_sizes = dict(sorted(effect_sizes.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)[:25])
                result["effect_sizes"] = effect_sizes

            # Only compute binary-style segment rates when target is exactly
            # 2-class. For multiclass targets, _positive_mask collapses all
            # Compute per-class segment distribution for all categorical columns.
            # Uses generic class rates (proportion of each target class per segment)
            # rather than _positive_mask, which only handles known binary encodings
            # (0/1, yes/no, true/false). Arbitrary two-class targets like
            # "churned/retained" or "paid/unpaid" would silently become all-zero
            # under _positive_mask since neither value matches the positive patterns.
            segment_rates: dict = {}
            _valid_tgt = df[target_col].notna()
            _df_valid = df[_valid_tgt]

            for col in cat_cols:
                if col == target_col:
                    continue
                try:
                    # Cross-tabulate: for each segment value, compute the count
                    # and proportion of each target class. This works for binary
                    # targets with arbitrary class names and for multiclass targets.
                    xtab = (
                        _df_valid
                        .groupby([col, target_col])
                        .size()
                        .unstack(fill_value=0)
                    )
                    totals = xtab.sum(axis=1)
                    rates = (xtab.div(totals, axis=0)).round(4)
                    # Surface as a flat dict keyed by segment value, with one
                    # key per target class plus the total count.
                    seg_dict: dict = {}
                    for seg_val in xtab.index:
                        entry: dict = {"count": int(totals[seg_val])}
                        for cls in xtab.columns:
                            entry[f"rate_{cls}"] = float(rates.loc[seg_val, cls])
                        # For binary targets, also add a "conversion_rate" alias
                        # using the minority class (lower overall frequency) as
                        # the "positive" class so existing report logic still works.
                        if n_unique == 2:
                            cls_counts = _df_valid[target_col].value_counts()
                            positive_cls = cls_counts.idxmin()
                            entry["conversion_rate"] = float(rates.loc[seg_val, positive_cls])
                        seg_dict[str(seg_val)] = entry
                    # Sort by count descending, cap at 20
                    seg_dict = dict(
                        sorted(seg_dict.items(), key=lambda x: x[1]["count"], reverse=True)[:20]
                    )
                    segment_rates[col] = seg_dict
                except Exception:
                    pass
            result["segment_rates"] = segment_rates

        else:
            result["target_stats"] = result["numeric_analysis"].get(target_col, {})

    # ── Time-series analysis ──
    # Detect date-like columns and, if found, bucket numeric columns by month
    # to compute real trends instead of letting the report-writing model guess
    # at trends from a snapshot. This directly answers "revenue over time",
    # "top products over time", and similar period-based questions.
    result["time_series"] = {}
    date_candidates = []
    for col in df.columns:
        if col in num_cols:
            continue
        try:
            sample = df[col].dropna().astype(str).head(20)
            if len(sample) == 0:
                continue
            # Cheap pre-filter before the real pd.to_datetime attempt — skips
            # obviously non-date text (names, categories, IDs) instead of
            # letting pandas try and emit a parse-format warning for each one,
            # which otherwise spams the logs across every non-numeric column
            # in a wide dataset.
            if not sample.str.contains(_DATE_LIKE_HINT_PATTERN, regex=True, na=False).any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().mean() > 0.8:
                date_candidates.append(col)
        except Exception:
            continue

    if date_candidates and num_cols:
        date_col = date_candidates[0]
        try:
            ts_df = df[[date_col] + num_cols].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
            ts_df = ts_df.dropna(subset=[date_col])

            if len(ts_df) >= 10:
                ts_df["_period"] = ts_df[date_col].dt.to_period("M").astype(str)
                monthly = ts_df.groupby("_period")[num_cols].agg(["sum", "mean", "count"])

                periods = sorted(ts_df["_period"].unique())
                result["time_series"]["date_column"] = date_col
                result["time_series"]["period_granularity"] = "month"
                result["time_series"]["periods"] = periods
                result["time_series"]["period_count"] = len(periods)

                trends = {}
                for col in num_cols:
                    try:
                        period_sums = monthly[(col, "sum")].reindex(periods)
                        period_counts = monthly[(col, "count")].reindex(periods)
                        if period_sums.isna().all():
                            continue
                        first_val = float(period_sums.iloc[0])
                        last_val = float(period_sums.iloc[-1])
                        pct_change = round(((last_val - first_val) / first_val * 100), 2) if first_val != 0 else None
                        trends[col] = {
                            "first_period": periods[0],
                            "last_period": periods[-1],
                            "first_period_sum": round(first_val, 2),
                            "last_period_sum": round(last_val, 2),
                            "pct_change_first_to_last": pct_change,
                            "trend_direction": "increasing" if (pct_change or 0) > 5 else "decreasing" if (pct_change or 0) < -5 else "flat",
                            "by_period_sum": {p: round(float(v), 2) for p, v in period_sums.items() if not pd.isna(v)},
                            "by_period_count": {p: int(v) for p, v in period_counts.items() if not pd.isna(v)},
                        }
                    except Exception:
                        continue
                result["time_series"]["numeric_trends"] = trends

                # Top categorical values per period (e.g. top product/category by month),
                # limited to the lowest-cardinality categorical column to stay token-cheap.
                low_card_cats = [c for c in cat_cols if result["categorical_analysis"].get(c, {}).get("cardinality") in ("low", "medium")]
                if low_card_cats:
                    cat_col = low_card_cats[0]
                    try:
                        cat_ts = df[[date_col, cat_col]].copy()
                        cat_ts[date_col] = pd.to_datetime(cat_ts[date_col], errors='coerce')
                        cat_ts = cat_ts.dropna(subset=[date_col, cat_col])
                        cat_ts["_period"] = cat_ts[date_col].dt.to_period("M").astype(str)
                        top_by_period = {}
                        for p in periods:
                            sub = cat_ts[cat_ts["_period"] == p][cat_col]
                            if len(sub) == 0:
                                continue
                            vc = sub.value_counts().head(3)
                            top_by_period[p] = {str(k): int(v) for k, v in vc.items()}
                        result["time_series"]["top_category_by_period"] = {
                            "category_column": cat_col,
                            "data": top_by_period,
                        }
                    except Exception:
                        pass
        except Exception:
            pass

    result["sample_rows"] = clean(df.head(5).to_dict(orient="records"))

    # ── Auto segment rates ──
    # Compute outcome-by-segment rates even in exploratory mode (no target_col).
    # Run impact summary for all detected binary outcome columns —
    # regardless of whether segment_rates already exists from a target_col path.
    # This ensures target-selected reports also get richer lost revenue totals,
    # cross-tabs, and segment_impact_summary.
    auto_segment_rates: dict = {}
    binary_outcome_cols = _find_binary_outcome_cols(df)
    low_card_cats = [
        c for c in cat_cols
        if result.get("categorical_analysis", {}).get(c, {}).get("cardinality") in ("low", "medium")
    ]

    # Prioritise specific high-value financial columns over generic ones.
    # "lost_revenue_estimate" and "gross_margin_dollars" are preferred over
    # generic "price" or "cost" columns that may not reflect actual impact.
    priority_col_keywords = [
        "lost_revenue", "gross_margin", "revenue", "sales",
        "margin", "profit", "amount", "gmv", "spend",
    ]

    # Exclude columns that look like ratios, rates, scores, or index values —
    # summing these as if they were currency amounts produces nonsense like
    # "$15.51 net_revenue_expansion impact." Only genuine currency/volume
    # columns should be used for dollar-impact segment ranking.
    ratio_exclude_keywords = (
        "_rate", "_ratio", "_pct", "_percent", "_score", "_index",
        "expansion", "contraction", "churn_rate", "retention_rate",
        "conversion_rate", "growth_rate", "mrr_growth",
    )

    def _is_ratio_col(col: str) -> bool:
        """Return True if the column name suggests a ratio/rate/score, not a currency."""
        col_lower = col.lower()
        return any(col_lower.endswith(excl) or excl in col_lower for excl in ratio_exclude_keywords)

    def _rank_impact_col(col: str) -> int:
        """Return sort key for impact column priority (lower = higher priority)."""
        col_lower = col.lower()
        for rank, kw in enumerate(priority_col_keywords):
            if kw in col_lower:
                return rank
        return len(priority_col_keywords)

    impact_cols: list[str] = sorted(
        [col for col in num_cols
         if any(kw in col.lower() for kw in priority_col_keywords)
         and not _is_ratio_col(col)],
        key=_rank_impact_col,
    )[:5]

    for outcome_col in binary_outcome_cols:
        overall_rate = float(_positive_mask(df[outcome_col].dropna()).mean())
        overall_positive_count = int(_positive_mask(df[outcome_col]).sum())
        primary_impact_col = impact_cols[0] if impact_cols else None
        total_primary_impact = (
            float(df[_positive_mask(df[outcome_col])][primary_impact_col].sum())
            if primary_impact_col else None
        )

        for cat_col in low_card_cats:
            if cat_col == outcome_col:
                continue
            try:
                _valid = df[outcome_col].notna()
                _tmp_df = df.loc[_valid, [cat_col]].copy()
                _tmp_df["_pos"] = _positive_mask(df.loc[_valid, outcome_col]).astype(int)
                grp = _tmp_df.groupby(cat_col)["_pos"].agg(["mean", "count", "sum"])
                grp.columns = ["rate", "record_count", "positive_count"]
                grp["rate"] = grp["rate"].round(4)
                grp["pp_above_overall"] = (grp["rate"] - overall_rate).round(4)
                best_rate = float(grp["rate"].min())
                grp["pp_above_best"] = (grp["rate"] - best_rate).round(4)
                grp["relative_pct_above_best"] = (
                    ((grp["rate"] - best_rate) / best_rate * 100).round(2)
                    if best_rate > 0 else 0
                )

                for imp_col in impact_cols:
                    try:
                        rev_grp = df.groupby(cat_col)[imp_col].agg(["sum", "mean"])
                        positive_rows = df[_positive_mask(df[outcome_col])]
                        rev_positive = positive_rows.groupby(cat_col)[imp_col].agg(["sum", "mean"])
                        grp[f"{imp_col}_total"] = rev_grp["sum"].round(2)
                        grp[f"{imp_col}_mean"] = rev_grp["mean"].round(2)
                        grp[f"{imp_col}_positive_total"] = rev_positive["sum"].round(2)
                        grp[f"{imp_col}_positive_mean"] = rev_positive["mean"].round(2)
                    except Exception:
                        continue

                # Compute highest/lowest rate from full data BEFORE sort/truncate
                # so low-volume high-rate segments aren't silently excluded.
                highest_rate_seg = str(grp["rate"].idxmax())
                lowest_rate_seg = str(grp["rate"].idxmin())

                sort_col = f"{primary_impact_col}_positive_total" if primary_impact_col and f"{primary_impact_col}_positive_total" in grp.columns else "rate"
                impact_sort_col = f"{primary_impact_col}_positive_total"
                highest_impact_seg = str(grp[impact_sort_col].idxmax()) if primary_impact_col and impact_sort_col in grp.columns else None
                grp = grp.sort_values(sort_col, ascending=False).head(20)

                key = f"{outcome_col}_by_{cat_col}"
                auto_segment_rates[key] = {
                    "outcome_col": outcome_col,
                    "segment_col": cat_col,
                    "overall_rate": round(overall_rate, 4),
                    "overall_rate_pct": round(overall_rate * 100, 2),
                    "overall_positive_count": overall_positive_count,
                    "total_primary_impact": round(total_primary_impact, 2) if total_primary_impact is not None else None,
                    "highest_rate_segment": highest_rate_seg,
                    "lowest_rate_segment": lowest_rate_seg,
                    "highest_primary_impact_segment": highest_impact_seg,
                    "impact_columns_included": impact_cols,
                    "note": (
                        "Sorted by primary impact for positive outcome rows where available. "
                        "Use highest_rate_segment/lowest_rate_segment for rate comparisons. "
                        "Use *_positive_total to rank by total business impact."
                    ) if impact_cols else (
                        "No currency/financial impact column detected — segments are sorted by "
                        "positive_count only. Do NOT estimate or invent dollar impact. "
                        "Rank by count and rate only."
                    ),
                    "rates": grp.to_dict(orient="index"),
                }
            except Exception:
                continue

        # Cross-tab: outcome by (cat_col_a x cat_col_b) for top 2 categoricals.
        # Surfaces insights like "Outdoor in Wholesale" that single-column
        # breakdowns miss entirely.
        if len(low_card_cats) >= 2:
            try:
                # Prefer semantically meaningful column names for cross-tab.
                # These produce more useful insights than arbitrary low-card cols.
                preferred_crosstab_keywords = [
                    "category", "channel", "region", "segment",
                    "plan", "product_type", "tier", "type",
                ]

                def _crosstab_rank(col: str) -> int:
                    """Return sort priority for cross-tab column selection."""
                    col_lower = col.lower()
                    for rank, kw in enumerate(preferred_crosstab_keywords):
                        if kw in col_lower:
                            return rank
                    return len(preferred_crosstab_keywords)

                ranked_cats = sorted(low_card_cats, key=_crosstab_rank)
                cat_a, cat_b = ranked_cats[0], ranked_cats[1]
                _valid_cross = df[outcome_col].notna()
                _cross_tmp = df.loc[_valid_cross, [cat_a, cat_b]].copy()
                _cross_tmp["_pos"] = _positive_mask(df.loc[_valid_cross, outcome_col]).astype(int)
                cross = _cross_tmp.groupby([cat_a, cat_b])["_pos"].agg(["mean", "count", "sum"])
                cross.columns = ["rate", "record_count", "positive_count"]
                cross["rate"] = cross["rate"].round(4)
                cross["pp_above_overall"] = (cross["rate"] - overall_rate).round(4)
                if primary_impact_col:
                    positive_rows_cross = df[_positive_mask(df[outcome_col])]
                    rev_cross = positive_rows_cross.groupby([cat_a, cat_b])[primary_impact_col].sum().round(2)
                    cross[f"{primary_impact_col}_positive_total"] = rev_cross
                    cross = cross.sort_values(f"{primary_impact_col}_positive_total", ascending=False).head(15)
                else:
                    cross = cross.sort_values("rate", ascending=False).head(15)
                cross.index = [f"{a} × {b}" for a, b in cross.index]
                result.setdefault("auto_crosstab_rates", {})
                result["auto_crosstab_rates"][outcome_col] = {
                    "outcome_col": outcome_col,
                    "dimensions": [cat_a, cat_b],
                    "overall_rate_pct": round(overall_rate * 100, 2),
                    "note": "Cross-tabulation of outcome by two categorical dimensions. Reveals interaction effects missed by single-column breakdowns.",
                    "rates": cross.to_dict(orient="index"),
                }
            except Exception:
                pass

        # Segment impact summary — pre-ranked list for executive section.
        if auto_segment_rates:
            try:
                summary: dict = {
                    "overall_rate_pct": round(overall_rate * 100, 2),
                    "overall_positive_count": overall_positive_count,
                    "total_primary_impact": round(total_primary_impact, 2) if total_primary_impact is not None else None,
                    "primary_impact_column": primary_impact_col,
                    "top_segments_by_primary_impact": [],
                    "top_segments_by_rate": [],
                }
                # Build a flat list of (segment_label, lost_revenue, rate) across all breakdowns
                rev_rows: list[tuple] = []
                rate_rows: list[tuple] = []
                for key, val in auto_segment_rates.items():
                    if val.get("outcome_col") != outcome_col:
                        continue
                    for seg, data in val.get("rates", {}).items():
                        imp_key = f"{primary_impact_col}_positive_total" if primary_impact_col else None
                        rev = data.get(imp_key) if imp_key else None
                        rate = data.get("rate", 0)
                        label = f"{val['segment_col']}={seg}"
                        if rev is not None:
                            rev_rows.append((label, rev, rate))
                        rate_rows.append((label, rate, rev))
                summary["top_segments_by_primary_impact"] = [
                    {"segment": r[0], "primary_impact": r[1], "positive_rate_pct": round(r[2] * 100, 2)}
                    for r in sorted(rev_rows, key=lambda x: x[1], reverse=True)[:5]
                ]
                summary["top_segments_by_rate"] = [
                    {"segment": r[0], "positive_rate_pct": round(r[1] * 100, 2), "primary_impact": r[2]}
                    for r in sorted(rate_rows, key=lambda x: x[1], reverse=True)[:5]
                ]
                result.setdefault("segment_impact_summary", {})
                result["segment_impact_summary"][outcome_col] = summary
            except Exception:
                pass

    if auto_segment_rates:
        result["auto_segment_rates"] = auto_segment_rates

    # Auto effect sizes for binary outcome columns — prevents the model
    # from inventing Cohen's d values. Computed as Cohen's d between
    # the positive and negative class for each numeric column.
    auto_effect_sizes: dict = {}
    for outcome_col in binary_outcome_cols:
        try:
            _valid_mask = df[outcome_col].notna()
            pos = df[_valid_mask & _positive_mask(df[outcome_col])]
            neg = df[_valid_mask & ~_positive_mask(df[outcome_col])]
            if len(pos) < 5 or len(neg) < 5:
                continue
            for num_col in num_cols[:20]:  # cap to avoid prompt bloat
                try:
                    a = pos[num_col].dropna()
                    b = neg[num_col].dropna()
                    if len(a) < 5 or len(b) < 5:
                        continue
                    pooled_std = float(np.sqrt(
                        ((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2)
                        / (len(a) + len(b) - 2)
                    ))
                    if pooled_std == 0:
                        continue
                    d = float((a.mean() - b.mean()) / pooled_std)
                    auto_effect_sizes[f"{outcome_col}_{num_col}"] = {
                        "outcome_col": outcome_col,
                        "feature_col": num_col,
                        "cohens_d": round(d, 4),
                        "magnitude": "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small",
                        "mean_positive_class": round(float(a.mean()), 4),
                        "mean_negative_class": round(float(b.mean()), 4),
                    }
                except Exception:
                    continue
        except Exception:
            continue
    if auto_effect_sizes:
        result["auto_effect_sizes"] = auto_effect_sizes

    return clean(result)


REPORT_SYSTEM_PROMPT = """You are a world-class senior data scientist and strategic business analyst at a top-tier consulting firm. You have been handed pre-computed statistics from a client dataset. Your job is to produce a Data Intelligence Report that is thorough, insightful, and actionable.

You will receive a JSON object containing pre-computed statistics. Your output must be a single, complete, self-contained HTML page — no markdown, no explanation outside the HTML, ONLY the HTML document starting with <!DOCTYPE html>.

REPORT STRUCTURE (use exactly these sections in this order):
1. EXECUTIVE SUMMARY — 5-7 punchy bullet points a CEO can read in 90 seconds. Lead with the single most important finding. Include one specific opportunity and one specific risk.
2. DATASET HEALTH AUDIT — Completeness score, missing value analysis, duplicate assessment, data type issues, outlier summary. Color-code each finding: green (healthy), amber (monitor), red (action required).
3. STATISTICAL DEEP DIVE — For each numeric column: interpret the distribution shape (not just the numbers — what does a skewness of 1.8 MEAN for this business?), flag outliers with business context, note if the column is normally distributed or not and why that matters.
4. CATEGORICAL INTELLIGENCE — For each categorical column: what does the dominant category tell us? What is the diversity/entropy of this column? Are any categories dangerously underrepresented?
5. TIME-BASED TRENDS — Only include this section if a "time_series" object is present in the data with a non-empty "numeric_trends" or "top_category_by_period". If present, this section is mandatory and must come before Pattern & Correlation Analysis. Report, by month: which numeric metrics increased or decreased over the observed periods and by how much (cite the exact pct_change_first_to_last and the first/last period sums), and which categories led each period if top_category_by_period is present. Call out the overall trend direction plainly (e.g. "Revenue rose 34% from 2023-01 to 2023-11" rather than vague language like "revenue has been growing"). If no time_series data is present (no usable date column was found), state in one sentence that no time-based analysis was possible because no reliable date column was detected, and skip the rest of this section.
6. PATTERN & CORRELATION ANALYSIS — Interpret every strong correlation. Don't just say "X and Y are correlated at 0.72" — explain the causal story, what it implies operationally, and what the business should do with it.
7. SEGMENT ANALYSIS — Use all computed segment rate data available. Check for "segment_rates", "auto_segment_rates", "auto_crosstab_rates", and "segment_impact_summary" in the statistics.

For auto_segment_rates: use "highest_rate_segment" and "lowest_rate_segment" for rate comparisons (not "best_segment" or "worst_segment" — those fields do not exist). Use "overall_rate_pct" for the company average. Use "pp_above_overall" for each segment's gap vs overall — do not recalculate this yourself. State exact rates and gaps (e.g. "Wholesale has a 39.3% stockout rate, 2.1pp above the company average of 37.2%").

For segment_impact_summary: each key is a binary outcome column name. Use "top_segments_by_primary_impact" to lead the executive summary with the biggest business impact, and "top_segments_by_rate" for rate-based rankings. Always distinguish between rate-based and revenue-impact-based rankings. The "primary_impact_column" field tells you which financial metric was used — name it correctly in the report rather than always calling it "lost revenue."

For auto_crosstab_rates: surface the top 3-5 combinations by total business impact (primary impact column), and mention stockout rate separately. Do not conflate rate and revenue impact — e.g. "Wholesale × Outdoor has the highest total lost revenue among all segment combinations, with a 42.6% stockout rate."

Never say you "cannot confirm" rates when computed data is present. Never infer or guess segment rates — if no computed rate exists, label it as a hypothesis.
8. EFFECT SIZE & PREDICTIVE SIGNALS — Only use Cohen's d values if an "effect_sizes" or "auto_effect_sizes" object is present in the statistics. If effect sizes were computed, use them to explain which features actually matter for predicting the outcome and distinguish statistical significance from practical significance. If neither object is present, state explicitly that effect sizes were not computed for this dataset and rely on correlations, segment rates, and impact totals instead. Never invent or estimate Cohen's d values.
9. BUSINESS RECOMMENDATIONS — Exactly 10 specific, prioritized recommendations. If evidence runs thin for later ones, frame them as investigative/process recommendations rather than quantified interventions — but never drop below 10 and never invent numbers to fill one out. Each must have: a bold headline action, the data evidence (cite specific numbers where real ones exist), estimated impact level (High/Medium/Low), implementation difficulty (Easy/Medium/Hard), and a specific first step.
10. RISK FLAGS — Minimum 4 specific risks identified in the data. Each must have a severity level and a specific mitigation strategy.
11. WHAT TO MEASURE NEXT — 5 specific data points or experiments the company should run to validate these findings.
12. APPENDIX — Raw statistics table for each numeric column.

STYLE RULES:
- Use the brand colors: background #060e1f, primary text #f0faff, accent cyan #00d4ff, magenta #e040fb, gold #ffd600, teal #00b894, coral #ff4444
- Each section must have a clear visual separator.
- Use color-coded badges: 🟢 healthy/positive, 🟡 monitor/neutral, 🔴 risk/negative
- Include small inline SVG bar charts for distributions and segment comparisons — build them with pure SVG, no libraries
- The report must be printable (include @media print CSS)
- Fonts: use system-ui or sans-serif, no external font imports
- List item spacing: give all <li> elements margin: 16px 0 and padding: 4px 0 so bullets are easy to scan. For executive summary bullets specifically, add a left border: border-left: 3px solid rgba(0,212,255,0.3); padding-left: 12px; list-style: none; to make each point feel like a distinct card row rather than a tight list.
- Make the recommendations SPECIFIC and NUMBERED. Every insight must cite a specific number from the statistics. Use a plain <div> or <ul> for recommendations — do NOT use <ol> tags, as the list numbers are already written inline in bold (e.g. '1. **Deploy Gradient Boosting**'). Using <ol> causes double numbering.
- PERCENTAGE CHANGE ACCURACY: Always calculate percentage differences correctly as ((B - A) / A) * 100. For example: group A = 0.47, group B = 0.66 → ((0.66 - 0.47) / 0.47) * 100 = 40% higher, NOT 66% (that would be the raw value of B, not the change). Double-check every percentage claim before writing it.
- The tone is clear, direct, and constructive — like a trusted advisor explaining what the numbers mean and what to do about it. Avoid dramatic or alarmist language (e.g. "catastrophic," "crisis," "disaster") even when describing serious issues. State the finding plainly, explain why it matters, and move to what can be done. A reader with no data background should feel informed and capable of acting, not alarmed.

CONTENT RULES:
- SEGMENT ANALYSIS: Never use "high-value" or "low-value" language for medical or clinical datasets. Use neutral descriptive labels like "Tumor Positive Profile" and "No Tumor Profile". For business datasets, "high-converting segment" is fine but avoid implying one human outcome is more valuable than another.
- RECOMMENDATIONS: If a predictive model recommendation is relevant, include it as one recommendation and explain why the model type fits the data characteristics (e.g. feature types, class balance, cardinality). Do not force it to be Recommendation 1 unless predictive modeling is genuinely the most important next action for this dataset. Do not estimate accuracy, R², lift, or recovery unless actual trained model metrics are provided in the statistics.
- RISK FLAGS: Ensure severity labels match the actual risk described. Do not label a speculative or low-probability risk as HIGH severity. HIGH = immediate action needed, data is currently compromised. MEDIUM = monitor and investigate. LOW = document and revisit.
- MISSING VALUES described as "outliers" must be clearly distinguished — a zero-inflated column is a distribution characteristic, not the same as IQR-based outliers. Describe them separately.
- For mixed technical/non-technical audiences, flag advanced recommendations (PCA, feature engineering) with "(Advanced)" so non-technical users know they can skip these.
- CONFOUNDING VARIABLES IN MEDICAL/IMAGE DATASETS: If image dimensions, file size, or other technical metadata appear as top predictive features in a medical imaging dataset, do NOT present them as clinical opportunities. Instead, flag them prominently as likely data collection artifacts or confounding variables. Explain clearly that: (1) the size difference probably reflects different image sources or equipment, not a biological signal; (2) a model trained on these features would likely fail on any standardized dataset; (3) the right action is to standardize image resolution before modeling, not to use dimensions as a feature. This warning must appear in the Executive Summary AND as a HIGH severity Risk Flag — not buried in recommendations. Never recommend building a classifier on technical metadata features in medical datasets without this caveat.
- INTERNAL CONSISTENCY: Before writing recommendations, cross-check them against risk flags. If a risk flag contradicts a recommendation (e.g. recommending use of a feature that is simultaneously flagged as a data leakage risk), resolve the contradiction explicitly. The recommendation should defer to the risk — flag the issue and explain why the feature should not be used as-is.
- LANGUAGE: Avoid alarmist bold lead-ins such as "CRITICAL FINDING," "CRISIS," "CATASTROPHIC," "VULNERABILITY," or similar dramatic framing — even in the Executive Summary or Risk Flags sections. Use plain, specific descriptions instead (e.g. "Customer identity gap:" instead of "CRITICAL FINDING:", "Revenue concentration:" instead of "CRISIS:"). The finding can still be serious and the language can still be direct — just not dramatic.
- BASE-RATE SANITY CHECK: Before presenting any percentage or share as a notable finding (e.g. "X% of orders come from California"), check whether that number is close to a natural baseline you can reasonably infer — population share, category count, number of days/months in the period, even split across N groups, etc. If the number is unremarkable once compared to its baseline (e.g. 11.6% of orders from California, when California is ~11.5% of the US population), either omit it entirely or state explicitly that it tracks the expected baseline and is not a meaningful pattern. Never present a baseline-matching statistic as if it were an insight. Only highlight percentages that meaningfully deviate from a reasonable baseline.
- SAMPLING DISCLOSURE: If a "SAMPLING NOTE" appears in the data you receive, you must include a short, clear disclosure in the report introduction — e.g. "Detailed statistics were computed on a representative 5,000-row sample of the full 48,000-row dataset." Use the exact numbers from the note. The top-level metadata (total row count, column count, missingness, duplicates) reflects the full dataset and should be presented as such. All statistical distributions and correlations are based on the sample and may differ slightly from full-dataset values.
- SPECIFIC, NOT GENERIC RECOMMENDATIONS: Every recommendation must be derivable only from a specific number, trend, or pattern actually present in THIS dataset — not generic e-commerce/business playbook advice that could apply to any company. Before writing each recommendation, identify the exact statistic, correlation, segment rate, or trend driving it and cite it in the same sentence as the headline action. If a recommendation cannot be tied to a specific number from the data provided, do not include it. Avoid recommending things that are standard practice regardless of the data (e.g. generic "diversify your sales channels" advice) unless the data shows a specific concentration risk that justifies it with an exact figure.
- NO EMOJIS IN HEADERS: Never use emoji characters in section headers or sub-headers. Plain text only.
- TONE CALIBRATION: Never use dramatic or crisis language (e.g. "bleeding," "crisis," "failing system," "catastrophic"). Use direct, factual descriptions instead — e.g. "Stockouts are reducing revenue" or "inventory allocation is underperforming." The finding can be serious without being theatrical.
- NO OUTSIDE BENCHMARKS: Never cite external industry benchmarks (e.g. "best-in-class targets sub-5% stockout rates") unless that benchmark is present in the dataset itself. If no benchmark data exists in the provided statistics, do not reference what competitors or industry standards achieve.
- NO UNSUPPORTED QUANTITATIVE CLAIMS: Never state specific dollar recovery figures (e.g. "could recover $217K," "recover $36K annually," "recovering $240-400K"), model accuracy percentages, event count ranges, or precise lift estimates unless those exact values are explicitly present in the computed statistics. This includes phrases like "could recover approximately $X" or "a X% reduction could recover $Y" — and vaguer-sounding but equally unsupported magnitude claims like "tens of thousands," "$100K+," or "six figures" are covered by this same rule; vagueness does not make an unsupported number acceptable. Do not derive new projections or estimates in recommendations. Use directional phrasing only: "could recover a meaningful portion of lost revenue," "would reduce lost revenue," "could reduce the stockout rate significantly." Never calculate and state a recovery dollar amount that is not in the stats.
- NO OVERSTATED COMPARISONS: When comparing segments, use the precomputed "pp_above_overall" and "pp_above_best" fields from auto_segment_rates — do not recalculate differences yourself. Only use strong language like "significantly underperforming" when the pp_above_overall gap is 5+ percentage points. For gaps under 5pp, use hedged language like "modestly higher stockout rate." Always verify that segment labels (highest/lowest) match the actual rates shown.
- NO EFFECT SIZES WITHOUT DATA: Use "effect_sizes" or "auto_effect_sizes" only if present in the statistics. If neither is present, state that effect sizes were not computed and rely on correlations, segment rates, and impact totals instead. Never invent or estimate Cohen's d values.
- CENSORED SALES / FORECAST ACCURACY: Never claim forecast accuracy or suggest the business "does not have a demand prediction problem" based on correlation between `forecast_demand_units` and `units_sold`. This is a hard prohibition, not a suggestion. When `stockout_flag` is present in the dataset, `units_sold` is inventory-capped observed demand — not true demand — and a high correlation between forecast and capped sales is not evidence of forecast accuracy. Any phrase like "forecasts are accurate," "predicted demand matches actual sales," "forecasting is strong," "no demand prediction problem," or "not a demand forecasting problem" must be replaced with a statement acknowledging the inventory-capping limitation. Never frame `forecast_demand_units` as an "uncensored demand signal" or any equivalent phrasing implying it reveals true, uncapped demand — it is a forecast input, not a demand measurement. Only claim forecast accuracy if `forecast_error`, `mape`, or `mae` is explicitly present in the statistics.
- LOW-CARDINALITY NUMERIC VALUE COUNTS: For any numeric column that has a `value_counts` field in `numeric_analysis`, only list the values that actually appear there. Do not invent categories or tiers (e.g. discount tiers, price bands, threshold groups) beyond what is present in `numeric_analysis[col].value_counts`. If `numeric_analysis.discount_rate.value_counts` shows {"0.0": 5200, "0.05": 1400, "0.1": 600}, only those three tiers exist — never add or imply additional tiers. Never infer tiers or buckets from histogram bin edges, mean, median, skewness, outlier counts, or example values. If `value_counts` is absent for a column, do not speculate about its distribution structure at all — only report the summary statistics that are explicitly provided.
- METRIC MEANINGS: The following field interpretations are fixed and must not be overridden by inference or correlation. Use the exact meaning when discussing each field. If a column matching these patterns is in the dataset, apply these definitions: `lost_revenue_estimate` = estimated revenue not captured due to stockout (not margin, not realized revenue); `gross_margin_dollars` = realized gross margin on sold units only; `units_sold` = observed sold units, may be capped by inventory when stockouts occur — not true demand; `forecast_demand_units` = model or planner forecast input, not a measure of forecast accuracy and not an uncensored true-demand signal; `week` = a categorical week label, not a continuous time series unless `time_series` stats are explicitly computed. Do not conflate `lost_revenue_estimate` with `gross_margin_dollars`. Do not claim "annual" figures unless `annualized_*` fields are present in the stats. Do not claim "regional spread" without computed `max_rate - min_rate` from region segment rates. Do not add `gross_margin_dollars` to `lost_revenue_estimate` and call the result "total lost gross profit" — these are different metrics and cannot be combined.
- NO DOLLAR IMPACT WITHOUT CURRENCY COLUMN: If `impact_columns_included` is empty in the segment stats, no financial impact column was detected. In this case, rank segments by count and rate only. Do not estimate, infer, or invent dollar impact figures. Never sum ratio, rate, score, or expansion columns and present the result as a dollar amount.
- SUPPORTED ALGORITHMS ONLY: When recommending a predictive model, only mention algorithms supported by this platform: Random Forest, Gradient Boosting, Logistic Regression, Ridge Regression, and K-Means Clustering. Do not recommend XGBoost, LightGBM, CatBoost, neural networks, deep learning, SVMs, or any other algorithm not in this list.
- BUSINESS IMPACT FIRST: When ranking problems, prioritize total business impact over rate alone. A large segment with a slightly lower stockout rate but much higher total lost revenue is usually more important than a small high-rate segment. Use the *_positive_total fields in auto_segment_rates to rank by business impact. Always note both rate and total impact when discussing segments.
- FINAL AUDIT: Before finalizing the report, check for math consistency: highest/lowest segment labels must match the rates shown; percentage point differences must equal rate A minus rate B using the precomputed pp_above_overall fields; relative differences must be labeled as relative not absolute; remove any unsupported event counts, recovery estimates, or model accuracy claims; remove any Cohen's d values unless present in auto_effect_sizes or effect_sizes. Also specifically scan for and remove these exact phrase patterns unless the exact supporting figures or fields they require are present in the stats: "not a demand forecasting problem" or "no demand prediction problem" (need forecast_error/mape/mae), "annualized estimate" or any annual dollar figure (need annualized_* fields), "recover $X," "tens of thousands," "$100K+," or "six figures" (need the exact figure in the stats), and "uncensored demand signal" (never acceptable framing for forecast_demand_units).
- ANNUALIZED ESTIMATES: Never state an "annualized estimate" or any annual/yearly dollar or unit figure unless `annualized_*` fields are explicitly present in the statistics. If the dataset covers a partial period (e.g. 26 weeks, 3 months) and no `annualized_*` field exists, do not derive your own annualized number by scaling a partial-period figure — report only the actual period covered (e.g. "$1.2M in lost revenue over the 26 weeks of data," not "$2.4M annualized"). If an `annualized_*` field IS present, use that exact value and label it clearly as an annualized estimate — never state it as a confirmed annual figure.
- NO DISCOUNTING STOCKOUT ITEMS: Never recommend discounting products that are at risk of stocking out or already stocked out. Discounting accelerates depletion and worsens the stockout. The correct levers are: discounting substitutes, discounting overstock of alternative SKUs, or demand-shifting promotions on available inventory.
- NO SKU RETIREMENT FROM STOCKOUT DATA ALONE: Never recommend retiring, discontinuing, or consolidating SKUs/products/categories based on low stockout rate or low lost-revenue total alone. Low lost revenue from stockouts says nothing about margin, sell-through velocity, strategic assortment value, customer demand, or substitutability — none of which are typically present in stockout stats. If assortment rationalization comes up, frame it as a question to investigate with the missing data, not a recommendation to act on.
- USER STEERING INSTRUCTIONS: If the user message includes steering instructions (e.g. "only focus on Product X", "exclude category Y", "I only care about repeat customers"), these are not optional context — they are binding constraints on the entire report. Apply them consistently across every section: the Executive Summary, statistical analysis, segment analysis, and recommendations must all reflect the requested focus. If honoring the instruction means a section has little or nothing to say (e.g. excluding a category that dominates the dataset leaves few rows), say so plainly rather than quietly reverting to analyzing the full unfiltered dataset.

IMPORTANT: Keep the HTML concise. Do not pad with unnecessary whitespace or repetition. Prioritize completing all 12 sections over decorative detail. Section numbers must be strictly sequential: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 — no gaps, no skipped numbers, no repeated numbers. Every section header must display its number.

- Never cut a sentence mid-thought. If you are running low on space, shorten earlier sections rather than leaving a sentence incomplete.
- Every statistic cited must be a complete statement: never write "r=" or "Cohen's d =" without the actual value following it.
- Write tight prose. One strong sentence beats three weak ones. The goal is a report a busy professional can read in 10 minutes.
- Every recommendation must be a complete thought with all four parts present: Evidence, Impact, Difficulty, and First Step. Never leave a recommendation half-written.
- Business Recommendations: exactly 10 recommendations. Do not stop at 8. If later recommendations have weaker evidence than the earlier ones, make them investigative or process recommendations (e.g. "investigate X," "review Y process," "audit Z before acting") rather than quantified interventions — but every one of the 10 must still be grounded in stats actually provided, even the investigative ones. Never invent recovery dollars, event counts, model performance figures, or exact intervention lift to fill out a recommendation.
- APPENDIX: The statistics table must include EVERY numeric column in the dataset without exception. Do not omit columns to save space — shorten descriptions elsewhere if needed.

CSS: Use exactly this stylesheet in the <style> tag — do not invent alternative colors, fonts, or layouts:
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #060e1f; color: #f0faff; font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; padding: 40px 20px; }
.container { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 2.5em; margin-bottom: 0.5em; background: linear-gradient(135deg, #00d4ff, #e040fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
h2 { font-size: 1.8em; margin: 2em 0 0.8em 0; color: #00d4ff; border-bottom: 2px solid rgba(0,212,255,0.3); padding-bottom: 0.3em; }
h3 { font-size: 1.3em; margin: 1.5em 0 0.5em 0; color: #ffd600; }
p { margin: 1em 0; }
ul, ol { margin: 1em 0; padding-left: 2em; }
li { margin: 16px 0; padding: 4px 0; }
.exec-summary li { border-left: 3px solid rgba(0,212,255,0.3); padding-left: 12px; list-style: none; margin: 16px 0; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; margin-right: 8px; }
.badge-green { background: rgba(0,184,148,0.2); color: #00b894; }
.badge-amber { background: rgba(255,214,0,0.2); color: #ffd600; }
.badge-red { background: rgba(255,68,68,0.2); color: #ff4444; }
.badge-high { background: rgba(255,68,68,0.2); color: #ff4444; }
.badge-medium { background: rgba(255,214,0,0.2); color: #ffd600; }
.badge-low { background: rgba(0,184,148,0.2); color: #00b894; }
.badge-easy { background: rgba(0,184,148,0.2); color: #00b894; }
.badge-hard { background: rgba(255,68,68,0.2); color: #ff4444; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 2em 0; }
.stat-card { background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.2); border-radius: 8px; padding: 20px; }
.stat-value { font-size: 2em; font-weight: 700; color: #00d4ff; margin: 0.2em 0; }
.stat-label { font-size: 0.9em; color: rgba(240,250,255,0.7); text-transform: uppercase; letter-spacing: 0.05em; }
table { width: 100%; border-collapse: collapse; margin: 2em 0; font-size: 0.9em; }
th, td { padding: 12px; text-align: left; border-bottom: 1px solid rgba(0,212,255,0.1); }
th { background: rgba(0,212,255,0.1); color: #00d4ff; font-weight: 600; }
tr:hover { background: rgba(0,212,255,0.03); }
.separator { height: 2px; background: linear-gradient(90deg, transparent, rgba(0,212,255,0.5), transparent); margin: 3em 0; }
.rec-item { background: rgba(224,64,251,0.05); border-left: 4px solid #e040fb; padding: 16px 20px; margin: 16px 0; border-radius: 4px; }
.rec-headline { font-size: 1.1em; font-weight: 700; color: #e040fb; margin-bottom: 8px; }
.risk-item { background: rgba(255,68,68,0.05); border-left: 4px solid #ff4444; padding: 16px 20px; margin: 16px 0; border-radius: 4px; }
.chart-container { margin: 2em 0; }
@media print { body { background: white; color: black; } h1, h2, h3 { color: black; } }

OUTPUT: Return ONLY the complete HTML document. No preamble, no explanation, no markdown fences."""


# ── Auth ──
_bearer = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Auth service not configured")
    try:
        url = f"{SUPABASE_URL}/auth/v1/user"
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_KEY,
        }
        resp = httpx.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        user_data = resp.json()
        user_id = user_data.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Could not identify user")
        return user_id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Auth check failed")



class BuildPromptRequest(BaseModel):
    q1: str = ""   # dataset description
    q2: str = ""   # main question to answer
    q3: str = ""   # exclusions / deprioritizations
    columns: list[str] = []
    file_id: str = ""  # optional — if provided, loads stats for richer context


@app.post("/build-prompt")
async def build_prompt(
    req: BuildPromptRequest,
    user_id: str = Depends(get_current_user)
):
    """Generate a focused report steering prompt from the user's answers.

    This endpoint used to ask Claude to write "a concise steering prompt
    (3-5 sentences)" with the quality guardrails embedded naturally into the
    prose — which produced exactly the undifferentiated paragraph problem
    that kept resurfacing downstream. It also duplicated, in a separate
    hand-rolled block, much of what _build_steering_block already does more
    thoroughly (real computed stats vs. this endpoint's own column-name
    keyword guessing).

    Fixed by narrowing what Claude is asked for — a single, short OBJECTIVE
    statement only, nothing else — and then routing that objective through
    the SAME _build_steering_block function used at actual report-generation
    time. No second API call was added: this is still exactly one call to
    Claude, just asked for something narrower, with the (already-existing)
    structuring function doing the rest deterministically in code. The
    practical effect: what a user sees here as their "prompt preview" now
    IS the real structured prompt, not a paragraph that gets restructured
    later out of view.

    Args:
        req: The user's answers to the three prompt-builder questions plus
            an optional file_id for richer dataset-aware context.
        user_id: Authenticated user ID from the request token.

    Returns:
        A dict with a "prompt" key containing the full structured steering
        prompt (REPORT OBJECTIVE / ANALYSIS FOCUS / METRICS TO PRIORITIZE /
        EVIDENCE RULES / LABELING RULES / DO NOT DO).
    """
    stats: Optional[dict] = None
    binary_outcome_hint = ""

    if req.file_id:
        data_path = DATA_DIR / f"{req.file_id}.pkl"
        ensure_file_owner(req.file_id, user_id)
        if data_path.exists():
            try:
                df = pd.read_pickle(str(data_path))
                df_for_stats, _, _ = sample_for_report(df, None, REPORT_SAMPLE_ROWS)
                stats = compute_dataset_stats(df_for_stats)
                stats["columns"] = list(df.columns)
                binary_outcome_cols = _find_binary_outcome_cols(df)
                if binary_outcome_cols:
                    binary_outcome_hint = f"\nDetected binary outcome columns: {', '.join(binary_outcome_cols)}."
            except Exception:
                pass  # degrade gracefully — still generate a prompt without stats

    col_list = f"Column names: {', '.join(req.columns[:30])}" if req.columns else ""

    user_message = f"""You are helping a user write a short, clear OBJECTIVE statement for a data intelligence report — NOT a full instruction set. Evidence rules, labeling requirements, and other guardrails are added automatically afterward in code; do not write any of those yourself.

{f"Dataset description: {req.q1}" if req.q1 else ""}
{col_list}{binary_outcome_hint}
Main question: {req.q2}
{f"Exclude or deprioritize: {req.q3}" if req.q3 else ""}

Write ONLY a 2-3 sentence objective describing what this report should focus on, based on the main question above. Reference the relevant columns by name where it helps clarity.

Do NOT do any of the following in the objective, even though it may feel natural:
- Do not invent a formula for a metric (e.g. "revenue (units_sold × unit_price)") — if a computed revenue figure exists, just say "revenue"; do not show your own arithmetic for how it's derived.
- Only reference a metric or field name if it actually appears in the column list provided above. Do not mention "revenue" (or any other field) as something to quantify unless it's literally one of the given columns — if there's no revenue column, describe the impact using the fields that ARE present (e.g. gross_margin_dollars, lost_revenue_estimate) instead of naming a metric that may not exist in this dataset.
- Do not use vague umbrella terms like "profit", "profitability", or "profit margin" — name the exact field instead (e.g. gross_margin_dollars). This applies to every variant of the word, not just the literal string "profit".
- Do not use causal language ("causing", "leads to", "drives down") for a relationship that's only ever been observed as a correlation or co-occurrence — say "coincide with" or "are associated with" instead of "cause."
- Do not reference a ratio, rate, or "relative to X" comparison unless you know it's a value the statistics actually compute — if unsure, describe the two quantities separately instead of a derived ratio.
- Do not claim or imply "conversion" or "retention" issues unless conversion/retention statistics are explicitly part of this dataset — for a plain sales/orders dataset, describe volume or margin patterns instead.
- Do not mention time trends, trends "over time," or trends by week/month/period — that is handled separately and conditionally elsewhere; if you include it here, it directly contradicts a guardrail that gets added after your text, so leave time entirely out of the objective.
- Do not include evidence rules, labeling instructions, or anything about avoiding estimates — none of that belongs in this objective statement.
- If describing disproportionate stockouts or revenue loss, phrase it as relative to computed segment totals or rates — never "relative to demand," since observed demand is censored (artificially capped) whenever a stockout occurs, so true demand isn't something the data can actually measure.

Example of a well-scoped objective for a similar dataset (for style/scope reference only, not to copy verbatim): "Identify which products by sku and category, regions, and acquisition channels are associated with the highest gross_margin_dollars and lowest lost_revenue_estimate. Surface segments where precomputed stockout_flag rates indicate inventory or conversion problems. Highlight only computed patterns involving units_sold, unit_price, discount_rate, and gross_margin_dollars."

Output the objective text only — no preamble, no explanation, no section headers."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": user_message}]
        )
        objective_text = response.content[0].text.strip()
        structured_prompt = _build_steering_block(objective_text, stats).strip()
        return {"prompt": structured_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_clarify_prompt(column_summary: str) -> str:
    """Build the prompt for the clarifying-questions AI call.

    Args:
        column_summary: A compact text description of the dataset's columns,
            their types, and sample values.

    Returns:
        The full prompt string to send to the AI.
    """
    return f"""You are a senior data analyst preparing to generate a Data Intelligence Report.
You have been given a summary of a dataset's columns. Your job is to decide whether any columns are genuinely ambiguous in a way that would meaningfully change how the report interprets the data.

DATASET COLUMN SUMMARY:
{column_summary}

Identify columns whose meaning is unclear enough to affect the report's conclusions — for example:
- A numeric column that could be a count, a currency amount, or an ID
- A status/flag column where the values (e.g. 0/1, Y/N, A/B) could mean different things
- A date column where the time period's business significance is unclear
- Columns with values like "Free Shipping" and "Free Economy" that might or might not be the same thing

Do NOT ask about:
- Columns whose meaning is obvious from the name and values
- Columns that are clearly identifiers (IDs, emails, names)
- Statistical or formatting preferences
- Anything the report can reasonably infer without external context

Respond ONLY with a valid JSON array of question objects, maximum 3 questions.
If the data is clear enough to analyze without clarification, return an empty array [].

Format:
[
  {{
    "column": "column_name_this_question_is_about",
    "question": "Plain English question for the user, one sentence, specific to what you see in the data."
  }}
]

Return ONLY the JSON array, no explanation, no markdown fences."""


@app.get("/clarify")
async def clarify(
    file_id: str,
    target_col: Optional[str] = None,
    user_id: str = Depends(get_current_user)
) -> dict:
    """Check whether clarifying questions are needed before generating a report.

    Loads the dataset, builds a compact column summary, and asks the AI
    whether any columns are genuinely ambiguous. Returns 0-3 questions, or
    an empty list if the data is clear enough to proceed without clarification.

    Args:
        file_id: The stored dataset file ID.
        target_col: Optional target column hint for the report.
        user_id: Authenticated user ID from the request token.

    Returns:
        A dict with a "questions" key containing a list of question objects,
        each with "column" and "question" fields.
    """
    data_path = DATA_DIR / f"{file_id}.pkl"
    ensure_file_owner(file_id, user_id)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")
    try:
        df = pd.read_pickle(str(data_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load dataset: {e}")

    # Build a compact column summary — enough context for the AI to ask smart
    # questions without sending the full stats JSON (which is expensive and
    # unnecessary for this lightweight check).
    lines: list[str] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:6].tolist()
        sample_str = ", ".join(str(v) for v in sample_vals)
        lines.append(f"- {col} ({dtype}, {n_unique} unique values) — sample: {sample_str}")

    column_summary = "\n".join(lines)
    prompt = _build_clarify_prompt(column_summary)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = []
        # Cap at 3 regardless of what the model returns
        questions = questions[:3]
    except Exception:
        # If parsing fails or the call errors, silently skip questions
        # so the user can still generate their report unimpeded.
        questions = []

    return {"questions": questions}



def _fix_section_numbers(html: str) -> str:
    """Renumber report sections sequentially if the model skipped any numbers.

    Finds h2 tags that start with a digit and a period (e.g. "3. Statistical
    Deep Dive", "5. Pattern Analysis") and rewrites them as 1, 2, 3... in the
    order they appear. Only touches the leading number — the section title is
    preserved exactly.

    Args:
        html: The assembled report HTML string.

    Returns:
        HTML with section numbers corrected to be strictly sequential.
    """
    import re
    pattern = re.compile(r'(<h2[^>]*>)\s*(\d+)\.\s*(.+?)(</h2>)', re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(html))
    if not matches:
        return html

    # Check if there's actually a gap
    numbers = [int(m.group(2)) for m in matches]
    expected = list(range(1, len(numbers) + 1))
    if numbers == expected:
        return html  # already sequential, nothing to do

    print(f"[section-numbers] found {numbers}, renumbering to {expected}")
    # Replace from right to left to preserve offsets
    result = html
    for i, m in reversed(list(enumerate(matches))):
        correct_num = i + 1
        replacement = f"{m.group(1)}{correct_num}. {m.group(3)}{m.group(4)}"
        result = result[:m.start()] + replacement + result[m.end():]

    return result


def _finalize_truncated_report_html(html: str, notice: str) -> str:
    """Force-close an HTML report that was cut off mid-stream.

    This is a different repair than _repair_html_tags, which fixes
    MISMATCHED closing tags in an otherwise-complete document. A report cut
    off by a length or time budget has no closing tags at all for whatever
    was still open at the cutoff point — potentially a dozen or more nested
    tags — and needs every one of them closed in the correct order, not just
    a handful of mismatches corrected. Also trims any dangling partial
    opening tag (e.g. the stream stopped mid `<div cla`) and appends a
    visible, honest notice that generation was shortened.

    Args:
        html: The partial HTML collected before the cutoff.
        notice: A short, plain-language sentence noting the truncation,
            shown to the reader inline in the report itself.

    Returns:
        A syntactically complete HTML document — always has matching
        <body></body> and <html></html>, regardless of how deep the
        original content's tags were nested at the cutoff point.
    """
    from html.parser import HTMLParser

    block_tags = {
        "p", "div", "section", "article", "aside", "main",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "table", "thead", "tbody", "tr", "td", "th",
        "blockquote", "pre", "code", "span", "a", "strong", "em",
        "header", "footer", "nav", "figure", "figcaption",
    }
    auto_close = {"p", "li", "td", "th", "tr", "dt", "dd"}
    void_tags = {"br", "img", "input", "link", "meta", "hr", "source", "area", "base", "col", "embed", "track", "wbr"}

    class OpenStackFinder(HTMLParser):
        """Track which block-level tags are still open at the end of a
        possibly-truncated document, and whether <body>/<html> were both
        opened AND properly closed (opened alone is not enough — that's
        exactly the truncated case this function exists to fix)."""

        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.stack: list[str] = []
            self.body_open = False
            self.body_closed = False
            self.html_open = False
            self.html_closed = False

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag == "body":
                self.body_open = True
            if tag == "html":
                self.html_open = True
            if tag in void_tags or tag not in block_tags:
                return
            if tag in auto_close and self.stack and self.stack[-1] == tag:
                self.stack.pop()
            self.stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag == "body":
                self.body_closed = True
            if tag == "html":
                self.html_closed = True
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
            elif tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()

    # Trim any dangling partial tag the cutoff landed inside (e.g. the stream
    # stopped mid `<div cla`) — cut back to the last fully-written '>'.
    last_gt = html.rfind(">")
    if last_gt != -1 and last_gt < len(html) - 1:
        html = html[: last_gt + 1]

    finder = OpenStackFinder()
    try:
        finder.feed(html)
    except Exception:
        pass  # if the parser itself chokes, still proceed with what we have — closing tags below are additive, not destructive

    closing_tags = "".join(f"</{tag}>" for tag in reversed(finder.stack))
    notice_html = (
        f'<div style="margin-top:2rem;padding:1rem 1.25rem;background:rgba(255,214,0,.08);'
        f'border:1px solid rgba(255,214,0,.25);border-radius:8px;font-size:.85rem;'
        f'color:#f0faff">{notice}</div>'
    )

    result = html + closing_tags + notice_html
    if finder.body_open and not finder.body_closed:
        result += "</body>"
    if finder.html_open and not finder.html_closed:
        result += "</html>"
    return result


def _validate_report_quality(html: str) -> list[str]:
    """Check a fully repaired/audited report for hard rejection criteria.

    This runs LAST, after every repair pass (_repair_html_tags,
    _repair_corrupted_list_items, section number fixes, etc.) has already had
    a chance to fix what it can. Anything still present here means the
    report must be REJECTED, not shipped with a best-effort patch — these
    are specific, real corruption/completion signatures observed in actual
    shipped output, not speculative edge cases.

    Args:
        html: The fully repaired report HTML, ready for final output.

    Returns:
        A list of rejection reasons. Empty list means the report passes and
        is safe to return as a completed report.
    """
    reasons: list[str] = []

    exec_match = re.search(r'<ul class="exec-summary">(.*?)</ul>', html, re.DOTALL)
    if exec_match:
        li_count = len(re.findall(r'<li>', exec_match.group(1)))
        if li_count < 7:
            reasons.append(f"Executive Summary has only {li_count} <li> items (need at least 7).")
    else:
        reasons.append('Executive Summary <ul class="exec-summary"> block not found.')

    # A bullet whose content starts with a stray numeric fragment like "95)"
    # or "6%)" is a strong signal that a sentence was cut and merged with the
    # start of another — the exact defect observed in a real shipped report.
    fragment_starts = re.findall(r'<li>\s*(?:<strong>)?\s*\d+(?:\.\d+)?%?\)', html)
    if fragment_starts:
        reasons.append(
            f"Found {len(fragment_starts)} <li> starting with a corrupted numeric "
            f"fragment (e.g. {fragment_starts[0]!r})."
        )

    if re.search(r'%\.\s*\d+%', html):
        reasons.append("Found the '%. N%' corruption signature (merged sentence fragments).")
    if re.search(r'\)\.\d+\)', html):
        reasons.append("Found the ').N)' corruption signature (merged sentence fragments).")

    rec_count = len(re.findall(r'<div class="rec-headline">', html))
    if rec_count != 10:
        reasons.append(f"Found {rec_count} recommendations (need exactly 10).")

    return reasons


def _repair_corrupted_list_items(html: str) -> tuple[str, list[str]]:
    """Remove empty bullets and auto-close unclosed <li> elements.

    This is a different defect class than _repair_html_tags (which fixes a
    closing tag that names the WRONG element, e.g. closing a <li> with
    </h3>). This function handles two specific things observed in a real
    generated report: an <li> that opens, has content, but is never closed
    before the next <li> starts (browsers auto-close this visually, but it's
    still malformed HTML worth fixing at the source), and a completely empty
    <li></li> with no content at all — always a defect, never intentional.

    Also detects (but does not attempt to silently rewrite) a specific text-
    corruption signature: a closing paren immediately followed by a period,
    digits, and another closing paren (e.g. "Cohen's d = -0.66).95) confirms"),
    which is a strong signal that two sentences were merged with a chunk of
    text missing between them. Reconstructing the intended sentence isn't
    something that can be done safely/automatically, so this is surfaced as
    a warning for review rather than guessed at and silently rewritten.

    Args:
        html: The report HTML to repair.

    Returns:
        A tuple (repaired_html, warnings) — warnings is empty if nothing
        suspicious was found.
    """
    warnings_found: list[str] = []

    # Remove genuinely empty bullets (with only whitespace between the tags).
    empty_li_count = len(re.findall(r'<li>\s*</li>', html))
    if empty_li_count:
        html = re.sub(r'<li>\s*</li>', '', html)
        warnings_found.append(f"Removed {empty_li_count} empty <li></li> bullet(s).")

    # Auto-close an <li> that runs straight into the next <li> without ever
    # closing — insert the missing </li> right before the next opening tag.
    unclosed_pattern = re.compile(r'(<li\b[^>]*>(?:(?!</li>|<li\b).)*?)(?=<li\b)', re.DOTALL)
    html, n_closed = unclosed_pattern.subn(r'\1</li>', html)
    if n_closed:
        warnings_found.append(f"Auto-closed {n_closed} <li> element(s) missing their closing tag.")

    # Flag (do not rewrite) the specific corruption signature: ").<digits>)"
    # with no space/operator before the digits — e.g. "-0.66).95) confirms".
    corruption_matches = re.findall(r'\)\.\d+\)', html)
    if corruption_matches:
        warnings_found.append(
            f"POSSIBLE TEXT CORRUPTION: found {len(corruption_matches)} instance(s) of a "
            f"').<digits>)' pattern (e.g. {corruption_matches[0]!r}) — this usually means two "
            f"sentences were merged with text missing between them. Needs manual review; not "
            f"auto-repaired since reconstructing the intended sentence isn't safe to guess at."
        )

    return html, warnings_found


def _repair_html_tags(html: str) -> str:
    """Repair mismatched HTML closing tags in generated report HTML.

    Uses a window-based scan to find closing tags that don't match the most
    recently opened block-level tag, and replaces them with the correct closer.
    Handles the common AI generation mistake of closing with </h3> when the
    open tag was <p>, or closing with </p> when the open tag was <li>.

    Non-blocking — returns original HTML unchanged if any error occurs.

    Args:
        html: The assembled report HTML string to repair.

    Returns:
        The HTML string with mismatched closing tags corrected.
    """
    try:
        from html.parser import HTMLParser

        # Block-level tags we track for mismatch detection.
        # Void elements (br, hr, img, input, etc.) are intentionally excluded
        # since they have no closing tag.
        BLOCK_TAGS = {
            "p", "div", "section", "article", "aside", "main",
            "h1", "h2", "h3", "h4", "h5", "h6",
            "ul", "ol", "li", "table", "thead", "tbody", "tr", "td", "th",
            "blockquote", "pre", "code", "span", "a", "strong", "em",
            "header", "footer", "nav", "figure", "figcaption",
        }

        # Auto-closing tags — opening one implicitly closes the previous same tag.
        # e.g. <p> followed by another <p> closes the first without an explicit </p>.
        AUTO_CLOSE = {"p", "li", "td", "th", "tr", "dt", "dd"}

        class MismatchFinder(HTMLParser):
            """Collect (open_tag, position) stack and mismatched close positions."""

            def __init__(self) -> None:
                super().__init__(convert_charrefs=False)
                self.stack: list[tuple[str, int]] = []  # (tag, char_offset)
                self.mismatches: list[tuple[int, int, str, str]] = []
                # (close_start, close_end, wrong_tag, correct_tag)
                self._html = ""

            def feed_html(self, html_str: str) -> None:
                self._html = html_str
                self.feed(html_str)

            def handle_starttag(self, tag: str, attrs: list) -> None:
                if tag not in BLOCK_TAGS:
                    return
                if tag in AUTO_CLOSE and self.stack and self.stack[-1][0] == tag:
                    self.stack.pop()
                self.stack.append((tag, self.getpos()))

            def handle_endtag(self, tag: str) -> None:
                if tag not in BLOCK_TAGS:
                    return
                if not self.stack:
                    return
                expected_tag, _ = self.stack[-1]
                if tag != expected_tag:
                    # Find the position of this closing tag in the raw HTML
                    # by scanning from the current parser position
                    line, col = self.getpos()
                    lines = self._html.split("\n")
                    # Approximate char offset for the line
                    char_offset = sum(len(l) + 1 for l in lines[:line - 1]) + col
                    close_str = f"</{tag}>"
                    # Search near char_offset for the exact close tag
                    window = self._html[max(0, char_offset - 5):char_offset + len(close_str) + 5]
                    idx = self._html.find(close_str, max(0, char_offset - 5))
                    if idx != -1:
                        self.mismatches.append((
                            idx,
                            idx + len(close_str),
                            tag,          # wrong closing tag
                            expected_tag, # what it should be
                        ))
                    self.stack.pop()
                else:
                    self.stack.pop()

        finder = MismatchFinder()
        finder.feed_html(html)

        if not finder.mismatches:
            return html

        # Apply fixes from right to left so character offsets stay valid
        result = html
        fixed = 0
        for start, end, wrong, correct in sorted(finder.mismatches, reverse=True):
            bad_close = f"</{wrong}>"
            good_close = f"</{correct}>"
            # Verify the text at this position still matches (earlier fixes may
            # have shifted things slightly — skip if not an exact match)
            if result[start:end] == bad_close:
                result = result[:start] + good_close + result[end:]
                fixed += 1
                print(f"[html-repair] fixed </{wrong}> → </{correct}> at char {start}")

        if fixed:
            print(f"[html-repair] {fixed} tag(s) repaired")
        return result

    except Exception as e:
        print(f"[html-repair] failed: {e}")
        return html


def _validate_rankings(html: str, stats: dict) -> str:
    """Deterministically fix ranking claims that don't match computed segment data.

    Covers two high-confidence patterns only:
    1. Explicit A > B > C ranking strings (e.g. "Apparel > Outdoor > Home Goods")
    2. "top N categories/channels/regions by lost revenue/revenue/impact" phrases

    Ground truth comes only from segment_impact_summary[outcome_col]
    .top_segments_by_primary_impact. If the dimension or metric can't be
    mapped cleanly to a computed ranking, the claim is left alone and a skip
    is logged — no guessing.

    Args:
        html: The report HTML after audit.
        stats: The full computed stats dict.

    Returns:
        The HTML with incorrect ranking claims replaced by data-backed sentences.
    """
    import re

    # ── Extract ground truth from segment_impact_summary ──────────────────
    # Flatten all outcomes into a single dict keyed by dimension name
    # (channel, category, region, etc.) → ordered list of segment labels
    # from top_segments_by_primary_impact.
    ranked_by_dim: dict[str, list[str]] = {}
    summary = stats.get("segment_impact_summary", {})
    if isinstance(summary, dict):
        # May be keyed by outcome_col or be a flat dict
        for v in summary.values():
            if not isinstance(v, dict):
                continue
            top_segs = v.get("top_segments_by_primary_impact", [])
            for entry in top_segs:
                seg_label = entry.get("segment", "")
                # seg_label looks like "category=Apparel" or "channel=Shopify"
                if "=" in seg_label:
                    dim, val = seg_label.split("=", 1)
                    dim = dim.strip().lower()
                    val = val.strip()
                    if dim not in ranked_by_dim:
                        ranked_by_dim[dim] = []
                    if val not in ranked_by_dim[dim]:
                        ranked_by_dim[dim].append(val)

    if not ranked_by_dim:
        print("[ranking-validator] no segment_impact_summary data — skipping")
        return html

    FALLBACK = "The highest-impact segments are listed in the segment analysis above."

    # ── Pattern 1: A > B > C ranking strings ──────────────────────────────
    # Each segment item must start with an uppercase letter to avoid capturing
    # trailing prose words like "Electronics based on stockout exposure."
    gt_pattern = re.compile(
        r'([A-Z][A-Za-z0-9&\'\-]*(?:\s+[A-Z][A-Za-z0-9&\'\-]*)*'
        r'(?:\s*>\s*[A-Z][A-Za-z0-9&\'\-]*(?:\s+[A-Z][A-Za-z0-9&\'\-]*)*)+)'
    )

    def _fix_gt_match(m: re.Match) -> str:
        full_match = m.group(0)
        parts = [p.strip() for p in re.split(r'\s*>\s*', full_match)]
        if len(parts) < 2:
            return full_match

        # Try to identify which dimension these belong to
        matched_dim = None
        for dim, ground_truth in ranked_by_dim.items():
            gt_lower = [g.lower() for g in ground_truth]
            parts_lower = [p.lower() for p in parts]
            # At least half the parts must appear in the ground truth
            overlap = sum(1 for p in parts_lower if p in gt_lower)
            if overlap >= max(2, len(parts) // 2):
                matched_dim = dim
                break

        if matched_dim is None:
            print(f"[ranking-validator] skip (no dim match): {full_match[:60]}")
            return full_match  # leave it alone

        ground_truth = ranked_by_dim[matched_dim]
        # Build correct order: only include parts that appear in ground truth
        parts_lower = {p.lower(): p for p in parts}
        gt_lower_map = {g.lower(): g for g in ground_truth}
        correct_order = [
            gt_lower_map[g] for g in [g.lower() for g in ground_truth]
            if g in parts_lower
        ]
        correct_str = " > ".join(correct_order) if correct_order else None

        if correct_str and correct_str != full_match:
            print(f"[ranking-validator] fixed ranking: '{full_match[:50]}' → '{correct_str[:50]}'")
            return correct_str
        return full_match

    html = gt_pattern.sub(_fix_gt_match, html)

    # ── Pattern 2: "top N X by Y" phrases ─────────────────────────────────
    top_n_pattern = re.compile(
        r'top\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)'
        r'\s+(categor(?:y|ies)|channel[s]?|region[s]?|segment[s]?)'
        r'\s+by\s+(lost\s+revenue|revenue|primary\s+impact|impact|business\s+impact)',
        re.IGNORECASE
    )

    dim_map = {
        "category": "category", "categories": "category",
        "channel": "channel", "channels": "channel",
        "region": "region", "regions": "region",
        "segment": None, "segments": None,  # too vague — skip
    }

    word_to_n = {
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    def _fix_top_n_match(m: re.Match) -> str:
        n_raw = m.group(1).lower()
        n = word_to_n.get(n_raw) or (int(n_raw) if n_raw.isdigit() else None)
        if n is None:
            print(f"[ranking-validator] skip top-N (can't parse n): {m.group(0)[:60]}")
            return m.group(0)
        dim_word = m.group(2).lower()
        dim_word = re.sub(r'ies$', 'y', dim_word)  # categories → category
        dim_word = dim_word.rstrip("s")              # channels → channel
        dim = dim_map.get(dim_word) or dim_map.get(dim_word + "y")
        if dim is None or dim not in ranked_by_dim:
            print(f"[ranking-validator] skip top-N (no dim): {m.group(0)[:60]}")
            return m.group(0)

        top = ranked_by_dim[dim][:n]
        if not top:
            return m.group(0)

        dim_plural = dim[:-1] + "ies" if dim.endswith("y") else dim + "s"

        # Format the list naturally: "A, B, and C"
        if len(top) == 1:
            top_list = top[0]
        elif len(top) == 2:
            top_list = f"{top[0]} and {top[1]}"
        else:
            top_list = ", ".join(top[:-1]) + f", and {top[-1]}"

        clean_sentence = f"The top {len(top)} {dim_plural} by primary impact are {top_list}."
        print(f"[ranking-validator] fixed top-N: '{m.group(0)[:50]}' → '{clean_sentence[:60]}'")

        # Use prefix+suffix sentinels so the expansion loop can find the exact
        # span of the clean sentence without ambiguity.
        _SENTINEL_PREFIX = "\x00TOPN\x00"
        _SENTINEL_SUFFIX = "\x00ENDTOPN\x00"
        return f"{_SENTINEL_PREFIX}{clean_sentence}{_SENTINEL_SUFFIX}"

    # Apply top-N fixes (with sentinels marking matches for sentence expansion)
    _SENTINEL_PREFIX = "\x00TOPN\x00"
    _SENTINEL_SUFFIX = "\x00ENDTOPN\x00"
    html = top_n_pattern.sub(_fix_top_n_match, html)

    # Expand each sentinel to replace its full containing sentence
    sentinel_re = re.compile(
        re.escape(_SENTINEL_PREFIX) + r'(.*?)' + re.escape(_SENTINEL_SUFFIX),
        re.DOTALL
    )

    def _expand_sentence(m2: re.Match) -> str:
        # We can't do boundary expansion inside sub() — do it in post-processing loop
        return m2.group(1)  # just return the clean sentence for now

    # Process sentinels one at a time so we can do boundary expansion on full html
    while _SENTINEL_PREFIX in html:
        si = html.find(_SENTINEL_PREFIX)
        ei = html.find(_SENTINEL_SUFFIX, si)
        if ei == -1:
            # Malformed sentinel — just strip the prefix and move on
            html = html[:si] + html[si + len(_SENTINEL_PREFIX):]
            continue

        clean_sentence = html[si + len(_SENTINEL_PREFIX):ei]
        after_sentinel_end = ei + len(_SENTINEL_SUFFIX)

        # Expand LEFT to find start of the containing sentence
        left_search = html[:si]
        left_boundaries = {
            ".": left_search.rfind("."),
            "!": left_search.rfind("!"),
            "?": left_search.rfind("?"),
            "<p>": left_search.rfind("<p>"),
            "<li>": left_search.rfind("<li>"),
        }
        best_left_key = max(left_boundaries, key=left_boundaries.get, default=None)
        best_left = left_boundaries.get(best_left_key, -1) if best_left_key else -1
        if best_left == -1:
            sent_start = 0
        elif best_left_key in ("<p>", "<li>"):
            # Advance past the full opening tag so we don't leave a stray '<'
            sent_start = best_left + len(best_left_key)
        else:
            sent_start = best_left + 1

        # Expand RIGHT: skip past the rest of the original sentence.
        # The original sentence ends at the next . ! ? or block close tag
        # AFTER the sentinel ends.
        right_search = html[after_sentinel_end:]
        right_boundaries = {
            ".": right_search.find("."),
            "!": right_search.find("!"),
            "?": right_search.find("?"),
        }
        valid_rights = {k: v for k, v in right_boundaries.items() if v != -1}
        if valid_rights:
            first_right_key = min(valid_rights, key=valid_rights.get)
            first_right = valid_rights[first_right_key]
            sent_end = after_sentinel_end + first_right + 1  # include the punctuation
        else:
            sent_end = after_sentinel_end  # no sentence end found — stop here

        # Build the replacement
        prefix = html[:sent_start].rstrip()
        suffix = html[sent_end:].lstrip()
        html = (prefix + " " + clean_sentence + " " + suffix).strip()
        print(f"[ranking-validator] sentence replaced: '{clean_sentence[:70]}'")

    # Defensive cleanup — strip any leftover sentinels that the expansion loop
    # may have missed (e.g. malformed HTML that had no sentence boundary).
    if _SENTINEL_PREFIX in html or _SENTINEL_SUFFIX in html:
        print("[ranking-validator] WARNING: leftover sentinels found — stripping")
        html = html.replace(_SENTINEL_PREFIX, "").replace(_SENTINEL_SUFFIX, "")

    return html


# Completeness-enforcement tuning. A dimension is only eligible for tier
# completeness checks when it has at most this many distinct values. This is
# deliberately well below the pipeline's own truncation ceiling of 20 (see
# auto_segment_rates' .head(20) in the stats builder) — dimensions in the
# 13-20 "medium cardinality" range are ones where the report is SUPPOSED to
# discuss only the top few by impact (highest_primary_impact_segment), not
# enumerate every value. 12 covers genuinely small, enumerable business
# dimensions (subscription tiers, regions, quarters, weekday buckets) without
# forcing full enumeration onto columns where selective top-N reporting is
# the intended, correct behavior. Raise only if your users' datasets commonly
# have 13+ tiers they'd expect fully enumerated rather than ranked.
_SEGMENT_MAX_TIERS: int = 12

# A tier is considered "covered" in the narrative only if its label appears as a
# whole token within this many characters of a percentage figure. This avoids
# treating an incidental generic-word match (e.g. the word "growth" in prose, or
# a region literally named "Central") as evidence the tier's rate was stated.
_SEGMENT_RATE_PROXIMITY_CHARS: int = 130


def _segment_label_covered(section: str, label: str, proximity: int) -> bool:
    """Report whether a segment tier's rate appears to be stated in the section.

    Coverage requires the label to appear as a whole token (not as a substring
    of a larger word) within ``proximity`` characters of a percentage figure
    (a ``%`` sign or the word "percent"). This is deliberately stricter than a
    bare substring test so generic-word tier labels ("Other", "Standard", a
    region named "Central") are not falsely counted as covered when they merely
    occur incidentally in surrounding prose.

    Args:
        section: The Segment Analysis section HTML (any case).
        label: The tier label to look for (e.g. "Scale", "Paid Search").
        proximity: Character window on each side of a label match within which a
            percentage figure must appear for the tier to count as covered.

    Returns:
        True if the label is present as a token near a percentage figure, or if
        the label is empty (nothing sensible to check); False otherwise.
    """
    import re

    clean_label = label.strip()
    if not clean_label:
        return True

    # Boundary via alphanumeric lookarounds rather than \b so multi-word labels
    # and labels containing "&" or "-" (e.g. "Home & Garden") behave correctly.
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(clean_label) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    for match in pattern.finditer(section):
        low = max(0, match.start() - proximity)
        high = min(len(section), match.end() + proximity)
        window = section[low:high].lower()
        if "%" in window or "percent" in window:
            return True
    return False


def _validate_segment_completeness(html: str, stats: dict) -> str:
    """Ensure every tier of each low-cardinality segment dimension appears in the report.

    The model sometimes narrates only a subset of a categorical dimension's
    values in the Segment Analysis section (e.g. describing three of four plan
    tiers and silently dropping the rest). For each dimension with a small,
    fully-enumerated set of tiers, this verifies each tier label is present and
    appends one compact, data-backed sentence for any that are missing. Ground
    truth comes only from auto_segment_rates; if the section anchor or a tier's
    rate cannot be resolved, that dimension is skipped rather than guessed at.

    This is a safety net behind the hard-constraint enumeration in
    _build_hard_constraints, which is the primary (prevention) mechanism.

    Args:
        html: The report HTML after ranking validation.
        stats: The full computed stats dict for this report run.

    Returns:
        The HTML with a data-backed sentence appended for any omitted tier, or
        the original HTML unchanged if nothing is missing or safe to add.
    """
    auto_rates = stats.get("auto_segment_rates", {})
    if not isinstance(auto_rates, dict) or not auto_rates:
        return html

    # Anchor on the section heading text, which is stable regardless of the
    # section number assigned by _fix_section_numbers.
    lower_html = html.lower()
    heading_end = lower_html.find("segment analysis</h2>")
    if heading_end == -1:
        print("[completeness] no Segment Analysis section found — skipping")
        return html
    section_start = heading_end

    # The section ends at the next separator or <h2> after its start.
    sep_idx = html.find('<div class="separator">', section_start + 1)
    h2_idx = lower_html.find("<h2", section_start + 1)
    candidates = [i for i in (sep_idx, h2_idx) if i != -1]
    section_end = min(candidates) if candidates else len(html)
    section = html[section_start:section_end]

    additions: list[str] = []
    for key, dim in auto_rates.items():
        if not isinstance(dim, dict):
            continue
        rates = dim.get("rates", {})
        seg_col = dim.get("segment_col", "segment")
        overall_pct = dim.get("overall_rate_pct")
        if not (isinstance(rates, dict) and 0 < len(rates) <= _SEGMENT_MAX_TIERS):
            continue
        if overall_pct is None:
            continue

        missing_lines: list[str] = []
        for seg_label, seg_data in rates.items():
            label = str(seg_label)
            if _segment_label_covered(section, label, _SEGMENT_RATE_PROXIMITY_CHARS):
                continue
            if not isinstance(seg_data, dict):
                continue
            rate = seg_data.get("rate")
            if rate is None:
                continue
            rate_pct = round(float(rate) * 100, 1)
            pp = seg_data.get("pp_above_overall")
            if pp is None:
                gap_txt = ""
            else:
                pp_pts = round(abs(float(pp)) * 100, 1)
                direction = "above" if float(pp) >= 0 else "below"
                gap_txt = f" ({pp_pts}pp {direction} the {overall_pct}% average)"
            missing_lines.append(f"{label} shows a {rate_pct}% rate{gap_txt}.")

        if missing_lines:
            joined = " ".join(missing_lines)
            additions.append(
                f'<p><strong>Additional {seg_col} segments (for completeness):'
                f'</strong> {joined}</p>'
            )
            print(f"[completeness] appended {len(missing_lines)} missing tier(s) for {key}")

    if not additions:
        return html

    insert_block = "\n" + "\n".join(additions) + "\n"
    return html[:section_end] + insert_block + html[section_end:]


def _build_steering_block(dataset_context: Optional[str], stats: Optional[dict] = None) -> str:
    """Render a user-supplied report focus/steering instruction as a
    structured, sectioned template — not a single paragraph.

    Two upgrades over a plain paragraph, both applied automatically by the
    prompt BUILDER rather than depending on the user writing a well-formed
    instruction:

    1. Sectioned output (REPORT OBJECTIVE / ANALYSIS FOCUS / METRICS TO
       PRIORITIZE / EVIDENCE RULES / LABELING RULES / DO NOT DO) gives the
       model separate, unambiguous "slots" to obey instead of one paragraph
       it can partially miss.
    2. ANALYSIS FOCUS and METRICS TO PRIORITIZE are populated dynamically
       FROM THE ACTUAL DATASET's computed stats (categorical and numeric
       columns actually present), not hardcoded to one dataset's field names
       like gross_margin_dollars/stockout_flag. A healthcare or marketing
       dataset gets its own real columns listed here, not ecommerce fields
       that don't exist in its data. Each categorical column is framed as
       "X segment patterns" (deterministic string formatting, not a second API call),
       and a "Stockout/lost revenue problem areas" line is added whenever
       stockout_flag or lost_revenue_estimate exist in the data.

    The user's own dataset_context text becomes the REPORT OBJECTIVE
    verbatim — this function does not attempt to rewrite or reinterpret it,
    only to structure the guardrails and dynamic metric list around it.

    Args:
        dataset_context: The raw user-supplied focus/steering text, or None
            if the user didn't provide one.
        stats: The computed dataset statistics (from compute_dataset_stats),
            used to populate the segment/metric lists dynamically. If not
            provided, those sections fall back to a generic placeholder
            rather than guessing at field names.

    Returns:
        The structured steering block ready to append to the prompt context,
        or an empty string if dataset_context was empty/None.
    """
    if not dataset_context:
        return ""

    stats = stats or {}
    all_columns = stats.get("columns") or []
    categorical_cols = list((stats.get("categorical_analysis") or {}).keys())[:10]
    numeric_cols = [
        col for col in (stats.get("numeric_analysis") or {}).keys()
        if not _ID_NAME_PATTERN.search(str(col))
    ][:12]

    # Directive "X segment patterns" framing instead of a bare column-name list — cheap
    # to derive deterministically from the same columns already extracted
    # for METRICS TO PRIORITIZE, no second API call needed to get this. A
    # dedicated stockout/lost-revenue line is added only when those specific
    # fields exist, since that's a distinct analysis focus (problem areas,
    # not just performance drivers) worth calling out on its own.
    #
    # Date/time-label columns (week, month, etc.) are deliberately EXCLUDED
    # from "segment patterns" framing — a raw time label with no computed time_series
    # stats behind it invites the model to invent a trend from a bare label,
    # which is exactly the failure mode this is guarding against. If such a
    # column exists, a plain caveat bullet replaces it instead.
    has_time_like_column = any(_TIME_LIKE_NAME_PATTERN.search(str(c)) for c in categorical_cols)
    segment_lines = [f"- {c} segment patterns" for c in categorical_cols if not _TIME_LIKE_NAME_PATTERN.search(str(c))]
    if "stockout_flag" in all_columns or "lost_revenue_estimate" in all_columns:
        segment_lines.append("- Stockout/lost revenue problem areas")
    if has_time_like_column:
        segment_lines.append("- Time trends only if time_series statistics were explicitly provided")
    segment_section = "\n".join(segment_lines) if segment_lines else "- (no categorical segment columns detected in this dataset)"
    metrics_section = "\n".join(f"- {c}" for c in numeric_cols) if numeric_cols else "- (no numeric metric columns detected in this dataset)"

    # Only mention these specific fields if they actually exist in THIS
    # dataset — a template that references fields absent from the data is
    # worse than no mention at all, since it invites the model to go looking
    # for something that isn't there.
    forecast_note = (
        "\n- forecast_demand_units: use only as context, never as proof of forecast accuracy — "
        "a correlation with units_sold does not confirm the forecast is accurate when stockouts "
        "or supply constraints exist."
    ) if "forecast_demand_units" in all_columns else ""
    stockout_note = (
        "\n- For stockout_flag specifically, cite only the precomputed segment rates provided in "
        "the statistics. Do not infer or calculate a stockout rate for any segment not included."
    ) if "stockout_flag" in all_columns else ""
    discount_rate_note = (
        "\n- Treat discount_rate as a rate only, not a summable revenue or impact metric."
    ) if "discount_rate" in all_columns else ""

    # Conditional, field-presence-gated rules — same pattern as forecast_note/
    # stockout_note/discount_rate_note above. These mirror protections that
    # already exist in the main report-generation system prompt, but that
    # prompt is invisible to anyone just reading this function's own output
    # (e.g. the /build-prompt preview) — so this function needs to be
    # self-sufficient on its own, not rely on a second prompt the reader
    # never sees.
    stockout_discount_rule = (
        "\n- Do not recommend discounting products that are stocked out or at risk of stocking out — "
        "discounting accelerates depletion and worsens the stockout. Use substitutes, available "
        "alternatives, overstock discounting, demand shifting, or inventory allocation adjustments instead."
    ) if "stockout_flag" in all_columns else ""
    margin_conflation_rule = (
        "\n- Do not treat lost_revenue_estimate and gross_margin_dollars as the same metric or add them "
        "together — do not rename either figure, or their sum, as \"profit,\" \"lost profit,\" or \"total margin.\""
    ) if "lost_revenue_estimate" in all_columns and "gross_margin_dollars" in all_columns else ""

    return f"""

REPORT OBJECTIVE
{dataset_context}

ANALYSIS FOCUS
{segment_section}

METRICS TO PRIORITIZE
{metrics_section}{forecast_note}

EVIDENCE RULES
- Use only actual computed statistics for all segment metrics.
- Never estimate values for segments, time periods, or outcomes not directly present in the data.{stockout_note}{discount_rate_note}

LABELING RULES
- Label claims as "confirmed by data" when backed by computed statistics.
- Label suggestions as "hypothesis/recommendation" when they go beyond what the numbers directly show.

DO NOT DO
- Do not invent recovery amounts, lift percentages, conversion rates, retention rates, event counts, model accuracy figures, or new ratios unless those exact values are present in the computed statistics.
- Do not use causal language ("causes," "drives," "leads to") for a relationship that is only ever observed as a correlation or co-occurrence — use association language ("is associated with," "coincides with") unless the data explicitly supports a causal claim.
- Do not extrapolate financial impacts.
- Do not annualize figures.
- Do not claim forecast accuracy from a correlation between forecast_demand_units and units_sold when stockouts or supply constraints exist.
- Deprioritize traffic or vanity metrics unless they directly connect to revenue, margin, units, or computed conversion behavior.{stockout_discount_rule}{margin_conflation_rule}"""


def _build_hard_constraints(stats: dict) -> str:
    """Build a plain-text block of mandatory facts derived deterministically from stats.

    These are injected into the user message as MANDATORY FACTS so the model
    cannot substitute its own inferences. Currently covers low-cardinality
    numeric column tier descriptions where the model consistently invents
    wrong values from skewness/mean rather than reading the actual value_counts.

    Args:
        stats: The full computed stats dict for this report run.

    Returns:
        A newline-separated string of hard constraints, or empty string if none.
    """
    lines: list[str] = []
    for col, col_data in stats.get("numeric_analysis", {}).items():
        vc = col_data.get("value_counts")
        if not vc:
            continue
        non_zero = [
            f"{round(float(k)*100)}%" if float(k) < 1 else str(round(float(k), 4))
            for k in sorted(vc.keys(), key=float)
            if float(k) > 0
        ]
        if not non_zero:
            continue
        zero_count = vc.get("0.0", vc.get("0", 0))
        total = sum(vc.values())
        zero_pct = round(zero_count / total * 100, 1) if total else 0
        tier_str = (
            ", ".join(non_zero[:-1]) + ", and " + non_zero[-1]
            if len(non_zero) > 1 else non_zero[0]
        )
        lines.append(
            f"- {col}: {zero_pct}% of records have a value of 0. "
            f"The only non-zero values that exist are: {tier_str}. "
            f"Do not describe any other values or tiers for this column."
        )

    # Segment coverage — force the model to name EVERY tier of each
    # low-cardinality categorical dimension so none is silently dropped from
    # the Segment Analysis section (root-cause prevention for omitted tiers).
    for dim in stats.get("auto_segment_rates", {}).values():
        if not isinstance(dim, dict):
            continue
        rates = dim.get("rates", {})
        seg_col = dim.get("segment_col")
        overall_pct = dim.get("overall_rate_pct")
        if not (isinstance(rates, dict) and seg_col and 0 < len(rates) <= _SEGMENT_MAX_TIERS):
            continue
        tier_bits: list[str] = []
        for seg_label, seg_data in rates.items():
            if not isinstance(seg_data, dict):
                continue
            rate = seg_data.get("rate")
            if rate is None:
                continue
            tier_bits.append(f"{seg_label} ({round(float(rate) * 100, 1)}%)")
        if tier_bits:
            lines.append(
                f"- {seg_col}: the Segment Analysis section MUST include one "
                f"explicit sentence for EVERY value listed here, each with its "
                f"exact rate. Values: {', '.join(tier_bits)}. "
                f"Overall rate: {overall_pct}%. Do not omit any value."
            )
    return "\n".join(lines)


def _inject_tier_notes(stats: dict) -> dict:
    """Inject canonical value-count tier descriptions into stats before prompting.

    This makes the actual discount/price tier distribution unavoidable in the
    report prompt so the model cannot invent tiers from skewness/mean/histogram.
    """
    tier_notes = {}
    for col, col_data in stats.get("numeric_analysis", {}).items():
        vc = col_data.get("value_counts")
        if not vc:
            continue
        non_zero = [
            f"{round(float(k)*100)}%"
            for k in sorted(vc.keys(), key=float)
            if float(k) > 0
        ]
        if non_zero:
            tier_str = (
                ", ".join(non_zero[:-1]) + ", and " + non_zero[-1]
                if len(non_zero) > 1 else non_zero[0]
            )
            tier_notes[col] = (
                f"Exact tiers present in data: {tier_str}. "
                "Do not invent or imply other tiers."
            )
    if not tier_notes:
        return stats
    result = dict(stats)
    result["_canonical_value_tiers"] = tier_notes
    return result


def _build_banned_phrase_replacements(stats: dict) -> list[dict]:
    """Return a list of deterministic banned-phrase patch items based on stats.

    These are semantic errors that can be caught without AI inference — phrases
    that are wrong by definition given the fields present (or absent) in the stats.
    Each item is in the same format as AI audit patch items so _audit_report_html
    can apply them identically. Uses replacement text rather than removal so the
    report doesn't get awkward holes.

    Args:
        stats: The full computed stats dict for this report run.

    Returns:
        A list of patch dicts, each with "bad", "fix", "type", and "source_stat".
    """
    patches: list[dict] = []
    has_forecast_error = any(
        k in stats.get("numeric_analysis", {})
        for k in ("forecast_error", "mape", "mae", "forecast_accuracy")
    )
    has_stockout_flag = "stockout_flag" in (stats.get("columns") or [])
    has_annualized = any(
        k.startswith("annualized_") for k in stats.get("numeric_analysis", {})
    )
    has_mrr = "mrr" in [str(c).lower() for c in (stats.get("columns") or [])]

    # Discount / value-count tiers
    # Value counts live at numeric_analysis[col]["value_counts"] — string keys
    discount_vc = (
        stats.get("numeric_analysis", {})
        .get("discount_rate", {})
        .get("value_counts")
    )

    # Build canonical tier string from actual value_counts so any invented
    # tier description can be replaced with the real distribution.
    if discount_vc:
        non_zero_tiers = [
            f"{round(float(k)*100)}%"
            for k in sorted(discount_vc.keys(), key=float)
            if float(k) > 0
        ]
        discount_tier_str = (
            ", ".join(non_zero_tiers[:-1]) + ", and " + non_zero_tiers[-1]
            if len(non_zero_tiers) > 1 else
            non_zero_tiers[0] if non_zero_tiers else "various levels"
        )
        zero_count = discount_vc.get("0.0", discount_vc.get("0", 0))
        total_count = sum(discount_vc.values())
        discount_zero_pct = round(zero_count / total_count * 100, 1) if total_count else 0
    else:
        discount_tier_str = None
        discount_zero_pct = None


    # Region spread — requires computed region segment rates
    region_spread_computed = any(
        "region" in k.lower()
        for k in stats.get("auto_segment_rates", {})
    )

    banned: list[tuple[str, str, str]] = [
        # (bad_phrase, replacement, type_label)

        # Forecast accuracy from censored sales — cover the specific phrases
        # that survive the softer prompt rule and appear in actual output.
        # Using case-insensitive matching so capitalisation variants are caught.
        (
            "forecast accuracy is excellent",
            "forecast demand and units sold are strongly correlated, but this does not confirm forecast accuracy — when stockouts occur, units_sold is capped by available inventory and does not reflect true demand",
            "censored_sales_claim",
        ),
        (
            "forecasts are accurate when inventory is available",
            "forecast demand and observed sales are strongly correlated, but this correlation is not evidence of forecast accuracy — stockouts cap units_sold, making the comparison unreliable",
            "censored_sales_claim",
        ),
        (
            "predicted demand matches actual sales almost perfectly",
            "forecast demand and observed sales are strongly correlated, though observed sales are capped by available inventory during stockout periods and do not reflect true demand",
            "censored_sales_claim",
        ),
        (
            "the business does not have a demand prediction problem",
            "the correlation between forecast and sales is high, but this cannot rule out a demand prediction problem — stockouts suppress observed sales and make the forecast appear more accurate than it may be",
            "censored_sales_claim",
        ),
        (
            "forecasts accurately predict sales when inventory is available",
            "forecast demand and observed sales are strongly correlated, though sales are inventory-constrained during stockout periods, making this correlation an unreliable measure of forecast accuracy",
            "censored_sales_claim",
        ),
        (
            "forecast is highly accurate",
            "forecast demand and units sold are strongly correlated, but this does not confirm forecast accuracy — observed sales may be inventory-constrained",
            "censored_sales_claim",
        ),
        (
            "forecasting is strong",
            "forecast demand correlates with sales, though sales may be capped by inventory during stockout periods",
            "censored_sales_claim",
        ),
        (
            "forecasting performance is strong",
            "forecast demand correlates with sales, though stockouts cap observed sales and make this an unreliable accuracy measure",
            "censored_sales_claim",
        ),
    ]

    if not has_forecast_error and has_stockout_flag:
        for bad, fix, type_label in banned:
            patches.append({"bad": bad, "fix": fix, "type": type_label, "source_stat": "stockout_flag present, forecast_error absent"})

    # Annual/annualized claims without an annualized stat
    if not has_annualized:
        annualized_phrases = [
            ("annual lost revenue", "lost revenue in the analyzed period"),
            ("annualized lost revenue", "lost revenue in the analyzed period"),
            ("annual revenue loss", "revenue loss in the analyzed period"),
            ("per year", "in the analyzed period"),
            ("annually", "in the analyzed period"),
        ]
        for bad, fix in annualized_phrases:
            patches.append({"bad": bad, "fix": fix, "type": "unsupported_annualized_claim", "source_stat": "no annualized_* field in stats"})

    # Margin/revenue conflation
    margin_conflation = [
        ("margin recovery", "revenue recovery"),
        ("recover margin", "recover revenue"),
        ("margin impact of", "revenue impact of"),
    ]
    for bad, fix in margin_conflation:
        patches.append({"bad": bad, "fix": fix, "type": "metric_conflation", "source_stat": "lost_revenue_estimate != gross_margin_dollars"})

    # Invented lost-margin / lost-gross-profit arithmetic.
    # gross_margin_dollars is realized margin on sold units, not a lost-margin
    # figure. Adding it to lost_revenue_estimate produces unsupported totals.
    has_lost_margin_col = any(
        "lost_margin" in col.lower() or "lost_profit" in col.lower()
        for col in (stats.get("columns") or [])
    )
    if not has_lost_margin_col:
        lost_margin_phrases = [
            ("lost margin", "lost revenue"),
            ("lost gross profit", "lost revenue"),
            ("total lost gross profit", "total lost revenue"),
            ("lost gross margin", "lost revenue"),
        ]
        for bad, fix in lost_margin_phrases:
            patches.append({"bad": bad, "fix": fix, "type": "invented_metric", "source_stat": "no lost_margin or lost_profit column in dataset"})

    # Unsupported specific recovery estimates
    recovery_phrases = [
        ("could recover an estimated", "could reduce"),
        ("could recover approximately $", "could recover a meaningful portion of lost revenue"),
        ("could recover $890,000+", "could recover a meaningful portion of the $2.97M in lost revenue"),
        ("could recover $890", "could recover a meaningful portion of lost revenue"),
        ("recover $50K–$80K", "recover a meaningful portion of lost revenue"),
        ("recover $75K–$100K", "recover revenue in those segments"),
        ("recovering $240-400K", "reducing lost revenue meaningfully"),
        ("recovering $240", "reducing lost revenue meaningfully"),
        ("industry norms for wholesale distribution", "typical B2B wholesale expectations"),
        ("95%+ fill rates", "high fill rate standards expected by wholesale partners"),
        ("industry standard", "common operational targets"),
    ]
    for bad, fix in recovery_phrases:
        patches.append({"bad": bad, "fix": fix, "type": "unsupported_claim", "source_stat": "no source calculation in stats"})

    # Dramatic language not already caught by the tone rule.
    # NOTE: matching is case-insensitive substring (no word boundaries), so only
    # add tokens that are NOT substrings of common words. "crisis" is safe;
    # words like "dire" are not (they would corrupt "direction", "directly").
    # Order matters: the more specific "in crisis" precedes bare "crisis" so it
    # claims its instance first.
    dramatic_phrases = [
        ("hemorrhaging revenue", "losing revenue"),
        ("hemorrhaging", "losing"),
        ("catastrophic", "significant"),
        ("bleeding revenue", "reducing revenue"),
        ("in crisis", "severely at risk"),
        ("crisis", "severe difficulty"),
    ]
    for bad, fix in dramatic_phrases:
        patches.append({"bad": bad, "fix": fix, "type": "dramatic_language", "source_stat": "tone rule"})

    # MRR timeframe contradictions. "MRR" is Monthly Recurring Revenue by
    # definition, so "quarterly/annual/yearly MRR" is a contradiction in terms.
    # Relabeling to "monthly MRR" is always terminologically correct and, in the
    # common failure mode (a monthly figure mislabeled as quarterly), also fixes
    # the label/math mismatch. It does NOT correct a dollar amount that was itself
    # miscomputed as a multiple — the AI math-audit pass handles that case.
    if has_mrr:
        mrr_timeframe = [
            ("quarterly MRR", "monthly MRR"),
            ("annualized MRR", "monthly MRR"),
            ("annual MRR", "monthly MRR"),
            ("yearly MRR", "monthly MRR"),
        ]
        for bad, fix in mrr_timeframe:
            patches.append({"bad": bad, "fix": fix, "type": "mrr_timeframe_contradiction", "source_stat": "MRR is monthly by definition"})

    # Unsupported algorithm recommendations — only list platform-supported ones
    unsupported_algos = [
        ("XGBoost", "Gradient Boosting"),
        ("LightGBM", "Gradient Boosting"),
        ("CatBoost", "Gradient Boosting"),
        ("neural network", "Gradient Boosting or Random Forest"),
        ("deep learning", "Gradient Boosting or Random Forest"),
        ("SVM", "Logistic Regression"),
        ("support vector machine", "Logistic Regression"),
    ]
    for bad, fix in unsupported_algos:
        patches.append({"bad": bad, "fix": fix, "type": "unsupported_algorithm", "source_stat": "platform supports: RandomForest, GradientBoosting, LogisticRegression, Ridge, KMeans"})

    # Discount tier descriptions — replace invented tier clusters with real value_counts.
    # The model reads skewness/mean and invents tiers like "3%, 9%, 15%, 18%"
    # when the actual tiers are "5%, 10%, 15%, 20%, 25%, 30%". Since we have
    # the exact distribution, catch the cluster description deterministically.
    if discount_tier_str and discount_zero_pct is not None:
        correct_desc = (
            f"{discount_zero_pct}% of records carry no discount. "
            f"When discounts are applied, the actual tiers are: {discount_tier_str}."
        )
        # Common phrases the model uses to describe invented tiers
        invented_tier_phrases = [
            "cluster at 3%",
            "cluster at 9%",
            "they cluster at",
            "cluster at 3%, 9%",
            "3%, 9%, 15%, 18%",
            "concentrate at 3",
            "concentrate at 9",
            "tiers of 3%",
            "discount tiers of 3",
            "discrete tiers (likely",
            "appear in discrete tiers (likely",
        ]
        for phrase in invented_tier_phrases:
            fix_str = (
                f"discrete tiers: {discount_tier_str}"
                if discount_tier_str else "discrete tiers based on actual data"
            )
            patches.append({
                "bad": phrase,
                "fix": fix_str,
                "type": "invented_category",
                "source_stat": f"numeric_analysis.discount_rate.value_counts: {discount_vc}",
            })

    # Regional spread without computed region segment rates
    if not region_spread_computed:
        patches.append({
            "bad": "regional spread",
            "fix": "variation across regions (exact rates not computed for this dataset)",
            "type": "unsupported_regional_claim",
            "source_stat": "region not in auto_segment_rates",
        })

    return patches


def _audit_report_html(report_html: str, stats: dict) -> tuple[str, list[dict], list[dict]]:
    """Run the two-pass surgical audit on a generated report HTML string.

    Pass 1: AI returns a compact JSON list of issues, each with exact bad/fix
    strings and (for math fixes) a source_stat dot-path into the computed stats.
    Pass 2: Python applies deterministic exact-string replacements. Ambiguous
    matches (bad string appears more than once) are skipped, not guessed at.

    Non-blocking — any failure returns the original HTML unchanged.

    Args:
        report_html: The assembled report HTML to audit and patch.
        stats: The full computed stats dict for this report run, used to
            build the audit context object so the auditor has real numbers
            to validate claims against.

    Returns:
        A 3-tuple of:
            - patched_html: The report HTML after applying all safe patches.
            - patches_applied: List of dicts describing each successful patch.
            - patches_skipped: List of dicts describing each skipped patch.
    """
    patches_applied: list[dict] = []
    patches_skipped: list[dict] = []

    # ── Pass 0: Deterministic banned-phrase replacements ─────────────────────
    # These are semantic errors detectable without AI — run them first so the
    # AI audit sees already-corrected text and can focus on math/label issues.
    for item in _build_banned_phrase_replacements(stats):
        bad: str = item["bad"]
        fix: str = item["fix"]
        # Case-insensitive search since the report may capitalize differently
        import re as _re
        pattern = _re.compile(_re.escape(bad), _re.IGNORECASE)
        matches = pattern.findall(report_html)
        if not matches:
            continue
        if len(matches) > 1:
            patches_skipped.append({
                "reason": "multiple_occurrences",
                "type": item.get("type", "banned_phrase"),
                "bad": bad[:80],
            })
            continue
        # Replace preserving the original case of the match by using the
        # regex sub with the fixed replacement string
        report_html = pattern.sub(fix, report_html, count=1)
        patches_applied.append({
            "type": item.get("type", "banned_phrase"),
            "action": "replaced",
            "bad": bad[:60],
            "fix": fix[:60],
            "source_stat": item.get("source_stat", ""),
        })
        print(f"[audit-banned] replaced: '{bad[:50]}' → '{fix[:50]}'")
    # ── End pass 0 ───────────────────────────────────────────────────────────

    try:
        # Build a purpose-built audit context — smaller and more reliable than
        # truncating the full stats JSON, which can cut off the exact values
        # the auditor needs to validate a math claim.
        audit_stats: dict = {}
        audit_stats["segment_impact_summary"] = stats.get("segment_impact_summary", {})
        auto_seg = stats.get("auto_segment_rates", {})
        # Named consistently with the actual field name so source_stat dot-paths work
        audit_stats["auto_segment_rates"] = {
            k: {
                "overall_rate_pct": v.get("overall_rate_pct"),
                "highest_rate_segment": v.get("highest_rate_segment"),
                "lowest_rate_segment": v.get("lowest_rate_segment"),
                "highest_primary_impact_segment": v.get("highest_primary_impact_segment"),
                "rates": {
                    seg: {kk: vv for kk, vv in data.items()
                          if kk in ("rate", "pp_above_overall", "positive_rate_pct", "primary_impact")}
                    for seg, data in list(v.get("rates", {}).items())[:8]
                }
            }
            for k, v in auto_seg.items()
        }
        # Cross-tab rates — include top 8 rows so the auditor can validate
        # specific combination claims like "Wholesale x Outdoor has 42.6% rate."
        audit_stats["auto_crosstab_rates"] = {
            k: {
                **{kk: vv for kk, vv in v.items() if kk != "rates"},
                "rates": {
                    seg: {
                        **{kk: vv for kk, vv in data.items()
                           if kk in ("rate", "pp_above_overall", "positive_count")},
                        # Normalize the dynamic *_positive_total field (e.g.
                        # lost_revenue_positive_total, gross_margin_positive_total)
                        # into a stable "primary_impact" key so the auditor can
                        # validate dollar/value claims without knowing the column name.
                        "primary_impact": next(
                            (vv for kk, vv in data.items() if kk.endswith("_positive_total")),
                            None
                        ),
                    }
                    for seg, data in list(v.get("rates", {}).items())[:8]
                }
            }
            for k, v in stats.get("auto_crosstab_rates", {}).items()
        }
        audit_stats["auto_effect_sizes"] = {
            k: {kk: vv for kk, vv in v.items() if kk in ("cohens_d", "magnitude", "feature_col", "outcome_col")}
            for k, v in list(stats.get("auto_effect_sizes", {}).items())[:10]
        }
        audit_stats["target_distribution"] = stats.get("target_distribution", {})
        audit_stats["target_pcts"] = stats.get("target_pcts", {})
        audit_stats["top_correlations"] = stats.get("top_correlations", [])[:10]
        # Value counts for low-cardinality numeric columns — lets the auditor
        # verify that discount tiers, price bands, etc. match the actual data.
        # Exposed at "numeric_value_counts" for brevity; actual path in full stats
        # is numeric_analysis[col].value_counts.
        audit_stats["numeric_value_counts"] = {
            col: data["value_counts"]
            for col, data in stats.get("numeric_analysis", {}).items()
            if data.get("value_counts") is not None
        }
        audit_stats["_note_value_counts_path"] = "In the full stats, value_counts lives at numeric_analysis.<col>.value_counts. Source-stat dot-paths should use that path."
        # Metric meanings — static definitions injected so the auditor knows
        # what each financial/operational field actually measures.
        audit_stats["metric_meanings"] = {
            "lost_revenue_estimate": "estimated revenue not captured due to stockout; NOT margin, NOT realized revenue",
            "gross_margin_dollars": "realized gross margin on sold units only",
            "units_sold": "observed sold units — may be capped by inventory when stockouts occur; NOT true demand",
            "forecast_demand_units": "forecast input — NOT a measure of forecast accuracy",
            "week": "categorical week label — NOT a continuous time series unless time_series stats are present",
            "stockout_flag": "binary indicator: 1=stockout occurred, 0=no stockout",
        }
        stats_snapshot = json.dumps(audit_stats, indent=1)

        audit_prompt = f"""You are auditing a data intelligence report HTML for factual and quality issues before it is shown to a client. Return ONLY a JSON array of issues. No preamble, no explanation, no markdown.

COMPUTED STATS (source of truth for math fixes):
{stats_snapshot}

REPORT HTML (first 16000 chars):
{report_html[:16000]}

For each issue found, return an object with these exact fields:
- "type": one of "math" | "unsupported_claim" | "dramatic_language" | "label_mismatch" | "outside_benchmark" | "invented_effect_size" | "week_range_guess" | "metric_conflation" | "censored_sales_claim" | "invented_category"
- "bad": the exact string to replace (must appear verbatim in the report)
- "fix": the replacement string, OR "remove" to delete the sentence containing "bad"
- "source_stat": for "math" type only — the dot-path into the stats above that supports the fix (e.g. "auto_segment_rates.stockout_flag_by_channel.rates.Wholesale.pp_above_overall"). If no computed stat supports the math fix, use type "unsupported_claim" with fix "remove" instead.

Rules:
- Only flag issues where the bad text appears literally in the report
- For math: only provide a fix if the correct value exists in the computed stats above. Otherwise flag as unsupported_claim.
- For dramatic_language: replace with factual equivalent (e.g. "bleeding revenue" → "reducing revenue")
- For unsupported_claim (accuracy %, dollar recovery, event count ranges not in stats): fix must be "remove"
- For invented_effect_size (Cohen's d not in auto_effect_sizes): fix must be "remove"
- For metric_conflation: if the report uses "margin" when the source field is lost_revenue_estimate (per metric_meanings), replace "margin" with "revenue" in that claim
- For censored_sales_claim: if the report claims forecast accuracy from units_sold correlation without forecast_error in stats, replace with a hedged claim noting inventory-capping
- For invented_category: if the report lists discount tiers, price bands, or other numeric categories not present in `numeric_value_counts`, flag as invented_category and replace with the actual values from `numeric_value_counts`. If the column has no `value_counts` entry, the report must not describe its tier structure at all. Tiers inferred from histogram bins, mean, median, skewness, or examples are not valid — only explicit `value_counts` entries count.
- For label_mismatch: also check category/segment rankings. If the report states a ranking order (e.g. "Apparel > Outdoor > Home Goods > Electronics") but the dollar amounts or rates listed do not match that order, flag as label_mismatch and provide the correct ranking derived from the listed values. The listed values always take precedence over any claimed order.
- Do not change HTML structure, CSS, or styling
- Do not rewrite sentences beyond the minimum needed to remove or correct the flagged text
- If no issues found, return []

Return ONLY the JSON array."""

        audit_resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            timeout=25.0,  # hard cap — audit must not block final_html delivery
            messages=[{"role": "user", "content": audit_prompt}]
        )
        audit_raw = audit_resp.content[0].text.strip()
        if audit_raw.startswith("```"):
            audit_raw = audit_raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            audit_items: list[dict] = json.loads(audit_raw)
            if not isinstance(audit_items, list):
                audit_items = []
        except Exception:
            audit_items = []

        for item in audit_items:
            bad: str = item.get("bad", "").strip()
            fix: str = item.get("fix", "").strip()
            issue_type: str = item.get("type", "unknown")
            source_stat: str = item.get("source_stat", "")

            if not bad or not fix:
                continue

            occurrences = report_html.count(bad)
            if occurrences == 0:
                continue
            if occurrences > 1:
                patches_skipped.append({
                    "reason": "multiple_occurrences",
                    "type": issue_type,
                    "bad": bad[:80],
                })
                print(f"[audit] skipped (x{occurrences} occurrences): {bad[:60]}")
                continue

            if fix == "remove":
                idx = report_html.find(bad)
                sent_start = max(0, report_html.rfind(".", 0, idx) + 1)
                for tag_end in ["<p>", "<li>", "<div>"]:
                    tag_idx = report_html.rfind(tag_end, 0, idx)
                    if tag_idx > sent_start:
                        sent_start = tag_idx + len(tag_end)
                sent_end_idx = report_html.find(".", idx + len(bad))
                sent_end = sent_end_idx + 1 if sent_end_idx != -1 else idx + len(bad)
                sentence = report_html[sent_start:sent_end].strip()
                report_html = report_html.replace(sentence, "", 1)
                patches_applied.append({"type": issue_type, "action": "removed", "bad": bad[:60]})
            else:
                report_html = report_html.replace(bad, fix, 1)
                patches_applied.append({
                    "type": issue_type,
                    "action": "replaced",
                    "bad": bad[:60],
                    "fix": fix[:60],
                    "source_stat": source_stat,
                })
            print(f"[audit] patched ({issue_type}): '{bad[:50]}' → '{fix[:50]}'")

    except Exception as audit_err:
        print(f"[audit] pass failed: {audit_err}")

    # ── Pass 3: Deterministic HTML tag repair ────────────────────────────────
    # Scans for closing tags that don't match their most recent opening tag.
    # Uses a simple window-based approach because Python's html.parser treats
    # <p> as auto-closing (it never appears on the stack), so the standard
    # parser misses mismatches like </h3> closing a <p>.
    # Non-blocking — any failure leaves the HTML unchanged.
    try:
        import re as _re_html

        # Find all closing tags and check if they have a matching opener
        # in the preceding 2000 chars. If not, they're likely mismatched.
        close_tag_pattern = _re_html.compile(r'</(h[1-6]|p|div|section|article|span|li|ul|ol|table|tr|td|th)>', _re_html.IGNORECASE)
        open_tag_pattern = _re_html.compile(r'<(h[1-6]|p|div|section|article|span|li|ul|ol|table|tr|td|th)(?:\s[^>]*)?>',  _re_html.IGNORECASE)

        for close_match in close_tag_pattern.finditer(report_html):
            close_tag = close_match.group(1).lower()
            close_start = close_match.start()
            # Look at the window before this closing tag
            window = report_html[max(0, close_start - 2000):close_start]
            # Find the last opening tag in the window
            open_matches = list(open_tag_pattern.finditer(window))
            if not open_matches:
                continue
            last_open = open_matches[-1].group(1).lower()
            if last_open != close_tag:
                bad = f"</{close_tag}>"
                good = f"</{last_open}>"
                # Only fix if unambiguous (bad tag appears exactly once nearby)
                nearby = report_html[max(0, close_start-100):close_start+100]
                if nearby.count(bad) == 1 and report_html.count(bad) == 1:
                    report_html = report_html.replace(bad, good, 1)
                    patches_applied.append({
                        "type": "html_tag_repair",
                        "action": "replaced",
                        "bad": bad,
                        "fix": good,
                        "source_stat": "html_tag_mismatch_detector",
                    })
                    print(f"[audit-html] repaired tag: {bad} → {good}")
    except Exception as html_err:
        print(f"[audit-html] tag repair failed: {html_err}")
    # ── End pass 3 ───────────────────────────────────────────────────────────

    # ── Pass 4: Segment ranking consistency check ─────────────────────────────
    # Scans for "A > B > C > D" ranking strings in the HTML and verifies that
    # the order matches any dollar amounts or percentages listed nearby.
    # If mismatched, re-orders the labels to match the values. Deterministic.
    try:
        import re as _re2

        ranking_pattern = _re2.compile(
            r'([A-Za-z][A-Za-z &\/\-]+?)\s*(?:&gt;|>)\s*'
            r'([A-Za-z][A-Za-z &\/\-]+?)\s*(?:&gt;|>)\s*'
            r'([A-Za-z][A-Za-z &\/\-]+?)(?:\s*(?:&gt;|>)\s*([A-Za-z][A-Za-z &\/\-]+?))?'
            r'(?=[,\.\s<])'
        )
        value_pattern = _re2.compile(r'\$?(\d[\d,]*(?:\.\d+)?)[KMB]?')

        for match in ranking_pattern.finditer(report_html):
            labels = [g.strip() for g in match.groups() if g and g.strip()]
            if len(labels) < 3:
                continue
            end_pos = match.end()
            context = report_html[end_pos:end_pos + 500]
            raw_values = value_pattern.findall(context)
            numeric_values = []
            for v in raw_values[:len(labels)]:
                try:
                    val = float(v.replace(",", ""))
                    if val > 0:
                        numeric_values.append(val)
                except (ValueError, AttributeError):
                    pass

            if len(numeric_values) == len(labels):
                paired = list(zip(labels, numeric_values))
                sorted_by_value = sorted(paired, key=lambda x: x[1], reverse=True)
                correct_order = [p[0] for p in sorted_by_value]
                if correct_order != labels:
                    bad_ranking = " > ".join(labels)
                    good_ranking = " > ".join(correct_order)
                    if report_html.count(bad_ranking) == 1:
                        report_html = report_html.replace(bad_ranking, good_ranking, 1)
                        patches_applied.append({
                            "type": "ranking_correction",
                            "action": "replaced",
                            "bad": bad_ranking[:60],
                            "fix": good_ranking[:60],
                            "source_stat": "values extracted from surrounding context",
                        })
                        print(f"[audit-ranking] corrected: '{bad_ranking}' → '{good_ranking}'")
    except Exception as rank_err:
        print(f"[audit-ranking] ranking check failed: {rank_err}")
    # ── End pass 4 ───────────────────────────────────────────────────────────

    return report_html, patches_applied, patches_skipped

@app.get("/analyze-report-stream")
async def analyze_report_stream(
    file_id: str,
    target_col: Optional[str] = None,
    dataset_context: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Stream the report via SSE — keeps connection alive past Railway 30s timeout."""
    from fastapi.responses import StreamingResponse
    import json as json_mod

    plan = get_user_plan(user_id)
    check_and_enforce_limit(user_id, plan, "report_count", PLAN_REPORT_LIMITS, "Data Intelligence Reports")

    data_path = DATA_DIR / f"{file_id}.pkl"
    ensure_file_owner(file_id, user_id)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")

    try:
        df = pd.read_pickle(str(data_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load dataset: {str(e)}")

    # Capture full-dataset metadata BEFORE sampling so the report always
    # shows real row/column counts, missingness totals, and column names.
    full_row_count: int = len(df)
    full_col_count: int = len(df.columns)
    full_duplicate_count: int = int(df.duplicated().sum())
    full_missing_total: int = int(df.isnull().sum().sum())
    full_columns: list[str] = list(df.columns)

    # Sample the dataframe if it exceeds the row cap.
    df_for_stats, was_sampled, sample_method = sample_for_report(
        df, target_col, REPORT_SAMPLE_ROWS
    )
    if was_sampled:
        sampling_note = (
            f"NOTE: Detailed statistics were computed on a representative "
            f"{len(df_for_stats):,}-row sample of the full {full_row_count:,}-row dataset "
            f"(sampling method: {sample_method}). Full-dataset metadata (row count, "
            f"column names, missingness totals, duplicate count) is preserved below. "
            f"Surface this sampling note visibly in the report introduction."
        )
    else:
        sampling_note = ""

    try:
        stats = compute_dataset_stats(df_for_stats, target_col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics computation failed: {str(e)}")

    # Overwrite sampled metadata with real full-dataset values.
    stats["shape"] = {"rows": full_row_count, "cols": full_col_count}
    stats["duplicates"] = full_duplicate_count
    stats["columns"] = full_columns
    total_cells = full_row_count * full_col_count
    stats["completeness_pct"] = (
        round((1 - full_missing_total / total_cells) * 100, 2)
        if total_cells else 100
    )
    if was_sampled:
        stats["sampling_note"] = sampling_note

    context_block = _build_steering_block(dataset_context, stats)
    if sampling_note:
        context_block = f"\n\nSAMPLING NOTE: {sampling_note}" + context_block

    # Build hard deterministic constraints from value_counts so the model
    # cannot invent tiers by reading skewness/mean. These appear as explicit
    # MANDATORY FACTS in the user message, not buried in the stats JSON.
    hard_constraints = _build_hard_constraints(stats)
    if hard_constraints:
        context_block = context_block + f"\n\nMANDATORY FACTS — use these exact values, do not infer alternatives:\n{hard_constraints}"

    stats_json = json.dumps(_inject_tier_notes(stats), indent=2)


    if len(stats_json) > 28000:
        stats["sample_rows"] = stats.get("sample_rows", [])[:2]
        if "top_correlations" in stats:
            stats["top_correlations"] = stats["top_correlations"][:10]
        if "effect_sizes" in stats:
            stats["effect_sizes"] = dict(list(stats["effect_sizes"].items())[:15])
        if "segment_rates" in stats:
            stats["segment_rates"] = {
                col: dict(list(groups.items())[:10])
                for col, groups in list(stats["segment_rates"].items())[:10]
            }
        for col in stats.get("numeric_analysis", {}):
            na = stats["numeric_analysis"][col]
            stats["numeric_analysis"][col] = {
                k: na[k] for k in ["mean","median","std","min","max","skewness","outlier_count","outlier_pct","skew_label","histogram"] if k in na
            }
        # Trim time_series per-period breakdowns to keep token usage bounded,
        # but keep the summary trend (first/last period, pct change) intact
        # since that's what the report actually needs to state the trend.
        ts = stats.get("time_series", {})
        for col, trend in ts.get("numeric_trends", {}).items():
            if "by_period_sum" in trend and len(trend["by_period_sum"]) > 12:
                periods_sorted = sorted(trend["by_period_sum"].keys())
                keep = periods_sorted[:3] + periods_sorted[-3:]
                trend["by_period_sum"] = {p: trend["by_period_sum"][p] for p in keep}
                trend["by_period_count"] = {p: trend["by_period_count"].get(p) for p in keep if p in trend.get("by_period_count", {})}
        stats_json = json.dumps(_inject_tier_notes(stats), indent=2)

    user_message = f"""Here are the pre-computed statistics for the client dataset:{context_block}

{stats_json}

Generate the complete Data Intelligence Report HTML page now. Remember: ONLY output the HTML document, nothing else."""

    rows_analyzed = len(df)
    cols_analyzed = len(df.columns)

    def generate():
        html_chunks = []
        try:
            # Immediate heartbeat to keep connection alive
            yield f"data: {json_mod.dumps({'type': 'heartbeat'})}\n\n"

            # Bounded retry: if the hard quality gate below rejects a report,
            # try once more before giving up. This is a full re-generation
            # (a second complete API call), not a text patch — the whole
            # point is that a corrupted/incomplete report gets a genuine
            # do-over, not a manual touch-up. Capped at 2 total attempts so
            # a dataset that reliably produces bad output doesn't silently
            # double every user's wait time and cost indefinitely.
            MAX_GENERATION_ATTEMPTS = 2
            report_html = ""
            rejection_reasons: list[str] = []
            truncated = False

            for attempt_num in range(1, MAX_GENERATION_ATTEMPTS + 1):
                html_chunks = []
                generation_start = time.monotonic()
                truncated = False

                if attempt_num > 1:
                    yield f"data: {json_mod.dumps({'type': 'progress', 'chars': 0})}\n\n"

                with client.messages.stream(
                    model="claude-sonnet-4-5",
                    max_tokens=20000,
                    system=REPORT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                ) as stream:
                    char_count = 0
                    for text in stream.text_stream:
                        html_chunks.append(text)
                        char_count += len(text)
                        # Send progress heartbeats (character count only, never raw HTML)
                        # so the frontend knows generation is active without rendering
                        # unaudited content. The final audited HTML arrives as final_html.
                        if char_count % 500 < len(text):
                            yield f"data: {json_mod.dumps({'type': 'progress', 'chars': char_count})}\n\n"

                        # Wall-clock insurance only — no character cap. Breaking
                        # here (still inside the `with` block) closes the
                        # underlying stream connection via its context manager on
                        # exit, rather than us just abandoning it.
                        if time.monotonic() - generation_start >= REPORT_MAX_GENERATION_SECONDS:
                            truncated = True
                            break

                # Clean up the assembled HTML
                attempt_html = "".join(html_chunks).strip()
                if '```' in attempt_html:
                    lines = attempt_html.split('\n')
                    lines = [l for l in lines if not l.strip().startswith('```')]
                    attempt_html = '\n'.join(lines).strip()

                if not attempt_html.startswith("<!DOCTYPE") and not attempt_html.startswith("<html"):
                    idx = attempt_html.find("<!DOCTYPE")
                    if idx == -1:
                        idx = attempt_html.find("<html")
                    if idx != -1:
                        attempt_html = attempt_html[idx:]

                # Only triggers if REPORT_MAX_GENERATION_SECONDS was actually
                # crossed — a rare, "something went wrong" case, not a routine
                # part of normal report generation. The document is likely
                # missing many closing tags at once (not just a mismatch —
                # _repair_html_tags below fixes mismatches, not wholesale
                # unclosed nesting), so it needs force-closing here first.
                if truncated:
                    notice = (
                        "Analysis was limited to the available generation window — generation took "
                        "longer than expected. The sections above reflect real, computed statistics; "
                        "anything past this point was not generated."
                    )
                    attempt_html = _finalize_truncated_report_html(attempt_html, notice)

                # ── Two-pass surgical audit (delegated to helper) ────────────────
                try:
                    attempt_html, patches_applied, patches_skipped = _audit_report_html(attempt_html, stats)
                    if patches_applied or patches_skipped:
                        yield f"data: {json_mod.dumps({'type': 'audit', 'applied': patches_applied, 'skipped': patches_skipped})}\n\n"
                except Exception as audit_ex:
                    print(f"[audit] skipped due to error: {audit_ex}")
                # ── End audit ────────────────────────────────────────────────────

                # ── Deterministic HTML tag repair ────────────────────────────────
                try:
                    attempt_html = _repair_html_tags(attempt_html)
                except Exception:
                    pass
                # ── End HTML repair ──────────────────────────────────────────────

                # ── Corrupted list-item repair (empty/unclosed <li>, text-merge signature) ──
                try:
                    attempt_html, li_repair_warnings = _repair_corrupted_list_items(attempt_html)
                    if li_repair_warnings:
                        print(f"[report] list-item repair: {li_repair_warnings}")
                        yield f"data: {json_mod.dumps({'type': 'audit', 'applied': li_repair_warnings, 'skipped': []})}\n\n"
                except Exception as li_repair_ex:
                    print(f"[list-item repair] skipped due to error: {li_repair_ex}")
                # ── End corrupted list-item repair ───────────────────────────────

                # ── Section number gap fix ────────────────────────────────────────
                try:
                    attempt_html = _fix_section_numbers(attempt_html)
                except Exception:
                    pass
                # ── End section number fix ────────────────────────────────────────

                # ── Deterministic ranking validator ──────────────────────────────
                try:
                    attempt_html = _validate_rankings(attempt_html, stats)
                except Exception:
                    pass
                # ── End ranking validator ─────────────────────────────────────────

                # ── Segment completeness safety net ──────────────────────────────
                try:
                    attempt_html = _validate_segment_completeness(attempt_html, stats)
                except Exception:
                    pass
                # ── End segment completeness ─────────────────────────────────────

                report_html = attempt_html

                # ── Hard quality gate — runs LAST, after every repair pass had
                # its chance to fix what it could. A truncated (timeout) report
                # is never subject to rejection/retry: it already represents a
                # deliberate best-effort partial result, and retrying would
                # just spend another full generation window on the same
                # underlying slowness that caused the timeout in the first
                # place. ─────────────────────────────────────────────────────
                if truncated:
                    rejection_reasons = []
                    break
                rejection_reasons = _validate_report_quality(report_html)
                if not rejection_reasons:
                    break
                print(f"[report] attempt {attempt_num}/{MAX_GENERATION_ATTEMPTS} REJECTED: {rejection_reasons}")
            # ── End generation attempt loop ───────────────────────────────────

            # Explicit status, not an implicit side effect of a boolean the
            # caller might not check. A partial/timeout-finalized report and a
            # rejected report are both materially different outcomes from a
            # clean completion and must be tracked as such — including NOT
            # silently consuming the same report_count credit a full report
            # does. This matters most on Free plan with a small report
            # allowance: a cut-short or rejected result burning someone's only
            # credit erodes trust even if some of the content is useful.
            if truncated:
                report_status = "partial_timeout"
            elif rejection_reasons:
                report_status = "rejected"
            else:
                report_status = "complete"

            if report_status == "complete":
                increment_usage_count(user_id, "report_count")
            elif report_status == "partial_timeout":
                # NOTE: this requires a `partial_report_count` column on the
                # Supabase `profiles` table. If that column doesn't exist yet,
                # this PATCH fails and is caught/logged by increment_usage_count
                # itself — harmless, but non-functional until the column is
                # added (a schema change outside what this code can do).
                increment_usage_count(user_id, "partial_report_count")
                print(f"[report] PARTIAL_TIMEOUT for user_id={user_id}: hit REPORT_MAX_GENERATION_SECONDS={REPORT_MAX_GENERATION_SECONDS}s — report_count NOT incremented, partial_report_count incremented instead")
            else:
                print(f"[report] REJECTED for user_id={user_id} after {MAX_GENERATION_ATTEMPTS} attempt(s): {rejection_reasons} — report_count NOT incremented")

            # Send the final audited HTML as a single payload — this is what
            # the frontend must render. Never render raw chunk content since
            # chunks were sent before the audit pass ran.
            yield f"data: {json_mod.dumps({'type': 'final_html', 'html': report_html, 'rows_analyzed': rows_analyzed, 'cols_analyzed': cols_analyzed, 'report_status': report_status, 'rejection_reasons': rejection_reasons})}\n\n"


        except Exception as e:
            import traceback
            print(traceback.format_exc())
            yield f"data: {json_mod.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/analyze-report")
async def analyze_report(
    file_id: str,
    target_col: Optional[str] = None,
    dataset_context: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Non-streaming fallback. Use /analyze-report-stream for production.

    Kept in sync with the streaming endpoint's limit enforcement and usage
    tracking — this was previously missing (a real gap: a user could
    generate reports through this route without it ever counting against
    their plan's report_count, and without the report_status field the
    frontend now expects).
    """
    plan = get_user_plan(user_id)
    check_and_enforce_limit(user_id, plan, "report_count", PLAN_REPORT_LIMITS, "Data Intelligence Reports")
    try:
        data_path = DATA_DIR / f"{file_id}.pkl"
        ensure_file_owner(file_id, user_id)
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")

        try:
            df = pd.read_pickle(str(data_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load dataset: {str(e)}")

        df_for_stats, was_sampled, sample_method = sample_for_report(
            df, target_col, REPORT_SAMPLE_ROWS
        )
        full_row_count = len(df); full_col_count = len(df.columns)
        full_duplicate_count = int(df.duplicated().sum())
        full_missing_total = int(df.isnull().sum().sum())
        sampling_note = (
            f"NOTE: Detailed statistics computed on a representative "
            f"{len(df_for_stats):,}-row sample of {full_row_count:,} total rows."
        ) if was_sampled else ""
        try:
            stats = compute_dataset_stats(df_for_stats, target_col)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Statistics computation failed: {str(e)}")
        stats["shape"] = {"rows": full_row_count, "cols": full_col_count}
        stats["duplicates"] = full_duplicate_count
        stats["columns"] = list(df.columns)
        total_cells = full_row_count * full_col_count
        stats["completeness_pct"] = round((1 - full_missing_total / total_cells) * 100, 2) if total_cells else 100
        if was_sampled:
            stats["sampling_note"] = sampling_note

        context_block = _build_steering_block(dataset_context, stats)
        if sampling_note:
            context_block = f"\n\nSAMPLING NOTE: {sampling_note}" + context_block
        hard_constraints = _build_hard_constraints(stats)
        if hard_constraints:
            context_block = context_block + f"\n\nMANDATORY FACTS — use these exact values, do not infer alternatives:\n{hard_constraints}"
        stats_json = json.dumps(_inject_tier_notes(stats), indent=2)

        if len(stats_json) > 28000:
            stats["sample_rows"] = stats.get("sample_rows", [])[:2]
            if "top_correlations" in stats:
                stats["top_correlations"] = stats["top_correlations"][:10]
            if "effect_sizes" in stats:
                stats["effect_sizes"] = dict(list(stats["effect_sizes"].items())[:15])
            if "segment_rates" in stats:
                stats["segment_rates"] = {
                    col: dict(list(groups.items())[:10])
                    for col, groups in list(stats["segment_rates"].items())[:10]
                }
            for col in stats.get("numeric_analysis", {}):
                na = stats["numeric_analysis"][col]
                stats["numeric_analysis"][col] = {
                    k: na[k] for k in ["mean","median","std","min","max","skewness","outlier_count","outlier_pct","skew_label","histogram"] if k in na
                }
            ts = stats.get("time_series", {})
            for col, trend in ts.get("numeric_trends", {}).items():
                if "by_period_sum" in trend and len(trend["by_period_sum"]) > 12:
                    periods_sorted = sorted(trend["by_period_sum"].keys())
                    keep = periods_sorted[:3] + periods_sorted[-3:]
                    trend["by_period_sum"] = {p: trend["by_period_sum"][p] for p in keep}
                    trend["by_period_count"] = {p: trend["by_period_count"].get(p) for p in keep if p in trend.get("by_period_count", {})}
            stats_json = json.dumps(_inject_tier_notes(stats), indent=2)

        user_message = f"""Here are the pre-computed statistics for the client dataset:{context_block}

{stats_json}

Generate the complete Data Intelligence Report HTML page now. Remember: ONLY output the HTML document, nothing else."""

        MAX_GENERATION_ATTEMPTS = 2
        rejection_reasons: list[str] = []
        patches_applied = []
        for attempt_num in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                message = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=20000,
                    system=REPORT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                )
                report_html = message.content[0].text.strip()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")

            if report_html.startswith("```"):
                report_html = re.sub(r'^```[a-z]*\n?', '', report_html)
                report_html = re.sub(r'\n?```$', '', report_html)
                report_html = report_html.strip()

            if not report_html.startswith("<!DOCTYPE") and not report_html.startswith("<html"):
                idx = report_html.find("<!DOCTYPE")
                if idx == -1:
                    idx = report_html.find("<html")
                if idx != -1:
                    report_html = report_html[idx:]
                else:
                    raise HTTPException(status_code=500, detail="Report generation produced invalid HTML. Please try again.")

            # Apply the same surgical audit as the streaming endpoint
            report_html, patches_applied, patches_skipped = _audit_report_html(report_html, stats)
            if patches_applied:
                print(f"[analyze-report fallback] audit applied {len(patches_applied)} patch(es)")
            report_html = _repair_html_tags(report_html)
            report_html, li_repair_warnings = _repair_corrupted_list_items(report_html)
            if li_repair_warnings:
                print(f"[analyze-report fallback] list-item repair: {li_repair_warnings}")
            report_html = _fix_section_numbers(report_html)
            report_html = _validate_rankings(report_html, stats)
            report_html = _validate_segment_completeness(report_html, stats)

            # Hard quality gate — same criteria as the streaming endpoint.
            # Runs LAST, after every repair pass had its chance to fix what
            # it could.
            rejection_reasons = _validate_report_quality(report_html)
            if not rejection_reasons:
                break
            print(f"[analyze-report fallback] attempt {attempt_num}/{MAX_GENERATION_ATTEMPTS} REJECTED: {rejection_reasons}")

        # This fallback has no generation-timeout concept of its own — it's
        # a single blocking call with no streaming loop to cut short — so the
        # only two outcomes here are "complete" or "rejected" (never
        # "partial_timeout"). A rejected report does NOT consume a
        # report_count credit, same principle as the streaming endpoint.
        report_status = "rejected" if rejection_reasons else "complete"
        if report_status == "complete":
            increment_usage_count(user_id, "report_count")
        else:
            print(f"[analyze-report fallback] REJECTED for user_id={user_id} after {MAX_GENERATION_ATTEMPTS} attempt(s): {rejection_reasons} — report_count NOT incremented")


        return {
            "report_html": report_html,
            "stats_computed": True,
            "audit_patches_applied": len(patches_applied),
            "rows_analyzed": len(df),
            "cols_analyzed": len(df.columns),
            "report_status": report_status,
            "rejection_reasons": rejection_reasons
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ============================================================
# STRIPE BILLING
# ============================================================

# Owner emails — always get unlimited regardless of plan
OWNER_EMAILS = ["dashfarrell@gmail.com", "freshpineapples1234@gmail.com"]

PLAN_MODEL_LIMITS = {
    "free": 1,
    "pro": 5,
    "team": 15,
    "unlimited": 999999,
}

def get_user_plan(user_id: str) -> str:
    """Fetch user plan from Supabase profiles table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "free"
    try:
        url = f"{SUPABASE_URL}/rest/v1/profiles?id=eq.{user_id}&select=plan"
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
        }
        resp = httpx.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                return data[0].get("plan", "free")
    except Exception as e:
        print(f"get_user_plan error: {e}")
    return "free"


# ── Lifetime usage limits (never reset — once spent, gone for that account) ──
# Free is intentionally low since it has no revenue backing it. Each paid tier
# steps up meaningfully so usage scales with what the subscriber is paying.
PLAN_REPORT_LIMITS = {
    "free": 1,
    "pro": 50,
    "team": 200,
    "unlimited": 999999,
}
PLAN_ORG_SESSION_LIMITS = {
    "free": 5,
    "pro": 100,
    "team": 400,
    "unlimited": 999999,
}


def _supabase_headers():
    return {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
    }


def get_usage_counts(user_id: str) -> dict:
    """Fetch lifetime report_count, org_session_count, and partial_report_count
    for a user. Defaults to 0 for any column if missing/not yet set.

    increment_usage_count reads through this function to know the CURRENT
    value before writing current+1 — a field left out of the select here
    would always read back as 0 regardless of what's actually stored, so
    every increment would write 1 instead of actually accumulating.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"report_count": 0, "org_session_count": 0, "partial_report_count": 0}
    try:
        url = f"{SUPABASE_URL}/rest/v1/profiles?id=eq.{user_id}&select=report_count,org_session_count,partial_report_count"
        resp = httpx.get(url, headers=_supabase_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                row = data[0]
                return {
                    "report_count": row.get("report_count") or 0,
                    "org_session_count": row.get("org_session_count") or 0,
                    "partial_report_count": row.get("partial_report_count") or 0,
                }
    except Exception as e:
        print(f"get_usage_counts error: {e}")
    return {"report_count": 0, "org_session_count": 0, "partial_report_count": 0}


def increment_usage_count(user_id: str, field: str):
    """Atomically-ish increments a lifetime usage counter in Supabase.
    Not a true atomic increment (read-then-write), but the resulting race
    window only affects exact limit enforcement by at most one extra unit
    under concurrent requests from the same account, which is an acceptable
    tradeoff against the complexity of an RPC-based atomic counter."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        current = get_usage_counts(user_id).get(field, 0)
        url = f"{SUPABASE_URL}/rest/v1/profiles?id=eq.{user_id}"
        httpx.patch(url, headers=_supabase_headers(), json={field: current + 1}, timeout=10)
    except Exception as e:
        print(f"increment_usage_count error ({field}): {e}")


def check_and_enforce_limit(user_id: str, plan: str, field: str, limits: dict, feature_name: str):
    """Raises HTTPException(403) if the user has hit their lifetime limit for
    this feature on their current plan. Owner emails bypass entirely."""
    email = get_user_email(user_id)
    if email in OWNER_EMAILS:
        return
    limit = limits.get(plan, limits.get("free", 1))
    if limit >= 999999:
        return  # unlimited tier, nothing to check
    counts = get_usage_counts(user_id)
    used = counts.get(field, 0)
    if used >= limit:
        raise HTTPException(
            status_code=403,
            detail=f"You've used all {limit} {feature_name} included on your plan. Upgrade for more."
        )


def get_user_email(user_id: str) -> str:
    """Fetch user email from Supabase auth."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return ""
    try:
        url = f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}"
        headers = {"Authorization": f"Bearer {SUPABASE_KEY}", "apikey": SUPABASE_KEY}
        resp = httpx.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("email", "")
    except Exception:
        pass
    return ""

def check_model_limit(user_id: str) -> tuple[bool, int, int]:
    """Returns (can_save, current_count, limit)."""
    # Check owner override
    email = get_user_email(user_id)
    if email in OWNER_EMAILS:
        return True, 0, 999999

    plan = get_user_plan(user_id)
    limit = PLAN_MODEL_LIMITS.get(plan, 1)

    # Count models in registry for this user
    registry = load_registry()
    user_models = [v for v in registry.values() if v.get("user_id") == user_id]
    count = len(user_models)

    return count < limit, count, limit

def _write_file_meta(file_id: str, user_id: str) -> None:
    """Write a sidecar .meta.json recording which user owns this dataset file.

    Args:
        file_id: The UUID identifying the stored dataset .pkl file.
        user_id: The authenticated user ID who uploaded this file.
    """
    import json as _json
    meta_path = DATA_DIR / f"{file_id}.meta.json"
    try:
        meta_path.write_text(_json.dumps({"user_id": user_id}))
    except Exception as e:
        print(f"[file_meta] failed to write sidecar for {file_id}: {e}")


def ensure_file_owner(file_id: str, user_id: str) -> None:
    """Raise HTTPException(403) if the user does not own this dataset file.

    Reads the sidecar .meta.json written at upload time. Owner emails always
    bypass this check. If no sidecar exists (legacy files pre-dating this fix),
    the check is skipped rather than blocking access.

    Args:
        file_id: The UUID of the dataset file to check.
        user_id: The authenticated user making the request.

    Raises:
        HTTPException: 403 if the file belongs to a different user.
    """
    import json as _json
    email = get_user_email(user_id)
    if email in OWNER_EMAILS:
        return
    meta_path = DATA_DIR / f"{file_id}.meta.json"
    if not meta_path.exists():
        return  # legacy file — skip check rather than block
    try:
        meta = _json.loads(meta_path.read_text())
        if meta.get("user_id") and meta["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied — this dataset belongs to another account.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[file_meta] could not read sidecar for {file_id}: {e}")


def ensure_model_owner(model_id: str, user_id: str, registry: Optional[dict] = None) -> dict:
    """Return model metadata only when the authenticated user owns the model."""
    registry = registry or load_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail="Model not found")
    meta = registry[model_id]
    if meta.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your model")
    return meta

def set_user_plan(user_id: str, plan: str, stripe_customer_id: str = None, stripe_subscription_id: str = None):
    """Upsert user plan in Supabase profiles table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        payload = {"id": user_id, "plan": plan}
        if stripe_customer_id:
            payload["stripe_customer_id"] = stripe_customer_id
        if stripe_subscription_id:
            payload["stripe_subscription_id"] = stripe_subscription_id

        # Use upsert so it works whether or not the row exists
        url = f"{SUPABASE_URL}/rest/v1/profiles"
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=10)
        print(f"set_user_plan response: {resp.status_code} {resp.text[:100]}")
        return resp.status_code in [200, 201, 204]
    except Exception as e:
        print(f"set_user_plan error: {e}")
        return False


@app.post("/create-checkout-session")
@app.get("/create-checkout-session")
async def create_checkout_session(
    plan: str,
    user_id: str = Depends(get_current_user)
):
    """Create a Stripe checkout session for the given plan."""
    if plan not in STRIPE_PRICES:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {plan}")

    price_id = STRIPE_PRICES[plan]
    if not price_id:
        raise HTTPException(status_code=500, detail=f"Price ID for {plan} not configured. Set STRIPE_PRICE_{plan.upper()} env var.")

    try:
        # Get user email from Supabase
        email = None
        try:
            url = f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}"
            headers = {
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "apikey": SUPABASE_KEY,
            }
            resp = httpx.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                email = resp.json().get("email")
        except Exception:
            pass

        session_params = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": f"{FRONTEND_URL}/success.html?session_id={{CHECKOUT_SESSION_ID}}&plan={plan}",
            "cancel_url": f"{FRONTEND_URL}/dashboard.html#billing",
            "metadata": {"user_id": user_id, "plan": plan},
            "subscription_data": {"metadata": {"user_id": user_id, "plan": plan}},
            "allow_promotion_codes": True,
        }

        if email:
            session_params["customer_email"] = email

        session = stripe.checkout.Session.create(**session_params)
        return {"checkout_url": session.url, "session_id": session.id}

    except stripe.StripeError as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checkout failed: {str(e)}")


@app.get("/verify-session")
async def verify_session(
    session_id: str,
    user_id: str = Depends(get_current_user)
):
    """Verify a completed Stripe checkout session and activate the plan."""
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            plan = session.metadata["plan"] if session.metadata and "plan" in session.metadata else "pro"
            stored_user_id = session.metadata["user_id"] if session.metadata and "user_id" in session.metadata else None

            # Security: make sure this session belongs to this user
            if stored_user_id != user_id:
                raise HTTPException(status_code=403, detail="Session does not belong to this user")

            set_user_plan(
                user_id,
                plan,
                stripe_customer_id=session.customer,
                stripe_subscription_id=session.subscription
            )
            return {"success": True, "plan": plan, "plan_name": PLAN_NAMES.get(plan, plan)}
        else:
            return {"success": False, "status": session.payment_status}
    except stripe.StripeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events for subscription lifecycle."""
    from fastapi import Request
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except stripe.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Handle subscription events
    if event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        metadata = dict(sub.get("metadata") or {})
        user_id = metadata.get("user_id")
        if user_id:
            set_user_plan(user_id, "free")
            print(f"Subscription cancelled — user {user_id} downgraded to free")

    elif event["type"] == "customer.subscription.updated":
        sub = event["data"]["object"]
        metadata = dict(sub.get("metadata") or {})
        user_id = metadata.get("user_id")
        status = sub.get("status")
        if user_id and status == "active":
            plan = metadata.get("plan", "pro")
            set_user_plan(user_id, plan, stripe_subscription_id=sub["id"])
            print(f"Subscription updated — user {user_id} plan: {plan}")

    elif event["type"] in ["checkout.session.completed", "invoice.payment_succeeded"]:
        if event["type"] == "checkout.session.completed":
            obj = event["data"]["object"]
            metadata = dict(obj.get("metadata") or {})
            user_id = metadata.get("user_id")
            plan = metadata.get("plan", "pro")
            if user_id and obj.get("payment_status") == "paid":
                set_user_plan(
                    user_id, plan,
                    stripe_customer_id=obj.get("customer"),
                    stripe_subscription_id=obj.get("subscription")
                )
                print(f"Checkout complete — user {user_id} activated {plan}")

    return {"received": True}


@app.get("/my-plan")
async def get_my_plan(user_id: str = Depends(get_current_user)):
    """Get the current user's plan."""
    plan = get_user_plan(user_id)
    return {"plan": plan, "plan_name": PLAN_NAMES.get(plan, "Free Plan")}


# ============================================================
# ORGANIZE YOUR DATA
# ============================================================

def summarize_dataframe(df: pd.DataFrame, filename: str) -> dict:
    """Generate a comprehensive but token-efficient summary of a dataframe."""
    # Some columns can contain unhashable values (dicts/lists), e.g. from
    # nested JSON fields that survived a merge. nunique/value_counts/duplicated
    # all hash values internally and will crash on these — coerce such columns
    # to strings first so profiling never throws.
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            sample = df[col].dropna().head(20)
            if any(isinstance(v, (dict, list, set)) for v in sample):
                df[col] = df[col].apply(lambda v: str(v) if isinstance(v, (dict, list, set)) else v)

    try:
        duplicate_rows = int(df.duplicated().sum())
    except Exception:
        duplicate_rows = 0

    summary = {
        "filename": filename,
        "rows": len(df),
        "cols": len(df.columns),
        "duplicate_rows": duplicate_rows,
        "columns": {}
    }
    for col in df.columns:
        series = df[col]
        try:
            unique_count = int(series.nunique())
        except Exception:
            unique_count = None
        col_info = {
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_pct": round(float(series.isnull().mean()) * 100, 1),
            "unique_count": unique_count,
            "sample_values": [str(v) for v in series.dropna().head(3).tolist()],
            "top_values": []
        }
        # Top 5 most frequent values
        try:
            vc = series.value_counts().head(5)
            col_info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        except Exception:
            pass
        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            col_info["min"] = float(series.min()) if not series.isnull().all() else None
            col_info["max"] = float(series.max()) if not series.isnull().all() else None
            col_info["mean"] = round(float(series.mean()), 4) if not series.isnull().all() else None
            col_info["median"] = round(float(series.median()), 4) if not series.isnull().all() else None
        # Date detection — only attempted for non-numeric columns whose sample
        # values at least loosely resemble a date (contain a date separator, a
        # 4-digit year, or a month name/abbreviation). This is a cheap filter
        # that skips the pd.to_datetime attempt entirely for obviously
        # non-date text (names, free-form notes, ID values), which is what
        # was producing noisy pandas date-parse warnings on every such column.
        # Warnings are also explicitly suppressed around the parse attempt
        # itself as a second layer, since even date-like-looking text can
        # still be ambiguous enough to warn.
        if not pd.api.types.is_numeric_dtype(series):
            date_sample = series.dropna().astype(str).head(10)
            looks_date_like = (
                not date_sample.empty
                and date_sample.str.contains(_DATE_LIKE_HINT_PATTERN, regex=True, na=False).any()
            )
            if looks_date_like:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(date_sample, errors='coerce')
                    if parsed.notna().any():
                        col_info["likely_date"] = True
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            full_parsed = pd.to_datetime(series.dropna(), errors='coerce')
                        col_info["date_min"] = str(full_parsed.min())
                        col_info["date_max"] = str(full_parsed.max())
                except Exception:
                    pass
        summary["columns"][col] = col_info
    return summary


def _promote_safe_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Promote string columns to numeric dtype only where it's provably safe.

    Called right after a string-safe file read (every column loaded as text)
    to restore proper numeric typing for genuinely numeric columns — needed
    so summarize_dataframe can still compute min/max/mean for them, and the
    planner still sees real numeric columns as numeric. A column is promoted
    to numeric ONLY if none of its non-null values would lose information by
    doing so: no leading-zero numeric-looking string (e.g. "007") and no
    ID-like column name. Every other column is kept as clean, whitespace-
    trimmed text, exactly as it appeared in the source file.

    This does not gate on a specific string dtype value (e.g. `== object`),
    because pandas represents "a column of strings" differently across
    versions/configs — numpy object, pandas' nullable StringDtype, or (as of
    pandas 3.0's default) an Arrow-backed "str" dtype. Gating on one specific
    spelling silently skips the others, which is the exact bug this function
    previously had.

    Args:
        df: A dataframe whose columns were all read as strings.

    Returns:
        A new dataframe: safely-numeric columns promoted to a numeric dtype,
        everything else left as trimmed text.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        is_na = series.isna()
        stringified = series.astype(str).mask(is_na).str.strip()
        trimmed = stringified.astype(object)
        df[col] = trimmed
        non_null = trimmed.dropna()
        if non_null.empty:
            continue
        if _is_id_like_column(col, non_null):
            continue  # protected identifier — never promote to numeric
        numeric = pd.to_numeric(non_null, errors='coerce')
        if numeric.isna().any():
            continue  # not cleanly numeric for every value — leave as text
        df[col] = pd.to_numeric(trimmed, errors='coerce')
    return df


def load_file_to_df(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load any supported file format into a DataFrame.

    CSV/TSV/Excel/JSON are all read with every column as text FIRST, then
    each column is promoted to numeric only where doing so is provably safe
    (see _promote_safe_numeric_columns). This exists specifically to stop
    type inference from silently turning an ID like "007" into the integer 7
    during the initial read — before join logic, ID protection, or even the
    file summary shown to the planner ever sees the original text. Once that
    information is gone at read time, nothing downstream can recover it.

    JSON note: pandas' default read_json coerces even an EXPLICITLY QUOTED
    string like "007" to the integer 7 — the quoting in the source file is
    not enough on its own to protect it, `dtype=False` is required. This is
    different from a source JSON that stored the id as a bare, unquoted
    number (e.g. 7 with no quotes) — in that case the leading zero was
    already gone in the source file itself, before this function runs, and
    no read option can recover it. Excel has an analogous, unfixable
    limitation: a cell Excel stored as a genuine number (its default for
    anything typed as digits, unless the column was explicitly formatted as
    Text) has already lost the leading zero before pandas ever sees the
    file.
    """
    import io
    if len(file_bytes) > MAX_ZIP_MEMBER_BYTES:
        limit_mb = round(MAX_ZIP_MEMBER_BYTES / (1024 * 1024), 1)
        raise HTTPException(status_code=413, detail=f"{filename} is too large to parse safely. Maximum inner file size is {limit_mb} MB.")
    fname = filename.lower()
    if fname.endswith('.json'):
        df = pd.read_json(io.BytesIO(file_bytes), dtype=False)
        df = _promote_safe_numeric_columns(df)
    elif fname.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
        df = _promote_safe_numeric_columns(df)
    elif fname.endswith('.tsv'):
        df = pd.read_csv(io.BytesIO(file_bytes), sep='\t', dtype=str)
        df = _promote_safe_numeric_columns(df)
    else:
        # .csv and any unrecognized extension fall back to CSV parsing.
        df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
        df = _promote_safe_numeric_columns(df)
    enforce_dataframe_limits(df, filename)
    return df


def enforce_dataframe_limits(df: pd.DataFrame, label: str = "dataset") -> None:
    """Reject parsed dataframes that are too large for safe synchronous processing."""
    if len(df) > MAX_PARSED_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"{label} has {len(df):,} rows. Maximum supported rows for this workflow is {MAX_PARSED_ROWS:,}."
        )
    if len(df.columns) > MAX_PARSED_COLUMNS:
        raise HTTPException(
            status_code=413,
            detail=f"{label} has {len(df.columns):,} columns. Maximum supported columns for this workflow is {MAX_PARSED_COLUMNS:,}."
        )


def validate_zip_archive(zf, label: str = "ZIP file"):
    infos = [
        info for info in zf.infolist()
        if not info.filename.startswith('__MACOSX') and not Path(info.filename).name.startswith('.')
    ]
    if len(infos) > MAX_ZIP_MEMBERS:
        raise HTTPException(status_code=413, detail=f"{label} contains too many files. Maximum is {MAX_ZIP_MEMBERS}.")
    total_uncompressed = sum(max(0, info.file_size) for info in infos)
    if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
        limit_mb = round(MAX_ZIP_UNCOMPRESSED_BYTES / (1024 * 1024), 1)
        raise HTTPException(status_code=413, detail=f"{label} expands to more than {limit_mb} MB.")
    for info in infos:
        if info.file_size > MAX_ZIP_MEMBER_BYTES:
            limit_mb = round(MAX_ZIP_MEMBER_BYTES / (1024 * 1024), 1)
            raise HTTPException(status_code=413, detail=f"{Path(info.filename).name} is too large. Maximum inner file size is {limit_mb} MB.")
    return infos


def expand_zip_files(filename: str, raw: bytes) -> list[tuple[str, bytes]]:
    """
    If raw is a ZIP archive, extract every tabular file inside it and
    return a list of (filename, bytes) pairs. Otherwise return [(filename, raw)]
    unchanged. Used by the Organize flow so users can upload a single ZIP
    containing many CSV/Excel/JSON/TSV files.
    """
    import zipfile
    import io
    if not filename.lower().endswith('.zip'):
        return [(filename, raw)]

    TABULAR_EXT = ('.csv', '.xlsx', '.xls', '.json', '.tsv', '.txt')
    extracted = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            infos = validate_zip_archive(zf, filename)
            for info in infos:
                name = info.filename
                base = Path(name).name
                if not base.lower().endswith(TABULAR_EXT):
                    continue
                with zf.open(name) as fp:
                    extracted.append((base, fp.read()))
    except HTTPException:
        raise
    except Exception:
        return [(filename, raw)]  # let the caller's error handling take over

    return extracted if extracted else [(filename, raw)]


@app.post("/organize/analyze")
async def organize_analyze(
    files: list[UploadFile] = File(...),
    scope_instructions: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user)
):
    """
    Pass 1: Read all uploaded files, generate summaries, send to Claude
    for a transformation plan. Returns the plan for user review.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    plan = get_user_plan(user_id)
    check_and_enforce_limit(user_id, plan, "org_session_count", PLAN_ORG_SESSION_LIMITS, "Data Organization sessions")

    summaries = []
    file_store = {}  # Keep raw bytes for execution later
    dfs_for_dry_run = {}  # Keep loaded dataframes to compute real pre-execution join diagnostics

    # Expand any uploaded ZIPs into their individual tabular files first
    expanded = []
    for f in files:
        raw = await f.read()
        for fname, fbytes in expand_zip_files(f.filename, raw):
            expanded.append((fname, fbytes))

    for fname, raw in expanded:
        try:
            df = load_file_to_df(raw, fname)
            summary = summarize_dataframe(df, fname)
            summaries.append(summary)
            dfs_for_dry_run[fname] = df
            # Store in temp location keyed by filename
            tmp_path = DATA_DIR / f"organize_{user_id}_{fname.replace('/', '_')}"
            with open(str(tmp_path), 'wb') as fp:
                fp.write(raw)
        except Exception as e:
            summaries.append({"filename": fname, "error": str(e)})

    # Build prompt for Claude
    summaries_text = json.dumps(summaries, indent=2)

    scope_block = ""
    if scope_instructions:
        scope_block = f"""

USER SCOPING INSTRUCTIONS (apply these as a binding filter on the final merged dataset — this is not optional context, it changes what rows end up in the output): {scope_instructions}
If this instruction asks to keep only certain rows (e.g. a specific category, product, date range, or to exclude something), include a "filter_rows" transformation in your plan that expresses this filter precisely, using the exact column name and value(s) from the file summaries above. The filter should be applied after merging so it works across all source files consistently."""

    plan_prompt = f"""You are a professional data engineer. A user has uploaded {len(summaries)} files to be merged into a single clean dataset.

Here are the detailed summaries of each file:

{summaries_text}{scope_block}

Your job is to create a precise transformation plan to merge these files into ONE clean, coherent dataset.

Respond ONLY with a JSON object in this exact format:
{{
  "assessment": "Brief 2-3 sentence assessment of the data quality and mergeability",
  "merge_strategy": "join" | "append" | "mixed",
  "merge_key": "column_name_to_join_on, ONLY for a single-key single-step join, otherwise null",
  "base_file": "filename.csv — the file every join step joins INTO, required whenever you use joins below. Can ALSO be a list of filenames (e.g. [\"orders_jan.csv\", \"orders_feb.csv\"]) when several files share the same shape and need to be appended together FIRST, before the joins below enrich the combined result — this is the correct way to express 'append these monthly/regional files, then join in lookup tables', not something joins below can express on its own.",
  "joins": [
    {{
      "file": "filename.csv — the file being joined into the result so far",
      "on": "column_name" or ["column_a", "column_b"],
      "how": "left" | "outer" | "inner",
      "reason": "why this file joins on this key, and what entity/grain it adds",
      "aggregate": "OPTIONAL. Include this ONLY when the user wants the result at a HIGHER grain than this file (e.g. one row per customer, but this file is one row per ticket/order/event). Collapses this file to one row per group BEFORE the join, so the join is 1:1 and never fans out. Omit entirely for a normal same-grain join. Shape: {{\"group_by\": \"customer_id\", \"columns\": [{{\"as\": \"ticket_count\", \"agg\": \"count\"}}, {{\"as\": \"high_severity_ticket_count\", \"agg\": \"count\", \"filter_column\": \"severity\", \"filter_equals\": \"high\"}}, {{\"as\": \"latest_ticket_date\", \"agg\": \"max\", \"source\": \"ticket_date\"}}, {{\"as\": \"avg_resolution_hours\", \"agg\": \"mean\", \"source\": \"resolution_hours\"}}]}}. Allowed agg values: count, count_distinct, sum, mean, min, max, first, last. Never use sum/mean/min/max on an identifier column — use count or count_distinct for those instead."
    }}
  ],
  "id_columns": ["list of column names across all files that are identifiers (customer_id, sku, order_id, zip, etc.) and must be kept as text, never cast to a number"],
  "estimated_output_rows": <number>,
  "estimated_output_cols": <number>,
  "transformations": [
    {{
      "type": "rename_column",
      "file": "filename.csv",
      "from": "old_name",
      "to": "new_name",
      "reason": "why"
    }},
    {{
      "type": "fill_nulls",
      "column": "column_name",
      "strategy": "mean" | "median" | "mode" | "zero" | "unknown" | "forward_fill",
      "reason": "why"
    }},
    {{
      "type": "drop_column",
      "file": "filename.csv",
      "column": "column_name",
      "reason": "why"
    }},
    {{
      "type": "standardize_dates",
      "column": "column_name",
      "target_format": "YYYY-MM-DD",
      "reason": "why"
    }},
    {{
      "type": "deduplicate",
      "key_columns": ["col1", "col2"],
      "reason": "why"
    }},
    {{
      "type": "cast_type",
      "column": "column_name",
      "target_type": "int" | "float" | "string" | "date",
      "reason": "why"
    }},
    {{
      "type": "parse_delimited_column",
      "file": "filename.csv",
      "column": "column_name_holding_raw_delimited_string",
      "delimiter": ";",
      "new_columns": ["col_a", "col_b", "col_c"],
      "reason": "why — use this when a file's rows are still glued together as one string column instead of properly split (e.g. a CSV read with the wrong separator)"
    }},
    {{
      "type": "standardize_text",
      "column": "column_name",
      "placeholder_values": ["unknown", "n/a", "-", "null"],
      "case": "lower" | "upper" | "title" | null,
      "reason": "why — use this to convert placeholder strings to real nulls and/or normalize text casing"
    }},
    {{
      "type": "replace_value",
      "column": "column_name",
      "old_values": ["bad_value_1", "bad_value_2"],
      "new_value": null,
      "reason": "why — use this for targeted replacement of specific garbage values; new_value of null means replace with a true null"
    }},
    {{
      "type": "filter_invalid_values",
      "column": "column_name",
      "min_value": 0,
      "max_value": 850,
      "invalid_values": [-1, 9999],
      "action": "null" | "drop_rows",
      "reason": "why — use this for out-of-range numeric values like impossible credit scores or negative income; action 'null' keeps the row but blanks the bad value, 'drop_rows' removes the row entirely"
    }},
    {{
      "type": "filter_rows",
      "column": "column_name",
      "keep_values": ["value_to_keep_1", "value_to_keep_2"],
      "exclude_values": [],
      "reason": "why — use this ONLY when the user has given explicit scoping instructions to keep or exclude specific rows. keep_values means only rows matching these values survive; exclude_values means rows matching these values are removed. Use the exact column name and values as they appear in the file summaries."
    }}
  ],
  "warnings": ["any data quality warnings the user should know about"],
  "confidence": "high" | "medium" | "low"
}}

CRITICAL: Only use the transformation "type" values shown above (rename_column, fill_nulls, drop_column, standardize_dates, deduplicate, cast_type, parse_delimited_column, standardize_text, replace_value, filter_invalid_values, filter_rows). Do not invent new type names — if a cleanup step does not fit one of these exactly, either omit it or express it using the closest matching type above. Every transformation you list will actually be executed verbatim, so an unsupported type silently does nothing and the data will not match your assessment. Only include a filter_rows transformation if the user provided explicit scoping instructions — never filter rows on your own initiative.

JOIN GUIDANCE:
- Use "joins" (not the legacy single "merge_key") whenever the files need MORE THAN ONE join key across the dataset — e.g. orders join to customers on customer_id, then that result joins to products on sku. List each step in the order it should be applied; each step joins one file into the accumulated result so far.
- Only use the legacy top-level "merge_key" for the simple case of one shared key across all files. If you set "joins", leave "merge_key" null.
- If several files share the same shape and should be stacked together BEFORE any joins happen — the common case of monthly or regional extracts, e.g. orders_jan.csv + orders_feb.csv — set "base_file" to a LIST of those filenames instead of a single string. They will be appended together first, and the joins below then enrich that combined result. Do not try to express this by adding an append-like step inside "joins" — "joins" only ever joins one file at a time into the result, it cannot append multiple files together.
- Every join key you name MUST actually exist, under that exact name, in both the left (accumulated result) and right (file being joined) sides — check the file summaries above before naming a key. If two files that should relate to each other don't share a common column name for the relevant entity, say so in "warnings" instead of inventing a join.
- Do NOT use "cast_type" to convert an ID column (customer_id, sku, order_id, zip, account number, etc.) to a number — list it in "id_columns" instead. IDs are automatically protected from numeric coercion by the backend, and casting them yourself risks the model choosing to do it anyway in a way that isn't caught.
- Currency strings (e.g. "$1,200.50") and percent strings (e.g. "15.5%") are automatically cleaned before any numeric cast_type is applied — you do not need a separate transformation for this, just specify cast_type to "float" for that column as normal.
- When the desired final grain is HIGHER than one of the files being joined (e.g. the user wants one row per customer, but a lookup file like support_tickets.csv or orders.csv is one row per ticket/order), use a join step's "aggregate" field to collapse that file to one row per group BEFORE the join, instead of letting the join fan out. This turns "3 customers become 9 rows because each has multiple tickets" into "3 customers stay 3 rows, with new summary columns like ticket_count and latest_ticket_date." Only skip aggregation if the user actually wants the wider, one-row-per-event result (say so explicitly in "warnings" if that's genuinely what the request calls for).

Be thorough but conservative — only transform what is clearly necessary. Never drop data without a very good reason. If files cannot be cleanly merged, explain why in warnings."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=16000,
            messages=[{"role": "user", "content": plan_prompt}]
        )
        plan_text = message.content[0].text.strip()
        # Strip markdown fences
        if '```' in plan_text:
            lines = plan_text.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            plan_text = '\n'.join(lines).strip()
        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            # Response may have been truncated mid-string if the model hit
            # max_tokens on a complex multi-file plan. Try a simple repair:
            # close any unterminated string/object/array and re-parse.
            repaired = plan_text
            if repaired.count('"') % 2 == 1:
                repaired += '"'
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            repaired += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
            plan = json.loads(repaired)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")

    # ── PRE-EXECUTION REVIEW: deterministic dry-run diagnostics ──
    # Computed directly from the actual uploaded data (not the model's guess),
    # so the user can see real numbers — expected output rows, unmatched rows,
    # duplicate-key risk, and whether the join will expand or shrink the
    # dataset — before committing to /organize/execute. This runs the same
    # key-existence and normalization logic as execution, but without
    # performing the merge itself.
    dry_run_steps = []
    joins_plan = plan.get("joins")
    legacy_key = plan.get("merge_key")
    merge_strategy = plan.get("merge_strategy", "append")

    def _dry_run_one_step(left_name, left_df, right_name, right_df, join_keys, how, known_id_columns=None):
        """Preview one join step using the exact same engine execution uses.

        This performs the real merge (not a lighter approximation based on
        distinct key counts) so estimated_output_rows and row_multiplier
        reflect ACTUAL row expansion from duplicate keys on either side —
        a distinct-key estimate can miss a one-to-many grain change entirely
        when, say, the right side has many duplicate keys but few distinct
        ones. The merged dataframe is returned so the caller can reuse it to
        advance to the next step without re-running the join a second time.
        """
        merged, diag, blocker = _execute_join_step(
            left_df, right_df, join_keys, left_name, right_name, how=how, known_id_columns=known_id_columns
        )
        if blocker:
            return {
                "left_file": left_name, "right_file": right_name, "join_keys": join_keys,
                "will_succeed": False,
                "missing_on_left": blocker.get("missing_on_left", []),
                "missing_on_right": blocker.get("missing_on_right", []),
                "message": blocker["message"],
            }, left_df
        return {
            **diag,
            "will_succeed": True,
            "estimated_output_rows": diag["output_rows"],
            "grain_change_risk": bool(diag["row_multiplier"] and diag["row_multiplier"] > _GRAIN_WARNING_ROW_MULTIPLIER),
            # Calls the EXACT SAME function /organize/execute uses to decide
            # whether to block — not a second, separately-tuned threshold
            # check that could quietly drift out of sync with it. That drift
            # is exactly what happened before this fix: this dry-run field
            # and the real execute-time gate used different duplicate-key
            # thresholds, so a preview could show "no risk" while execution
            # still blocked, or vice versa.
            "requires_confirmation": _check_risky_fanout(diag, right_file, join_keys, confirmed=False) is not None,
            "duplicate_key_risk": bool(diag["right_duplicate_key_rows"] > 0),
        }, merged

    id_columns_hint = set(plan.get("id_columns") or [])

    if joins_plan and dfs_for_dry_run:
        fnames = list(dfs_for_dry_run.keys())
        base_file_spec = plan.get("base_file", fnames[0])
        base_files = base_file_spec if isinstance(base_file_spec, list) else [base_file_spec]
        base_files = [f for f in base_files if f in dfs_for_dry_run] or [fnames[0]]
        if len(base_files) > 1:
            running_df = pd.concat([dfs_for_dry_run[f] for f in base_files], ignore_index=True, sort=False)
            running_name = "+".join(base_files)
        else:
            running_df = dfs_for_dry_run[base_files[0]]
            running_name = base_files[0]
        for step in joins_plan:
            right_file = step.get("file")
            on = step.get("on")
            how = step.get("how", "left")
            if right_file not in dfs_for_dry_run or not on:
                dry_run_steps.append({
                    "left_file": running_name, "right_file": right_file, "will_succeed": False,
                    "message": f"Join step references '{right_file}', which is not among the uploaded files, or has no 'on' key.",
                })
                continue
            join_keys = on if isinstance(on, list) else [on]

            right_df_for_preview = dfs_for_dry_run[right_file]
            agg_spec = step.get("aggregate")
            if agg_spec:
                # Preview must reflect the AGGREGATED join, not the raw one —
                # otherwise the preview would show a fanout-risk warning for
                # a join that execution will actually perform safely (or,
                # worse, the reverse: preview looks clean while an invalid
                # aggregation spec later blocks at execute time).
                agg_df, agg_blocker = _prepare_aggregated_lookup(
                    right_df_for_preview, agg_spec, right_file, known_id_columns=id_columns_hint
                )
                if agg_blocker:
                    dry_run_steps.append({
                        "left_file": running_name, "right_file": right_file, "will_succeed": False,
                        "message": agg_blocker["message"],
                    })
                    continue
                right_df_for_preview = agg_df

            step_result, running_df = _dry_run_one_step(
                running_name, running_df, right_file, right_df_for_preview, join_keys, how, id_columns_hint
            )
            dry_run_steps.append(step_result)
            if step_result["will_succeed"]:
                running_name = f"{running_name}+{right_file}"
    elif legacy_key and dfs_for_dry_run:
        fnames = list(dfs_for_dry_run.keys())
        running_df = dfs_for_dry_run[fnames[0]]
        running_name = fnames[0]
        for right_file in fnames[1:]:
            step_result, running_df = _dry_run_one_step(
                running_name, running_df, right_file, dfs_for_dry_run[right_file], [legacy_key], "outer", id_columns_hint
            )
            dry_run_steps.append(step_result)
            if step_result["will_succeed"]:
                running_name = f"{running_name}+{right_file}"

    # Catch a real bug found in production: a plan can declare
    # merge_strategy="join" (or "mixed") while providing no usable join
    # instruction at all (empty "joins", no "merge_key"). Naive branching
    # would silently treat this as append and report success with no
    # warning — surfacing it here, in the SAME dry_run_steps list the
    # dashboard already checks via any_step_will_block, means the user sees
    # this before ever clicking execute, with zero new frontend wiring
    # required.
    plan_classification = _classify_merge_plan(merge_strategy, joins_plan, legacy_key)
    if plan_classification == "incomplete_join_plan":
        dry_run_steps.append({
            "left_file": None, "right_file": None, "will_succeed": False,
            "message": (
                f"The plan specified merge_strategy '{merge_strategy}' but provided no usable join "
                f"instruction — 'joins' was empty and 'merge_key' was not set. Executing this plan as-is "
                f"would silently fall back to appending the files instead, which contradicts what the "
                f"plan said it would do."
            ),
        })

    pre_execution_review = {
        "operation_type": "join" if (joins_plan or legacy_key) else "append",
        "join_steps": dry_run_steps,
        "any_step_will_block": any(not s.get("will_succeed", True) for s in dry_run_steps),
        "any_grain_change_risk": any(s.get("grain_change_risk") for s in dry_run_steps),
        "any_duplicate_key_risk": any(s.get("duplicate_key_risk") for s in dry_run_steps),
        "any_requires_confirmation": any(s.get("requires_confirmation") for s in dry_run_steps),
    }

    return {
        "plan": plan,
        "file_summaries": summaries,
        "session_key": f"organize_{user_id}",
        "pre_execution_review": _json_safe(pre_execution_review),
    }


def _json_safe(obj):
    """Recursively replace NaN/Infinity floats with None.

    json.dumps and FastAPI's response serializer both choke on NaN/Infinity,
    which pandas produces constantly (e.g. a mean of an all-null column).
    Shared by both /organize/analyze and /organize/execute so there is one
    place responsible for this, not two copies that could drift.

    Args:
        obj: Any JSON-serializable-ish structure (dict, list, tuple, or scalar).

    Returns:
        The same structure with NaN/Infinity floats replaced by None.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# Join risk tuning. Two different thresholds, deliberately: a lower one just
# WARNS (informational — e.g. any meaningful one-to-many expansion is worth
# mentioning), and a higher one BLOCKS execution until the client explicitly
# confirms — reserved for genuinely large, easy-to-miss fanout (e.g. 50,000
# rows quietly becoming 83,000). Warning at the same low bar used for
# blocking would make routine one-to-many joins (customers->orders) require
# confirmation constantly, which trains users to click through warnings
# without reading them — the opposite of what a confirmation gate is for.
_GRAIN_WARNING_ROW_MULTIPLIER: float = 1.05
_RISKY_ROW_MULTIPLIER_THRESHOLD: float = 1.5
_RISKY_DUPLICATE_KEY_FRACTION: float = 0.2

_ID_NAME_PATTERN = re.compile(
    r'(?:^|[_\s])(id|sku|code|zip|zipcode|postal|account|acct|upc|isbn)(?:[_\s]|$)',
    re.IGNORECASE,
)
# Catches date/time-label columns (week, month, quarter, year, date, period,
# day) so they're never framed as ordinary segment "drivers" — a raw time
# label with no computed time_series stats behind it is exactly the kind of
# thing that invites the model to invent a trend it has no real numbers for.
_TIME_LIKE_NAME_PATTERN = re.compile(
    r'(?:^|_)(week|month|quarter|year|date|period|day)(?:_|$)',
    re.IGNORECASE,
)
# Catches human-style compound names like "Account Number", "Invoice Number",
# "PO Number", "Reference No", "Tracking#" — none of which match the plain
# token list above, since the identifying word only becomes ID-like when
# paired with "number"/"no"/"#". Requiring that pairing (rather than adding
# bare "po", "order", etc. to the list above) keeps this from false-firing on
# unrelated columns like "PO Box" or "Order Total".
_ID_NUMBER_SUFFIX_PATTERN = re.compile(
    r'(?:^|[_\s])(account|invoice|po|order|reference|ref|confirmation|tracking|'
    r'serial|routing|policy|ticket|case|phone|fax|ssn|ein)[_\s]*(number|num|no\.?|#)(?:[_\s]|$)',
    re.IGNORECASE,
)
# Coarse pre-filter used only to skip an expensive, warning-prone
# pd.to_datetime() attempt on columns that obviously aren't dates (names,
# free text, ID values). Deliberately loose (a bare 4-digit run, a common
# date separator, or a month name/abbreviation) — it exists to cut down
# wasted attempts, not to be a precise date classifier; pd.to_datetime with
# errors='coerce' plus a notna() check is still what actually decides.
_DATE_LIKE_HINT_PATTERN = re.compile(
    r'\d{4}|[\-/]\d{1,2}[\-/]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec',
    re.IGNORECASE,
)


def _is_id_like_column(col_name: str, series: pd.Series, known_id_columns: Optional[set] = None) -> bool:
    """Decide whether a column holds identifier values that must never be
    silently coerced to a numeric type.

    A column is treated as ID-like if any of the following hold:
    - The planner explicitly named it in "id_columns".
    - Its name matches a common identifier naming pattern (e.g. ends in
      "_id", or is "sku"/"code"/"zip").
    - Its name is a human-style compound like "Account Number", "Invoice
      Number", "PO Number", or "Tracking #" (see _ID_NUMBER_SUFFIX_PATTERN).
    - Any of its non-null string values have a leading zero that numeric
      casting would destroy (e.g. "007" -> 7).
    - Most of its non-null values are long (8+ digit) pure-digit strings —
      far more often an account/invoice/tracking number than a genuine
      measurement, even without a matching column name. This is a heuristic,
      not a certainty: it requires a clear majority of values to match, to
      avoid falsely protecting a real large-magnitude numeric column (e.g.
      revenue in cents). If it misfires on a specific dataset, this is the
      threshold to revisit.

    Args:
        col_name: The column's header name.
        series: The column's data, in its original (pre-cast) form.
        known_id_columns: Optional set of column names the planner explicitly
            identified as identifiers (from the plan's "id_columns" field).
            Compared case/whitespace-insensitively against col_name.

    Returns:
        True if this column should be protected from numeric coercion.
    """
    if known_id_columns:
        normalized_known = {str(c).strip().lower() for c in known_id_columns}
        if str(col_name).strip().lower() in normalized_known:
            return True
    name_str = str(col_name)
    if _ID_NAME_PATTERN.search(name_str) or _ID_NUMBER_SUFFIX_PATTERN.search(name_str):
        return True
    sample = series.dropna().astype(str).str.strip()
    if sample.empty:
        return False
    if sample.str.match(r'^0\d+$').any():
        return True
    long_numeric = sample.str.match(r'^\d{8,}$')
    if long_numeric.mean() > 0.8:
        return True
    return False


def _normalize_id_key(series: pd.Series) -> pd.Series:
    """Produce a normalized comparison key for ID-like join columns.

    Strips surrounding whitespace, normalizes casing to uppercase, and strips
    leading zeros from purely-numeric-looking strings, so formatting
    differences like "007" vs "7" or "abc-1" vs "ABC-1" are treated as the
    same key when matching rows in a join. This is used ONLY to decide which
    rows match — the caller is responsible for keeping the original,
    unmodified ID values in the output dataset; this function never replaces
    a column in place.

    Args:
        series: The raw join-key column from one side of a join.

    Returns:
        A Series of normalized string keys suitable for use as a merge key.
    """
    def strip_leading_zeros(value) -> str:
        # .astype(str) on an object-dtype Series does NOT stringify null values
        # (None/NaN survive as literal Python floats) — only .str accessor
        # methods skip nulls automatically, plain .map() does not. Without this
        # guard, a blank/placeholder join-key value (e.g. an empty promo code)
        # crashes here with AttributeError instead of being treated as "no key,
        # doesn't match anything" — which is the semantically correct outcome
        # for a genuinely missing key.
        if not isinstance(value, str):
            return value
        if value.isdigit():
            stripped = value.lstrip('0')
            return stripped if stripped else '0'
        return value

    cleaned = series.astype(str).str.strip().str.upper()
    return cleaned.map(strip_leading_zeros)


def _parse_currency_or_percent(series: pd.Series) -> pd.Series:
    """Clean common currency and percent string formats before numeric casting.

    Handles a leading dollar sign ("$1,200.50"), thousands separators
    ("1,200"), parenthesized negatives ("(400.00)" -> "-400.00"), and a
    trailing percent sign ("10%", "15.5%" -> "10", "15.5" — the literal
    number as shown, not divided by 100, since that is what a report should
    display unless the user asks for a fraction). Values that already look
    like plain numbers pass through unchanged, so this is safe to run before
    every numeric cast, not just ones known to contain currency/percent data.

    Args:
        series: A column of raw string/object values to clean before casting.

    Returns:
        A new Series of cleaned strings ready for pd.to_numeric.
    """
    def clean(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        text = str(value).strip()
        if text == "":
            return None
        negative = False
        if text.startswith('(') and text.endswith(')'):
            negative = True
            text = text[1:-1]
        text = text.replace('$', '').replace(',', '').replace('%', '').strip()
        if negative and text:
            text = '-' + text
        return text

    # Gate on "not already numeric" rather than "is object dtype" — pandas can
    # report string columns under several different dtypes depending on
    # version/config (numpy object, pandas' nullable StringDtype, Arrow-backed
    # strings). Gating only on `dtype == object` misses the others entirely,
    # silently skipping the cleanup and letting currency/percent values null
    # out during the numeric cast that follows.
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.map(clean)


def _compute_join_diagnostics(
    left_df: pd.DataFrame, right_df: pd.DataFrame, norm_cols: list[str],
    left_name: str, right_name: str
) -> dict:
    """Compute match-rate diagnostics for one join step, using normalized keys.

    Rows with a null normalized key (a genuinely missing/blank join-key
    value) are excluded from the matched/left-only/right-only merge and
    counted separately instead. Without this, pandas' merge would treat
    NaN == NaN as a match — two unrelated rows with a missing key would be
    counted as "matched" to each other, which both misreports the real match
    rate and (if not also handled in the actual merge) would silently
    fabricate a join between two unrelated records.

    Args:
        left_df: The left-hand dataframe, already carrying normalized key column(s).
        right_df: The right-hand dataframe, already carrying normalized key column(s).
        norm_cols: The normalized key column name(s) shared by both sides.
        left_name: Display name of the left file/accumulated result, for messages.
        right_name: Display name of the right file being joined in.

    Returns:
        A dict with left/right row counts, matched/left-only/right-only key
        counts, null-key row counts on each side, and duplicate-key row
        counts on each side.
    """
    left_null_mask = left_df[norm_cols].isna().any(axis=1)
    right_null_mask = right_df[norm_cols].isna().any(axis=1)
    left_keys = left_df.loc[~left_null_mask, norm_cols].drop_duplicates()
    right_keys = right_df.loc[~right_null_mask, norm_cols].drop_duplicates()
    merged_keys = left_keys.merge(right_keys, on=norm_cols, how='outer', indicator=True)
    matched = int((merged_keys['_merge'] == 'both').sum())
    left_only = int((merged_keys['_merge'] == 'left_only').sum())
    right_only = int((merged_keys['_merge'] == 'right_only').sum())
    return {
        "left_file": left_name,
        "right_file": right_name,
        "left_rows": int(len(left_df)),
        "right_rows": int(len(right_df)),
        "matched_keys": matched,
        "left_only_keys": left_only,
        "right_only_keys": right_only,
        "left_null_key_rows": int(left_null_mask.sum()),
        "right_null_key_rows": int(right_null_mask.sum()),
        "left_duplicate_key_rows": int(left_df.loc[~left_null_mask].duplicated(subset=norm_cols).sum()),
        "right_duplicate_key_rows": int(right_df.loc[~right_null_mask].duplicated(subset=norm_cols).sum()),
    }


def _check_risky_fanout(diag: dict, right_file: str, join_keys, confirmed: bool) -> Optional[dict]:
    """Decide whether a join step's row expansion is risky enough to block
    execution until the user explicitly confirms it.

    Two independent signals count as risky: the row count grew by more than
    _RISKY_ROW_MULTIPLIER_THRESHOLD, or a large fraction of the right-side
    file's rows share a duplicate key (a strong predictor of fanout even in
    edge cases where the resulting multiplier alone might look modest).

    Args:
        diag: The diagnostics dict returned by _execute_join_step for this step.
        right_file: Display name of the file being joined in, for the message.
        join_keys: The join key(s) used, for the message.
        confirmed: Whether the client already confirmed risky operations for
            this run (echoed back after a prior blocked response).

    Returns:
        A blocker dict if this step is risky and not yet confirmed, else None.
    """
    multiplier = diag.get("row_multiplier")
    right_rows = diag.get("right_rows") or 0
    dup_rows = diag.get("right_duplicate_key_rows") or 0
    dup_fraction = (dup_rows / right_rows) if right_rows else 0.0

    is_risky = bool(
        (multiplier and multiplier > _RISKY_ROW_MULTIPLIER_THRESHOLD)
        or dup_fraction > _RISKY_DUPLICATE_KEY_FRACTION
    )
    if not is_risky or confirmed:
        return None
    return {
        "type": "risky_row_expansion",
        "file": right_file,
        "join_keys": join_keys,
        "left_rows": diag.get("left_rows"),
        "output_rows": diag.get("output_rows"),
        "row_multiplier": multiplier,
        "right_duplicate_key_rows": dup_rows,
        "message": (
            f"Joining '{right_file}' on {join_keys} would expand the dataset from "
            f"{diag.get('left_rows')} rows to {diag.get('output_rows')} rows "
            f"({multiplier}x). This usually means the lookup file has duplicate keys "
            f"causing rows to repeat, not a data-entry error — but it's easy to miss, "
            f"so we stop here rather than silently produce a much larger dataset. "
            f"Review this, then resend the request with confirm_risky_operations: true "
            f"to proceed as-is."
        ),
    }


def _generate_join_plain_english_insights(join_summary: list[dict]) -> list[str]:
    """Generate plain-English sentences describing what each join actually did.

    These are computed directly from the join numbers (percentages, counts),
    not produced by pattern-matching existing technical warning strings —
    a percentage like "12% of orders did not match" needs to be calculated
    from matched/unmatched counts, it isn't present in any single warning
    string to translate. This is what actually answers the questions a user
    cares about: how much of my data matched, and why might row counts have
    changed.

    Args:
        join_summary: The per-join-step list built by _build_change_summary
            (joined_file, matched_rows, unmatched_left_rows,
            unmatched_right_rows, duplicate_keys_in_lookup_file,
            null_or_blank_key_rows, row_multiplier).

    Returns:
        A list of plain-English sentences, one or more per join step that
        has something worth surfacing. A clean join with no issues produces
        no sentences for that step.
    """
    insights = []
    for step in join_summary:
        joined_file = step.get("joined_file", "the lookup file")
        matched = step.get("matched_rows") or 0
        unmatched_left = step.get("unmatched_left_rows") or 0
        total_left = matched + unmatched_left
        dup_keys = step.get("duplicate_keys_in_lookup_file") or 0
        null_keys = step.get("null_or_blank_key_rows") or 0
        multiplier = step.get("row_multiplier")

        if total_left > 0 and unmatched_left > 0:
            pct = round((unmatched_left / total_left) * 100)
            insights.append(
                f"{pct}% of rows did not find a match in '{joined_file}' ({unmatched_left} of {total_left} rows)."
            )
        if dup_keys > 0:
            insights.append(
                f"'{joined_file}' has {dup_keys} duplicate key value(s), so some rows in the result were "
                f"duplicated to account for every match."
            )
        if null_keys > 0:
            insights.append(
                f"{null_keys} row(s) had a blank or missing join key and were left unmatched, rather than "
                f"being guessed at or dropped."
            )
        if multiplier and multiplier > _RISKY_ROW_MULTIPLIER_THRESHOLD:
            insights.append(
                f"Joining '{joined_file}' substantially increased the row count ({multiplier}x) — this was "
                f"confirmed before the dataset was finalized."
            )
    return insights


def _build_change_summary(
    fnames_in_order: list[str], original_counts: dict, total_original_rows: int,
    output_rows: int, output_cols: int, join_diagnostics: list[dict],
    base_files_used: list[str], transformations: list[dict],
) -> dict:
    """Assemble a single, human-readable record of what the organizer did.

    This exists so a user (or anyone auditing the output later) can see
    exactly what happened without having to reverse-engineer it from the raw
    join diagnostics — input files, whether each was appended or joined,
    rows before/after, unmatched-row counts per join, duplicate-key
    warnings, and every column that was renamed, cast, or newly created.

    Args:
        fnames_in_order: The uploaded filenames, in upload order.
        original_counts: Row count per input file, before any processing.
        total_original_rows: Sum of all input row counts.
        output_rows: Row count of the final organized dataset.
        output_cols: Column count of the final organized dataset.
        join_diagnostics: Per-join-step diagnostics, in execution order.
        base_files_used: The file(s) that formed the starting point before any
            joins ran — length > 1 means those files were appended together
            first (see the "joins" plan schema's list-valued "base_file").
        transformations: The plan's transformation list as executed.

    Returns:
        A dict with input_files, files_appended, files_joined,
        rows_before/after, join_summary (per step), and columns_changed.
    """
    files_joined = [d["right_file"] for d in join_diagnostics]
    # base_files_used has length > 1 only when those files were appended
    # together as the starting point (list-valued "base_file"). A plain
    # append run (no joins at all) passes every uploaded file here instead —
    # the caller is responsible for that distinction, since it already knows
    # which merge path executed and this function shouldn't have to
    # re-derive it from indirect signals.
    files_appended = list(base_files_used) if len(base_files_used) > 1 or not join_diagnostics else []

    join_summary = []
    for d in join_diagnostics:
        join_summary.append({
            "joined_file": d.get("right_file"),
            "matched_rows": d.get("matched_keys"),
            "unmatched_left_rows": d.get("left_only_keys"),
            "unmatched_right_rows": d.get("right_only_keys"),
            "duplicate_keys_in_lookup_file": d.get("right_duplicate_key_rows", 0),
            "null_or_blank_key_rows": d.get("left_null_key_rows", 0) + d.get("right_null_key_rows", 0),
            "row_count_after_join": d.get("output_rows"),
            "row_multiplier": d.get("row_multiplier"),
        })

    columns_changed = {
        "renamed": [
            {"file": t.get("file"), "from": t.get("from"), "to": t.get("to")}
            for t in transformations if t.get("type") == "rename_column"
        ],
        "cast": [
            {"column": t.get("column"), "target_type": t.get("target_type")}
            for t in transformations if t.get("type") == "cast_type"
        ],
        "created": [
            {"file": t.get("file"), "source_column": t.get("column"), "new_columns": t.get("new_columns")}
            for t in transformations if t.get("type") in ("parse_delimited_column", "split_column")
        ],
        "dropped": [
            {"file": t.get("file"), "column": t.get("column")}
            for t in transformations if t.get("type") == "drop_column"
        ],
    }

    return {
        "input_files": [{"filename": f, "rows": original_counts.get(f, 0)} for f in fnames_in_order],
        "files_appended": files_appended,
        "files_joined": files_joined,
        "rows_before_total": total_original_rows,
        "rows_after": output_rows,
        "columns_after": output_cols,
        "joins": join_summary,
        "columns_changed": columns_changed,
    }


def _build_organization_report_text(
    change_summary: dict, validation: dict, verification: dict,
    plain_english_insights: list[str], generated_at: str,
) -> str:
    """Render a downloadable, human-readable QA report for one organizer run.

    This is the artifact a user can save or hand to someone else as proof of
    exactly what the organizer did to their data — every input file, what
    was appended vs. joined, match rates, duplicate-key and null-key
    warnings, every column that was renamed/cast/created, and the final
    pass/fail verdict. Plain text on purpose: readable in any editor, safe
    to email, no rendering dependencies.

    Args:
        change_summary: The dict built by _build_change_summary.
        validation: The deterministic validation dict from the execute run.
        verification: The AI second-pass verification dict from the execute run.
        plain_english_insights: Sentences from _generate_join_plain_english_insights.
        generated_at: An ISO-8601 timestamp string for the report header.

    Returns:
        The full report as a single plain-text string.
    """
    lines = []
    lines.append("ARCHIMEDES DATA ORGANIZATION REPORT")
    lines.append(f"Generated: {generated_at}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("INPUT FILES")
    lines.append("-" * 60)
    for f in change_summary["input_files"]:
        lines.append(f"  {f['filename']}: {f['rows']:,} rows")
    lines.append("")

    lines.append("WHAT HAPPENED")
    lines.append("-" * 60)
    if change_summary["files_appended"]:
        lines.append(f"  Appended together: {', '.join(change_summary['files_appended'])}")
    if change_summary["files_joined"]:
        lines.append(f"  Joined in: {', '.join(change_summary['files_joined'])}")
    lines.append(f"  Rows before (sum of inputs): {change_summary['rows_before_total']:,}")
    lines.append(f"  Rows after: {change_summary['rows_after']:,}")
    lines.append(f"  Columns in final dataset: {change_summary['columns_after']}")
    lines.append("")

    if change_summary["joins"]:
        lines.append("JOIN DETAILS")
        lines.append("-" * 60)
        for j in change_summary["joins"]:
            lines.append(f"  Join with {j['joined_file']}:")
            lines.append(f"    Matched: {j['matched_rows']:,} rows")
            lines.append(f"    Did not match (left side): {j['unmatched_left_rows']:,} rows")
            lines.append(f"    Did not match (lookup side): {j['unmatched_right_rows']:,} rows")
            if j["duplicate_keys_in_lookup_file"]:
                lines.append(f"    Duplicate keys in lookup file: {j['duplicate_keys_in_lookup_file']:,}")
            if j["null_or_blank_key_rows"]:
                lines.append(f"    Blank/missing key rows: {j['null_or_blank_key_rows']:,}")
            lines.append(f"    Row count after this join: {j['row_count_after_join']:,}")
            lines.append("")

    changed = change_summary["columns_changed"]
    if any(changed.values()):
        lines.append("COLUMNS CHANGED")
        lines.append("-" * 60)
        for r in changed["renamed"]:
            lines.append(f"  Renamed: {r['from']} -> {r['to']} ({r['file']})")
        for c in changed["cast"]:
            lines.append(f"  Cast to {c['target_type']}: {c['column']}")
        for cr in changed["created"]:
            lines.append(f"  Split into new columns: {cr['source_column']} -> {cr['new_columns']} ({cr['file']})")
        for d in changed["dropped"]:
            lines.append(f"  Dropped: {d['column']} ({d['file']})")
        lines.append("")

    if plain_english_insights:
        lines.append("WHAT THIS MEANS FOR YOUR DATA")
        lines.append("-" * 60)
        for insight in plain_english_insights:
            lines.append(f"  - {insight}")
        lines.append("")

    if validation.get("warnings"):
        lines.append("TECHNICAL WARNINGS")
        lines.append("-" * 60)
        for w in validation["warnings"]:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("VALIDATION VERDICT")
    lines.append("-" * 60)
    lines.append(f"  Deterministic checks: {'PASSED' if validation.get('passed') else 'FAILED'}")
    lines.append(f"  AI review verdict: {verification.get('verdict', 'unknown')}")
    lines.append(f"  Summary: {verification.get('summary', 'n/a')}")
    if verification.get("issues"):
        lines.append("  Issues flagged by AI review:")
        for issue in verification["issues"]:
            lines.append(f"    - {issue}")

    return "\n".join(lines)



def _classify_merge_plan(merge_strategy: str, joins_plan, legacy_merge_key) -> str:
    """Decide which merge path a plan actually calls for, catching contradictions.

    This exists specifically to catch the case that let a real bug through:
    a plan can declare merge_strategy="join" (or "mixed") while providing no
    usable join instruction at all — an empty/missing "joins" list AND no
    "merge_key". Naive branching on "joins_plan is truthy, else fall back to
    append" silently treats that contradiction as a normal append, which
    directly contradicts what the plan claimed to do. This function makes
    that classification explicit and independently testable, rather than
    leaving it as an implicit side effect of an if/elif chain.

    Args:
        merge_strategy: The plan's "merge_strategy" field.
        joins_plan: The plan's "joins" field (may be None, [], or a list).
        legacy_merge_key: The plan's "merge_key" field (may be None or a string).

    Returns:
        One of: "multi_step_join" (use the joins_plan engine),
        "legacy_single_join" (use the single merge_key engine),
        "plain_append" (stack all files), or "incomplete_join_plan" (the
        contradiction described above — caller must block, not append).
    """
    if joins_plan:
        return "multi_step_join"
    if merge_strategy in ("join", "mixed") and not legacy_merge_key:
        return "incomplete_join_plan"
    if merge_strategy == "append" or not legacy_merge_key:
        return "plain_append"
    return "legacy_single_join"


_AGG_NUMERIC_TYPES = {"sum", "mean"}
_AGG_ORDERABLE_TYPES = {"min", "max"}
_AGG_ANY_TYPE = {"count", "count_distinct", "first", "last"}
_AGG_WHITELIST = _AGG_NUMERIC_TYPES | _AGG_ORDERABLE_TYPES | _AGG_ANY_TYPE


def _prepare_aggregated_lookup(
    df: pd.DataFrame, agg_spec: dict, file_name: str, known_id_columns: Optional[set] = None
) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Collapse a lower-grain lookup file to one row per group before a join.

    This is what lets "join customers to support tickets" produce one clean
    row per customer (ticket_count, latest_ticket_date, etc.) instead of
    either fanning out to one row per ticket or requiring the user to
    confirm a large row expansion every time. Every aggregation is validated
    against a strict whitelist before anything is computed — an unsupported
    aggregation type, a missing source column, or summing/averaging an
    identifier column is a blocker, never a silent guess. This mirrors the
    same principle as the rest of this join engine: an ambiguous operation
    must stop and ask, not produce a confidently-wrong number.

    Args:
        df: The raw (pre-aggregation) lookup dataframe, e.g. support_tickets.csv.
        agg_spec: The plan step's "aggregate" dict: {"group_by": "customer_id",
            "columns": [{"as": "ticket_count", "agg": "count"}, ...]}.
        file_name: Display name of the file being aggregated, for messages.
        known_id_columns: Planner-flagged identifier columns, passed through
            to the same ID-protection heuristic used elsewhere.

    Returns:
        A tuple (aggregated_df, blocker). Exactly one is non-None: if
        blocker is set, nothing was computed and the caller must stop.
    """
    group_by = agg_spec.get("group_by")
    columns_spec = agg_spec.get("columns") or []

    if not group_by:
        return None, {"type": "invalid_aggregation_spec", "file": file_name,
                       "message": f"Aggregation for '{file_name}' has no 'group_by' column specified."}
    if group_by not in df.columns:
        return None, {"type": "invalid_aggregation_spec", "file": file_name,
                       "message": f"Aggregation group_by column '{group_by}' does not exist in '{file_name}'."}
    if not columns_spec:
        return None, {"type": "invalid_aggregation_spec", "file": file_name,
                       "message": f"Aggregation for '{file_name}' specifies no output columns."}

    result_cols: dict[str, pd.Series] = {}
    count_like_columns: set[str] = set()
    grouped = df.groupby(group_by, dropna=False)

    for col_spec in columns_spec:
        as_name = col_spec.get("as")
        agg = col_spec.get("agg")
        source = col_spec.get("source")
        filter_column = col_spec.get("filter_column")
        filter_equals = col_spec.get("filter_equals")

        if not as_name:
            return None, {"type": "invalid_aggregation_spec", "file": file_name,
                           "message": f"An aggregation column in '{file_name}' has no 'as' output name."}
        if agg not in _AGG_WHITELIST:
            return None, {"type": "invalid_aggregation_spec", "file": file_name,
                           "message": f"Aggregation type '{agg}' for output column '{as_name}' is not "
                                      f"supported. Allowed: {sorted(_AGG_WHITELIST)}."}

        if agg == "count" and filter_column:
            if filter_column not in df.columns:
                return None, {"type": "invalid_aggregation_spec", "file": file_name,
                               "message": f"Aggregation filter column '{filter_column}' does not exist in '{file_name}'."}
            mask = df[filter_column].astype(str).str.strip().str.lower() == str(filter_equals).strip().lower()
            result_cols[as_name] = df[mask].groupby(group_by, dropna=False).size()
            count_like_columns.add(as_name)
            continue

        if agg == "count":
            result_cols[as_name] = grouped.size()
            count_like_columns.add(as_name)
            continue

        if not source:
            return None, {"type": "invalid_aggregation_spec", "file": file_name,
                           "message": f"Aggregation column '{as_name}' (agg='{agg}') has no 'source' column specified."}
        if source not in df.columns:
            return None, {"type": "invalid_aggregation_spec", "file": file_name,
                           "message": f"Aggregation source column '{source}' does not exist in '{file_name}'."}

        series = df[source]
        is_id = _is_id_like_column(source, series, known_id_columns)

        if agg == "count_distinct":
            result_cols[as_name] = grouped[source].nunique()
            count_like_columns.add(as_name)
            continue

        if agg in ("first", "last"):
            result_cols[as_name] = grouped[source].first() if agg == "first" else grouped[source].last()
            continue

        if agg in _AGG_NUMERIC_TYPES:
            if is_id:
                return None, {"type": "invalid_aggregation_spec", "file": file_name,
                               "message": f"Cannot {agg} column '{source}' — it looks like an identifier "
                                          f"column, {agg}ming it would produce a meaningless number."}
            cleaned = _parse_currency_or_percent(series) if not pd.api.types.is_numeric_dtype(series) else series
            numeric = pd.to_numeric(cleaned, errors='coerce')
            if numeric.notna().sum() == 0:
                return None, {"type": "invalid_aggregation_spec", "file": file_name,
                               "message": f"Column '{source}' could not be reliably treated as numeric for {agg}."}
            temp = pd.DataFrame({group_by: df[group_by], "_val": numeric})
            grouped_val = temp.groupby(group_by, dropna=False)["_val"]
            result_cols[as_name] = grouped_val.sum() if agg == "sum" else grouped_val.mean()
            continue

        if agg in _AGG_ORDERABLE_TYPES:
            if is_id:
                return None, {"type": "invalid_aggregation_spec", "file": file_name,
                               "message": f"Cannot compute {agg} of column '{source}' — it looks like an "
                                          f"identifier column."}
            cleaned = _parse_currency_or_percent(series) if not pd.api.types.is_numeric_dtype(series) else series
            numeric = pd.to_numeric(cleaned, errors='coerce')
            if numeric.notna().sum() > 0 and numeric.notna().mean() > 0.5:
                temp = pd.DataFrame({group_by: df[group_by], "_val": numeric})
                grouped_val = temp.groupby(group_by, dropna=False)["_val"]
                result_cols[as_name] = grouped_val.max() if agg == "max" else grouped_val.min()
                continue
            parsed_dates = pd.to_datetime(series, errors='coerce')
            if parsed_dates.notna().sum() > 0 and parsed_dates.notna().mean() > 0.5:
                temp = pd.DataFrame({group_by: df[group_by], "_val": parsed_dates})
                grouped_val = temp.groupby(group_by, dropna=False)["_val"]
                result_cols[as_name] = grouped_val.max() if agg == "max" else grouped_val.min()
                continue
            return None, {"type": "invalid_aggregation_spec", "file": file_name,
                           "message": f"Column '{source}' is neither reliably numeric nor a date — cannot "
                                      f"compute {agg} safely."}

    agg_df = pd.DataFrame(result_cols)
    for col in count_like_columns:
        agg_df[col] = agg_df[col].fillna(0).astype(int)
    agg_df = agg_df.reset_index().rename(columns={"index": group_by})
    return agg_df, None


def _execute_join_step(
    left_df: pd.DataFrame, right_df: pd.DataFrame, join_keys: list[str],
    left_name: str, right_name: str, how: str = "left",
    known_id_columns: Optional[set] = None,
) -> tuple[pd.DataFrame, Optional[dict], Optional[dict]]:
    """Execute one join step with ID-safe key matching and full diagnostics.

    Verifies every join key exists on both sides before doing anything — if a
    key is missing, this returns a blocker instead of silently falling back
    to an append (the exact failure mode that let a mixed join/append output
    look successful). Join keys are matched using a normalized comparison
    (whitespace-trimmed, case-normalized, and leading-zero-stripped for
    numeric-looking IDs) so formatting differences like "007" vs "7" match
    correctly. For "left" joins the original left-side ID values are kept
    as-is; for "outer"/"right" joins, right-only rows (which have no left-side
    value to keep) fall back to the right side's own original ID value so
    they are never left null.

    Args:
        left_df: The accumulated result so far (or the first file's data).
        right_df: The next file's data to join in.
        join_keys: Column name(s), present under the same name on both sides,
            to join on.
        left_name: Display name for the left side, used in diagnostics/messages.
        right_name: Display name for the right side, used in diagnostics/messages.
        how: Pandas join type. Defaults to "left" so unmatched right-side rows
            don't manufacture new left-side rows unless explicitly requested.
        known_id_columns: Optional set of column names the planner explicitly
            flagged as identifiers, passed through to ID detection so a column
            like "Account Number" is protected even without a matching name
            pattern or leading-zero values.

    Returns:
        A 3-tuple (merged_df, diagnostics, blocker). Exactly one of
        diagnostics/blocker is non-None: if blocker is set, the join was NOT
        executed and merged_df is just left_df unchanged — the caller must
        stop and surface the blocker rather than proceeding as if it succeeded.
    """
    missing_left = [k for k in join_keys if k not in left_df.columns]
    missing_right = [k for k in join_keys if k not in right_df.columns]
    if missing_left or missing_right:
        side = []
        if missing_left:
            side.append(f"missing from '{left_name}': {missing_left}")
        if missing_right:
            side.append(f"missing from '{right_name}': {missing_right}")
        blocker = {
            "type": "missing_join_key",
            "left_file": left_name,
            "right_file": right_name,
            "join_keys": join_keys,
            "missing_on_left": missing_left,
            "missing_on_right": missing_right,
            "message": (
                f"Cannot join '{right_name}' on {join_keys}: the key is "
                f"{' and '.join(side)}. Choose a different join key for this "
                f"file, or confirm it should be appended instead of joined."
            ),
        }
        return left_df, None, blocker

    left_work = left_df.copy()
    right_work = right_df.copy()
    norm_cols = []
    id_format_mismatch = False

    for key in join_keys:
        id_like = (
            _is_id_like_column(key, left_work[key], known_id_columns)
            or _is_id_like_column(key, right_work[key], known_id_columns)
        )
        norm_col = f"__normkey__{key}"
        if id_like:
            left_work[norm_col] = _normalize_id_key(left_work[key])
            right_work[norm_col] = _normalize_id_key(right_work[key])
        else:
            left_work[norm_col] = left_work[key].astype(str).str.strip()
            right_work[norm_col] = right_work[key].astype(str).str.strip()
        # An empty string after normalization means "no key" just as much as
        # a real NaN does — without this, two rows with a blank string key
        # (as opposed to a true null) would still spuriously match each
        # other, the same bug just for a different representation of "empty".
        left_work[norm_col] = left_work[norm_col].replace('', pd.NA)
        right_work[norm_col] = right_work[norm_col].replace('', pd.NA)
        norm_cols.append(norm_col)

        # Precisely detect a formatting-only mismatch: for keys that DO match
        # (present on both sides after normalization), did the raw string
        # representation actually differ between the two sides (e.g. "007" on
        # the left, "7" on the right)? Comparing whole raw-value SETS (as an
        # earlier version did) overfires constantly, since two files rarely
        # have identical customer populations for unrelated reasons — that is
        # normal partial overlap, not a formatting problem. Restricting the
        # comparison to only the keys that actually matched avoids that.
        matched_norm_keys = set(left_work[norm_col]) & set(right_work[norm_col])
        if matched_norm_keys:
            combined_raw = pd.concat([
                left_work[[norm_col, key]].rename(columns={key: "_raw"}),
                right_work[[norm_col, key]].rename(columns={key: "_raw"}),
            ])
            combined_raw["_raw"] = combined_raw["_raw"].astype(str).str.strip()
            matched_subset = combined_raw[combined_raw[norm_col].isin(matched_norm_keys)]
            distinct_raw_per_key = matched_subset.groupby(norm_col)["_raw"].nunique()
            if (distinct_raw_per_key > 1).any():
                id_format_mismatch = True

    diagnostics = _compute_join_diagnostics(left_work, right_work, norm_cols, left_name, right_name)

    # Keep a copy of the right side's own original key column(s) so an
    # outer/right join can recover the ID value for right-only rows, which
    # would otherwise be null (the left side has no row to supply it).
    right_key_rename = {k: f"__rightkey__{k}" for k in join_keys}
    right_work_with_backup_keys = right_work.rename(columns=right_key_rename)
    right_reduced = right_work_with_backup_keys.drop(columns=join_keys, errors="ignore")

    # pandas' merge treats NaN == NaN as a MATCH by default — this is easy to
    # assume is safe (two unrelated missing values "shouldn't" be treated as
    # the same key) but is NOT how pandas actually behaves. Left unguarded,
    # two rows with a genuinely missing/blank join key (e.g. two orders with
    # no promo code) would be silently matched to each other, or to a
    # right-side row that also happens to have a missing key — exactly the
    # kind of silent-wrong-data bug this whole join engine exists to prevent.
    # To guarantee this never happens regardless of how "missing" ends up
    # represented, rows with any null normalized key are pulled out of the
    # merge entirely and reattached afterward as unconditionally-unmatched.
    left_null_mask = left_work[norm_cols].isna().any(axis=1)
    right_null_mask = right_reduced[norm_cols].isna().any(axis=1)
    left_mergeable = left_work[~left_null_mask]
    left_nullkey = left_work[left_null_mask]
    right_mergeable = right_reduced[~right_null_mask]
    right_nullkey = right_reduced[right_null_mask]

    merged_core = pd.merge(
        left_mergeable, right_mergeable, on=norm_cols, how=how,
        suffixes=("", f"__{right_name}"),
    )

    extra_pieces = []
    if len(left_nullkey) and how in ("left", "outer"):
        padded = left_nullkey.reindex(columns=merged_core.columns)
        extra_pieces.append(padded)
    if len(right_nullkey) and how in ("right", "outer"):
        padded = right_nullkey.reindex(columns=merged_core.columns)
        extra_pieces.append(padded)
    merged = pd.concat([merged_core, *extra_pieces], ignore_index=True, sort=False) if extra_pieces else merged_core

    for key in join_keys:
        backup_col = f"__rightkey__{key}"
        if backup_col in merged.columns:
            if key in merged.columns:
                merged[key] = merged[key].fillna(merged[backup_col])
            else:
                merged[key] = merged[backup_col]
            merged = merged.drop(columns=[backup_col])
    merged = merged.drop(columns=norm_cols)

    output_rows = len(merged)
    left_rows = len(left_df)
    diagnostics["output_rows"] = output_rows
    diagnostics["row_multiplier"] = round(output_rows / left_rows, 3) if left_rows else None
    diagnostics["id_format_mismatch_detected"] = id_format_mismatch

    return merged, diagnostics, None


@app.post("/organize/execute")
async def organize_execute(
    request: Request,
    user_id: str = Depends(get_current_user)
):
    """
    Pass 2: Execute the approved transformation plan, validate output,
    run second-pass verification, return the clean dataset.
    """
    try:
        result = await _organize_execute_impl(request, user_id)
        increment_usage_count(user_id, "org_session_count")
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"organize_execute crashed: {tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


async def _organize_execute_impl(
    request: Request,
    user_id: str
):
    import math

    body = await request.json()
    plan = body.get("plan")
    file_summaries = body.get("file_summaries", [])
    session_key = body.get("session_key", f"organize_{user_id}")
    # The client sends this back as True only after the user has explicitly
    # reviewed and approved a risky join (large row expansion / heavy
    # duplicate-key fanout). Until then, such a join is treated as a blocker
    # — the same mechanism already used for a missing join key — so a
    # dataset that would quietly balloon in size can never be produced
    # without the user having seen and confirmed that first.
    confirm_risky_operations = bool(body.get("confirm_risky_operations", False))

    if not plan:
        raise HTTPException(status_code=400, detail="No plan provided")

    # Columns the planner explicitly flagged as identifiers (e.g. "Account
    # Number") — honored by both cast_type protection and join key matching,
    # even when the column doesn't match a naming pattern or have obvious
    # leading zeros.
    id_columns_hint = set(plan.get("id_columns") or [])

    # Reload files from temp storage
    dfs = {}
    for summary in file_summaries:
        if "error" in summary:
            continue
        fname = summary["filename"]
        tmp_path = DATA_DIR / f"organize_{user_id}_{fname.replace('/', '_')}"
        if not tmp_path.exists():
            raise HTTPException(status_code=400, detail=f"File {fname} not found — please re-upload")
        with open(str(tmp_path), 'rb') as f:
            raw = f.read()
        dfs[fname] = load_file_to_df(raw, fname)

    if not dfs:
        raise HTTPException(status_code=400, detail="No valid files to process")

    # Record original row counts for validation
    original_counts = {fname: len(df) for fname, df in dfs.items()}
    total_original_rows = sum(original_counts.values())
    original_column_counts = {fname: len(df.columns) for fname, df in dfs.items()}

    # Apply transformations
    transformations = plan.get("transformations", [])
    pre_merge_warnings = []

    for t in transformations:
        t_type = t.get("type")
        try:
            if t_type == "rename_column":
                fname = t.get("file")
                if fname in dfs:
                    dfs[fname] = dfs[fname].rename(columns={t["from"]: t["to"]})

            elif t_type == "drop_column":
                fname = t.get("file")
                col = t.get("column")
                if fname in dfs and col in dfs[fname].columns:
                    dfs[fname] = dfs[fname].drop(columns=[col])

            elif t_type == "cast_type":
                col = t.get("column")
                target = t.get("target_type")
                for fname, df in dfs.items():
                    if col in df.columns:
                        if target in ("int", "float"):
                            if _is_id_like_column(col, df[col], id_columns_hint):
                                # Protect identifiers: never coerce to a numeric
                                # type even if the plan asked for it, since that
                                # silently destroys leading zeros (e.g. "007" -> 7)
                                # and format (e.g. mixed alphanumeric SKUs).
                                pre_merge_warnings.append(
                                    f"Skipped numeric cast of '{col}' in {fname} — it looks like an "
                                    f"identifier column (e.g. leading zeros or an ID-style name). "
                                    f"IDs are kept as text to avoid corrupting them."
                                )
                                dfs[fname][col] = df[col].astype(str).str.strip()
                                continue
                            cleaned = _parse_currency_or_percent(df[col])
                            if target == "int":
                                dfs[fname][col] = pd.to_numeric(cleaned, errors='coerce').astype('Int64')
                            else:
                                dfs[fname][col] = pd.to_numeric(cleaned, errors='coerce')
                        elif target == "string":
                            dfs[fname][col] = df[col].astype(str)
                        elif target == "date":
                            dfs[fname][col] = pd.to_datetime(df[col], errors='coerce')

            elif t_type == "standardize_dates":
                col = t.get("column")
                for fname, df in dfs.items():
                    if col in df.columns:
                        dfs[fname][col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')

            elif t_type == "parse_delimited_column":
                # A column whose raw value is itself a delimited string that never
                # got split into real columns (e.g. a semicolon-delimited row that
                # arrived as one giant glued-together string). Split it into the
                # named new columns for the affected file.
                fname = t.get("file")
                col = t.get("column")
                delimiter = t.get("delimiter", ";")
                new_columns = t.get("new_columns")
                if fname in dfs and col in dfs[fname].columns:
                    split_cols = dfs[fname][col].astype(str).str.split(delimiter, expand=True)
                    if new_columns and len(new_columns) == split_cols.shape[1]:
                        split_cols.columns = new_columns
                    else:
                        split_cols.columns = [f"{col}_{i}" for i in range(split_cols.shape[1])]
                    dfs[fname] = pd.concat(
                        [dfs[fname].drop(columns=[col]), split_cols], axis=1
                    )

            elif t_type == "split_column":
                # Alias for parse_delimited_column under a different name the
                # planner sometimes uses.
                fname = t.get("file")
                col = t.get("column")
                delimiter = t.get("delimiter", ";")
                new_columns = t.get("new_columns")
                if fname in dfs and col in dfs[fname].columns:
                    split_cols = dfs[fname][col].astype(str).str.split(delimiter, expand=True)
                    if new_columns and len(new_columns) == split_cols.shape[1]:
                        split_cols.columns = new_columns
                    else:
                        split_cols.columns = [f"{col}_{i}" for i in range(split_cols.shape[1])]
                    dfs[fname] = pd.concat(
                        [dfs[fname].drop(columns=[col]), split_cols], axis=1
                    )

            elif t_type == "standardize_text":
                # Normalize casing/whitespace on a text column, and optionally
                # replace known placeholder strings with real nulls.
                col = t.get("column")
                placeholders = t.get("placeholder_values", ["unknown", "Unknown", "UNKNOWN", "n/a", "N/A",
                                                              "na", "NA", "-", "--", "null", "NULL", "None", "nan", "NaN"])
                case_mode = t.get("case", None)  # "lower" | "upper" | "title" | None
                for fname, df in dfs.items():
                    if col in df.columns:
                        series = df[col].astype(str).str.strip()
                        series = series.replace(placeholders, pd.NA)
                        if case_mode == "lower":
                            series = series.str.lower()
                        elif case_mode == "upper":
                            series = series.str.upper()
                        elif case_mode == "title":
                            series = series.str.title()
                        dfs[fname][col] = series

            elif t_type == "replace_value":
                # Replace specific placeholder/garbage values with null or a
                # given replacement, across one column.
                col = t.get("column")
                old_values = t.get("old_values") or ([t["old_value"]] if "old_value" in t else [])
                new_value = t.get("new_value", None)
                replacement = pd.NA if new_value is None else new_value
                for fname, df in dfs.items():
                    if col in df.columns and old_values:
                        dfs[fname][col] = df[col].replace(old_values, replacement)

        except Exception as e:
            print(f"Transformation error ({t_type}): {e}")
            # Continue — don't fail the whole job for one transformation

    # ── MERGE / JOIN EXECUTION ──
    # Two plan shapes are supported:
    #   Legacy: {"merge_strategy": "join"|"append", "merge_key": "col"}
    #   New:    {"joins": [{"file": "products.csv", "on": "sku", "how": "left"}, ...],
    #            "base_file": "orders.csv"}
    # Both paths run through the same _execute_join_step engine so there is
    # exactly one place that knows how to join two dataframes safely — no
    # separate, divergent code path that could silently fall back to append.
    merge_strategy = plan.get("merge_strategy", "append")
    legacy_merge_key = plan.get("merge_key")
    joins_plan = plan.get("joins")
    fnames_in_order = list(dfs.keys())

    join_diagnostics: list[dict] = []
    grain_warnings: list[str] = []
    blockers: list[dict] = []
    base_files_used: list[str] = []  # populated by whichever merge path runs, used by _build_change_summary

    if joins_plan:
        base_file_spec = plan.get("base_file", fnames_in_order[0])
        base_files_requested = base_file_spec if isinstance(base_file_spec, list) else [base_file_spec]
        missing_base_files = [f for f in base_files_requested if f not in dfs]
        if missing_base_files:
            # Previously this silently dropped any nonexistent filename and
            # fell back to the first uploaded file — exactly the "looks
            # successful but is actually wrong" failure mode this whole join
            # engine exists to prevent. A typo'd base_file must block, not
            # quietly substitute a different file the user never asked for.
            blockers.append({
                "type": "missing_base_file",
                "files": missing_base_files,
                "message": f"base_file references file(s) not among the uploaded files: {missing_base_files}. "
                           f"Check for a typo, or confirm the intended file was actually uploaded.",
            })
        base_files = [f for f in base_files_requested if f in dfs] or [fnames_in_order[0]]
        base_files_used = base_files

        if len(base_files) > 1:
            # Multiple base files means "append these together first, THEN
            # join the rest in" — the common real-world pattern of several
            # monthly/regional extracts that all share the same shape (e.g.
            # orders_jan.csv + orders_feb.csv) needing to become one dataset
            # before being enriched with lookup tables. pd.concat aligns
            # columns the same way plain append does, so files with slightly
            # different column sets still combine sensibly.
            result_df = pd.concat([dfs[f] for f in base_files], ignore_index=True, sort=False)
            result_name = "+".join(base_files)
        else:
            result_df = dfs[base_files[0]]
            result_name = base_files[0]
        joined_files = set(base_files)

        for step in joins_plan:
            right_file = step.get("file")
            on = step.get("on")
            how = step.get("how", "left")
            if right_file not in dfs:
                blockers.append({
                    "type": "unknown_file_in_join_step",
                    "message": f"Join step references '{right_file}', which was not among the uploaded files.",
                })
                continue
            join_keys = on if isinstance(on, list) else [on]

            right_df_for_join = dfs[right_file]
            agg_spec = step.get("aggregate")
            if agg_spec:
                # Collapse the lookup file to one row per group BEFORE
                # joining — this is what lets "join customers to support
                # tickets" produce one clean row per customer instead of
                # fanning out to one row per ticket. An invalid aggregation
                # spec blocks here rather than falling through to a raw,
                # un-aggregated join that would then trigger (or silently
                # miss) the fanout confirmation gate for the wrong reason.
                agg_df, agg_blocker = _prepare_aggregated_lookup(
                    right_df_for_join, agg_spec, right_file, known_id_columns=id_columns_hint
                )
                if agg_blocker:
                    blockers.append(agg_blocker)
                    continue
                right_df_for_join = agg_df

            merged, diag, blocker = _execute_join_step(
                result_df, right_df_for_join, join_keys, result_name, right_file, how=how,
                known_id_columns=id_columns_hint,
            )
            if blocker:
                blockers.append(blocker)
                continue
            result_df = merged
            result_name = f"{result_name}+{right_file}"
            joined_files.add(right_file)
            join_diagnostics.append(diag)
            if diag["row_multiplier"] and diag["row_multiplier"] > _GRAIN_WARNING_ROW_MULTIPLIER:
                grain_warnings.append(
                    f"Joining '{right_file}' on {join_keys} changed the dataset grain: the result "
                    f"went from {diag['left_rows']} rows to {diag['output_rows']} rows "
                    f"({diag['row_multiplier']}x) — each row on the left side is no longer unique, "
                    f"consistent with a one-to-many relationship (e.g. customer-level joining to "
                    f"order/ticket-level rows)."
                )
            risk_blocker = _check_risky_fanout(diag, right_file, join_keys, confirm_risky_operations)
            if risk_blocker:
                blockers.append(risk_blocker)
            if diag["id_format_mismatch_detected"]:
                grain_warnings.append(
                    f"Some IDs in the join with '{right_file}' matched only after normalizing "
                    f"formatting differences (e.g. '007' vs '7'). The rows were still matched, but "
                    f"consider standardizing ID formats at the source."
                )

        unused_files = [f for f in fnames_in_order if f not in joined_files]
        if unused_files:
            blockers.append({
                "type": "unused_file_in_join_plan",
                "files": unused_files,
                "message": f"These uploaded files were never joined or appended: {unused_files}. "
                           f"They would be silently excluded from the output — confirm this is intended.",
            })

    elif merge_strategy in ("join", "mixed") and not legacy_merge_key:
        # The plan explicitly declared it would join files (merge_strategy is
        # "join" or "mixed"), but provided no usable join instruction — an
        # empty/missing "joins" list AND no "merge_key". Falling through to
        # the append branch below would silently produce a result that
        # directly contradicts what the plan claimed to do: exactly the
        # "looks successful but is actually wrong" failure mode this whole
        # engine exists to prevent. A real example that triggered this: the
        # planner returned merge_strategy="join", merge_key=null, joins=[] —
        # execution then silently appended 12 rows with no join and no
        # warning, instead of stopping to flag the contradiction.
        blockers.append({
            "type": "incomplete_join_plan",
            "message": (
                f"The plan specified merge_strategy '{merge_strategy}' but provided no usable join "
                f"instruction — 'joins' was empty and 'merge_key' was not set. Rather than silently "
                f"falling back to appending the files (which would contradict what the plan said it "
                f"would do), this run stopped. Provide an explicit join key or join steps, or set "
                f"merge_strategy to 'append' if stacking the files is actually what's intended."
            ),
        })
        result_df = pd.concat(list(dfs.values()), ignore_index=True, sort=False)
        base_files_used = list(fnames_in_order)

    elif merge_strategy == "append" or not legacy_merge_key:
        df_list = list(dfs.values())
        result_df = pd.concat(df_list, ignore_index=True, sort=False)
        base_files_used = list(fnames_in_order)

    else:
        items = list(dfs.items())
        result_name, result_df = items[0]
        base_files_used = [items[0][0]]
        for right_name, right_df in items[1:]:
            merged, diag, blocker = _execute_join_step(
                result_df, right_df, [legacy_merge_key], result_name, right_name, how="outer",
                known_id_columns=id_columns_hint,
            )
            if blocker:
                blockers.append(blocker)
                continue
            result_df = merged
            result_name = f"{result_name}+{right_name}"
            join_diagnostics.append(diag)
            if diag["row_multiplier"] and diag["row_multiplier"] > _GRAIN_WARNING_ROW_MULTIPLIER:
                grain_warnings.append(
                    f"Joining '{right_name}' on '{legacy_merge_key}' changed the dataset grain: the "
                    f"result went from {diag['left_rows']} rows to {diag['output_rows']} rows "
                    f"({diag['row_multiplier']}x)."
                )
            risk_blocker = _check_risky_fanout(diag, right_name, [legacy_merge_key], confirm_risky_operations)
            if risk_blocker:
                blockers.append(risk_blocker)
            if diag["id_format_mismatch_detected"]:
                grain_warnings.append(
                    f"Some '{legacy_merge_key}' values matched only after normalizing formatting "
                    f"differences (e.g. '007' vs '7'). The rows were still matched, but consider "
                    f"standardizing ID formats at the source."
                )

    # Apply fill_nulls and deduplicate after merge
    for t in transformations:
        t_type = t.get("type")
        try:
            if t_type == "fill_nulls":
                col = t.get("column")
                strategy = t.get("strategy", "unknown")
                if col in result_df.columns:
                    if strategy == "mean" and pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(result_df[col].median())
                    elif strategy == "mode":
                        result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if len(result_df[col].mode()) > 0 else None)
                    elif strategy == "zero":
                        result_df[col] = result_df[col].fillna(0)
                    elif strategy == "forward_fill":
                        result_df[col] = result_df[col].ffill()
                    else:
                        result_df[col] = result_df[col].fillna("Unknown")

            elif t_type == "deduplicate":
                key_cols = t.get("key_columns", [])
                valid_cols = [c for c in key_cols if c in result_df.columns]
                if valid_cols:
                    result_df = result_df.drop_duplicates(subset=valid_cols)
                else:
                    result_df = result_df.drop_duplicates()

            elif t_type == "filter_invalid_values":
                # Remove or null-out values outside an allowed range/set for a column
                # (e.g. credit scores of -1 or 9999, negative income).
                col = t.get("column")
                if col in result_df.columns:
                    min_val = t.get("min_value")
                    max_val = t.get("max_value")
                    invalid_values = t.get("invalid_values")
                    action = t.get("action", "null")  # "null" sets to NaN, "drop_rows" removes the row

                    mask_invalid = pd.Series(False, index=result_df.index)
                    if min_val is not None and pd.api.types.is_numeric_dtype(result_df[col]):
                        mask_invalid |= result_df[col] < min_val
                    if max_val is not None and pd.api.types.is_numeric_dtype(result_df[col]):
                        mask_invalid |= result_df[col] > max_val
                    if invalid_values:
                        mask_invalid |= result_df[col].isin(invalid_values)

                    if action == "drop_rows":
                        result_df = result_df[~mask_invalid]
                    else:
                        result_df.loc[mask_invalid, col] = pd.NA

            elif t_type == "filter_rows":
                # User-requested scoping (e.g. "only keep Colorants category" or
                # "exclude Amazon orders"). Only applied when the plan includes
                # this step, which only happens when the user gave explicit
                # scope instructions — see the plan-generation prompt.
                col = t.get("column")
                if col in result_df.columns:
                    keep_values = t.get("keep_values") or []
                    exclude_values = t.get("exclude_values") or []
                    if keep_values:
                        result_df = result_df[result_df[col].isin(keep_values)]
                    if exclude_values:
                        result_df = result_df[~result_df[col].isin(exclude_values)]
        except Exception as e:
            print(f"Post-merge transformation error ({t_type}): {e}")

    result_df = result_df.reset_index(drop=True)
    enforce_dataframe_limits(result_df, "organized output")

    # ── VALIDATION PASS ──
    validation = {
        "passed": True,
        "checks": [],
        "warnings": []
    }

    # Check 1: Row count sanity
    output_rows = len(result_df)
    row_reducing_transformations = [
        t for t in transformations
        if t.get("type") in ("deduplicate", "filter_rows")
        or (t.get("type") == "filter_invalid_values" and t.get("action") == "drop_rows")
    ]
    used_join = bool(joins_plan or legacy_merge_key)
    row_expectation = {
        "merge_strategy": "join" if used_join else "append",
        "merge_key": legacy_merge_key,
        "input_total_rows": total_original_rows,
        "input_row_counts": original_counts,
        "input_column_counts": original_column_counts,
        "output_rows": output_rows,
        "output_columns": len(result_df.columns),
        "expected_logic": "",
        "expected_min_rows": None,
        "expected_max_rows": None,
        "join_diagnostics": join_diagnostics,
        "grain_warnings": grain_warnings,
        "blockers": blockers,
        "row_reducing_transformations_present": bool(row_reducing_transformations),
        "row_reducing_transformation_types": [t.get("type") for t in row_reducing_transformations],
    }

    if not used_join:
        expected_min = max(original_counts.values()) if original_counts else 1
        expected_max = total_original_rows
        row_expectation.update({
            "expected_logic": (
                "append stacks files vertically, so output rows should usually be close to "
                "the sum of input rows unless deduplication or explicit row filtering was applied"
            ),
            "expected_min_rows": expected_min,
            "expected_max_rows": expected_max,
        })
        if output_rows < expected_min and not row_reducing_transformations:
            validation["warnings"].append(
                f"Output has {output_rows} rows but largest input had {expected_min} rows — some data may have been lost"
            )
    else:
        row_expectation["expected_logic"] = (
            "join combines files using the key(s) diagnosed per step in join_diagnostics; input rows "
            "should NOT be summed. See join_diagnostics for match rates and grain_warnings for "
            "one-to-many expansion."
        )
        if join_diagnostics:
            expected_min_last = join_diagnostics[-1]["matched_keys"]
            row_expectation["expected_min_rows"] = expected_min_last
            if expected_min_last and output_rows < expected_min_last and not row_reducing_transformations:
                validation["warnings"].append(
                    f"Output has {output_rows} rows but the last join step matched "
                    f"{expected_min_last} keys — some matched rows may have been lost."
                )

    validation["warnings"].extend(grain_warnings)
    validation["warnings"].extend(pre_merge_warnings)

    validation["checks"].append({
        "name": "Row count",
        "strategy": row_expectation["merge_strategy"],
        "input_total": total_original_rows,
        "output": output_rows,
        "expectation": row_expectation,
        "ok": output_rows > 0 and (
            row_reducing_transformations or
            row_expectation["expected_min_rows"] is None or output_rows >= row_expectation["expected_min_rows"]
        ),
    })

    if blockers:
        validation["passed"] = False
        validation["blockers"] = blockers

    # Check 2: No empty dataset
    if len(result_df) == 0:
        validation["passed"] = False
        validation["checks"].append({"name": "Non-empty output", "ok": False})

    # Check 3: Column count reasonable
    validation["checks"].append({"name": "Column count", "value": len(result_df.columns), "ok": len(result_df.columns) > 0})

    # Check 4: No fully null columns introduced
    null_cols = [col for col in result_df.columns if result_df[col].isnull().all()]
    if null_cols:
        validation["warnings"].append(f"These columns are entirely null after merging: {null_cols}")

    # If any join step hit a blocker (missing key, unresolvable file reference,
    # or a file silently excluded from the plan), stop here. Do NOT run the
    # second-pass verification or save a downloadable file — a blocked run
    # must never be handed to the user dressed up as a success. The caller
    # needs to choose how to proceed (different key, confirm append, etc.)
    # before anything is finalized.
    if blockers:
        return {
            "success": False,
            "blocked": True,
            "rows": len(result_df),
            "cols": len(result_df.columns),
            "validation": _json_safe(validation),
            "blockers": _json_safe(blockers),
            "message": (
                "The organizer stopped before finishing because it hit a problem it can't safely "
                "resolve on its own. Review the blockers below and choose how to proceed — for "
                "example, pick a different join key for the affected file, or confirm it should be "
                "appended instead of joined."
            ),
        }

    # ── SECOND PASS: Claude verifies the output ──
    try:
        output_summary = summarize_dataframe(result_df, "merged_output.csv")
    except Exception as e:
        output_summary = {"filename": "merged_output.csv", "rows": len(result_df), "cols": len(result_df.columns), "error": str(e)}

    # Build the original-files summary defensively — file_summaries entries should
    # always be plain dicts, but we've seen unexplained TypeErrors here in production,
    # so coerce/skip anything unexpected instead of crashing the whole request.
    original_files_summary = []
    for idx, s in enumerate(file_summaries):
        if not isinstance(s, dict):
            continue
        if "error" in s:
            continue
        original_files_summary.append({
            "filename": s.get("filename", "unknown"),
            "rows": s.get("rows", 0),
            "cols": s.get("cols", 0)
        })

    verification_prompt = f"""You are a data quality engineer. A data merge operation just completed.

ORIGINAL FILES:
{json.dumps(original_files_summary, indent=2)}

MERGE STRATEGY AND ROW-COUNT EXPECTATION:
{json.dumps(_json_safe(row_expectation), indent=2)}

TRANSFORMATION PLAN THAT WAS APPLIED:
{json.dumps(_json_safe(plan.get("transformations", [])), indent=2)}

OUTPUT DATASET SUMMARY:
{json.dumps(_json_safe(output_summary), indent=2)}

Verify the output. Check for:
1. Data loss (missing rows or columns that should be there)
2. Type corruption (numbers became strings, dates became nulls, etc.)
3. Unexpected null increases
4. Any red flags

ROW-COUNT RULES:
- If merge_strategy is "append", files are stacked vertically. In that case, expected rows are usually near the sum of input rows, unless deduplication or explicit row filtering was applied.
- If merge_strategy is "join", files are combined using the per-step diagnostics in join_diagnostics (each entry has left_rows, right_rows, matched_keys, left_only_keys, right_only_keys, output_rows, row_multiplier). DO NOT sum rows across input files.
- A row_multiplier notably above 1.0 for a join step is EXPECTED and healthy when it represents a real one-to-many relationship (e.g. customers to orders) — it is only a problem if grain_warnings does not already flag it, or if the multiplier is implausibly large relative to the data.
- Only flag data loss if output rows are materially below matched_keys for the relevant join step, or if a blocker/missing-key situation is present.
- If row_reducing_transformations_present is true, smaller row counts may be intentional. Evaluate whether the reduction matches the listed transformations before flagging data loss.

Respond ONLY with JSON:
{{
  "verdict": "pass" | "pass_with_warnings" | "fail",
  "issues": ["list of specific issues found, or empty list if none"],
  "summary": "One sentence summary of the output quality"
}}"""

    try:
        verify_msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": verification_prompt}]
        )
        verify_text = verify_msg.content[0].text.strip()
        if '```' in verify_text:
            lines = verify_text.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            verify_text = '\n'.join(lines).strip()
        verification = json.loads(verify_text)
    except Exception as e:
        verification = {"verdict": "pass_with_warnings", "issues": [f"Verification check failed: {str(e)}"], "summary": "Could not complete second-pass verification"}

    # Save result CSV to temp storage
    result_path = DATA_DIR / f"organized_{user_id}.csv"
    result_df.to_csv(str(result_path), index=False)

    # Also save as uploadable file_id for instant model training
    file_id = str(uuid.uuid4())
    pkl_path = DATA_DIR / f"{file_id}.pkl"
    result_df.to_pickle(str(pkl_path))
    _write_file_meta(file_id, user_id)

    # Clean up temp files
    for summary in file_summaries:
        if "error" not in summary:
            tmp_path = DATA_DIR / f"organize_{user_id}_{summary['filename'].replace('/', '_')}"
            if tmp_path.exists():
                tmp_path.unlink()

    preview_rows = [_json_safe(row) for row in result_df.head(5).to_dict(orient="records")]

    # Build the human-readable change summary, plain-English insights, and a
    # downloadable QA report — the record of exactly what happened to this
    # data, independent of whether the user reads the technical diagnostics.
    change_summary = _build_change_summary(
        fnames_in_order, original_counts, total_original_rows,
        len(result_df), len(result_df.columns), join_diagnostics,
        base_files_used, transformations,
    )
    plain_english_insights = _generate_join_plain_english_insights(change_summary["joins"])
    generated_at = datetime.utcnow().isoformat() + "Z"
    report_text = _build_organization_report_text(
        change_summary, validation, verification, plain_english_insights, generated_at,
    )
    report_path = DATA_DIR / f"organize_report_{user_id}.txt"
    report_path.write_text(report_text)

    return {
        "success": True,
        "rows": len(result_df),
        "cols": len(result_df.columns),
        "columns": list(result_df.columns),
        "validation": _json_safe(validation),
        "verification": _json_safe(verification),
        "file_id": file_id,  # Can be used directly for model training
        "preview": preview_rows,
        "download_ready": True,
        "change_summary": _json_safe(change_summary),
        "plain_english_insights": plain_english_insights,
        "report_ready": True
    }


@app.get("/organize/download")
async def organize_download(
    user_id: str = Depends(get_current_user)
):
    """Download the organized dataset as CSV."""
    result_path = DATA_DIR / f"organized_{user_id}.csv"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="No organized dataset found — please run the organizer first")
    return FileResponse(
        str(result_path),
        filename="organized_dataset.csv",
        media_type="text/csv"
    )


@app.get("/organize/download-report")
async def organize_download_report(
    user_id: str = Depends(get_current_user)
):
    """Download the QA report explaining exactly what the organizer did.

    Covers input files, what was appended vs. joined, match rates,
    duplicate-key and null-key warnings, every column that was renamed/cast/
    created, plain-English insights, and the final validation verdict — the
    artifact a user can save or share as proof of what happened to their
    data, independent of the clean CSV itself.
    """
    report_path = DATA_DIR / f"organize_report_{user_id}.txt"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No organization report found — please run the organizer first")
    return FileResponse(
        str(report_path),
        filename="organization_report.txt",
        media_type="text/plain"
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"service": "Archimedes MD", "status": "operational", "version": "2.0.0", "mode": "real_training"}

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    from fastapi.responses import Response
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...), target_col: Optional[str] = None, user_id: str = Depends(get_current_user)):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = round(MAX_UPLOAD_BYTES / (1024 * 1024), 1)
        raise HTTPException(status_code=413, detail=f"File is too large. Maximum upload size is {limit_mb} MB.")

    try:
        if ext == '.csv':
            sample = content[:2000].decode('utf-8', errors='ignore')
            delimiter = ';' if sample.count(';') > sample.count(',') else ','
            df = pd.read_csv(io.BytesIO(content), sep=delimiter)

        elif ext == '.json':
            try:
                df = pd.read_json(io.BytesIO(content))
            except Exception:
                # Try as JSON lines
                df = pd.read_json(io.BytesIO(content), lines=True)

        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(io.BytesIO(content))

        elif ext == '.zip':
            import zipfile
            df = None
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zip_infos = validate_zip_archive(zf, file.filename)
                all_names = [info.filename for info in zip_infos]

                # First try: find a tabular data file (CSV, Excel, JSON, TSV)
                tabular = sorted(
                    [n for n in all_names if Path(n).suffix.lower() in ['.csv','.xlsx','.xls','.json','.tsv']],
                    key=lambda n: (
                        0 if n.endswith('.csv') else
                        1 if n.endswith(('.xlsx','.xls')) else
                        2 if n.endswith('.json') else 3
                    )
                )
                for name in tabular:
                    inner_ext = Path(name).suffix.lower()
                    try:
                        inner = zf.read(name)
                        if inner_ext == '.csv':
                            sample = inner[:2000].decode('utf-8', errors='ignore')
                            delim = ';' if sample.count(';') > sample.count(',') else ','
                            df = pd.read_csv(io.BytesIO(inner), sep=delim)
                        elif inner_ext in ['.xlsx', '.xls']:
                            df = pd.read_excel(io.BytesIO(inner))
                        elif inner_ext == '.json':
                            try:
                                df = pd.read_json(io.BytesIO(inner))
                            except Exception:
                                df = pd.read_json(io.BytesIO(inner), lines=True)
                        elif inner_ext == '.tsv':
                            df = pd.read_csv(io.BytesIO(inner), sep='\t')
                        if df is not None and len(df) > 0 and len(df.columns) > 1:
                            enforce_dataframe_limits(df, name)
                            break
                        else:
                            df = None
                    except Exception:
                        df = None
                        continue

                # Second try: image classification folder structure
                # Pattern: folder/ClassName/image.jpg  or  split/ClassName/image.jpg
                if df is None:
                    image_exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
                    image_files = [n for n in all_names if Path(n).suffix.lower() in image_exts]

                    if image_files:
                        try:
                            from PIL import Image as PILImage
                        except ImportError:
                            PILImage = None

                        rows = []
                        for name in image_files:
                            parts = Path(name).parts
                            # Label = the immediate parent folder name
                            label = parts[-2] if len(parts) >= 2 else 'unknown'
                            # Split = grandparent folder (Training/Testing/etc) if present
                            split = parts[-3] if len(parts) >= 3 else 'all'
                            row = {
                                'filename': Path(name).name,
                                'label': label,
                                'split': split,
                            }
                            # Extract basic image stats if PIL available
                            if PILImage:
                                try:
                                    img_bytes = zf.read(name)
                                    img = PILImage.open(io.BytesIO(img_bytes)).convert('L')  # grayscale
                                    img_small = img.resize((32, 32))
                                    arr = np.array(img_small, dtype=float)
                                    row['mean_brightness'] = round(float(arr.mean()), 4)
                                    row['std_brightness']  = round(float(arr.std()),  4)
                                    row['min_brightness']  = round(float(arr.min()),  4)
                                    row['max_brightness']  = round(float(arr.max()),  4)
                                    row['width']  = img.width
                                    row['height'] = img.height
                                    # Top-left / center / bottom-right quadrant means
                                    h, w = arr.shape
                                    row['q_topleft']     = round(float(arr[:h//2, :w//2].mean()), 4)
                                    row['q_topright']    = round(float(arr[:h//2, w//2:].mean()), 4)
                                    row['q_bottomleft']  = round(float(arr[h//2:, :w//2].mean()), 4)
                                    row['q_bottomright'] = round(float(arr[h//2:, w//2:].mean()), 4)
                                    row['center_mean']   = round(float(arr[h//4:3*h//4, w//4:3*w//4].mean()), 4)
                                except Exception:
                                    pass
                            rows.append(row)

                        if rows:
                            df = pd.DataFrame(rows)
                            enforce_dataframe_limits(df, file.filename)
                            # Use 'label' as the auto-detected target
                            target_col = 'label'

            if df is None:
                raise HTTPException(status_code=400, detail="No readable data found inside the ZIP. Supported: CSV, Excel, JSON, or an image folder dataset (e.g. Training/ClassName/image.jpg).")

        elif ext == '.ipynb':
            import json as json_mod
            notebook = json_mod.loads(content.decode('utf-8', errors='ignore'))
            # Extract dataframe-like outputs from notebook cells
            rows = []
            headers = None
            for cell in notebook.get('cells', []):
                for output in cell.get('outputs', []):
                    text = output.get('text', [])
                    if isinstance(text, list):
                        text = ''.join(text)
                    if not text:
                        data_out = output.get('data', {})
                        text = ''.join(data_out.get('text/plain', []))
                    # Look for pipe-table or space-aligned table output
                    lines = [l.rstrip() for l in text.split('\n') if l.strip()]
                    if len(lines) >= 2 and ('  ' in lines[0] or '|' in lines[0]):
                        try:
                            from io import StringIO
                            df_try = pd.read_csv(StringIO(text), sep=r'\s{2,}', engine='python', skiprows=0)
                            if len(df_try.columns) > 1 and len(df_try) > 0:
                                rows.append(df_try)
                        except Exception:
                            pass
            # Also look for CSV data saved in notebook source
            for cell in notebook.get('cells', []):
                src = ''.join(cell.get('source', []))
                if 'to_csv' in src or 'read_csv' in src:
                    pass  # source analysis only, can't run code
            if rows:
                df = pd.concat(rows, ignore_index=True).drop_duplicates()
            else:
                raise HTTPException(status_code=400, detail="No tabular data found in the notebook. Make sure cells have DataFrame output (e.g. df.head()) or save the data as a CSV inside the notebook.")

        elif ext in ['.html', '.htm']:
            try:
                tables = pd.read_html(io.BytesIO(content))
                if not tables:
                    raise HTTPException(status_code=400, detail="No tables found in the HTML file.")
                # Use the largest table
                df = max(tables, key=len)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not extract table from HTML: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: CSV, Excel, JSON, ZIP, IPYNB, HTML.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    enforce_dataframe_limits(df, file.filename)

    data_path = DATA_DIR / f"{file_id}.pkl"
    ensure_file_owner(file_id, user_id)
    df.to_pickle(str(data_path))
    _write_file_meta(file_id, user_id)

    analysis = analyze_dataframe(df, target_col)
    analysis["file_id"] = file_id
    analysis["filename"] = file.filename
    analysis["data_path"] = str(data_path)
    analysis["preview"] = df.head(5).to_dict(orient='records')
    analysis["column_samples"] = {
        col: {
            "samples": [str(v) for v in df[col].dropna().unique()[:5]],
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "missing": int(df[col].isna().sum())
        }
        for col in df.columns
    }
    warnings = []
    if analysis["rows"] < 1000:
        warnings.append(f"Small dataset ({analysis['rows']} rows) — results may not be reliable")
    if target_col and target_col in df.columns:
        target_n_unique = df[target_col].nunique()
        if analysis["task"] == "classification" and target_n_unique > 20:
            warnings.append(f"Target column '{target_col}' has {target_n_unique} unique values — consider using Regression instead")
        if analysis["task"] == "regression" and target_n_unique < 10:
            warnings.append(f"Target column '{target_col}' has only {target_n_unique} unique values — consider using Classification instead")
    analysis["warnings"] = warnings
    analysis["describe"] = df.describe().to_dict()

    return clean(analysis)


@app.post("/train-real")
async def train_real(
    file_id: str,
    model_name: str,
    target_col: str,
    task: str,
    algorithm: str,
    domain: str = "general",
    user_id: str = Depends(get_current_user)
):
    data_path = DATA_DIR / f"{file_id}.pkl"
    ensure_file_owner(file_id, user_id)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Data file not found. Please upload again.")

    try:
        df = pd.read_pickle(str(data_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load data: {str(e)}")

    # Check model limit before training
    can_save, current_count, limit = check_model_limit(user_id)
    if not can_save:
        raise HTTPException(
            status_code=403,
            detail=f"Model limit reached. Your plan allows {limit} saved model{'s' if limit != 1 else ''}. You have {current_count}. Please upgrade to save more."
        )

    model_id = str(uuid.uuid4())

    try:
        metrics, model_path = train_real_model(df, target_col, task, algorithm, model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    registry = load_registry()
    registry[model_id] = {
        "id":           model_id,
        "file_id":      file_id,
        "name":         model_name,
        "domain":       domain,
        "task":         task,
        "algorithm":    algorithm,
        "target_col":   target_col,
        "metrics":      metrics,
        "model_path":   model_path,
        "created":      datetime.utcnow().strftime("%Y-%m-%d"),
        "user_id":      user_id,
        "version":      1,
        "feature_names": metrics.get("feature_names", []),
        "buffer_count": 0,
        "total_samples": metrics.get("train_samples", 0) + metrics.get("test_samples", 0),
        "real_model":   True,
        "accuracy":     f"{metrics.get('accuracy', metrics.get('r2_score', 0))}{'%' if task == 'classification' else ' R²'}"
    }
    save_registry(registry)

    return {
        "model_id":   model_id,
        "model_name": model_name,
        "task":       task,
        "algorithm":  algorithm,
        "metrics":    metrics,
        "version":    1,
        "message":    f"Real model trained successfully on {metrics.get('train_samples',0)} samples. Tested on {metrics.get('test_samples',0)} held-out samples."
    }


@app.post("/predict/{model_id}")
async def predict(model_id: str, input_data: dict, user_id: str = Depends(get_current_user)):
    try:
        registry = load_registry()
        ensure_model_owner(model_id, user_id, registry)
        result = predict_with_model(model_id, input_data)
        append_to_buffer(model_id, {
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_data,
            "prediction": result
            # true_label is intentionally absent — predictions don't carry ground truth.
            # Use POST /model/{model_id}/feedback to submit the actual outcome later.
        })
        if model_id in registry:
            buf = load_buffer(model_id)
            registry[model_id]["buffer_count"] = len(buf)
            save_registry(registry)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class FeedbackRequest(BaseModel):
    """Request body for submitting ground-truth outcome after a prediction.

    Attributes:
        input: The original input row that was predicted on.
        prediction: The model's prediction result (optional — include if known).
        true_label: The actual real-world outcome. Required — this is what
            retraining uses as the target label, not the model's own prediction.
    """
    input: dict
    prediction: Optional[dict] = None
    true_label: str  # required — no ground truth = no retraining value


@app.post("/model/{model_id}/feedback")
async def submit_feedback(
    model_id: str,
    req: FeedbackRequest,
    user_id: str = Depends(get_current_user)
):
    """Record a ground-truth outcome for a previous prediction.

    This is how the retraining buffer accumulates real-world data rather than
    model guesses. The /predict endpoint intentionally does not record true_label
    because the actual outcome is only known later (e.g. did the customer churn?
    did the transaction succeed?). Clients call this endpoint once the outcome
    is confirmed.

    Args:
        model_id: The model this feedback relates to.
        req: Feedback payload with the original input and confirmed true_label.
        user_id: Authenticated user ID from the request token.

    Returns:
        A dict with ground_truth_count reflecting how many labelled entries
        are now in the buffer and whether retrain threshold is met.
    """
    registry = load_registry()
    ensure_model_owner(model_id, user_id, registry)
    meta = registry.get(model_id, {})

    # Coerce true_label to the correct type based on model task.
    # Regression targets must be numeric — storing a string like "3.7" would
    # poison the target column during retraining and fail silently.
    task = meta.get("task", "classification")
    coerced_label: str | float = req.true_label
    if task == "regression":
        try:
            coerced_label = float(req.true_label)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"This model uses regression (numeric target). "
                       f"true_label must be a number, got: {req.true_label!r}"
            )

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": req.input,
        "true_label": coerced_label,
        "source": "feedback",
    }
    if req.prediction is not None:
        entry["prediction"] = req.prediction

    append_to_buffer(model_id, entry)

    buf = load_buffer(model_id)
    ground_truth_count = sum(1 for e in buf if e.get("true_label") is not None)
    retrain_ready = ground_truth_count >= 10

    if model_id in registry:
        registry[model_id]["buffer_count"] = len(buf)
        registry[model_id]["ground_truth_count"] = ground_truth_count
        save_registry(registry)

    return {
        "model_id": model_id,
        "buffer_count": len(buf),
        "ground_truth_count": ground_truth_count,
        "retrain_ready": retrain_ready,
    }



@app.get("/model/{model_id}")
def get_model(model_id: str, user_id: str = Depends(get_current_user)):
    registry = load_registry()
    return ensure_model_owner(model_id, user_id, registry)


@app.get("/models")
def list_models(user_id: str = Depends(get_current_user)):
    registry = load_registry()
    models = [v for v in registry.values() if v.get("user_id") == user_id]
    models.sort(key=lambda x: x["created"], reverse=True)
    return {"models": models, "count": len(models)}


@app.get("/model/{model_id}/download")
def download_model(model_id: str, user_id: str = Depends(get_current_user)):
    registry = load_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail="Model not found")
    if registry[model_id].get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your model")
    meta = registry[model_id]
    path = Path(meta.get("model_path", ""))

    # Restore from Supabase if not on local disk
    if not path.exists():
        print(f"Model {model_id} not on disk — restoring from Supabase for download...")
        if not download_model_from_supabase(model_id, str(path)):
            raise HTTPException(status_code=404, detail="Model file not found in storage")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file could not be restored")

    return FileResponse(path, filename=f"{meta['name']}_v{meta.get('version',1)}.pkl", media_type="application/octet-stream")


@app.post("/model/{model_id}/retrain")
async def retrain_model(model_id: str, user_id: str = Depends(get_current_user)):
    registry = load_registry()
    meta = ensure_model_owner(model_id, user_id, registry)
    buf = load_buffer(model_id)
    if not buf:
        return {"message": "No new data in buffer.", "version": meta.get("version", 1)}

    file_id = meta.get("file_id")
    if file_id:
        data_path = DATA_DIR / f"{file_id}.pkl"
        ensure_file_owner(file_id, user_id)
        if data_path.exists():
            original_df = pd.read_pickle(str(data_path))
            new_rows = []
            for entry in buf:
                # Require explicit true_label — never use model's own prediction
                # as the training label. Self-training on guesses reinforces errors
                # and degrades accuracy silently over time.
                true_label = entry.get("true_label")
                if true_label is None:
                    continue  # skip entries without ground truth
                row = entry.get("input", {})
                row[meta["target_col"]] = true_label
                new_rows.append(row)
            if not new_rows:
                return {"message": "No entries with ground-truth labels in buffer. Add true_label when logging predictions to enable retraining.", "version": meta.get("version", 1)}
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                merged_df = pd.concat([original_df, new_df], ignore_index=True)
                metrics, model_path = train_real_model(
                    merged_df, meta["target_col"], meta["task"], meta["algorithm"], model_id
                )
                new_version = meta.get("version", 1) + 1
                registry[model_id].update({
                    "version":       new_version,
                    "model_path":    model_path,
                    "metrics":       metrics,
                    "feature_names":  metrics.get("feature_names", []),
                    "buffer_count":  0,
                    "last_retrained": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                    "accuracy":      f"{metrics.get('accuracy', metrics.get('r2_score', 0))}{'%' if meta['task'] == 'classification' else ' R²'}"
                })
                save_registry(registry)
                clear_buffer(model_id)
                return {"model_id": model_id, "new_version": new_version, "metrics": metrics, "samples_absorbed": len(buf)}

    return {"message": "Could not retrain — original data not found.", "version": meta.get("version", 1)}


@app.delete("/model/{model_id}")
def delete_model(model_id: str, user_id: str = Depends(get_current_user)):
    registry = load_registry()
    if model_id not in registry:
        raise HTTPException(status_code=404, detail="Model not found")
    if registry[model_id].get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your model")
    meta = registry.pop(model_id)
    Path(meta.get("model_path", "")).unlink(missing_ok=True)
    clear_buffer(model_id)
    save_registry(registry)
    return {"deleted": model_id}


@app.post("/summarize")
async def summarize(data: dict, user_id: str = Depends(get_current_user)):
    try:
        model_name = data.get("model_name", "")
        algo       = data.get("algo", "")
        domain     = data.get("domain", "")
        result     = data.get("result", {})
        input_data = data.get("input_data", {})

        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system="""You are an AI assistant that explains machine learning predictions accurately.
CRITICAL RULE: You must interpret the prediction label in the context of what the model was trained to predict.
The target column and its meaning is provided. For example:
- If target is "default" and prediction is "Yes" = this person WILL DEFAULT (high risk, negative outcome)
- If target is "default" and prediction is "No" = this person will NOT default (low risk, positive outcome)
- If target is "Outcome" and prediction is "1" = DIABETIC (positive for disease)
- If target is "Survived" and prediction is "1" = SURVIVED
Always interpret the label correctly based on the target column context.
NEVER reverse or soften a negative outcome.
Format as clean bullet points:
• **Prediction:** State what the prediction actually means in plain language based on the target column
• **Confidence:** What the confidence percentage means practically
• **Key Factors:** 2-3 most important input values driving this prediction
• **Recommendation:** One honest practical next step matching the actual prediction
If Prediction result contains input_warnings, model_reliability_warnings, out_of_range_values, unknown_categories, or low input_completeness_pct, clearly say the prediction should be treated cautiously. Do not describe confidence as reliability when warnings are present.
Be accurate and clear. No jargon.""",
            messages=[{"role": "user", "content": f"""Model: {model_name} ({algo})
Domain: {domain}
Target column being predicted: {data.get("target_col", "unknown")}
Prediction result: {json.dumps(result)}
Input data: {json.dumps(input_data)}

Write a plain English summary. Remember to interpret the prediction label correctly based on what the target column means."""}]
        )
        return {"summary": message.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/{model_id}/buffer")
def check_buffer(model_id: str, user_id: str = Depends(get_current_user)):
    registry = load_registry()
    ensure_model_owner(model_id, user_id, registry)
    buf = load_buffer(model_id)
    ground_truth_count = sum(1 for e in buf if e.get("true_label") is not None)
    return {
        "model_id": model_id,
        "buffer_count": len(buf),
        "ground_truth_count": ground_truth_count,
        "retrain_ready": ground_truth_count >= 10,
    }
