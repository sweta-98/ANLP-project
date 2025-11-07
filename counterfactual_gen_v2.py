# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertForMaskedLM, GPT2LMHeadModel, GPT2TokenizerFast
)
from sentence_transformers import SentenceTransformer, util
from lime.lime_text import LimeTextExplainer
import shap
from difflib import SequenceMatcher

# %%
cache_dir = "/ssd_scratch/sweta.jena/new"
dataset_name="imdb"
class_names = ["negative", "positive"]
# sim_threshold=0.85
# ppl_threshold=100

# %%
# sentiment clf
clf_name = "textattack/bert-base-uncased-imdb"   
clf_tokenizer = AutoTokenizer.from_pretrained(clf_name, cache_dir = cache_dir)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name, cache_dir=cache_dir)
clf_model.eval()

# masked LM
mlm_name = "bert-base-uncased"                   
mlm_tokenizer = AutoTokenizer.from_pretrained(mlm_name, cache_dir = cache_dir)
mlm_model = BertForMaskedLM.from_pretrained(mlm_name, cache_dir=cache_dir)
mlm_model.eval()


# fluency
gpt2_name = "gpt2"                               
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_name, cache_dir = cache_dir)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_name, cache_dir=cache_dir)
gpt2_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    clf_model = torch.nn.DataParallel(clf_model)
    mlm_model = torch.nn.DataParallel(mlm_model)
    gpt2_model = torch.nn.DataParallel(gpt2_model)
clf_model.to(device)
mlm_model.to(device)
gpt2_model.to(device)

#semantic sim
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

# %%


# %%
def predict_proba(texts, batch_size=64):
    
    processed_texts = []
    for t in texts:
        if isinstance(t, (list, np.ndarray)):
            processed_texts.append(clf_tokenizer.decode(t, skip_special_tokens=True))
        else:
            processed_texts.append(str(t))

    all_probs = []
    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i:i+batch_size]
        encodings = clf_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = clf_model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)


# %%
def generate_candidates(text, word, top_k=10):
    tokens = mlm_tokenizer.tokenize(text, max_length=512, truncation=True)
    if word not in tokens:
        return []

    idx = tokens.index(word)
    tokens[idx] = mlm_tokenizer.mask_token
    masked_text = mlm_tokenizer.convert_tokens_to_string(tokens)
    inputs = mlm_tokenizer(masked_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        logits = mlm_model(**inputs).logits
    mask_index = torch.where(inputs["input_ids"][0] == mlm_tokenizer.mask_token_id)[0].item()
    probs = torch.softmax(logits[0, mask_index], dim=0)
    top_tokens = torch.topk(probs, top_k).indices.tolist()
    candidates = [mlm_tokenizer.decode([t]) for t in top_tokens]
    return candidates

def mask_subword(text, subword):
    # tokenize to wordpiece ids
    encoding = mlm_tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = encoding.input_ids[0]

    # tokenize target into wordpieces
    target_ids = mlm_tokenizer.encode(subword, add_special_tokens=False)

    # find matching subsequence
    for i in range(len(ids) - len(target_ids)):
        if ids[i:i+len(target_ids)].tolist() == target_ids:
            # replace first piece with [MASK], rest with padding token to remove them
            ids[i] = mlm_tokenizer.mask_token_id
            for j in range(1, len(target_ids)):
                ids[i+j] = mlm_tokenizer.pad_token_id
            return ids.unsqueeze(0)  # return masked input

    return None  # means no match


def generate_candidates_steered(text, subword, steering_vec, layer=12, alpha=2.0, top_k=10):
    masked_ids = mask_subword(text, subword)
    if masked_ids is None:
        return []  # no match -> safe exit
    
    inputs = {"input_ids": masked_ids.to(device)}

    with torch.no_grad():
        outputs = mlm_model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]  # (1, seq, dim)

    mask_index = (inputs["input_ids"] == mlm_tokenizer.mask_token_id).nonzero(as_tuple=False)[0][1]

    # apply steering
    hidden[0, mask_index, :] += alpha * steering_vec

    # re-run MLM head correctly 
    logits = mlm_model.module.cls(hidden)
    probs = torch.softmax(logits[0, mask_index], dim=0)

    top_ids = torch.topk(probs, top_k).indices.tolist()
    return [mlm_tokenizer.decode([i]).strip() for i in top_ids]


# def generate_candidates_steered(text, word, steering_vec, alpha=2.0, layer=12, top_k=10):
#     encoding = mlm_tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512
#     )

#     tokens = mlm_tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

#     if word not in tokens:
#         return []

#     idx = tokens.index(word)
#     tokens[idx] = mlm_tokenizer.mask_token
#     masked_text = mlm_tokenizer.convert_tokens_to_string(tokens)


#     inputs = mlm_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = mlm_model(**inputs, output_hidden_states=True)

#     mask_index = (inputs["input_ids"] == mlm_tokenizer.mask_token_id).nonzero()[0,1]
#     hidden = outputs.hidden_states[layer][0, mask_index]
#     hidden = hidden + alpha * steering_vec.to(device)

#     logits = mlm_model.module.cls(hidden.unsqueeze(0))
#     probs = torch.softmax(logits, dim=-1)

#     top_tokens = torch.topk(probs, top_k).indices[0].tolist()
#     candidates = [mlm_tokenizer.decode([t]).strip() for t in top_tokens]

#     return candidates


# %%
def semantic_similarity(text1, text2):
    emb1 = sbert.encode(text1, convert_to_tensor=True)
    emb2 = sbert.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

# %%
def perplexity(text):

    model_config = gpt2_model.module.config if hasattr(gpt2_model, "module") else gpt2_model.config
    max_length = model_config.n_positions
    encodings = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len
        lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return float(ppl)

# %%
def edit_distance(a, b):
    return 1 - SequenceMatcher(None, a.split(), b.split()).ratio()

# %%
# def generate_counterfactual_with_lime(
#     text, num_samples=1000, top_k_tokens=5
# ):
#     explainer = LimeTextExplainer(class_names=class_names)
#     exp = explainer.explain_instance(
#         text, predict_proba, num_features=5, labels=[0, 1], num_samples=num_samples
#     )

#     pred_label = np.argmax(predict_proba([text])[0])
#     influential_tokens = [w for w, score in exp.as_list(label=pred_label)]
#     if not influential_tokens:
#         return None

#     for target_word in influential_tokens[:top_k_tokens]:
#         candidates = generate_candidates(text, target_word)
#         for cand in candidates:
#             new_text = text.replace(target_word, cand)
#             new_pred = np.argmax(predict_proba([new_text])[0])
#             if new_pred != pred_label:
#                 sim = semantic_similarity(text, new_text)
#                 flu = perplexity(new_text)
#                 if sim >= sim_threshold and flu <= ppl_threshold:
#                     return {
#                         "original_text": text,
#                         "original_pred": pred_label,
#                         "counterfactual_text": new_text,
#                         "counterfactual_pred": new_pred,
#                         "changed_word": (target_word, cand),
#                         "semantic_similarity": sim,
#                         "perplexity": flu
#                     }
#     return None


# %%
# def truncate_texts(texts, max_len=512):
#     if isinstance(texts, str):
#         texts = [texts]

#     enc = clf_tokenizer(
#         texts,
#         truncation=True,
#         padding=False,
#         max_length=max_len,
#         return_tensors="pt"
#     )
#     truncated_texts = [
#         clf_tokenizer.decode(ids, skip_special_tokens=True)
#         for ids in enc["input_ids"]
#     ]
#     return truncated_texts


# %%
# def generate_counterfactuals_with_shap_batch(texts, max_evals=1000):
#     masker = shap.maskers.Text(tokenizer=clf_tokenizer)
#     model_config = clf_model.module.config if hasattr(clf_model, "module") else clf_model.config
#     labels = [model_config.id2label[i] for i in range(len(model_config.id2label))]

#     explainer = shap.Explainer(predict_proba, masker, output_names=labels)
#     shap_values = explainer(texts, max_evals=max_evals)   #batched call

#     results = []
#     for i, text in enumerate(texts):
#         pred_label = np.argmax(predict_proba([text])[0])
#         token_importances = shap_values.values[i, :, pred_label]
#         tokens = shap_values.data[i]
#         influential_tokens = [t for t, score in zip(tokens, token_importances) if score > 0]

#         candidates = generate_candidates(text, influential_tokens)
#         cf = None
#         for cand, orig_token, repl_token in candidates:
#             new_pred = np.argmax(predict_proba([cand])[0])
#             if new_pred != pred_label:
#                 sim = semantic_similarity(text, cand)
#                 ppl = perplexity(cand)
#                 cf = {
#                     "counterfactual_text": cand,
#                     "changed_word": (orig_token, repl_token),
#                     "semantic_similarity": sim,
#                     "perplexity": ppl
#                 }
#                 break
#         results.append(cf)
#     return results


# %%
def evaluate_counterfactual(cf, orig_text, method, sim_threshold=0.75, ppl_threshold=200):
    if cf is None:
        return {
            "success": -1,
            "method": method,
            "original_text": orig_text,
            "counterfactual_text": None,
            "changed_word": None,
            "semantic_similarity": None,
            "perplexity": None,
            "edit_distance": None,
            "original_embedding": None,
            "counterfactual_embedding": None,
            "mced": None
        }

    orig_emb = sbert.encode(orig_text, convert_to_tensor=False)
    cf_emb = sbert.encode(cf.get("counterfactual_text"), convert_to_tensor=False)
    ed = edit_distance(orig_text, cf.get("counterfactual_text"))
    mced = ed / max(1, len(orig_text.split()))  # normalized edit distance

    if (cf.get("semantic_similarity") >= sim_threshold) and (cf.get("perplexity") <= ppl_threshold):
        success = 1
    else:
        success = 0

    return {
        "success": success,
        "method": method,
        "original_text": cf.get("original_text", orig_text),
        "counterfactual_text": cf.get("counterfactual_text"),
        "changed_word": cf.get("changed_word"),
        "semantic_similarity": cf.get("semantic_similarity"),
        "perplexity": cf.get("perplexity"),
        "edit_distance": ed,
        "original_embedding": orig_emb.tolist(),       
        "counterfactual_embedding": cf_emb.tolist(),
        "mced": mced
    }


# %%
def batch_evaluate(
    sample_size=-1,
    dataset_name="imdb",
    batch_size=64,
    max_len=512,
    lime_num_samples=500,
    shap_max_evals=500
):
    dataset = load_dataset(dataset_name)
    test_data = dataset["test"]


    steering_vector_pos_neg=torch.load("steering_vector_pos_neg.pt").to(device)
    steering_vector_neg_pos=torch.load("steering_vector_neg_pos.pt").to(device)

    if sample_size == -1:
        sample_size = len(test_data)
        examples=list(test_data)
        
        
    else:
        examples = random.sample(list(test_data), sample_size)

    print("Sample size:", sample_size)

    results = []

    masker = shap.maskers.Text(tokenizer=clf_tokenizer)
    model_config = clf_model.module.config if hasattr(clf_model, "module") else clf_model.config
    labels = [model_config.id2label[i] for i in range(len(model_config.id2label))]
    shap_explainer = shap.Explainer(predict_proba, masker, output_names=labels)

    lime_explainer = LimeTextExplainer(class_names=class_names)


    for i in range(0, len(examples), batch_size):

        # last run till- 2496
        # if i<= 2496:
        #     continue

        batch = examples[i:i+batch_size]
        texts = []
        labels=[]
        for ex in batch:
            enc = clf_tokenizer(
                ex["text"],
                truncation=True,
                padding=False,
                max_length=max_len,
                return_tensors="pt"
            )
            truncated_text = clf_tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)
            texts.append(truncated_text)
            labels.append(ex["label"])

        
        # SHAP counterfactuals ####################################
        
        shap_values = shap_explainer(texts, max_evals=shap_max_evals)

        for j, text in enumerate(texts):
            pred_label = np.argmax(predict_proba([text])[0])
            if labels[j] == 1: #pos
                steering_vec = steering_vector_pos_neg
            else:
                steering_vec = steering_vector_neg_pos

            
            vals = shap_values.values[j]
            tokens = shap_values.data[j]

            # handle multi-class output
            if vals.ndim == 1:
                token_importances = vals
            else:
                token_importances = vals[:, pred_label]

            top_indices = np.argsort(np.abs(token_importances))[-5:]
            influential_tokens_shap = [tokens[idx].strip() for idx in top_indices if len(tokens[idx].strip()) > 0]

            #print("influential_tokens_shap",influential_tokens_shap)
            for target_word in influential_tokens_shap:
                #convert to MLM-compatible subword
                subword = mlm_tokenizer.tokenize(target_word)
                # print("subword", subword)
                if len(subword) == 0:
                    continue
                subword = subword[0]
                # candidates = generate_candidates(text, subword)
                candidates = generate_candidates_steered(text, subword, steering_vec)
                
                for cand in candidates:
                    new_text = text.replace(target_word, cand)
                    new_pred = np.argmax(predict_proba([new_text])[0])
                    if new_pred != pred_label:
                        sim = semantic_similarity(text, new_text)
                        flu = perplexity(new_text)
                        cf_shap = {
                            "original_text": text,
                            "original_pred": pred_label,
                            "counterfactual_text": new_text,
                            "counterfactual_pred": new_pred,
                            "changed_word": (target_word, cand),
                            "semantic_similarity": sim,
                            "perplexity": flu
                        }
                        metrics = evaluate_counterfactual(cf_shap, text, method="SHAP")
                        results.append(metrics)
        pd.DataFrame(results).to_csv(cache_dir+f"/intermediate_counterfactuals_{i}_shap.csv", index=False)

        # LIME counterfactuals ####################################
  
        for j, text in enumerate(texts):

            if labels[j] == 1: #pos
                steering_vec = steering_vector_pos_neg
            else:
                steering_vec = steering_vector_neg_pos

            exp = lime_explainer.explain_instance(
                text,
                predict_proba,
                num_features=5,
                labels=[0, 1],
                num_samples=lime_num_samples
            )

            pred_label = np.argmax(predict_proba([text])[0])
            influential_tokens = [w for w, score in sorted(exp.as_list(label=pred_label), key=lambda x: abs(x[1]), reverse=True)][:5]
            if not influential_tokens:
                continue

            #print("influential_tokens_lime",influential_tokens)
            for target_word in influential_tokens:
                subword = mlm_tokenizer.tokenize(target_word)
                # print("subword", subword)
                if len(subword) == 0:
                    continue
                subword = subword[0]
                # candidates = generate_candidates(text, subword)
                candidates = generate_candidates_steered(text, subword, steering_vec)

                for cand in candidates:
                    new_text = text.replace(target_word, cand)
                    new_pred = np.argmax(predict_proba([new_text])[0])
                    if new_pred != pred_label:
                        sim = semantic_similarity(text, new_text)
                        flu = perplexity(new_text)
                        cf_lime = {
                            "original_text": text,
                            "original_pred": pred_label,
                            "counterfactual_text": new_text,
                            "counterfactual_pred": new_pred,
                            "changed_word": (target_word, cand),
                            "semantic_similarity": sim,
                            "perplexity": flu
                        }
                        metrics = evaluate_counterfactual(cf_lime, text, method="LIME")
                        results.append(metrics)

        pd.DataFrame(results).to_csv(cache_dir+f"/intermediate_counterfactuals_{i}_shap_lime.csv", index=False)
        print(f"Processed {min(i + batch_size, sample_size)}/{sample_size} examples")

    df = pd.DataFrame(results)
    df.to_csv("all_counterfactuals.csv", index=False)

    summary = df[df["success"] == 1].groupby("method").mean(numeric_only=True).to_dict()

    return df, summary


# %%


# %%
df, summary = batch_evaluate(sample_size=-1, dataset_name=dataset_name)
summary

# %%


# %%



