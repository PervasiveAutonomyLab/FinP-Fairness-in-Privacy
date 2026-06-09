import copy
import math
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from models.Update import DatasetSplit


def discover_attack_weight_keys(model):
    keys = []
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            keys.append(f"{module_name}.weight")
    return keys


def _femnist_image_to_pil_l(img):
    if hasattr(img, "convert"):
        return img.convert("L")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(BytesIO(img["bytes"])).convert("L")
    raise TypeError(f"Unsupported image type for FEMNIST record: {type(img)}")


def _compute_single_record_gradient(model, x, y, device, target_weight_keys):
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    x_b = x.unsqueeze(0).to(device)
    y_b = torch.tensor([int(y)], dtype=torch.long, device=device)
    logits = model(x_b)
    loss = criterion(logits, y_b)
    loss.backward()

    grad_dict = {}
    for name, param in model.named_parameters():
        if name in target_weight_keys and param.grad is not None and torch.is_floating_point(param.grad):
            grad_dict[name] = param.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    del x_b, y_b, logits, loss
    return grad_dict


def _train_mock_local_delta(args, global_model, dataset_train, subset_indices, global_state):
    local_model = copy.deepcopy(global_model).to(args.device)
    local_model.train()
    loader = DataLoader(DatasetSplit(dataset_train, subset_indices), batch_size=args.local_bs, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)

    for _ in range(args.local_ep):
        for images, labels in loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = local_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    local_state = local_model.state_dict()
    delta_mock = {}
    with torch.no_grad():
        for key, value in local_state.items():
            if key in global_state and value.is_floating_point():
                delta_mock[key] = (value - global_state[key]).detach().clone()

    del loader, criterion, optimizer, local_model, local_state
    return delta_mock


def _layerwise_cosine_features(grad_dict, delta_dict, target_weight_keys):
    features = []
    with torch.no_grad():
        for key in target_weight_keys:
            grad = grad_dict.get(key, None)
            delta = delta_dict.get(key, None)
            if grad is None or delta is None:
                continue
            if not (torch.is_floating_point(grad) and torch.is_floating_point(delta)):
                continue
            sim = torch.nn.functional.cosine_similarity(
                grad.view(-1), delta.view(-1), dim=0, eps=1e-12
            )
            features.append(sim.item())
    return features


def _sample_from_femnist_writer(hf_split, row_idx, transform):
    row = hf_split[row_idx]
    image = transform(_femnist_image_to_pil_l(row["image"]))
    label = int(row["character"])
    return image, label


def prepare_mia_static_data(args, dataset_train, dict_party_user, rng):
    if not hasattr(dataset_train, "hf") or not hasattr(dataset_train, "row_indices"):
        raise ValueError("--mia currently expects FEMNISTHFDataset-backed training data.")
    if args.num_users < 2:
        raise ValueError("--mia requires at least 2 users (dummy + victim clients).")

    hf_train = dataset_train.hf
    hf_writer_ids = hf_train["writer_id"]
    row_indices = dataset_train.row_indices

    client_writer_map = {}
    for client_id in range(args.num_users):
        if len(dict_party_user[client_id]) == 0:
            raise ValueError(f"--mia requires non-empty data for client {client_id}.")
        local_pos = dict_party_user[client_id][0]
        hf_row = row_indices[local_pos]
        client_writer_map[client_id] = hf_writer_ids[hf_row]

    writer_to_indices = defaultdict(list)
    for idx, wid in enumerate(hf_writer_ids):
        writer_to_indices[wid].append(idx)

    in_fl_writers = set(client_writer_map.values())
    unseen_writers = [wid for wid in writer_to_indices.keys() if wid not in in_fl_writers]
    if len(unseen_writers) < 2:
        raise ValueError("--mia requires at least two unseen writers for Writer A and Writer B.")

    writer_a, writer_b = rng.choice(np.array(unseen_writers, dtype=object), size=2, replace=False)
    writer_a_rows = list(writer_to_indices[writer_a])
    writer_b_rows = list(writer_to_indices[writer_b])

    victim_ids = [cid for cid in range(1, args.num_users)]
    mia_testing_dict = {}
    for victim_id in victim_ids:
        victim_pool = list(dict_party_user[victim_id])
        victim_member_positions = rng.choice(
            victim_pool, size=50, replace=(len(victim_pool) < 50)
        ).tolist()
        writer_b_samples = rng.choice(
            writer_b_rows, size=50, replace=(len(writer_b_rows) < 50)
        ).tolist()

        static_test_records = []
        for pos in victim_member_positions:
            x_m, y_m = dataset_train[pos]
            static_test_records.append((x_m.detach().cpu(), int(y_m), 1))
        for row_idx in writer_b_samples:
            x_nm, y_nm = _sample_from_femnist_writer(hf_train, row_idx, dataset_train.transform)
            static_test_records.append((x_nm.detach().cpu(), int(y_nm), 0))
        rng.shuffle(static_test_records)
        mia_testing_dict[victim_id] = static_test_records

    return {
        "dummy_client_id": 0,
        "victim_ids": victim_ids,
        "writer_a_id": writer_a,
        "writer_b_id": writer_b,
        "writer_a_rows": writer_a_rows,
        "mia_testing_dict": mia_testing_dict,
    }


def run_whitebox_mia_round(
    args,
    rng,
    global_model,
    global_state,
    dataset_train,
    dict_party_user,
    client_to_weights,
    mia_ctx,
    target_weight_keys,
):
    if len(target_weight_keys) == 0:
        print("MIA skipped this round: no eligible weight layers for cosine features.")
        return {}

    grad_model = copy.deepcopy(global_model).to(args.device)
    grad_model.train()
    client0_pool = list(dict_party_user[0])

    rf_features, rf_labels = [], []
    for _ in range(30):
        x_m_pos = int(rng.choice(client0_pool))
        x_m, y_m = dataset_train[x_m_pos]

        x_nm_row = int(rng.choice(mia_ctx["writer_a_rows"]))
        x_nm, y_nm = _sample_from_femnist_writer(dataset_train.hf, x_nm_row, dataset_train.transform)

        g_m = _compute_single_record_gradient(grad_model, x_m, y_m, args.device, target_weight_keys)
        g_nm = _compute_single_record_gradient(grad_model, x_nm, y_nm, args.device, target_weight_keys)

        subset_size = max(1, int(math.ceil(0.8 * len(client0_pool))))
        if subset_size >= len(client0_pool):
            mock_subset = list(client0_pool)
        else:
            remaining = [idx for idx in client0_pool if idx != x_m_pos]
            picked = rng.choice(remaining, size=subset_size - 1, replace=False).tolist()
            mock_subset = picked + [x_m_pos]

        delta_w_mock = _train_mock_local_delta(
            args=args,
            global_model=global_model,
            dataset_train=dataset_train,
            subset_indices=mock_subset,
            global_state=global_state,
        )

        member_vec = _layerwise_cosine_features(g_m, delta_w_mock, target_weight_keys)
        nonmember_vec = _layerwise_cosine_features(g_nm, delta_w_mock, target_weight_keys)
        if len(member_vec) == len(target_weight_keys):
            rf_features.append(member_vec)
            rf_labels.append(1)
        if len(nonmember_vec) == len(target_weight_keys):
            rf_features.append(nonmember_vec)
            rf_labels.append(0)

        del g_m, g_nm, delta_w_mock, member_vec, nonmember_vec

    if len(rf_features) < 10 or len(set(rf_labels)) < 2:
        print("MIA skipped this round: insufficient RandomForest training rows.")
        del grad_model
        return {}

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=args.manualseed,
        n_jobs=-1,
    )
    rf_clf.fit(np.array(rf_features), np.array(rf_labels))

    round_metrics = {}
    for victim_id in mia_ctx["victim_ids"]:
        if victim_id not in client_to_weights:
            continue
        victim_state = client_to_weights[victim_id]

        delta_victim = {}
        with torch.no_grad():
            for key in target_weight_keys:
                if key in victim_state and key in global_state and torch.is_floating_point(victim_state[key]):
                    delta_victim[key] = (victim_state[key] - global_state[key]).detach().clone()

        x_eval, y_true = [], []
        for x_test, y_test_cls, mia_label in mia_ctx["mia_testing_dict"][victim_id]:
            g_test = _compute_single_record_gradient(
                grad_model, x_test, y_test_cls, args.device, target_weight_keys
            )
            feat = _layerwise_cosine_features(g_test, delta_victim, target_weight_keys)
            if len(feat) == len(target_weight_keys):
                x_eval.append(feat)
                y_true.append(mia_label)
            del g_test, feat

        if len(x_eval) == 0:
            continue
        x_eval_np = np.array(x_eval)
        y_true_np = np.array(y_true)
        y_pred = rf_clf.predict(x_eval_np)
        y_prob_all = rf_clf.predict_proba(x_eval_np)
        if y_prob_all.shape[1] >= 2:
            y_score = y_prob_all[:, 1]
        else:
            y_score = y_pred.astype(float)

        round_metrics[victim_id] = {
            "accuracy": float(accuracy_score(y_true_np, y_pred)),
            "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true_np, y_score)),
        }

        del delta_victim, x_eval, y_true, x_eval_np, y_true_np, y_pred, y_prob_all, y_score

    del rf_clf, grad_model
    return round_metrics
