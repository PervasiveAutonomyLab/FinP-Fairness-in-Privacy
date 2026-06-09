"""
Federated FEMNIST: one client = one writer_id (same split idea as FEMNIST/femnist_train.py).
Train/test per writer: (1 - test_fraction) / test_fraction (default 80% / 20%).
"""
from collections import Counter, defaultdict
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset

try:
    from datasets import load_dataset
except ImportError as e:
    load_dataset = None  # type: ignore


def _image_to_pil_l(img):
    if hasattr(img, "convert"):
        return img.convert("L")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(BytesIO(img["bytes"])).convert("L")
    raise TypeError(f"Unsupported image type: {type(img)}")


class FEMNISTHFDataset(Dataset):
    """Rows from a HuggingFace split indexed by integer positions into that split."""

    def __init__(self, hf_split, row_indices, transform):
        self.hf = hf_split
        self.row_indices = list(row_indices)
        self.transform = transform

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, i):
        row = self.hf[self.row_indices[i]]
        x = self.transform(_image_to_pil_l(row["image"]))
        y = int(row["character"])
        return x, torch.tensor(y, dtype=torch.long)


def _empty_placeholder_batch():
    """Unused in main_fed; keeps get_dataset return shape valid."""
    return TensorDataset(
        torch.zeros(0, 1, 28, 28, dtype=torch.float32),
        torch.zeros(0, dtype=torch.long),
    )


def print_federated_femnist_stats(stats):
    """Same header style as FEMNIST/femnist_train.py; adds train/test/sample split breakdown."""
    print("\n=== Federated FEMNIST split stats ===")
    g = stats.get("global", {})
    print(
        "Global: "
        f"dataset_train_len={g.get('dataset_train_len')}, "
        f"dataset_test_len={g.get('dataset_test_len')}, "
        f"num_users={g.get('num_users')}, "
        f"num_samples_cap={g.get('num_samples_cap')}, "
        f"test_fraction={g.get('test_fraction')}, "
        f"femnist_remove={g.get('femnist_remove', 0)}"
    )
    print("Selected writer_ids:", stats["selected_writer_ids"])
    for c in stats["clients"]:
        print(
            f"\n  client_id={c['client_id']} writer_id={c['writer_id']}: "
            f"n_total_writer_rows={c.get('n_total_writer_rows', c.get('n_total'))}, "
            f"n_after_class_exclusion={c.get('n_after_class_exclusion', c.get('n_total'))}, "
            f"distinct_characters={c['n_distinct_characters']}, "
            f"n_train={c['n_train']}, n_test={c['n_test']}, n_sample={c['n_sample']}"
        )
        ex = c.get("excluded_character_classes") or []
        if ex:
            print(f"    excluded_character_classes ({len(ex)}): {ex}")
        print("    per-character counts (pool after exclusion, before train/test):", c["character_counts"])
        print("    train split - counts:", c["character_counts_train"])
        print("    test split  - counts:", c["character_counts_test"])
        print("    MIA sample  - counts:", c["character_counts_sample"])
    print("=== end stats ===\n")


def prepare_femnist_federated(args):
    if load_dataset is None:
        raise ImportError(
            "FEMNIST requires the `datasets` package. Install with: pip install datasets"
        )

    from torchvision import transforms

    rng = np.random.default_rng(args.manualseed)
    hf_name = getattr(args, "femnist_hf_name", "flwrlabs/femnist")
    test_fraction = float(getattr(args, "femnist_test_fraction", 0.2))

    hf_train = load_dataset(hf_name, split="train")
    character_col = hf_train["character"]

    femnist_remove = int(getattr(args, "femnist_remove", 0))
    num_character_classes = int(getattr(args, "num_classes", 62))
    if femnist_remove < 0:
        raise ValueError("--femnist_remove must be >= 0")
    if femnist_remove > num_character_classes:
        raise ValueError(
            f"--femnist_remove ({femnist_remove}) cannot exceed num_classes ({num_character_classes})"
        )

    writer_to_indices = defaultdict(list)
    writer_ids = hf_train["writer_id"]
    for i, wid in enumerate(writer_ids):
        writer_to_indices[wid].append(i)

    all_writers = list(writer_to_indices.keys())
    if len(all_writers) < args.num_users:
        raise ValueError(
            f"Need {args.num_users} distinct writer_ids but split only has {len(all_writers)}."
        )

    chosen_writers = list(
        rng.choice(np.array(all_writers, dtype=object), size=args.num_users, replace=False)
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    global_train_rows = []
    dict_party_user = {}
    dict_sample_user = {}
    global_test_rows = []
    stats = {"selected_writer_ids": list(chosen_writers), "clients": []}

    for client_id, wid in enumerate(chosen_writers):
        all_row_idxs = list(writer_to_indices[wid])
        n_writer_full = len(all_row_idxs)

        if femnist_remove > 0:
            excluded_labels = set(
                rng.choice(
                    num_character_classes,
                    size=femnist_remove,
                    replace=False,
                ).tolist()
            )
            idxs = [
                i
                for i in all_row_idxs
                if int(character_col[i]) not in excluded_labels
            ]
        else:
            excluded_labels = set()
            idxs = all_row_idxs

        if len(idxs) == 0:
            raise ValueError(
                f"client_id={client_id} writer_id={wid}: after excluding classes "
                f"{sorted(excluded_labels)}, no samples remain. Lower --femnist_remove or change seed."
            )

        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_fraction * n))
        if n_test == 0 and n > 1:
            n_test = 1
        if n_test >= n and n > 1:
            n_test = n - 1

        if n <= 1:
            train_idx, test_idx = idxs, []
        else:
            test_idx = idxs[:n_test]
            train_idx = idxs[n_test:]

        offset = len(global_train_rows)
        global_train_rows.extend(train_idx)
        dict_party_user[client_id] = list(range(offset, offset + len(train_idx)))
        global_test_rows.extend(test_idx)

        n_take = min(args.num_samples, len(dict_party_user[client_id]))
        if n_take == 0:
            dict_sample_user[client_id] = []
        else:
            rel = rng.choice(len(dict_party_user[client_id]), size=n_take, replace=False)
            dict_sample_user[client_id] = [
                dict_party_user[client_id][int(j)] for j in rel
            ]

        cnt_pool = Counter(int(character_col[i]) for i in idxs)
        cnt_train = Counter(character_col[i] for i in train_idx)
        cnt_test = Counter(character_col[i] for i in test_idx)
        cnt_sample = Counter()
        for pos in dict_sample_user[client_id]:
            hf_row = global_train_rows[pos]
            cnt_sample[character_col[hf_row]] += 1

        stats["clients"].append(
            {
                "writer_id": wid,
                "client_id": client_id,
                "n_total_writer_rows": n_writer_full,
                "excluded_character_classes": sorted(excluded_labels),
                "n_after_class_exclusion": len(idxs),
                "n_total": n_writer_full,
                "n_distinct_characters": len(cnt_pool),
                "character_counts": dict(sorted(cnt_pool.items())),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_sample": len(dict_sample_user[client_id]),
                "character_counts_train": dict(sorted(cnt_train.items())),
                "character_counts_test": dict(sorted(cnt_test.items())),
                "character_counts_sample": dict(sorted(cnt_sample.items())),
            }
        )

    stats["global"] = {
        "dataset_train_len": len(global_train_rows),
        "dataset_test_len": len(global_test_rows),
        "num_users": args.num_users,
        "num_samples_cap": args.num_samples,
        "test_fraction": test_fraction,
        "femnist_remove": femnist_remove,
    }
    print_federated_femnist_stats(stats)

    dataset_train = FEMNISTHFDataset(hf_train, global_train_rows, transform)
    dataset_test = FEMNISTHFDataset(hf_train, global_test_rows, transform)

    empty = _empty_placeholder_batch()
    test_subsets = [empty for _ in range(args.num_users)]
    class_test = [empty for _ in range(args.num_classes)]

    return (
        dataset_train,
        dataset_test,
        dict_party_user,
        dict_sample_user,
        test_subsets,
        class_test,
    )
