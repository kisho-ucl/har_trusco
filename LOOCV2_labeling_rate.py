import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, random_split

from preprocess import load_session_trusco, window_split_trusco
from models import (
	CNN_Encoder,
	SimCLR_new,
	TaskClassifier_DCL,
	TaskClassifier_DCT,
	TaskClassifier_LSTM,
	TaskClassifier_Linear,
	TaskClassifier_Transformer,
)


class IMUDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.long)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, i):
		return self.X[i], self.y[i]


class EarlyStopping:
	def __init__(self, patience=5, delta=0.0):
		self.patience = patience
		self.delta = delta
		self.best_loss = None
		self.counter = 0
		self.early_stop = False
		self.best_state = None
		self.best_epoch = 0

	def __call__(self, epoch, val_loss, model):
		if self.best_loss is None or val_loss < self.best_loss - self.delta:
			self.best_loss = val_loss
			self.best_state = deepcopy(model.state_dict())
			self.best_epoch = epoch
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True


@dataclass
class ExperimentConfig:
	model_type: str
	pretrain_type: str
	frozen: bool
	lr_type: str
	base_dir: str
	save_dir: str
	seed: int = 0


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

import numpy as np

# 各クラスから rate% を抽出
def sample_train_windows_by_rate(X, y, rate, seed):
    if rate >= 1.0:
        return X, y
    if rate <= 0.0:
        return X[:0], y[:0]

    rng = np.random.default_rng(seed)
    selected_indices = []

    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        n = len(class_indices)
        if n == 0:
            continue

        # 固定割合で抽出（論文的に明確）
        sample_size = int(np.floor(rate * n))

        if sample_size == 0:
            continue

        chosen = rng.choice(class_indices, size=sample_size, replace=False)
        selected_indices.append(chosen)

    if len(selected_indices) == 0:
        return X[:0], y[:0]

    # 時系列順を保つためにソート
    selected_indices = np.sort(np.concatenate(selected_indices))

    return X[selected_indices], y[selected_indices]

def build_loaders_for_subject(
	test_user,
	YDs,
	IDs,
	Hours,
	all_entries,
	config,
	labeling_rate=1.0,
	sampling_seed=0,
):
	if config.model_type == "cnn-linear":
		window_size = 400 * 1
		train_stride = 200 * 1
		test_stride = 400 * 1
	elif config.model_type in {"multi-trans", "dct", "dcl"}:
		window_size = 400 * 10
		train_stride = 400 * 1
		test_stride = 400 * 1
	else:
		raise ValueError(f"Unknown model_type: {config.model_type}")

	X_train, y_train = [], []
	X_test, y_test = [], []
	train_subject_data = {}

	for Year, Date in YDs:
		for ID in IDs:
			for h in Hours:
				if h not in all_entries or ID not in all_entries[h]:
					continue

				imu, labels = load_session_trusco(ID, Year, Date, h)
				if imu is None:
					continue

				if ID == test_user:
					X, y = window_split_trusco(
						imu,
						labels,
						window_size=window_size,
						stride=test_stride,
					)
					valid = y != -1
					X_test.append(X[valid])
					y_test.append(y[valid])
				else:
					X, y = window_split_trusco(
						imu,
						labels,
						window_size=window_size,
						stride=train_stride,
					)
					valid = y != -1
					X = X[valid]
					y = y[valid]

					if ID not in train_subject_data:
						train_subject_data[ID] = {"X": [], "y": []}

					train_subject_data[ID]["X"].append(X)
					train_subject_data[ID]["y"].append(y)

	for ID in sorted(train_subject_data.keys()):
		if len(train_subject_data[ID]["X"]) == 0:
			continue

		X_id = np.concatenate(train_subject_data[ID]["X"])
		y_id = np.concatenate(train_subject_data[ID]["y"])
		X_id, y_id = sample_train_windows_by_rate(X_id, y_id, labeling_rate, sampling_seed)

		if len(X_id) == 0:
			continue

		X_train.append(X_id)
		y_train.append(y_id)

	if len(X_train) == 0:
		raise ValueError(f"No train samples available for test_user={test_user}, rate={labeling_rate}")
	if len(X_test) == 0:
		raise ValueError(f"No test samples available for test_user={test_user}")

	X_train = np.concatenate(X_train)
	y_train = np.concatenate(y_train)
	X_test = np.concatenate(X_test)
	y_test = np.concatenate(y_test)

	train_dataset = IMUDataset(X_train, y_train)
	test_dataset = IMUDataset(X_test, y_test)

	train_len = int(len(train_dataset) * (1 - 0.1))
	valid_len = len(train_dataset) - train_len

	train_ds, valid_ds = random_split(
		train_dataset,
		[train_len, valid_len],
		generator=torch.Generator().manual_seed(config.seed),
	)

	train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
	valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

	print(
		f"[test_user={test_user} | rate={labeling_rate:.2f}] "
		f"Train={len(train_ds)}, Valid={len(valid_ds)}, Test={len(test_dataset)}"
	)

	return train_loader, valid_loader, test_loader


def compute_class_weights(train_loader, num_classes):
	counts = torch.zeros(num_classes)

	for _, labels in train_loader:
		for c in range(num_classes):
			counts[c] += (labels == c).sum()

	total = counts.sum()
	counts[counts == 0] = 1
	weights = total / (num_classes * counts)
	return weights


def train_one_epoch(classifier, loader, optimizer, criterion, device):
	classifier.train()
	total_loss = 0
	total_n = 0

	for signals, labels in loader:
		signals = signals.to(device)
		labels = labels.to(device)

		logits = classifier(signals)
		loss = criterion(logits, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * len(labels)
		total_n += len(labels)

	return total_loss / total_n


def evaluate(classifier, loader, device, criterion=None):
	classifier.eval()
	total_loss = 0
	total_n = 0
	Y, P = [], []

	with torch.no_grad():
		for signals, labels in loader:
			signals = signals.to(device)
			labels = labels.to(device)

			logits = classifier(signals)
			preds = logits.argmax(dim=1)

			Y.extend(labels.cpu().numpy())
			P.extend(preds.cpu().numpy())

			if criterion is not None:
				loss = criterion(logits, labels)
				total_loss += loss.item() * len(labels)
				total_n += len(labels)

	if criterion is None:
		return Y, P, None
	return Y, P, total_loss / total_n


def record_results(test_user, y_true, y_pred, results_dir="results_loso"):
	os.makedirs(results_dir, exist_ok=True)
	save_path = os.path.join(results_dir, f"{test_user}.json")

	result = {
		"user": test_user,
		"y_true": list(map(int, y_true)),
		"y_pred": list(map(int, y_pred)),
	}
	with open(save_path, "w") as f:
		json.dump(result, f, indent=2)

	print(f"Saved: {save_path}")


def build_model(config):
	if config.pretrain_type == "None":
		encoder = CNN_Encoder(
			input_dim=6,
			seq_len=400,
			hidden_dim=64,
			feature_dim=128,
			dropout=0.3,
		)
		print("Created new encoder.")
	else:
		sim = SimCLR_new(
			input_dim=6,
			seq_len=400,
			hidden_dim=64,
			feature_dim=128,
			projection_dim=64,
			feedforward_dim=256,
			num_layers=2,
			num_heads=4,
		)

		if config.pretrain_type == "csshar":
			path = "/home/kisho_ucl/kisho_ws/deep_HAR/trusco/experiments/model/1214/imu_encoder_100.pth"
		elif config.pretrain_type == "mysimclr":
			path = "/home/kisho_ucl/kisho_ws/deep_HAR/trusco/experiments/model/0413/imu_encoder_100.pth"
		elif config.pretrain_type == "mysimclr2":
			path = "/home/kisho_ucl/kisho_ws/deep_HAR/trusco/experiments/model/1212/imu_encoder_100.pth"
		else:
			raise ValueError(f"Unknown pretrain_type: {config.pretrain_type}")

		sim.load_state_dict(torch.load(path, map_location="cpu"))
		encoder = sim.encoder
		print(f"Loaded pretrained encoder: {config.pretrain_type}")

	if config.frozen:
		for p in encoder.parameters():
			p.requires_grad = False
		print("Encoder is frozen.")
	else:
		print("Encoder is trainable.")

	print(f"Building model: {config.model_type}")
	if config.model_type == "cnn-linear":
		return TaskClassifier_Linear(
			encoder=encoder,
			feature_dim=128,
			hidden_dim=64,
			num_classes=3,
			dropout=0.1,
			freeze_encoder=config.frozen,
		)

	if config.model_type == "multi-trans":
		return TaskClassifier_Transformer(
			encoder=encoder,
			hidden_dim=128,
			window_len=400,
			overlap=0.5,
			num_heads=4,
			num_layers=3,
			num_classes=3,
			freeze_encoder=config.frozen,
		)

	if config.model_type == "dcl":
		return TaskClassifier_DCL(
			input_dim=6,
			hidden_dim=64,
			feature_dim=128,
			lstm_hidden=128,
			num_layers=2,
			num_classes=3,
			dropout=0.1,
		)

	if config.model_type == "dct":
		return TaskClassifier_DCT(
			input_dim=6,
			hidden_dim=64,
			feature_dim=128,
			num_heads=4,
			num_layers=3,
			num_classes=3,
			dropout=0.1,
		)

	raise ValueError(f"Unknown model_type: {config.model_type}")


def build_optimizer(model, config):
	if config.frozen:
		print("Optimizer: frozen encoder -> uniform LR 1e-3")
		params = [p for p in model.parameters() if p.requires_grad]
		return torch.optim.Adam(params, lr=1e-3)

	if config.lr_type == "none":
		print("Optimizer: frozen encoder -> uniform LR 1e-3")
		params = [p for p in model.parameters() if p.requires_grad]
		return torch.optim.Adam(params, lr=1e-3)

	if config.lr_type == "uniform":
		print("Optimizer: uniform lr=1e-3")
		return torch.optim.Adam(model.parameters(), lr=1e-3)

	if config.lr_type == "layerwise":
		encoder_lr = 1e-4
		head_lr = 1e-3

		if isinstance(model, TaskClassifier_Transformer):
			print("Optimizer: layerwise (Transformer)")
			return torch.optim.Adam(
				[
					{"params": model.encoder.parameters(), "lr": encoder_lr},
					{"params": model.transformer.parameters(), "lr": head_lr},
					{"params": model.classifier.parameters(), "lr": head_lr},
				]
			)

		if isinstance(model, TaskClassifier_LSTM):
			print("Optimizer: layerwise (LSTM)")
			return torch.optim.Adam(
				[
					{"params": model.encoder.parameters(), "lr": encoder_lr},
					{"params": model.lstm.parameters(), "lr": head_lr},
					{"params": model.classifier.parameters(), "lr": head_lr},
				]
			)

		if isinstance(model, TaskClassifier_Linear):
			print("Optimizer: layerwise (Linear)")
			return torch.optim.Adam(
				[
					{"params": model.encoder.parameters(), "lr": encoder_lr},
					{"params": model.classifier.parameters(), "lr": head_lr},
				]
			)

		raise ValueError("Unknown model architecture for layerwise LR")

	raise ValueError(f"Unknown lr_type {config.lr_type}")


def run_loso_one(
	train_loader,
	valid_loader,
	test_loader,
	weight_tensor,
	test_user="UNKNOWN",
	max_epochs=30,
	patience=5,
	config=None,
	device=None,
):
	set_seed(config.seed)

	classifier = build_model(config).to(device)
	optimizer = build_optimizer(classifier, config)

	weight_tensor = weight_tensor.to(device)
	criterion = nn.CrossEntropyLoss(weight=weight_tensor)

	early_stopper = EarlyStopping(patience=patience)

	for epoch in range(1, max_epochs + 1):
		train_loss = train_one_epoch(classifier, train_loader, optimizer, criterion, device)
		_, _, valid_loss = evaluate(classifier, valid_loader, device, criterion)
		Y_tmp, P_tmp, test_loss = evaluate(classifier, test_loader, device, criterion)
		f1_tmp = f1_score(Y_tmp, P_tmp, average="weighted")

		print(
			f"[Epoch {epoch}/{max_epochs}] "
			f"Train={train_loss:.4f} | "
			f"Valid={valid_loss:.4f} | "
			f"Test={test_loss:.4f} | F1={f1_tmp:.4f}"
		)

		early_stopper(epoch, valid_loss, classifier)
		if early_stopper.early_stop:
			print("Early stopping.")
			break

	classifier.load_state_dict(early_stopper.best_state)

	Y, P, _ = evaluate(classifier, test_loader, device, criterion)
	final_f1 = f1_score(Y, P, average="weighted")

	print(f"[{test_user}] Final F1 = {final_f1:.4f} | BestEpoch={early_stopper.best_epoch}")

	record_results(test_user, Y, P, results_dir=config.save_dir)
	return final_f1, Y, P, classifier


def main():
	device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	YDs = [[2024, 1003]]
	Hours = range(7, 18, 1)
	IDs = range(0, 39, 1)

	all_entries = {
		8: [1],
		9: [21],
		10: [13],
		11: [2, 6, 9, 23, 27, 36, 16],
	}

	configs = [

	ExperimentConfig("cnn-linear", "None", False, "uniform", "T1_cnn", ""),
    ExperimentConfig("cnn-linear", "mysimclr2", True, "uniform", "T3_precnn_adj", ""),

    ExperimentConfig("dcl", "None", False, "uniform", "T4_dcl", ""),


    ExperimentConfig("multi-trans", "None", False, "uniform", "T5_cnn-transformer", ""),
    ExperimentConfig("multi-trans", "mysimclr2", True, "uniform", "T7_precnn-transformer_adj", ""),
	]

	seeds = [0, 2, 3, 4]
	#labeling_rates = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
	labeling_rates = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
	results_root = Path(__file__).resolve().parent / "results_loso_rate"
	num_classes = 3

	all_records = []

	for seed in seeds:
		print(f"\n=========== SEED {seed} ===========")
		seed_records = []
		sampling_seed = seed

		for rate in labeling_rates:
			print(f"\n----- Labeling rate: {rate:.2f} -----")
			rate_records = []

			for cfg in configs:
				cfg.seed = seed
				model_name = os.path.basename(cfg.base_dir.rstrip("/"))

				print("===================================")
				print(f" Variant: {model_name}")
				print("===================================")

				for test_hour, subjects in all_entries.items():
					for test_subject in subjects:
						set_seed(seed)

						train_loader, valid_loader, test_loader = build_loaders_for_subject(
							test_user=test_subject,
							YDs=YDs,
							IDs=IDs,
							Hours=Hours,
							all_entries=all_entries,
							labeling_rate=rate,
							sampling_seed=sampling_seed,
							config=cfg,
						)

						weight_tensor = compute_class_weights(train_loader, num_classes)
						print("Class weights:", weight_tensor.numpy())

						save_dir = (
							results_root
							/ f"seed{seed}"
							/ f"rate{int(rate * 100):03d}"
							/ model_name
						)
						os.makedirs(save_dir, exist_ok=True)
						cfg.save_dir = str(save_dir)

						f1, Y, P, classifier = run_loso_one(
							train_loader,
							valid_loader,
							test_loader,
							weight_tensor,
							test_user=test_subject,
							max_epochs=30,
							patience=5,
							config=cfg,
							device=device,
						)

						record = {
							"seed": seed,
							"rate": rate,
							"model": model_name,
							"test_hour": test_hour,
							"test_subject": test_subject,
							"f1": float(f1),
						}
						rate_records.append(record)
						seed_records.append(record)
						all_records.append(record)
						print(f"Seed={seed}, Rate={rate:.2f}, User={test_subject}, F1={f1:.4f}")

			if len(rate_records) == 0:
				continue

			rate_df = pd.DataFrame(rate_records)
			rate_summary_df = (
				rate_df.groupby("model", as_index=False)
				.agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"), n_folds=("f1", "size"))
				.sort_values("model")
			)
			rate_summary_df["std_f1"] = rate_summary_df["std_f1"].fillna(0.0)

			print(rate_summary_df)

			rate_dir = results_root / f"seed{seed}" / f"rate{int(rate * 100):03d}"
			os.makedirs(rate_dir, exist_ok=True)
			rate_df.to_csv(rate_dir / "fold_results.csv", index=False)
			rate_summary_df.to_csv(rate_dir / "summary.csv", index=False)

		if len(seed_records) > 0:
			all_df = pd.DataFrame(seed_records)
			seed_summary_df = (
				all_df.groupby(["rate", "model"], as_index=False)
				.agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"), n_folds=("f1", "size"))
				.sort_values(["rate", "model"])
			)
			seed_summary_df["std_f1"] = seed_summary_df["std_f1"].fillna(0.0)
			seed_dir = results_root / f"seed{seed}"
			os.makedirs(seed_dir, exist_ok=True)
			all_df.to_csv(seed_dir / "all_fold_results.csv", index=False)
			seed_summary_df.to_csv(seed_dir / "seed_summary.csv", index=False)
			print(seed_summary_df)

	if len(all_records) > 0:
		all_records_df = pd.DataFrame(all_records)
		all_records_df.to_csv(results_root / "all_records.csv", index=False)
		print(f"Saved all records: {results_root / 'all_records.csv'}")


if __name__ == "__main__":
	main()
