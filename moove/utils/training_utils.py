# utils/training_utils.py
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from moove.models.CNN import CNN
from moove.models.ConvMLP import ConvMLP
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

from PyQt6.QtWidgets import QLabel, QApplication, QDialog, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from moove.qt_helpers import show_info, show_confirm_action_window


def _torch_major_minor():
    """Return torch major/minor as tuple, e.g. (2, 6)."""
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def _load_checkpoint_with_version_fallback(model_path):
    """Load checkpoint with version-aware fallback for torch >= 2.6."""
    if _torch_major_minor() >= (2, 6):
        try:
            return torch.load(model_path)
        except Exception:
            # Trusted local checkpoint created by this app: use legacy object load.
            return torch.load(model_path, weights_only=False)
    return torch.load(model_path)


def _show_status(window, text):
    """Show a status message on the training/dialog window."""
    if hasattr(window, 'status_label'):
        window.status_label.setText(text)
        window.status_label.show()
        QApplication.processEvents()


def _hide_status(window):
    if hasattr(window, 'status_label'):
        window.status_label.hide()
        QApplication.processEvents()


def _set_training_running(app_state, running):
    """Store training state on the training dialog instance."""
    win = getattr(app_state, 'training_window', None)
    if win is not None:
        win._training_running = bool(running)
        if running:
            win._training_cancel_requested = False


def _training_cancel_requested(app_state):
    """Return True if user requested to cancel via dialog close."""
    win = getattr(app_state, 'training_window', None)
    return bool(win is not None and getattr(win, '_training_cancel_requested', False))


def _ask_user_for_small_dataset(parent, n_files):
    """Show a blocking dialog asking if user wants to continue with few files."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Small Dataset detected")
    dlg.setModal(True)
    layout = QVBoxLayout(dlg)

    msg = QLabel(
        f"Number of files given is very small! (n = {n_files})\n"
        "Are you training on one file containing multiple bouts?\n"
        "If not, please add more song files for training.\n"
        "Using data from multiple files is recommended.\n\n"
        "Do you want to continue?"
    )
    msg.setWordWrap(True)
    layout.addWidget(msg)

    btn_row = QHBoxLayout()
    btn_continue = QPushButton("Continue with few files")
    btn_cancel = QPushButton("Cancel")
    btn_row.addWidget(btn_continue)
    btn_row.addWidget(btn_cancel)
    layout.addLayout(btn_row)

    result = {'answer': None}
    btn_continue.clicked.connect(lambda: (result.update({'answer': 'continue'}), dlg.accept()))
    btn_cancel.clicked.connect(lambda: (result.update({'answer': 'cancel'}), dlg.reject()))

    dlg.exec()
    return result.get('answer')


def start_segmentation_training(parent, app_state, training_dataset_name):
    """Start training of segmentation model using provided dataset and parameters."""

    if training_dataset_name == "Select Training Dataset":
        show_info(parent, "Error", "Selected training dataset not valid! Perhaps you forgot to pick a dataset?")
        return

    _show_status(app_state.training_window, "Checking files...")

    imbalance_strategy = app_state.train_segmentation_params['imbalance_strategy'].get()
    epochs = int(app_state.train_segmentation_params['epochs'].get())
    batch_size = int(app_state.train_segmentation_params['batch_size'].get())
    learning_rate = float(app_state.train_segmentation_params['learning_rate'].get())
    early_stopping_patience = int(
        app_state.train_segmentation_params['early_stopping_patience'].get()
    )

    dataset_path = os.path.join(app_state.config['global_dir'], 'training_data', f'{training_dataset_name}')

    with open(dataset_path, 'rb') as f:
        data_dict = pickle.load(f)

    features = np.array(data_dict['features'])
    metadata = data_dict['metadata']
    num_segs = data_dict['syllables']
    del data_dict

    if features.ndim == 2:
        file_indices = np.unique(features[:, 0])
    else:
        _hide_status(app_state.training_window)
        show_info(parent, "Error", "Given dataset is empty!")
        return

    if num_segs <= 7:
        _hide_status(app_state.training_window)
        show_info(parent, "Error", f"Not enough segments given (n = {num_segs}), "
                                   f"need at least 7 to train a network!\n You might want to adjust the threshold.")
        return

    if len(file_indices) >= 7:
        def filter_data_by_files(data, file_set):
            return data[np.isin(data[:, 0], file_set)]

        train_files, temp_files = train_test_split(file_indices, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        if not show_confirm_action_window(parent, "Info", "Training of segmentation model started. "
                                                          "This may take a while, please wait!"):
            # stop execution if closed with [Close]
            _hide_status(app_state.training_window)
            return
        _show_status(app_state.training_window, "Training in Progress...")

        train_data = filter_data_by_files(features, train_files)
        val_data = filter_data_by_files(features, val_files)
        test_data = filter_data_by_files(features, test_files)
    else:
        answer = _ask_user_for_small_dataset(parent, len(file_indices))
        if answer == 'cancel' or answer is None:
            _hide_status(app_state.training_window)
            return

        train_data, temp_data = train_test_split(features, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        if not show_confirm_action_window(parent, "Info", "Training of segmentation model started. "
                                                          "This may take a while, please wait!"):
            # stop execution if closed with [X]
            _hide_status(app_state.training_window)
            return
        _show_status(app_state.training_window, "Training in Progress...")

    _set_training_running(app_state, True)

    train_data = train_data[:, 1:]
    val_data = val_data[:, 1:]
    test_data = test_data[:, 1:]

    X_train = train_data[:, :-1].astype('float32')
    y_train = train_data[:, -1].astype('float32')
    X_val = val_data[:, :-1].astype('float32')
    y_val = val_data[:, -1].astype('float32')
    X_test = test_data[:, :-1].astype('float32')
    y_test = test_data[:, -1].astype('float32')

    def downsample_data(data, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = np.min(counts)
        downsampled_data, downsampled_labels = [], []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled = np.random.choice(label_indices, size=min_count, replace=False)
            downsampled_data.append(data[sampled])
            downsampled_labels.append(labels[sampled])
        return np.vstack(downsampled_data), np.hstack(downsampled_labels)

    if imbalance_strategy == 'downsampling':
        X_train, y_train = downsample_data(X_train, y_train)
        X_val, y_val = downsample_data(X_val, y_val)
        # Test set is NOT downsampled: evaluate on the natural class distribution.

    pos_weight = None
    if imbalance_strategy == 'weighted_loss':
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0])

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)

    mean = X_train_tensor.mean()
    std = X_train_tensor.std()

    metadata['mean'] = mean.item()
    metadata['std'] = std.item()
    metadata['hyperparameters'] = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'imbalance_strategy': imbalance_strategy,
    }

    X_train_tensor = (X_train_tensor - mean) / std
    X_val_tensor = (X_val_tensor - mean) / std
    X_test_tensor = (X_test_tensor - mean) / std
    X_train_tensor[torch.isnan(X_train_tensor)] = 0
    X_val_tensor[torch.isnan(X_val_tensor)] = 0
    X_test_tensor[torch.isnan(X_test_tensor)] = 0

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvMLP(input_size=X_train.shape[1]).to(device)
    pw = pos_weight.to(device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    def calculate_accuracy(outputs, labels):
        probs = torch.sigmoid(outputs)
        preds = probs > 0.5
        return (preds == labels).float().mean()

    prefix = f"{training_dataset_name.split('.')[0]}"

    for epoch in range(epochs):
        QApplication.processEvents()
        if _training_cancel_requested(app_state):
            _set_training_running(app_state, False)
            _hide_status(app_state.training_window)
            show_info(parent, "Info", "Training aborted.")
            return

        model.train()
        train_loss, train_accuracy = 0.0, 0.0
        for inputs, labels in train_loader:
            if _training_cancel_requested(app_state):
                _set_training_running(app_state, False)
                _hide_status(app_state.training_window)
                show_info(parent, "Info", "Training aborted.")
                return
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += acc.item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if _training_cancel_requested(app_state):
                    _set_training_running(app_state, False)
                    _hide_status(app_state.training_window)
                    show_info(parent, "Info", "Training aborted.")
                    return
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)
                val_loss += loss.item()
                val_accuracy += acc.item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            app_state.logger.info(f'Early stopping at epoch {epoch + 1}')
            break

        app_state.logger.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}")

    if best_model is not None:
        best_model.eval()
        save_path = os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth')
        torch.save({'model': best_model, 'metadata': metadata}, save_path)

    model_path = os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth')
    checkpoint = _load_checkpoint_with_version_fallback(model_path)
    model = checkpoint['model']
    metadata = checkpoint['metadata']
    model.to(device)

    if _training_cancel_requested(app_state):
        _set_training_running(app_state, False)
        _hide_status(app_state.training_window)
        show_info(parent, "Info", "Training aborted.")
        return

    test_loss, test_accuracy = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if _training_cancel_requested(app_state):
                _set_training_running(app_state, False)
                _hide_status(app_state.training_window)
                show_info(parent, "Info", "Training aborted.")
                return
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            test_loss += loss.item()
            test_accuracy += acc.item()
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    _set_training_running(app_state, False)
    _hide_status(app_state.training_window)
    app_state.training_window.close()
    show_info(parent, "Info",
              f"Model \"{training_dataset_name}\" trained successfully!\nTest Accuracy: {test_accuracy:.4f}")


def start_classification_training(parent, app_state, dataset_name, bird):
    """Start training of classification model using provided dataset and parameters."""

    if dataset_name == "Select Training Dataset":
        show_info(parent, "Error", "Selected training dataset not valid! Perhaps you forgot to pick a dataset?")
        return

    _show_status(app_state.training_window, "Checking files...")

    imbalance_strategy = app_state.train_classification_params['imbalance_strategy'].get()
    epochs = int(app_state.train_classification_params['epochs'].get())
    batch_size = int(app_state.train_classification_params['batch_size'].get())
    learning_rate = float(app_state.train_classification_params['learning_rate'].get())
    early_stopping_patience = int(app_state.train_classification_params['early_stopping_patience'].get())

    aug_params = {k: v.get() for k, v in app_state.augmentation_params.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(app_state.config['global_dir'], 'training_data', dataset_name), 'rb') as f:
        data = pickle.load(f)

    df = data['dataframe']
    metadata = data['metadata']

    df['taf_unflattend_spectrogram'] = df['taf_unflattend_spectrogram'].apply(np.array).apply(torch.tensor)
    inputs = df['taf_unflattend_spectrogram'].tolist()
    labels = df['label'].tolist()

    all_labels, counts = np.unique(labels, return_counts=True)
    labels_below_threshold = all_labels[counts < 6]
    counts_below_threshold = counts[counts < 6]

    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    labels = [label_to_int[label] for label in labels]
    labels = torch.tensor(labels).long()
    metadata.update({"label_to_int": label_to_int, "int_to_label": int_to_label})
    num_classes = len(unique_labels)

    def preprocess_data(data_list):
        return [F.pad(torch.tensor(array).float().unsqueeze(0), (0, 1, 0, 1)) for array in data_list]

    file_groups = df.groupby('file')
    filenames = list(file_groups.groups.keys())

    if not filenames:
        _hide_status(app_state.training_window)
        show_info(parent, "Error", "Given dataset is empty!")
        return

    if any(counts_below_threshold):
        _hide_status(app_state.training_window)
        show_info(parent, "Error",
                  f"Number of labels for syllable {labels_below_threshold} is too small (n = {counts_below_threshold})!\nYou need at least 6 labeled syllables per syllable type.")
        return

    if len(filenames) >= 7:
        train_files, temp_files = train_test_split(filenames, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        if not show_confirm_action_window(parent, "Info", "Training of classification model started. "
                                                          "This may take a while, please wait!"):
            # stop execution if closed with [Close]
            _hide_status(app_state.training_window)
            return
        _show_status(app_state.training_window, "Training in Progress...")

        df_train = df[df['file'].isin(train_files)]
        df_val = df[df['file'].isin(val_files)]
        df_test = df[df['file'].isin(test_files)]

        train_data = preprocess_data(df_train['taf_unflattend_spectrogram'].tolist())
        train_labels = [label_to_int[l] for l in df_train['label'].tolist()]
        val_data = preprocess_data(df_val['taf_unflattend_spectrogram'].tolist())
        val_labels = [label_to_int[l] for l in df_val['label'].tolist()]
        test_data = preprocess_data(df_test['taf_unflattend_spectrogram'].tolist())
        test_labels = [label_to_int[l] for l in df_test['label'].tolist()]
        input_shape = train_data[0].shape
    else:
        input_data = preprocess_data(inputs)
        answer = _ask_user_for_small_dataset(parent, len(filenames))
        if answer == 'cancel' or answer is None:
            _hide_status(app_state.training_window)
            return

        train_data, temp_data, train_labels, temp_labels = train_test_split(
            input_data, labels, test_size=0.3, stratify=labels, random_state=42)
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)
        input_shape = train_data[0].shape
        if not show_confirm_action_window(parent, "Info", "Training of classification model started. "
                                                          "This may take a while, please wait!"):
            # stop execution if closed with [Close]
            _hide_status(app_state.training_window)
            return
        _show_status(app_state.training_window, "Training in Progress...")

    _set_training_running(app_state, True)

    train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
    val_data, val_labels = shuffle(val_data, val_labels, random_state=42)
    test_data, test_labels = shuffle(test_data, test_labels, random_state=42)

    train_data = [(a - a.mean()) / a.std() if a.std() != 0 else a for a in train_data]
    val_data = [(a - a.mean()) / a.std() if a.std() != 0 else a for a in val_data]
    test_data = [(a - a.mean()) / a.std() if a.std() != 0 else a for a in test_data]

    def downsample_data(data, labels):
        data_df = pd.DataFrame({'data': data, 'labels': labels})
        min_size = data_df['labels'].value_counts().min()
        downsampled = pd.DataFrame(columns=data_df.columns)
        for label, group in data_df.groupby('labels'):
            downsampled = pd.concat([downsampled, group.sample(min_size, random_state=42)])
        return downsampled['data'].tolist(), downsampled['labels'].tolist()

    if imbalance_strategy == 'downsampling':
        train_data, train_labels = downsample_data(train_data, train_labels)
        val_data, val_labels = downsample_data(val_data, val_labels)
        # Test set is NOT downsampled: evaluate on the natural class distribution.

    class_weights = None
    if imbalance_strategy == 'weighted_loss':
        counts_dict = Counter(train_labels)
        n_total = sum(counts_dict.values())
        class_weights_vals = [n_total / (num_classes * counts_dict.get(i, 1)) for i in range(num_classes)]
        class_weights = torch.tensor(class_weights_vals, dtype=torch.float)

    train_labels = torch.tensor(train_labels).long()
    val_labels = torch.tensor(val_labels).long()
    test_labels = torch.tensor(test_labels).long()

    train_dataset = TensorDataset(torch.stack(train_data), train_labels)
    val_dataset = TensorDataset(torch.stack(val_data), val_labels)
    test_dataset = TensorDataset(torch.stack(test_data), test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(input_shape=input_shape, num_classes=num_classes).to(device)
    cw = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    prefix = f"{dataset_name.replace('.pkl', '')}"

    def calc_accuracy(loader, model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inp, lab in loader:
                inp, lab = inp.to(device), lab.to(device)
                _, pred = torch.max(model(inp), 1)
                total += lab.size(0)
                correct += (pred == lab).sum().item()
        model.train()
        return correct / total

    metadata['augmentation'] = aug_params
    metadata['hyperparameters'] = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'imbalance_strategy': imbalance_strategy,
    }

    for epoch in range(epochs):
        QApplication.processEvents()
        if _training_cancel_requested(app_state):
            _set_training_running(app_state, False)
            _hide_status(app_state.training_window)
            show_info(parent, "Info", "Training aborted.")
            return

        model.train()
        running_loss = 0.0
        for inp, lab in train_loader:
            if _training_cancel_requested(app_state):
                _set_training_running(app_state, False)
                _hide_status(app_state.training_window)
                show_info(parent, "Info", "Training aborted.")
                return
            augmented = [torch.from_numpy(augment_spectrogram(t.cpu().numpy(), aug_params)).float() for t in inp]
            augmented = torch.stack(augmented).to(device)
            lab = lab.to(device)
            optimizer.zero_grad()
            out = model(augmented)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inp, lab in val_loader:
                if _training_cancel_requested(app_state):
                    _set_training_running(app_state, False)
                    _hide_status(app_state.training_window)
                    show_info(parent, "Info", "Training aborted.")
                    return
                inp, lab = inp.to(device), lab.to(device)
                out = model(inp)
                val_loss += criterion(out, lab).item()
        val_loss /= len(val_loader)

        app_state.logger.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {calc_accuracy(train_loader, model):.4f}, Val Acc: {calc_accuracy(val_loader, model):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            torch.save({'model': model, 'metadata': metadata},
                       os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                app_state.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    model_path = os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth')
    checkpoint = _load_checkpoint_with_version_fallback(model_path)
    model = checkpoint['model']
    model.to(device)

    if _training_cancel_requested(app_state):
        _set_training_running(app_state, False)
        _hide_status(app_state.training_window)
        show_info(parent, "Info", "Training aborted.")
        return

    test_accuracy = calc_accuracy(test_loader, model)

    if _training_cancel_requested(app_state):
        _set_training_running(app_state, False)
        _hide_status(app_state.training_window)
        show_info(parent, "Info", "Training aborted.")
        return

    predictions, targets = get_predictions_and_targets(model, test_loader, device)
    cm = confusion_matrix(targets, predictions)
    labels_range = [int_to_label[i] for i in range(num_classes)]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels_range, yticklabels=labels_range)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_confusion_matrix.svg'))
    plt.close()

    _set_training_running(app_state, False)
    _hide_status(app_state.training_window)
    app_state.training_window.close()
    show_info(parent, "Info",
              f"Model \"{dataset_name}\" trained successfully!\nTest Accuracy: {test_accuracy:.4f}")


def calculate_accuracy_and_percent_probabilities(loader, model, device):
    model.eval()
    correct, total = 0, 0
    all_probs = []
    with torch.no_grad():
        for inp, lab in loader:
            inp, lab = inp.to(device), lab.to(device)
            out = model(inp)
            probs = torch.softmax(out, dim=1) * 100
            _, pred = torch.max(out, 1)
            total += lab.size(0)
            correct += (pred == lab).sum().item()
            all_probs.append(probs.cpu())
    model.train()
    return correct / total, torch.cat(all_probs)


def get_predictions_and_targets(model, data_loader, device):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for inp, lab in data_loader:
            inp, lab = inp.to(device), lab.to(device)
            _, pred = torch.max(model(inp), 1)
            preds.extend(pred.cpu().numpy())
            targs.extend(lab.cpu().numpy())
    return preds, targs


# Default augmentation parameters (used when no config is provided)
DEFAULT_AUGMENTATION_PARAMS = {
    'enabled': True,
    'probability': 0.2,
    'noise_level': 0.0001,
    'freq_mask_width': 10,
    'time_mask_width': 10,
    'compression_factor': 0.5,
}


def augment_spectrogram(spec, aug_params=None):
    import random
    if aug_params is None:
        aug_params = DEFAULT_AUGMENTATION_PARAMS
    if not aug_params.get('enabled', True):
        return spec
    prob = float(aug_params.get('probability', 0.2))
    if np.random.rand() < prob:
        noise_level = float(aug_params.get('noise_level', 0.0001))
        freq_w = int(aug_params.get('freq_mask_width', 10))
        time_w = int(aug_params.get('time_mask_width', 10))
        comp = float(aug_params.get('compression_factor', 0.5))
        augmentations = [
            lambda s: add_noise_to_spectrogram(s, noise_level=noise_level),
            lambda s: dynamic_range_compression(s, compression_factor=comp),
            lambda s: frequency_mask(s, F=freq_w),
            lambda s: time_mask(s, T=time_w),
        ]
        chosen = random.choice(augmentations)
        spec = chosen(spec)
    return spec


def add_noise_to_spectrogram(spec, noise_level=0.0001):
    return spec + noise_level * np.random.randn(*spec.shape)


def frequency_mask(spec, F=10, num_masks=1, replace_with_zero=False):
    cloned = spec.copy()
    nf = spec.shape[0]
    for _ in range(num_masks):
        f = int(np.random.uniform(1, min(F, nf)))
        f = max(1, min(f, nf - 1))
        ms = nf - f
        if ms <= 0:
            continue
        f0 = np.random.randint(0, ms)
        cloned[f0:f0 + f, :] = 0 if replace_with_zero else cloned.mean()
    return cloned


def time_mask(spec, T=10, num_masks=1, replace_with_zero=False):
    cloned = spec.copy()
    nt = spec.shape[1]
    for _ in range(num_masks):
        t = int(np.random.uniform(1, min(T, nt)))
        t = max(1, min(t, nt - 1))
        ms = nt - t
        if ms <= 0:
            continue
        t0 = np.random.randint(0, ms)
        cloned[:, t0:t0 + t] = 0 if replace_with_zero else cloned.mean()
    return cloned


def dynamic_range_compression(spec, compression_factor=0.5):
    return np.log1p(compression_factor * np.expm1(spec))
