import os
import time
from datetime import datetime

import numpy as np
import torch


class BaseEngine:
    """
    Base training engine for PulseTrafficAI.

    Checkpoint strategy
    -------------------
    1. final_model_sXXXX.pt
       - Legacy/model-only checkpoint.
       - Contains model.state_dict().
       - Kept for compatibility with the existing evaluation flow.

    2. best_checkpoint_sXXXX.pt
       - Full resumable checkpoint.
       - Saved whenever the monitored validation metric improves.

    3. last_checkpoint_sXXXX.pt
       - Full resumable checkpoint.
       - Saved after every completed epoch.
       - Used to resume training safely after interruption.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        loss_fn,
        scaler,
        max_epochs,
        patience,
        scheduler,
        seed,
        log_dir,
        clip_grad_value=5,
        null_val=0.0,
        cl_step=0,
        warm_step=0,
        device="cuda",
        experiment_name="experiment",
        incident=False,
        model_config=None,
        resume=True,
        monitor_metric="val_loss",
    ):
        self.model = model
        self._dataloader = dataloader
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scaler = scaler

        self._max_epochs = max_epochs
        self._patience = patience
        self._scheduler = scheduler
        self._seed = seed
        self._save_path = log_dir
        self._clip_grad_value = clip_grad_value
        self._null_val = null_val

        self._cl_step = cl_step
        self._warm_step = warm_step

        self._device = device

        # New checkpoint information
        self._experiment_name = experiment_name
        self._incident = incident
        self._model_config = model_config or {}

        # If True, automatically resume from checkpoint if available.
        self._resume = resume

        # Default remains validation loss so Phase-1 reproduction
        # continues to use the original training criterion.
        self._monitor_metric = monitor_metric

        # Curriculum-learning iteration counter.
        self._iter_cnt = 0

        # Resume state
        self._start_epoch = 1
        self._best_metric = np.inf
        self._best_epoch = 0
        self._wait = 0

        os.makedirs(self._save_path, exist_ok=True)

        if self._resume:
            self._restore_checkpoint_if_available()

    # ------------------------------------------------------------------
    # CHECKPOINT PATHS
    # ------------------------------------------------------------------

    @property
    def _seed_suffix(self):
        return f"s{self._seed}"

    @property
    def _last_checkpoint_path(self):
        return os.path.join(
            self._save_path,
            f"last_checkpoint_{self._seed_suffix}.pt",
        )

    @property
    def _best_checkpoint_path(self):
        return os.path.join(
            self._save_path,
            f"best_checkpoint_{self._seed_suffix}.pt",
        )

    @property
    def _legacy_model_path(self):
        return os.path.join(
            self._save_path,
            f"final_model_{self._seed_suffix}.pt",
        )

    # ------------------------------------------------------------------
    # EXISTING MODEL SAVE/LOAD
    # ------------------------------------------------------------------

    def save_model(self, save_path):
        """
        Preserve the existing behavior.

        final_model_sXXXX.pt contains only model.state_dict().
        This is intentionally retained so existing evaluation code
        remains compatible.
        """

        os.makedirs(save_path, exist_ok=True)

        filename = f"final_model_{self._seed_suffix}.pt"
        model_path = os.path.join(save_path, filename)

        torch.save(
            self.model.state_dict(),
            model_path,
        )

        return model_path

    def load_model(self, save_path):
        """
        Load the legacy/model-only checkpoint.
        """

        filename = f"final_model_{self._seed_suffix}.pt"
        model_path = os.path.join(save_path, filename)

        checkpoint = torch.load(
            model_path,
            map_location=self._device,
        )

        # Existing final_model file contains model.state_dict().
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            self.model.load_state_dict(checkpoint)

    # ------------------------------------------------------------------
    # CHECKPOINT CREATION
    # ------------------------------------------------------------------

    def _get_learning_rate(self):
        """
        Return the current optimizer learning rate.
        """

        if not self._optimizer.param_groups:
            return None

        return self._optimizer.param_groups[0]["lr"]

    def _build_checkpoint(
        self,
        epoch,
        train_loss,
        train_mape,
        train_rmse,
        val_loss,
        val_mape,
        val_rmse,
        epoch_time_sec,
    ):
        """
        Build a complete resumable checkpoint.

        Important:
        model_state_dict is the same information that was previously
        stored inside final_model_sXXXX.pt.

        Additional training state is now stored alongside it.
        """

        checkpoint = {
            # ----------------------------------------------------------
            # Training position
            # ----------------------------------------------------------
            "epoch": epoch,
            "iter_cnt": self._iter_cnt,

            # ----------------------------------------------------------
            # Model
            # ----------------------------------------------------------
            "model_state_dict": self.model.state_dict(),

            # ----------------------------------------------------------
            # Optimizer / scheduler
            # ----------------------------------------------------------
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": (
                self._scheduler.state_dict()
                if self._scheduler is not None
                else None
            ),

            # ----------------------------------------------------------
            # Current epoch metrics
            # ----------------------------------------------------------
            "train_loss": float(train_loss),
            "train_rmse": float(train_rmse),
            "train_mape": float(train_mape),

            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse),
            "val_mape": float(val_mape),

            # ----------------------------------------------------------
            # Best checkpoint information
            # ----------------------------------------------------------
            "best_metric": float(self._best_metric),
            "best_epoch": int(self._best_epoch),
            "wait": int(self._wait),

            # ----------------------------------------------------------
            # Training configuration
            # ----------------------------------------------------------
            "learning_rate": self._get_learning_rate(),
            "seed": self._seed,
            "incident": bool(self._incident),
            "experiment": self._experiment_name,
            "monitor_metric": self._monitor_metric,

            # ----------------------------------------------------------
            # Model configuration
            # ----------------------------------------------------------
            "model_config": self._model_config,

            # ----------------------------------------------------------
            # Timing
            # ----------------------------------------------------------
            "epoch_time_sec": float(epoch_time_sec),
            "timestamp": datetime.now().isoformat(),

            # ----------------------------------------------------------
            # Version / purpose information
            # ----------------------------------------------------------
            "checkpoint_version": 2,
        }

        return checkpoint

    def _save_full_checkpoint(
        self,
        path,
        epoch,
        train_loss,
        train_mape,
        train_rmse,
        val_loss,
        val_mape,
        val_rmse,
        epoch_time_sec,
    ):
        checkpoint = self._build_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            train_mape=train_mape,
            train_rmse=train_rmse,
            val_loss=val_loss,
            val_mape=val_mape,
            val_rmse=val_rmse,
            epoch_time_sec=epoch_time_sec,
        )

        # Save to a temporary file first.
        #
        # This reduces the chance of leaving a corrupted checkpoint
        # if the process is interrupted during torch.save().
        temp_path = path + ".tmp"

        torch.save(
            checkpoint,
            temp_path,
        )

        os.replace(
            temp_path,
            path,
        )

    # ------------------------------------------------------------------
    # CHECKPOINT RESTORE
    # ------------------------------------------------------------------

    def _restore_checkpoint_if_available(self):
        """
        Restore training state.

        Priority:
            1. last_checkpoint
            2. best_checkpoint
            3. old final_model

        The last checkpoint has highest priority because it represents
        the most recently completed training epoch.
        """

        if os.path.exists(self._last_checkpoint_path):
            print(
                "\n[CHECKPOINT] Last checkpoint found."
            )
            print(
                f"[CHECKPOINT] Loading: {self._last_checkpoint_path}"
            )

            self._load_full_checkpoint(
                self._last_checkpoint_path
            )

            return

        if os.path.exists(self._best_checkpoint_path):
            print(
                "\n[CHECKPOINT] Best checkpoint found."
            )
            print(
                f"[CHECKPOINT] Loading: {self._best_checkpoint_path}"
            )

            self._load_full_checkpoint(
                self._best_checkpoint_path
            )

            return

        if os.path.exists(self._legacy_model_path):
            print(
                "\n[CHECKPOINT] Existing legacy model found."
            )
            print(
                f"[CHECKPOINT] Loading model weights: "
                f"{self._legacy_model_path}"
            )

            checkpoint = torch.load(
                self._legacy_model_path,
                map_location=self._device,
            )

            if (
                isinstance(checkpoint, dict)
                and "model_state_dict" in checkpoint
            ):
                self.model.load_state_dict(
                    checkpoint["model_state_dict"]
                )
            else:
                self.model.load_state_dict(checkpoint)

            print(
                "[CHECKPOINT] WARNING:"
                " The existing final_model file contains only model"
                " weights. Optimizer, scheduler and epoch information"
                " are not available."
            )

            print(
                "[CHECKPOINT] Therefore this run starts a NEW"
                " resumable checkpoint history from the loaded weights."
            )

            self._start_epoch = 1
            self._iter_cnt = 0
            self._best_metric = np.inf
            self._best_epoch = 0
            self._wait = 0

            return

        print(
            "\n[CHECKPOINT] No existing checkpoint found."
        )
        print(
            "[CHECKPOINT] Training will start from Epoch 1."
        )

    def _load_full_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self._device,
        )

        required_keys = [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "train_loss",
            "val_loss",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in checkpoint
        ]

        if missing_keys:
            raise RuntimeError(
                "Checkpoint is missing required fields: "
                + ", ".join(missing_keys)
            )

        # Restore model
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Restore optimizer
        self._optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # Restore scheduler
        if (
            self._scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            self._scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        # Restore training position
        saved_epoch = int(checkpoint["epoch"])

        self._start_epoch = saved_epoch + 1

        # Restore curriculum-learning state
        self._iter_cnt = int(
            checkpoint.get("iter_cnt", 0)
        )

        # Restore best metric
        self._best_metric = float(
            checkpoint.get(
                "best_metric",
                np.inf,
            )
        )

        self._best_epoch = int(
            checkpoint.get(
                "best_epoch",
                0,
            )
        )

        self._wait = int(
            checkpoint.get(
                "wait",
                0,
            )
        )

        print(
            "\n[CHECKPOINT] Resume information"
        )
        print(
            f"  Experiment      : "
            f"{checkpoint.get('experiment', 'unknown')}"
        )
        print(
            f"  Incident        : "
            f"{checkpoint.get('incident', 'unknown')}"
        )
        print(
            f"  Last Epoch      : {saved_epoch}"
        )
        print(
            f"  Next Epoch      : {self._start_epoch}"
        )
        print(
            f"  Best Epoch      : {self._best_epoch}"
        )
        print(
            f"  Best Metric     : {self._best_metric:.6f}"
        )
        print(
            f"  Monitor Metric  : "
            f"{checkpoint.get('monitor_metric', self._monitor_metric)}"
        )
        print(
            f"  Learning Rate   : "
            f"{checkpoint.get('learning_rate')}"
        )
        print(
            f"  Last Train MAE  : "
            f"{checkpoint.get('train_loss')}"
        )
        print(
            f"  Last Train MAPE : "
            f"{checkpoint.get('train_mape')}"
        )
        print(
            f"  Last Val MAE    : "
            f"{checkpoint.get('val_loss')}"
        )
        print(
            f"  Last Val MAPE   : "
            f"{checkpoint.get('val_mape')}"
        )
        print(
            f"  Saved At        : "
            f"{checkpoint.get('timestamp')}"
        )

    # ------------------------------------------------------------------
    # MONITORING
    # ------------------------------------------------------------------

    def _get_monitor_value(
        self,
        val_loss,
        val_mape,
        val_rmse,
    ):
        if self._monitor_metric == "val_loss":
            return float(val_loss)

        if self._monitor_metric == "val_mape":
            return float(val_mape)

        if self._monitor_metric == "val_rmse":
            return float(val_rmse)

        raise ValueError(
            f"Unsupported monitor metric: "
            f"{self._monitor_metric}"
        )

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train(self):
        """
        Train model with resumable checkpoints.

        Important:
        max_epochs represents the total epoch number.

        Example:
            Run 1:
                epochs 1-10

            Run 2:
                checkpoint says epoch 10
                training continues 11-20

            Run 3:
                checkpoint says epoch 20
                training continues 21-30
        """

        if self._start_epoch > self._max_epochs:
            print(
                "\n[TRAINING] Checkpoint already reached "
                f"Epoch {self._start_epoch - 1}."
            )

            print(
                f"[TRAINING] max_epochs is currently "
                f"{self._max_epochs}."
            )

            print(
                "[TRAINING] No new epochs will run."
            )

            print(
                "[TRAINING] Increase max_epochs to continue."
            )

            self.evaluate("test")
            return

        print("\n" + "=" * 70)
        print("IGSTGNN TRAINING")
        print("=" * 70)

        print(
            f"Experiment      : {self._experiment_name}"
        )
        print(
            f"Incident        : {self._incident}"
        )
        print(
            f"Start Epoch     : {self._start_epoch}"
        )
        print(
            f"End Epoch       : {self._max_epochs}"
        )
        print(
            f"Monitor Metric  : {self._monitor_metric}"
        )
        print(
            f"Batch Size      : "
            f"{self._dataloader['train_loader'].bs}"
        )

        for epoch in range(
            self._start_epoch,
            self._max_epochs + 1,
        ):
            epoch_start = time.time()

            # ----------------------------------------------------------
            # TRAIN
            # ----------------------------------------------------------
            train_start = time.time()

            (
                mtrain_loss,
                mtrain_mape,
                mtrain_rmse,
            ) = self.train_batch()

            train_time = time.time() - train_start

            # ----------------------------------------------------------
            # VALIDATION
            # ----------------------------------------------------------
            val_start = time.time()

            (
                mvalid_loss,
                mvalid_mape,
                mvalid_rmse,
            ) = self.evaluate("val")

            val_time = time.time() - val_start

            # ----------------------------------------------------------
            # Scheduler
            # ----------------------------------------------------------
            if self._scheduler is not None:
                self._scheduler.step()

            # ----------------------------------------------------------
            # Epoch timing
            # ----------------------------------------------------------
            epoch_time = time.time() - epoch_start

            current_lr = self._get_learning_rate()

            # ----------------------------------------------------------
            # Check whether current epoch is best
            # ----------------------------------------------------------
            monitor_value = self._get_monitor_value(
                val_loss=mvalid_loss,
                val_mape=mvalid_mape,
                val_rmse=mvalid_rmse,
            )

            is_best = (
                monitor_value < self._best_metric
            )

            if is_best:
                self._best_metric = monitor_value
                self._best_epoch = epoch
                self._wait = 0
            else:
                self._wait += 1

            # ----------------------------------------------------------
            # Console output
            # ----------------------------------------------------------
            print(
                "\n"
                + "-" * 70
            )

            print(
                f"Epoch {epoch}/{self._max_epochs}"
            )

            print(
                f"Train | "
                f"Loss/MAE: {mtrain_loss:.4f} | "
                f"RMSE: {mtrain_rmse:.4f} | "
                f"MAPE: {mtrain_mape:.4f}"
            )

            print(
                f"Valid | "
                f"Loss/MAE: {mvalid_loss:.4f} | "
                f"RMSE: {mvalid_rmse:.4f} | "
                f"MAPE: {mvalid_mape:.4f}"
            )

            print(
                f"LR: {current_lr}"
            )

            print(
                f"Epoch Time: {epoch_time:.2f} sec"
            )

            print(
                f"Best Epoch: {self._best_epoch}"
            )

            print(
                f"Best {self._monitor_metric}: "
                f"{self._best_metric:.6f}"
            )

            print(
                f"Early Stop Wait: "
                f"{self._wait}/{self._patience}"
            )

            # ----------------------------------------------------------
            # Save LAST checkpoint after every completed epoch
            # ----------------------------------------------------------
            self._save_full_checkpoint(
                path=self._last_checkpoint_path,
                epoch=epoch,
                train_loss=mtrain_loss,
                train_mape=mtrain_mape,
                train_rmse=mtrain_rmse,
                val_loss=mvalid_loss,
                val_mape=mvalid_mape,
                val_rmse=mvalid_rmse,
                epoch_time_sec=epoch_time,
            )

            print(
                "[CHECKPOINT] Last checkpoint saved."
            )

            # ----------------------------------------------------------
            # Save BEST checkpoint
            # ----------------------------------------------------------
            if is_best:
                self._save_full_checkpoint(
                    path=self._best_checkpoint_path,
                    epoch=epoch,
                    train_loss=mtrain_loss,
                    train_mape=mtrain_mape,
                    train_rmse=mtrain_rmse,
                    val_loss=mvalid_loss,
                    val_mape=mvalid_mape,
                    val_rmse=mvalid_rmse,
                    epoch_time_sec=epoch_time,
                )

                # Also preserve the old model-only checkpoint.
                self.save_model(
                    self._save_path
                )

                print(
                    "[CHECKPOINT] New best checkpoint saved."
                )

                print(
                    "[CHECKPOINT] final_model_sXXXX.pt updated."
                )

            # ----------------------------------------------------------
            # Early stopping
            # ----------------------------------------------------------
            if self._wait >= self._patience:
                print(
                    "\n[EARLY STOPPING]"
                )

                print(
                    f"No improvement in "
                    f"{self._monitor_metric} for "
                    f"{self._patience} epochs."
                )

                break

        # --------------------------------------------------------------
        # Final test evaluation
        # --------------------------------------------------------------
        print(
            "\n[TRAINING] Loading best model for final test evaluation."
        )

        if os.path.exists(
            self._best_checkpoint_path
        ):
            best_checkpoint = torch.load(
                self._best_checkpoint_path,
                map_location=self._device,
            )

            self.model.load_state_dict(
                best_checkpoint["model_state_dict"]
            )

        elif os.path.exists(
            self._legacy_model_path
        ):
            self.load_model(
                self._save_path
            )

        self.evaluate("test")