import numpy as np
import torch
from tqdm import tqdm

from .base_engine import BaseEngine
from .metrics import compute_all_metrics, masked_mape, masked_rmse


class IGSTGNN_Engine(BaseEngine):
    """
    Incident-aware IGSTGNN engine.

    Responsibilities:
    - Prepare traffic and incident batches
    - Run IGSTGNN forward pass
    - Train the model
    - Calculate MAE, MAPE and RMSE
    - Evaluate validation/test data
    - Work with the resumable checkpoint system in BaseEngine
    - Detect the first non-finite numerical value during training
    """

    def __init__(
        self,
        cl_step,
        warm_step,
        horizon,
        incident=False,
        **args
    ):
        super(IGSTGNN_Engine, self).__init__(**args)

        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = horizon
        self._cl_len = 0
        self._incident = incident

    # =================================================================
    # TENSOR CONVERSION
    # =================================================================

    def _as_float_tensor(self, value):
        return torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self._device,
        )

    def _as_tensor(self, value):
        return torch.as_tensor(
            value,
            device=self._device,
        )

    # =================================================================
    # NUMERICAL DIAGNOSTICS
    # =================================================================

    def _tensor_diagnostics(self, name, value):
        """
        Print numerical information for a tensor.

        Returns True when the tensor contains only finite values.
        """
        if not torch.is_tensor(value):
            print(
                f"[NUMERICAL CHECK] {name}: "
                f"not a torch.Tensor"
            )
            return False

        finite = torch.isfinite(value)

        is_finite = bool(torch.all(finite).item())

        if is_finite:
            return True

        nan_count = int(
            torch.isnan(value).sum().item()
        )

        inf_count = int(
            torch.isinf(value).sum().item()
        )

        finite_values = value[finite]

        if finite_values.numel() > 0:
            min_value = finite_values.min().item()
            max_value = finite_values.max().item()
        else:
            min_value = float("nan")
            max_value = float("nan")

        print(
            f"[NUMERICAL CHECK] {name} contains "
            f"non-finite values."
        )
        print(
            f"  shape       : {tuple(value.shape)}"
        )
        print(
            f"  NaN count   : {nan_count}"
        )
        print(
            f"  Inf count   : {inf_count}"
        )
        print(
            f"  finite min  : {min_value}"
        )
        print(
            f"  finite max  : {max_value}"
        )

        return False

    def _check_gradients(self, batch_number):
        """
        Check model gradients immediately after loss.backward().
        """
        bad_parameters = []

        for name, parameter in self.model.named_parameters():

            if parameter.grad is None:
                continue

            if not torch.isfinite(parameter.grad).all():
                bad_parameters.append(name)

        if bad_parameters:

            print(
                "\n[NUMERICAL CHECK] Non-finite gradient detected."
            )
            print(
                f"  Batch       : {batch_number}"
            )
            print(
                f"  Parameters  : "
                f"{len(bad_parameters)}"
            )

            for name in bad_parameters[:10]:
                print(
                    f"    - {name}"
                )

            return False

        return True

    # =================================================================
    # INVERSE TRANSFORMATION
    # =================================================================

    def _inverse_transform(self, values):
        """
        Convert normalized traffic values back to original traffic
        scale using the StandardScaler supplied by the data loader.
        """
        return [
            self._scaler.inverse_transform(value)
            for value in values
        ]

    # =================================================================
    # BATCH PREPARATION
    # =================================================================

    def _prepare_batch(self, batch):

        if isinstance(batch, dict):

            X = self._as_float_tensor(
                batch["x_data"]
            )

            label = self._as_float_tensor(
                batch["y_data"]
            )

            incident_data = None

            if self._incident:

                incident_data = {
                    "incident": self._as_float_tensor(
                        batch["incident_features"]
                    ),

                    "position": self._as_tensor(
                        batch["incident_position"]
                    ),

                    "distances": self._as_float_tensor(
                        batch["incident_distances"]
                    ),
                }

            sensor_data = None

            if "sensor_data" in batch:

                sensor_data = {
                    key: self._as_tensor(value)
                    for key, value in batch["sensor_data"].items()
                }

            return (
                X,
                label,
                incident_data,
                sensor_data,
            )

        X, label = batch

        X = self._as_float_tensor(X)

        label = self._as_float_tensor(label)

        return (
            X,
            label,
            None,
            None,
        )

    # =================================================================
    # MODEL PREDICTION
    # =================================================================

    def _predict(
        self,
        X,
        label,
        incident_data,
        sensor_data,
    ):

        if incident_data is not None:

            return self.model(
                X,
                label,
                incident_data=incident_data,
                sensor_data=sensor_data,
            )

        return self.model(
            X,
            label,
        )

    # =================================================================
    # MASK VALUE
    # =================================================================

    def _mask_value(self, label):

        mask_value = torch.tensor(
            0.0,
            device=label.device,
        )

        if label.min() < 1:
            mask_value = label.min()

        return mask_value

    # =================================================================
    # CURRICULUM LEARNING
    # =================================================================

    def _update_curriculum_length(self):
        """Update curriculum-learning horizon safely.

        When both warm_step and cl_step are zero, curriculum learning is
        disabled. In that configuration the complete forecast horizon must
        be used from the first batch.
        """

        self._iter_cnt += 1

        # Curriculum learning disabled: use the complete horizon.
        if self._warm_step == 0 and self._cl_step == 0:
            self._cl_len = self._horizon
            return

        if self._iter_cnt < self._warm_step:
            self._cl_len = self._horizon

        elif self._iter_cnt == self._warm_step:
            self._cl_len = 1

        elif self._cl_step > 0:
            if (
                (self._iter_cnt - self._warm_step)
                % self._cl_step == 0
                and self._cl_len < self._horizon
            ):
                self._cl_len += 1

    # =================================================================
    # TRAIN ONE EPOCH
    # =================================================================

    def train_batch(self):

        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []

        self._dataloader["train_loader"].shuffle()

        iterator = (
            self._dataloader[
                "train_loader"
            ].get_iterator()
        )

        total_batches = (
            self._dataloader[
                "train_loader"
            ].num_batch
        )

        progress_bar = tqdm(
            iterator,
            total=total_batches,
            desc="Training",
            unit="batch",
            leave=False,
            position=0,
            dynamic_ncols=True,
            colour="green",
        )

        try:

            for batch_number, batch in enumerate(
                progress_bar,
                start=1,
            ):

                self._optimizer.zero_grad()

                (
                    X,
                    label,
                    incident_data,
                    sensor_data,
                ) = self._prepare_batch(batch)

                # -----------------------------------------------------
                # Check raw tensors before model forward
                # -----------------------------------------------------

                if not self._tensor_diagnostics(
                    "Input X",
                    X,
                ):
                    raise FloatingPointError(
                        f"Non-finite Input X detected at batch "
                        f"{batch_number}."
                    )

                if not self._tensor_diagnostics(
                    "Input label",
                    label,
                ):
                    raise FloatingPointError(
                        f"Non-finite input label detected at batch "
                        f"{batch_number}."
                    )

                if incident_data is not None:

                    if not self._tensor_diagnostics(
                        "Incident features",
                        incident_data["incident"],
                    ):
                        raise FloatingPointError(
                            "Non-finite incident features detected "
                            f"at batch {batch_number}."
                        )

                    if not self._tensor_diagnostics(
                        "Incident distances",
                        incident_data["distances"],
                    ):
                        raise FloatingPointError(
                            "Non-finite incident distances detected "
                            f"at batch {batch_number}."
                        )

                # -----------------------------------------------------
                # Forward pass
                # -----------------------------------------------------

                pred = self._predict(
                    X,
                    label,
                    incident_data,
                    sensor_data,
                )

                if not self._tensor_diagnostics(
                    "Normalized prediction",
                    pred,
                ):
                    raise FloatingPointError(
                        "Non-finite model prediction detected at "
                        f"batch {batch_number}."
                    )

                # -----------------------------------------------------
                # IMPORTANT:
                # Keep the original model prediction for the training
                # loss.  The scaler's inverse_transform() is intended
                # for metric reporting and may detach the autograd graph.
                # If we calculate loss after inverse_transform(), PyTorch
                # can produce a tensor without grad_fn and loss.backward()
                # will fail.
                # -----------------------------------------------------

                pred_for_loss = pred
                label_for_loss = label

                # -----------------------------------------------------
                # Curriculum learning
                # -----------------------------------------------------

                self._update_curriculum_length()

                pred_for_loss = pred_for_loss[
                    :,
                    :self._cl_len,
                    :,
                    :,
                ]

                label_for_loss = label_for_loss[
                    :,
                    :self._cl_len,
                    :,
                    :,
                ]

                # -----------------------------------------------------
                # Training loss is calculated in normalized space.
                # This preserves the model's autograd graph.
                # -----------------------------------------------------

                mask_value_train = self._mask_value(
                    label_for_loss
                )

                if self._iter_cnt == 1:

                    print(
                        "check normalized mask value",
                        mask_value_train,
                    )

                loss = self._loss_fn(
                    pred_for_loss,
                    label_for_loss,
                    mask_value_train,
                )

                if not torch.isfinite(loss):

                    print(
                        "\n[NUMERICAL CHECK] "
                        "Non-finite loss detected."
                    )
                    print(
                        f"  Batch       : {batch_number}"
                    )
                    print(
                        f"  Loss        : {loss.item()}"
                    )

                    self._tensor_diagnostics(
                        "Prediction before loss",
                        pred_for_loss,
                    )

                    self._tensor_diagnostics(
                        "Label before loss",
                        label_for_loss,
                    )

                    raise FloatingPointError(
                        f"Non-finite loss detected at batch "
                        f"{batch_number}."
                    )

                # -----------------------------------------------------
                # Inverse transformation is ONLY for reporting metrics.
                # -----------------------------------------------------

                pred_metric, label_metric = self._inverse_transform(
                    [
                        pred_for_loss.detach(),
                        label_for_loss.detach(),
                    ]
                )

                if not self._tensor_diagnostics(
                    "Inverse-transformed prediction",
                    pred_metric,
                ):
                    raise FloatingPointError(
                        "Non-finite inverse-transformed prediction "
                        f"detected at batch {batch_number}."
                    )

                if not self._tensor_diagnostics(
                    "Inverse-transformed label",
                    label_metric,
                ):
                    raise FloatingPointError(
                        "Non-finite inverse-transformed label "
                        f"detected at batch {batch_number}."
                    )

                # Use original-scale values only for MAPE/RMSE reporting.
                mask_value_metric = self._mask_value(
                    label_metric
                )

                if not torch.isfinite(loss):

                    print(
                        "\n[NUMERICAL CHECK] "
                        "Non-finite loss detected."
                    )
                    print(
                        f"  Batch       : {batch_number}"
                    )
                    print(
                        f"  Loss        : {loss.item()}"
                    )

                    self._tensor_diagnostics(
                        "Prediction before loss",
                        pred,
                    )

                    self._tensor_diagnostics(
                        "Label before loss",
                        label,
                    )

                    raise FloatingPointError(
                        f"Non-finite loss detected at batch "
                        f"{batch_number}."
                    )

                # -----------------------------------------------------
                # Metrics
                # -----------------------------------------------------

                mape_tensor = masked_mape(
                    pred_metric,
                    label_metric,
                    mask_value_metric,
                )

                rmse_tensor = masked_rmse(
                    pred_metric,
                    label_metric,
                    mask_value_metric,
                )

                if not torch.isfinite(mape_tensor):

                    raise FloatingPointError(
                        "Non-finite MAPE detected at batch "
                        f"{batch_number}."
                    )

                if not torch.isfinite(rmse_tensor):

                    raise FloatingPointError(
                        "Non-finite RMSE detected at batch "
                        f"{batch_number}."
                    )

                mape = mape_tensor.item()
                rmse = rmse_tensor.item()

                # -----------------------------------------------------
                # Backpropagation
                # -----------------------------------------------------

                loss.backward()

                if not self._check_gradients(
                    batch_number
                ):
                    raise FloatingPointError(
                        "Non-finite gradient detected at batch "
                        f"{batch_number}."
                    )

                # -----------------------------------------------------
                # Gradient clipping
                # -----------------------------------------------------

                if self._clip_grad_value != 0:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self._clip_grad_value,
                    )

                # Check again after clipping.
                if not self._check_gradients(
                    batch_number
                ):
                    raise FloatingPointError(
                        "Non-finite gradient detected after "
                        f"gradient clipping at batch {batch_number}."
                    )

                # -----------------------------------------------------
                # Optimizer update
                # -----------------------------------------------------

                self._optimizer.step()

                # -----------------------------------------------------
                # Check model parameters after optimizer update
                # -----------------------------------------------------

                bad_parameters = []

                for name, parameter in self.model.named_parameters():

                    if not torch.isfinite(parameter).all():
                        bad_parameters.append(name)

                if bad_parameters:

                    print(
                        "\n[NUMERICAL CHECK] "
                        "Non-finite model parameter detected "
                        "after optimizer step."
                    )
                    print(
                        f"  Batch       : {batch_number}"
                    )

                    for name in bad_parameters[:10]:
                        print(
                            f"    - {name}"
                        )

                    raise FloatingPointError(
                        "Model parameters became non-finite after "
                        f"optimizer step at batch {batch_number}."
                    )

                # -----------------------------------------------------
                # Store metrics
                # -----------------------------------------------------

                train_loss.append(
                    loss.item()
                )

                train_mape.append(
                    mape
                )

                train_rmse.append(
                    rmse
                )

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "mape": f"{mape:.4f}",
                        "rmse": f"{rmse:.4f}",
                    }
                )

        finally:

            progress_bar.close()

        return (
            np.mean(train_loss),
            np.mean(train_mape),
            np.mean(train_rmse),
        )

    # =================================================================
    # EVALUATION
    # =================================================================

    def evaluate(self, mode):

        if mode == "test":

            self.load_model(
                self._save_path
            )

        self.model.eval()

        preds = []
        labels = []

        with torch.no_grad():

            iterator = (
                self._dataloader[
                    mode + "_loader"
                ].get_iterator()
            )

            for batch in iterator:

                (
                    X,
                    label,
                    incident_data,
                    sensor_data,
                ) = self._prepare_batch(batch)

                pred = self._predict(
                    X,
                    label,
                    incident_data,
                    sensor_data,
                )

                pred, label = self._inverse_transform(
                    [
                        pred,
                        label,
                    ]
                )

                if not self._tensor_diagnostics(
                    "Evaluation prediction",
                    pred,
                ):
                    raise FloatingPointError(
                        f"Non-finite prediction during {mode} evaluation."
                    )

                preds.append(
                    pred.squeeze(-1).cpu()
                )

                labels.append(
                    label.squeeze(-1).cpu()
                )

        preds = torch.cat(
            preds,
            dim=0,
        )

        labels = torch.cat(
            labels,
            dim=0,
        )

        mask_value = torch.tensor(
            0.0
        )

        if labels.min() < 1:
            mask_value = labels.min()

        if mode == "val":

            mae = self._loss_fn(
                preds,
                labels,
                mask_value,
            ).item()

            mape = masked_mape(
                preds,
                labels,
                mask_value,
            ).item()

            rmse = masked_rmse(
                preds,
                labels,
                mask_value,
            ).item()

            return (
                mae,
                mape,
                rmse,
            )

        if mode == "test":

            test_mae = []
            test_mape = []
            test_rmse = []

            print(
                "Check mask value",
                mask_value,
            )

            for i in range(
                self.model.horizon
            ):

                res = compute_all_metrics(
                    preds[:, i, :],
                    labels[:, i, :],
                    mask_value,
                )

                log = (
                    "Horizon {:d}, "
                    "Test MAE: {:.4f}, "
                    "Test RMSE: {:.4f}, "
                    "Test MAPE: {:.4f}"
                )

                if hasattr(self, "_logger"):

                    self._logger.info(
                        log.format(
                            i + 1,
                            res[0],
                            res[2],
                            res[1],
                        )
                    )

                else:

                    print(
                        log.format(
                            i + 1,
                            res[0],
                            res[2],
                            res[1],
                        )
                    )

                test_mae.append(
                    res[0]
                )

                test_mape.append(
                    res[1]
                )

                test_rmse.append(
                    res[2]
                )

            log = (
                "Average Test MAE: {:.4f}, "
                "Test RMSE: {:.4f}, "
                "Test MAPE: {:.4f}"
            )

            if hasattr(self, "_logger"):

                self._logger.info(
                    log.format(
                        np.mean(test_mae),
                        np.mean(test_rmse),
                        np.mean(test_mape),
                    )
                )

            else:

                print(
                    log.format(
                        np.mean(test_mae),
                        np.mean(test_rmse),
                        np.mean(test_mape),
                    )
                )
