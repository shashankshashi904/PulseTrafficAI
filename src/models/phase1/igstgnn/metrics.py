import torch


def _build_mask(labels, null_val):
    """
    Build a finite mask for metric calculations.

    Values equal to null_val are ignored.
    NaN/Inf labels are also ignored.
    """
    if torch.is_tensor(null_val):
        if torch.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = labels != null_val
    else:
        if torch.isnan(torch.tensor(null_val, device=labels.device)):
            mask = ~torch.isnan(labels)
        else:
            mask = labels != null_val

    mask = mask & torch.isfinite(labels)

    mask = mask.float()

    valid_count = torch.sum(mask)

    if valid_count > 0:
        mask = mask / valid_count * mask.numel()
    else:
        mask = torch.zeros_like(mask)

    return mask


def masked_mse(preds, labels, null_val):
    """
    Calculate masked mean squared error safely.
    """
    mask = _build_mask(labels, null_val)

    finite = torch.isfinite(preds) & torch.isfinite(labels)

    mask = mask * finite.float()

    valid_count = torch.sum(mask)

    if valid_count > 0:
        mask = mask / valid_count * mask.numel()
    else:
        return torch.zeros(
            (),
            dtype=preds.dtype,
            device=preds.device,
        )

    loss = (preds - labels) ** 2

    loss = torch.where(
        torch.isfinite(loss),
        loss,
        torch.zeros_like(loss),
    )

    loss = loss * mask

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    """
    Calculate masked root mean squared error safely.
    """
    mse = masked_mse(
        preds=preds,
        labels=labels,
        null_val=null_val,
    )

    return torch.sqrt(
        torch.clamp(
            mse,
            min=0.0,
        )
    )


def masked_mae(preds, labels, null_val):
    """
    Calculate masked mean absolute error safely.
    """
    mask = _build_mask(labels, null_val)

    finite = torch.isfinite(preds) & torch.isfinite(labels)

    mask = mask * finite.float()

    valid_count = torch.sum(mask)

    if valid_count > 0:
        mask = mask / valid_count * mask.numel()
    else:
        return torch.zeros(
            (),
            dtype=preds.dtype,
            device=preds.device,
        )

    loss = torch.abs(preds - labels)

    loss = torch.where(
        torch.isfinite(loss),
        loss,
        torch.zeros_like(loss),
    )

    loss = loss * mask

    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    """
    Calculate masked mean absolute percentage error safely.

    Labels equal to zero are excluded from the percentage calculation
    because division by zero is undefined.
    """
    mask = _build_mask(labels, null_val)

    finite = (
        torch.isfinite(preds)
        & torch.isfinite(labels)
        & (torch.abs(labels) > 1e-8)
    )

    mask = mask * finite.float()

    valid_count = torch.sum(mask)

    if valid_count > 0:
        mask = mask / valid_count * mask.numel()
    else:
        return torch.zeros(
            (),
            dtype=preds.dtype,
            device=preds.device,
        )

    loss = torch.abs(preds - labels) / torch.clamp(
        torch.abs(labels),
        min=1e-8,
    )

    loss = torch.where(
        torch.isfinite(loss),
        loss,
        torch.zeros_like(loss),
    )

    loss = loss * mask

    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    """
    Calculate MAE, MAPE and RMSE.
    """
    mae = masked_mae(
        preds,
        labels,
        null_val,
    ).item()

    mape = masked_mape(
        preds,
        labels,
        null_val,
    ).item()

    rmse = masked_rmse(
        preds,
        labels,
        null_val,
    ).item()

    return mae, mape, rmse
