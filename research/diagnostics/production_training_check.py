import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_config
from core.ml.train import load_ml_ohlc, _prepare_symbol_dataset, _to_sequences, resolve_forecast_horizons
from core.ml.dataset import purged_walk_forward_splits
from core.ml.date_validation import chronological_purged_split
from core.ml.tcn_model import TCNTrendModel

cfg = load_config(PROJECT_ROOT / 'configs' / 'training_crypto_intraday.yaml')
raw = load_ml_ohlc(cfg.data, cfg.ml)
ohlc = raw['BTC/USDT']
H = resolve_forecast_horizons(cfg.ml)
X, y, t1_pos, mfe, mae, horizon_labels = _prepare_symbol_dataset(ohlc, None, cfg.ml, forecast_horizons=H)
X_seq, y_seq, t1_seq, index_seq, mfe_seq, mae_seq = _to_sequences(X, y, t1_pos, cfg.ml.sequence_length, mfe, mae)
folds = purged_walk_forward_splits(len(y_seq), t1_seq, cfg.ml.n_splits, cfg.ml.embargo_fraction)
wf4 = folds[3]
print('PRODUCTION_CONFIG=configs/training_crypto_intraday.yaml')
print('PRODUCTION_HORIZONS=' + str(H))
print('PRODUCTION_DEVICE=' + cfg.ml.training_device)
print('MODEL_DEVICE=' + TCNTrendModel(cfg.ml, X_seq.shape[2], forecast_horizons=H, prefer_cuda=True, asset='BTC/USDT').device.type.upper())
print('WF4_INDEX=3')
print('WF4_TRAIN_SAMPLES=' + str(len(wf4.train_idx)))
print('WF4_TEST_SAMPLES=' + str(len(wf4.test_idx)))
print('WF4_VALIDATION_SAMPLES=0')
# chronological split for the final full-history fit to inspect the validation slice used by the real training pipeline
split = chronological_purged_split(
    index_seq,
    t1_seq,
    train_end=cfg.ml.train_end,
    validation_start=cfg.ml.validation_start,
    validation_end=cfg.ml.validation_end,
    test_start=cfg.ml.test_start,
    test_end=cfg.ml.test_end,
    purge_hours=cfg.ml.purge_hours,
    minimum_train_samples=cfg.ml.minimum_train_samples,
    minimum_validation_samples=cfg.ml.minimum_validation_samples,
    minimum_test_samples=cfg.ml.minimum_test_samples,
)
print('CHRONO_TRAIN_SAMPLES=' + str(len(split.train_idx)))
print('CHRONO_VALIDATION_SAMPLES=' + str(len(split.validation_idx)))
print('CHRONO_TEST_SAMPLES=' + str(len(split.test_idx)))
