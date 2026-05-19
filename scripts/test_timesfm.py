"""
Quick test to verify TimesFM loads and runs inference on Apple Silicon.
"""
import numpy as np
import timesfm

# Load the smallest available model
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=10,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

# Generate a simple test series
np.random.seed(42)
test_series = np.cumsum(np.random.randn(100)) + 50

# Run inference
forecast_input = [test_series]
frequency_input = [0]  # 0 = high frequency (daily)

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

print("TimesFM loaded successfully!")
print(f"Input length: {len(test_series)}")
print(f"Forecast shape: {point_forecast.shape}")
print(f"First 5 forecast values: {point_forecast[0][:5]}")