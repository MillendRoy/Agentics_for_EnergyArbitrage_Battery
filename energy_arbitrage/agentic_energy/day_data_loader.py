"""
Day-Specific Data Loader with Forecast Support

This module loads energy data for specific days, with options to use
forecasted data from different models.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta

from .schemas import DayInputs, ForecastModel


class DayDataLoader:
    """Load energy data for specific days with forecast support"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the day data loader
        
        Args:
            data_dir: Path to directory containing data and forecast CSV files
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        # Define file paths - use forecast files as primary source (2018-2019 data)
        self.forecast_prices_file = self.data_dir / "predictions_prices.csv"
        self.forecast_consumption_file = self.data_dir / "predictions_consumption.csv"
        
        # Validate files exist
        if not self.forecast_prices_file.exists():
            raise FileNotFoundError(f"Forecast prices file not found: {self.forecast_prices_file}")
        if not self.forecast_consumption_file.exists():
            raise FileNotFoundError(f"Forecast consumption file not found: {self.forecast_consumption_file}")
    
    def get_load_statistics(self) -> dict:
        """Calculate load statistics for battery sizing"""
        df = pd.read_csv(self.forecast_consumption_file)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # Use actual consumption column
        consumption = df['consumption'].dropna()
        
        stats = {
            'mean': consumption.mean(),
            'median': consumption.median(),
            'p25': consumption.quantile(0.25),
            'p75': consumption.quantile(0.75),
            'iqr': consumption.quantile(0.75) - consumption.quantile(0.25),
            'min': consumption.min(),
            'max': consumption.max(),
            'std': consumption.std()
        }
        
        # Calculate recommended battery capacity: IQR * 4 hours
        stats['recommended_capacity_mwh'] = stats['iqr'] * 4
        stats['recommended_power_mw'] = stats['recommended_capacity_mwh'] / 4
        
        return stats
    
    def load_day_data(
        self,
        date: str,
        use_forecast: bool = False,
        forecast_models: Optional[ForecastModel] = None,
        allow_export: bool = False,
        dt_hours: float = 1.0
    ) -> Tuple[DayInputs, dict]:
        """
        Load data for a specific day
        
        Args:
            date: Date string in YYYY-MM-DD format
            use_forecast: Whether to use forecasted data
            forecast_models: Forecast model selections (required if use_forecast=True)
            allow_export: Whether grid export is allowed
            dt_hours: Time step duration in hours
            
        Returns:
            Tuple of (DayInputs, metadata_dict)
        """
        date_obj = pd.to_datetime(date)
        
        metadata = {
            "date": date,
            "use_forecast": use_forecast,
            "data_source": "forecast" if use_forecast else "actual",
            "dt_hours": dt_hours
        }
        
        if use_forecast:
            if forecast_models is None:
                raise ValueError("forecast_models must be provided when use_forecast=True")
            
            prices, consumption = self._load_forecast_data(date_obj, forecast_models)
            metadata["forecast_models"] = {
                "prices": forecast_models.prices_model,
                "consumption": forecast_models.consumption_model
            }
        else:
            prices, consumption = self._load_actual_data(date_obj)
        
        # Create prices_sell (same as buy for now)
        prices_sell = prices if allow_export else None
        
        day_inputs = DayInputs(
            prices_buy=prices,
            demand_kw=consumption,
            prices_sell=prices_sell,
            allow_export=allow_export,
            dt_hours=dt_hours
        )
        
        metadata["num_timesteps"] = len(prices)
        metadata["price_range"] = {"min": min(prices), "max": max(prices), "mean": sum(prices)/len(prices)}
        metadata["demand_range"] = {"min": min(consumption), "max": max(consumption), "mean": sum(consumption)/len(consumption)}
        
        return day_inputs, metadata
    
    def _load_actual_data(self, date: pd.Timestamp) -> Tuple[list, list]:
        """Load actual data for a specific day from predictions files"""
        
        # Read from predictions files (which contain actual data in 'prices' and 'consumption' columns)
        df_prices = pd.read_csv(self.forecast_prices_file)
        df_prices['timestamps'] = pd.to_datetime(df_prices['timestamps'])
        
        df_consumption = pd.read_csv(self.forecast_consumption_file)
        df_consumption['timestamps'] = pd.to_datetime(df_consumption['timestamps'])
        
        # Filter for the specific day
        day_prices = df_prices[df_prices['timestamps'].dt.date == date.date()].sort_values('timestamps')
        day_consumption = df_consumption[df_consumption['timestamps'].dt.date == date.date()].sort_values('timestamps')
        
        if len(day_prices) == 0:
            raise ValueError(f"No price data found for date {date.date()}")
        if len(day_consumption) == 0:
            raise ValueError(f"No consumption data found for date {date.date()}")
        
        prices = day_prices['prices'].fillna(method='ffill').fillna(method='bfill').tolist()
        consumption = day_consumption['consumption'].fillna(method='ffill').fillna(method='bfill').tolist()
        
        # Ensure equal length
        min_len = min(len(prices), len(consumption))
        return prices[:min_len], consumption[:min_len]
    
    def _load_forecast_data(
        self, 
        date: pd.Timestamp, 
        forecast_models: ForecastModel
    ) -> Tuple[list, list]:
        """Load forecasted data for a specific day"""
        
        # Load prices
        df_prices = pd.read_csv(self.forecast_prices_file)
        df_prices['timestamps'] = pd.to_datetime(df_prices['timestamps'])
        day_prices = df_prices[df_prices['timestamps'].dt.date == date.date()].sort_values('timestamps')
        
        if len(day_prices) == 0:
            raise ValueError(f"No price data found for date {date.date()}")
        
        if forecast_models.prices_model:
            # Use forecast column
            price_column = forecast_models.prices_model
            if price_column not in day_prices.columns:
                available = [col for col in day_prices.columns if col not in ['timestamps', 'consumption']]
                raise ValueError(f"Price forecast column '{price_column}' not found. Available: {available}")
            prices = day_prices[price_column].fillna(method='ffill').fillna(method='bfill').tolist()
        else:
            # Use actual prices
            prices = day_prices['prices'].fillna(method='ffill').fillna(method='bfill').tolist()
        
        # Load consumption
        df_consumption = pd.read_csv(self.forecast_consumption_file)
        df_consumption['timestamps'] = pd.to_datetime(df_consumption['timestamps'])
        day_consumption = df_consumption[df_consumption['timestamps'].dt.date == date.date()].sort_values('timestamps')
        
        if len(day_consumption) == 0:
            raise ValueError(f"No consumption data found for date {date.date()}")
        
        if forecast_models.consumption_model:
            # Use forecast column
            consumption_column = forecast_models.consumption_model
            if consumption_column not in day_consumption.columns:
                available = [col for col in day_consumption.columns if col not in ['timestamps', 'prices']]
                raise ValueError(f"Consumption forecast column '{consumption_column}' not found. Available: {available}")
            consumption = day_consumption[consumption_column].fillna(method='ffill').fillna(method='bfill').tolist()
        else:
            # Use actual consumption
            consumption = day_consumption['consumption'].fillna(method='ffill').fillna(method='bfill').tolist()
        
        # Ensure equal length
        min_len = min(len(prices), len(consumption))
        return prices[:min_len], consumption[:min_len]
    
    def get_available_dates(self, use_forecast: bool = False) -> list:
        """Get list of available dates in the dataset"""
        
        df = pd.read_csv(self.forecast_prices_file)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        dates = sorted(df['timestamps'].dt.date.unique())
        return [str(d) for d in dates]