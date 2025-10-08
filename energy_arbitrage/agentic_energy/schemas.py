from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class MetricStats(BaseModel):
    count: Optional[int] = Field(None, description="Number of valid data points")
    min: Optional[float] = Field(None, description="Minimum value")
    max: Optional[float] = Field(None, description="Maximum value")
    avg: Optional[float] = Field(None, description="Average value")
    median: Optional[float] = Field(None, description="Median value")
    p25: Optional[float] = Field(None, description="25th percentile")
    p75: Optional[float] = Field(None, description="75th percentile")
    std: Optional[float] = Field(None, description="Standard deviation")
    var: Optional[float] = Field(None, description="Variance")

class DateRange(BaseModel):
    start: Optional[str]
    end: Optional[str]

class SummaryStats(BaseModel):
    region: str
    total_records: int
    date_range: DateRange
    prices: Optional[MetricStats]
    consumption: Optional[MetricStats]

class EnergyDataRecord(BaseModel):
    """Base energy data record with common fields across all regions"""
    timestamps: str = Field(description="Timestamp in ISO format")
    prices: Optional[float] = Field(None, description="Energy price at timestamp")
    consumption: Optional[float] = Field(None, description="Energy consumption")
    year: Optional[int] = Field(None, description="Year extracted from timestamp")
    region: Optional[str] = Field(None, description="Energy market region")


# Schemas for forecasting
class ForecastRecord(BaseModel):
    """Single forecast record comparing actual vs predicted"""
    timestamp: str = Field(description="Timestamp of the forecast")
    actual: float = Field(description="Actual observed value")
    predicted: float = Field(description="Predicted value from model")
    error: float = Field(description="Prediction error (predicted - actual)")

class ForecastMetrics(BaseModel):
    """Forecast quality metrics"""
    mse: float = Field(description="Mean Squared Error")
    rmse: float = Field(description="Root Mean Squared Error")
    mae: float = Field(description="Mean Absolute Error")
    num_predictions: int = Field(description="Number of predictions made")

class ForecastResult(BaseModel):
    """Complete forecast result for a target variable"""
    region: str = Field(description="Energy market region")
    target: str = Field(description="Target variable (prices or consumption)")
    start_date: str = Field(description="Forecast start date")
    end_date: str = Field(description="Forecast end date")
    lookback: int = Field(description="Number of historical points used")
    horizon: int = Field(description="Forecast horizon length")
    metrics: ForecastMetrics = Field(description="Forecast quality metrics")
    forecasts: List[ForecastRecord] = Field(description="Individual forecast records")


# Schemas for heuristic trading
class TradingRecord(BaseModel):
    """Single trading record with action and profit"""
    timestamp: str = Field(description="Timestamp of the trade")
    hour: int = Field(description="Hour of day (0-23)")
    action: str = Field(description="Trading action (charge/discharge/idle)")
    energy: float = Field(description="Energy charged or discharged (MWh)")
    price: float = Field(description="Energy price at timestamp")
    soc: float = Field(description="State of charge after action (MWh)")
    profit: float = Field(description="Profit/loss from this action (negative for charging)")

class TradingMetrics(BaseModel):
    """Trading performance metrics"""
    total_profit: float = Field(description="Total profit over trading period")
    total_energy_charged: float = Field(description="Total energy charged (MWh)")
    total_energy_discharged: float = Field(description="Total energy discharged (MWh)")
    total_charge_cost: float = Field(description="Total cost of charging")
    total_discharge_revenue: float = Field(description="Total revenue from discharging")
    num_charges: int = Field(description="Number of charging actions")
    num_discharges: int = Field(description="Number of discharging actions")
    avg_charge_price: float = Field(description="Average charging price")
    avg_discharge_price: float = Field(description="Average discharging price")
    initial_soc: float = Field(description="Initial state of charge (MWh)")
    final_soc: float = Field(description="Final state of charge (MWh)")

class TradingResult(BaseModel):
    """Complete trading result for a period"""
    region: str = Field(description="Energy market region")
    start_date: str = Field(description="Trading start date")
    end_date: str = Field(description="Trading end date")
    storage_capacity: float = Field(description="Storage capacity (MWh)")
    round_trip_efficiency: float = Field(description="Round-trip efficiency")
    metrics: TradingMetrics = Field(description="Trading performance metrics")
    trades: List[TradingRecord] = Field(description="Individual trading records")



# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict

# class MetricStats(BaseModel):
#     count: Optional[int] = Field(None, description="Number of valid data points")
#     min: Optional[float] = Field(None, description="Minimum value")
#     max: Optional[float] = Field(None, description="Maximum value")
#     avg: Optional[float] = Field(None, description="Average value")
#     median: Optional[float] = Field(None, description="Median value")
#     p25: Optional[float] = Field(None, description="25th percentile")
#     p75: Optional[float] = Field(None, description="75th percentile")
#     std: Optional[float] = Field(None, description="Standard deviation")
#     var: Optional[float] = Field(None, description="Variance")

# class DateRange(BaseModel):
#     start: Optional[str]
#     end: Optional[str]

# class SummaryStats(BaseModel):
#     region: str
#     total_records: int
#     date_range: DateRange
#     prices: Optional[MetricStats]
#     consumption: Optional[MetricStats]

# class EnergyDataRecord(BaseModel):
#     """Base energy data record with common fields across all regions"""
#     timestamps: str = Field(description="Timestamp in ISO format")
#     prices: Optional[float] = Field(None, description="Energy price at timestamp")
#     consumption: Optional[float] = Field(None, description="Energy consumption")
#     year: Optional[int] = Field(None, description="Year extracted from timestamp")
#     region: Optional[str] = Field(None, description="Energy market region")


# # Schemas for forecasting
# class ForecastRecord(BaseModel):
#     """Single forecast record comparing actual vs predicted"""
#     timestamp: str = Field(description="Timestamp of the forecast")
#     actual: float = Field(description="Actual observed value")
#     predicted: float = Field(description="Predicted value from model")
#     error: float = Field(description="Prediction error (predicted - actual)")

# class ForecastMetrics(BaseModel):
#     """Forecast quality metrics"""
#     mse: float = Field(description="Mean Squared Error")
#     rmse: float = Field(description="Root Mean Squared Error")
#     mae: float = Field(description="Mean Absolute Error")
#     num_predictions: int = Field(description="Number of predictions made")

# class ForecastResult(BaseModel):
#     """Complete forecast result for a target variable"""
#     region: str = Field(description="Energy market region")
#     target: str = Field(description="Target variable (prices or consumption)")
#     start_date: str = Field(description="Forecast start date")
#     end_date: str = Field(description="Forecast end date")
#     lookback: int = Field(description="Number of historical points used")
#     horizon: int = Field(description="Forecast horizon length")
#     metrics: ForecastMetrics = Field(description="Forecast quality metrics")
#     forecasts: List[ForecastRecord] = Field(description="Individual forecast records")


# # Schemas for heuristic trading
# class TradingRecord(BaseModel):
#     """Single trading record with action and profit"""
#     timestamp: str = Field(description="Timestamp of the trade")
#     hour: int = Field(description="Hour of day (0-23)")
#     action: str = Field(description="Trading action (charge/discharge/idle)")
#     energy: float = Field(description="Energy charged or discharged (MWh)")
#     price: float = Field(description="Energy price at timestamp")
#     soc: float = Field(description="State of charge after action (MWh)")
#     profit: float = Field(description="Profit/loss from this action (negative for charging)")

# class TradingMetrics(BaseModel):
#     """Trading performance metrics"""
#     total_profit: float = Field(description="Total profit over trading period")
#     total_energy_charged: float = Field(description="Total energy charged (MWh)")
#     total_energy_discharged: float = Field(description="Total energy discharged (MWh)")
#     total_charge_cost: float = Field(description="Total cost of charging")
#     total_discharge_revenue: float = Field(description="Total revenue from discharging")
#     num_charges: int = Field(description="Number of charging actions")
#     num_discharges: int = Field(description="Number of discharging actions")
#     avg_charge_price: float = Field(description="Average charging price")
#     avg_discharge_price: float = Field(description="Average discharging price")
#     final_soc: float = Field(description="Final state of charge (MWh)")

# class TradingResult(BaseModel):
#     """Complete trading result for a period"""
#     region: str = Field(description="Energy market region")
#     start_date: str = Field(description="Trading start date")
#     end_date: str = Field(description="Trading end date")
#     storage_capacity: float = Field(description="Storage capacity (MWh)")
#     round_trip_efficiency: float = Field(description="Round-trip efficiency")
#     metrics: TradingMetrics = Field(description="Trading performance metrics")
#     trades: List[TradingRecord] = Field(description="Individual trading records")


# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict

# class MetricStats(BaseModel):
#     count: Optional[int] = Field(None, description="Number of valid data points")
#     min: Optional[float] = Field(None, description="Minimum value")
#     max: Optional[float] = Field(None, description="Maximum value")
#     avg: Optional[float] = Field(None, description="Average value")
#     median: Optional[float] = Field(None, description="Median value")
#     p25: Optional[float] = Field(None, description="25th percentile")
#     p75: Optional[float] = Field(None, description="75th percentile")
#     std: Optional[float] = Field(None, description="Standard deviation")
#     var: Optional[float] = Field(None, description="Variance")

# class DateRange(BaseModel):
#     start: Optional[str]
#     end: Optional[str]

# class SummaryStats(BaseModel):
#     region: str
#     total_records: int
#     date_range: DateRange
#     prices: Optional[MetricStats]
#     consumption: Optional[MetricStats]

# class EnergyDataRecord(BaseModel):
#     """Base energy data record with common fields across all regions"""
#     timestamps: str = Field(description="Timestamp in ISO format")
#     prices: Optional[float] = Field(None, description="Energy price at timestamp")
#     consumption: Optional[float] = Field(None, description="Energy consumption")
#     year: Optional[int] = Field(None, description="Year extracted from timestamp")
#     region: Optional[str] = Field(None, description="Energy market region")


# # New schemas for forecasting
# class ForecastRecord(BaseModel):
#     """Single forecast record comparing actual vs predicted"""
#     timestamp: str = Field(description="Timestamp of the forecast")
#     actual: float = Field(description="Actual observed value")
#     predicted: float = Field(description="Predicted value from model")
#     error: float = Field(description="Prediction error (predicted - actual)")

# class ForecastMetrics(BaseModel):
#     """Forecast quality metrics"""
#     mse: float = Field(description="Mean Squared Error")
#     rmse: float = Field(description="Root Mean Squared Error")
#     mae: float = Field(description="Mean Absolute Error")
#     num_predictions: int = Field(description="Number of predictions made")

# class ForecastResult(BaseModel):
#     """Complete forecast result for a target variable"""
#     region: str = Field(description="Energy market region")
#     target: str = Field(description="Target variable (prices or consumption)")
#     start_date: str = Field(description="Forecast start date")
#     end_date: str = Field(description="Forecast end date")
#     lookback: int = Field(description="Number of historical points used")
#     horizon: int = Field(description="Forecast horizon length")
#     metrics: ForecastMetrics = Field(description="Forecast quality metrics")
#     forecasts: List[ForecastRecord] = Field(description="Individual forecast records")



# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict

# class MetricStats(BaseModel):
#     count: Optional[int] = Field(None, description="Number of valid data points")
#     min: Optional[float] = Field(None, description="Minimum value")
#     max: Optional[float] = Field(None, description="Maximum value")
#     avg: Optional[float] = Field(None, description="Average value")
#     median: Optional[float] = Field(None, description="Median value")
#     p25: Optional[float] = Field(None, description="25th percentile")
#     p75: Optional[float] = Field(None, description="75th percentile")
#     std: Optional[float] = Field(None, description="Standard deviation")
#     var: Optional[float] = Field(None, description="Variance")

# class DateRange(BaseModel):
#     start: Optional[str]
#     end: Optional[str]

# class SummaryStats(BaseModel):
#     region: str
#     total_records: int
#     date_range: DateRange
#     prices: Optional[MetricStats]
#     consumption: Optional[MetricStats]

# class EnergyDataRecord(BaseModel):
#     """Base energy data record with common fields across all regions"""
#     timestamps: str = Field(description="Timestamp in ISO format")
#     prices: Optional[float] = Field(None, description="Energy price at timestamp")
#     consumption: Optional[float] = Field(None, description="Energy consumption")
#     year: Optional[int] = Field(None, description="Year extracted from timestamp")
#     region: Optional[str] = Field(None, description="Energy market region")