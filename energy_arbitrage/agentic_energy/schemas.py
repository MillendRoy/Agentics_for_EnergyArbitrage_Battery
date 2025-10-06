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


# New schemas for forecasting
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