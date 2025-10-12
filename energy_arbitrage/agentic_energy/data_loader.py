"""
Energy Data Loader using Agentics Framework

This module provides utilities to load and process energy market data from various regions
(CAISO, ERCOT, Germany, Italy, NewYork) using the Agentics framework for structured data handling.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# from agentics import agentics as AG
from agentics.core.agentics import AG

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union
import pandas as pd
from datetime import datetime
from .schemas import EnergyDataRecord, MetricStats, SummaryStats, DateRange
import numpy as np


class EnergyDataLoader:
    """
    Energy Data Loader using Agentics framework for structured energy market data loading
    """
    
    def __init__(self, region: str, data_dir: Union[str, Path] = None):
        """
        Initialize the data loader
        
        Args:
            region: Region name (CAISO, ERCOT, GERMANY, ITALY, NEWYORK)
            data_dir: Path to the directory containing CSV files
        """
        if data_dir is None:
            # Default to current directory's data folder
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.region = region.upper()
        self.data = None
        self.available_regions = {
            "CAISO": "CAISO_data.csv",
            "ERCOT": "Ercot_energy_data.csv", 
            "GERMANY": "Germany_energy_Data.csv",
            "ITALY": "Italy_data.csv",
            "NEWYORK": "NewYork_energy_data.csv"
        }
    
    def load_region_data(self) -> AG:
        """
        Load data for a specific region using Agentics
            
        Raises:
            ValueError: If region is not supported
            FileNotFoundError: If data file doesn't exist
        """        
        if self.region not in self.available_regions:
            raise ValueError(f"Region {self.region} not supported. Available: {list(self.available_regions.keys())}")
        
        file_path = self.data_dir / self.available_regions[self.region]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data using Agentics
        energy_data = AG.from_csv(file_path, atype=EnergyDataRecord)
        
        # Add region information to each record
        for state in energy_data.states:
            state.region = self.region
            
        self.data = energy_data
        return self.data

    async def get_filtered_data(self, start_date: str = None, end_date: str = None, 
                                price_range: tuple = None) -> AG:
        """
        Filter region data asynchronously using Agentics areduce (no amap needed).
        """

        async def _filter_reduce(states: list):
            """Reduce function: keep only states that meet date/price filters."""
            filtered = []
            for state in states:
                try:
                    # --- Date filter ---
                    if start_date or end_date:
                        record_date = pd.to_datetime(state.timestamps).date()
                        if start_date and record_date < pd.to_datetime(start_date).date():
                            continue
                        if end_date and record_date > pd.to_datetime(end_date).date():
                            continue

                    # --- Price filter ---
                    if price_range and state.prices is not None:
                        min_price, max_price = price_range
                        if state.prices < min_price or state.prices > max_price:
                            continue

                    filtered.append(state)
                except Exception:
                    continue  # skip invalid records

            return filtered

        # apply areduce directly to the dataset
        filtered_states = await self.data.areduce(_filter_reduce)

        # construct a new Agentics group
        return AG(atype=EnergyDataRecord, states=filtered_states)

    @staticmethod
    async def get_summary_stats_from_ag(ag_data: AG, column: Optional[str] = None) -> SummaryStats | Dict:
        """
        Compute summary statistics (min, max, avg, median, percentiles, std, var)
        and return as Agentics Object for (SummaryStats).
        """
        prices = np.array([s.prices for s in ag_data.states if s.prices is not None], dtype=float)
        consumption = np.array([s.consumption for s in ag_data.states if s.consumption is not None], dtype=float)
        timestamps = [s.timestamps for s in ag_data.states if getattr(s, "timestamps", None)]

        async def summarize(arr: np.ndarray) -> MetricStats:
            if arr.size == 0:
                return MetricStats()
            return MetricStats(
                count=int(arr.size),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                avg=float(np.mean(arr)),
                median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                std=float(np.std(arr)),
                var=float(np.var(arr))
            )

        stats_obj = SummaryStats(
            region=ag_data[0].region,
            total_records=len(ag_data.states),
            date_range=DateRange(
                start=min(timestamps) if timestamps else None,
                end=max(timestamps) if timestamps else None
            ),
            prices=await summarize(prices),
            consumption=await summarize(consumption)
        )

        if column:
            if column not in ["prices", "consumption"]:
                raise ValueError("Column must be 'prices' or 'consumption'.")
            return  AG(atype = MetricStats, states=[getattr(stats_obj, column)])
        
        #create an agentic object for stats_obj
        return AG(atype=SummaryStats, states=[stats_obj])

    
    # async def get_summary_stats_from_ag(ag_data: AG, column: Optional[str] = None) -> SummaryStats | Dict:
    #     """
    #     Compute summary statistics (min, max, avg, median, percentiles, std, var)
    #     and return as Pydantic schema (SummaryStats).
    #     """

    #     # source = AG(
    #     #     atype = EnergyDataRecord,
    #     #     verbose_agent = True
    #     #     state = ag_data.states
    #     # )
    #     if column:
    #         answer = await(
    #             AG(
    #                 atype = SummaryStats,
    #                 verbose_agent = True,
    #                 instructions = f"Compute summary statistics for the '{column}' column only. "
    #             ) 
    #             << ag_data(column)
    #         )
    #         return answer
    #     else:
    #         answer = await(
    #             AG(
    #                 atype = SummaryStats,
    #                 verbose_agent = True,
    #                 instructions = "Compute summary statistics for all relevant columns."
    #             ) 
    #             << ag_data
    #         )
    #         return answer
        
    