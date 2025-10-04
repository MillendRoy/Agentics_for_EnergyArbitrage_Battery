"""
Energy Data Loader using Agentics Framework

This module provides utilities to load and process energy market data from various regions
(CAISO, ERCOT, Germany, Italy, NewYork) using the Agentics framework for structured data handling.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from agentics import Agentics as AG
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union
import pandas as pd
from datetime import datetime
from .schemas import EnergyDataRecord


class EnergyDataLoader:
    """
    Energy Data Loader using Agentics framework for structured energy market data loading
    """
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to the directory containing CSV files
        """
        if data_dir is None:
            # Default to current directory's data folder
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.available_regions = {
            "CAISO": "CAISO_data.csv",
            "ERCOT": "Ercot_energy_data.csv", 
            "GERMANY": "Germany_energy_Data.csv",
            "ITALY": "Italy_data.csv",
            "NEWYORK": "NewYork_energy_data.csv"
        }
    
    def load_region_data(self, region: str) -> AG:
        """
        Load data for a specific region using Agentics
        
        Args:
            region: Region name (CAISO, ERCOT, GERMANY, ITALY, NEWYORK)
            
        Returns:
            Agentics object containing energy data records
            
        Raises:
            ValueError: If region is not supported
            FileNotFoundError: If data file doesn't exist
        """
        region = region.upper()
        
        if region not in self.available_regions:
            raise ValueError(f"Region {region} not supported. Available: {list(self.available_regions.keys())}")
        
        file_path = self.data_dir / self.available_regions[region]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data using Agentics
        energy_data = AG.from_csv(file_path, atype=EnergyDataRecord)
        
        # Add region information to each record
        for state in energy_data.states:
            state.region = region
            
        return energy_data
    
    def load_all_regions(self) -> Dict[str, AG]:
        """
        Load data for all available regions
        
        Returns:
            Dictionary mapping region names to their Agentics data objects
        """
        all_data = {}
        
        for region in self.available_regions.keys():
            try:
                all_data[region] = self.load_region_data(region)
                print(f"✓ Loaded {region}: {len(all_data[region])} records")
            except (FileNotFoundError, Exception) as e:
                print(f"✗ Failed to load {region}: {e}")
                
        return all_data
    
    async def get_filtered_data(self, region: str, start_date: str = None, end_date: str = None, 
                                price_range: tuple = None) -> AG:
        """
        Filter region data asynchronously using Agentics areduce (no amap needed).
        """
        data = self.load_region_data(region)

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
        filtered_states = await data.areduce(_filter_reduce)

        # construct a new Agentics group
        return AG(atype=EnergyDataRecord, states=filtered_states)
    
    def get_summary_stats(self, region: str) -> Dict:
        """
        Get summary statistics for a region's data
        
        Args:
            region: Region name
            
        Returns:
            Dictionary with summary statistics
        """
        data = self.load_region_data(region)
        
        prices = [state.prices for state in data.states if state.prices is not None]
        consumption = [state.consumption for state in data.states if state.consumption is not None]
        
        stats = {
            "region": region,
            "total_records": len(data),
            "date_range": {
                "start": min([state.timestamps for state in data.states]),
                "end": max([state.timestamps for state in data.states])
            },
            "prices": {
                "count": len(prices),
                "min": min(prices) if prices else None,
                "max": max(prices) if prices else None,
                "avg": sum(prices) / len(prices) if prices else None
            },
            "consumption": {
                "count": len(consumption),
                "min": min(consumption) if consumption else None,
                "max": max(consumption) if consumption else None,
                "avg": sum(consumption) / len(consumption) if consumption else None
            }
        }
        
        return stats


# Convenience functions for direct usage
def load_caiso_data() -> AG:
    """Load CAISO energy data"""
    loader = EnergyDataLoader()
    return loader.load_region_data("CAISO")


def load_ercot_data() -> AG:
    """Load ERCOT energy data"""
    loader = EnergyDataLoader()
    return loader.load_region_data("ERCOT")


def load_all_energy_data() -> Dict[str, AG]:
    """Load all available energy data"""
    loader = EnergyDataLoader()
    return loader.load_all_regions()


def get_energy_stats(region: str) -> Dict:
    """Get summary statistics for a region"""
    loader = EnergyDataLoader()
    return loader.get_summary_stats(region)


# Example usage and testing
if __name__ == "__main__":
    print("Energy Data Loader - Agentics Framework")
    print("=" * 50)
    
    # Initialize loader
    loader = EnergyDataLoader()
    
    # Test loading CAISO data
    print("\n1. Loading CAISO data...")
    try:
        caiso = load_caiso_data()
        print(f"✓ Loaded CAISO: {len(caiso)} records")
        print(f"First record: {caiso[0]}")
    except Exception as e:
        print(f"✗ Error loading CAISO: {e}")
    
    # Test loading all regions
    print("\n2. Loading all regions...")
    all_data = load_all_energy_data()
    
    # Print summary statistics
    print("\n3. Summary statistics:")
    for region in ["CAISO", "ERCOT"]:
        try:
            stats = get_energy_stats(region)
            print(f"\n{region}:")
            print(f"  Records: {stats['total_records']}")
            print(f"  Price range: ${stats['prices']['min']:.2f} - ${stats['prices']['max']:.2f}")
            if stats['consumption']['avg']:
                print(f"  Avg consumption: {stats['consumption']['avg']:.2f}")
        except Exception as e:
            print(f"  Error getting stats for {region}: {e}")
