"""
Heuristic Storage Trading Engine using Agentics Framework

This module provides utilities to simulate heuristic-based energy storage trading
with predefined charge/discharge windows and profit calculation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
from datetime import datetime, timedelta

from agentics.core.agentics import AG
from .schemas import EnergyDataRecord, TradingRecord, TradingMetrics, TradingResult


class HeuristicTrader:
    """
    Heuristic-based energy storage trading simulator
    
    Trading Rules:
    - 00:00-01:59: Discharge
    - 02:00-05:59: Charge
    - 06:00-09:59: Discharge
    - 10:00-15:59: Charge
    - 16:00-19:59: Discharge
    - 20:00-21:59: Charge
    - 22:00-23:59: Discharge
    
    Initial SOC: 50% of capacity (0.5 * storage_capacity)
    """
    
    def __init__(
        self,
        storage_capacity: float = 1.0,  # MWh
        round_trip_efficiency: float = 0.95,
        max_charge_rate: float = 1.0,  # MW (can charge/discharge full capacity in 1 hour)
        max_discharge_rate: float = 1.0  # MW
    ):
        """
        Initialize the heuristic trader
        
        Args:
            storage_capacity: Storage capacity in MWh (default: 1.0)
            round_trip_efficiency: Round-trip efficiency (default: 0.95)
            max_charge_rate: Maximum charge rate in MW (default: 1.0)
            max_discharge_rate: Maximum discharge rate in MW (default: 1.0)
        """
        self.storage_capacity = storage_capacity
        self.efficiency = round_trip_efficiency
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        
        # Define trading windows (hour ranges)
        self.charge_windows = [(2, 6), (10, 16), (20, 22)]
        self.discharge_windows = [(0, 2), (6, 10), (16, 20), (22, 24)]
    
    def _get_action(self, hour: int) -> str:
        """
        Determine action based on hour of day
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            'charge', 'discharge', or 'idle'
        """
        for start, end in self.charge_windows:
            if start <= hour < end:
                return 'charge'
        
        for start, end in self.discharge_windows:
            if start <= hour < end:
                return 'discharge'
        
        return 'idle'
    
    def _calculate_charge_energy(
        self,
        current_soc: float,
        price: float,
        duration_hours: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate energy to charge and new SOC
        
        Args:
            current_soc: Current state of charge (MWh)
            price: Current price
            duration_hours: Duration in hours (default: 1.0)
            
        Returns:
            (energy_charged, new_soc)
        """
        # Maximum energy we can charge
        max_charge = min(
            self.max_charge_rate * duration_hours,
            self.storage_capacity - current_soc
        )
        
        # Energy charged (accounting for efficiency on input)
        energy_charged = max_charge
        
        # New SOC (efficiency applied when energy goes into storage)
        new_soc = current_soc + (energy_charged * self.efficiency)
        
        return energy_charged, new_soc
    
    def _calculate_discharge_energy(
        self,
        current_soc: float,
        price: float,
        duration_hours: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate energy to discharge and new SOC
        
        Args:
            current_soc: Current state of charge (MWh)
            price: Current price
            duration_hours: Duration in hours (default: 1.0)
            
        Returns:
            (energy_discharged, new_soc)
        """
        # Maximum energy we can discharge (limited by SOC)
        max_discharge = min(
            self.max_discharge_rate * duration_hours,
            current_soc
        )
        
        # Energy discharged
        energy_discharged = max_discharge
        
        # New SOC
        new_soc = current_soc - energy_discharged
        
        return energy_discharged, new_soc
    
    async def simulate_trading(
        self,
        ag_data: AG,
        start_date: str,
        end_date: str
    ) -> AG:
        """
        Simulate heuristic trading over specified period
        
        Args:
            ag_data: Agentics object with energy data (must have timestamps and prices)
            start_date: Start date for trading (YYYY-MM-DD)
            end_date: End date for trading (YYYY-MM-DD)
            
        Returns:
            AG object with TradingResult
            
        Note:
            - Initial SOC is set to 50% of storage capacity
            - SOC is stitched across consecutive days within the trading period
        """
        # Get region
        region = ag_data.states[0].region if ag_data.states else "UNKNOWN"
        
        # Convert data to dataframe
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(s.timestamps),
                'prices': s.prices
            }
            for s in ag_data.states
        ]).sort_values('timestamp')
        
        # Filter to trading period
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
        
        trading_df = df[
            (df['timestamp'] >= start_dt) & 
            (df['timestamp'] <= end_dt)
        ].copy()
        
        if len(trading_df) == 0:
            raise ValueError(f"No data available for period {start_date} to {end_date}")
        
        print(f"\nSimulating heuristic trading for {region}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Total data points: {len(trading_df)}")
        print(f"Storage capacity: {self.storage_capacity} MWh")
        print(f"Round-trip efficiency: {self.efficiency * 100}%")
        print(f"Initial SOC: {self.storage_capacity * 0.5} MWh (50%)")
        
        # Initialize tracking variables
        soc = self.storage_capacity * 0.5  # Start at 50% capacity
        total_profit = 0.0
        total_energy_charged = 0.0
        total_energy_discharged = 0.0
        total_charge_cost = 0.0
        total_discharge_revenue = 0.0
        
        trading_records = []
        
        # Simulate trading for each timestamp
        for idx, row in trading_df.iterrows():
            timestamp = row['timestamp']
            price = row['prices']
            hour = timestamp.hour
            
            # Determine action
            action = self._get_action(hour)
            
            energy = 0.0
            profit = 0.0
            
            if action == 'charge':
                # Try to charge
                energy_charged, new_soc = self._calculate_charge_energy(soc, price)
                
                if energy_charged > 0:
                    cost = energy_charged * price
                    total_charge_cost += cost
                    total_energy_charged += energy_charged
                    profit = -cost  # Negative profit (it's a cost)
                    soc = new_soc
                    energy = energy_charged
                else:
                    action = 'idle'  # Can't charge more
            
            elif action == 'discharge':
                # Try to discharge
                energy_discharged, new_soc = self._calculate_discharge_energy(soc, price)
                
                if energy_discharged > 0:
                    revenue = energy_discharged * price
                    total_discharge_revenue += revenue
                    total_energy_discharged += energy_discharged
                    profit = revenue  # Positive profit
                    soc = new_soc
                    energy = energy_discharged
                else:
                    action = 'idle'  # Can't discharge more
            
            # Update total profit
            total_profit += profit
            
            # Record the trade
            trading_records.append(TradingRecord(
                timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                hour=hour,
                action=action,
                energy=float(energy),
                price=float(price),
                soc=float(soc),
                profit=float(profit)
            ))
        
        # Calculate metrics
        num_charges = sum(1 for r in trading_records if r.action == 'charge')
        num_discharges = sum(1 for r in trading_records if r.action == 'discharge')
        
        avg_charge_price = (
            total_charge_cost / total_energy_charged 
            if total_energy_charged > 0 else 0.0
        )
        avg_discharge_price = (
            total_discharge_revenue / total_energy_discharged 
            if total_energy_discharged > 0 else 0.0
        )
        
        metrics = TradingMetrics(
            total_profit=float(total_profit),
            total_energy_charged=float(total_energy_charged),
            total_energy_discharged=float(total_energy_discharged),
            total_charge_cost=float(total_charge_cost),
            total_discharge_revenue=float(total_discharge_revenue),
            num_charges=num_charges,
            num_discharges=num_discharges,
            avg_charge_price=float(avg_charge_price),
            avg_discharge_price=float(avg_discharge_price),
            initial_soc=float(self.storage_capacity * 0.5),
            final_soc=float(soc)
        )
        
        # Create result
        result = TradingResult(
            region=region,
            start_date=start_date,
            end_date=end_date,
            storage_capacity=self.storage_capacity,
            round_trip_efficiency=self.efficiency,
            metrics=metrics,
            trades=trading_records
        )
        
        print(f"\n=== Trading Summary ===")
        print(f"Initial SOC: {self.storage_capacity * 0.5:.2f} MWh (50%)")
        print(f"Final SOC: {soc:.2f} MWh")
        print(f"Total Profit: ${total_profit:,.2f}")
        print(f"Total Energy Charged: {total_energy_charged:,.2f} MWh")
        print(f"Total Energy Discharged: {total_energy_discharged:,.2f} MWh")
        print(f"Avg Charge Price: ${avg_charge_price:.2f}/MWh")
        print(f"Avg Discharge Price: ${avg_discharge_price:.2f}/MWh")
        
        # Return as Agentics object
        return AG(atype=TradingResult, states=[result])
    
    async def simulate_trading_for_day(
        self,
        ag_data: AG,
        date: str
    ) -> AG:
        """
        Simulate trading for a specific day
        
        Args:
            ag_data: Agentics object with energy data
            date: Date to trade (YYYY-MM-DD)
            
        Returns:
            AG object with TradingResult
        """
        return await self.simulate_trading(ag_data, date, date)


# """
# Heuristic Storage Trading Engine using Agentics Framework

# This module provides utilities to simulate heuristic-based energy storage trading
# with predefined charge/discharge windows and profit calculation.
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Optional, List, Tuple, Dict, Union
# from datetime import datetime, timedelta

# from agentics.core.agentics import AG
# from .schemas import EnergyDataRecord, TradingRecord, TradingMetrics, TradingResult


# class HeuristicTrader:
#     """
#     Heuristic-based energy storage trading simulator
    
#     Trading Rules:
#     - 00:00-05:59: Charge
#     - 06:00-09:59: Discharge
#     - 10:00-15:59: Charge
#     - 16:00-19:59: Discharge
#     - 20:00-21:59: Charge
#     - 22:00-23:59: Discharge
#     """
    
#     def __init__(
#         self,
#         storage_capacity: float = 1.0,  # MWh
#         round_trip_efficiency: float = 0.95,
#         max_charge_rate: float = 1.0,  # MW (can charge/discharge full capacity in 1 hour)
#         max_discharge_rate: float = 1.0  # MW
#     ):
#         """
#         Initialize the heuristic trader
        
#         Args:
#             storage_capacity: Storage capacity in MWh (default: 1.0)
#             round_trip_efficiency: Round-trip efficiency (default: 0.95)
#             max_charge_rate: Maximum charge rate in MW (default: 1.0)
#             max_discharge_rate: Maximum discharge rate in MW (default: 1.0)
#         """
#         self.storage_capacity = storage_capacity
#         self.efficiency = round_trip_efficiency
#         self.max_charge_rate = max_charge_rate
#         self.max_discharge_rate = max_discharge_rate
        
#         # Define trading windows (hour ranges)
#         self.charge_windows = [(0, 6), (10, 16), (20, 22)]
#         self.discharge_windows = [(6, 10), (16, 20), (22, 24)]
    
#     def _get_action(self, hour: int) -> str:
#         """
#         Determine action based on hour of day
        
#         Args:
#             hour: Hour of day (0-23)
            
#         Returns:
#             'charge', 'discharge', or 'idle'
#         """
#         for start, end in self.charge_windows:
#             if start <= hour < end:
#                 return 'charge'
        
#         for start, end in self.discharge_windows:
#             if start <= hour < end:
#                 return 'discharge'
        
#         return 'idle'
    
#     def _calculate_charge_energy(
#         self,
#         current_soc: float,
#         price: float,
#         duration_hours: float = 1.0
#     ) -> Tuple[float, float]:
#         """
#         Calculate energy to charge and new SOC
        
#         Args:
#             current_soc: Current state of charge (MWh)
#             price: Current price
#             duration_hours: Duration in hours (default: 1.0)
            
#         Returns:
#             (energy_charged, new_soc)
#         """
#         # Maximum energy we can charge
#         max_charge = min(
#             self.max_charge_rate * duration_hours,
#             self.storage_capacity - current_soc
#         )
        
#         # Energy charged (accounting for efficiency on input)
#         energy_charged = max_charge
        
#         # New SOC (efficiency applied when energy goes into storage)
#         new_soc = current_soc + (energy_charged * self.efficiency)
        
#         return energy_charged, new_soc
    
#     def _calculate_discharge_energy(
#         self,
#         current_soc: float,
#         price: float,
#         duration_hours: float = 1.0
#     ) -> Tuple[float, float]:
#         """
#         Calculate energy to discharge and new SOC
        
#         Args:
#             current_soc: Current state of charge (MWh)
#             price: Current price
#             duration_hours: Duration in hours (default: 1.0)
            
#         Returns:
#             (energy_discharged, new_soc)
#         """
#         # Maximum energy we can discharge (limited by SOC)
#         max_discharge = min(
#             self.max_discharge_rate * duration_hours,
#             current_soc
#         )
        
#         # Energy discharged
#         energy_discharged = max_discharge
        
#         # New SOC
#         new_soc = current_soc - energy_discharged
        
#         return energy_discharged, new_soc
    
#     async def simulate_trading(
#         self,
#         ag_data: AG,
#         start_date: str,
#         end_date: str
#     ) -> AG:
#         """
#         Simulate heuristic trading over specified period
        
#         Args:
#             ag_data: Agentics object with energy data (must have timestamps and prices)
#             start_date: Start date for trading (YYYY-MM-DD)
#             end_date: End date for trading (YYYY-MM-DD)
            
#         Returns:
#             AG object with TradingResult
#         """
#         # Get region
#         region = ag_data.states[0].region if ag_data.states else "UNKNOWN"
        
#         # Convert data to dataframe
#         df = pd.DataFrame([
#             {
#                 'timestamp': pd.to_datetime(s.timestamps),
#                 'prices': s.prices
#             }
#             for s in ag_data.states
#         ]).sort_values('timestamp')
        
#         # Filter to trading period
#         start_dt = pd.to_datetime(start_date)
#         end_dt = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
        
#         trading_df = df[
#             (df['timestamp'] >= start_dt) & 
#             (df['timestamp'] <= end_dt)
#         ].copy()
        
#         if len(trading_df) == 0:
#             raise ValueError(f"No data available for period {start_date} to {end_date}")
        
#         print(f"\nSimulating heuristic trading for {region}")
#         print(f"Period: {start_date} to {end_date}")
#         print(f"Total data points: {len(trading_df)}")
#         print(f"Storage capacity: {self.storage_capacity} MWh")
#         print(f"Round-trip efficiency: {self.efficiency * 100}%")
        
#         # Initialize tracking variables
#         soc = 0.0  # Start discharged
#         total_profit = 0.0
#         total_energy_charged = 0.0
#         total_energy_discharged = 0.0
#         total_charge_cost = 0.0
#         total_discharge_revenue = 0.0
        
#         trading_records = []
        
#         # Simulate trading for each timestamp
#         for idx, row in trading_df.iterrows():
#             timestamp = row['timestamp']
#             price = row['prices']
#             hour = timestamp.hour
            
#             # Determine action
#             action = self._get_action(hour)
            
#             energy = 0.0
#             profit = 0.0
            
#             if action == 'charge':
#                 # Try to charge
#                 energy_charged, new_soc = self._calculate_charge_energy(soc, price)
                
#                 if energy_charged > 0:
#                     cost = energy_charged * price
#                     total_charge_cost += cost
#                     total_energy_charged += energy_charged
#                     profit = -cost  # Negative profit (it's a cost)
#                     soc = new_soc
#                     energy = energy_charged
#                 else:
#                     action = 'idle'  # Can't charge more
            
#             elif action == 'discharge':
#                 # Try to discharge
#                 energy_discharged, new_soc = self._calculate_discharge_energy(soc, price)
                
#                 if energy_discharged > 0:
#                     revenue = energy_discharged * price
#                     total_discharge_revenue += revenue
#                     total_energy_discharged += energy_discharged
#                     profit = revenue  # Positive profit
#                     soc = new_soc
#                     energy = energy_discharged
#                 else:
#                     action = 'idle'  # Can't discharge more
            
#             # Update total profit
#             total_profit += profit
            
#             # Record the trade
#             trading_records.append(TradingRecord(
#                 timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                 hour=hour,
#                 action=action,
#                 energy=float(energy),
#                 price=float(price),
#                 soc=float(soc),
#                 profit=float(profit)
#             ))
        
#         # Calculate metrics
#         num_charges = sum(1 for r in trading_records if r.action == 'charge')
#         num_discharges = sum(1 for r in trading_records if r.action == 'discharge')
        
#         avg_charge_price = (
#             total_charge_cost / total_energy_charged 
#             if total_energy_charged > 0 else 0.0
#         )
#         avg_discharge_price = (
#             total_discharge_revenue / total_energy_discharged 
#             if total_energy_discharged > 0 else 0.0
#         )
        
#         metrics = TradingMetrics(
#             total_profit=float(total_profit),
#             total_energy_charged=float(total_energy_charged),
#             total_energy_discharged=float(total_energy_discharged),
#             total_charge_cost=float(total_charge_cost),
#             total_discharge_revenue=float(total_discharge_revenue),
#             num_charges=num_charges,
#             num_discharges=num_discharges,
#             avg_charge_price=float(avg_charge_price),
#             avg_discharge_price=float(avg_discharge_price),
#             final_soc=float(soc)
#         )
        
#         # Create result
#         result = TradingResult(
#             region=region,
#             start_date=start_date,
#             end_date=end_date,
#             storage_capacity=self.storage_capacity,
#             round_trip_efficiency=self.efficiency,
#             metrics=metrics,
#             trades=trading_records
#         )
        
#         print(f"\n=== Trading Summary ===")
#         print(f"Total Profit: ${total_profit:,.2f}")
#         print(f"Total Energy Charged: {total_energy_charged:,.2f} MWh")
#         print(f"Total Energy Discharged: {total_energy_discharged:,.2f} MWh")
#         print(f"Avg Charge Price: ${avg_charge_price:.2f}/MWh")
#         print(f"Avg Discharge Price: ${avg_discharge_price:.2f}/MWh")
#         print(f"Final SOC: {soc:.2f} MWh")
        
#         # Return as Agentics object
#         return AG(atype=TradingResult, states=[result])
    
#     async def simulate_trading_for_day(
#         self,
#         ag_data: AG,
#         date: str
#     ) -> AG:
#         """
#         Simulate trading for a specific day
        
#         Args:
#             ag_data: Agentics object with energy data
#             date: Date to trade (YYYY-MM-DD)
            
#         Returns:
#             AG object with TradingResult
#         """
#         return await self.simulate_trading(ag_data, date, date)



# """
# Heuristic Storage Trading Engine using Agentics Framework

# This module provides utilities to simulate heuristic-based energy storage trading
# with predefined charge/discharge windows and profit calculation.
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Optional, List, Tuple, Dict, Union
# from datetime import datetime, timedelta

# from agentics.core.agentics import AG
# from .schemas import EnergyDataRecord, TradingRecord, TradingMetrics, TradingResult


# class HeuristicTrader:
#     """
#     Heuristic-based energy storage trading simulator
    
#     Trading Rules:
#     - 00:00-05:59: Charge
#     - 06:00-09:59: Discharge
#     - 10:00-15:59: Charge
#     - 16:00-19:59: Discharge
#     - 20:00-23:59: Charge
#     """
    
#     def __init__(
#         self,
#         storage_capacity: float = 1.0,  # MWh
#         round_trip_efficiency: float = 0.95,
#         max_charge_rate: float = 1.0,  # MW (can charge/discharge full capacity in 1 hour)
#         max_discharge_rate: float = 1.0  # MW
#     ):
#         """
#         Initialize the heuristic trader
        
#         Args:
#             storage_capacity: Storage capacity in MWh (default: 1.0)
#             round_trip_efficiency: Round-trip efficiency (default: 0.95)
#             max_charge_rate: Maximum charge rate in MW (default: 1.0)
#             max_discharge_rate: Maximum discharge rate in MW (default: 1.0)
#         """
#         self.storage_capacity = storage_capacity
#         self.efficiency = round_trip_efficiency
#         self.max_charge_rate = max_charge_rate
#         self.max_discharge_rate = max_discharge_rate
        
#         # Define trading windows (hour ranges)
#         self.charge_windows = [(0, 6), (10, 16), (20, 24)]
#         self.discharge_windows = [(6, 10), (16, 20)]
    
#     def _get_action(self, hour: int) -> str:
#         """
#         Determine action based on hour of day
        
#         Args:
#             hour: Hour of day (0-23)
            
#         Returns:
#             'charge', 'discharge', or 'idle'
#         """
#         for start, end in self.charge_windows:
#             if start <= hour < end:
#                 return 'charge'
        
#         for start, end in self.discharge_windows:
#             if start <= hour < end:
#                 return 'discharge'
        
#         return 'idle'
    
#     def _calculate_charge_energy(
#         self,
#         current_soc: float,
#         price: float,
#         duration_hours: float = 1.0
#     ) -> Tuple[float, float]:
#         """
#         Calculate energy to charge and new SOC
        
#         Args:
#             current_soc: Current state of charge (MWh)
#             price: Current price
#             duration_hours: Duration in hours (default: 1.0)
            
#         Returns:
#             (energy_charged, new_soc)
#         """
#         # Maximum energy we can charge
#         max_charge = min(
#             self.max_charge_rate * duration_hours,
#             self.storage_capacity - current_soc
#         )
        
#         # Energy charged (accounting for efficiency on input)
#         energy_charged = max_charge
        
#         # New SOC (efficiency applied when energy goes into storage)
#         new_soc = current_soc + (energy_charged * self.efficiency)
        
#         return energy_charged, new_soc
    
#     def _calculate_discharge_energy(
#         self,
#         current_soc: float,
#         price: float,
#         duration_hours: float = 1.0
#     ) -> Tuple[float, float]:
#         """
#         Calculate energy to discharge and new SOC
        
#         Args:
#             current_soc: Current state of charge (MWh)
#             price: Current price
#             duration_hours: Duration in hours (default: 1.0)
            
#         Returns:
#             (energy_discharged, new_soc)
#         """
#         # Maximum energy we can discharge (limited by SOC)
#         max_discharge = min(
#             self.max_discharge_rate * duration_hours,
#             current_soc
#         )
        
#         # Energy discharged
#         energy_discharged = max_discharge
        
#         # New SOC
#         new_soc = current_soc - energy_discharged
        
#         return energy_discharged, new_soc
    
#     async def simulate_trading(
#         self,
#         ag_data: AG,
#         start_date: str,
#         end_date: str
#     ) -> AG:
#         """
#         Simulate heuristic trading over specified period
        
#         Args:
#             ag_data: Agentics object with energy data (must have timestamps and prices)
#             start_date: Start date for trading (YYYY-MM-DD)
#             end_date: End date for trading (YYYY-MM-DD)
            
#         Returns:
#             AG object with TradingResult
#         """
#         # Get region
#         region = ag_data.states[0].region if ag_data.states else "UNKNOWN"
        
#         # Convert data to dataframe
#         df = pd.DataFrame([
#             {
#                 'timestamp': pd.to_datetime(s.timestamps),
#                 'prices': s.prices
#             }
#             for s in ag_data.states
#         ]).sort_values('timestamp')
        
#         # Filter to trading period
#         start_dt = pd.to_datetime(start_date)
#         end_dt = pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)
        
#         trading_df = df[
#             (df['timestamp'] >= start_dt) & 
#             (df['timestamp'] <= end_dt)
#         ].copy()
        
#         if len(trading_df) == 0:
#             raise ValueError(f"No data available for period {start_date} to {end_date}")
        
#         print(f"\nSimulating heuristic trading for {region}")
#         print(f"Period: {start_date} to {end_date}")
#         print(f"Total data points: {len(trading_df)}")
#         print(f"Storage capacity: {self.storage_capacity} MWh")
#         print(f"Round-trip efficiency: {self.efficiency * 100}%")
        
#         # Initialize tracking variables
#         soc = 0.0  # Start discharged
#         total_profit = 0.0
#         total_energy_charged = 0.0
#         total_energy_discharged = 0.0
#         total_charge_cost = 0.0
#         total_discharge_revenue = 0.0
        
#         trading_records = []
        
#         # Simulate trading for each timestamp
#         for idx, row in trading_df.iterrows():
#             timestamp = row['timestamp']
#             price = row['prices']
#             hour = timestamp.hour
            
#             # Determine action
#             action = self._get_action(hour)
            
#             energy = 0.0
#             profit = 0.0
            
#             if action == 'charge':
#                 # Try to charge
#                 energy_charged, new_soc = self._calculate_charge_energy(soc, price)
                
#                 if energy_charged > 0:
#                     cost = energy_charged * price
#                     total_charge_cost += cost
#                     total_energy_charged += energy_charged
#                     profit = -cost  # Negative profit (it's a cost)
#                     soc = new_soc
#                     energy = energy_charged
#                 else:
#                     action = 'idle'  # Can't charge more
            
#             elif action == 'discharge':
#                 # Try to discharge
#                 energy_discharged, new_soc = self._calculate_discharge_energy(soc, price)
                
#                 if energy_discharged > 0:
#                     revenue = energy_discharged * price
#                     total_discharge_revenue += revenue
#                     total_energy_discharged += energy_discharged
#                     profit = revenue  # Positive profit
#                     soc = new_soc
#                     energy = energy_discharged
#                 else:
#                     action = 'idle'  # Can't discharge more
            
#             # Update total profit
#             total_profit += profit
            
#             # Record the trade
#             trading_records.append(TradingRecord(
#                 timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
#                 hour=hour,
#                 action=action,
#                 energy=float(energy),
#                 price=float(price),
#                 soc=float(soc),
#                 profit=float(profit)
#             ))
        
#         # Calculate metrics
#         num_charges = sum(1 for r in trading_records if r.action == 'charge')
#         num_discharges = sum(1 for r in trading_records if r.action == 'discharge')
        
#         avg_charge_price = (
#             total_charge_cost / total_energy_charged 
#             if total_energy_charged > 0 else 0.0
#         )
#         avg_discharge_price = (
#             total_discharge_revenue / total_energy_discharged 
#             if total_energy_discharged > 0 else 0.0
#         )
        
#         metrics = TradingMetrics(
#             total_profit=float(total_profit),
#             total_energy_charged=float(total_energy_charged),
#             total_energy_discharged=float(total_energy_discharged),
#             total_charge_cost=float(total_charge_cost),
#             total_discharge_revenue=float(total_discharge_revenue),
#             num_charges=num_charges,
#             num_discharges=num_discharges,
#             avg_charge_price=float(avg_charge_price),
#             avg_discharge_price=float(avg_discharge_price),
#             final_soc=float(soc)
#         )
        
#         # Create result
#         result = TradingResult(
#             region=region,
#             start_date=start_date,
#             end_date=end_date,
#             storage_capacity=self.storage_capacity,
#             round_trip_efficiency=self.efficiency,
#             metrics=metrics,
#             trades=trading_records
#         )
        
#         print(f"\n=== Trading Summary ===")
#         print(f"Total Profit: ${total_profit:,.2f}")
#         print(f"Total Energy Charged: {total_energy_charged:,.2f} MWh")
#         print(f"Total Energy Discharged: {total_energy_discharged:,.2f} MWh")
#         print(f"Avg Charge Price: ${avg_charge_price:.2f}/MWh")
#         print(f"Avg Discharge Price: ${avg_discharge_price:.2f}/MWh")
#         print(f"Final SOC: {soc:.2f} MWh")
        
#         # Return as Agentics object
#         return AG(atype=TradingResult, states=[result])
    
#     async def simulate_trading_for_day(
#         self,
#         ag_data: AG,
#         date: str
#     ) -> AG:
#         """
#         Simulate trading for a specific day
        
#         Args:
#             ag_data: Agentics object with energy data
#             date: Date to trade (YYYY-MM-DD)
            
#         Returns:
#             AG object with TradingResult
#         """
#         return await self.simulate_trading(ag_data, date, date)