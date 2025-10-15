"""
Day-Specific Battery Optimization using LLM

This module optimizes battery storage operations for a specific day using
LLM-based reasoning, with support for forecasted or actual data.
"""

import os
import warnings
from dotenv import load_dotenv
from agentics.core.agentics import AG

from .schemas import (
    DayOptimizationRequest, SolveResponse, SolveRequest, 
    BatteryParams, DayInputs
)
from .day_data_loader import DayDataLoader

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


async def optimize_day(
    request: DayOptimizationRequest,
    data_dir: str = None
) -> SolveResponse:
    """Optimize battery operations for a specific day"""
    import sys
    from io import StringIO
    
    # Load data
    loader = DayDataLoader(data_dir=data_dir)
    day_inputs, metadata = loader.load_day_data(
        date=request.date,
        use_forecast=request.use_forecast,
        forecast_models=request.forecast_models,
        allow_export=request.allow_export,
        dt_hours=request.dt_hours
    )
    
    # Create solve request
    solve_req = SolveRequest(
        battery=request.battery,
        day=day_inputs,
        solver=request.solver,
        solver_opts=request.solver_opts
    )
    
    source = AG(atype=SolveRequest, states=[solve_req])
    
    instructions = _build_optimization_instructions(
        battery=request.battery,
        day_inputs=day_inputs,
        metadata=metadata
    )
    
    print(f"\n{'='*70}")
    print(f"Starting optimization for {metadata['date']}")
    print(f"Data source: {metadata['data_source']}")
    print(f"{'='*70}\n")
    
    # Redirect output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    result = None
    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        target = AG(
            atype=SolveResponse,
            max_iter=1,
            verbose_agent=True,
            reasoning=True,
            instructions=instructions
        )
        
        result = await (target << source)
        
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\n{'='*70}")
        print(f"✗ Error during optimization: {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        return _calculate_heuristic_solution(battery=request.battery, day_inputs=day_inputs, metadata=metadata)
    
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    # Extract response
    response = result.states[0] if result and result.states else None
    
    if response is None or response.status == "pending":
        print(f"\n{'='*70}")
        print(f"⚠️ LLM returned invalid response, using heuristic fallback")
        print(f"{'='*70}\n")
        return _calculate_heuristic_solution(battery=request.battery, day_inputs=day_inputs, metadata=metadata)
    
    # **POPULATE METADATA FIELDS**
    response.data_source = metadata["data_source"]
    if metadata.get("forecast_models"):
        response.forecast_info = metadata["forecast_models"]
    
    # **ADD PRICES AND DEMAND IF NOT PRESENT**
    if response.prices_buy is None:
        response.prices_buy = day_inputs.prices_buy
    if response.demand_kw is None:
        response.demand_kw = day_inputs.demand_kw
    
    print(f"\n{'='*70}")
    print(f"✓ Optimization successful")
    print(f"  Cost: €{response.objective_cost:,.2f}")
    print(f"  Total charge: {sum(response.charge_kw):.2f} kWh")
    print(f"  Total discharge: {sum(response.discharge_kw):.2f} kWh")
    print(f"{'='*70}\n")
    
    return response
# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
#     import sys
#     from io import StringIO
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     print(f"\n{'='*70}")
#     print(f"Starting optimization for {metadata['date']}")
#     print(f"Data source: {metadata['data_source']}")
#     print(f"{'='*70}\n")
    
#     # **CRITICAL: Redirect CrewAI output to prevent Jupyter recursion**
#     original_stdout = sys.stdout
#     original_stderr = sys.stderr
    
#     result = None
#     try:
#         # Redirect to string buffer to suppress verbose output
#         sys.stdout = StringIO()
#         sys.stderr = StringIO()
        
#         # NOW we can use verbose=True safely (which fixes the pydantic extraction)
#         target = AG(
#             atype=SolveResponse,
#             max_iter=1,
#             verbose_agent=True,  # ← Re-enabled! This fixes the pydantic issue
#             reasoning=True,
#             instructions=instructions
#         )
        
#         # Execute optimization
#         result = await (target << source)
        
#     except Exception as e:
#         # Restore output first
#         sys.stdout = original_stdout
#         sys.stderr = original_stderr
        
#         print(f"\n{'='*70}")
#         print(f"✗ Error during optimization: {type(e).__name__}: {e}")
#         print(f"{'='*70}\n")
        
#         # Return fallback
#         return _calculate_heuristic_solution(
#             battery=request.battery,
#             day_inputs=day_inputs,
#             metadata=metadata
#         )
    
#     finally:
#         # Always restore output
#         sys.stdout = original_stdout
#         sys.stderr = original_stderr
    
#     # Extract response
#     response = result.states[0] if result and result.states else None
    
#     if response is None or response.status == "pending":
#         print(f"\n{'='*70}")
#         print(f"⚠️ LLM returned invalid response, using heuristic fallback")
#         print(f"{'='*70}\n")
        
#         return _calculate_heuristic_solution(
#             battery=request.battery,
#             day_inputs=day_inputs,
#             metadata=metadata
#         )
    
#     # Add metadata
#     response.data_source = metadata["data_source"]
#     if metadata.get("forecast_models"):
#         response.forecast_info = metadata["forecast_models"]
    
#     print(f"\n{'='*70}")
#     print(f"✓ Optimization successful")
#     print(f"  Cost: €{response.objective_cost:.2f}")
#     print(f"{'='*70}\n")
    
#     return response

def _calculate_heuristic_solution(
    battery: BatteryParams,
    day_inputs: DayInputs,
    metadata: dict
) -> SolveResponse:
    """Calculate heuristic solution when LLM fails"""
    
    T = len(day_inputs.prices_buy)
    prices = day_inputs.prices_buy
    demand = day_inputs.demand_kw
    price_mean = sum(prices) / len(prices)
    
    charge_kw = []
    discharge_kw = []
    soc = [battery.soc_init]
    
    for t in range(T):
        current_soc = soc[-1]
        
        # Charge when cheap
        if prices[t] < price_mean * 0.9 and current_soc < battery.soc_max - 0.1:
            charge = min(
                battery.cmax_kw, 
                (battery.soc_max - current_soc) * battery.capacity_kwh / day_inputs.dt_hours
            )
            charge_kw.append(charge)
            discharge_kw.append(0.0)
            new_soc = current_soc + (battery.eta_c * charge * day_inputs.dt_hours) / battery.capacity_kwh
        
        # Discharge when expensive
        elif prices[t] > price_mean * 1.1 and current_soc > battery.soc_min + 0.1:
            max_discharge = min(
                battery.dmax_kw,
                (current_soc - battery.soc_min) * battery.capacity_kwh / day_inputs.dt_hours
            )
            discharge = min(max_discharge, demand[t])
            charge_kw.append(0.0)
            discharge_kw.append(discharge)
            new_soc = current_soc - (discharge * day_inputs.dt_hours) / (battery.eta_d * battery.capacity_kwh)
        
        # Idle
        else:
            charge_kw.append(0.0)
            discharge_kw.append(0.0)
            new_soc = current_soc
        
        soc.append(max(battery.soc_min, min(battery.soc_max, new_soc)))
    
    import_kw = [demand[t] + charge_kw[t] - discharge_kw[t] for t in range(T)]
    export_kw = [0.0] * T if day_inputs.allow_export else None
    decision = [1 if c > 0 else -1 if d > 0 else 0 for c, d in zip(charge_kw, discharge_kw)]
    cost = sum(prices[t] * import_kw[t] * day_inputs.dt_hours for t in range(T))
    
    return SolveResponse(
        status="success",
        message=f"Heuristic solution for {metadata['date']}: charged during low prices (<€{price_mean*0.9:.2f}), discharged during high prices (>€{price_mean*1.1:.2f}).",
        objective_cost=cost,
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        import_kw=import_kw,
        export_kw=export_kw,
        soc=soc,
        decision=decision,
        data_source=metadata["data_source"],
        forecast_info=metadata.get("forecast_models")
    )

# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     # Create target AG object with SIMPLIFIED settings
#     target = AG(
#         atype=SolveResponse,
#         max_iter=1,  # Back to 1 since we're providing full example
#         verbose_agent=False,
#         verbose_transduction=False,
#         reasoning=False,  # TURN OFF reasoning for simpler generation
#         instructions=instructions
#     )
    
#     # Execute optimization with error handling
#     print(f"\n{'='*70}")
#     print(f"Starting optimization for {metadata['date']}")
#     print(f"Data source: {metadata['data_source']}")
#     print(f"{'='*70}\n")
    
#     result = None
#     try:
#         result = await (target << source)
        
#         # Extract response and add metadata
#         response = result.states[0] if result.states else None
        
#         if response is None:
#             print("Warning: LLM returned no states")
#             raise ValueError("LLM returned no valid response")
        
#         print(f"\n{'='*70}")
#         print(f"✓ Optimization successful")
#         print(f"{'='*70}\n")
            
#     except Exception as e:
#         print(f"\n{'='*70}")
#         print(f"✗ Error during LLM optimization:")
#         print(f"  {type(e).__name__}: {e}")
#         print(f"{'='*70}\n")
        
#         # Print more debug info
#         if result is not None and hasattr(result, 'states'):
#             print(f"Result has {len(result.states)} states")
        
#         print("Returning fallback response with naive strategy...")
        
#         # Calculate naive solution (just import everything)
#         naive_cost = sum(day_inputs.prices_buy[t] * day_inputs.demand_kw[t] * day_inputs.dt_hours 
#                         for t in range(len(day_inputs.prices_buy)))
        
#         # Return a fallback error response
#         return SolveResponse(
#             status="error",
#             message=f"LLM optimization failed: {str(e)}. Returning naive solution (no battery usage). Try checking your API key or model configuration.",
#             objective_cost=naive_cost,
#             charge_kw=[0.0] * len(day_inputs.prices_buy),
#             discharge_kw=[0.0] * len(day_inputs.prices_buy),
#             import_kw=day_inputs.demand_kw,
#             export_kw=[0.0] * len(day_inputs.prices_buy) if day_inputs.allow_export else None,
#             soc=[request.battery.soc_init] * (len(day_inputs.prices_buy) + 1),
#             decision=[0] * len(day_inputs.prices_buy),
#             data_source=metadata["data_source"]
#         )
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response
def _build_optimization_instructions(
    battery: BatteryParams,
    day_inputs: DayInputs,
    metadata: dict
) -> str:
    """Build instructions that encourage actual optimization thinking"""
    
    T = len(day_inputs.prices_buy)
    prices = day_inputs.prices_buy
    demand = day_inputs.demand_kw
    
    # Calculate key statistics
    price_mean = sum(prices) / len(prices)
    price_min, price_max = min(prices), max(prices)
    price_std = (sum((p - price_mean)**2 for p in prices) / len(prices)) ** 0.5
    
    # Find price patterns
    low_price_threshold = price_mean - 0.5 * price_std
    high_price_threshold = price_mean + 0.5 * price_std
    
    low_hours = [i for i, p in enumerate(prices) if p <= low_price_threshold]
    high_hours = [i for i, p in enumerate(prices) if p >= high_price_threshold]
    
    # Calculate arbitrage potential
    if high_hours and low_hours:
        avg_low_price = sum(prices[i] for i in low_hours) / len(low_hours)
        avg_high_price = sum(prices[i] for i in high_hours) / len(high_hours)
        spread = avg_high_price - avg_low_price
        arbitrage_potential = spread * battery.eta_c * battery.eta_d  # Account for efficiency losses
    else:
        arbitrage_potential = 0
    
    instructions = f'''You are an expert energy arbitrage optimizer solving a battery scheduling problem for {metadata['date']}.

**MARKET CONDITIONS:**
- Timesteps: {T} hours
- Price statistics:
  * Min: €{price_min:.2f}/kWh at hour {prices.index(price_min)}
  * Max: €{price_max:.2f}/kWh at hour {prices.index(price_max)}
  * Mean: €{price_mean:.2f}/kWh
  * Std Dev: €{price_std:.2f}/kWh
- Low price hours ({len(low_hours)}): {low_hours[:8]}{"..." if len(low_hours) > 8 else ""}
- High price hours ({len(high_hours)}): {high_hours[:8]}{"..." if len(high_hours) > 8 else ""}
- Arbitrage spread: €{arbitrage_potential:.2f}/kWh after efficiency losses

**BATTERY SPECIFICATIONS:**
- Capacity: {battery.capacity_kwh:,.0f} kWh ({battery.capacity_kwh/1000:.1f} MWh)
- Power: {battery.cmax_kw:,.0f} kW charge / {battery.dmax_kw:,.0f} kW discharge
- Round-trip efficiency: {battery.eta_c * battery.eta_d * 100:.1f}%
- SoC range: [{battery.soc_min*100:.0f}%, {battery.soc_max*100:.0f}%]
- Starting SoC: {battery.soc_init*100:.0f}%
- Target end SoC: {(battery.soc_target or battery.soc_init)*100:.0f}%

**YOUR OPTIMIZATION TASK:**

1. **Analyze the price profile:** Identify the best times to charge (low prices) and discharge (high prices) to maximize arbitrage value.

2. **Plan battery cycles:** The battery can cycle multiple times per day. Plan when to:
   - CHARGE: During periods with prices below €{low_price_threshold:.2f}/kWh
   - DISCHARGE: During periods with prices above €{high_price_threshold:.2f}/kWh
   - IDLE: During medium-price periods

3. **Manage SoC trajectory:** Ensure SoC stays within [{battery.soc_min}, {battery.soc_max}] and ends at {battery.soc_target or battery.soc_init}.

4. **Consider efficiency losses:** Each kWh stored costs {(1-battery.eta_c)*100:.1f}% in charging losses and {(1-battery.eta_d)*100:.1f}% in discharging losses.

5. **Always meet demand:** At every hour, ensure: demand_kw[t] = import_kw[t] + discharge_kw[t] - charge_kw[t]

**OPTIMIZATION STRATEGY:**

Think through these questions:
- Which hours offer the best charging opportunities? (lowest prices with available capacity)
- Which hours offer the best discharging opportunities? (highest prices with available energy)
- Should I fully charge/discharge or partially? (consider SoC limits and future opportunities)
- Is the price spread large enough to justify the efficiency losses?

**REQUIRED OUTPUT FORMAT:**

Return a SolveResponse JSON with:

{{
  "status": "success",
  "message": "**CRITICAL: Provide detailed reasoning here (200-300 words):**
             - Explain your price analysis and identified patterns
             - Describe your charging strategy (which hours and why)
             - Describe your discharging strategy (which hours and why)  
             - Justify key decisions (e.g., why not fully charge, why discharge at hour X)
             - State the achieved cost and potential savings compared to no battery",
  
  "objective_cost": <total cost in €>,
  
  "prices_buy": {prices},
  "demand_kw": {demand},
  
  "charge_kw": [list of {T} hourly charge values in kW],
  "discharge_kw": [list of {T} hourly discharge values in kW],
  "import_kw": [list of {T} hourly grid import values in kW],
  "export_kw": [{"list of " + str(T) + " values" if day_inputs.allow_export else "null"}],
  "soc": [list of {T+1} SoC fractions from 0-1, starting at {battery.soc_init}],
  "decision": [list of {T} decisions: +1=charge, -1=discharge, 0=idle],
  "confidence": null,
  "data_source": null,
  "forecast_info": null
}}

**CONSTRAINTS (MUST SATISFY ALL):**
- SoC[t+1] = SoC[t] + ({battery.eta_c} × charge[t] - discharge[t] / {battery.eta_d}) × {day_inputs.dt_hours} / {battery.capacity_kwh}
- {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t
- 0 ≤ charge[t] ≤ {battery.cmax_kw}, 0 ≤ discharge[t] ≤ {battery.dmax_kw}
- Cannot charge AND discharge simultaneously
- import[t] = demand[t] + charge[t] - discharge[t] - export[t]

**NOW OPTIMIZE:** Generate the best possible schedule with detailed reasoning explaining your decisions.'''
    
    return instructions

# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build SIMPLE, concrete optimization instructions"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate simple heuristic solution
#     price_mean = sum(prices) / len(prices)
    
#     charge_kw = []
#     discharge_kw = []
#     soc = [battery.soc_init]
    
#     for t in range(T):
#         current_soc = soc[-1]
        
#         if prices[t] < price_mean * 0.9 and current_soc < battery.soc_max - 0.1:
#             charge = min(battery.cmax_kw, (battery.soc_max - current_soc) * battery.capacity_kwh / day_inputs.dt_hours)
#             charge_kw.append(round(charge, 2))
#             discharge_kw.append(0.0)
#             new_soc = current_soc + (battery.eta_c * charge * day_inputs.dt_hours) / battery.capacity_kwh
#         elif prices[t] > price_mean * 1.1 and current_soc > battery.soc_min + 0.1:
#             discharge = min(battery.dmax_kw, (current_soc - battery.soc_min) * battery.capacity_kwh / day_inputs.dt_hours)
#             discharge = min(discharge, demand[t])
#             charge_kw.append(0.0)
#             discharge_kw.append(round(discharge, 2))
#             new_soc = current_soc - (discharge * day_inputs.dt_hours) / (battery.eta_d * battery.capacity_kwh)
#         else:
#             charge_kw.append(0.0)
#             discharge_kw.append(0.0)
#             new_soc = current_soc
        
#         soc.append(round(max(battery.soc_min, min(battery.soc_max, new_soc)), 4))
    
#     import_kw = [round(demand[t] + charge_kw[t] - discharge_kw[t], 2) for t in range(T)]
#     export_kw = [0.0] * T if day_inputs.allow_export else None
#     decision = [1 if c > 0 else -1 if d > 0 else 0 for c, d in zip(charge_kw, discharge_kw)]
    
#     cost = round(sum(prices[t] * import_kw[t] * day_inputs.dt_hours for t in range(T)), 2)
    
#     # Create a CONCRETE example output
#     instructions = f'''You are optimizing battery storage for {T} hours on {metadata['date']}.

# COPY THIS EXACT SOLUTION (you may adjust values to improve cost):

# {{
#   "status": "success",
#   "message": "Optimized battery schedule for {metadata['date']}. Charged during low-price hours (below €{price_mean*0.9:.2f}) and discharged during high-price hours (above €{price_mean*1.1:.2f}) to minimize total cost.",
#   "objective_cost": {cost},
#   "charge_kw": {charge_kw},
#   "discharge_kw": {discharge_kw},
#   "import_kw": {import_kw},
#   "export_kw": {export_kw},
#   "soc": {soc},
#   "decision": {decision},
#   "confidence": null,
#   "data_source": null,
#   "forecast_info": null
# }}

# Return this EXACT JSON structure with all {T} timesteps. You can optimize the values but keep the same format.'''
    
#     return instructions


# Convenience function for notebook use
async def optimize_day_simple(
    date: str,
    capacity_kwh: float = None,
    power_kw: float = None,
    battery_sizing: str = "manual",
    use_forecast: bool = False,
    prices_model: str = None,
    consumption_model: str = None,
    allow_export: bool = False,
    data_dir: str = None
) -> SolveResponse:
    """
    Simplified interface for day optimization
    
    Args:
        date: Date string (YYYY-MM-DD)
        capacity_kwh: Battery capacity in kWh (if battery_sizing="manual")
        power_kw: Max charge/discharge power in kW (if battery_sizing="manual")
        battery_sizing: "manual" or "interquartile"
            - "manual": use provided capacity_kwh and power_kw
            - "interquartile": auto-calculate based on load IQR * 4 hours
        use_forecast: Whether to use forecast data
        prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
        consumption_model: Forecast model for consumption
        allow_export: Allow grid export
        data_dir: Data directory path
        
    Returns:
        SolveResponse with optimization results
    """
    from .schemas import ForecastModel
    
    # Determine battery capacity based on sizing method
    if battery_sizing == "interquartile":
        # Calculate battery size based on load statistics
        loader = DayDataLoader(data_dir=data_dir)
        stats = loader.get_load_statistics()
        
        # IQR * 4 hours rule (convert from MW to kW)
        capacity_kwh = stats['recommended_capacity_mwh'] * 1000  # MWh to kWh
        power_kw = stats['recommended_power_mw'] * 1000  # MW to kW
        
        print(f"📊 Automatic Battery Sizing (IQR Method):")
        print(f"   Load Mean: {stats['mean']:.2f} MW")
        print(f"   Load IQR: {stats['iqr']:.2f} MW (P25: {stats['p25']:.2f}, P75: {stats['p75']:.2f})")
        print(f"   → Battery Capacity: {capacity_kwh:.2f} kWh ({capacity_kwh/1000:.2f} MWh)")
        print(f"   → Charge/Discharge Power: {power_kw:.2f} kW ({power_kw/1000:.2f} MW)")
        print()
    else:
        # Manual sizing
        if capacity_kwh is None or power_kw is None:
            raise ValueError("capacity_kwh and power_kw must be provided when battery_sizing='manual'")
    
    battery = BatteryParams(
        capacity_kwh=capacity_kwh,
        soc_init=0.5,
        soc_min=0.1,
        soc_max=0.9,
        cmax_kw=power_kw,
        dmax_kw=power_kw,
        eta_c=0.95,
        eta_d=0.95,
        soc_target=0.5
    )
    
    forecast_models = None
    if use_forecast:
        forecast_models = ForecastModel(
            prices_model=prices_model,
            consumption_model=consumption_model
        )
    
    request = DayOptimizationRequest(
        date=date,
        battery=battery,
        use_forecast=use_forecast,
        forecast_models=forecast_models,
        allow_export=allow_export,
        dt_hours=1.0
    )
    
    return await optimize_day(request, data_dir=data_dir)
# """
# Day-Specific Battery Optimization using LLM

# This module optimizes battery storage operations for a specific day using
# LLM-based reasoning, with support for forecasted or actual data.
# """

# import os
# import warnings
# from dotenv import load_dotenv
# from agentics.core.agentics import AG

# from .schemas import (
#     DayOptimizationRequest, SolveResponse, SolveRequest, 
#     BatteryParams, DayInputs
# )
# from .day_data_loader import DayDataLoader

# warnings.filterwarnings("ignore", category=UserWarning)
# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     target = AG(
#         atype=SolveResponse,
#         max_iter=1,  # Match working example
#         verbose_agent=True,
#         reasoning=True,
#         instructions=instructions
#     )
    
#     # Execute optimization with error handling
#     print(f"\n{'='*70}")
#     print(f"Starting optimization for {metadata['date']}")
#     print(f"Data source: {metadata['data_source']}")
#     print(f"{'='*70}\n")
    
#     result = None
#     try:
#         result = await (target << source)
        
#         # Extract response and add metadata
#         response = result.states[0] if result.states else None
        
#         if response is None:
#             print("Warning: LLM returned no states")
#             raise ValueError("LLM returned no valid response")
        
#         print(f"\n{'='*70}")
#         print(f"✓ Optimization successful")
#         print(f"{'='*70}\n")
            
#     except Exception as e:
#         print(f"\n{'='*70}")
#         print(f"✗ Error during LLM optimization:")
#         print(f"  {type(e).__name__}: {e}")
#         print(f"{'='*70}\n")
        
#         # Print more debug info
#         if result is not None and hasattr(result, 'states'):
#             print(f"Result has {len(result.states)} states")
        
#         print("Returning fallback response with naive strategy...")
        
#         # Calculate naive solution (just import everything)
#         naive_cost = sum(day_inputs.prices_buy[t] * day_inputs.demand_kw[t] * day_inputs.dt_hours 
#                         for t in range(len(day_inputs.prices_buy)))
        
#         # Return a fallback error response
#         return SolveResponse(
#             status="error",
#             message=f"LLM optimization failed: {str(e)}. Returning naive solution (no battery usage). Try checking your API key or model configuration.",
#             objective_cost=naive_cost,
#             charge_kw=[0.0] * len(day_inputs.prices_buy),
#             discharge_kw=[0.0] * len(day_inputs.prices_buy),
#             import_kw=day_inputs.demand_kw,
#             export_kw=[0.0] * len(day_inputs.prices_buy) if day_inputs.allow_export else None,
#             soc=[request.battery.soc_init] * (len(day_inputs.prices_buy) + 1),
#             decision=[0] * len(day_inputs.prices_buy),
#             data_source=metadata["data_source"]
#         )
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate statistics
#     price_mean = sum(prices) / len(prices)
#     price_min, price_max = min(prices), max(prices)
    
#     instructions = f'''
# You are solving a daily battery scheduling optimization problem using reasoning and constraint satisfaction.

# You are given a request object containing:
# - Hourly energy prices for buying and selling electricity: {prices[:5]}... (total {T} hours)
# - Hourly electricity demand: {demand[:5]}... (total {T} hours)
# - Battery technical parameters including:
#   * capacity_kwh: {battery.capacity_kwh}
#   * charge/discharge power limits: cmax_kw: {battery.cmax_kw}, dmax_kw: {battery.dmax_kw}
#   * efficiencies: eta_c: {battery.eta_c}, eta_d: {battery.eta_d}
#   * state-of-charge bounds: soc_min: {battery.soc_min}, soc_max: {battery.soc_max}
#   * initial SoC: soc_init: {battery.soc_init}
#   * target end SoC: soc_target: {battery.soc_target or battery.soc_init}

# Your task is to:
# 1. Determine the hourly charge, discharge, grid import, grid export, and SoC schedule for {T} hours.

# 2. Minimize the total operational cost:
#    total_cost = Σ_t [ (price_buy[t] * import_kw[t] - price_sell[t] * export_kw[t]) * dt_hours ]
#    where dt_hours = {day_inputs.dt_hours}

# 3. Ensure all constraints are satisfied:
#    - SoC dynamics: SoC[t+1] = SoC[t] + (eta_c * charge_kw[t] - discharge_kw[t] / eta_d) * dt_hours / capacity_kwh
#      i.e., SoC[t+1] = SoC[t] + ({battery.eta_c} * charge_kw[t] - discharge_kw[t] / {battery.eta_d}) * {day_inputs.dt_hours} / {battery.capacity_kwh}
   
#    - SoC bounds: {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t
   
#    - Power limits:
#      * 0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
#      * 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}
   
#    - Energy balance: import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
#      Therefore: import_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t] - export_kw[t]
#      (must be ≥ 0)
   
#    - Export constraint: export_kw[t] ≥ 0 only if allow_export = {day_inputs.allow_export}
   
#    - Initial condition: SoC[0] = {battery.soc_init}
   
#    - End condition: SoC[{T}] ≥ {battery.soc_target or battery.soc_init}
   
#    - No simultaneous charge/discharge: The battery can either charge OR discharge OR stay idle in a given hour, not both.
#      This means: NOT(charge_kw[t] > 0 AND discharge_kw[t] > 0)

# STRATEGY GUIDANCE:
# - Price range: min={price_min:.2f}, max={price_max:.2f}, mean={price_mean:.2f}
# - Charge the battery when prices are LOW (below mean)
# - Discharge the battery when prices are HIGH (above mean)
# - Always meet demand at every timestep

# 4. Output a SolveResponse object with these fields:
#    - status: "success" or "failure"
#    - message: Brief explanation of your optimization strategy (2-3 sentences)
#    - objective_cost: the minimized total cost in currency units
#    - charge_kw: list of {T} hourly charge power values in kW
#    - discharge_kw: list of {T} hourly discharge power values in kW
#    - import_kw: list of {T} hourly grid import values in kW
#    - export_kw: list of {T} hourly grid export values in kW (or None if not allowed)
#    - soc: list of {T+1} state of charge values as fraction of capacity (0 to 1), including initial state
#    - decision: list of {T} decision values: +1 (charge), -1 (discharge), 0 (idle)

# Make sure:
# - All lists have the correct length ({T} for hourly values, {T+1} for soc)
# - All constraints are satisfied at every timestep
# - The objective function (total cost) is minimized
# - The schedule is physically feasible

# Think step by step:
# 1. Identify low-price hours for charging
# 2. Identify high-price hours for discharging  
# 3. Calculate optimal charge/discharge amounts respecting battery limits
# 4. Verify SoC stays within bounds
# 5. Ensure demand is always met
# 6. Calculate total cost

# Generate your complete SolveResponse now.
# '''
    
#     return instructions


# # Convenience function for notebook use
# async def optimize_day_simple(
#     date: str,
#     capacity_kwh: float = None,
#     power_kw: float = None,
#     battery_sizing: str = "manual",
#     use_forecast: bool = False,
#     prices_model: str = None,
#     consumption_model: str = None,
#     allow_export: bool = False,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Simplified interface for day optimization
    
#     Args:
#         date: Date string (YYYY-MM-DD)
#         capacity_kwh: Battery capacity in kWh (if battery_sizing="manual")
#         power_kw: Max charge/discharge power in kW (if battery_sizing="manual")
#         battery_sizing: "manual" or "interquartile"
#             - "manual": use provided capacity_kwh and power_kw
#             - "interquartile": auto-calculate based on load IQR * 4 hours
#         use_forecast: Whether to use forecast data
#         prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
#         consumption_model: Forecast model for consumption
#         allow_export: Allow grid export
#         data_dir: Data directory path
        
#     Returns:
#         SolveResponse with optimization results
#     """
#     from .schemas import ForecastModel
    
#     # Determine battery capacity based on sizing method
#     if battery_sizing == "interquartile":
#         # Calculate battery size based on load statistics
#         loader = DayDataLoader(data_dir=data_dir)
#         stats = loader.get_load_statistics()
        
#         # IQR * 4 hours rule (convert from MW to kW)
#         capacity_kwh = stats['recommended_capacity_mwh'] * 1000  # MWh to kWh
#         power_kw = stats['recommended_power_mw'] * 1000  # MW to kW
        
#         print(f"📊 Automatic Battery Sizing (IQR Method):")
#         print(f"   Load Mean: {stats['mean']:.2f} MW")
#         print(f"   Load IQR: {stats['iqr']:.2f} MW (P25: {stats['p25']:.2f}, P75: {stats['p75']:.2f})")
#         print(f"   → Battery Capacity: {capacity_kwh:.2f} kWh ({capacity_kwh/1000:.2f} MWh)")
#         print(f"   → Charge/Discharge Power: {power_kw:.2f} kW ({power_kw/1000:.2f} MW)")
#         print()
#     else:
#         # Manual sizing
#         if capacity_kwh is None or power_kw is None:
#             raise ValueError("capacity_kwh and power_kw must be provided when battery_sizing='manual'")
    
#     battery = BatteryParams(
#         capacity_kwh=capacity_kwh,
#         soc_init=0.5,
#         soc_min=0.1,
#         soc_max=0.9,
#         cmax_kw=power_kw,
#         dmax_kw=power_kw,
#         eta_c=0.95,
#         eta_d=0.95,
#         soc_target=0.5
#     )
    
#     forecast_models = None
#     if use_forecast:
#         forecast_models = ForecastModel(
#             prices_model=prices_model,
#             consumption_model=consumption_model
#         )
    
#     request = DayOptimizationRequest(
#         date=date,
#         battery=battery,
#         use_forecast=use_forecast,
#         forecast_models=forecast_models,
#         allow_export=allow_export,
#         dt_hours=1.0
#     )
    
#     return await optimize_day(request, data_dir=data_dir)

# """
# Day-Specific Battery Optimization using LLM

# This module optimizes battery storage operations for a specific day using
# LLM-based reasoning, with support for forecasted or actual data.
# """

# import os
# import warnings
# from dotenv import load_dotenv
# from agentics.core.agentics import AG

# from .schemas import (
#     DayOptimizationRequest, SolveResponse, SolveRequest, 
#     BatteryParams, DayInputs
# )
# from .day_data_loader import DayDataLoader

# warnings.filterwarnings("ignore", category=UserWarning)
# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     target = AG(
#         atype=SolveResponse,
#         max_iter=3,  # Increased iterations for better convergence
#         verbose_agent=True,
#         reasoning=True,
#         instructions=instructions
#     )
    
#     # Execute optimization with error handling
#     try:
#         result = await (target << source)
        
#         # Extract response and add metadata
#         response = result.states[0] if result.states else None
        
#         if response is None:
#             raise ValueError("LLM returned no valid response")
            
#     except Exception as e:
#         print(f"Error during LLM optimization: {e}")
#         print("Returning fallback response...")
        
#         # Return a fallback error response
#         return SolveResponse(
#             status="error",
#             message=f"LLM optimization failed: {str(e)}. Please try again or adjust parameters.",
#             objective_cost=0.0,
#             charge_kw=[0.0] * len(day_inputs.prices_buy),
#             discharge_kw=[0.0] * len(day_inputs.prices_buy),
#             import_kw=day_inputs.demand_kw,
#             export_kw=[0.0] * len(day_inputs.prices_buy) if day_inputs.allow_export else None,
#             soc=[request.battery.soc_init] * (len(day_inputs.prices_buy) + 1),
#             data_source=metadata["data_source"]
#         )
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate statistics
#     price_mean = sum(prices) / len(prices)
#     price_min, price_max = min(prices), max(prices)
    
#     # Generate simple heuristic solution as example/baseline
#     charge_kw = []
#     discharge_kw = []
#     import_kw = []
#     export_kw = []
#     soc = [battery.soc_init]
#     decision = []
    
#     for t in range(T):
#         # Simple heuristic: charge when price < mean, discharge when price > mean
#         if prices[t] < price_mean * 0.85 and soc[-1] < battery.soc_max - 0.1:
#             # Charge
#             charge = min(battery.cmax_kw, (battery.soc_max - soc[-1]) * battery.capacity_kwh / day_inputs.dt_hours)
#             charge_kw.append(charge)
#             discharge_kw.append(0.0)
#             import_kw.append(demand[t] + charge)
#             export_kw.append(0.0)
#             decision.append(1)
#             new_soc = soc[-1] + (battery.eta_c * charge * day_inputs.dt_hours) / battery.capacity_kwh
#         elif prices[t] > price_mean * 1.15 and soc[-1] > battery.soc_min + 0.1:
#             # Discharge
#             discharge = min(battery.dmax_kw, (soc[-1] - battery.soc_min) * battery.capacity_kwh / day_inputs.dt_hours)
#             discharge = min(discharge, demand[t])  # Don't discharge more than needed
#             charge_kw.append(0.0)
#             discharge_kw.append(discharge)
#             import_kw.append(max(0, demand[t] - discharge))
#             export_kw.append(0.0)
#             decision.append(-1)
#             new_soc = soc[-1] - (discharge * day_inputs.dt_hours) / (battery.eta_d * battery.capacity_kwh)
#         else:
#             # Idle
#             charge_kw.append(0.0)
#             discharge_kw.append(0.0)
#             import_kw.append(demand[t])
#             export_kw.append(0.0)
#             decision.append(0)
#             new_soc = soc[-1]
        
#         soc.append(max(battery.soc_min, min(battery.soc_max, new_soc)))
    
#     # Calculate baseline cost
#     baseline_cost = sum(prices[t] * import_kw[t] * day_inputs.dt_hours for t in range(T))
    
#     instructions = f"""
# You must generate a valid SolveResponse for battery optimization.

# **PROBLEM:**
# Date: {metadata['date']}
# Timesteps: {T} hours
# Battery: {battery.capacity_kwh} kWh capacity, {battery.cmax_kw} kW power
# Prices: min={price_min:.2f}, max={price_max:.2f}, mean={price_mean:.2f} €/kWh

# **STRATEGY:**
# 1. Charge battery when prices are LOW (< {price_mean * 0.85:.2f})
# 2. Discharge battery when prices are HIGH (> {price_mean * 1.15:.2f})
# 3. Stay idle otherwise
# 4. Always meet demand

# **BASELINE SOLUTION (Simple Heuristic):**
# Total Cost: €{baseline_cost:.2f}
# Charges: {sum(1 for d in decision if d > 0)} hours
# Discharges: {sum(1 for d in decision if d < 0)} hours

# **YOUR TASK:** 
# Return a VALID SolveResponse JSON object with these EXACT fields:

# {{
#   "status": "success",
#   "message": "I am an energy optimization AI. For {metadata['date']}, I analyzed the price pattern (range €{price_min:.2f}-{price_max:.2f}). I charged the battery during low-price hours and discharged during high-price hours to minimize costs while meeting all demand and battery constraints.",
#   "objective_cost": {baseline_cost:.2f},
#   "charge_kw": {charge_kw},
#   "discharge_kw": {discharge_kw},
#   "import_kw": {import_kw},
#   "export_kw": {'null' if not day_inputs.allow_export else export_kw},
#   "soc": {soc},
#   "decision": {decision},
#   "confidence": null
# }}

# You can IMPROVE on this baseline by optimizing better, but you MUST return a valid JSON matching this structure with all {T} timesteps.

# Output the JSON now:"""
    
#     return instructions


# # Convenience function for notebook use
# async def optimize_day_simple(
#     date: str,
#     capacity_kwh: float = None,
#     power_kw: float = None,
#     battery_sizing: str = "manual",
#     use_forecast: bool = False,
#     prices_model: str = None,
#     consumption_model: str = None,
#     allow_export: bool = False,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Simplified interface for day optimization
    
#     Args:
#         date: Date string (YYYY-MM-DD)
#         capacity_kwh: Battery capacity in kWh (if battery_sizing="manual")
#         power_kw: Max charge/discharge power in kW (if battery_sizing="manual")
#         battery_sizing: "manual" or "interquartile"
#             - "manual": use provided capacity_kwh and power_kw
#             - "interquartile": auto-calculate based on load IQR * 4 hours
#         use_forecast: Whether to use forecast data
#         prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
#         consumption_model: Forecast model for consumption
#         allow_export: Allow grid export
#         data_dir: Data directory path
        
#     Returns:
#         SolveResponse with optimization results
#     """
#     from .schemas import ForecastModel
    
#     # Determine battery capacity based on sizing method
#     if battery_sizing == "interquartile":
#         # Calculate battery size based on load statistics
#         loader = DayDataLoader(data_dir=data_dir)
#         stats = loader.get_load_statistics()
        
#         # IQR * 4 hours rule (convert from MW to kW)
#         capacity_kwh = stats['recommended_capacity_mwh'] * 1000  # MWh to kWh
#         power_kw = stats['recommended_power_mw'] * 1000  # MW to kW
        
#         print(f"📊 Automatic Battery Sizing (IQR Method):")
#         print(f"   Load Mean: {stats['mean']:.2f} MW")
#         print(f"   Load IQR: {stats['iqr']:.2f} MW (P25: {stats['p25']:.2f}, P75: {stats['p75']:.2f})")
#         print(f"   → Battery Capacity: {capacity_kwh:.2f} kWh ({capacity_kwh/1000:.2f} MWh)")
#         print(f"   → Charge/Discharge Power: {power_kw:.2f} kW ({power_kw/1000:.2f} MW)")
#         print()
#     else:
#         # Manual sizing
#         if capacity_kwh is None or power_kw is None:
#             raise ValueError("capacity_kwh and power_kw must be provided when battery_sizing='manual'")
    
#     battery = BatteryParams(
#         capacity_kwh=capacity_kwh,
#         soc_init=0.5,
#         soc_min=0.1,
#         soc_max=0.9,
#         cmax_kw=power_kw,
#         dmax_kw=power_kw,
#         eta_c=0.95,
#         eta_d=0.95,
#         soc_target=0.5
#     )
    
#     forecast_models = None
#     if use_forecast:
#         forecast_models = ForecastModel(
#             prices_model=prices_model,
#             consumption_model=consumption_model
#         )
    
#     request = DayOptimizationRequest(
#         date=date,
#         battery=battery,
#         use_forecast=use_forecast,
#         forecast_models=forecast_models,
#         allow_export=allow_export,
#         dt_hours=1.0
#     )
    
#     return await optimize_day(request, data_dir=data_dir)

# """
# Day-Specific Battery Optimization using LLM

# This module optimizes battery storage operations for a specific day using
# LLM-based reasoning, with support for forecasted or actual data.
# """

# import os
# import warnings
# from dotenv import load_dotenv
# from agentics.core.agentics import AG

# from .schemas import (
#     DayOptimizationRequest, SolveResponse, SolveRequest, 
#     BatteryParams, DayInputs
# )
# from .day_data_loader import DayDataLoader

# warnings.filterwarnings("ignore", category=UserWarning)
# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     target = AG(
#         atype=SolveResponse,
#         max_iter=3,  # Increased iterations for better convergence
#         verbose_agent=True,
#         reasoning=True,
#         instructions=instructions
#     )
    
#     # Execute optimization with error handling
#     try:
#         result = await (target << source)
        
#         # Extract response and add metadata
#         response = result.states[0] if result.states else None
        
#         if response is None:
#             raise ValueError("LLM returned no valid response")
            
#     except Exception as e:
#         print(f"Error during LLM optimization: {e}")
#         print("Returning fallback response...")
        
#         # Return a fallback error response
#         return SolveResponse(
#             status="error",
#             message=f"LLM optimization failed: {str(e)}. Please try again or adjust parameters.",
#             objective_cost=0.0,
#             charge_kw=[0.0] * len(day_inputs.prices_buy),
#             discharge_kw=[0.0] * len(day_inputs.prices_buy),
#             import_kw=day_inputs.demand_kw,
#             export_kw=[0.0] * len(day_inputs.prices_buy) if day_inputs.allow_export else None,
#             soc=[request.battery.soc_init] * (len(day_inputs.prices_buy) + 1),
#             data_source=metadata["data_source"]
#         )
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate statistics
#     price_mean = sum(prices) / len(prices)
#     price_min, price_max = min(prices), max(prices)
    
#     instructions = f"""
# You are an expert energy storage optimization AI solving a battery scheduling problem.

# **OBJECTIVE:** Minimize total cost = Σ [(price_buy[t] × import_kw[t] - price_sell[t] × export_kw[t]) × {day_inputs.dt_hours}]

# **GIVEN DATA FOR {metadata['date']}:**
# - {T} hourly timesteps
# - Prices (€/kWh): range [{price_min:.2f}, {price_max:.2f}], mean {price_mean:.2f}
# - Demand (kW): {demand[:3]}...
# - Data source: {metadata['data_source']}

# **BATTERY CONSTRAINTS:**
# - Capacity: {battery.capacity_kwh} kWh
# - Power: charge ≤ {battery.cmax_kw} kW, discharge ≤ {battery.dmax_kw} kW
# - Efficiency: charge {battery.eta_c}, discharge {battery.eta_d}
# - SoC limits: [{battery.soc_min}, {battery.soc_max}]
# - Initial SoC: {battery.soc_init}, Target end SoC: {battery.soc_target or battery.soc_init}

# **REQUIRED CONSTRAINTS:**
# 1. Energy balance: import - export = demand + charge - discharge
# 2. SoC update: soc[t+1] = soc[t] + ({battery.eta_c}*charge[t] - discharge[t]/{battery.eta_d})*{day_inputs.dt_hours}/{battery.capacity_kwh}
# 3. Cannot charge AND discharge simultaneously
# 4. Meet demand: demand[t] ≤ import[t] + discharge[t] - charge[t]

# **YOUR TASK:**
# Generate optimal {T}-hour schedule that:
# - Charges when prices are LOW
# - Discharges when prices are HIGH  
# - Always meets demand
# - Respects all battery constraints

# **OUTPUT AS SolveResponse with:**
# - status: "success"
# - message: Brief explanation (3-5 sentences) of your strategy
# - objective_cost: total cost in €
# - charge_kw: list of {T} charge values
# - discharge_kw: list of {T} discharge values
# - import_kw: list of {T} import values
# - export_kw: list of {T} export values {'(or null)' if not day_inputs.allow_export else ''}
# - soc: list of {T+1} SoC fractions [including initial]
# - decision: list of {T} values: +1=charge, -1=discharge, 0=idle

# Generate the response now.
# """
    
#     return instructions


# # Convenience function for notebook use
# async def optimize_day_simple(
#     date: str,
#     capacity_kwh: float = 100.0,
#     power_kw: float = 50.0,
#     use_forecast: bool = False,
#     prices_model: str = None,
#     consumption_model: str = None,
#     allow_export: bool = False,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Simplified interface for day optimization
    
#     Args:
#         date: Date string (YYYY-MM-DD)
#         capacity_kwh: Battery capacity in kWh
#         power_kw: Max charge/discharge power in kW
#         use_forecast: Whether to use forecast data
#         prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
#         consumption_model: Forecast model for consumption
#         allow_export: Allow grid export
#         data_dir: Data directory path
        
#     Returns:
#         SolveResponse with optimization results
#     """
#     from .schemas import ForecastModel
    
#     battery = BatteryParams(
#         capacity_kwh=capacity_kwh,
#         soc_init=0.5,
#         soc_min=0.1,
#         soc_max=0.9,
#         cmax_kw=power_kw,
#         dmax_kw=power_kw,
#         eta_c=0.95,
#         eta_d=0.95,
#         soc_target=0.5
#     )
    
#     forecast_models = None
#     if use_forecast:
#         forecast_models = ForecastModel(
#             prices_model=prices_model,
#             consumption_model=consumption_model
#         )
    
#     request = DayOptimizationRequest(
#         date=date,
#         battery=battery,
#         use_forecast=use_forecast,
#         forecast_models=forecast_models,
#         allow_export=allow_export,
#         dt_hours=1.0
#     )
    
#     return await optimize_day(request, data_dir=data_dir)


# """
# Day-Specific Battery Optimization using LLM

# This module optimizes battery storage operations for a specific day using
# LLM-based reasoning, with support for forecasted or actual data.
# """

# import os
# import warnings
# from dotenv import load_dotenv
# from agentics.core.agentics import AG

# from .schemas import (
#     DayOptimizationRequest, SolveResponse, SolveRequest, 
#     BatteryParams, DayInputs
# )
# from .day_data_loader import DayDataLoader

# warnings.filterwarnings("ignore", category=UserWarning)
# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     target = AG(
#         atype=SolveResponse,
#         max_iter=1,
#         verbose_agent=True,
#         reasoning=True,
#         instructions=instructions
#     )
    
#     # Execute optimization
#     result = await (target << source)
    
#     # Extract response and add metadata
#     response = result.states[0] if result.states else None
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate statistics
#     price_mean = sum(prices) / len(prices)
#     price_min, price_max = min(prices), max(prices)
#     demand_mean = sum(demand) / len(demand)
    
#     # Identify price patterns
#     low_price_hours = [i for i, p in enumerate(prices) if p < price_mean * 0.9]
#     high_price_hours = [i for i, p in enumerate(prices) if p > price_mean * 1.1]
    
#     instructions = f"""
# You are an AI energy storage optimization agent tasked with minimizing operational costs for a battery energy storage system.

# **YOUR IDENTITY:**
# You are an expert energy arbitrage optimizer with deep knowledge of:
# - Mixed Integer Linear Programming (MILP) for energy storage
# - Electricity market dynamics and price patterns
# - Battery physics and operational constraints
# - Multi-stage decision-making under uncertainty

# **PROBLEM CONTEXT:**
# Date: {metadata['date']}
# Data Source: {metadata['data_source'].upper()}
# {f"Forecast Models: Prices={metadata.get('forecast_models', {}).get('prices', 'actual')}, Consumption={metadata.get('forecast_models', {}).get('consumption', 'actual')}" if metadata['data_source'] == 'forecast' else ''}
# Time Steps: {T} hours
# Time Resolution: {day_inputs.dt_hours} hours per step

# **GIVEN DATA:**
# Prices (€/kWh): {prices[:5]}... (range: {price_min:.2f} - {price_max:.2f}, mean: {price_mean:.2f})
# Demand (kW): {demand[:5]}... (mean: {demand_mean:.2f})
# Allow Export: {day_inputs.allow_export}

# **BATTERY SPECIFICATIONS:**
# - Capacity: {battery.capacity_kwh} kWh
# - Max Charge Rate: {battery.cmax_kw} kW
# - Max Discharge Rate: {battery.dmax_kw} kW
# - Charge Efficiency: {battery.eta_c * 100}%
# - Discharge Efficiency: {battery.eta_d * 100}%
# - Initial SoC: {battery.soc_init * 100}%
# - SoC Limits: [{battery.soc_min * 100}%, {battery.soc_max * 100}%]
# - Target End SoC: {(battery.soc_target or battery.soc_init) * 100}%

# **OPTIMIZATION OBJECTIVE:**
# Minimize total cost = Σ_t [(price_buy[t] × import_kw[t] - price_sell[t] × export_kw[t]) × dt_hours]

# **HARD CONSTRAINTS (MUST SATISFY):**
# 1. **Energy Balance at each timestep t:**
#    import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
   
# 2. **State of Charge Dynamics:**
#    soc[t+1] = soc[t] + (eta_c × charge_kw[t] × dt - discharge_kw[t] × dt / eta_d) / capacity_kwh
#    - soc[0] = {battery.soc_init}
#    - soc[T] ≥ {battery.soc_target or battery.soc_init} (end target)
   
# 3. **SoC Limits:**
#    {battery.soc_min} ≤ soc[t] ≤ {battery.soc_max} for all t

# 4. **Power Limits:**
#    - 0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
#    - 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}

# 5. **No Simultaneous Charge/Discharge:**
#    Battery can either charge OR discharge OR idle at each timestep, but NOT both charge and discharge simultaneously.

# 6. **Non-negative Grid Operations:**
#    - import_kw[t] ≥ 0 (always)
#    - export_kw[t] ≥ 0 (only if allow_export = {day_inputs.allow_export})

# **OPTIMIZATION STRATEGY:**

# **Phase 1: Price Pattern Analysis**
# Analyze the price profile to identify:
# - Low price periods (below {price_mean * 0.9:.2f}): hours {low_price_hours[:5]}...
# - High price periods (above {price_mean * 1.1:.2f}): hours {high_price_hours[:5]}...
# - Price trends and volatility

# **Phase 2: Multi-Cycle Planning**
# The battery should cycle MULTIPLE times throughout the day if profitable:
# - Round-trip efficiency loss: {(1 - battery.eta_c * battery.eta_d) * 100:.1f}%
# - Breakeven spread: price_sell > price_buy × {1/(battery.eta_c * battery.eta_d):.3f}
# - Identify ALL profitable charge-discharge opportunities

# **Phase 3: SoC Trajectory Planning**
# Plan the SoC trajectory to:
# - Start at {battery.soc_init * 100}%
# - Never violate [{battery.soc_min * 100}%, {battery.soc_max * 100}%] bounds
# - End at {(battery.soc_target or battery.soc_init) * 100}%
# - Maximize utilization without hitting limits prematurely

# **Phase 4: Demand Coverage**
# At each hour, ensure:
# demand_kw[t] ≤ import_kw[t] + discharge_kw[t] - charge_kw[t] - export_kw[t]

# **DECISION-MAKING EXAMPLE:**

# Consider a simple case with prices = [100, 50, 150] €/kWh, demand = [10, 10, 10] kW:

# Hour 0 (€100, medium price):
# - Price neither high nor low → Consider battery state
# - If SoC allows, might discharge to avoid high import cost
# - Decision: discharge 5 kW, import 5 kW
# - Cost: 100 × 5 × 1 = €500

# Hour 1 (€50, LOW price):
# - Excellent charging opportunity
# - Demand + charge from grid, possibly prepare for Hour 2
# - Decision: charge 10 kW, import 20 kW total (10 demand + 10 charge)
# - Cost: 50 × 20 × 1 = €1000

# Hour 2 (€150, HIGH price):
# - Most expensive hour → maximize battery discharge
# - Can discharge what we charged in Hour 1
# - Decision: discharge 15 kW (covers 10 kW demand + 5 kW export)
# - Revenue: 150 × 5 × 1 = €750 (export)
# - Cost: 0 (no import needed)
# - Net Hour 2: -€750 (profit)

# Total cost: €500 + €1000 - €750 = €750

# **YOUR TASK:**

# 1. **Analyze** the full price and demand profiles
# 2. **Plan** a multi-cycle charging strategy
# 3. **Generate** hour-by-hour decisions for:
#    - charge_kw[t]: Battery charging power
#    - discharge_kw[t]: Battery discharging power
#    - import_kw[t]: Grid import power
#    - export_kw[t]: Grid export power (if allowed)
#    - soc[t]: State of charge trajectory
#    - decision[t]: +1 (charge), -1 (discharge), 0 (idle)

# 4. **Verify** all constraints are satisfied
# 5. **Calculate** the total objective cost
# 6. **Explain** your strategy comprehensively

# **OUTPUT FORMAT:**

# Return a SolveResponse object with:
# - status: "success" (if feasible) or "failure"
# - message: YOUR COMPREHENSIVE EXPLANATION (see below)
# - objective_cost: Total cost in €
# - charge_kw: List of {T} charging values
# - discharge_kw: List of {T} discharging values
# - import_kw: List of {T} import values
# - export_kw: List of {T} export values (or None if not allowed)
# - soc: List of {T+1} SoC values (including initial state)
# - decision: List of {T} decisions (+1, -1, or 0)

# **MESSAGE FIELD REQUIREMENTS:**

# Your message field must include a COMPREHENSIVE explanation with:

# 1. **Introduction (2-3 sentences):**
#    - Introduce yourself as an energy optimization AI
#    - State the problem you're solving
#    - Mention the date and data source

# 2. **Price Analysis (3-4 sentences):**
#    - Describe the price pattern (volatility, trends)
#    - Identify cheapest and most expensive periods
#    - Calculate price spreads and arbitrage opportunities

# 3. **Strategy Overview (4-5 sentences):**
#    - Explain your high-level approach
#    - Number of charge-discharge cycles planned
#    - How you're balancing demand coverage and arbitrage
#    - Key decision-making principles

# 4. **Hour-by-Hour Decisions (1 paragraph):**
#    - Summarize key decision points
#    - Explain WHY you charge/discharge at specific hours
#    - Mention any constraints that limited your options

# 5. **Battery Utilization (2-3 sentences):**
#    - SoC trajectory summary
#    - How you managed SoC limits
#    - Whether you achieved target end SoC

# 6. **Cost Breakdown (3-4 sentences):**
#    - Total import costs
#    - Total export revenues (if any)
#    - Net cost and potential savings
#    - Compare to naive strategy (e.g., no battery)

# 7. **Validation (2-3 sentences):**
#    - Confirm all constraints satisfied
#    - Mention any challenges or trade-offs
#    - Confidence in solution optimality

# Make your explanation clear, technical, and insightful. Use specific numbers and hours. This will help users understand your decision-making process.

# **CRITICAL REMINDERS:**
# - NEVER charge and discharge simultaneously in the same hour
# - ALWAYS satisfy demand: demand_kw[t] ≤ total supply at time t
# - Battery SoC MUST stay within [{battery.soc_min}, {battery.soc_max}]
# - End SoC MUST be ≥ {battery.soc_target or battery.soc_init}
# - Import costs money, export generates revenue (if allowed)
# - Minimize TOTAL COST across all hours

# Begin your optimization now.
# """
    
#     return instructions


# # Convenience function for notebook use
# async def optimize_day_simple(
#     date: str,
#     capacity_kwh: float = 100.0,
#     power_kw: float = 50.0,
#     use_forecast: bool = False,
#     prices_model: str = None,
#     consumption_model: str = None,
#     allow_export: bool = False,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Simplified interface for day optimization
    
#     Args:
#         date: Date string (YYYY-MM-DD)
#         capacity_kwh: Battery capacity in kWh
#         power_kw: Max charge/discharge power in kW
#         use_forecast: Whether to use forecast data
#         prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
#         consumption_model: Forecast model for consumption
#         allow_export: Allow grid export
#         data_dir: Data directory path
        
#     Returns:
#         SolveResponse with optimization results
#     """
#     from .schemas import ForecastModel
    
#     battery = BatteryParams(
#         capacity_kwh=capacity_kwh,
#         soc_init=0.5,
#         soc_min=0.1,
#         soc_max=0.9,
#         cmax_kw=power_kw,
#         dmax_kw=power_kw,
#         eta_c=0.95,
#         eta_d=0.95,
#         soc_target=0.5
#     )
    
#     forecast_models = None
#     if use_forecast:
#         forecast_models = ForecastModel(
#             prices_model=prices_model,
#             consumption_model=consumption_model
#         )
    
#     request = DayOptimizationRequest(
#         date=date,
#         battery=battery,
#         use_forecast=use_forecast,
#         forecast_models=forecast_models,
#         allow_export=allow_export,
#         dt_hours=1.0
#     )
    
#     return await optimize_day(request, data_dir=data_dir)