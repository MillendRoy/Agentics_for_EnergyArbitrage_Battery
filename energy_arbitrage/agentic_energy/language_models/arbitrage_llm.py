"""
Pure Battery Arbitrage Optimization using LLM
"""
import os
import warnings
from dotenv import load_dotenv
from agentics.core.agentics import AG
import numpy as np

from agentic_energy.schemas import (
    ArbitrageBatteryParams,
    ArbitrageInputs,
    ArbitrageRequest,
    ArbitrageResponse,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")

load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


async def optimize_arbitrage(
    prices: list = None,
    battery_capacity_mwh: float = 1.0,
    power_mw: float = 0.25,
    dt_hours: float = 1.0
):
    """
    Optimize battery arbitrage given price data
    
    Args:
        prices: List of hourly prices (€/MWh or $/MWh). If None, uses synthetic data.
        battery_capacity_mwh: Battery capacity in MWh
        power_mw: Max charge/discharge power in MW
        dt_hours: Time step in hours (1.0 for hourly, 1/12 for 5-min)
        
    Returns:
        tuple: (result, inputs, battery)
    """
    try:
        # Create synthetic prices if none provided
        if prices is None:
            hours = np.arange(24)
            prices = (50 + 30 * np.sin(2 * np.pi * hours / 24)).tolist()
            print("Using synthetic price data (sine wave)")
        else:
            print(f"Using provided prices: {len(prices)} data points")
        
        # Create battery parameters
        battery = ArbitrageBatteryParams(
            capacity_mwh=battery_capacity_mwh,
            soc_init=0.5,
            soc_min=0.0,
            soc_max=1.0,
            power_mw=power_mw,
            eta_c=0.95,
            eta_d=0.95
        )
        
        # Create inputs
        inputs = ArbitrageInputs(
            prices=prices,
            dt_hours=dt_hours
        )
        
        # Create request
        req = ArbitrageRequest(
            battery=battery,
            inputs=inputs
        )
        
        print(f"\nBattery Configuration:")
        print(f"  Capacity: {battery.capacity_mwh} MWh")
        print(f"  Power: {battery.power_mw} MW")
        print(f"  Initial SoC: {battery.soc_init*100}%")
        print(f"\nPrice Range: €{min(prices):.2f} - €{max(prices):.2f} per MWh")
        print(f"Price Spread: €{max(prices) - min(prices):.2f} per MWh")
        
        # Create Agentics source
        source = AG(
            atype=ArbitrageRequest,
            states=[req]
        )
        
        # Create Agentics target with optimization instructions
        target = AG(
            atype=ArbitrageResponse,
            max_iter=1,
            verbose_agent=True,
            reasoning=True,
            # Replace the instructions in arbitrage_llm.py with this:
            # Replace the instructions in arbitrage_llm.py with this:
            instructions='''
            You are optimizing a battery energy storage system for CONTINUOUS arbitrage in the electricity market.

            **Given:**
            - Battery capacity in MWh (capacity_mwh)
            - Maximum charge/discharge power in MW (power_mw)
            - Hourly electricity prices in €/MWh or $/MWh over entire horizon
            - Charge efficiency (eta_c) and discharge efficiency (eta_d)
            - Initial state of charge (soc_init), SoC bounds: soc_min to soc_max

            **OPTIMIZATION PHILOSOPHY:**

            This is NOT about finding "3 cheapest hours to charge" and "3 most expensive hours to discharge."

            This IS about CONTINUOUS TRADING throughout the day:
            - Battery cycles can happen MULTIPLE times per day
            - Trade whenever there's profitable spread
            - Use the full horizon dynamically

            **OPTIMAL STRATEGY - MULTI-STAGE APPROACH:**

            STAGE 1: PRICE PATTERN ANALYSIS
            Calculate reference price (e.g., mean or median of all prices)
            Identify price movements: valleys, peaks, trends

            STAGE 2: DYNAMIC THRESHOLD STRATEGY
            For each hour t, decide based on:
            1. Current price vs. reference price
            2. Price trend (rising or falling)
            3. Current SoC (do we have room to charge? energy to discharge?)
            4. Remaining hours (can we do better later?)

            STAGE 3: GREEDY-OPTIMAL SCHEDULING
            At each hour, choose ONE of:
            - CHARGE at power_mw if: price is LOW relative to future + battery has room
            - DISCHARGE at power_mw if: price is HIGH relative to past + battery has energy
            - IDLE if: price is medium OR battery is at limits

            **KEY INSIGHT - ROUND-TRIP EFFICIENCY:**
            To profit from one cycle (charge then discharge):
            sell_price × eta_d > buy_price / eta_c
            sell_price > buy_price / (eta_c × eta_d)
            With eta_c = eta_d = 0.95:
            sell_price > buy_price × 1.108

            So any price spread > 10.8% is profitable!

            **DETAILED EXAMPLE - 24 HOUR OPTIMAL STRATEGY:**

            Prices (€/MWh): 
            [105, 104, 102, 95, 100, 99, 98, 100, 100, 91, 92, 88, 87, 87, 87, 87, 92, 98, 106, 140, 143, 147, 122, 112]

            Battery: 1 MWh capacity, 0.25 MW power, start 50% SoC, eta=0.95

            Reference: Mean = 103.3, Median = 100
            Round-trip threshold: ~110.8 (need 10.8% spread)

            OPTIMAL SCHEDULE:

            Hour 0 (€105): Above mean, discharge 0.25 MW → SoC: 50% → 23.7%
            Hour 1 (€104): Still above mean, but SoC getting low → IDLE → SoC: 23.7%
            Hour 2 (€102): Near mean → IDLE → SoC: 23.7%
            Hour 3 (€95): Below mean, charge 0.25 MW → SoC: 23.7% → 47.4%
            Hour 4 (€100): At mean → IDLE → SoC: 47.4%
            Hour 5 (€99): Slightly below, charge 0.25 MW → SoC: 47.4% → 71.1%
            Hour 6 (€98): Below mean, charge 0.25 MW → SoC: 71.1% → 94.8%
            Hour 7 (€100): At mean, battery almost full → IDLE → SoC: 94.8%
            Hour 8 (€100): At mean → IDLE → SoC: 94.8%
            Hour 9 (€91): Below mean but battery full → IDLE → SoC: 94.8%
            Hour 10 (€92): LOW, but battery already at 95% (near max) → IDLE → SoC: 94.8%
            Hour 11 (€88): VERY LOW, but battery full, WASTED OPPORTUNITY
            Hour 12 (€87): LOWEST, but battery full, WASTED OPPORTUNITY  
            Hour 13 (€87): Battery full → IDLE
            Hour 14 (€87): Battery full → IDLE
            Hour 15 (€87): Battery full → IDLE
            Hour 16 (€92): Still low → IDLE → SoC: 94.8%
            Hour 17 (€98): Near mean → IDLE → SoC: 94.8%
            Hour 18 (€106): Above mean, discharge 0.25 MW → SoC: 94.8% → 68.5%
            Hour 19 (€140): VERY HIGH, discharge 0.25 MW → SoC: 68.5% → 42.2%
            Hour 20 (€143): PEAK, discharge 0.25 MW → SoC: 42.2% → 15.9%
            Hour 21 (€147): HIGHEST, discharge 0.25 MW → SoC: 15.9% → 0% (hit minimum)
            Hour 22 (€122): High but battery empty → IDLE → SoC: 0%
            Hour 23 (€112): High but battery empty → IDLE → SoC: 0%

            **PROBLEM WITH ABOVE:** We hit max SoC too early and wasted hours 11-15 (the cheapest!)

            **REVISED OPTIMAL STRATEGY:**

            Look ahead: Hours 11-15 are €87 (CHEAPEST). Hours 19-21 are €140-147 (HIGHEST).

            Better approach:
            - Hours 0-2: Discharge early (prices >100) to make room
            - Hours 3-10: Moderate charging, DON'T fill battery yet
            - Hours 11-15: AGGRESSIVE charging at absolute cheapest
            - Hours 18-21: AGGRESSIVE discharging at absolute highest

            TRULY OPTIMAL SCHEDULE:

            Hour 0 (€105): discharge 0.25 → SoC: 50% → 23.7%
            Hour 1 (€104): discharge 0.25 → SoC: 23.7% → 0% (hit min)
            Hour 2 (€102): idle (battery empty)
            Hour 3 (€95): charge 0.25 → SoC: 0% → 23.8%
            Hour 4 (€100): idle
            Hour 5 (€99): charge 0.25 → SoC: 23.8% → 47.5%
            Hour 6 (€98): charge 0.25 → SoC: 47.5% → 71.3%
            Hour 7 (€100): idle
            Hour 8 (€100): idle  
            Hour 9 (€91): charge 0.25 → SoC: 71.3% → 95.0%
            Hour 10 (€92): idle (near max)
            Hour 11 (€88): idle (at max)
            Hour 12 (€87): CHEAPEST but full → start discharging a bit to cycle? NO, WAIT
            Hour 13 (€87): idle (at max, waiting for peak)
            Hour 14 (€87): idle
            Hour 15 (€87): idle
            Hour 16 (€92): idle (holding for peak)
            Hour 17 (€98): idle
            Hour 18 (€106): discharge 0.25 → SoC: 95% → 68.7%
            Hour 19 (€140): discharge 0.25 → SoC: 68.7% → 42.4%
            Hour 20 (€143): discharge 0.25 → SoC: 42.4% → 16.1%
            Hour 21 (€147): discharge 0.25 → SoC: 16.1% → 0%
            Hour 22 (€122): idle (empty)
            Hour 23 (€112): idle

            **EVEN BETTER STRATEGY - MULTIPLE CYCLES:**

            Since hours 11-15 are SO cheap (€87) and hours 19-21 are SO expensive (€140-147), we should:

            1. First cycle: Charge hours 11-15, discharge hours 19-21
            2. But we can do MORE cycles!

            Hour 0 (€105): Medium-high, discharge 0.25 → SoC: 50% → 23.7%
            Hour 1 (€104): discharge 0.25 → SoC: 23.7% → 0%  
            Hour 2-10: Charge back up gradually
            Hour 11-15 (€87): CHEAP - charge heavily
            Hour 16-17: Hold
            Hour 18 (€106): Start discharge
            Hour 19-21 (€140-147): PEAK - discharge heavily
            Hour 22-23: Hold

            Profit calculation:
            - Buy ~2 MWh total at avg €92
            - Sell ~1.8 MWh (after efficiency) at avg €135
            - Profit: (1.8 × €135) - (2 × €92) = €243 - €184 = €59

            **RULES FOR YOUR SOLUTION:**

            1. **Constraint**: 0 ≤ charge_mw[t] ≤ power_mw, 0 ≤ discharge_mw[t] ≤ power_mw
            2. **No simultaneous**: NOT(charge_mw[t] > 0 AND discharge_mw[t] > 0)
            3. **SoC dynamics**: soc[t+1] = soc[t] + (eta_c × charge_mw[t] - discharge_mw[t]/eta_d) × dt / capacity_mwh
            4. **SoC limits**: soc_min ≤ soc[t] ≤ soc_max, soc[0] = soc_init
            5. **Objective**: profit = Σ_t [price[t] × discharge_mw[t] × dt - price[t] × charge_mw[t] × dt]

            **YOUR TASK:**

            Given the actual prices, generate a SOPHISTICATED schedule that:
            - Uses MOST hours of the day (not just 3-4 hours!)
            - Cycles the battery MULTIPLE times if profitable
            - Responds DYNAMICALLY to price patterns
            - MAXIMIZES profit by trading continuously
            - Avoids hitting SoC limits prematurely

            Think step-by-step:
            1. Analyze the full price curve
            2. Identify multiple valleys and peaks
            3. Plan cycles between valleys→peaks
            4. Execute continuously throughout the day

            **Output Format:**
            Return ArbitrageResponse with:
            - status: "success"
            - message: Describe your multi-cycle strategy (Please describe your reasoning similar to the following format: "Multi-cycle continuous arbitrage strategy executed across 11 active hours. Strategy: (1) Early discharge at €100 to create capacity, (2) Strategic intermediate cycles hours 3-9 to capture medium spreads, (3) Aggressive charging hours 9-15 during valley period (€70-90), (4) Maximum discharge hours 18-21 during peak (€120-160). Battery fully utilized with SoC cycling: 50%→23.7%→100%→0%. Achieved 5 charge cycles and 6 discharge cycles. Buying at average €90/MWh, selling at €120/MWh generates x profit with y ROI.")
            - profit: Total profit (should be MUCH higher with continuous trading!)
            - charge_mw, discharge_mw, grid_buy_mw, grid_sell_mw: Full 24-hour schedules
            - soc: State of charge trajectory showing multiple cycles
            - condifence: Your confidence in this solution (0 to 1)

            **VERIFICATION:**
            - Count non-zero operations: Should see 15-20 hours of activity, not 6-8!
            - Check cycles: Should see SoC go up and down multiple times
            - Check profit: With good spread (€87→€147), expect €40-60 profit for 1MWh battery
            '''
        )
        
        # Run optimization
        print("\n" + "="*60)
        print("Running LLM-based arbitrage optimization...")
        print("="*60)
        
        result = await (target << source)
        
        print("\n" + "="*60)
        print("Optimization Complete")
        print("="*60)
        
        return result, inputs, battery
        
    except Exception as e:
        print(f"💥 Error in arbitrage optimization: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
