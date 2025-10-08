# milp_mcp_client_direct.py
import os, sys, asyncio
import warnings
import contextlib
import io
from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter
import json
from agentics import Agentics as AG

from energy_arbitrage.agentic_energy.schemas import (
    BatteryParams, DayInputs, SolveRequest, SolveResponse,
    EnergyDataRecord, SolveFromRecordsRequest,
)
# Point to your server file
# SERVER_PATH = os.getenv("MCP_MILP_SERVER", "milp_mcp_server.py")

# Comprehensive error suppression for CrewAI stream issues
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")

# Suppress stderr during cleanup to hide the CrewAI stream error
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    old_stderr = sys.stderr
    sys.stderr = mystderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "energy_arbitrage.agentic_energy.milp.milp_mcp_server"],
    env=os.environ,
)

async def main():
    # Start server and get tool adapter
    try:
        with MCPServerAdapter(params) as tools:
            print("✅ Connected to MCP server")
            print("🛠️  Available tools:", [t.name for t in tools])

            def get_tool(name: str):
                for t in tools:
                    if t.name == name:
                        return t
                raise RuntimeError(f"Tool {name!r} not found")

            milp_solve = get_tool("milp_solve")

            # -------- A) Arrays path (SolveRequest) --------
            prices_buy = [0.12]*6 + [0.15]*6 + [0.22]*6 + [0.16]*6
            prices_sell = prices_buy  # or different if allowing export
            req = SolveRequest(
                battery=BatteryParams(
                    capacity_kwh=20.0, soc_init=0.5, soc_min=0.10, soc_max=0.90,
                    cmax_kw=6.0, dmax_kw=6.0, eta_c=0.95, eta_d=0.95, soc_target=0.5
                ),
                day=DayInputs(
                    prices_buy=[0.12]*6 + [0.15]*6 + [0.22]*6 + [0.16]*6,
                    demand_kw=[0.9]*24,
                    prices_sell=prices_sell,
                    allow_export=True,
                    dt_hours=1.0
                ),
                solver=None,
                solver_opts=None
                # solver_opts = {
                #     "TimeLimit": 300,        # Maximum solve time in seconds
                #     "MIPGap": 0.01,         # Stop when gap between best solution and bound < 1%
                #     "Threads": 4,           # Number of threads to use
                #     "OutputFlag": 1,        # 1 = show solver output, 0 = silent
                #     "LogToConsole": 1       # Print log to console
                # }
            )


            source = AG(
                atype = SolveRequest,
                tools = [milp_solve],
                states = [req]
            )
            target = AG(
                atype=SolveResponse,
                tools=[milp_solve],
                max_iter=1,
                verbose_agent=False,
                reasoning=False,
            )

            arr_res = await (target << source)

            print("\n— Arrays call —")
            print(arr_res.pretty_print())

    except Exception as e:
        print(f"💥 MCP client error: {e}")
        import traceback
        traceback.print_exc()

            # records = [
            #     EnergyDataRecord(timestamps=f"2025-01-01T{h:02d}:00:00Z", prices=p, consumption=c)
            #     for h, (p, c) in enumerate(
            #         [(0.12, 0.9)]*6 + [(0.15, 1.0)]*6 + [(0.22, 1.4)]*6 + [(0.16, 1.1)]*6
            #     )
            # ]
            # req2 = SolveFromRecordsRequest(
            #     battery=req.battery,
            #     records=records,
            #     dt_hours=1.0,
            #     allow_export=False,
            #     prices_sell=None,
            #     solver=None,
            #     solver_opts=None
            # )

            # rec_prompt = {
            #     "tool": "milp_solve_from_records",
            #     "args": req2.model_dump()  
            # }

            # rec_res = await (AG(
            #     atype=SolveResponse,
            #     tools=tools,
            #     max_iter=1,
            #     verbose_agent=False,
            #     reasoning=False,
            #     description="Run MILP with EnergyDataRecord rows; return SolveResponse.",
            # ) << [rec_prompt])

            # print("\n— Records call —")
            # rec_res.pretty_print()


if __name__ == "__main__":
    try:
        # Run the main function
        asyncio.run(main())
        print("\n🎉 Client completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n💥 Application error: {e}")
    finally:
        # Suppress any cleanup errors from CrewAI
        with suppress_stderr():
            import time
            time.sleep(0.2)  # Give time for cleanup
        print("👋 Goodbye!")

