# milp_mcp_client_direct.py (adapter form without .call_tool)
import os, sys, asyncio
from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

from agentic_energy.schemas import (
    BatteryParams, DayInputs, SolveRequest, SolveResponse,
    EnergyDataRecord, SolveFromRecordsRequest,
)

load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agentic_energy.milp.milp_mcp_server"],
    env=os.environ,
)

async def main():
    with MCPServerAdapter(params) as tools:
        print("Tools:", [t.name for t in tools])

        def get_tool(name: str):
            for t in tools:
                if t.name == name:
                    return t
            raise RuntimeError(f"Tool {name!r} not found")

        milp_solve = get_tool("milp_solve")
        # milp_solve_from_records = get_tool("milp_solve_from_records")

        # Arrays
        req = SolveRequest(
            battery=BatteryParams(
                capacity_kwh=20.0, soc_init=0.5, soc_min=0.10, soc_max=0.90,
                cmax_kw=6.0, dmax_kw=6.0, eta_c=0.95, eta_d=0.95, soc_target=0.5
            ),
            day=DayInputs(
                prices_buy=[0.12]*6 + [0.15]*6 + [0.22]*6 + [0.16]*6,
                demand_kw=[0.9]*24, allow_export=False, dt_hours=1.0
            ),
            solver=None, solver_opts=None
        )

        # Most adapter tool objects expose one of these; try in this order:
        call_fn = getattr(milp_solve, "call", None) or getattr(milp_solve, "run", None) or getattr(milp_solve, "__call__", None)
        if call_fn is None:
            raise RuntimeError("Adapter tool object has no call/run/callable interface")

        raw = call_fn(args=req.model_dump())
        res = SolveResponse(**raw)
        print("\n— Arrays call —")
        print("Status:", res.status, "Objective ($):", res.objective_cost)

        # Records
        # records = [
        #     EnergyDataRecord(timestamps=f"2025-01-01T{h:02d}:00:00Z", prices=p, consumption=c)
        #     for h,(p,c) in enumerate([(0.12,0.9)]*6 + [(0.15,1.0)]*6 + [(0.22,1.4)]*6 + [(0.16,1.1)]*6)
        # ]
        # req2 = SolveFromRecordsRequest(
        #     battery=req.battery, records=records, dt_hours=1.0,
        #     allow_export=False, prices_sell=None, solver=None, solver_opts=None
        # )
        # call_fn2 = getattr(milp_solve_from_records, "call", None) or getattr(milp_solve_from_records, "run", None) or getattr(milp_solve_from_records, "__call__", None)
        # raw2 = call_fn2(args=req2.model_dump())
        # res2 = SolveResponse(**raw2)
        # print("\n— Records call —")
        # print("Status:", res2.status, "Objective ($):", res2.objective_cost)

if __name__ == "__main__":
    asyncio.run(main())
