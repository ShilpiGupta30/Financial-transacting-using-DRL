from fastapi import APIRouter

from ..schemas import SchedulerItem, SchedulerListResponse

router = APIRouter()


@router.get("/schedulers", response_model=SchedulerListResponse)
def schedulers() -> SchedulerListResponse:
    """List algorithms implemented by the C++ engine and report DQN honestly."""
    return SchedulerListResponse(
        items=[
            SchedulerItem(
                id="fcfs",
                name="First Come First Served",
                type="non-preemptive",
                available=True,
                supports_time_quantum=False,
                description="Runs processes in arrival order.",
            ),
            SchedulerItem(
                id="round_robin",
                name="Round Robin",
                type="preemptive",
                available=True,
                supports_time_quantum=True,
                description="Shares CPU time using a configurable time quantum.",
            ),
            SchedulerItem(
                id="priority",
                name="Priority Scheduling",
                type="non-preemptive",
                available=True,
                supports_time_quantum=False,
                description="Runs the highest-priority ready process first.",
            ),
            SchedulerItem(
                id="dqn",
                name="Deep Q-Network",
                type="reinforcement-learning",
                available=False,
                supports_time_quantum=False,
                description="A Python agent implementation exists, but no verified inference path is connected to the engine.",
                reason="No trained model artifact or C++ to Python inference path is available.",
            ),
        ]
    )
