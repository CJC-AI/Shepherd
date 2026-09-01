from simulation.clock import SimulationClock
from simulation.customer_state import CustomerState
from simulation.device_state import DeviceState


class DeviceGenerator:
    def __init__(self, clock: SimulationClock):
        self.clock = clock

    def generate_for_customer(
        self,
        customer: CustomerState,
    ) -> list[DeviceState]:

        first_seen = self.clock.now()

        return [
            DeviceState(
                device_id=device_id,
                customer_id=customer.customer_id,
                device_trust_score=0.5,
                first_seen=first_seen,
            )
            for device_id in customer.known_devices
        ]