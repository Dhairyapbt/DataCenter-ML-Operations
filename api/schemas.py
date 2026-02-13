from pydantic import BaseModel


class SensorInput(BaseModel):
    mean_temperature: float
    max_temperature: float
    std_temperature: float
    mean_power_load: float
    mean_vibration: float
    humidity_mean: float
    maintenance_count: int
    corrective_ratio: float
    avg_maintenance_cost: float
    capacity_kw: float
    asset_age_days: int
    asset_type: str
