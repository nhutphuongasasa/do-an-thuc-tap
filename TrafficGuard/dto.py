from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class PacketEvent:
    ts: float                 
    flow_key: Tuple           
    direction: str            
    length: int                
    flags: Dict[str, int]      
    protocol: str             


@dataclass
class FeatureVector:
    flow_key: Tuple
    window_size: int           
    ts: float                  
    values: Dict[str, float]    


@dataclass
class Prediction:
    flow_key: Tuple
    window_size: int
    ts: float
    label: str
    confidence: float