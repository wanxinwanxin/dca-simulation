"""Gas fee models."""

from dataclasses import dataclass
from typing import Protocol


class GasModel(Protocol):
    """Protocol for gas fee calculation."""
    
    def gas_fee(self, gas_used: int) -> float:
        """Calculate gas fee for given gas usage."""
        ...


@dataclass(frozen=True)
class Evm1559:
    """EIP-1559 gas fee model."""
    
    base_fee: float  # base fee per gas in ETH
    tip: float  # priority fee per gas in ETH
    
    def gas_fee(self, gas_used: int) -> float:
        """Calculate total gas fee."""
        return (self.base_fee + self.tip) * gas_used 