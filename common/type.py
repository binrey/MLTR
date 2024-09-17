from enum import Enum


class Side(Enum):
    BUY = 1
    SELL = -1

    @staticmethod
    def from_str(side: str):
        if side.lower() == "buy":
            return Side.BUY
        elif side.lower() == "sell":
            return Side.SELL
        else:
            raise ValueError(f"{side} is not valid value, set ['buy' or 'sell']")

    @staticmethod
    def from_int(side: int):
        if side == 1:
            return Side.BUY
        elif side == -1:
            return Side.SELL
        else:
            raise ValueError(f"{side} is not valid value, set [1 or -1]")

    def __str__(self):
        return "BUY" if self == Side.BUY else "SELL"
    
if __name__ == "__main__":
    print(Side.from_str("buy"))
    print(Side.from_int(1))
    print(Side.BUY)
    print(Side.SELL)