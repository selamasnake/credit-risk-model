from pydantic import BaseModel
from typing import Optional

class CustomerTransaction(BaseModel):
    CustomerId: int
    TotalTransactionAmount: float
    AvgTransactionAmount: float
    TransactionCount: int
    TransactionStdAmount: float
    TransactionHour: Optional[int] = None
    TransactionDay: Optional[int] = None
    TransactionMonth: Optional[int] = None
    TransactionYear: Optional[int] = None
    # Add other engineered features your model expects

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: int  # 1 = High Risk, 0 = Low Risk
