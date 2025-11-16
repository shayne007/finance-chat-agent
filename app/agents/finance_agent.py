from typing import List, Dict, Optional


class FinanceAgent:
    def __init__(self):
        pass

    async def run(self, user_input: str, conversation_history: Optional[List[Dict]] = None) -> str:
        text = user_input.lower()
        if any(k in text for k in ["budget", "save", "saving", "savings"]):
            return "Consider allocating 20% of income to savings, 50% to needs, 30% to wants (50/30/20 rule)."
        if any(k in text for k in ["invest", "investment", "portfolio"]):
            return "Diversify across index funds, bonds, and a small allocation to alternatives based on your risk tolerance."
        if any(k in text for k in ["retire", "retirement", "401k", "ira"]):
            return "Max out tax-advantaged accounts like 401(k) and IRA; target a savings rate aligned to your retirement timeline."
        if any(k in text for k in ["debt", "loan", "credit"]):
            return "Use the avalanche method: pay off highest-interest debt first while making minimums on others."
        return "I can help with budgeting, investing, retirement planning, and debt strategies. What is your goal?"