# pricing.py

# Preços da API da OpenAI em USD por 1.000 tokens (valores de Outubro de 2025)
# Fonte: https://openai.com/pricing
PRICES = {
    "gpt-3.5-turbo": {
        "prompt": 0.0005,
        "completion": 0.0015
    },
    "text-embedding-3-small": {
        "usage": 0.00002
    }
}

def calculate_cost(model_name, prompt_tokens, completion_tokens=0):
    """
    Calcula o custo de uma chamada de API com base no modelo e no número de tokens.
    """
    if model_name not in PRICES:
        return 0.0

    model_prices = PRICES[model_name]

    if "usage" in model_prices:
        # Para modelos de embedding
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * model_prices["usage"]
    else:
        # Para modelos de chat
        prompt_cost = (prompt_tokens / 1000) * model_prices.get("prompt", 0)
        completion_cost = (completion_tokens / 1000) * model_prices.get("completion", 0)
        cost = prompt_cost + completion_cost
    
    return cost