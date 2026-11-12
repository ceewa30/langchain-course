import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import traceable


load_dotenv(override=True)

print("Tracing enabled:", os.getenv("LANGSMITH_TRACING"))
print("API key loaded:", bool(os.getenv("LANGSMITH_API_KEY")))


MAX_ITERATIONS = 10
MODEL = "openai:gpt-4o-mini"
# MODEL ="google_genai:gemini-3.8-flash"


# ---- Tools (LangChain @tool decorator) ----
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1000,
        "phone": 500,
        "tablet": 300,
        "headphones": 149.95,
        "keyboard": 89.50,
    }
    return prices.get(product.lower(), 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold"""
    print(
        f" >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


# ------- Agent Loop ------
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(MODEL, temperature=0)
    # llm = init_chat_model(f"google_genai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 80)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned.\n"
                "3. Never calculate discounts yourself. Always use the apply_discount tool."
                "You are a helpful shopping assistant. "
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If the LLM didn't request any tool calls, we have our final answer
        if not response.tool_calls:
            print(f"\nFinal Answer:\n{response.content}")
            return response.content

        # Execute each requested tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            selected_tool = tools_dict.get(tool_name)

            if selected_tool is None:
                raise ValueError(f"Tool '{tool_name}' not found")
                
            observation = selected_tool.invoke(tool_args)

            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )

    print("Reached max iterations without a final answer.")
    return None


if __name__ == "__main__":
    try:
        print("Hello LangChain Agent (.bind_tools)!")
        print()
        run_agent("What is the price of a laptop after applying a gold discount?")
    finally:
        # Ensures background LangSmith trace uploads complete before script exit
        wait_for_all_tracers()

