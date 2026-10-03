"""Agent 3: Customer Reply Agent. Drafts the reply in the customer's language using ONLY the facts."""
from crewai import Agent, Task

LANGUAGE_RULE = {
    "english": "English",
    "roman_urdu": "Roman Urdu (Urdu written in English letters), the way the customer writes",
    "urdu": "Urdu",
    "mixed": "the same mix of English and Roman Urdu that the customer used",
}


def create_reply_agent(llm, shop_name: str) -> Agent:
    return Agent(
        role="Customer Support Writer",
        goal="Write a short, friendly and accurate reply to the customer using only the facts provided.",
        backstory=(
            f"You write chat replies for {shop_name}, a small clothing shop. "
            "You are warm and concise. You never invent prices, stock, sizes, delivery times "
            "or policies, and you never promise refunds, discounts or special exceptions."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )


def create_reply_task(agent: Agent, message: str, language: str, history: str, faq_task: Task,
                      owner_contact: str, examples: str = "") -> Task:
    style_block = ""
    if examples:
        style_block = (
            "\nSimilar past replies the shop owner wrote or approved (copy their TONE and wording style only; "
            "facts must still come ONLY from the previous task):\n"
            f"{examples}\n"
        )
    lang = LANGUAGE_RULE.get(language, "the same language the customer wrote in")
    return Task(
        description=(
            "Recent conversation (may be empty):\n"
            f"{history}\n\n"
            f"Customer's latest message: {message}\n"
            f"{style_block}\n"
            f"Write the reply in {lang}. Use ONLY the facts from the previous task. Rules:\n"
            "- Maximum 3 short sentences, friendly tone, at most one emoji.\n"
            "- Quote prices with the currency exactly as given.\n"
            "- If a size is out of stock, say so and suggest an available size if there is one.\n"
            "- If the facts say NO_MATCH or do not answer the question, say you are not sure and that "
            f"the shop owner will confirm ({owner_contact}). Do not guess.\n"
            "- Never promise refunds, discounts or exceptions."
        ),
        expected_output="Only the text of the reply to the customer. No labels, no quotes, no explanations.",
        agent=agent,
        context=[faq_task],
    )
