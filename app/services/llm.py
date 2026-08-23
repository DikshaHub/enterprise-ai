from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def generate_answer(question: str, context: str):
    prompt = f"""
    You are an enterprise FAQ assistant.

    Answer the user's question using only the context.

    Rules:
    - Give only the answer.
    - Be concise.
    - Do not explain your reasoning.
    - Do not add a justification.
    - Do not discuss the context.
    - Do not ask for additional sources.
    - If the answer is not in the context, say exactly:
    "I couldn't find the answer in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer