import os
import gradio as gr
from huggingface_hub import InferenceClient,login
from transformers import pipeline

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. In Spaces, add it under Settings → Repository secrets.")

login(token=HF_TOKEN)


# --- Emissions factors --------------------------------------------------------
EMISSIONS_FACTORS = {
    "transportation": {"car": 2.3, "bus": 0.1, "train": 0.04, "plane": 0.25},
    "food": {"meat": 6.0, "vegetarian": 1.5, "vegan": 1.0},
}

def calculate_footprint(car_km, bus_km, train_km, air_km,
                        meat_meals, vegetarian_meals, vegan_meals):
    transport_emissions = (
        car_km * EMISSIONS_FACTORS["transportation"]["car"] +
        bus_km * EMISSIONS_FACTORS["transportation"]["bus"] +
        train_km * EMISSIONS_FACTORS["transportation"]["train"] +
        air_km * EMISSIONS_FACTORS["transportation"]["plane"]
    )
    food_emissions = (
        meat_meals * EMISSIONS_FACTORS["food"]["meat"] +
        vegetarian_meals * EMISSIONS_FACTORS["food"]["vegetarian"] +
        vegan_meals * EMISSIONS_FACTORS["food"]["vegan"]
    )
    total_emissions = transport_emissions + food_emissions
    stats = {
        "trees": round(total_emissions / 21),
        "flights": round(total_emissions / 500),
        "driving100km": round(total_emissions / 230)
    }
    return total_emissions, stats

# --- Default system prompt ----------------------------------------------------
system_message = """
You are Sustainable.ai, a friendly, encouraging, and knowledgeable AI assistant.
Always provide practical sustainability suggestions that are easy to adopt,
while keeping a supportive and positive tone. Prefer actionable steps over theory.
Reasoning: medium
"""

# --- Local pipeline (initialized once) ----------------------------------------
pipe = pipeline("text-generation", model="google/gemma-3-270m-it")

# --- Chat callback ------------------------------------------------------------
def respond(
    message,
    history: list[dict[str, str]],
    car_km,
    bus_km,
    train_km,
    air_km,
    meat_meals,
    vegetarian_meals,
    vegan_meals,
    use_local_model,   # checkbox
):
    # Compute personalized footprint summary
    footprint, stats = calculate_footprint(
        car_km, bus_km, train_km, air_km,
        meat_meals, vegetarian_meals, vegan_meals
    )

    custom_prompt = (
        f"This user’s estimated weekly footprint is **{footprint:.1f} kg CO2**.\n"
        f"That’s roughly planting {stats['trees']} trees 🌳 or taking {stats['flights']} short flights ✈️.\n"
        f"Breakdown includes transportation and food choices.\n"
        f"Your job is to give practical, friendly suggestions to lower this footprint.\n"
        f"{system_message}"
    )

    # Build chat context
    chat_context = custom_prompt + "\n"
    for turn in (history or []):
        role, content = turn["role"], turn["content"]
        chat_context += f"{role.upper()}: {content}\n"
    chat_context += f"USER: {message}\nASSISTANT:"

    # --- Local branch ---------------------------------------------------------
    if use_local_model:
        out = pipe(chat_context, max_new_tokens=300, do_sample=True)
        yield out[0]["generated_text"]
        return

    model_id = "openai/gpt-oss-20b"
    client = InferenceClient(model=model_id, token=HF_TOKEN)

    response = ""
    for chunk in client.chat_completion(
        [{"role": "system", "content": custom_prompt}] + (history or []) + [{"role": "user", "content": message}],
        max_tokens=3000,
        temperature=0.7,
        top_p=0.95,
        stream=True,
    ):
        token_piece = ""
        if chunk.choices and getattr(chunk.choices[0], "delta", None):
            token_piece = chunk.choices[0].delta.content or ""
        else:
            token_piece = getattr(chunk, "message", {}).get("content", "") or ""
        if token_piece:
            response += token_piece
            yield response

# --- UI -----------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    additional_inputs=[
        gr.Slider(0, 500, value=50, step=10, label="Car km/week"),
        gr.Slider(0, 500, value=20, step=10, label="Bus km/week"),
        gr.Slider(0, 500, value=20, step=10, label="Train km/week"),
        gr.Slider(0, 5000, value=200, step=50, label="Air km/week"),
        gr.Slider(0, 21, value=7, step=1, label="Meat meals/week"),
        gr.Slider(0, 21, value=7, step=1, label="Vegetarian meals/week"),
        gr.Slider(0, 21, value=7, step=1, label="Vegan meals/week"),
        gr.Checkbox(label="Use Local Model (google/gemma-3-270m-it)", value=False),
    ],
    title="🌱 Sustainable.ai",
    description=(
        "Chat with an AI that helps you understand and reduce your carbon footprint. "
        "Toggle 'Use Local Model' to run locally with google/gemma-3-270m-it, or leave it off "
        "to call Hugging Face Inference API (gpt-oss-20b)."
    ),
)

if __name__ == "__main__":
    demo.launch()
