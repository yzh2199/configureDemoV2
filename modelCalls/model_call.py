import logging
import google.generativeai as genai
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

def call_gemini(prompt:str):
    # Select the model - use a newer, capable model
    # model = genai.GenerativeModel('gemini-pro')
    # Using 1.5 Flash for potentially faster responses, adjust if needed
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Set safety settings to be less restrictive for this task if needed, but be cautious
    # safety_settings = [
    #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    # ]

    # Force JSON output if the model supports it (Check Gemini documentation)
    # Note: As of early 2024, direct JSON mode might be limited.
    # The prompt strongly instructs JSON output, which is often sufficient.
    response = model.generate_content(
        prompt,
        # safety_settings=safety_settings,
        generation_config=genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=['\n\n'], # Optional: Stop generation earlier
            # max_output_tokens=2048, # Adjust as needed
            temperature=0.1 # Lower temperature for more deterministic/factual output
        )
    )

    logging.info("----- Received Response from LLM -----")
    # Check for blocked response due to safety
    if not response.candidates:
        logging.error("LLM response blocked or empty.")
        # Try to get blocking reason if available
        block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
        return {"success": False, "message": f"LLM response was blocked or empty. Reason: {block_reason}"}

    raw_response_text = response.text
    logging.info(raw_response_text)
    logging.info("----- End of LLM Response -----")
    return raw_response_text

def call_deepseek_V3(prompt:str):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass(("Enter API key for deespeek: "))

    model = init_chat_model("deepseek-chat", model_provider="deepseek")

    response = model.invoke(prompt)
    logging.info(f"call deepseek finished, res is {response}")
    return response.text()

def call_deepseek_R1(prompt:str):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass(("Enter API key for deespeek: "))

    model = init_chat_model("deepseek-reasoner", model_provider="deepseek")

    response = model.invoke(prompt)
    logging.info(f"call deepseek finished, res is {response}")
    return response.text()
