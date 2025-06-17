import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import time

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)

start_time = time.time()
input_text = "Hello, how are you?"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = llm_model.generate(inputs, max_new_tokens=20)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, max_new_tokens=100)
print(generated_text)
end_time = time.time()
print(("Time for Eluether call: %s") %  (end_time - start_time))

start_time = time.time()
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"  # Replace with your API key
)

model_name =  "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF" 
combined_docs=['Ronaldo Cristiano']
retrieved_context = "\n".join(combined_docs)
question = 'who is Cristiano'

retrieved_context = "\n".join(combined_docs)
context = (f"You are a knowledgeable soccer fan. Answer the question in a friendly, engaging, and "
            f"enthusiastic manner, as if you are talking to a fellow sports fan. "
            f"Quote information from this context to complete your answer"
            f"{retrieved_context}")

prompt = (
    f"Context: {context}\n\n"
    "Summarize the response to be under 200 words"
    "Make sure to answer the question with a complete answer with an upbeat enthusiastic mood" 
    "include interesting facts or statistics about the topic or football if relevant."
    "Ask a follow up question based on the context provided to check if the user would like more information."
    "Avoid repeating the context or question, and keep the tone light and conversational."
    "Avoid any advise or profanity or discussion about political, illegal or controversial issues"
    "Remove any html tags or markdown tags from your response to make it easily speakble so it can be converted to speech"
)
# Prepare the message content
messages = [
    {"role": "user", "content": question},
    {"role": "system", "content": prompt}
]

# Check for empty content before making the API request
if not any(msg["content"].strip() for msg in messages):
    print("Error: Both question and context are empty. Please try again with a valid input.")

# Create a completion request
try:
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=messages,
        temperature=0.5,
        top_p=1,
        max_tokens=512,
        stream=True
    )

    # Collect the streamed response
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    print(f"LLM reponse is {response}")
    print(response)

except Exception as e:
    print(f"Error during API request: {e}")
    print("An error occurred while processing your request. Please try again.")
end_time = time.time()
print(("Time for Open call: %s") %  (end_time - start_time))
