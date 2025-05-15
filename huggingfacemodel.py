from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableLambda, RunnableMap
from dotenv import load_dotenv
import os
import json
import csv
import pandas as pd
import re

# Load environment variables for Hugging Face
load_dotenv()

# Get Hugging Face API token
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face model with parameters passed directly
model = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token=hf_api_token,
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.95,
    do_sample=False,  # More deterministic output
    return_full_text=False  # Don't repeat the input in the output
)

# Create a function to fix potentially malformed JSON
def fix_json_output(text):
    # Debug: print raw output
    print("Raw model output:", text)
    
    # Find the JSON part - everything between first { or [ and last } or ]
    json_match = re.search(r'(\[|\{).*(\]|\})', text, re.DOTALL)
    if json_match:
        potential_json = json_match.group(0)
        try:
            # Try to parse it to make sure it's valid
            parsed_json = json.loads(potential_json)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            raise Exception(f"Could not extract valid JSON: {e}")
    else:
        raise Exception("No JSON-like structure found in the output")

# Use a more explicit JSON output parser
parser = JsonOutputParser()

# Make the prompt even more explicit about JSON formatting
template = PromptTemplate.from_template(
    """
    You are a helpful assistant that can convert raw table-like text into structured JSON.

    The raw extracted table content is:
    {pdf_content}

    Instructions:
    - Identify rows and columns in the table.
    - Create a list of dictionaries, where each dictionary represents one row.
    - Each key in the dictionary should be a column header.
    - Use the same column headers for all rows.
    - ONLY RESPOND WITH THE JSON. Do not include any explanation or other text.
    - Format your entire response as a valid, parsable JSON array.
    - Start your response with '[' and end with ']'
    
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Chain the components together with the fixing function
chain = template | model | RunnableLambda(fix_json_output)


def json_to_csv(json_data, output_file="extracted_data_huggingface.csv"):
    """
    Convert JSON data to CSV format and save it to a file
    
    Args:
        json_data (dict): The JSON data to convert
        output_file (str): Path to save the CSV file
    """
    # For nested JSON structures, flatten to one row
    # Convert lists to comma-separated strings
    flattened_data = {}
    
    def flatten_json(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    flatten_json(value, new_key)
                else:
                    flattened_data[new_key] = value
        elif isinstance(data, list):
            if all(not isinstance(item, (dict, list)) for item in data):
                # If simple list, join as string
                flattened_data[prefix] = ", ".join(str(item) for item in data)
            else:
                # For complex lists, add index to key
                for i, item in enumerate(data):
                    flatten_json(item, f"{prefix}_{i}")
    
    # Handle different data structures
    if isinstance(json_data, list):
        # Multiple entries - use pandas
        df = pd.DataFrame(json_data)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        # Single entry - direct flattening
        flatten_json(json_data)
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(flattened_data.keys())
            # Write values
            writer.writerow(flattened_data.values())
        print(f"Data saved to {output_file}")

    return output_file


if __name__ == "__main__":
    pdf_text = """
   'Food item Unit Global average water footprint (litres) Apple or pear 1 kg 700 Banana 1 kg 860 Beef 1 kg 15,500 Beer (from barley) 1 glass of 250 ml 75 Bread (from wheat) 1 kg 1,300 Cabbage 1 kg 200 Cheese 1 kg 5,000 Chicken 1 kg 3,900 Chocolate 1 kg 24,000 Coffee 1 cup of 125 ml 140 Cucumber or pumpkin 1 kg 240 Dates 1 kg 3,000 Groundnuts (in shell) 1 kg 3,100 Lettuce 1 kg 130 Maize 1 kg 900 Mango 1 kg 1,600 Milk 1 glass of 250 ml 250 Olives 1 kg 4,400 Orange 1 kg 460 Peach or nectarine 1 kg 1,200 Pork 1 kg 4,800 Potato 1 kg 250 Rice 1 kg 3,400 Sugar (from sugar cane) 1 kg 1,500 Tea 1 cup of 250 ml 30 Tomato 1 kg 180 Wine 1 glass of 125 ml 120'
    """

    try:
        result = chain.invoke({"pdf_content": pdf_text})
        print("\nParsed JSON Result:")
        print(result)
        
        # Convert and save to CSV
        csv_file = json_to_csv(result)
        print(f"\nConverted to CSV: {csv_file}")
    except Exception as e:
        print(f"Error processing document: {e}")