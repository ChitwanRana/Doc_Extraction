from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableMap
from dotenv import load_dotenv
import os
import json
import csv
import pandas as pd

# Load environment variables for Azure OpenAI
load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
end_point=os.getenv("AZURE_OPENAI_ENDPOINT")
model_name=os.getenv("AZURE_OPENAI_MODEL_NAME")
api_version=os.getenv("AZURE_OPENAI_API_VERSION")
deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


model = AzureChatOpenAI(
    azure_deployment=deployment, 
    api_version=api_version,
    azure_endpoint=end_point,
    api_key=api_key,
    temperature=0,    
)

# Step 2: Setup the output parser
parser = JsonOutputParser()

# Step 3: Define the dynamic prompt
template = PromptTemplate.from_template(
    """
    You are a helpful assistant that can convert raw table-like text extracted from a PDF into a structured table.

    The raw extracted table content is:
    {pdf_content}

    Instructions:
    - Identify rows and columns from the text.
    - Structure the output as a list of rows in JSON format, where each row is a dictionary with column headers as keys.
    - If the table headers are not clear, infer meaningful column names.
    - Make sure the final output is a *valid JSON list of dictionaries*.

    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser


def json_to_csv(json_data, output_file="extracted_data_openai.csv"):
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
        print("JSON Result:")
        print(result)
        
        # Convert and save to CSV
        csv_file = json_to_csv(result)
        print(f"\nConverted to CSV: {csv_file}")
    except Exception as e:
        print(f"Error processing document: {e}")