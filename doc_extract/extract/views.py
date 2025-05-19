import os
import json
import re
import glob
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from .forms import PDFUploadForm
from .models import UploadedPDF
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
api_key = os.getenv("OPENAI_API_KEY")
end_point = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize model
model = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=end_point,
    api_key=api_key,
    temperature=0,
)

# JSON parser
parser = JsonOutputParser()

# Fix AIMessage to raw string
def fix_json_output(text):
    if not isinstance(text, str):
        text = str(text.content if hasattr(text, "content") else text)
    match = re.search(r'(\[|\{).*(\]|\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON output")
    raise ValueError("No JSON structure found")

# Prompt template for table extraction
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
    - ONLY RESPOND WITH THE JSON.
    - Format your response as a valid, parsable JSON array.
    - Start your response with '[' and end with ']'
    
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# LangChain pipeline
chain = template | model | RunnableLambda(lambda x: fix_json_output(x))

# Convert JSON data to CSV
def json_to_csv(json_data, path):
    df = pd.DataFrame(json_data)
    df.to_csv(path, index=False)
    return path

# Main view
def extract_view(request):
    csv_paths = []
    extracted_texts = []
    image_urls = []

    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_instance = form.save()
            pdf_path = pdf_instance.file.path

            # Partition PDF and extract tables/images
            output_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_data')
            os.makedirs(output_dir, exist_ok=True)

            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res", # do "fast" for more processing speed 
                extract_images_in_pdf=True,
                extract_image_block_types=["Table"],
                extract_image_block_output_dir=output_dir,
            )

            # Extract all saved table images (for preview)
            for img_file in sorted(glob.glob(os.path.join(output_dir, '*'))):
                rel_path = os.path.relpath(img_file, settings.MEDIA_ROOT)
                image_urls.append(os.path.join(settings.MEDIA_URL, rel_path))

            # Extract table texts from PDF
            table_texts = [str(el) for el in elements if "Table" in str(type(el))]

            # Process each table text
            for i, raw_text in enumerate(table_texts):
                try:
                    result = chain.invoke({"pdf_content": raw_text})
                    extracted_texts.append({
                        'index': i + 1,
                        'content': json.dumps(result, indent=2)
                    })
                    output_path = os.path.join(settings.MEDIA_ROOT, f"output_table_{i+1}.csv")
                    json_to_csv(result, output_path)
                    csv_paths.append({
                        'index': i + 1,
                        'url': os.path.join(settings.MEDIA_URL, f"output_table_{i+1}.csv")
                    })
                except Exception as e:
                    extracted_texts.append({
                        'index': i + 1,
                        'content': f"Error processing table {i+1}: {str(e)}"
                    })
    else:
        form = PDFUploadForm()

    return render(request, 'extract/extract.html', {
        'form': form,
        'extracted_texts': extracted_texts,
        'csv_paths': csv_paths,
        'image_urls': image_urls,
    })
