from bs4 import BeautifulSoup
import json
import re
import os
import asyncio
from playwright.async_api import async_playwright
from typing import Dict, List, Any
from urllib.parse import urljoin
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Load URLs from JSON file
with open('city_of_ottawa_pages.json', 'r') as f:
    data = json.load(f)
    urls = data['city_of_ottawa_pages']  # Get the array of URLs

class KnowledgeBaseParser:
    def __init__(self, html_content: str):
        self.soup = BeautifulSoup(html_content, 'html.parser')
        
    def extract_title(self) -> str:
        """Extract the main title from the page."""
        title = self.soup.find('h1', class_='page-title')
        if title:
            return title.get_text(strip=True)
        return ""

    def extract_section_content(self, section: BeautifulSoup) -> Dict[str, Any]:
        """Extract content from a section including text and tables."""
        content = []
        
        # Find all text blocks and article references
        text_blocks = section.find_all(['div', 'article'], class_=['clearfix text-formatted', 'node--type-article', 'node--type-service'])
        
        for block in text_blocks:
            section_content = {
                "description": "",
                "details": []
            }
            
            # Handle both direct paragraphs and nested article content
            paragraphs = block.find_all(['p', 'ul', 'ol', 'h3'])
            current_heading = None
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                if not text:
                    continue
                    
                # Skip external links text markers
                if text == "(link is external)":
                    continue
                    
                if p.name == 'h3':
                    # Add heading as a detail with special formatting
                    current_heading = text
                    section_content["details"].append(f"Heading: {text}")
                elif p.name == 'p':
                    # Extract text without the external link marker
                    links = p.find_all('a')
                    if links:
                        # If paragraph contains links, extract them properly
                        link_texts = []
                        for link in links:
                            link_text = link.get_text(strip=True)
                            if link_text and link_text != "(link is external)":
                                link_texts.append(link_text)
                        if link_texts:
                            text = " | ".join(link_texts)
                    
                    if not section_content["description"]:
                        section_content["description"] = text
                    else:
                        if current_heading:
                            section_content["details"].append(text)
                        else:
                            section_content["details"].append(text)
                elif p.name in ['ul', 'ol']:
                    list_items = []
                    for li in p.find_all('li', recursive=True):
                        li_text = li.get_text(strip=True)
                        if li_text and li_text != "(link is external)":
                            # Check if this li contains a nested list
                            nested_list = li.find(['ul', 'ol'])
                            if nested_list:
                                # Only get the direct text of the li, without its nested list
                                direct_text = ''.join(t.strip() for t in li.strings 
                                                    if t.parent.name == 'li' and t.strip() != "(link is external)")
                                list_items.append(direct_text.strip())
                                # Add nested items with indentation
                                for nested_li in nested_list.find_all('li', recursive=False):
                                    nested_text = nested_li.get_text(strip=True)
                                    if nested_text and nested_text != "(link is external)":
                                        list_items.append(f"  - {nested_text}")
                            else:
                                # Handle links in list items
                                links = li.find_all('a')
                                if links:
                                    link_texts = []
                                    for link in links:
                                        link_text = link.get_text(strip=True)
                                        if link_text and link_text != "(link is external)":
                                            link_texts.append(link_text)
                                    if link_texts:
                                        list_items.append(" | ".join(link_texts))
                                else:
                                    list_items.append(li_text)
                    
                    if list_items:
                        if current_heading:
                            section_content["details"].extend(list_items)
                        else:
                            section_content["details"].extend(list_items)
            
            if section_content["description"] or section_content["details"]:
                content.append(section_content)

        # Find and parse tables
        tables = section.find_all('table')
        if tables:
            for table in tables:
                table_data = {
                    "name": table.find('caption').get_text(strip=True) if table.find('caption') else "",
                    "infractions": []
                }
                
                rows = table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        # Extract fine amounts with better error handling
                        early_payment_text = cols[1].get_text(strip=True)
                        set_fine_text = cols[2].get_text(strip=True)
                        
                        try:
                            early_payment = int(re.sub(r'[^\d]', '', early_payment_text)) if early_payment_text else 0
                        except ValueError:
                            early_payment = 0
                            
                        try:
                            set_fine = int(re.sub(r'[^\d]', '', set_fine_text)) if set_fine_text else 0
                        except ValueError:
                            set_fine = 0
                            
                        infraction = {
                            "violation": cols[0].get_text(strip=True),
                            "early_payment": early_payment,
                            "set_fine": set_fine
                        }
                        table_data["infractions"].append(infraction)
                
                if table_data["infractions"]:
                    content.append(table_data)

        return content

    def parse(self) -> Dict[str, Any]:
        """Parse the entire HTML document into a structured JSON format."""
        result = {
            "title": self.extract_title(),
            "sections": []
        }

        # Find all collapsible sections
        sections = self.soup.find_all('div', class_='collapse-wrapper')
        
        for section in sections:
            # Get section title
            title_elem = section.find_previous('h2')
            if title_elem:
                section_data = {
                    "title": title_elem.get_text(strip=True),
                    "content": self.extract_section_content(section)
                }
                result["sections"].append(section_data)

        return result

async def fetch_content(url: str) -> str:
    """Fetch content from a URL using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        try:
            # Navigate to the page and wait for the content to load
            await page.goto(url)
            await page.wait_for_selector('div.region.region-content')
            
            # Extract the specific content we want
            content = await page.query_selector('div.region.region-content')
            html_content = await content.inner_html()
            
            return html_content
        finally:
            await browser.close()

def generate_qa_pairs(content: str) -> str:
    """Use Gemini to generate Q/A pairs from content."""
    prompt = """Analyze this official documentation and generate realistic user questions with concise answers. 
Format responses EXACTLY like:

MESSAGE user "[question]"
MESSAGE assistant "[direct answer]"

Requirements:
1. Answers should be factual and contain only specific information from the content
2. Never use phrases like "according to documentation" or "City of Ottawa states"
3. Include exact numbers, dates, or amounts when present in content
4. Keep answers under 2 sentences
5. Use natural variations of the same information for different questions"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            "You are a municipal information extractor creating clean Q/A pairs",
            prompt,
            content
        ]
    )
    return response.text

def save_qa_pairs(qa_text: str, output_file: str):
    """Save Q/A pairs to training file with proper formatting."""
    print("\nSaving Q/A pairs:")
    with open(output_file, 'a', encoding='utf-8') as f:
        messages = qa_text.strip().split("\n")
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i].strip()
                assistant_msg = messages[i + 1].strip()
                
                # Skip empty messages or invalid formats
                if not user_msg or not assistant_msg:
                    continue
                    
                # Remove any remaining citations
                assistant_msg = assistant_msg.replace("According to City of Ottawa documentation", "")
                assistant_msg = assistant_msg.replace("per City guidelines", "")
                assistant_msg = re.sub(r'\[.*?\]', '', assistant_msg).strip()
                
                # Validate message format
                if not user_msg.startswith("MESSAGE user") or not assistant_msg.startswith("MESSAGE assistant"):
                    continue
                    
                print(f"\n{user_msg}\n{assistant_msg}")
                
                # Write pair to file with single newline between pairs
                f.write(f"{user_msg}\n{assistant_msg}\n\n")

async def process_url(url: str, training_file: str):
    """Process URL and save Q/A pairs to training file."""
    print(f"\nProcessing URL: {url}")
    try:
        html_content = await fetch_content(url)
        print(f"Successfully fetched HTML content for: {url}")
        parser = KnowledgeBaseParser(html_content)
        parsed_data = parser.parse()
        print(f"Successfully parsed data for: {url}")
        
        # Convert parsed data to text for Gemini
        content_text = json.dumps(parsed_data, indent=2)
        
        # Generate and save Q/A pairs
        qa_text = generate_qa_pairs(content_text)
        save_qa_pairs(qa_text, training_file)
        print(f"Successfully generated and saved Q/A pairs for: {url}")
        
    except Exception as e:
        print(f"Error processing URL '{url}': {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def process_file(input_file: str, output_file: str):
    """Process an HTML file and save the result as JSON."""
    try:
        # Read HTML file
        with open(input_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Parse HTML
        parser = KnowledgeBaseParser(html_content)
        result = parser.parse()

        # Save JSON output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully processed {input_file} to {output_file}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def clean_training_data(input_file: str, output_file: str):
    """Clean training data by removing low-quality entries."""
    cleaned_examples = []
    
    def is_valid_example(example: Dict[str, str]) -> bool:
        """Check if an example meets quality criteria."""
        # Check if response is too short
        if len(example['response'].strip()) < 20:
            return False
            
        # Check if response contains incomplete sentences or lists
        if example['response'].strip().endswith(('...', ',')):
            return False
            
        # Check if response is just a list of links
        if example['response'].count('(link is external)') > 3:
            return False
            
        # Check if response is mostly repeated content
        lines = example['response'].split('\n')
        unique_lines = set(lines)
        if len(lines) > 10 and len(unique_lines) < len(lines) * 0.5:
            return False
            
        # Check if response contains actual content (not just navigation elements)
        if all(line.startswith(('Browse ', 'View ', 'Register ')) for line in lines if line.strip()):
            return False
            
        return True

    try:
        # Read existing examples
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if is_valid_example(example):
                        cleaned_examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        # Write cleaned examples
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in cleaned_examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"Cleaned training data: {len(cleaned_examples)} examples retained")
        
    except Exception as e:
        print(f"Error cleaning training data: {str(e)}")

# Main execution
if __name__ == "__main__":
    print(f"Loaded URLs: {json.dumps(urls, indent=2)}")
    training_file = "model_training.txt"
    
    # Clear existing training file
    if os.path.exists(training_file):
        os.remove(training_file)

    # Process URLs and generate training data
    async def process_all_urls():
        for url in urls:
            print(f"Processing: {url}")
            await process_url(url, training_file)
    
    asyncio.run(process_all_urls())