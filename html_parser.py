from bs4 import BeautifulSoup
import json
import re
import os
import asyncio
from playwright.async_api import async_playwright
from typing import Dict, List, Any
from urllib.parse import urljoin

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

async def process_url(url: str, output_file: str):
    """Process a URL and save the result as JSON."""
    try:
        # Fetch HTML content
        print(f"Fetching content from {url}")
        html_content = await fetch_content(url)

        # Parse HTML
        parser = KnowledgeBaseParser(html_content)
        result = parser.parse()

        # Save JSON output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully processed URL to {output_file}")
            
    except Exception as e:
        print(f"Error processing URL: {str(e)}")

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

# Main execution
if __name__ == "__main__":
    # Read URLs from parking_links.json
    try:
        with open('city_of_ottawa_pages.json', 'r') as f:
            parking_data = json.load(f)
            urls = parking_data.get('city_of_ottawa_pages', [])
    except Exception as e:
        print(f"Error reading parking_links.json: {str(e)}")
        urls = []

    output_directory = "./rawdata"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    
    # Process each URL
    async def process_all_urls():
        for url in urls:
            # Extract the last part of the URL to use as filename
            filename = url.rstrip('/').split('/')[-1] + '.json'
            output_file = os.path.join(output_directory, filename)
            
            print(f"\nProcessing: {url}")
            await process_url(url, output_file)

    # Run the async function to process all URLs
    asyncio.run(process_all_urls())