import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import pandas as pd
from tqdm import tqdm
import re
import concurrent.futures
import os
from typing import List, Dict, Set
import tiktoken
import hashlib
import urllib3
import time

class OttawaSiteScraper:
    def __init__(self, base_url="https://ottawa.ca"):
        self.base_url = base_url
        self.visited_urls = set()
        self.content_data = []
        self.session = requests.Session()
        # Add SSL verification handling
        self.session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Add error handling for tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            self.tokenizer = None
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to ottawa.ca domain."""
        parsed = urlparse(url)
        excluded_patterns = [
            '.pdf', '.jpg', '.png', '.gif', '.jpeg', '.doc', '.docx',
            'calendar', 'search', 'login', 'signin', 'signup',
            '/fr/', '/fr-ca/', 'mailto:', 'tel:', 'javascript:'
        ]
        return (
            parsed.netloc.endswith('ottawa.ca') and
            not any(pattern in url.lower() for pattern in excluded_patterns) and
            not url.endswith(('.css', '.js', '.xml', '.rss'))
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters while preserving important punctuation
        text = re.sub(r'[^\w\s.,?!;:()\-\'\"$%]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_page_content(self, url: str) -> Dict:
        """Extract and structure content from a single page."""
        try:
            # Add delay between requests
            time.sleep(1)  # 1 second delay
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove irrelevant elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find(class_=re.compile('(content|main)', re.I))
            if not main_content:
                return None
            
            # Extract structured data
            title = soup.title.string if soup.title else ''
            headings = [h.get_text().strip() for h in main_content.find_all(['h1', 'h2', 'h3'])]
            
            # Extract and clean content
            content = self.clean_text(main_content.get_text())
            
            # Extract metadata
            meta_description = ''
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_description = meta_tag.get('content', '')
            
            # Extract links for crawling
            links = {urljoin(self.base_url, a['href']) for a in soup.find_all('a', href=True)}
            valid_links = {link for link in links if self.is_valid_url(link)}
            
            return {
                'url': url,
                'title': self.clean_text(title),
                'meta_description': self.clean_text(meta_description),
                'headings': [self.clean_text(h) for h in headings if h.strip()],
                'content': content,
                'links': valid_links
            }
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return None
    
    def chunk_content(self, text: str, max_tokens: int = 1000) -> List[str]:
        """Split content into chunks of approximately max_tokens tokens."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_length += 1
            
            if current_length >= max_tokens:
                # Decode chunk and clean it
                chunk_text = self.tokenizer.decode(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
        
        # Add remaining tokens
        if current_chunk:
            chunk_text = self.tokenizer.decode(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def crawl(self, max_pages: int = 100) -> None:
        """Crawl the website and extract content."""
        to_visit = {self.base_url}
        failed_urls = set()  # Track failed URLs
        
        with tqdm(total=max_pages, desc="Crawling pages") as pbar:
            while to_visit and len(self.visited_urls) < max_pages:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    for _ in range(min(5, len(to_visit))):
                        if not to_visit:
                            break
                        url = to_visit.pop()
                        if url not in self.visited_urls and url not in failed_urls:
                            futures.append(executor.submit(self.extract_page_content, url))
                    
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            self.visited_urls.add(result['url'])
                            self.content_data.append(result)
                            to_visit.update(result['links'] - self.visited_urls - failed_urls)
                            pbar.update(1)
                        else:
                            failed_urls.add(url)  # Track failed URLs
                        
                        if len(self.visited_urls) >= max_pages:
                            break
        
        if failed_urls:
            print(f"\nFailed to process {len(failed_urls)} URLs")
    
    def save_content(self, output_dir: str = 'ottawa_data') -> None:
        """Save extracted content and prepare training data."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert content data links from sets to lists for JSON serialization
        serializable_content = []
        for item in self.content_data:
            item_copy = item.copy()
            item_copy['links'] = list(item_copy['links'])  # Convert set to list
            serializable_content.append(item_copy)
        
        # Save raw content
        with open(f'{output_dir}/content.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_content, f, ensure_ascii=False, indent=2)
        
        # Create CSV with metadata
        df = pd.DataFrame([{
            'url': item['url'],
            'title': item['title'],
            'meta_description': item['meta_description'],
            'content_length': len(item['content'])
        } for item in self.content_data])
        df.to_csv(f'{output_dir}/metadata.csv', index=False)
        
        # Prepare training data for Mistral
        training_data = []
        for item in self.content_data:
            # Create a content summary
            summary = f"Title: {item['title']}\n"
            if item['meta_description']:
                summary += f"Description: {item['meta_description']}\n"
            if item['headings']:
                summary += f"Topics: {' | '.join(item['headings'])}\n"
            summary += f"Source: {item['url']}\n\n"
            
            # Split content into chunks
            content_chunks = self.chunk_content(item['content'])
            
            # Create training examples
            for chunk in content_chunks:
                if len(chunk.strip()) > 100:  # Ignore very small chunks
                    # Create multiple training examples with different prompt patterns
                    examples = [
                        {
                            "input": f"What information can you provide about {item['title']}?",
                            "output": f"{summary}{chunk}"
                        },
                        {
                            "input": f"Tell me about {item['title']}.",
                            "output": f"{summary}{chunk}"
                        },
                        {
                            "input": "What services does the City of Ottawa provide in this area?",
                            "output": f"{summary}{chunk}"
                        }
                    ]
                    training_data.extend(examples)
        
        # Save training data
        with open(f'{output_dir}/training_data.json', 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

def create_modelfile(output_dir: str = 'ottawa_data') -> None:
    """Create a Modelfile for an Ollama custom model using the Mistral framework."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    modelfile_path = os.path.join(output_dir, 'Modelfile')

    modelfile_content = '''\
        FROM mistral

        # Optimized parameters for precise and reliable information retrieval
        PARAMETER temperature 0.3  # Controls randomness (0.0=deterministic, 1.0=creative)
        PARAMETER top_k 50        # Consider top 50 tokens for each prediction step
        PARAMETER top_p 0.9       # Choose from tokens covering 90% probability mass
        PARAMETER repeat_penalty 1.1  # Penalize repeated content (1.0=no penalty)
        PARAMETER num_ctx 4096    # Context window size (4k tokens for history+response)
        PARAMETER num_thread 8    # Use 8 CPU threads for parallel processing

        # System prompt optimized for a City of Ottawa assistant
        SYSTEM """You are OttawaGPT, a trusted AI assistant dedicated to providing accurate and official information 
        about City of Ottawa services and programs. Your responses must be:
        1. Fact-based and derived from verified City of Ottawa data.
        2. Clear, concise, and free of unnecessary technical jargon.
        3. Sensitive to bilingual (English and French) needs, noting when services are available in both languages.
        4. Professional, yet approachable in tone.
        5. Only include URLs that you are absolutely certain exist and are valid.

        Important URL guidelines:
        - Only use URLs that appear in your training data
        - The main website is https://ottawa.ca/en/ for English content
        - French content uses https://ottawa.ca/fr/
        - Never construct URLs - only use complete URLs from your training data
        - If unsure about a URL, direct users to https://ottawa.ca/en/ instead

        If you are ever uncertain about an answer:
        - Clearly acknowledge the uncertainty
        - Direct users to the main website: https://ottawa.ca/en/
        - Suggest contacting 3-1-1 for the most current information"""

        # Template for consistent, professional responses
        TEMPLATE """{{ .System }}

        User: {{ .Prompt }}
        Assistant: {{ .Response }}"""
        '''
    
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

def main():
    # Initialize scraper
    scraper = OttawaSiteScraper()
    
    print("Starting City of Ottawa website crawl...")
    scraper.crawl(max_pages=100)  # Adjust max_pages as needed
    
    print("Processing and saving content...")
    scraper.save_content()
    
    print("Creating Modelfile for Mistral...")
    create_modelfile()
    
    print("""
PoC Setup Complete!

Next steps:
1. Review the extracted content:
   - ottawa_data/content.json (raw data)
   - ottawa_data/metadata.csv (site structure)
   - ottawa_data/training_data.json (training examples)

2. Build the custom Mistral model:
   ollama create ottawa-assistant -f ottawa_data/Modelfile

3. Set up the chat interface:
   git clone https://github.com/schmitech/ollama-chat
   cd ollama-chat
   npm install
   npm run dev

Key improvements in this version:
- Better content chunking using token-aware splitting
- Enhanced metadata extraction and organization
- Improved prompt engineering for Mistral
- Bilingual service awareness
- Professional response formatting
- Comprehensive data cleaning and structuring

The model will provide accurate, sourced information about City of Ottawa services
with a professional yet approachable tone suitable for citizen interactions.
""")

if __name__ == "__main__":
    main()