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
import yaml
from queue import PriorityQueue
import random
from urllib.robotparser import RobotFileParser

class OttawaSiteScraper:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_url = self.config['scraper']['base_url']
        self.visited_urls = set()
        self.content_data = []
        self.session = requests.Session()
        # Add SSL verification handling
        self.session.verify = self.config['scraper']['verify_ssl']
        
        if not self.session.verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Add error handling for tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(self.config['content']['tokenizer'])
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            self.tokenizer = None
        
        # Add request headers
        self.session.headers.update({
            'User-Agent': self.config['scraper']['user_agent'],
            'Accept-Language': 'en-CA, fr-CA;q=0.8',
            'From': 'your-email@example.com',
            'Accept-Encoding': 'gzip',  # Reduce bandwidth
            'Connection': 'keep-alive',  # Reuse connections
        })
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to ottawa.ca domain."""
        parsed = urlparse(url)
        return (
            parsed.netloc in ['ottawa.ca', 'www.ottawa.ca'] and  # Only allow main domain
            not any(p in url.lower() for p in self.config['content']['excluded_patterns']) and
            not url.endswith(('.css', '.js', '.xml', '.rss')) and
            not re.search(r'(/\w{2}_[A-Z]{2}/|locale=\w{2}_[A-Z]{2})', url) and
            not re.search(r'/form/|/pay-or-purchase', parsed.path)  # Block form and payment paths
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content with null handling."""
        if not text:  # Handle None or empty input
            return ''
        
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
            # Add randomized delay with jitter
            base_delay = self.config['scraper']['request_delay']
            jitter = base_delay * self.config['scraper']['jitter']
            actual_delay = base_delay * self.config['scraper']['politeness'] + random.uniform(-jitter, jitter)
            time.sleep(max(0.5, actual_delay))  # Never less than 0.5s
            
            # Add retry logic
            for attempt in range(self.config['scraper']['max_retries']):
                response = self.session.get(url, timeout=self.config['scraper']['timeout'])
                if response.status_code == 429:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
                    continue
                break
            
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
            
            # Extract structured data - handle None case
            title_tag = soup.title
            title = title_tag.get_text(strip=True) if title_tag else url  # Use URL as fallback title
            
            # Safely extract headings
            headings = []
            for h in main_content.find_all(['h1', 'h2', 'h3']):
                heading_text = h.get_text(strip=True)
                if heading_text:
                    headings.append(heading_text)
            
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
            
            # Detect pagination
            pagination = soup.find('nav', {'aria-label': 'Pagination'})
            if pagination:
                for page_link in pagination.find_all('a', href=True):
                    absolute_link = urljoin(self.base_url, page_link['href'])
                    if self.is_valid_url(absolute_link):
                        valid_links.add(absolute_link)
            
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
    
    def chunk_content(self, text: str) -> List[str]:
        """Split content into chunks of approximately max_tokens tokens."""
        max_tokens = self.config['content']['max_tokens_per_chunk']
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
    
    def link_priority(self, url: str) -> int:
        priority = 1
        if '/programs/' in url: priority = 3
        if '/registration/' in url: priority = 5
        if '/facilities/' in url: priority = 2
        return priority
    
    def crawl(self) -> None:
        """Crawl the website and extract content."""
        max_pages = self.config['scraper']['max_pages']
        max_workers = self.config['scraper']['max_workers']
        to_visit = PriorityQueue()
        failed_urls = set()
        
        # Seed the queue with initial URL
        initial_priority = -self.link_priority(self.base_url)
        to_visit.put((initial_priority, self.base_url))
        
        # Add failure threshold
        max_consecutive_failures = 5
        consecutive_failures = 0
        
        with tqdm(total=max_pages, desc="Crawling pages") as pbar:
            while not to_visit.empty() and len(self.visited_urls) < max_pages:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for _ in range(min(5, to_visit.qsize())):
                        if to_visit.empty():
                            break
                        _, url = to_visit.get()
                        if url not in self.visited_urls and url not in failed_urls:
                            futures.append(executor.submit(self.extract_page_content, url))
                    
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:  # First check if result exists
                            if result['content'].strip():
                                self.visited_urls.add(result['url'])
                                self.content_data.append(result)
                                for link in result['links']:
                                    to_visit.put((-self.link_priority(link), link))
                                pbar.update(1)
                            else:
                                print(f"Skipping empty page: {result['url']}")
                                failed_urls.add(url)
                        else:
                            print(f"Failed to process URL: {url}")
                            failed_urls.add(url)
                        
                        if len(self.visited_urls) >= max_pages:
                            print(f"Reached max pages ({max_pages}). Stopping crawl.")
                            break
                        
                        if not result:
                            consecutive_failures += 1
                            if consecutive_failures >= max_consecutive_failures:
                                print("Too many consecutive failures. Stopping crawl.")
                                break
                        else:
                            consecutive_failures = 0
        
        if failed_urls:
            print(f"\nFailed to process {len(failed_urls)} URLs")
    
    def save_content(self) -> None:
        """Save extracted content and prepare training data."""
        output_dir = self.config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert content data links from sets to lists for JSON serialization
        serializable_content = []
        for item in self.content_data:
            item_copy = item.copy()
            item_copy['links'] = list(item_copy['links'])  # Convert set to list
            serializable_content.append(item_copy)
        
        # Save raw content
        with open(f"{output_dir}/{self.config['output']['files']['content']}", 'w', encoding='utf-8') as f:
            json.dump(serializable_content, f, ensure_ascii=False, indent=2)
        
        # Create CSV with metadata
        df = pd.DataFrame([{
            'url': item['url'],
            'title': item['title'],
            'meta_description': item['meta_description'],
            'content_length': len(item['content'])
        } for item in self.content_data])
        df.to_csv(f"{output_dir}/{self.config['output']['files']['metadata']}", index=False)
        
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
                if len(chunk.strip()) > self.config['content']['min_chunk_size']:  # Ignore very small chunks
                    training_data.append({
                        "input": f"What information can you provide about {item['title']}?",
                        "output": f"{summary}{chunk}"
                    })
                    training_data.append({
                        "input": f"Tell me about {item['title']}.",
                        "output": f"{summary}{chunk}"
                    })
                    training_data.append({
                        "input": "What services does the City of Ottawa provide in this area?",
                        "output": f"{summary}{chunk}"
                    })
        
        # Save training data
        with open(f"{output_dir}/{self.config['output']['files']['training']}", 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

    def check_robots_txt(self):
        robots_url = urljoin(self.base_url, "/robots.txt")
        response = self.session.get(robots_url)
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser

    def adaptive_throttling(self, response):
        remaining = int(response.headers.get('X-RateLimit-Remaining', 100))
        if remaining < 10:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 60))
            print(f"Approaching rate limit. Pausing for {reset_time} seconds")
            time.sleep(reset_time + 5)

def create_modelfile(config_path="config.yaml") -> None:
    """Create a Modelfile for an Ollama custom model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)
    modelfile_path = os.path.join(output_dir, config['output']['files']['modelfile'])

    # Load model configuration
    base_model = config['model']['base_model']
    print(f"\n[Model Configuration] Creating Modelfile for base model: {base_model}")

    # Load model instructions from file
    instruction_file = config['model']['instruction_file']
    try:
        with open(instruction_file, 'r', encoding='utf-8') as f:
            model_instructions = f.read()
    except FileNotFoundError:
        print(f"Error: Model instruction file not found: {instruction_file}")
        return
    except Exception as e:
        print(f"Error reading model instruction file: {e}")
        return

    model_params = config['model']['parameters'].get(base_model, {})
    
    # Build parameter string based on available parameters
    parameter_str = ""
    for param, value in model_params.items():
        parameter_str += f"PARAMETER {param} {value}\n        "

    # Fix the template string formatting by removing leading whitespace
    modelfile_content = f'''FROM {base_model}

# Model-specific parameters
{parameter_str}

{model_instructions}
'''  # Removed leading whitespace in the template
    
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

def main():
    config_path = "config.yaml"
    
    # Initialize scraper
    scraper = OttawaSiteScraper(config_path)
    
    # Verify robots.txt first
    robots = scraper.check_robots_txt()
    if not robots.can_fetch(scraper.config['scraper']['user_agent'], scraper.base_url):
        print(f"ERROR: Cannot crawl {scraper.base_url} per robots.txt")
        return
    
    print("Starting City of Ottawa website crawl...")
    scraper.crawl()
    
    print("Processing and saving content...")
    scraper.save_content()
    
    print("Creating Modelfile...")
    create_modelfile(config_path)
    
    # Add final model confirmation
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"\n=== Training Configuration ===")
    print(f"Base Model: {config['model']['base_model']}")
    print(f"Instruction File: {config['model']['instruction_file']}")
    print(f"Parameters: {json.dumps(config['model']['parameters'][config['model']['base_model']], indent=2)}")
    print("===============================")

if __name__ == "__main__":
    main()