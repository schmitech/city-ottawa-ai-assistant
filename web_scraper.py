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
        
        # Prepare training data for model
        training_data = []
        seen_chunks = set()  # To avoid duplicate content
        
        def clean_text(text):
            """Clean and format text content."""
            # Remove navigation elements
            text = re.sub(r'On this page.*?\.\.\.', '', text, flags=re.DOTALL)
            # Remove UI elements
            text = re.sub(r'What you need.*?Apply', '', text, flags=re.DOTALL)
            # Remove redundant whitespace
            text = ' '.join(text.split())
            # Remove navigation artifacts
            text = re.sub(r'\.\.\.\s*$', '', text)
            return text.strip()
        
        # Improved parking-related content detection
        parking_keywords = {
            'nouns': ['parking', 'permit', 'garage', 'lot', 'space', 'zone', 'meter', 'valet'],
            'verbs': ['park', 'restrict', 'enforce', 'violate', 'ticket', 'tow'],
            'adj': ['resident', 'timed', 'paid', 'restricted', 'accessible']
        }
        
        # Enhanced instruction templates with variations
        instruction_templates = {
            'rates': [
                "What are the parking rates for {location}?",
                "How much does parking cost at {location}?",
                "What are the fees for parking in {area}?"
            ],
            'hours': [
                "What are the operating hours for {location} parking?",
                "When is parking allowed at {location}?",
                "What are the time restrictions for parking in {area}?"
            ],
            # ... other categories with similar variations
        }

        # Improved content chunk processing
        for item in self.content_data:
            # Enhanced parking content detection using combined criteria
            content_lower = ' '.join([item['title'], item['meta_description'], ' '.join(item['headings'])]).lower()
            is_parking_content = any(
                re.search(rf'\b{term}\b', content_lower)
                for term in parking_keywords['nouns'] + parking_keywords['verbs'] + parking_keywords['adj']
            ) and not any(excl in content_lower for excl in ['event parking', 'special occasion'])

            if not is_parking_content:
                continue

            # Initialize content_chunks for this item
            content_chunks = []

            # Improved chunking with sentence-aware splitting
            sentences = re.split(r'(?<=[.!?])\s+', item['content'])
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_tokens = self.tokenizer.encode(sentence)
                if current_length + len(sentence_tokens) > self.config['content']['max_tokens_per_chunk']:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunk_text = clean_text(chunk_text)
                        if len(chunk_text) >= self.config['content']['min_chunk_size']:
                            content_chunks.append(chunk_text)
                        current_chunk = []
                        current_length = 0
                current_chunk.append(sentence)
                current_length += len(sentence_tokens)
            
            # Process remaining sentences
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_text = clean_text(chunk_text)
                if len(chunk_text) >= self.config['content']['min_chunk_size']:
                    content_chunks.append(chunk_text)

            # Enhanced information extraction with context awareness
            for chunk in content_chunks:
                # Improved cleaning with more specific patterns
                chunk = re.sub(r'\b(?:Note:|Please note:|Important:).*$', '', chunk, flags=re.IGNORECASE)
                chunk = re.sub(r'\d{3}-\d{3}-\d{4}', '[PHONE]', chunk)  # Redact phone numbers
                chunk = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', chunk)
                
                if len(chunk) < self.config['content']['min_chunk_size']:
                    continue

                # Extract location context from headings
                location_matches = re.findall(r'\b(?:downtown|centretown|byward market|west end|east end)\b', 
                                            ' '.join(item['headings']), re.IGNORECASE)
                location = location_matches[0].title() if location_matches else "this location"

                # Create multiple question types per chunk with varied phrasing
                category_found = False
                
                # Rate detection with currency validation
                if re.search(r'\$(?:\d+\.?\d*|\.\d+)(?:\s*-\s*\$(?:\d+\.?\d*|\.\d+))?', chunk):
                    template = random.choice(instruction_templates['rates'])
                    training_data.append({
                        "system": "You are a knowledgeable parking advisor for Ottawa. Provide detailed, accurate information about parking regulations and services.",
                        "instruction": template.format(location=location, area=location),
                        "response": chunk,
                        "source_url": item['url'],
                        "content_type": "rate_info"
                    })
                    category_found = True
                
                # Time detection with hour format validation
                if re.search(r'\b(?:mon|tues|wednes|thurs|fri|satur|sun)day?s?\b.*?\d{1,2}(?::\d{2})?\s*[ap]\.?m?\.?', chunk, re.IGNORECASE):
                    template = random.choice(instruction_templates['hours'])
                    training_data.append({
                        "system": "You are a knowledgeable parking advisor for Ottawa. Provide detailed, accurate information about parking regulations and services.",
                        "instruction": template.format(location=location, area=location),
                        "response": chunk,
                        "source_url": item['url'],
                        "content_type": "operating_hours"
                    })
                    category_found = True
                
                # Add fallback general parking question
                if not category_found and is_parking_content:
                    training_data.append({
                        "system": "You are a knowledgeable parking advisor for Ottawa. Provide detailed, accurate information about parking regulations and services.",
                        "instruction": f"What should I know about parking regulations in {location}?",
                        "response": chunk,
                        "source_url": item['url'],
                        "content_type": "general_info"
                    })

        # Add data validation step
        valid_training_data = []
        seen_responses = set()
        for entry in training_data:
            response_hash = hashlib.md5(entry['response'].encode()).hexdigest()
            if (len(entry['response']) >= 100 and 
                response_hash not in seen_responses and
                not re.search(r'\[\w+\]', entry['response'])):  # Filter redacted content
                valid_training_data.append(entry)
                seen_responses.add(response_hash)
        
        # Save validated data
        training_file = f"{output_dir}/{self.config['output']['files']['training']}"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(valid_training_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nProcessed {len(self.content_data)} pages")
        print(f"Generated {len(valid_training_data)} validated instruction-response pairs")
        print(f"Training data saved to {training_file}")

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

if __name__ == "__main__":
    main()