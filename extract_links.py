import requests
from bs4 import BeautifulSoup
import json
import sys
import urllib.parse

def extract_links(url):
    try:
        # Send HTTP request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all 'a' tags that match the specific structure
        links = soup.find_all('a', attrs={'hreflang': 'en'})
        
        # Extract and process the href attributes
        city_links = []
        base_url = "https://ottawa.ca"
        
        for link in links:
            href = link.get('href')
            if href:
                # Convert relative URLs to absolute URLs
                if href.startswith('/'):
                    absolute_url = urllib.parse.urljoin(base_url, href)
                    city_links.append(absolute_url)
                elif href.startswith(base_url):
                    city_links.append(href)
        
        # Create the JSON structure
        output_data = {
            "city_of_ottawa_pages": city_links
        }
        
        # Write to JSON file
        with open('ottawa_links.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Successfully extracted {len(city_links)} links and saved to ottawa_links.json")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    extract_links(url)

if __name__ == "__main__":
    main()