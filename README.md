# Ottawa AI Assistant PoC

This project creates an AI-powered assistant for the City of Ottawa's website using web scraping and Mistral AI. The assistant helps citizens quickly find information without having to browse through multiple pages.

## Features

- Web scraping of ottawa.ca with intelligent content extraction
- Token-aware content chunking for optimal training
- Mistral AI model integration for natural language understanding
- Bilingual content awareness (English/French)
- Source URL preservation for verification
- Professional response formatting suitable for government services

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (for the chat interface)
- Ollama installed on your system ([Ollama Installation Guide](https://github.com/jmorganca/ollama))

## Project Structure

```
ottawa-ai-assistant/
├── ottawa_data/          # For storing scraped and generated data
├── src/
│   └── ottawa_scraper.py # Main scraping script
├── requirements.txt      # Project dependencies
├── venv/                # Virtual environment (git ignored)
└── .gitignore          # Git ignore file
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd ottawa-ai-assistant
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Run the scraper:
```bash
python src/ottawa_scraper.py
```

This will:
- Crawl the City of Ottawa website
- Extract and process content
- Create training data
- Generate a Modelfile for Mistral

3. Build the custom Mistral model:
```bash
ollama create ottawa-assistant -f ottawa_data/Modelfile
```

4. Set up the chat interface:
```bash
git clone https://github.com/schmitech/ollama-chat
cd ollama-chat
npm install
npm run dev
```

## Sample Queries

After setting up the chat interface, try these example queries:

```
# City Services & Programs
"What are the current recreation program registration dates?"
"How do I apply for a residential parking permit?"
"What documents do I need for a marriage license?"

# Waste & Recycling
"When is my next garbage collection day?"
"What items are accepted at the Trail Road waste facility?"
"How do I dispose of household hazardous waste?"

# Property & Development
"What are the current property tax payment deadlines?"
"How do I contest a parking ticket?"
"What permits do I need to renovate my bathroom?"

# Parks & Recreation
"Which city pools offer swimming lessons?"
"How can I reserve a park pavilion?"
"What are the winter skating rink locations?"

# Public Health & Safety
"Where are the nearest COVID-19 testing centers?"
"How do I report a bylaw violation?"
"What are the current snow removal standards?"

# Bilingual Services
"Quels services sont disponibles en français?"
"Where can I get city services in French?"
"Comment puis-je obtenir des documents municipaux en français?"
```

Each query is designed to return specific, actionable information with relevant URLs and contact details when available.

## Output Files

The scraper generates several files in the `ottawa_data` directory:

- `content.json`: Raw extracted content from the website
- `metadata.csv`: Site structure and metadata information
- `training_data.json`: Processed training examples for the model
- `Modelfile`: Configuration file for creating the Mistral model

## Model Features

The AI assistant:
- Provides accurate information from the City of Ottawa website
- Includes source URLs for verification
- Maintains a professional yet approachable tone
- Handles both English and French service information
- Suggests relevant contact information when appropriate

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```
3. Make your changes and commit:
```bash
git commit -am 'Add some feature'
```
4. Push to the branch:
```bash
git push origin feature/your-feature-name
```
5. Submit a pull request

## Customization

You can modify the script behavior by adjusting these parameters:

- `max_pages`: Number of pages to crawl (default: 100)
- `num_ctx`: Context window size for Mistral (default: 4096)
- `temperature`: Model creativity (default: 0.7)
- Other parameters in the Modelfile

## Dependencies

Main dependencies include:
- beautifulsoup4: Web scraping
- requests: HTTP requests
- pandas: Data processing
- tiktoken: Token-aware content chunking
- tqdm: Progress bars

## Notes

- The scraper respects website structure and avoids overloading the server
- Content is cleaned and normalized for better training
- The model is optimized for municipal service queries
- All source URLs are preserved for verification

## Data Synchronization

The project includes an automated synchronization system to keep the AI assistant's knowledge up-to-date with the website content.

### Sync Features

1. **Efficient Change Detection**
- Uses HTTP headers (Last-Modified, ETag) to check for changes
- Content hashing to detect actual content modifications
- Only processes pages that have been modified
- Minimizes unnecessary processing and server load

2. **Automated Synchronization Process**
```bash
# Run manual sync
python src/ottawa_sync.py

# Set up automatic daily sync (cron job)
0 2 * * * cd /path/to/ottawa-ai-assistant && ./venv/bin/python src/ottawa_sync.py
```

3. **Smart Resource Usage**
- Implements incremental updates instead of full reprocessing
- Respects server resources with controlled request rates
- Maintains a cache of previous states to optimize performance

4. **Monitoring and Reporting**
- Detailed logging of all changes
- Tracks new, updated, and removed pages
- Email notifications for significant changes (configurable)

5. **Error Handling and Reliability**
- Robust error handling for network issues
- Automatic retries for failed requests
- Preservation of existing data if updates fail

### Sync Process

The synchronization process:
1. Checks for modified pages using HTTP headers
2. Downloads and processes only changed content
3. Updates the model only when changes are detected
4. Maintains detailed logs of all updates

## Limitations

- Only crawls public-facing content
- Requires Ollama to be installed separately
- Limited to content available on ottawa.ca
- Model updates require re-running the scraping process

## Contributing

Contributions are welcome! Please read the contributing guidelines before making any changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://github.com/jmorganca/ollama) for the model serving infrastructure
- [Ollama Chat](https://github.com/schmitech/ollama-chat) for the chat interface
- Mistral AI for the base model

## Support

For support, please open an issue in the GitHub repository.