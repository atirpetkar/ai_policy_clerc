# Sue-per GPT: Legal RAG Assistant 🤖⚖️

A powerful Legal RAG (Retrieval-Augmented Generation) application using Streamlit that provides data upload, analysis, and chatbot capabilities for legal documents.

## Features 🌟

- **Document Processing**
  - CSV upload with flexible column mapping
  - Support for large documents with automatic chunking
  - Metadata extraction and tracking

- **Advanced Analysis**
  - Basic statistics and visualizations
  - Word clouds and key phrase extraction
  - Topic modeling
  - Document similarity analysis
  - Text length analysis

- **AI-Powered Chat**
  - OpenAI GPT-4 integration for intelligent responses
  - Context-aware responses using RAG
  - Advanced filtering options
  - Export chat history

- **Trial Simulator**
  - Speech-to-text and text-to-speech capabilities using:
      - SpeechRecognition for converting speech to text
      - gTTS (Google Text-to-Speech) for converting text to speech
  - Interactive trial simulation with:
      - Case type and topic selection
      - Choice between text or speech input
      - Generation of opposition arguments based on your legal document corpus
      - Citation of relevant cases
      - Potential counter-arguments
      - Trial history tracking
  - Integration with your existing vector store to:
      - Pull relevant legal precedents
      -  Generate contextually appropriate responses
      -  Maintain consistency with your document corpus
  - User-friendly interface features:
      - Audio controls for speech playback
      - Expandable sections for trial history
      - Option to toggle between text and speech output
      - Easy reset functionality

## Installation 🚀



### Option: Local Installation

1. Clone the repository
2. Ensure you have Python 3.9 or later installed (Python 3.9 - 3.11 supported)
3. Create and activate a Python virtual environment:
   ```bash
   # For Python 3.9
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Set up environment variables for API key:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   On Windows:
   ```bash
   set OPENAI_API_KEY=your-openai-api-key
   ```
6. Run the application:
   ```bash
   streamlit run main.py
   ```

### Required Dependencies
The application requires the following main packages (compatible with Python 3.9+):
- streamlit (1.21.0 or later)
- openai (1.0.0 or later)
- pandas (1.3.0 or later)
- numpy (1.21.0 or later)
- faiss-cpu (1.7.0 or later)
- nltk (3.6.0 or later)
- plotly (5.8.0 or later)
- wordcloud (1.8.0 or later)
- scikit-learn (1.0.0 or later)

## Usage Guide 📚

1. **Data Upload**
   - Prepare your CSV file with legal documents
   - Required columns: docid, previous_text, gold_text, short_citations
   - Upload through the interface
   - Select content and metadata columns

2. **Data Analysis**
   - Basic Analysis tab: View statistics and distributions
   - Text Analysis tab: Access advanced text analytics

3. **Chat Interface**
   - Configure OpenAI API key in sidebar
   - Use filters for targeted searches
   - Export chat history

## File Structure 📁

```
├── .streamlit/
│   └── config.toml
├── assets/
│   └── legal_icon.svg
├── components/
│   ├── chatbot.py
│   ├── data_analysis.py
│   └── data_upload.py
├── utils/
│   ├── data_processor.py
│   ├── export_utils.py
│   ├── openai_helper.py
│   ├── text_analysis.py
│   └── vector_store.py
└── main.py
```

## Technology Stack 💻

- **Frontend**: Streamlit
- **AI Model**: OpenAI GPT-4
- **Data Processing**: 
  - Pandas
  - NLTK
  - scikit-learn
- **Vector Search**: FAISS
- **Visualization**: Plotly

## Configuration ⚙️

### API Key

1. **OpenAI Configuration**
   - Get your API key from [OpenAI Platform](https://platform.openai.com)
   - Enter in the sidebar of the application

### Environment Variables

The application uses the following environment variable:
- `OPENAI_API_KEY`: Your OpenAI API key

## Features in Detail 🔍

### Document Processing
- Automatic text chunking for large documents
- Flexible column mapping
- Metadata preservation
- Token limit handling

### Analysis Capabilities
- Word clouds and key phrase extraction
- Topic modeling with adjustable topics
- Document similarity matrix
- Text length distribution
- Basic statistics and visualizations

### Chat Features
- OpenAI GPT-4 powered responses
- Context-aware responses
- Advanced filtering options:
  - Text length filtering
  - Metadata-based filtering
- Chat history export (JSON/CSV)

## License 📄

MIT License

## Contributing 🤝

Contributions are welcome! Feel free to submit issues and enhancement requests.

## Support 💬

For support or questions, please open an issue in the repository.
