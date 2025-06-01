# Multilingual RAG System for Food and Nutrition

A sophisticated Retrieval-Augmented Generation (RAG) system that supports both English and Urdu languages, specifically focused on food, nutrition, and traditional cuisine information.

## 🌟 Features

- **Multilingual Support**: Handles both English and Urdu content seamlessly
- **Advanced Chunking Strategies**:
  - Fixed-length chunking
  - Sentence-based chunking
  - Semantic chunking
- **Enhanced Context Processing**:
  - Context summarization
  - Redundancy removal
  - Relevance ranking
- **Quality Evaluation**: Built-in response quality assessment
- **Pinecone Integration**: Efficient vector storage and retrieval
- **Groq LLM Integration**: High-performance language model for response generation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Pinecone account and API key
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multilingual-rag.git
cd multilingual-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
```

### Usage

1. Run the main script:
```bash
python nlp_rag.py
```

2. The system will:
   - Collect data from Wikipedia and custom sources
   - Process and chunk the content
   - Generate embeddings
   - Store vectors in Pinecone
   - Enable querying in both English and Urdu

## 📚 Project Structure

```
multilingual-rag/
├── nlp_rag.py          # Main implementation file
├── requirements.txt    # Project dependencies
├── README.md          # This file
└── embeddings/        # Generated embeddings (if saved locally)
```

## 🔧 Configuration

The system uses several key components:

- **Pinecone**: Vector database for storing embeddings
  - Index name: "urdu-eng-rag"
  - Dimension: 768
  - Region: AWS us-east-1

- **Groq**: Language model for response generation
  - Model: llama-3.3-70b-versatile
  - Temperature: 0.3
  - Max tokens: 512

## 💡 Example Queries

The system can handle queries in both English and Urdu:

```python
# English queries
"What are the health benefits of Pakistani cuisine?"
"How can traditional Pakistani food be part of a balanced diet?"

# Urdu queries
"پاکستانی کھانوں کے صحت کے فوائد کیا ہیں؟"
"روایتی پاکستانی کھانوں کو متوازن غذا کا حصہ کیسے بنایا جا سکتا ہے؟"
```

## 🛠️ Technical Details

### Chunking Strategies

1. **Fixed-length Chunking**:
   - Size: 500 characters
   - Overlap: 50 characters

2. **Sentence-based Chunking**:
   - Language-aware sentence splitting
   - Maximum chunk size: 500 characters

3. **Semantic Chunking**:
   - Paragraph-based splitting
   - Preserves semantic units

### Embedding Model

- Model: sentence-transformers/LaBSE
- Dimension: 768
- Language support: Multilingual (including English and Urdu)

## 📊 Performance

The system includes built-in evaluation metrics for:
- Response relevance
- Accuracy
- Completeness
- Clarity
- Context utilization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Wikipedia for content
- Pinecone for vector storage
- Groq for language model capabilities
- Sentence-Transformers for embedding generation 