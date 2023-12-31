# Resume-Parser-LLM

## Introduction
This is a Resume Parser tool built using Streamlit and Langchain. The tool allows users to upload resumes in PDF format, and it extracts relevant information based on user queries. The underlying technology uses the Mistral AI model for natural language understanding and Pinecone for vector storage.

## Features
- **Resume Upload:** Users can upload one or multiple resumes in PDF format.
- **Query Processing:** The tool processes user queries and extracts relevant information from the resumes.
- **History Tracking:** The tool keeps track of user queries and the corresponding extracted information.

## Getting Started

### Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add the following variables with your API keys:

     ```env
     HUGGINGFACE_API_TOKEN=your_huggingface_api_token
     OPENAI_API_KEY=your_openai_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     ```

### Usage
Run the following command to start the Streamlit app:

```bash
streamlit run resume_parser.py
```

Visit the provided URL in your web browser to access the Resume Parser.

## Customization
- **Model Configuration:** You can customize the Mistral AI model's parameters by modifying the `initialize_session_state` function in `resume_parser.py`.
- **Prompt Template:** The tool uses a prompt template for querying the Mistral AI model. You can customize the template in the `create_retrieval_chain` function in `resume_parser.py`.

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [Langchain](https://github.com/Mistral-7B/langchain)
- [Pinecone](https://www.pinecone.io/)
- [Hugging Face](https://huggingface.co/)
