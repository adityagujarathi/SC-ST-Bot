# SC/ST Atrocities Act Research Assistant

A powerful research assistant for frontline justice workers dealing with cases under the Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989.

## Features

- Interactive query interface for legal assistance
- Integration with Gemini 2.0 Flash Lite AI for intelligent responses
- Automatic processing of PDF, Word, and PowerPoint documents
- Vector search for relevant information in your document collection
- Real-time internet search capabilities when local data is insufficient
- User-friendly interface built with Streamlit

## Setup Instructions

1. Clone this repository
2. Run the setup script to install dependencies:
   ```bash
   setup.bat
   ```
   
   Or manually install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and add your Gemini API key:
   ```bash
   cp .env.template .env
   ```
4. Edit `.env` and add your Gemini API key

## Deploying to Streamlit Cloud (Easiest Method)

Follow these simple steps to deploy the application to Streamlit Cloud:

1. **Push your code to GitHub**
   - Create a GitHub account if you don't have one
   - Create a new repository named "SC_ST_Bot"
   - Run the deployment script:
     ```
     deploy_to_streamlit_cloud.bat
     ```

2. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Select your "SC_ST_Bot" repository
   - Set the main file path to: `app.py`

3. **Add your API key**
   - In Streamlit Cloud, go to "Advanced settings" > "Secrets"
   - Add the following:
     ```
     GEMINI_API_KEY = "your-actual-api-key"
     ```

4. **Deploy!**
   - Click "Deploy"
   - Wait for the deployment to complete
   - Share the provided URL with anyone who needs access

## Running the Application

```bash
streamlit run app.py
```

## Dataset Integration

The application automatically processes all PDF, DOCX, and PPTX files in the root directory. The first time you run the application, it will:

1. Extract text from all documents
2. Split the text into manageable chunks
3. Create embeddings for semantic search
4. Store everything in a local ChromaDB database

This process may take some time depending on the number and size of your documents.

## How It Works

1. When you enter a query, the system searches for the most relevant document chunks in the database
2. If insufficient information is found locally, it performs a web search
3. The Gemini 2.0 Flash Lite model analyzes the context and generates a comprehensive response
4. The response includes timing information for transparency

## Security Note

Never commit your `.env` file containing API keys to version control.
