# SC/ST Atrocities Act Research Assistant

A multi-language AI-powered research assistant for frontline justice workers dealing with cases under the Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989.

## Features

- **AI-Powered Responses**: Uses Google's Gemini 2.0 Flash Lite model
- **Multi-language Support**: Available in 11 Indian languages
- **Document Processing**: Automatically processes PDF, DOCX, and PPTX files
- **Vector Search**: Uses FAISS for efficient document retrieval
- **Web Search**: Fallback to web search when local documents don't have answers

## Supported Languages

- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Marathi (mr)
- Gujarati (gu)
- Punjabi (pa)
- Bengali (bn)
- Odia (or)

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
4. Create a `.streamlit/secrets.toml` file with your Gemini API key:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
5. Run the app: `streamlit run app.py`

## Deployment to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Set the main file path to `app.py`
5. Add your Gemini API key to the Streamlit Cloud secrets:
   - Go to your app settings
   - Click on "Secrets"
   - Add the following:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```
6. Deploy the app

## Note on Dependencies

- The app requires Python 3.9 or later
- Table extraction from PDFs using `tabula-py` is optional and requires Java to be installed
- For the best experience, ensure all dependencies in `requirements.txt` are installed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
