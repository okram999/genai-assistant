# GenAI-Assistant-demo


## What is this?
- A chatbot thats interacting with a Generative AI backend. In this project, contents from a text document (blog1.txt) is embedded in a vector store (Pinecone db). These vectors are fed as context to the LLM (`GPT-3`) by LangChain to respond to the user queries.


## To run this project in your local workspace:
 1. Populate the API Keys in .env file at the projects root
    ```
    OPENAI_API_KEY='sk-lxxccecdsvsfvsfv'
    PINECONE_API_KEY=wverveve-evevfev-erergerge
    PINECONE_ENVIRONMENT_REGION=us-west4-gcp-xxx
    ```
 3. Start the virtual env `pipenv shell`
 4. Install the required project packages, `pipenv install`
 5. Run the app, e.g. `streamlit run main.py` or set up the `.vscode/launch.json`. e.g.
    ```
    {
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "streamlit",
            "type": "python",
            "request": "launch",
            "program": "<path_to_streamlit_binary>",
            "args": [
                "run",
                "main.py"
            ],
            "justMyCode": true
        }
     ]
    }
    ```

<img width="500" alt="chat-screen" src="https://github.com/okram999/genai-assistant/assets/10067711/21794718-17f8-47cf-a173-c032bca18d09">





### Demo



https://github.com/okram999/genai-assistant/assets/10067711/c7b44934-f097-414b-b31b-3b1f9e77f087

