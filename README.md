# GenAI-Assistant-demo


## What is this?
- A chatbot thats interacting with a Generative AI backend. In this project, multiple data files of various types (.pdf/.csv/.txt) in `Amazon S3` is embedded in a vector store (Pinecone db). These vectors are fed as context to an LLM in Amazon Bedrock. LangChain is the framework used in this project.


## To run this project in your local workspace:
 1. Populate the API Keys in .env file at the projects root
    ```
    OPENAI_API_KEY='sk-lxxccecdsvsfvsfv'
    PINECONE_API_KEY=wverveve-evevfev-erergerge
    PINECONE_ENVIRONMENT_REGION=us-west4-gcp-xxx
    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=
    BWB_PROFILE_NAME=''
    BWB_REGION_NAME=''

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


![Screenshot 2023-08-07 at 12 58 44](https://github.com/okram999/genai-assistant/assets/10067711/387e8f47-781e-401a-8a87-95872fc6dfbd)





### Demo



https://github.com/okram999/genai-assistant/assets/10067711/728ac601-7035-43e9-b773-be7dbc511db5





