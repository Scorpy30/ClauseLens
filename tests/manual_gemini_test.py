from backend.services.ai.gemini_client import GeminiClient


client = GeminiClient()

result = client.generate_legal_answer(
    situation="I am considering resigning from my job. What notice period do I need to give?",
    evidence=[
        {
            "chunk_id": "chunk-1",
            "page_number": 2,
            "section": "4",
            "text": "Either party may terminate the employment relationship by providing 60 days' written notice to the other party.",
        }
    ],
)

print(result)
