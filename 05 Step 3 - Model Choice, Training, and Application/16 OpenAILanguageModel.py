import openai
import json
    
# Set up your OpenAI API key
openai.api_key = 'YOUR_API_KEY_HERE'
    
def analyze_sentiment(text):
    response = openai.chat.completions.create(
        model="gpt–4",
        messages=[
            {"role": "user", "content": f "Analyze the sentiment of the following text and provide a sentiment score from 0 to 10 in JSON in the property called sentiment_score:\n\n{text}\n\""}
        ]
    )
    return json.loads(response.choices[0].message.content)['sentiment_score']
    
def test_analyze_sentiment():
    test_sentences = [
       ""I absolutely love this new phone!It's fantastic and exceeds all my expectations"",
       ""This is the worst experience I have ever had with a company. Totally unacceptable"",
       ""Meh, the movie was just okay. Not too good, not too bad"",
       ""I am thrilled with my new job. The team is great and the work is fulfilling"",
       ""The food was terrible, and the service was even worse. I am never coming back here"",
       ""Wow, what a beautiful day! The sun is shining and everything feels perfect"",
       ""This product is a complete waste of money. It broke after one use"",
       ""I am feeling pretty neutral about this situation, neither happy nor sad"",
       ""The concert last night was amazing! Best performance I've ever seen"",
       ""Ugh, what a horrible traffic jam. Made me late for work and ruined my day""
    ]
    
    for i, sentence in enumerate(test_sentences):
        try:
            sentiment_score = analyze_sentiment(sentence)
            print(f"Test {i+1}: Sentence: \"{sentence}\"")
            print(f"Sentiment Score: {sentiment_score}\n")
        except Exception as e:
            print(f"Test {i+1}: Sentence: \"{sentence}\"")
            print(f"Error: {e}\n")
    
# Run the test method
test_analyze_sentiment()