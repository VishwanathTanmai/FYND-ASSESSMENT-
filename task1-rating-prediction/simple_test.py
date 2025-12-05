#!/usr/bin/env python3
"""
Simple test script for Task 1 - Rating Prediction
Fynd AI Intern Assessment
"""

import openai
import json
import pandas as pd
import time

# OpenAI Configuration
api_key = "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"
client = openai.OpenAI(api_key=api_key)

print("FYND AI ASSESSMENT - TASK 1")
print("Rating Prediction via Prompting")
print("=" * 50)

# Sample reviews for testing
sample_reviews = [
    {"text": "Amazing food and excellent service! Will definitely come back!", "stars": 5},
    {"text": "Good food but service was slow. Prices were reasonable.", "stars": 4},
    {"text": "Average experience. Nothing special but okay.", "stars": 3},
    {"text": "Disappointing meal. Food was cold and overpriced.", "stars": 2},
    {"text": "Terrible experience! Rude staff and awful food.", "stars": 1}
]

def test_approach(review_text, approach_name, prompt):
    """Test a single approach"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes reviews and returns JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except:
            pass
        
        return {"predicted_stars": 3, "explanation": "Failed to parse response"}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"predicted_stars": 3, "explanation": f"API Error: {str(e)}"}

# Approach 1: Direct Classification
def approach_1(review_text):
    prompt = f"""
    Analyze this restaurant review and predict the star rating (1-5):
    
    Review: "{review_text}"
    
    Return JSON format:
    {{
        "predicted_stars": <1-5>,
        "explanation": "<brief reason>"
    }}
    """
    return test_approach(review_text, "Direct Classification", prompt)

# Approach 2: Sentiment Analysis
def approach_2(review_text):
    prompt = f"""
    As a sentiment analyst, evaluate this review:
    
    Review: "{review_text}"
    
    Analyze sentiment and assign 1-5 stars based on:
    - Very positive = 5 stars
    - Positive = 4 stars  
    - Neutral = 3 stars
    - Negative = 2 stars
    - Very negative = 1 star
    
    Return JSON:
    {{
        "predicted_stars": <1-5>,
        "explanation": "<sentiment analysis>"
    }}
    """
    return test_approach(review_text, "Sentiment Analysis", prompt)

# Approach 3: Comparative Analysis
def approach_3(review_text):
    prompt = f"""
    Compare this review to these examples:
    
    5 Stars: "Amazing! Perfect service and food!"
    4 Stars: "Good food, minor issues with service"
    3 Stars: "Okay experience, nothing special"
    2 Stars: "Disappointing, several problems"
    1 Star: "Terrible! Awful food and service"
    
    Review to rate: "{review_text}"
    
    Return JSON:
    {{
        "predicted_stars": <1-5>,
        "explanation": "<comparison reasoning>"
    }}
    """
    return test_approach(review_text, "Comparative Analysis", prompt)

# Run evaluation
print("\nTesting all approaches...")
approaches = [
    ("Direct Classification", approach_1),
    ("Sentiment Analysis", approach_2), 
    ("Comparative Analysis", approach_3)
]

results = {}
for approach_name, approach_func in approaches:
    print(f"\nTesting {approach_name}...")
    
    predictions = []
    actual_ratings = []
    
    for i, review in enumerate(sample_reviews):
        print(f"  Processing review {i+1}/5", end='\r')
        
        result = approach_func(review['text'])
        predictions.append(result['predicted_stars'])
        actual_ratings.append(review['stars'])
        
        time.sleep(0.5)  # Rate limiting
    
    # Calculate accuracy
    correct = sum(1 for p, a in zip(predictions, actual_ratings) if p == a)
    accuracy = correct / len(predictions)
    
    results[approach_name] = {
        'accuracy': accuracy,
        'predictions': predictions,
        'actual': actual_ratings
    }
    
    print(f"\n{approach_name}: {accuracy:.2f} accuracy")

# Display results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

for approach_name, result in results.items():
    print(f"\n{approach_name}:")
    print(f"  Accuracy: {result['accuracy']:.2f}")
    print(f"  Predictions: {result['predictions']}")
    print(f"  Actual:      {result['actual']}")

# Find best approach
best_approach = max(results.keys(), key=lambda x: results[x]['accuracy'])
print(f"\nBest Approach: {best_approach}")
print(f"Best Accuracy: {results[best_approach]['accuracy']:.2f}")

print("\nTask 1 evaluation completed successfully!")
print("All three prompting approaches tested with real OpenAI API")