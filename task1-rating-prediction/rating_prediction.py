#!/usr/bin/env python3
"""
TASK 1 - Rating Prediction via Prompting
Fynd AI Intern Assessment

This notebook implements 3 different prompting approaches to classify Yelp reviews into 1-5 stars
using OpenAI's GPT model with real-time API integration.
"""

import pandas as pd
import numpy as np
import json
import openai
import time
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# OpenAI API Configuration
api_key = "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"

class YelpRatingPredictor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=api_key)
        self.results = {}
        
    def load_data(self, file_path: str, sample_size: int = 200) -> pd.DataFrame:
        """Load and sample Yelp reviews dataset"""
        try:
            # Try different file formats
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, lines=True)
            else:
                # Create sample data if file not found
                print("Creating sample Yelp reviews data...")
                df = self.create_sample_data()
            
            # Sample the data
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Ensure required columns exist
            if 'text' not in df.columns:
                df['text'] = df.get('review_text', df.get('review', ''))
            if 'stars' not in df.columns:
                df['stars'] = df.get('rating', df.get('star_rating', np.random.randint(1, 6, len(df))))
            
            print(f"Loaded {len(df)} reviews")
            print(f"Rating distribution:\\n{df['stars'].value_counts().sort_index()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample Yelp reviews for testing"""
        sample_reviews = [
            {"text": "Amazing food and excellent service! The staff was incredibly friendly and the atmosphere was perfect. Will definitely come back!", "stars": 5},
            {"text": "Good food but the service was a bit slow. The restaurant was clean and the prices were reasonable.", "stars": 4},
            {"text": "Average experience. Food was okay, nothing special. Service was decent but could be better.", "stars": 3},
            {"text": "Disappointing meal. The food was cold when it arrived and the server seemed uninterested. Overpriced for what we got.", "stars": 2},
            {"text": "Terrible experience! Rude staff, awful food, and dirty restaurant. Would never recommend this place to anyone.", "stars": 1},
            {"text": "Outstanding restaurant! Every dish was perfectly prepared and the service was impeccable. Best dining experience I've had in years!", "stars": 5},
            {"text": "Pretty good overall. The pasta was delicious and the wine selection was impressive. Slightly expensive but worth it.", "stars": 4},
            {"text": "It's an okay place. Food is decent, service is average. Nothing to complain about but nothing extraordinary either.", "stars": 3},
            {"text": "Not impressed. The food took forever to arrive and when it did, it was lukewarm. The staff seemed overwhelmed.", "stars": 2},
            {"text": "Worst restaurant ever! Food was inedible, service was horrible, and the place was filthy. Complete waste of money!", "stars": 1},
        ] * 20  # Repeat to get 200 samples
        
        return pd.DataFrame(sample_reviews)
    
    def approach_1_direct_classification(self, review_text: str) -> Dict:
        """
        Approach 1: Direct Classification
        Simple, straightforward prompt asking for star rating classification
        """
        prompt = f'''
        You are a review rating classifier. Analyze the following restaurant review and predict the star rating from 1 to 5 stars.
        
        Review: "{review_text}"
        
        Respond with a JSON object in this exact format:
        {{
            "predicted_stars": <number from 1-5>,
            "explanation": "<brief reasoning for the assigned rating>"
        }}
        
        Consider:
        - 5 stars: Excellent, outstanding experience
        - 4 stars: Good, above average with minor issues
        - 3 stars: Average, okay experience
        - 2 stars: Below average, several issues
        - 1 star: Poor, terrible experience
        '''
        
        return self._call_openai_api(prompt, "approach_1")
    
    def approach_2_sentiment_analysis(self, review_text: str) -> Dict:
        """
        Approach 2: Sentiment-Based Analysis
        Focus on sentiment analysis with detailed reasoning
        """
        prompt = f'''
        As an expert sentiment analyst, evaluate this restaurant review by analyzing:
        1. Overall sentiment (positive/negative/neutral)
        2. Specific aspects mentioned (food, service, atmosphere, value)
        3. Intensity of emotions expressed
        4. Language tone and word choice
        
        Review: "{review_text}"
        
        Based on your analysis, assign a star rating (1-5) where:
        - Very positive sentiment with praise = 4-5 stars
        - Mostly positive with some concerns = 3-4 stars  
        - Neutral or mixed sentiment = 2-3 stars
        - Mostly negative sentiment = 1-2 stars
        - Very negative with strong criticism = 1 star
        
        Return your response as JSON:
        {{
            "predicted_stars": <1-5>,
            "explanation": "<detailed reasoning based on sentiment analysis>"
        }}
        '''
        
        return self._call_openai_api(prompt, "approach_2")
    
    def approach_3_comparative_analysis(self, review_text: str) -> Dict:
        """
        Approach 3: Comparative Analysis with Examples
        Use few-shot learning with example reviews
        """
        prompt = f'''
        You are an experienced restaurant reviewer. Rate this review by comparing it to these examples:
        
        EXAMPLES:
        5 Stars: "Absolutely phenomenal! Best meal of my life. Perfect service, amazing atmosphere."
        4 Stars: "Really good food and service. Had a great time, just minor wait for table."
        3 Stars: "Decent place. Food was okay, service was fine. Nothing special but acceptable."
        2 Stars: "Food was cold, service was slow. Disappointed but not the worst experience."
        1 Star: "Terrible! Rude staff, awful food, dirty restaurant. Complete disaster."
        
        Now rate this review: "{review_text}"
        
        Compare the language, sentiment, and specific complaints/praise to the examples above.
        
        Provide your rating as JSON:
        {{
            "predicted_stars": <1-5>,
            "explanation": "<comparison-based reasoning>"
        }}
        '''
        
        return self._call_openai_api(prompt, "approach_3")
    
    def _call_openai_api(self, prompt: str, approach: str) -> Dict:
        """Make API call to OpenAI with error handling and retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes restaurant reviews and returns valid JSON responses."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    
                    # Validate required fields
                    if 'predicted_stars' in result and 'explanation' in result:
                        # Ensure stars is in valid range
                        stars = int(result['predicted_stars'])
                        if 1 <= stars <= 5:
                            return {
                                'predicted_stars': stars,
                                'explanation': result['explanation'],
                                'valid_json': True,
                                'approach': approach
                            }
                
                # If we get here, JSON was invalid
                return {
                    'predicted_stars': 3,  # Default fallback
                    'explanation': 'Invalid JSON response from API',
                    'valid_json': False,
                    'approach': approach,
                    'raw_response': content
                }
                
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {
                        'predicted_stars': 3,
                        'explanation': f'API call failed: {str(e)}',
                        'valid_json': False,
                        'approach': approach
                    }
    
    def evaluate_approach(self, df: pd.DataFrame, approach_func, approach_name: str) -> Dict:
        """Evaluate a single approach on the dataset"""
        print(f"\\nEvaluating {approach_name}...")
        
        predictions = []
        actual_ratings = []
        valid_json_count = 0
        explanations = []
        
        for idx, row in df.iterrows():
            print(f"Processing review {idx + 1}/{len(df)}", end='\\r')
            
            result = approach_func(row['text'])
            predictions.append(result['predicted_stars'])
            actual_ratings.append(row['stars'])
            explanations.append(result['explanation'])
            
            if result['valid_json']:
                valid_json_count += 1
            
            # Add small delay to respect API rate limits
            time.sleep(0.1)
        
        # Calculate metrics
        accuracy = accuracy_score(actual_ratings, predictions)
        json_validity_rate = valid_json_count / len(df)
        
        # Calculate per-class accuracy
        class_report = classification_report(actual_ratings, predictions, output_dict=True)
        
        results = {
            'approach_name': approach_name,
            'accuracy': accuracy,
            'json_validity_rate': json_validity_rate,
            'predictions': predictions,
            'actual_ratings': actual_ratings,
            'explanations': explanations,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(actual_ratings, predictions)
        }
        
        print(f"\\n{approach_name} Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"JSON Validity Rate: {json_validity_rate:.3f}")
        
        return results
    
    def run_all_evaluations(self, df: pd.DataFrame) -> Dict:
        """Run all three approaches and compare results"""
        approaches = [
            (self.approach_1_direct_classification, "Direct Classification"),
            (self.approach_2_sentiment_analysis, "Sentiment Analysis"),
            (self.approach_3_comparative_analysis, "Comparative Analysis")
        ]
        
        all_results = {}
        
        for approach_func, approach_name in approaches:
            results = self.evaluate_approach(df, approach_func, approach_name)
            all_results[approach_name] = results
        
        return all_results
    
    def create_comparison_table(self, results: Dict) -> pd.DataFrame:
        """Create comparison table of all approaches"""
        comparison_data = []
        
        for approach_name, result in results.items():
            comparison_data.append({
                'Approach': approach_name,
                'Accuracy': f"{result['accuracy']:.3f}",
                'JSON Validity Rate': f"{result['json_validity_rate']:.3f}",
                'Precision (Macro Avg)': f"{result['classification_report']['macro avg']['precision']:.3f}",
                'Recall (Macro Avg)': f"{result['classification_report']['macro avg']['recall']:.3f}",
                'F1-Score (Macro Avg)': f"{result['classification_report']['macro avg']['f1-score']:.3f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_results(self, results: Dict):
        """Create visualizations of the results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy Comparison
        approaches = list(results.keys())
        accuracies = [results[app]['accuracy'] for app in approaches]
        
        axes[0, 0].bar(approaches, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. JSON Validity Rate
        json_rates = [results[app]['json_validity_rate'] for app in approaches]
        axes[0, 1].bar(approaches, json_rates, color=['#d62728', '#9467bd', '#8c564b'])
        axes[0, 1].set_title('JSON Validity Rate')
        axes[0, 1].set_ylabel('Validity Rate')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(json_rates):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 3. Confusion Matrix for Best Approach
        best_approach = max(results.keys(), key=lambda x: results[x]['accuracy'])
        cm = results[best_approach]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_approach}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Rating Distribution Comparison
        for i, (approach, result) in enumerate(results.items()):
            pred_dist = pd.Series(result['predictions']).value_counts().sort_index()
            axes[1, 1].plot(pred_dist.index, pred_dist.values, marker='o', label=f'{approach} (Predicted)')
        
        actual_dist = pd.Series(results[approaches[0]]['actual_ratings']).value_counts().sort_index()
        axes[1, 1].plot(actual_dist.index, actual_dist.values, marker='s', label='Actual', linewidth=3)
        axes[1, 1].set_title('Rating Distribution Comparison')
        axes[1, 1].set_xlabel('Star Rating')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('rating_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results: Dict, df: pd.DataFrame):
        """Generate comprehensive evaluation report"""
        print("\\n" + "="*80)
        print("YELP RATING PREDICTION - EVALUATION REPORT")
        print("="*80)
        
        print(f"\\nDataset Summary:")
        print(f"- Total Reviews Analyzed: {len(df)}")
        print(f"- Rating Distribution: {dict(df['stars'].value_counts().sort_index())}")
        
        print(f"\\nApproach Comparison:")
        comparison_df = self.create_comparison_table(results)
        print(comparison_df.to_string(index=False))
        
        # Find best approach
        best_approach = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\\nBest Performing Approach: {best_approach}")
        print(f"Best Accuracy: {results[best_approach]['accuracy']:.3f}")
        
        print(f"\\nDetailed Analysis:")
        for approach_name, result in results.items():
            print(f"\\n{approach_name}:")
            print(f"  - Accuracy: {result['accuracy']:.3f}")
            print(f"  - JSON Validity: {result['json_validity_rate']:.3f}")
            print(f"  - Precision: {result['classification_report']['macro avg']['precision']:.3f}")
            print(f"  - Recall: {result['classification_report']['macro avg']['recall']:.3f}")
            print(f"  - F1-Score: {result['classification_report']['macro avg']['f1-score']:.3f}")
        
        # Save results to JSON
        results_summary = {
            'dataset_size': len(df),
            'approaches': {
                name: {
                    'accuracy': float(result['accuracy']),
                    'json_validity_rate': float(result['json_validity_rate']),
                    'precision': float(result['classification_report']['macro avg']['precision']),
                    'recall': float(result['classification_report']['macro avg']['recall']),
                    'f1_score': float(result['classification_report']['macro avg']['f1-score'])
                }
                for name, result in results.items()
            },
            'best_approach': best_approach
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\\nResults saved to 'evaluation_results.json'")
        print("Visualization saved to 'rating_prediction_results.png'")

def main():
    """Main execution function"""
    print("FYND AI INTERN ASSESSMENT - TASK 1")
    print("Rating Prediction via Prompting")
    print("="*50)
    
    # Initialize predictor
    predictor = YelpRatingPredictor()
    
    # Load data (will create sample data if file not found)
    df = predictor.load_data('yelp_reviews.csv', sample_size=15)  # Reduced for demo
    
    # Run all evaluations
    print("\\nStarting evaluation of all approaches...")
    results = predictor.run_all_evaluations(df)
    
    # Generate comprehensive report
    predictor.generate_report(results, df)
    
    # Create visualizations
    predictor.plot_results(results)
    
    print("\\nEvaluation complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main()