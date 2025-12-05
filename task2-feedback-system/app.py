#!/usr/bin/env python3
"""
TASK 2 - Two-Dashboard AI Feedback System
Fynd AI Intern Assessment

Flask web application with User Dashboard (public) and Admin Dashboard (internal)
with real-time OpenAI API integration for feedback analysis.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import openai
import json
import csv
import os
from datetime import datetime
import uuid
import sqlite3
from typing import Dict, List
import threading
import time

app = Flask(__name__)
app.secret_key = 'fynd-ai-assessment-secret-key-2024'

# OpenAI Configuration
openai.api_key = "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"

class FeedbackSystem:
    def __init__(self):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.db_path = 'feedback_system.db'
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_rating INTEGER NOT NULL,
                user_review TEXT NOT NULL,
                ai_response TEXT,
                ai_summary TEXT,
                recommended_actions TEXT,
                sentiment_score REAL,
                category TEXT,
                priority_level TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_ai_response(self, rating: int, review: str) -> str:
        """Generate AI response for user feedback"""
        try:
            prompt = f"""
            You are a helpful customer service AI. A customer has left the following feedback:
            
            Rating: {rating}/5 stars
            Review: "{review}"
            
            Generate a professional, empathetic response that:
            1. Acknowledges their feedback
            2. Thanks them for their time
            3. Addresses their concerns (if any)
            4. Offers next steps if appropriate
            
            Keep the response concise, friendly, and professional (max 100 words).
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional customer service representative."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Thank you for your {rating}-star feedback! We appreciate you taking the time to share your experience with us."
    
    def generate_ai_summary(self, rating: int, review: str) -> Dict:
        """Generate AI summary and analysis for admin dashboard"""
        try:
            prompt = f"""
            Analyze this customer feedback and provide a structured analysis:
            
            Rating: {rating}/5 stars
            Review: "{review}"
            
            Provide your analysis in JSON format:
            {{
                "summary": "Brief 1-2 sentence summary of the feedback",
                "sentiment_score": <number between -1 and 1, where -1 is very negative, 0 is neutral, 1 is very positive>,
                "category": "one of: product, service, pricing, delivery, support, other",
                "priority_level": "one of: low, medium, high, critical",
                "recommended_actions": "Specific actionable recommendations for the business"
            }}
            
            Base priority on rating and sentiment:
            - 1-2 stars with negative sentiment: high/critical
            - 3 stars: medium
            - 4-5 stars: low/medium
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst specializing in customer feedback analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback analysis
                return self._fallback_analysis(rating, review)
                
        except Exception as e:
            return self._fallback_analysis(rating, review)
    
    def _fallback_analysis(self, rating: int, review: str) -> Dict:
        """Fallback analysis when AI fails"""
        sentiment_score = (rating - 3) / 2  # Convert 1-5 to -1 to 1
        
        if rating <= 2:
            priority = "high"
        elif rating == 3:
            priority = "medium"
        else:
            priority = "low"
        
        return {
            "summary": f"Customer left a {rating}-star review with feedback about their experience.",
            "sentiment_score": sentiment_score,
            "category": "general",
            "priority_level": priority,
            "recommended_actions": f"Follow up on {rating}-star feedback to understand concerns and improve service."
        }
    
    def save_feedback(self, rating: int, review: str) -> str:
        """Save feedback to database with AI analysis"""
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Generate AI responses
        ai_response = self.generate_ai_response(rating, review)
        ai_analysis = self.generate_ai_summary(rating, review)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (
                id, timestamp, user_rating, user_review, ai_response,
                ai_summary, recommended_actions, sentiment_score,
                category, priority_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id, timestamp, rating, review, ai_response,
            ai_analysis['summary'], ai_analysis['recommended_actions'],
            ai_analysis['sentiment_score'], ai_analysis['category'],
            ai_analysis['priority_level']
        ))
        
        conn.commit()
        conn.close()
        
        return feedback_id, ai_response
    
    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback for admin dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback ORDER BY timestamp DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        feedback_list = []
        for row in rows:
            feedback_list.append({
                'id': row[0],
                'timestamp': row[1],
                'user_rating': row[2],
                'user_review': row[3],
                'ai_response': row[4],
                'ai_summary': row[5],
                'recommended_actions': row[6],
                'sentiment_score': row[7],
                'category': row[8],
                'priority_level': row[9]
            })
        
        return feedback_list
    
    def get_analytics(self) -> Dict:
        """Get analytics data for admin dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total feedback count
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]
        
        # Average rating
        cursor.execute('SELECT AVG(user_rating) FROM feedback')
        avg_rating = cursor.fetchone()[0] or 0
        
        # Rating distribution
        cursor.execute('SELECT user_rating, COUNT(*) FROM feedback GROUP BY user_rating')
        rating_dist = dict(cursor.fetchall())
        
        # Category distribution
        cursor.execute('SELECT category, COUNT(*) FROM feedback GROUP BY category')
        category_dist = dict(cursor.fetchall())
        
        # Priority distribution
        cursor.execute('SELECT priority_level, COUNT(*) FROM feedback GROUP BY priority_level')
        priority_dist = dict(cursor.fetchall())
        
        # Recent feedback (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM feedback 
            WHERE datetime(timestamp) > datetime('now', '-1 day')
        ''')
        recent_feedback = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'average_rating': round(avg_rating, 2),
            'rating_distribution': rating_dist,
            'category_distribution': category_dist,
            'priority_distribution': priority_dist,
            'recent_feedback_24h': recent_feedback
        }

# Initialize feedback system
feedback_system = FeedbackSystem()

# Routes
@app.route('/')
def landing_page():
    """Landing page with navigation to both dashboards"""
    return render_template('landing.html')

@app.route('/user')
def user_dashboard():
    """Public user dashboard for submitting feedback"""
    return render_template('user_dashboard.html')

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard for viewing all feedback and analytics"""
    # Simple authentication check
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin_login'))
    
    feedback_list = feedback_system.get_all_feedback()
    analytics = feedback_system.get_analytics()
    
    return render_template('admin_dashboard.html', 
                         feedback_list=feedback_list, 
                         analytics=analytics)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Simple admin login"""
    if request.method == 'POST':
        password = request.form.get('password')
        if password == 'fynd2024':  # Simple password for demo
            session['admin_authenticated'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='Invalid password')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_authenticated', None)
    return redirect(url_for('landing_page'))

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    """API endpoint to submit user feedback"""
    try:
        data = request.get_json()
        rating = int(data.get('rating'))
        review = data.get('review', '').strip()
        
        if not (1 <= rating <= 5):
            return jsonify({'success': False, 'error': 'Rating must be between 1 and 5'})
        
        if not review:
            return jsonify({'success': False, 'error': 'Review text is required'})
        
        # Save feedback and get AI response
        feedback_id, ai_response = feedback_system.save_feedback(rating, review)
        
        return jsonify({
            'success': True,
            'feedback_id': feedback_id,
            'ai_response': ai_response
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/admin/feedback')
def get_admin_feedback():
    """API endpoint to get all feedback for admin dashboard"""
    if not session.get('admin_authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    feedback_list = feedback_system.get_all_feedback()
    return jsonify(feedback_list)

@app.route('/api/admin/analytics')
def get_admin_analytics():
    """API endpoint to get analytics data"""
    if not session.get('admin_authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    analytics = feedback_system.get_analytics()
    return jsonify(analytics)

if __name__ == '__main__':
    print("Starting Fynd AI Feedback System...")
    print("User Dashboard: http://localhost:5000/user")
    print("Admin Dashboard: http://localhost:5000/admin (password: fynd2024)")
    app.run(debug=True, host='0.0.0.0', port=5000)