"""
Customer Feedback Analyzer - Main Application

This is the main entry point for the Customer Feedback Analyzer application.
It provides a command-line interface for users to:
- Add new customer feedback
- Analyze existing feedback
- Generate visualizations
- View statistics and reports

Author: Adyasha Khandai
Copyright (c) 2025 Adyasha Khandai
License: MIT License
Date: 2025
"""

import os
import sys
from typing import Optional, List, Dict
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from database.db_manager import DatabaseManager
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.text_preprocessor import TextPreprocessor
from visualization.charts import FeedbackVisualizer


class FeedbackAnalyzerApp:
    """Main application class for the Customer Feedback Analyzer."""
    
    def __init__(self, db_path: str = "feedback_analyzer.db"):
        """
        Initialize the application.
        
        Args:
            db_path: Path to the SQLite database file
        """
        print("🚀 Initializing Customer Feedback Analyzer...")
        
        try:
            # Initialize components
            self.db_manager = DatabaseManager(db_path)
            self.sentiment_analyzer = SentimentAnalyzer()
            self.text_preprocessor = TextPreprocessor()
            self.visualizer = FeedbackVisualizer()
            
            print("✅ Application initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing application: {e}")
            print("Please check your environment and try again.")
            sys.exit(1)
    
    def display_main_menu(self):
        """Display the main menu options."""
        print("\\n" + "="*60)
        print("📊 CUSTOMER FEEDBACK ANALYZER")
        print("="*60)
        print("1. 📝 Add New Feedback")
        print("2. 🔍 View All Feedback")
        print("3. 🧠 Analyze Sentiment (All Feedback)")
        print("4. 📈 Generate Visualizations")
        print("5. 📊 View Statistics & Reports")
        print("6. 🔍 Search & Filter Feedback")
        print("7. ⚙️  Manage Data")
        print("8. ❓ Help & Information")
        print("0. 🚪 Exit")
        print("="*60)
    
    def add_new_feedback(self):
        """Add new customer feedback to the database."""
        print("\\n📝 Adding New Customer Feedback")
        print("-" * 40)
        
        try:
            # Collect feedback information
            customer_name = input("Customer Name: ").strip()
            if not customer_name:
                print("❌ Customer name cannot be empty.")
                return
            
            email = input("Email (optional): ").strip()
            
            print("\\nProduct Categories:")
            categories = ["Electronics", "Clothing", "Food & Beverage", "Healthcare", 
                         "Software", "Books", "Home & Garden", "Sports", "Other"]
            for i, cat in enumerate(categories, 1):
                print(f"  {i}. {cat}")
            
            while True:
                try:
                    cat_choice = input("\\nSelect category (1-9): ").strip()
                    if cat_choice == "":
                        product_category = "Other"
                        break
                    cat_idx = int(cat_choice) - 1
                    if 0 <= cat_idx < len(categories):
                        product_category = categories[cat_idx]
                        break
                    else:
                        print("❌ Please enter a number between 1-9.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            
            while True:
                try:
                    rating = int(input("Rating (1-5 stars): "))
                    if 1 <= rating <= 5:
                        break
                    else:
                        print("❌ Rating must be between 1 and 5.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            
            feedback_text = input("\\nFeedback text: ").strip()
            if not feedback_text:
                print("❌ Feedback text cannot be empty.")
                return
            
            # Add to database
            feedback_id = self.db_manager.add_feedback(
                customer_name=customer_name,
                email=email,
                product_category=product_category,
                rating=rating,
                feedback_text=feedback_text
            )
            
            print(f"\\n✅ Feedback added successfully! (ID: {feedback_id})")
            
            # Ask if user wants to analyze sentiment immediately
            analyze = input("\\nAnalyze sentiment for this feedback now? (y/N): ").strip().lower()
            if analyze in ['y', 'yes']:
                self.analyze_single_feedback(feedback_id, feedback_text)
                
        except KeyboardInterrupt:
            print("\\n❌ Operation cancelled by user.")
        except Exception as e:
            print(f"❌ Error adding feedback: {e}")
    
    def view_all_feedback(self):
        """View all feedback in the database."""
        print("\\n🔍 All Customer Feedback")
        print("-" * 40)
        
        try:
            feedback_list = self.db_manager.get_all_feedback()
            
            if not feedback_list:
                print("📭 No feedback found in the database.")
                print("   Use option 1 to add some feedback first.")
                return
            
            print(f"Found {len(feedback_list)} feedback entries:\\n")
            
            for feedback in feedback_list:
                print(f"ID: {feedback['id']}")
                print(f"Customer: {feedback['customer_name']}")
                print(f"Category: {feedback['product_category']}")
                print(f"Rating: {'⭐' * feedback['rating']} ({feedback['rating']}/5)")
                print(f"Date: {feedback['created_date']}")
                print(f"Feedback: {feedback['feedback_text'][:100]}{'...' if len(feedback['feedback_text']) > 100 else ''}")
                print("-" * 40)
            
            # Pagination for large datasets
            if len(feedback_list) > 10:
                print(f"\\nShowing all {len(feedback_list)} entries.")
                print("💡 Tip: Use option 6 to search and filter feedback.")
        
        except Exception as e:
            print(f"❌ Error retrieving feedback: {e}")
    
    def analyze_single_feedback(self, feedback_id: int, feedback_text: str):
        """Analyze sentiment for a single feedback item."""
        try:
            # Perform sentiment analysis
            results = self.sentiment_analyzer.analyze_sentiment_comprehensive(feedback_text)
            overall = results['overall']
            
            # Extract keywords
            keywords = self.sentiment_analyzer.extract_keywords_with_sentiment(feedback_text, top_n=5)
            
            # Save results to database
            self.db_manager.save_sentiment_analysis(
                feedback_id=feedback_id,
                sentiment_score=overall['overall_score'],
                sentiment_label=overall['overall_label'],
                confidence=overall['confidence']
            )
            
            # Save keywords
            keyword_tuples = [(kw[0], 1, kw[1]) for kw in keywords]
            if keyword_tuples:
                self.db_manager.save_keywords(feedback_id, keyword_tuples)
            
            # Display results
            print(f"\\n🎯 Sentiment Analysis Results:")
            print(f"   Overall Sentiment: {overall['overall_label'].upper()}")
            print(f"   Confidence Score: {overall['confidence']:.2f}")
            print(f"   Sentiment Score: {overall['overall_score']:.2f}")
            
            if keywords:
                print(f"   Top Keywords: {', '.join([kw[0] for kw in keywords[:3]])}")
            
            print("✅ Analysis results saved to database.")
            
        except Exception as e:
            print(f"❌ Error analyzing feedback: {e}")
    
    def analyze_all_sentiment(self):
        """Analyze sentiment for all feedback in the database."""
        print("\\n🧠 Analyzing Sentiment for All Feedback")
        print("-" * 40)
        
        try:
            # Get all feedback
            all_feedback = self.db_manager.get_all_feedback()
            
            if not all_feedback:
                print("📭 No feedback found to analyze.")
                return
            
            print(f"Found {len(all_feedback)} feedback entries to analyze...")
            
            analyzed_count = 0
            skipped_count = 0
            
            for i, feedback in enumerate(all_feedback, 1):
                print(f"\\rProcessing {i}/{len(all_feedback)}... ", end="", flush=True)
                
                # Check if already analyzed
                existing_analysis = self.db_manager.get_feedback_with_sentiment()
                feedback_analyzed = any(
                    item['id'] == feedback['id'] and item['sentiment_label'] 
                    for item in existing_analysis
                )
                
                if feedback_analyzed:
                    skipped_count += 1
                    continue
                
                # Analyze sentiment
                try:
                    results = self.sentiment_analyzer.analyze_sentiment_comprehensive(
                        feedback['feedback_text']
                    )
                    overall = results['overall']
                    
                    # Extract keywords
                    keywords = self.sentiment_analyzer.extract_keywords_with_sentiment(
                        feedback['feedback_text'], top_n=5
                    )
                    
                    # Save results
                    self.db_manager.save_sentiment_analysis(
                        feedback_id=feedback['id'],
                        sentiment_score=overall['overall_score'],
                        sentiment_label=overall['overall_label'],
                        confidence=overall['confidence']
                    )
                    
                    # Save keywords
                    keyword_tuples = [(kw[0], 1, kw[1]) for kw in keywords]
                    if keyword_tuples:
                        self.db_manager.save_keywords(feedback['id'], keyword_tuples)
                    
                    analyzed_count += 1
                    
                except Exception as e:
                    print(f"\\n⚠️  Error analyzing feedback ID {feedback['id']}: {e}")
            
            print(f"\\n\\n✅ Analysis complete!")
            print(f"   📊 Analyzed: {analyzed_count} new entries")
            print(f"   ⏭️  Skipped: {skipped_count} already analyzed")
            print(f"   📈 Total in database: {len(all_feedback)} entries")
        
        except Exception as e:
            print(f"❌ Error during batch analysis: {e}")
    
    def generate_visualizations(self):
        """Generate and display visualizations."""
        print("\\n📈 Generate Visualizations")
        print("-" * 40)
        
        try:
            # Get feedback with sentiment data
            feedback_data = self.db_manager.get_feedback_with_sentiment()
            
            if not feedback_data:
                print("📭 No data available for visualization.")
                print("   Please add feedback and run sentiment analysis first.")
                return
            
            print("Available visualization options:")
            print("1. 🥧 Sentiment Distribution (Pie Chart)")
            print("2. 📊 Rating Distribution (Bar Chart)")
            print("3. 🏷️  Category Analysis (Horizontal Bar)")
            print("4. ☁️  Word Cloud")
            print("5. 📋 Summary Dashboard")
            print("6. 💾 Save All Charts to Folder")
            print("0. 🔙 Back to Main Menu")
            
            choice = input("\\nSelect visualization (0-6): ").strip()
            
            if choice == "1":
                self.visualizer.plot_sentiment_distribution(feedback_data)
                
            elif choice == "2":
                self.visualizer.plot_rating_distribution(feedback_data)
                
            elif choice == "3":
                self.visualizer.plot_category_analysis(feedback_data)
                
            elif choice == "4":
                text_data = [item['feedback_text'] for item in feedback_data 
                           if item.get('feedback_text')]
                if text_data:
                    self.visualizer.generate_wordcloud(text_data, 
                                                     title="Customer Feedback Word Cloud")
                else:
                    print("❌ No text data available for word cloud.")
                    
            elif choice == "5":
                self.visualizer.create_summary_dashboard(feedback_data)
                
            elif choice == "6":
                output_dir = input("Enter output directory (default: 'charts'): ").strip() or "charts"
                self.visualizer.save_all_charts(feedback_data, output_dir)
                print(f"✅ Charts saved to '{output_dir}/' directory")
                
            elif choice == "0":
                return
                
            else:
                print("❌ Invalid choice. Please select 0-6.")
        
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            traceback.print_exc()
    
    def view_statistics(self):
        """Display statistics and reports."""
        print("\\n📊 Statistics & Reports")
        print("-" * 40)
        
        try:
            # Get database statistics
            stats = self.db_manager.get_database_stats()
            
            print(f"📈 OVERALL STATISTICS")
            print(f"   Total Feedback Entries: {stats['total_feedback']}")
            print(f"   Average Rating: {stats['avg_rating']}/5.0")
            print(f"   Analysis Completion: {stats['analysis_completion_rate']}%")
            
            # Category distribution
            if stats['category_distribution']:
                print(f"\\n🏷️  FEEDBACK BY CATEGORY")
                for category, count in stats['category_distribution'].items():
                    percentage = (count / stats['total_feedback']) * 100
                    print(f"   {category}: {count} ({percentage:.1f}%)")
            
            # Sentiment summary
            sentiment_summary = self.db_manager.get_sentiment_summary()
            if sentiment_summary and sentiment_summary.get('total', 0) > 0:
                print(f"\\n🎭 SENTIMENT ANALYSIS SUMMARY")
                total_analyzed = sentiment_summary['total']
                for sentiment, data in sentiment_summary.items():
                    if sentiment != 'total' and isinstance(data, dict):
                        count = data['count']
                        avg_score = data['avg_score']
                        percentage = (count / total_analyzed) * 100
                        print(f"   {sentiment.title()}: {count} ({percentage:.1f}%) - Avg Score: {avg_score}")
            
            # Top keywords
            top_keywords = self.db_manager.get_top_keywords(limit=10)
            if top_keywords:
                print(f"\\n🔑 TOP KEYWORDS")
                for i, keyword_data in enumerate(top_keywords, 1):
                    keyword = keyword_data['keyword']
                    occurrences = keyword_data['total_occurrences']
                    relevance = keyword_data.get('avg_relevance', 0)
                    print(f"   {i:2d}. {keyword} (occurs {occurrences} times, relevance: {relevance:.2f})")
        
        except Exception as e:
            print(f"❌ Error retrieving statistics: {e}")
    
    def search_and_filter(self):
        """Search and filter feedback options."""
        print("\\n🔍 Search & Filter Feedback")
        print("-" * 40)
        
        try:
            print("Filter options:")
            print("1. 🏷️  By Category")
            print("2. ⭐ By Rating Range")
            print("3. 🎭 By Sentiment (if analyzed)")
            print("0. 🔙 Back to Main Menu")
            
            choice = input("\\nSelect filter option (0-3): ").strip()
            
            if choice == "1":
                self._filter_by_category()
            elif choice == "2":
                self._filter_by_rating()
            elif choice == "3":
                self._filter_by_sentiment()
            elif choice == "0":
                return
            else:
                print("❌ Invalid choice. Please select 0-3.")
        
        except Exception as e:
            print(f"❌ Error in search/filter: {e}")
    
    def _filter_by_category(self):
        """Filter feedback by category."""
        try:
            # Get all categories
            all_feedback = self.db_manager.get_all_feedback()
            categories = list(set(item['product_category'] for item in all_feedback))
            categories.sort()
            
            print("\\nAvailable categories:")
            for i, category in enumerate(categories, 1):
                print(f"  {i}. {category}")
            
            choice = int(input(f"\\nSelect category (1-{len(categories)}): ")) - 1
            if 0 <= choice < len(categories):
                selected_category = categories[choice]
                filtered_feedback = self.db_manager.get_feedback_by_category(selected_category)
                
                print(f"\\n📊 Feedback for '{selected_category}' ({len(filtered_feedback)} entries):")
                print("-" * 50)
                
                for feedback in filtered_feedback:
                    print(f"• {feedback['customer_name']} - {feedback['rating']}/5 stars")
                    print(f"  {feedback['feedback_text'][:80]}{'...' if len(feedback['feedback_text']) > 80 else ''}")
                    print()
            else:
                print("❌ Invalid category selection.")
                
        except ValueError:
            print("❌ Please enter a valid number.")
        except Exception as e:
            print(f"❌ Error filtering by category: {e}")
    
    def _filter_by_rating(self):
        """Filter feedback by rating range."""
        try:
            min_rating = int(input("Minimum rating (1-5): "))
            max_rating = int(input("Maximum rating (1-5): "))
            
            if not (1 <= min_rating <= 5 and 1 <= max_rating <= 5 and min_rating <= max_rating):
                print("❌ Invalid rating range.")
                return
            
            filtered_feedback = self.db_manager.get_feedback_by_rating(min_rating, max_rating)
            
            print(f"\\n📊 Feedback with ratings {min_rating}-{max_rating} stars ({len(filtered_feedback)} entries):")
            print("-" * 50)
            
            for feedback in filtered_feedback:
                stars = "⭐" * feedback['rating']
                print(f"• {feedback['customer_name']} - {stars} ({feedback['rating']}/5)")
                print(f"  Category: {feedback['product_category']}")
                print(f"  {feedback['feedback_text'][:80]}{'...' if len(feedback['feedback_text']) > 80 else ''}")
                print()
                
        except ValueError:
            print("❌ Please enter valid numbers.")
        except Exception as e:
            print(f"❌ Error filtering by rating: {e}")
    
    def _filter_by_sentiment(self):
        """Filter feedback by sentiment."""
        try:
            print("\\nAvailable sentiments:")
            print("1. 😊 Positive")
            print("2. 😞 Negative") 
            print("3. 😐 Neutral")
            
            choice = input("Select sentiment (1-3): ").strip()
            sentiment_map = {"1": "positive", "2": "negative", "3": "neutral"}
            
            if choice not in sentiment_map:
                print("❌ Invalid sentiment selection.")
                return
                
            selected_sentiment = sentiment_map[choice]
            
            # Get feedback with sentiment analysis
            all_feedback = self.db_manager.get_feedback_with_sentiment()
            filtered_feedback = [
                item for item in all_feedback 
                if item.get('sentiment_label') == selected_sentiment
            ]
            
            if not filtered_feedback:
                print(f"📭 No feedback found with '{selected_sentiment}' sentiment.")
                print("   Make sure to run sentiment analysis first (option 3).")
                return
            
            print(f"\\n📊 {selected_sentiment.title()} Feedback ({len(filtered_feedback)} entries):")
            print("-" * 50)
            
            for feedback in filtered_feedback:
                confidence = feedback.get('confidence', 0)
                print(f"• {feedback['customer_name']} - {feedback['rating']}/5 stars")
                print(f"  Confidence: {confidence:.2f} | Category: {feedback['product_category']}")
                print(f"  {feedback['feedback_text'][:80]}{'...' if len(feedback['feedback_text']) > 80 else ''}")
                print()
                
        except Exception as e:
            print(f"❌ Error filtering by sentiment: {e}")
    
    def manage_data(self):
        """Data management options."""
        print("\\n⚙️  Data Management")
        print("-" * 40)
        print("1. 🗑️  Delete Feedback Entry")
        print("2. 📤 Export Data to CSV")
        print("3. 🔄 Reset Database")
        print("0. 🔙 Back to Main Menu")
        
        choice = input("\\nSelect option (0-3): ").strip()
        
        if choice == "1":
            self._delete_feedback()
        elif choice == "2":
            self._export_data()
        elif choice == "3":
            self._reset_database()
        elif choice == "0":
            return
        else:
            print("❌ Invalid choice. Please select 0-3.")
    
    def _delete_feedback(self):
        """Delete a feedback entry."""
        try:
            feedback_id = int(input("Enter feedback ID to delete: "))
            
            # Confirm deletion
            confirm = input(f"Are you sure you want to delete feedback ID {feedback_id}? (y/N): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                success = self.db_manager.delete_feedback(feedback_id)
                if success:
                    print(f"✅ Feedback ID {feedback_id} deleted successfully.")
                else:
                    print(f"❌ Feedback ID {feedback_id} not found.")
            else:
                print("❌ Deletion cancelled.")
                
        except ValueError:
            print("❌ Please enter a valid feedback ID.")
        except Exception as e:
            print(f"❌ Error deleting feedback: {e}")
    
    def _export_data(self):
        """Export data to CSV."""
        try:
            import csv
            from datetime import datetime
            
            # Get all data
            feedback_data = self.db_manager.get_feedback_with_sentiment()
            
            if not feedback_data:
                print("📭 No data available to export.")
                return
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_export_{timestamp}.csv"
            
            # Write CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if feedback_data:
                    fieldnames = feedback_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(feedback_data)
            
            print(f"✅ Data exported to '{filename}'")
            print(f"   Exported {len(feedback_data)} records")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
    
    def _reset_database(self):
        """Reset the database (delete all data)."""
        print("\\n⚠️  WARNING: This will delete ALL data in the database!")
        confirm1 = input("Are you sure? Type 'yes' to continue: ").strip().lower()
        
        if confirm1 == "yes":
            confirm2 = input("This action cannot be undone! Type 'DELETE ALL' to confirm: ").strip()
            
            if confirm2 == "DELETE ALL":
                try:
                    # This would require implementing a reset method in DatabaseManager
                    print("✅ Database reset functionality would go here.")
                    print("   (Not implemented in this demo for safety)")
                except Exception as e:
                    print(f"❌ Error resetting database: {e}")
            else:
                print("❌ Reset cancelled.")
        else:
            print("❌ Reset cancelled.")
    
    def show_help(self):
        """Display help information."""
        print("\\n❓ Help & Information")
        print("-" * 40)
        print("""
🎯 ABOUT THIS APPLICATION:
   The Customer Feedback Analyzer helps you collect, analyze, and visualize
   customer feedback using Natural Language Processing (NLP) techniques.

📝 GETTING STARTED:
   1. Add customer feedback using option 1
   2. Analyze sentiment using option 3
   3. Generate visualizations using option 4
   
🧠 SENTIMENT ANALYSIS:
   - Positive: Happy, satisfied customers
   - Negative: Disappointed, frustrated customers  
   - Neutral: Mixed or objective feedback
   
📊 VISUALIZATIONS:
   - Pie charts for sentiment distribution
   - Bar charts for ratings and categories
   - Word clouds for key terms
   - Summary dashboards for overview
   
🔍 FILTERING:
   - Filter by product category
   - Filter by rating (1-5 stars)
   - Filter by sentiment (after analysis)
   
💡 TIPS:
   - Run sentiment analysis after adding feedback
   - Use visualizations to identify trends
   - Export data for external analysis
   - Check statistics regularly for insights
        """)
    
    def run(self):
        """Main application loop."""
        print("\\n🎉 Welcome to Customer Feedback Analyzer!")
        print("   A tool for analyzing customer sentiment using NLP")
        
        while True:
            try:
                self.display_main_menu()
                choice = input("\\nSelect an option (0-8): ").strip()
                
                if choice == "1":
                    self.add_new_feedback()
                elif choice == "2":
                    self.view_all_feedback()
                elif choice == "3":
                    self.analyze_all_sentiment()
                elif choice == "4":
                    self.generate_visualizations()
                elif choice == "5":
                    self.view_statistics()
                elif choice == "6":
                    self.search_and_filter()
                elif choice == "7":
                    self.manage_data()
                elif choice == "8":
                    self.show_help()
                elif choice == "0":
                    print("\\n👋 Thank you for using Customer Feedback Analyzer!")
                    print("   Happy analyzing! 📊")
                    break
                else:
                    print("❌ Invalid choice. Please select a number from 0-8.")
                
                # Wait for user to continue (except for exit)
                if choice != "0":
                    input("\\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Application interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Unexpected error: {e}")
                print("Please try again or contact support.")
                traceback.print_exc()


def main():
    """Entry point of the application."""
    try:
        # Create and run the application
        app = FeedbackAnalyzerApp()
        app.run()
        
    except KeyboardInterrupt:
        print("\\n\\n👋 Goodbye!")
    except Exception as e:
        print(f"\\n💥 Fatal error: {e}")
        print("\\nPlease check your installation and try again.")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()