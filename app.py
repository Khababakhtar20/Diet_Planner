import os
import streamlit as st
from typing import Dict, List, Any
import json
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Get API key from environment variable with a default message
API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL', 'https://api.aimlapi.com/v1')

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

class HealthAssistant:
    def __init__(self):
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY
        )
        self.nutrition_db = self._load_nutrition_db()
        self.regional_foods = self._load_regional_foods()
        
    def _load_nutrition_db(self) -> Dict:
        """Load comprehensive nutrition database"""
        return {
            "global_foods": {
                "chicken_breast": {"protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0, "calories": 165, "cost": "medium"},
                "salmon": {"protein": 20, "carbs": 0, "fat": 13, "fiber": 0, "calories": 208, "cost": "high"},
                "tofu": {"protein": 8, "carbs": 2, "fat": 4, "fiber": 0.3, "calories": 76, "cost": "low"},
                "brown_rice": {"protein": 2.6, "carbs": 23, "fat": 0.9, "fiber": 1.8, "calories": 112, "cost": "low"},
                "quinoa": {"protein": 4.4, "carbs": 21.3, "fat": 1.9, "fiber": 2.8, "calories": 120, "cost": "medium"},
                "sweet_potato": {"protein": 1.6, "carbs": 20.1, "fat": 0.1, "fiber": 3, "calories": 86, "cost": "low"},
                "spinach": {"protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2, "calories": 23, "cost": "low"},
                "chickpeas": {"protein": 8.9, "carbs": 27.4, "fat": 2.6, "fiber": 7.6, "calories": 164, "cost": "low"},
                "eggs": {"protein": 12.6, "carbs": 0.7, "fat": 9.5, "fiber": 0, "calories": 143, "cost": "low"},
                "greek_yogurt": {"protein": 10, "carbs": 3.6, "fat": 0.4, "fiber": 0, "calories": 59, "cost": "medium"},
                "avocado": {"protein": 2, "carbs": 8.5, "fat": 15, "fiber": 6.7, "calories": 160, "cost": "medium"},
                "almonds": {"protein": 21.2, "carbs": 21.7, "fat": 49.4, "fiber": 12.2, "calories": 579, "cost": "medium"},
                "oats": {"protein": 13.2, "carbs": 67.7, "fat": 6.9, "fiber": 10.1, "calories": 381, "cost": "low"},
                "banana": {"protein": 1.1, "carbs": 22.8, "fat": 0.3, "fiber": 2.6, "calories": 89, "cost": "low"},
                "beef": {"protein": 26.1, "carbs": 0, "fat": 11.8, "fiber": 0, "calories": 217, "cost": "high"},
                "lentils": {"protein": 9, "carbs": 20, "fat": 0.4, "fiber": 7.9, "calories": 116, "cost": "low"},
            },
            "nutrition_goals": {
                "weight_loss": {"protein": 30, "carbs": 40, "fat": 30, "calories_modifier": -300},
                "muscle_gain": {"protein": 40, "carbs": 40, "fat": 20, "calories_modifier": 300},
                "maintenance": {"protein": 30, "carbs": 45, "fat": 25, "calories_modifier": 0},
                "heart_health": {"protein": 25, "carbs": 50, "fat": 25, "focus": "omega3"},
                "diabetes_management": {"protein": 30, "carbs": 35, "fat": 35, "focus": "low_GI"},
                "anti_aging": {"protein": 30, "carbs": 40, "fat": 30, "focus": "antioxidants"},
                "athletic_performance": {"protein": 35, "carbs": 55, "fat": 10, "focus": "complex_carbs"},
            },
            "medical_considerations": {
                "diabetes": {"avoid": ["refined_sugar", "white_bread"], "prefer": ["low_GI_foods"]},
                "hypertension": {"avoid": ["excess_sodium"], "prefer": ["potassium_rich_foods"]},
                "celiac": {"avoid": ["gluten"], "prefer": ["gluten_free_grains"]},
                "lactose_intolerance": {"avoid": ["dairy"], "prefer": ["plant_based_alternatives"]},
                "gout": {"avoid": ["red_meat", "seafood"], "prefer": ["plant_proteins"]},
                "ibs": {"avoid": ["trigger_foods"], "prefer": ["fodmap_friendly_foods"]}
            }
        }
    
    def _load_regional_foods(self) -> Dict:
        """Load regional food availability database"""
        return {
            "North America": ["chicken_breast", "beef", "salmon", "sweet_potato", "kale", "quinoa", "almonds"],
            "South America": ["beans", "corn", "quinoa", "plantains", "cassava", "beef"],
            "Europe": ["chicken", "pork", "potatoes", "dairy", "rye_bread", "olive_oil"],
            "East Asia": ["rice", "tofu", "fish", "bok_choy", "seaweed", "mushrooms"],
            "South Asia": ["lentils", "rice", "chickpeas", "spinach", "yogurt", "chicken"],
            "Middle East": ["chickpeas", "lamb", "bulgur", "dates", "olive_oil", "yogurt"],
            "Africa": ["cassava", "plantains", "beans", "fish", "millet", "peanuts"],
            "Australia/Oceania": ["beef", "lamb", "fish", "sweet_potato", "macadamia_nuts"]
        }

    def diet_chatbot(self, message: str) -> str:
        """Interactive diet planning chatbot"""
        st.session_state.chat_history.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "system", "content": """You are a nutrition expert chatbot. Provide:
                    - Personalized meal advice considering location, medical conditions, and taste preferences
                    - Nutritional facts and calculations
        - Budget-friendly options
                    - Cultural food considerations
                    Be specific, helpful, and consider the user's region when suggesting foods."""},
                    *[{"role": msg["role"], "content": msg["content"]} 
                      for msg in st.session_state.chat_history[-6:]]
                ]
            )
            bot_message = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": bot_message})
            return bot_message
        except Exception as e:
            error_message = str(e)
            if "403" in error_message and "resource limit" in error_message:
                return "⚠️ API usage limit reached. Please update your payment method at https://aimlapi.com/app/billing to continue using the service."
            return f"Error: {error_message}"

    def generate_meal_plan(self, profile: Dict) -> Dict:
        """Generate meal plan with nutrition analysis"""
        try:
            # Adjust prompt based on user's region, medical conditions and preferences
            region_foods = self.regional_foods.get(profile.get("location", "North America"), [])
            region_foods_str = ", ".join(region_foods)
            
            medical_considerations = ""
            if profile.get("medical_conditions"):
                for condition in profile.get("medical_conditions", []):
                    if condition in self.nutrition_db["medical_considerations"]:
                        avoid = ", ".join(self.nutrition_db["medical_considerations"][condition]["avoid"])
                        prefer = ", ".join(self.nutrition_db["medical_considerations"][condition]["prefer"])
                        medical_considerations += f"\n- For {condition}: Avoid {avoid}. Prefer {prefer}."
            
            system_prompt = f"""Create a detailed 7-day meal plan considering:
            - Location: {profile.get('location', 'Not specified')} (common foods: {region_foods_str})
            - Age: {profile.get('age')} years
            - Diet type: {profile.get('diet_type')}
            - Goal: {profile.get('goal')}
            - Budget preference: {profile.get('budget', 'Medium')}
            - Taste preferences: {profile.get('taste_preferences', 'Not specified')}
            - Medical conditions: {', '.join(profile.get('medical_conditions', ['None']))}
            {medical_considerations}
            
            Format the meal plan day by day, with breakfast, lunch, dinner and 1-2 snacks.
            Include specific portion sizes and preparation methods.
            Focus on practical, easy-to-follow meals that align with the user's preferences.
            """
            
            response = self.client.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(profile)}
                ]
            )
            meal_plan = response.choices[0].message.content
            
            # Calculate estimated nutrition facts
            nutrition = self._analyze_meal_plan(meal_plan, profile.get("goal", "maintenance"))
            cost = self._estimate_cost(meal_plan)
        
            return {
                "plan": meal_plan,
                "nutrition": nutrition,
                "cost": cost
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_meal_plan(self, meal_plan: str, goal: str = "maintenance") -> Dict[str, Any]:
        """Calculate detailed nutrition facts for the meal plan"""
        nutrients = {
            "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "calories": 0,
            "estimated_daily": {
                "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "calories": 0
            }
        }
        
        # Count matches for foods in the nutrition database
        food_matches = 0
        for food, data in self.nutrition_db["global_foods"].items():
            count = meal_plan.lower().count(food)
            if count > 0:
                food_matches += count
                nutrients["protein"] += data["protein"] * count
                nutrients["carbs"] += data["carbs"] * count
                nutrients["fat"] += data["fat"] * count
                nutrients["fiber"] += data.get("fiber", 0) * count
                nutrients["calories"] += data["calories"] * count
        
        # Calculate daily estimates (assuming 7-day plan)
        if "day" in meal_plan.lower():
            days = 7
        else:
            days = 1
            
        for key in nutrients["estimated_daily"]:
            if key != "calories" and days > 0:
                nutrients["estimated_daily"][key] = round(nutrients[key] / days, 1)
        
        # Adjust based on goal
        goal_data = self.nutrition_db["nutrition_goals"].get(goal.lower().replace(" ", "_"), 
                                                          {"calories_modifier": 0})
        base_calories = 2000  # Default base
        if days > 0:
            nutrients["estimated_daily"]["calories"] = round((nutrients["calories"] / days) + 
                                                         goal_data.get("calories_modifier", 0), 0)
        
        # Add goal alignment data
        nutrients["goal_alignment"] = self._calculate_goal_alignment(nutrients["estimated_daily"], goal)
        
        return nutrients

    def _calculate_goal_alignment(self, daily_nutrients: Dict[str, float], goal: str) -> Dict[str, Any]:
        """Calculate how well the meal plan aligns with the nutrition goal"""
        goal_key = goal.lower().replace(" ", "_")
        if goal_key in self.nutrition_db["nutrition_goals"]:
            goal_data = self.nutrition_db["nutrition_goals"][goal_key]
            
            # Calculate macronutrient percentages from the meal plan
            total_calories = daily_nutrients["calories"]
            if total_calories > 0:
                protein_pct = (daily_nutrients["protein"] * 4 / total_calories) * 100
                carbs_pct = (daily_nutrients["carbs"] * 4 / total_calories) * 100
                fat_pct = (daily_nutrients["fat"] * 9 / total_calories) * 100
                
                # Calculate alignment scores (0-100%)
                protein_alignment = float(100 - min(abs(protein_pct - goal_data["protein"]) * 2, 100))
                carbs_alignment = float(100 - min(abs(carbs_pct - goal_data["carbs"]) * 2, 100))
                fat_alignment = float(100 - min(abs(fat_pct - goal_data["fat"]) * 2, 100))
                
                overall_alignment = float((protein_alignment + carbs_alignment + fat_alignment) / 3)
                
                return {
                    "protein_alignment": round(protein_alignment, 1),
                    "carbs_alignment": round(carbs_alignment, 1),
                    "fat_alignment": round(fat_alignment, 1),
                    "overall_alignment": round(overall_alignment, 1),
                    "macros_actual": {
                        "protein": round(protein_pct, 1),
                        "carbs": round(carbs_pct, 1),
                        "fat": round(fat_pct, 1)
                    },
                    "macros_target": {
                        "protein": goal_data["protein"],
                        "carbs": goal_data["carbs"],
                        "fat": goal_data["fat"]
                    }
                }
        
        # Default return if goal not found
        return {"overall_alignment": "N/A"}

    def _estimate_cost(self, meal_plan: str) -> Dict[str, Any]:
        """Estimate cost category and breakdown"""
        costs = []
        cost_breakdown = {"low": 0, "medium": 0, "high": 0}
        
        for food, data in self.nutrition_db["global_foods"].items():
            count = meal_plan.lower().count(food)
            if count > 0:
                costs.extend([data["cost"]] * count)
                cost_breakdown[data["cost"]] += count
        
        if not costs:
            return {"category": "Unknown", "breakdown": cost_breakdown}
        
        avg = sum(1 if c == "low" else 2 if c == "medium" else 3 for c in costs)/len(costs)
        category = ["Low", "Medium", "High"][int(avg)-1]
        
        total = sum(cost_breakdown.values())
        if total > 0:
            percentage_breakdown = {
                k: round((v / total) * 100, 1) for k, v in cost_breakdown.items()
            }
        else:
            percentage_breakdown = cost_breakdown
            
        return {
            "category": category,
            "breakdown": cost_breakdown,
            "percentages": percentage_breakdown
        }
        
    def get_specialized_advice(self, module: str, profile: Dict) -> str:
        """Get specialized health advice based on module"""
        try:
            module_prompts = {
                "Women's Health": f"Provide personalized women's health advice for a {profile.get('age')}-year-old with cycle length {profile.get('cycle')} days, pregnancy status: {profile.get('pregnancy')}. Address these concerns: {profile.get('concerns')}",
                "Child Health": f"Provide pediatric health advice for a {profile.get('age')}-year-old child weighing {profile.get('weight')}kg with these concerns: {profile.get('concerns')}",
                "Elderly Health": f"Provide geriatric health advice for a {profile.get('age')}-year-old with these health conditions: {profile.get('conditions')} and concerns: {profile.get('concerns')}"
            }
            
            response = self.client.chat.completions.create(
                model="o1",
                messages=[
                    {"role": "system", "content": module_prompts.get(module, "Provide health advice")},
                    {"role": "user", "content": json.dumps(profile)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating advice: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Health & Nutrition Assistant",
        page_icon="🍏",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        /* Enhanced theme colors */
        :root {
            --primary-color: #2ECC71;
            --primary-hover: #27AE60;
            --secondary-color: #3498DB;
            --background-color: #1A1A1A;
            --card-background: #2D2D2D;
            --text-color: #FFFFFF;
            --text-muted: #B0B0B0;
            --accent-color: #16A085;
            --border-color: #404040;
            --shadow-color: rgba(0,0,0,0.3);
            --success-color: #2ECC71;
            --error-color: #E74C3C;
        }

        /* Enhanced global styles */
        .stApp {
            background: linear-gradient(135deg, var(--background-color), #252525) !important;
            color: var(--text-color);
        }

        /* Enhanced navigation bar */
        .nav-container {
            background: linear-gradient(to right, var(--card-background), #363636);
            padding: 1.2rem 2.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px var(--shadow-color);
            margin-bottom: 2.5rem;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }

        .nav-container:hover {
            box-shadow: 0 6px 20px var(--shadow-color);
            transform: translateY(-2px);
        }

        .nav-links {
            display: flex;
            gap: 2.5rem;
            margin-top: 1.2rem;
        }

        .nav-link {
            color: var(--text-color);
            text-decoration: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }

        .nav-link:before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 2px;
            background-color: var(--primary-color);
            transition: width 0.3s ease;
        }

        .nav-link:hover:before {
            width: 100%;
        }

        .nav-link:hover {
            background-color: var(--primary-color);
            color: white;
            transform: translateY(-2px);
        }

        /* Enhanced card styling */
        .card {
            background: linear-gradient(145deg, var(--card-background), #333333);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 20px var(--shadow-color);
            margin-bottom: 1.5rem;
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .card:hover {
            transform: translateY(-8px) scale(1.01);
            box-shadow: 0 12px 30px var(--shadow-color);
        }

        /* Enhanced form elements */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {
            background-color: rgba(45, 45, 45, 0.9) !important;
            color: var(--text-color) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 0.8rem !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(46, 204, 113, 0.2) !important;
            transform: translateY(-2px);
        }

        /* Enhanced button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.8rem 1.5rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3) !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, var(--primary-hover), var(--primary-color)) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 6px 20px rgba(46, 204, 113, 0.4) !important;
        }

        /* Enhanced metric containers */
        .metric-container {
            background: linear-gradient(145deg, #2D2D2D, #333333);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid var(--primary-color);
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px var(--shadow-color);
            transition: all 0.3s ease;
        }

        .metric-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px var(--shadow-color);
        }

        .metric-value {
            color: var(--primary-color) !important;
            font-size: 28px !important;
            font-weight: bold !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .metric-label {
            color: var(--text-muted) !important;
            font-size: 16px !important;
            margin-top: 5px;
        }

        /* Enhanced progress bar */
        .stProgress > div > div {
            background: linear-gradient(to right, var(--primary-color), var(--primary-hover)) !important;
            height: 8px !important;
            border-radius: 4px !important;
        }

        /* Enhanced plot container */
        .plot-container {
            background: linear-gradient(145deg, #2D2D2D, #333333);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 20px var(--shadow-color);
            transition: all 0.3s ease;
        }

        .plot-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px var(--shadow-color);
        }

        /* Enhanced success/error messages */
        .success-msg {
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(39, 174, 96, 0.1));
            color: var(--success-color);
            padding: 1.2rem;
            border-radius: 10px;
            border-left: 5px solid var(--success-color);
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.1);
            margin: 1.5rem 0;
        }

        .error-msg {
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(192, 57, 43, 0.1));
            color: var(--error-color);
            padding: 1.2rem;
            border-radius: 10px;
            border-left: 5px solid var(--error-color);
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.1);
            margin: 1.5rem 0;
        }

        /* Enhanced headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color) !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            margin-bottom: 1rem !important;
        }

        /* Enhanced chat interface */
        .stChatMessage {
            background: linear-gradient(145deg, var(--card-background), #333333) !important;
            border-radius: 12px !important;
            padding: 1.2rem !important;
            margin: 1rem 0 !important;
            box-shadow: 0 4px 15px var(--shadow-color) !important;
            transition: all 0.3s ease !important;
        }

        .stChatMessage:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px var(--shadow-color) !important;
        }

        /* Enhanced tabs */
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 8px !important;
            padding: 1rem 1.5rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(46, 204, 113, 0.1) !important;
            transform: translateY(-2px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
        }

        /* Loading animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .loading {
            animation: pulse 1.5s infinite;
        }

        /* Fade-in animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-out forwards;
        }

        /* Add tooltips */
        [data-tooltip] {
            position: relative;
            cursor: help;
        }

        [data-tooltip]:before {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.5rem 1rem;
            background: rgba(0,0,0,0.8);
            color: white;
            border-radius: 4px;
            font-size: 14px;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }

        [data-tooltip]:hover:before {
            opacity: 1;
            visibility: visible;
        }

        /* Button click effect */
        .stButton > button:active {
            transform: scale(0.95) !important;
        }

        /* Input focus highlight */
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            transform: scale(1.02);
        }

        /* Card hover effect */
        .card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

    # Add this right after the CSS, before your main content
    st.markdown("""
    <div class="nav-container">
        <h1>🍏 Health & Nutrition Assistant</h1>
        <div class="nav-links">
            <a href="#" class="nav-link active">Home</a>
            <a href="#" class="nav-link">Diet Planner</a>
            <a href="#" class="nav-link">Chat Assistant</a>
            <a href="#" class="nav-link">Health Modules</a>
            <a href="#" class="nav-link">About</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API Key setup - only show if not already set in .env
    if not API_KEY:
        with st.sidebar:
            api_key = st.text_input("Enter API Key:", type="password")
            if api_key:
                os.environ["API_KEY"] = api_key
                st.success("API Key set!")
            else:
                st.warning("Please provide an API Key to use the assistant")
                st.stop()
    
    # Add this in the sidebar section
    with st.sidebar:
        st.markdown("""
        <div class="card">
            <h3>💡 API Usage Note</h3>
            <p>This application uses API credits for generating responses. If you encounter any issues, please ensure your API key has sufficient credits.</p>
        </div>
        """, unsafe_allow_html=True)
    
    assistant = HealthAssistant()
    
    st.title("🍏 Health & Nutrition Assistant")
    
    tab1, tab2, tab3 = st.tabs(["🍽️ Diet Planner", "💬 Chat Assistant", "🏥 Health Modules"])
    
    with tab1:
        st.markdown("""
            <div class="card">
                <h2>Personalized Meal Planner</h2>
                <p>Create your customized meal plan based on your preferences and goals.</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("diet_planner_form"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 👤 Personal Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", 1, 120, 30)
                weight = st.number_input("Weight (kg)", 30, 200, 70)
                height = st.number_input("Height (cm)", 100, 250, 170)
                
            with col2:
                gender = st.radio("Gender", ["Male", "Female", "Other"])
                location = st.selectbox("Location", 
                    ["North America", "South America", "Europe", "East Asia", 
                     "South Asia", "Middle East", "Africa", "Australia/Oceania"])
                
            with col3:
                diet_type = st.selectbox(
                    "Diet Type",
                    ["Omnivore", "Vegetarian", "Vegan", "Pescatarian", "Flexitarian", "Keto", "Paleo"])
                medical_conditions = st.multiselect(
                    "Medical Conditions",
                    ["None", "Diabetes", "Hypertension", "Celiac", "Lactose Intolerance", 
                     "Gout", "IBS", "Food Allergies"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Diet Preferences")
            col1, col2 = st.columns(2)
            
            with col1:
                activity = st.select_slider(
                    "Activity Level",
                    ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
                goal = st.selectbox(
                    "Primary Goal",
                    ["Weight Loss", "Muscle Gain", "Maintenance", "Heart Health", 
                     "Diabetes Management", "Anti-Aging", "Athletic Performance"])
                budget = st.select_slider(
                    "Budget Preference",
                    ["Low", "Medium", "High"])
                
            with col2:
                taste_preferences = st.text_area("Taste Preferences (e.g., spicy, sweet, savory)", height=80)
                food_dislikes = st.text_area("Foods You Dislike", height=80)
            st.markdown('</div>', unsafe_allow_html=True)
            
            submit = st.form_submit_button("Generate Meal Plan")
            
        if submit:
            with st.spinner("🔮 Creating your personalized nutrition plan... Please wait while we analyze your preferences..."):
                profile = {
                    "age": age,
                    "weight": weight,
                    "height": height,
                    "gender": gender,
                    "location": location,
                    "diet_type": diet_type,
                    "activity": activity,
                    "goal": goal,
                    "budget": budget,
                    "taste_preferences": taste_preferences,
                    "food_dislikes": food_dislikes,
                    "medical_conditions": [c for c in medical_conditions if c != "None"]
                }
                result = assistant.generate_meal_plan(profile)
                
                if "error" in result:
                    st.markdown(f"""
                        <div class="error-msg">
                            {result["error"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="success-msg">
                            Your personalized meal plan is ready!
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📋 Meal Plan")
                        st.markdown(result["plan"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📥 Download Meal Plan",
                            data=result["plan"],
                            file_name="my_meal_plan.txt"
                        )
                    
                    with col2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📊 Nutrition Analysis")
                        
                        if "estimated_daily" in result["nutrition"]:
                            daily = result["nutrition"]["estimated_daily"]
                            
                            # Display metrics in styled containers
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-value">{daily.get('calories', 0)} kcal</div>
                            <div class="metric-label">Daily Calories</div>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.metric("Protein", f"{daily.get('protein', 0)}g")
                                st.metric("Carbs", f"{daily.get('carbs', 0)}g")
                                st.markdown('</div>', unsafe_allow_html=True)
                            with col_b:
                                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                st.metric("Fat", f"{daily.get('fat', 0)}g")
                                st.metric("Fiber", f"{daily.get('fiber', 0)}g")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.metric("Cost", result["cost"].get("category", "Unknown"))
                            
                            if result["nutrition"].get("goal_alignment"):
                                alignment = result["nutrition"]["goal_alignment"]
                                
                                if isinstance(alignment, dict) and alignment.get("overall_alignment") != "N/A":
                                    st.markdown("### 🎯 Goal Alignment")
                                    st.progress(float(alignment["overall_alignment"])/100)
                                    st.write(f"Overall: {alignment['overall_alignment']}%")
                                    
                                    if "macros_actual" in alignment and "macros_target" in alignment:
                                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                        actual = alignment["macros_actual"]
                                        target = alignment["macros_target"]
                                        
                                        plt.style.use('dark_background')
                                        fig, ax = plt.subplots(figsize=(5, 3))
                                        ax.set_facecolor('#2D2D2D')
                                        fig.patch.set_facecolor('#2D2D2D')
                                        
                                        comparison_df = pd.DataFrame({
                                            'Macronutrient': ['Protein', 'Carbs', 'Fat'],
                                            'Your Plan': [actual.get("protein", 0), actual.get("carbs", 0), actual.get("fat", 0)],
                                            'Target': [target.get("protein", 0), target.get("carbs", 0), target.get("fat", 0)]
                                        })
                                        
                                        comparison_df.plot(x='Macronutrient', kind='bar', ax=ax)
                                        plt.ylabel('Percentage')
                                        plt.title('Macronutrient Distribution')
                                        st.pyplot(fig)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.write("Goal alignment could not be calculated")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Nutrition Chat Assistant")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if prompt := st.chat_input("Ask about nutrition..."):
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking deeply about nutrition..."):
                    response = assistant.diet_chatbot(prompt)
                    if response.startswith("⚠️"):
                        st.error(response)  # Display as error message
                    else:
                        st.write(response)
    
    with tab3:
        st.subheader("Specialized Health Modules")
        
        module = st.radio(
            "Select Module",
            ["Women's Health", "Child Health", "Elderly Health"],
            horizontal=True)
        
        if module == "Women's Health":
            with st.form("womens_health_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age", min_value=12, max_value=100, value=30)
                    cycle = st.number_input("Cycle Length (days)", min_value=20, max_value=40, value=28)
                with col2:
                    pregnancy = st.selectbox(
                        "Pregnancy Status",
                        ["Not Pregnant", "Pregnant", "Postpartum", "Trying to Conceive"])
                    concerns = st.text_area("Specific Concerns")
                
                submit_button = st.form_submit_button("Get Advice")
                
            if submit_button:
                with st.spinner("📊 Analyzing health data and generating personalized recommendations..."):
                    profile = {
                        "age": age,
                        "cycle": cycle,
                        "pregnancy": pregnancy,
                        "concerns": concerns
                    }
                    advice = assistant.get_specialized_advice(module, profile)
                    st.success("Here's your personalized health guidance:")
                    st.write(advice)
        
        elif module == "Child Health":
            with st.form("child_health_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Child's Age", min_value=1, max_value=18, value=8)
                    weight = st.number_input("Weight (kg)", min_value=5, max_value=100, value=30)
                with col2:
                    development = st.selectbox(
                        "Development Stage",
                        ["Toddler", "Preschool", "School Age", "Teenager"])
                    concerns = st.text_area("Health Concerns")
                
                submit_button = st.form_submit_button("Get Child Health Advice")
                
            if submit_button:
                with st.spinner("📊 Analyzing health data and generating personalized recommendations..."):
                    profile = {
                        "age": age,
                        "weight": weight,
                        "development": development,
                        "concerns": concerns
                    }
                    advice = assistant.get_specialized_advice(module, profile)
                    st.success("Child Health Recommendations:")
                    st.write(advice)
        
        elif module == "Elderly Health":
            with st.form("elderly_health_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age", min_value=60, max_value=120, value=70)
                    conditions = st.multiselect(
                        "Existing Conditions",
                        ["Hypertension", "Diabetes", "Arthritis", "Heart Disease", "Osteoporosis", "None"],
                        default=["None"])
                with col2:
                    mobility = st.select_slider(
                        "Mobility Level",
                        options=["Bedridden", "Uses Wheelchair", "Uses Walker", "Uses Cane", "Independent"])
                    concerns = st.text_area("Specific Concerns")
                
                submit_button = st.form_submit_button("Get Senior Health Advice")
                
            if submit_button:
                with st.spinner("📊 Analyzing health data and generating personalized recommendations..."):
                    profile = {
                        "age": age,
                        "conditions": ", ".join(conditions),
                        "mobility": mobility,
                        "concerns": concerns
                    }
                    advice = assistant.get_specialized_advice(module, profile)
                    st.success("Senior Health Recommendations:")
                    st.write(advice)

if __name__ == "__main__":
    main()