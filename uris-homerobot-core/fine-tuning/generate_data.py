#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Qwen2-VL Home Robot Fine-tuning
Generates personalized conversation data under 800 tokens per sample
"""

import asyncio
import json
import os
from typing import List, Dict, Optional
from datetime import datetime

import tiktoken
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import random

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL", "gpt-4")
MAX_TOKENS_PER_SAMPLE = 800
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "100"))
MAX_RETRIES = 5
OUTPUT_FILE = "dataset_personalization.json"


# Pydantic Models for Structured Generation
class UserProfile(BaseModel):
    """User persona with habits and preferences"""
    dietary_preference: str = Field(description="e.g., Vegetarian, Vegan, Keto, No restrictions")
    allergies: Optional[str] = Field(description="Food allergies if any")
    favorite_cuisine: str = Field(description="e.g., Italian, Chinese, Japanese")
    health_goal: Optional[str] = Field(description="e.g., Weight loss, Muscle gain, General health")
    lifestyle: str = Field(description="e.g., Busy professional, Student, Retiree")


class ScenarioData(BaseModel):
    """Complete scenario with visual and interaction data"""
    location: str = Field(description="Kitchen, Living room, Bedroom, etc.")
    visual_description: str = Field(description="Detailed description of objects/scene (150-200 words)")
    user_command: str = Field(description="Natural user request (20-40 words)")
    assistant_response: str = Field(description="Personalized response referencing user profile (80-120 words)")


# Token Counter
encoder = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    return len(encoder.encode(text))


def generate_user_profile_text(profile: UserProfile) -> str:
    """Convert UserProfile to concise text"""
    parts = [
        f"Dietary: {profile.dietary_preference}",
        f"Cuisine: {profile.favorite_cuisine}",
        f"Lifestyle: {profile.lifestyle}"
    ]
    if profile.allergies:
        parts.append(f"Allergies: {profile.allergies}")
    if profile.health_goal:
        parts.append(f"Goal: {profile.health_goal}")
    return ", ".join(parts)


# Predefined variety for rich generation
LOCATIONS = [
    "Kitchen with modern appliances",
    "Living room with smart home devices",
    "Home office workspace",
    "Bedroom with morning sunlight",
    "Dining room table",
    "Kitchen pantry area",
    "Balcony garden",
    "Home gym corner"
]

DIETARY_OPTIONS = [
    "Vegetarian", "Vegan", "Pescatarian", "Keto", "Paleo",
    "Low-carb", "Gluten-free", "Dairy-free", "No restrictions"
]

CUISINES = [
    "Italian", "Chinese", "Japanese", "Mexican", "Indian",
    "Mediterranean", "Thai", "Korean", "French", "American"
]

LIFESTYLES = [
    "Busy professional", "Student", "Retiree", "Parent with kids",
    "Fitness enthusiast", "Work-from-home", "Night shift worker"
]

ALLERGIES = [None, "Nuts", "Shellfish", "Dairy", "Gluten", "Soy", None, None]

HEALTH_GOALS = [
    None, "Weight loss", "Muscle gain", "Heart health",
    "Energy boost", "Better sleep", None, None
]


async def generate_sample(client: AsyncOpenAI, semaphore: asyncio.Semaphore) -> Optional[Dict]:
    """Generate a single training sample with retry logic"""
    
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                # Generate random user profile
                profile = UserProfile(
                    dietary_preference=random.choice(DIETARY_OPTIONS),
                    allergies=random.choice(ALLERGIES),
                    favorite_cuisine=random.choice(CUISINES),
                    health_goal=random.choice(HEALTH_GOALS),
                    lifestyle=random.choice(LIFESTYLES)
                )
                
                profile_text = generate_user_profile_text(profile)
                location = random.choice(LOCATIONS)
                
                # Construct generation prompt
                generation_prompt = f"""You are a data generation assistant for a home robot AI.

**Task:** Generate ONE complete training sample for personalized home assistance.

**User Profile:** {profile_text}
**Location:** {location}

**Requirements:**
1. **Visual Description**: Describe a realistic home scene with 5-8 objects/items. Be specific (e.g., "3 red tomatoes, a bottle of olive oil"). 150-200 words.

2. **User Command**: A natural question or request the user might say while looking at the scene. Must be 20-40 words. Examples:
   - "What can I make for dinner with these ingredients?"
   - "Is this meal suitable for my diet?"
   - "Suggest a recipe using what's here"

3. **Assistant Response**: 
   - MUST explicitly reference the user's dietary preference, lifestyle, or health goal
   - Provide helpful, personalized advice (80-120 words)
   - Be conversational and warm
   - Suggest specific actions based on the visual scene

**Output Format (JSON):**
{{
  "visual_description": "...",
  "user_command": "...",
  "assistant_response": "..."
}}

Keep responses concise to stay under token limits. Generate now:"""

                # Call OpenAI API
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise data generator. Output only valid JSON."},
                        {"role": "user", "content": generation_prompt}
                    ],
                    temperature=0.9,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                generated_text = response.choices[0].message.content
                scenario = json.loads(generated_text)
                
                # Build final sample
                system_message = f"Current User Profile: {profile_text}"
                user_message = f"<image> {scenario['user_command']}"
                assistant_message = scenario['assistant_response']
                
                # Token validation
                total_tokens = (
                    count_tokens(system_message) +
                    count_tokens(user_message) +
                    count_tokens(assistant_message)
                )
                
                if total_tokens > MAX_TOKENS_PER_SAMPLE:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        return None  # Skip this sample after max retries
                
                # Construct final sample in ShareGPT format
                sample = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": system_message
                        },
                        {
                            "from": "user",
                            "value": user_message
                        },
                        {
                            "from": "assistant",
                            "value": assistant_message
                        }
                    ],
                    "images": ["images/placeholder.jpg"],
                    "metadata": {
                        "location": location,
                        "visual_description": scenario['visual_description'],
                        "token_count": total_tokens
                    }
                }
                
                return sample
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"\n⚠️  Sample generation failed after {MAX_RETRIES} attempts: {str(e)}")
                    return None
    
    return None


async def generate_dataset(num_samples: int):
    """Generate complete dataset with progress tracking"""
    
    print(f"🚀 Starting dataset generation...")
    print(f"📊 Target: {num_samples} samples")
    print(f"🔒 Token limit: {MAX_TOKENS_PER_SAMPLE} per sample")
    print(f"🌐 API Endpoint: {BASE_URL}")
    print(f"🤖 Model: {MODEL}\n")
    
    # Initialize async OpenAI client
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    
    # Semaphore to limit concurrent requests (avoid rate limits)
    semaphore = asyncio.Semaphore(10)
    
    # Generate samples with progress bar
    tasks = [generate_sample(client, semaphore) for _ in range(num_samples)]
    samples = await tqdm_asyncio.gather(*tasks, desc="🔄 Generating samples")
    
    # Filter out None values (failed generations)
    valid_samples = [s for s in samples if s is not None]
    
    print(f"\n✅ Successfully generated: {len(valid_samples)}/{num_samples} samples")
    
    # Calculate statistics
    token_counts = [s['metadata']['token_count'] for s in valid_samples]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    
    print(f"\n📈 Token Statistics:")
    print(f"   Average: {avg_tokens:.1f} tokens")
    print(f"   Min: {min_tokens} tokens")
    print(f"   Max: {max_tokens} tokens")
    print(f"   Limit: {MAX_TOKENS_PER_SAMPLE} tokens")
    
    # Save to file (remove metadata for training)
    output_samples = []
    for sample in valid_samples:
        output_sample = {
            "conversations": sample["conversations"],
            "images": sample["images"]
        }
        output_samples.append(output_sample)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Dataset saved to: {OUTPUT_FILE}")
    print(f"📦 File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")
    
    # Save metadata separately
    metadata_file = OUTPUT_FILE.replace('.json', '_metadata.json')
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "total_samples": len(valid_samples),
        "token_limit": MAX_TOKENS_PER_SAMPLE,
        "model_used": MODEL,
        "statistics": {
            "avg_tokens": avg_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens
        }
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {metadata_file}")
    print("\n🎉 Generation complete!")


def main():
    """Main entry point"""
    
    # Validate environment
    if not API_KEY:
        raise ValueError("❌ OPENAI_API_KEY not found in environment variables!")
    
    # Run async generation
    asyncio.run(generate_dataset(NUM_SAMPLES))


if __name__ == "__main__":
    main()






