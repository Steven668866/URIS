#!/usr/bin/env python3
"""
Dataset Validator - Ensures all samples meet token requirements
Validates the generated dataset before fine-tuning
"""

import json
import tiktoken
from pathlib import Path
from collections import Counter


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    encoder = tiktoken.encoding_for_model("gpt-4")
    return len(encoder.encode(text))


def validate_dataset(dataset_path: str = "dataset_personalization.json"):
    """Validate dataset format and token counts"""
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    print(f"📖 Loading dataset: {dataset_path}\n")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"✅ Total samples: {len(samples)}\n")
    
    # Validation results
    valid_samples = 0
    invalid_samples = []
    token_counts = []
    issues = []
    
    # Profile statistics
    dietary_prefs = []
    
    for idx, sample in enumerate(samples):
        # Check structure
        if "conversations" not in sample:
            issues.append(f"Sample {idx}: Missing 'conversations' field")
            continue
        
        if "images" not in sample:
            issues.append(f"Sample {idx}: Missing 'images' field")
            continue
        
        conversations = sample["conversations"]
        
        # Check conversation format
        if len(conversations) != 3:
            issues.append(f"Sample {idx}: Expected 3 conversation turns, got {len(conversations)}")
            continue
        
        # Extract messages
        try:
            system_msg = conversations[0]["value"]
            user_msg = conversations[1]["value"]
            assistant_msg = conversations[2]["value"]
        except (KeyError, IndexError) as e:
            issues.append(f"Sample {idx}: Invalid conversation structure - {e}")
            continue
        
        # Count tokens
        total_tokens = (
            count_tokens(system_msg) +
            count_tokens(user_msg) +
            count_tokens(assistant_msg)
        )
        
        token_counts.append(total_tokens)
        
        # Check token limit
        if total_tokens > 800:
            invalid_samples.append({
                "index": idx,
                "tokens": total_tokens,
                "system_tokens": count_tokens(system_msg),
                "user_tokens": count_tokens(user_msg),
                "assistant_tokens": count_tokens(assistant_msg)
            })
        else:
            valid_samples += 1
        
        # Extract dietary preference
        if "Dietary:" in system_msg:
            dietary = system_msg.split("Dietary:")[1].split(",")[0].strip()
            dietary_prefs.append(dietary)
    
    # Print results
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\n✅ Valid samples (≤800 tokens): {valid_samples}/{len(samples)}")
    print(f"❌ Invalid samples (>800 tokens): {len(invalid_samples)}/{len(samples)}")
    print(f"⚠️  Structural issues: {len(issues)}")
    
    if token_counts:
        print(f"\n📊 Token Statistics:")
        print(f"   Average: {sum(token_counts)/len(token_counts):.1f} tokens")
        print(f"   Min: {min(token_counts)} tokens")
        print(f"   Max: {max(token_counts)} tokens")
        print(f"   Median: {sorted(token_counts)[len(token_counts)//2]} tokens")
        
        # Token distribution
        ranges = [
            (0, 400, "0-400"),
            (400, 600, "400-600"),
            (600, 700, "600-700"),
            (700, 800, "700-800"),
            (800, 1000, "800-1000 (INVALID)")
        ]
        
        print(f"\n📈 Token Distribution:")
        for min_t, max_t, label in ranges:
            count = sum(1 for t in token_counts if min_t <= t < max_t)
            percentage = (count / len(token_counts)) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {label:15} | {bar:25} {count:3} ({percentage:5.1f}%)")
    
    # Dietary preference diversity
    if dietary_prefs:
        print(f"\n🍽️  Dietary Preference Diversity:")
        pref_counts = Counter(dietary_prefs)
        for pref, count in pref_counts.most_common():
            percentage = (count / len(dietary_prefs)) * 100
            print(f"   {pref:20} {count:3} ({percentage:5.1f}%)")
    
    # Show invalid samples
    if invalid_samples:
        print(f"\n❌ Invalid Samples (>800 tokens):")
        for item in invalid_samples[:5]:  # Show first 5
            print(f"   Sample {item['index']:3}: {item['tokens']} tokens")
            print(f"      System: {item['system_tokens']}, User: {item['user_tokens']}, Assistant: {item['assistant_tokens']}")
        
        if len(invalid_samples) > 5:
            print(f"   ... and {len(invalid_samples) - 5} more")
    
    # Show structural issues
    if issues:
        print(f"\n⚠️  Structural Issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue}")
        
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
    
    print("\n" + "=" * 60)
    
    # Final verdict
    if len(invalid_samples) == 0 and len(issues) == 0:
        print("✅ DATASET IS VALID - Ready for fine-tuning!")
        return True
    elif len(invalid_samples) > 0:
        print(f"⚠️  WARNING: {len(invalid_samples)} samples exceed token limit")
        print("   Consider regenerating with stricter length constraints")
        return False
    else:
        print(f"❌ DATASET HAS ISSUES - Please fix before fine-tuning")
        return False


def main():
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_personalization.json"
    
    is_valid = validate_dataset(dataset_path)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()






