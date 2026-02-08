#!/usr/bin/env python3
"""
Data preparation script for ATP 6-0.5 military manual fine-tuning.

This script processes raw text from the ATP 6-0.5 manual and converts it
into a format suitable for fine-tuning language models.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import random


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare ATP 6-0.5 manual data for fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw ATP 6-0.5 text file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training data split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Evaluation data split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and headers
    text = re.sub(r'ATP 6-0\.5\s+\d+\s+\w+\s+\d{4}', '', text)
    # Clean up newlines
    text = text.strip()
    return text


def extract_sections(text: str) -> List[Dict[str, str]]:
    """
    Extract sections from ATP 6-0.5 manual.
    
    ATP 6-0.5 typically has chapters, sections, and subsections.
    This extracts meaningful chunks for training.
    """
    sections = []
    
    # Split by chapter/section markers (adjust regex based on actual format)
    # Common patterns: "Chapter 1", "Section I", "1-1.", etc.
    section_pattern = r'(?:Chapter|Section|CHAPTER|SECTION)\s+[IVX\d]+[.-]?\s*[:\n]'
    parts = re.split(section_pattern, text)
    
    for i, part in enumerate(parts):
        if len(part.strip()) < 50:  # Skip very short sections
            continue
            
        # Clean the section
        cleaned = clean_text(part)
        
        if cleaned:
            sections.append({
                "section_id": i,
                "text": cleaned
            })
    
    return sections


def create_qa_pairs(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Create question-answer pairs from sections.
    
    For a real implementation, you would want to:
    1. Use more sophisticated chunking
    2. Generate diverse questions
    3. Possibly use GPT-4 to create high-quality Q&A pairs
    
    This is a simplified version for demonstration.
    """
    qa_pairs = []
    
    for section in sections:
        text = section["text"]
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, para in enumerate(paragraphs):
            if len(para) < 100:  # Skip very short paragraphs
                continue
            
            # Create different types of questions
            questions = [
                f"What does ATP 6-0.5 say about the following topic?",
                f"Explain the concept described in this section of ATP 6-0.5:",
                f"According to ATP 6-0.5, what are the key points about:",
                f"Summarize the ATP 6-0.5 guidance on:",
            ]
            
            # Use the beginning of the paragraph as context for the question
            context_words = para.split()[:10]
            context = " ".join(context_words)
            
            question = random.choice(questions)
            
            qa_pairs.append({
                "input": f"{question} {context}...",
                "output": para,
                "source": f"section_{section['section_id']}_para_{i}"
            })
    
    return qa_pairs


def create_instruction_pairs(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Create instruction-following pairs for CP delegation tasks.
    
    These are task-specific prompts related to Command Post delegation.
    """
    instruction_pairs = []
    
    # Example instruction templates for CP delegation
    templates = [
        {
            "input": "Describe the command post delegation procedures according to ATP 6-0.5.",
            "task": "delegation_procedures"
        },
        {
            "input": "What are the responsibilities of the Chief of Staff in CP operations?",
            "task": "roles_responsibilities"
        },
        {
            "input": "Explain the decision-making process for CP commanders.",
            "task": "decision_making"
        },
        {
            "input": "What are the key considerations for effective mission command?",
            "task": "mission_command"
        },
    ]
    
    # For each section, try to match it with relevant templates
    for section in sections:
        text = section["text"].lower()
        
        for template in templates:
            # Simple keyword matching (improve this with better NLP)
            task_keywords = {
                "delegation_procedures": ["delegate", "authority", "responsibility"],
                "roles_responsibilities": ["role", "duty", "responsibility", "chief"],
                "decision_making": ["decision", "assess", "analyze", "evaluate"],
                "mission_command": ["mission", "command", "principle", "philosophy"],
            }
            
            keywords = task_keywords.get(template["task"], [])
            if any(keyword in text for keyword in keywords):
                instruction_pairs.append({
                    "input": template["input"],
                    "output": clean_text(section["text"][:1000]),  # Limit length
                    "source": f"section_{section['section_id']}"
                })
    
    return instruction_pairs


def save_jsonl(data: List[Dict], filepath: str):
    """Save data in JSONL format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_data(
    data: List[Dict],
    train_split: float,
    eval_split: float,
    seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, eval, and test sets."""
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    train_size = int(n * train_split)
    eval_size = int(n * eval_split)
    
    train_data = data[:train_size]
    eval_data = data[train_size:train_size + eval_size]
    test_data = data[train_size + eval_size:]
    
    return train_data, eval_data, test_data


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print(f"Reading input file: {args.input}")
    
    # Read the input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("\nTo use this script, place your ATP 6-0.5 manual text file in data/raw/")
        print("Example: data/raw/atp_6_0_5.txt")
        print("\nCreating a sample dataset for demonstration...")
        
        # Create sample data for demonstration
        sample_data = create_sample_data()
        train_data, eval_data, test_data = split_data(
            sample_data,
            args.train_split,
            args.eval_split,
            args.seed
        )
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print("Extracting sections...")
        sections = extract_sections(text)
        print(f"Extracted {len(sections)} sections")
        
        print("Creating Q&A pairs...")
        qa_pairs = create_qa_pairs(sections)
        print(f"Created {len(qa_pairs)} Q&A pairs")
        
        print("Creating instruction pairs...")
        instruction_pairs = create_instruction_pairs(sections)
        print(f"Created {len(instruction_pairs)} instruction pairs")
        
        # Combine all data
        all_data = qa_pairs + instruction_pairs
        print(f"Total training examples: {len(all_data)}")
        
        # Split the data
        print("Splitting data...")
        train_data, eval_data, test_data = split_data(
            all_data,
            args.train_split,
            args.eval_split,
            args.seed
        )
    
    # Save the datasets
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"
    test_file = output_dir / "test.jsonl"
    
    print(f"\nSaving datasets:")
    print(f"  Train: {train_file} ({len(train_data)} examples)")
    print(f"  Eval:  {eval_file} ({len(eval_data)} examples)")
    print(f"  Test:  {test_file} ({len(test_data)} examples)")
    
    save_jsonl(train_data, str(train_file))
    save_jsonl(eval_data, str(eval_file))
    save_jsonl(test_data, str(test_file))
    
    print("\n✓ Data preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the processed data in {args.output}")
    print(f"  2. Run training: python src/train.py --config configs/llama_3_2_3b.yaml")


def create_sample_data() -> List[Dict[str, str]]:
    """Create sample data for demonstration when no input file is available."""
    sample_data = [
        {
            "input": "What is the purpose of mission command according to ATP 6-0.5?",
            "output": "Mission command is the conduct of military operations through decentralized execution based on mission-type orders. It empowers subordinate decision-making and decentralized execution appropriate to the situation, enabling commanders to accomplish missions and adapt to changing circumstances.",
            "source": "sample_1"
        },
        {
            "input": "Explain the role of the command post in military operations.",
            "output": "A command post (CP) is a facility that houses the commander, staff, and special units needed to control and direct operations. CPs enable commanders and staffs to synchronize operations, allocate resources, and direct forces. They provide the infrastructure for planning, preparing, executing, and assessing operations.",
            "source": "sample_2"
        },
        {
            "input": "What are the key principles of mission command?",
            "output": "The principles of mission command include: building cohesive teams through mutual trust, creating shared understanding, providing clear commander's intent, exercising disciplined initiative, using mission orders, and accepting prudent risk. These principles enable effective decentralized execution and adaptability in complex operations.",
            "source": "sample_3"
        },
        {
            "input": "Describe the decision-making process for CP operations.",
            "output": "The decision-making process involves understanding the operational environment, analyzing the mission, developing courses of action, comparing courses of action, and making and implementing the decision. Commanders use their experience, judgment, and intuition within this framework to make timely and effective decisions.",
            "source": "sample_4"
        },
        {
            "input": "What are the responsibilities of staff officers in a command post?",
            "output": "Staff officers assist commanders in planning, preparing, executing, and assessing operations. They coordinate activities across functional areas, provide recommendations, analyze information, synchronize operations, and maintain situational understanding. Staff officers enable commanders to exercise mission command effectively.",
            "source": "sample_5"
        },
    ]
    return sample_data


if __name__ == "__main__":
    main()
