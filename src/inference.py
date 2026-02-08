#!/usr/bin/env python3
"""
Inference script for fine-tuned SLM.

This script loads a fine-tuned model and runs inference
for ATP 6-0.5 related queries.
"""

import argparse
import torch
from unsloth import FastLanguageModel


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned SLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt/question to ask the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 to 2.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Enable inference mode (faster)
        FastLanguageModel.for_inference(model)
        
        print("✓ Model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have trained a model first:")
        print("  python src/train.py --config configs/llama_3_2_3b.yaml")
        raise


def format_prompt(question: str) -> str:
    """Format the input question as a prompt."""
    prompt = f"""### Instruction:
You are an expert on ATP 6-0.5 (Mission Command). Answer the following question based on the manual.

### Question:
{question}

### Answer:
"""
    return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a response from the model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part (after "### Answer:")
    if "### Answer:" in response:
        response = response.split("### Answer:")[-1].strip()
    
    return response


def run_interactive(model, tokenizer, args):
    """Run in interactive mode."""
    print("\n" + "="*60)
    print("Interactive Mode - ATP 6-0.5 Expert System")
    print("="*60)
    print("Ask questions about ATP 6-0.5 (Mission Command)")
    print("Type 'quit' or 'exit' to end the session")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            question = input("Question: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            if not question:
                continue
            
            # Format prompt
            prompt = format_prompt(question)
            
            # Generate response
            print("\nGenerating response...\n")
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            
            print("Answer:")
            print("-" * 60)
            print(response)
            print("-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Run in interactive or single-query mode
    if args.interactive:
        run_interactive(model, tokenizer, args)
    else:
        # Single query mode
        if not args.prompt:
            print("Error: --prompt is required in non-interactive mode")
            print("Use --interactive for interactive mode")
            return
        
        # Format prompt
        prompt = format_prompt(args.prompt)
        
        print(f"\nQuestion: {args.prompt}\n")
        print("Generating response...\n")
        
        # Generate response
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        
        print("Answer:")
        print("="*60)
        print(response)
        print("="*60)


if __name__ == "__main__":
    main()
