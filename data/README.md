# Data Directory

This directory contains training data for fine-tuning SLMs on ATP 6-0.5.

## Structure

```
data/
├── raw/              # Raw ATP 6-0.5 manual data
│   └── atp_6_0_5.txt (place your manual here)
└── processed/        # Processed training data
    ├── train.jsonl
    ├── eval.jsonl
    └── test.jsonl
```

## Data Preparation

1. Place your ATP 6-0.5 manual text file in `data/raw/`
2. Run the data preparation script:

```bash
python src/prepare_data.py --input data/raw/atp_6_0_5.txt --output data/processed/
```

This will:
- Extract sections from the manual
- Create Q&A pairs
- Generate instruction-following examples
- Split into train/eval/test sets

## Data Format

The processed data is in JSONL format with the following structure:

```json
{
  "input": "Question or instruction",
  "output": "Answer or response",
  "source": "Source reference"
}
```

## ATP 6-0.5 Information

ATP 6-0.5 (Mission Command) is a U.S. Army field manual that describes:
- Mission command philosophy and principles
- Command post organization and operations
- Decision-making processes
- Staff responsibilities and procedures
- CP delegation tasks

For the actual manual, visit: https://armypubs.army.mil/

## Sample Data

If you don't have the manual yet, the script will generate sample data for testing.
