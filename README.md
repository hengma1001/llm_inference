# Python Project Template

This project is to simplify llm query to the ALCF hosted models and Huggingface.


#### Setup
```bash
git clone https://github.com/hengma1001/llm_inference.git
cd llm_inference
pip install .
```

## Getting Started

1. Get authenticated with ALCF via Globus tools
```bash
python src/llm_inference/inference_auth_token.py authenticate
```

2. Run example runs
```bash
python examples/test_alcf.py
```

## Contributing

