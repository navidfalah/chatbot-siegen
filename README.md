# Chatbot Siegen

An intelligent conversational AI system that uses advanced analysis to understand and respond to user queries across multiple domains.

## Features

- **Intelligent Relevance Analysis**: Determines if questions are within the system's knowledge domains
- **Context-Aware Responses**: Maintains conversation history for better understanding
- **Multi-Domain Support**: Handles questions about programming, science, technology, and general topics
- **Adaptive Clarification**: Asks specific questions when more information is needed
- **Confidence Scoring**: Uses confidence metrics to determine response strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/navidfalah/chatbot-siegen.git
cd chatbot-siegen

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

```python
from src.main import run_chatbot

# Start the chatbot
run_chatbot()
```

Or run directly from command line:

```bash
python -m src.main
```

## Project Structure

```
chatbot-siegen/
├── src/
│   ├── ai/              # AI engine and analysis components
│   ├── conversation/    # Conversation management system
│   ├── models/         # Data models and structures
│   └── utils/          # Utilities and constants
├── tests/              # Unit tests
└── examples/           # Usage examples
```

## Usage Example

```python
from src.conversation.conversation_system import ConversationSystem

# Initialize the conversation system
chatbot = ConversationSystem()

# Start interactive conversation
chatbot.start_point()
```

## Configuration

You can customize the chatbot's behavior by modifying the constants in `src/utils/constants.py`:

- Knowledge domains
- Confidence thresholds
- Maximum clarification attempts
- Response templates

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Navid Falah

## Acknowledgments

- Built with Python 3.8+
- Inspired by modern conversational AI systems
