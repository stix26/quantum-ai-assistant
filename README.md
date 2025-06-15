# Quantum AI Assistant

A sophisticated quantum-powered AI assistant that leverages IBM Quantum Experience to process and respond to messages using quantum computing principles.

## Features

- Real-time quantum-enhanced chat interface
- Advanced quantum state visualization
- Quantum-based confidence scoring
- IBM Quantum Experience integration
- Modern, responsive UI with Material-UI
- Secure API key management

## Prerequisites

- Docker and Docker Compose
 - IBM Quantum Experience API key
 - (Optional) OpenAI API key for enhanced language processing
- Node.js 16+ (for local development)
- Python 3.11 (for local development)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantum-ai-assistant.git
cd quantum-ai-assistant
```

2. Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

3. Edit the `.env` file and add your API keys:
```
# SECURITY WARNING: Never commit this file to version control
# SECURITY WARNING: Keep your API keys private and secure

# API key used by the backend for authentication
API_KEY=your_secure_api_key_here

# IBM Quantum Experience credentials
IBM_QUANTUM_API_KEY=your_ibm_quantum_api_key_here
IBM_QUANTUM_HUB=ibm-q
IBM_QUANTUM_GROUP=open
IBM_QUANTUM_PROJECT=main

# Optional OpenAI API key for enhanced language features
OPENAI_API_KEY=your_openai_api_key_here
```

4. Generate a secure API key:
```bash
openssl rand -hex 32
```

5. Add the generated key to your `.env` file.

## Running the Application

### Using Docker (Recommended)

1. Build and start the containers:
```bash
docker-compose up --build
```

2. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Local Development

#### Backend

1. Create a Python virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn main:app --reload
```

#### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
# Install TypeScript types for D3
npm install --save-dev @types/d3
```

2. Start the development server:
```bash
npm start
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and private
- Regularly rotate your API keys
- Monitor your IBM Quantum account for unusual activity
- The IBM Quantum API key and the OpenAI API key are used for different
  services and are not interchangeable. IBM credentials give you access to
  quantum hardware and simulators, while the optional OpenAI key enables
  language processing features.

## Using IBM Quantum and OpenAI Credentials

The IBM Quantum and OpenAI API keys **cannot be merged** into a single key. The
backend expects them as two distinct environment variables so that each service
can be used independently. Store them separately in your `.env` file:

```bash
IBM_QUANTUM_API_KEY=your_ibm_quantum_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # optional
```

The IBM key enables access to real quantum hardware or simulators, while the
OpenAI key is only used when language processing features are enabled.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 