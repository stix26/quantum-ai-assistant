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
- PostgreSQL 15 (local or Docker)
- IBM Quantum Experience API key
- Node.js 16+ (for local development)
- Python 3.11 (for local development and Docker builds)

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

# API Key for authentication (generate a new secure key)
API_KEY=your_secure_api_key_here
```

4. Generate a secure API key:
```bash
openssl rand -hex 32
```

5. Add the generated key to your `.env` file.

6. Ensure a PostgreSQL database is available. Docker Compose will automatically
   launch one, but for local development without Docker you'll need a running
   PostgreSQL 15 instance and to update `DATABASE_URL` in your `.env` file.

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
 - PostgreSQL: port 5432 on `db` service

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
4. Make sure PostgreSQL is running locally and that `DATABASE_URL` in your `.env`
   points to it.

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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 