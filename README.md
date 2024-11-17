# Generative AI-Based Cybersecurity for Connected Cars and IoT Systems

### Overview
This project leverages Generative AI (GANs) to simulate and prevent cybersecurity threats in connected cars and IoT-based automotive systems. The solution addresses the increasing risks of hacking, unauthorized access, and system vulnerabilities in the modern automotive and IoT ecosystem. By generating realistic cyberattack patterns and analyzing vulnerabilities, this project ensures proactive security measures and enhances user safety.

---

### Key Features
- Threat Simulation:  
   Generates simulated cyberattacks to test system vulnerabilities.  
- Proactive Security:  
   Identifies risks and provides insights to mitigate potential threats before they occur. 
- API Integration:  
   Flask-based REST API to interact with the GAN model for real-time inference.  
- Scalable Design:  
   Future-ready for real-time monitoring and adaptive security measures.

---

### Tech Stack
- Backend: Python, Flask, PyTorch, TorchMetrics  
- Frontend: React.js (Optional UI components for monitoring)
- Deployment: Waitress (production-ready server), Postman (API testing)

---

### Folder Structure
```
GENAI_PROJECT/
│
├── models/                    # Contains trained GAN model files
├── server/                    # Flask backend
│   ├── inference_api.py       # API for inference
│   ├── requirements.txt       # Python dependencies
│   └── server.py              # Server entry point
├── client/                    # Frontend (if implemented)
├── docs/                      # Documentation and assets (e.g., PPT, Video)
├── tests/                     # Unit tests for API endpoints
├── README.md                  # Project documentation
└── .gitignore                 # Git ignored files
```

---

### Setup Instructions

#### Prerequisites
1. Python 3.10 or above  
2. PyTorch installed (for the GAN model)  
3. Flask and Waitress for API hosting  
4. Postman (for testing API endpoints)  

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hritik664/genai.git
   cd genai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask server:
   ```bash
   python inference_api.py
   ```
4. For production:
   ```bash
   waitress-serve --port=5000 inference_api:app
   ```

#### Testing the API
- Use Postman to test API endpoints:
   - Endpoint: `http://127.0.0.1:5000/generate`
   - Method: `POST`
   - Body: JSON input, e.g., `{ "input_vector": [0.1, 0.2, 0.3, ...] }`

---

### Usage
1. Generate simulated threats using the GAN model via the `/generate` API.  
2. Visualize or log the generated data for security analysis.  
3. Leverage the output to enhance your system’s cybersecurity measures.

---

### Future Enhancements
- Real-time monitoring for connected cars and IoT systems.  
- Adaptive security protocols using reinforcement learning.  
- Integration with live automotive telemetry systems.  

---

### Contributors
- Hritik Singh

---

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Contact
For inquiries, please reach out to:  
Email: singhhritik560@gmail.com  
GitHub: [Hritik Singh](https://github.com/your-github-profile)
