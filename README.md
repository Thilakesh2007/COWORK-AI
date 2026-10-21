# COWORK AI 🤖

### Your personal AI coworker. Built locally. Designed to get work done.

COWORK AI is an evolving, locally run AI assistant built with Python, Ollama, and Gemma. The goal is to develop a desktop coworker that goes beyond answering questions: an AI that understands your context, remembers your preferences, researches information when necessary, and eventually helps carry out real tasks on your computer.

Inspired by the vision of AI-powered workspaces such as Claude Cowork, COWORK AI aims to combine conversational intelligence, persistent memory, practical tools, and desktop task automation in one application.

> **Project status:** Active development. The AI foundation is working; the graphical interface and advanced computer-work capabilities are planned.

## ✨ Features

### 🧠 Persistent Personal Memory
- Stores useful personal facts and preferences across sessions.
- Automatically detects supported personal information.
- Keeps personal memory separate from conversation history.
- Supports remembering, recalling, updating, and forgetting stored information.

### 💬 Conversation History
- Saves recent conversations locally.
- Loads conversation history across sessions.
- Keeps chat history separate from long-term personal memory.

### 🌐 Selective Web Research
- Searches the web when current or external information is useful.
- Avoids unnecessary searches for ordinary conversations.
- Uses search results to help answer research questions.

### 🛠️ AI Tool Integration
- Supports tools such as calculations, web research, and memory management.
- Uses a tool-enabled AI workflow to handle supported requests.

### ⚡ Local AI Inference
- Runs the language model locally through Ollama.
- Uses Gemma for natural-language understanding and responses.
- Includes response-time measurement to help evaluate performance.

### 🔒 Local-First Design
- Stores personal memory and conversation history in local JSON files.
- Excludes private memory files and virtual environments from Git using `.gitignore`.
- Aims to give users greater control over their data.

## 🎯 The Vision

COWORK AI is being developed toward a desktop AI coworker that can help users accomplish tasks rather than simply explain how to do them.

The long-term vision includes:

- **A modern desktop GUI:** A polished workspace with chat, conversation history, and task progress.
- **File and folder assistance:** Read, search, summarize, create, and organize files in user-approved locations.
- **Multi-step task execution:** Break requests into plans, use tools, perform actions, and verify results.
- **Computer interaction:** Progress toward assisting with applications and desktop workflows, with appropriate permissions and safeguards.
- **Context-aware assistance:** Use relevant personal memories and project context without repeatedly asking for known information.
- **Reliable task completion:** Handle errors, communicate progress, and ask for confirmation before sensitive or destructive actions.

These are development goals, not claims that every capability is already implemented.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and agent logic |
| Ollama | Local model runtime |
| Gemma | Language model |
| `ddgs` | Web search |
| JSON | Local memory and conversation storage |
| PySide6 *(planned)* | Native desktop GUI |

## 🚀 Getting Started

### Prerequisites

- Windows or another compatible Python environment
- Python
- [Ollama](https://ollama.com/)
- The Gemma model used by the project
- Git

### Installation

**1. Clone the repository**

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment on Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**4. Install the required Python packages**

Install the dependencies documented by the project. The dependency list and installation instructions will be finalized as the application evolves.

**5. Set up Ollama**

Install Ollama and download the model configured in `cowork.py`. For the current configuration, the model is:

```bash
ollama pull gemma4:12b-it-q4_K_M
```

Make sure Ollama is running before launching the application.

**6. Run COWORK AI**

```bash
python cowork.py
```

> Setup instructions may change as the project develops. Ensure your installed model name matches the configuration in `cowork.py`.

## 📁 Memory and Privacy

COWORK AI keeps two separate types of local data:

- `memory.json` — conversation history.
- `personal_memory.json` — persistent personal facts and preferences.

These files contain user-specific information and should not be committed to a public repository. Git ignores them through `.gitignore`; new installations should start with empty memory files.

**Privacy note:** Local execution does not automatically guarantee that all data stays local. Web searches send queries to an external search provider, and any future integrations may have their own data-handling requirements.

## 🗺️ Roadmap

- [x] Local language model integration
- [x] Persistent personal memory
- [x] Conversation history
- [x] Selective web research
- [x] Basic tool integration
- [x] Response-time measurement
- [ ] Modern desktop GUI
- [ ] Streaming responses in the GUI
- [ ] Conversation management through the GUI
- [ ] User-approved file and folder access
- [ ] File creation and organization tools
- [ ] Multi-step task planning and execution
- [ ] Desktop application interaction
- [ ] Better task progress reporting and error recovery
<img width="788" height="591" alt="image" src="https://github.com/user-attachments/assets/bf33a21f-eb10-42b1-a411-2f05c5efd206" />

## 🤝 Contributing

Ideas, bug reports, and suggestions are welcome. As the project grows, contributions can help improve its reliability, usability, privacy, and desktop capabilities.

## 📜 License

A license has not yet been specified. Until one is added, all rights remain with the copyright holder; do not assume the project is open for unrestricted reuse.

---

**COWORK AI — Moving from AI that answers to AI that works.** 💙

