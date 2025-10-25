# gemini-bulk-classifier-gui
A Python desktop application for bulk-classifying CSV data using the Google Gemini API, featuring asynchronous processing and real-time cost estimation.

# Gemini Bulk Classifier GUI
A user-friendly desktop application for bulk-classifying CSV data using the Google Gemini API, featuring asynchronous processing and real-time cost estimation. This tool is designed to empower non-technical users to leverage the power of large language models for their data analysis needs.

## The Problem
Data analysts and business teams often face the challenge of categorizing thousands of rows of messy, unstructured text data. This process is typically manual, slow, and prone to error, or requires writing complex, one-off scripts that are not reusable or accessible to non-coders.

## The Solution
This application provides a simple, powerful, and self-contained GUI that allows any user to:
- Load any CSV file.
- Intuitively select the data columns they want to analyze.
- Write and refine a custom, natural-language prompt to define their exact classification needs.
- Run the classification in bulk against the Gemini API with a single click.
- Monitor the progress in real-time and receive a final, classified CSV file.

## Key Features
- **Intuitive GUI:** A clean, step-by-step interface that requires no coding knowledge.
- **Asynchronous Processing:** The application remains fully responsive while processing thousands of API calls in the background, thanks to a multi-threaded architecture.
- **Real-Time Cost Management:** Provides an upfront cost estimate before running and a final, detailed cost report after, translating API usage into direct business expense.
- **Robust Error Handling:** Gracefully manages API errors and logs all progress in real-time for full transparency.
- **Advanced Prompt Engineering:** The flexible prompt window allows for sophisticated, expert-level prompt design directly in the UI.

## Core Architectural Concepts
This application was architected with two key principles in mind:
1.  **User Experience:** An asynchronous design using `QThread` was chosen to ensure the UI never freezes during long-running network operations, providing a smooth and professional user experience.
2.  **Efficiency & Transparency:** The system uses a batching method to process API calls efficiently, and includes a real-time cost calculation module to make the financial impact of the work transparent to the user.

## How to Use
1.  Clone the repository: `git clone https://github.com/your-username/gemini-bulk-classifier-gui.git`
2.  Install the required packages: `pip install pandas PySide6 google-generativeai`
3.  Run the main Python script.
4.  Add your Google AI API key, load a CSV, and customize the prompt for your task.

---
*This project was developed as a personal portfolio piece and is not affiliated with any past or present employer.* 
