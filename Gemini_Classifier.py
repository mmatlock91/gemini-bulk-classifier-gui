import sys
import os
import pandas as pd
import json
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QStatusBar, QGroupBox, QTextEdit, QPlainTextEdit, QProgressBar, QListWidget
)
from PySide6.QtGui import QFont
from PySide6.QtCore import QThread, QObject, Signal

# NOTE: Check the official Google AI Platform pricing for the model you use. These are illustrative.
INPUT_PRICE_PER_1K_TOKENS = 0.0001
OUTPUT_PRICE_PER_1K_TOKENS = 0.0004
CHARS_PER_TOKEN = 4.0  # A general approximation
BATCH_SIZE = 50

# A generic, example prompt. Users should customize this for their specific task.
DEFAULT_PROMPT = """You are an expert data analyst. Your task is to classify a data record into one, and only one, of the following predefined categories.

**Allowed Categories:**
[
  "Electronics",
  "Software",
  "Books",
  "Home Goods",
  "Clothing",
  "Groceries",
  "Services",
  "UNKNOWN"
]

Here are some guidelines for the categories:

- **Electronics**: Includes devices like phones, computers, and accessories.
- **Software**: Includes applications, operating systems, and digital licenses.
- **Services**: Includes consulting, subscriptions (non-software), and maintenance.
- **UNKNOWN**: Use this if the category cannot be determined from the provided data.

**Rules:**
1. You will be given multiple data records in a JSON list format.
2. Each record has one or more fields providing details about the item.
3. If the data looks garbled, unreadable, or is nonsense, classify it as "UNKNOWN".
4. Evaluate all the provided fields for each record to make the best classification.
5. Return a JSON array where each element corresponds to one record in order. The JSON object for each record should have a single key "category".
   Example: [{"category": "Software"}, {"category": "Electronics"}, ...]

"""

class CategorizationWorker(QObject):
    """
    Worker thread for processing data classification via the Gemini API.
    """
    finished = Signal(pd.DataFrame, int, int, bool)
    progress = Signal(int, int)
    log_message = Signal(str)

    def __init__(self, df, api_key, prompt_template, selected_columns, batch_size=BATCH_SIZE):
        super().__init__()
        self.df = df
        self.api_key = api_key
        self.prompt_template = prompt_template
        self.selected_columns = selected_columns
        self.batch_size = batch_size
        self.is_running = True

    def run(self):
        try:
            from google import genai
            from google.generativeai.client import Client
            from google.generativeai.types import GenerationConfig
            from google.api_core import exceptions as google_exceptions
        except ImportError:
            self.log_message.emit("ERROR: `google-generativeai` package not installed. Run: pip install google-generativeai")
            self.finished.emit(None, 0, 0, False)
            return

        try:
            client = Client(api_key=self.api_key)
        except Exception as e:
            self.log_message.emit(f"ERROR initializing Gemini client: {e}")
            self.finished.emit(None, 0, 0, False)
            return

        # Use a publicly available, recent model
        model_name = "gemini-1.5-flash-latest"
        
        total_rows = len(self.df)
        self.df["category"] = "NOT_PROCESSED"
        total_input_tokens = total_output_tokens = 0

        for start in range(0, total_rows, self.batch_size):
            if not self.is_running:
                break

            batch_df = self.df.iloc[start:start + self.batch_size]
            records = [
                {col: str(row.get(col, "")) for col in self.selected_columns}
                for _, row in batch_df.iterrows()
            ]
            batch_json = json.dumps(records, ensure_ascii=False)
            prompt = (
                self.prompt_template
                + f"\n\nClassify the following records:\n{batch_json}\n"
                "Return a JSON array of objects: [{'category': '...'}, ...]"
            )

            try:
                response = client.generate_content(
                    model=model_name,
                    contents=prompt,
                    generation_config=GenerationConfig(temperature=0.2)
                )
                
                content = response.text.strip()
                
                if response.usage_metadata:
                    usage = response.usage_metadata
                    total_input_tokens += usage.prompt_token_count
                    total_output_tokens += usage.candidates_token_count
                
                # Clean the response to ensure it's valid JSON
                cleaned = content.strip().replace("`", "")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:].strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()

                results = json.loads(cleaned)
                
                self.log_message.emit(f"--- Processing Batch {start} to {start + len(results) - 1} ---")

                for i, result in enumerate(results):
                    if start + i < total_rows:
                        category = result.get("category", "PARSE_ERROR")
                        self.df.at[start + i, "category"] = category
                        
                        record_row = batch_df.iloc[i]
                        # Dynamically display the first two selected columns in the log
                        val1 = record_row.get(self.selected_columns[0], "N/A") if len(self.selected_columns) > 0 else "N/A"
                        val2 = record_row.get(self.selected_columns[1], "N/A") if len(self.selected_columns) > 1 else ""
                        
                        log_entry = f"{val1} | {val2} -> {category}"
                        self.log_message.emit(log_entry)
                        
            except (google_exceptions.GoogleAPICallError, json.JSONDecodeError, Exception) as e:
                self.log_message.emit(f"API/JSON error at rows {start}-{start+self.batch_size}: {e}")
                time.sleep(1) # Wait a bit before retrying the next batch

            self.progress.emit(min(start + self.batch_size, total_rows), total_rows)
            time.sleep(0.1) # Small delay to keep UI responsive

        self.finished.emit(self.df, total_input_tokens, total_output_tokens, not self.is_running)

    def stop(self):
        self.is_running = False

class BulkClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bulk Data Classifier with Gemini API")
        self.setGeometry(100, 100, 800, 950)
        self.df, self.worker, self.thread = None, None, None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.create_widgets()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Load a CSV file to begin.")

    def create_widgets(self):
        # Setup monospaced font for logs and prompts
        log_font = QFont("Consolas", 11)
        if sys.platform == "darwin":
             log_font = QFont("Monaco", 11)
        elif sys.platform.startswith("linux"):
             log_font = QFont("Monospace", 11)
        
        # Step 1: Configuration Group
        config_group = QGroupBox("Step 1: Configuration")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Input CSV File:"), 0, 0)
        self.file_path_entry = QLineEdit()
        self.file_path_entry.setReadOnly(True)
        config_layout.addWidget(self.file_path_entry, 0, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.load_csv)
        config_layout.addWidget(browse_button, 0, 2)

        config_layout.addWidget(QLabel("Gemini API Key:"), 1, 0)
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setPlaceholderText("Enter your Google AI API Key")
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        config_layout.addWidget(self.api_key_entry, 1, 1, 1, 2)
        self.main_layout.addWidget(config_group)

        # Step 2: Column Selection
        column_group = QGroupBox("Step 2: Select Columns to Analyze")
        column_layout = QVBoxLayout(column_group)
        self.column_list_widget = QListWidget()
        self.column_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.column_list_widget.itemSelectionChanged.connect(self.update_estimated_cost)
        column_layout.addWidget(self.column_list_widget)
        self.main_layout.addWidget(column_group, stretch=1)

        # Step 3: Prompt Editor
        prompt_group = QGroupBox("Step 3: API Prompt (Customize for your task)")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = QPlainTextEdit(DEFAULT_PROMPT)
        self.prompt_edit.setFont(log_font)
        prompt_layout.addWidget(self.prompt_edit)
        self.main_layout.addWidget(prompt_group, stretch=2)

        # Step 4: Execution
        run_group = QGroupBox("Step 4: Execution")
        run_layout = QHBoxLayout(run_group)
        self.est_cost_label = QLabel("Estimated Cost: $0.00")
        self.run_button = QPushButton("Run Classification")
        self.run_button.setFixedHeight(40)
        self.run_button.clicked.connect(self.start_processing)
        self.run_button.setEnabled(False)
        run_layout.addWidget(self.est_cost_label, 1)
        run_layout.addWidget(self.run_button, 2)
        self.main_layout.addWidget(run_group)

        # Progress & Logs
        progress_group = QGroupBox("Progress & Logs")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(log_font)
        self.final_cost_label = QLabel("Final Cost: $0.00")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_area)
        progress_layout.addWidget(self.final_cost_label)
        self.main_layout.addWidget(progress_group, stretch=4)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if path:
            try:
                # Read CSV and sanitize column names for easier access
                df = pd.read_csv(path, low_memory=False)
                sanitized_columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
                df.columns = sanitized_columns
                self.df = df

                self.file_path_entry.setText(path)
                self.column_list_widget.clear()
                self.column_list_widget.addItems(self.df.columns)
                
                # Pre-select common column names for convenience
                common_cols = ["name", "description", "title", "details", "text"]
                for i in range(self.column_list_widget.count()):
                    item = self.column_list_widget.item(i)
                    if item.text() in common_cols:
                        item.setSelected(True)
                        
                self.log_area.clear()
                self.append_log(f"Loaded {len(self.df)} rows from {os.path.basename(path)}.")
                self.run_button.setEnabled(True)
                self.update_estimated_cost()
            except Exception as e:
                QMessageBox.critical(self, "Error Loading File", str(e))

    def update_estimated_cost(self):
        if self.df is None:
            return
        selected_items = self.column_list_widget.selectedItems()
        selected_columns = [item.text() for item in selected_items]
        if not selected_columns:
            self.est_cost_label.setText("Estimated Cost: $0.00 (No columns selected)")
            return
        
        # Estimate token count and cost
        num_rows = len(self.df)
        prompt_template_size = len(self.prompt_edit.toPlainText())
        # Calculate average row size based on selected columns
        avg_data_size = self.df[selected_columns].astype(str).apply(lambda x: x.str.len()).mean().sum()
        avg_output_size = 30 # Average size of a category response (e.g., {"category":"Electronics"})
        
        # Estimate total tokens
        est_input_tokens = num_rows * (prompt_template_size + avg_data_size) / CHARS_PER_TOKEN
        est_output_tokens = num_rows * avg_output_size / CHARS_PER_TOKEN
        
        total_est_cost = ((est_input_tokens / 1000) * INPUT_PRICE_PER_1K_TOKENS) + ((est_output_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS)
        self.est_cost_label.setText(f"Estimated Cost: ${total_est_cost:.4f}")

    def start_processing(self):
        api_key = self.api_key_entry.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API Key", "Please enter your Gemini API key in the configuration field (Step 1).")
            return
        
        selected_items = self.column_list_widget.selectedItems()
        selected_columns = [item.text() for item in selected_items]
        if not selected_columns:
            QMessageBox.warning(self, "No Columns Selected", "Please select at least one column to send to the API for analysis.")
            return

        # Change button to "Cancel" and connect to cancel method
        self.run_button.setText("Cancel")
        self.run_button.clicked.disconnect()
        self.run_button.clicked.connect(self.cancel_processing)
        self.progress_bar.setValue(0)
        self.log_area.clear()

        # Setup and start worker thread
        self.thread = QThread()
        self.worker = CategorizationWorker(self.df.copy(), api_key, self.prompt_edit.toPlainText(), selected_columns)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.processing_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.log_message.connect(self.append_log)
        self.thread.start()

    def cancel_processing(self):
        if self.worker:
            self.append_log("--- Cancelling... ---")
            self.worker.stop()

    def update_progress(self, processed, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(processed)
        self.statusBar().showMessage(f"Processing... {processed}/{total}")

    def append_log(self, message):
        self.log_area.append(message)
        self.log_area.ensureCursorVisible()

    def processing_finished(self, result_df, input_tokens, output_tokens, was_cancelled):
        self.statusBar().showMessage("Finished.")
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None

        # Reset run button
        self.run_button.setText("Run Classification")
        self.run_button.clicked.disconnect()
        self.run_button.clicked.connect(self.start_processing)
        
        if result_df is not None:
            # Calculate and display final cost
            final_input_cost = (input_tokens / 1000) * INPUT_PRICE_PER_1K_TOKENS
            final_output_cost = (output_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS
            total_final_cost = final_input_cost + final_output_cost
            self.final_cost_label.setText(f"Final Cost: ${total_final_cost:.4f} (Input: {input_tokens:,}, Output: {output_tokens:,})")
            
            # Save the results to a new CSV file
            try:
                base, _ = os.path.splitext(self.file_path_entry.text())
                output_path = f"{base}_classified.csv"
                result_df.to_csv(output_path, index=False)
                
                if was_cancelled:
                    QMessageBox.warning(self, "Process Cancelled", f"A partial file has been saved to:\n{output_path}")
                else:
                    QMessageBox.information(self, "Success", f"Classification complete. File saved to:\n{output_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save the output file.\nError: {e}")

if __name__ == "__main__":
    try:
        import pandas as pd
        import google.generativeai
    except ImportError as e:
        error_message = (
            f"Error: A required library is not installed: {e.name}\n\n"
            "Please install the necessary packages by running:\n"
            "pip install pandas PySide6 google-generativeai"
        )
        print(error_message)
        # Show a message box if GUI can be initialized
        app_temp = QApplication.instance() or QApplication(sys.argv)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(error_message)
        msg_box.setWindowTitle("Dependency Error")
        msg_box.exec()
        sys.exit(1)

    app = QApplication(sys.argv)
    window = BulkClassifierApp()
    window.show()
    sys.exit(app.exec())