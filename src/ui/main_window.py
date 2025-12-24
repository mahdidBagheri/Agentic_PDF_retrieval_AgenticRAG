import sys
# 1. PyQt imports MUST come first
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QFileDialog, QListWidget,
    QTabWidget, QProgressBar
)
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# 2. Then qt_material
from qt_material import apply_stylesheet

# 3. Then your app logic
from src.app_controller import AppController

class WorkerSignals(QObject):
    update_chat = pyqtSignal(str, str)  # For Chat messages
    update_status = pyqtSignal(str)  # For Status bar
    upload_finished = pyqtSignal(str)  # ✅ NEW: For Upload completion


class RAGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = AppController()
        self.signals = WorkerSignals()

        # --- Connect Signals to UI Updates ---
        self.signals.update_chat.connect(self.append_message)
        self.signals.update_status.connect(self.update_status_label)

        # ✅ Connect the new upload signal to the UI updater
        self.signals.upload_finished.connect(self.finalize_upload_ui)

        self.setWindowTitle("RAG Knowledge Assistant")
        self.setGeometry(100, 100, 900, 700)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_chat_tab(), "💬 Chat")
        tabs.addTab(self.create_kb_tab(), "📂 Knowledge Base")
        layout.addWidget(tabs)

        # Status Bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def create_chat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about your documents...")
        self.input_field.returnPressed.connect(self.handle_send)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.handle_send)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(send_btn)
        layout.addLayout(input_layout)

        return tab

    def create_kb_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("<h2>Upload Documents</h2>"))
        layout.addWidget(QLabel("Add PDFs to the knowledge base here."))

        upload_btn = QPushButton("Select PDF")
        upload_btn.clicked.connect(self.handle_upload)
        layout.addWidget(upload_btn)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_display = QListWidget()
        layout.addWidget(self.log_display)

        return tab

    # --- Logic ---

    def handle_send(self):
        query = self.input_field.text().strip()
        if not query:
            return

        self.append_message("You", query)
        self.input_field.clear()
        self.status_label.setText("Thinking...")

        self.controller.submit_query(query, self.on_query_complete)

    def on_query_complete(self, answer):
        # Background thread calls this -> Emits signal -> Main thread updates UI
        self.signals.update_chat.emit("Bot", answer)
        self.signals.update_status.emit("Ready")

    def append_message(self, sender, text):
        color = "#4caf50" if sender == "Bot" else "#2196f3"
        formatted = f'<div style="margin-bottom: 10px;"><b><span style="color:{color};">{sender}:</span></b><br>{text}</div>'
        self.chat_display.append(formatted)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_display.setTextCursor(cursor)

    def update_status_label(self, text):
        self.status_label.setText(text)

    def handle_upload(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open PDF', '', "PDF Files (*.pdf)")
        if fname:
            self.log_display.addItem(f"Uploading: {fname}...")
            self.progress.setValue(20)
            self.status_label.setText("Ingesting...")
            # Start background task
            self.controller.upload_file(fname, self.on_upload_complete)

    def on_upload_complete(self, message):
        """
        Runs on Background Thread.
        CANNOT touch UI directly. Must emit signal.
        """
        self.signals.upload_finished.emit(message)
        self.signals.update_status.emit("Ready")

    def finalize_upload_ui(self, message):
        """
        Runs on Main UI Thread.
        Safe to update widgets.
        """
        self.log_display.addItem(message)
        self.progress.setValue(100)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Theme applied cleanly
    apply_stylesheet(app, theme='dark_teal.xml')

    window = RAGApp()
    window.show()
    sys.exit(app.exec_())