import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit, QComboBox, QMessageBox,
                             QProgressBar, QFrame, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
                             QTabWidget, QTreeWidget, QTreeWidgetItem, QMenu, QAction)
from PyQt5.QtGui import QPixmap, QImage, QTextDocument, QTextCursor, QFont, QTextCharFormat, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import json
import numpy as np
from datetime import datetime

from database import FaceDatabase

class DatabaseManager(QMainWindow):
    """GUI for managing the face database"""

    def __init__(self):
        super().__init__()
        self.db = FaceDatabase()
        self.init_ui()

    def init_ui(self):
        """Initialize the database manager UI"""
        self.setWindowTitle('Face Database Manager')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget for different views
        self.tab_widget = QTabWidget()

        # Dashboard tab
        self.create_dashboard_tab()

        # Users tab
        self.create_users_tab()

        # Analyses tab
        self.create_analyses_tab()

        # Verification History tab
        self.create_verification_tab()

        # Search tab
        self.create_search_tab()

        # Settings tab
        self.create_settings_tab()

        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.statusBar().showMessage("Database Manager Ready")

        # Load initial data
        self.refresh_all_data()

    def create_dashboard_tab(self):
        """Create dashboard tab with overview statistics"""
        dashboard_widget = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_widget)

        # Statistics group
        stats_group = QGroupBox("Database Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        stats_layout.addWidget(self.stats_table)

        # Refresh button
        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self.refresh_stats)
        stats_layout.addWidget(refresh_btn)

        dashboard_layout.addWidget(stats_group)

        # Recent activity group
        activity_group = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout(activity_group)

        self.activity_table = QTableWidget()
        self.activity_table.setColumnCount(4)
        self.activity_table.setHorizontalHeaderLabels(["Type", "Description", "Date", "Details"])
        self.activity_table.horizontalHeader().setStretchLastSection(True)
        self.activity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.activity_table.setSelectionBehavior(QTableWidget.SelectRows)
        activity_layout.addWidget(self.activity_table)

        dashboard_layout.addWidget(activity_group)

        self.tab_widget.addTab(dashboard_widget, "📊 Dashboard")

    def create_users_tab(self):
        """Create users management tab"""
        users_widget = QWidget()
        users_layout = QVBoxLayout(users_widget)

        # Controls
        controls_layout = QHBoxLayout()

        add_user_btn = QPushButton("Add User")
        add_user_btn.clicked.connect(self.add_user_dialog)
        controls_layout.addWidget(add_user_btn)

        delete_user_btn = QPushButton("Delete User")
        delete_user_btn.clicked.connect(self.delete_user)
        controls_layout.addWidget(delete_user_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_users)
        controls_layout.addWidget(refresh_btn)

        controls_layout.addStretch()
        users_layout.addLayout(controls_layout)

        # Users table
        self.users_table = QTableWidget()
        self.users_table.setColumnCount(4)
        self.users_table.setHorizontalHeaderLabels(["ID", "Name", "Email", "Created"])
        self.users_table.horizontalHeader().setStretchLastSection(True)
        self.users_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.users_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.users_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.users_table.customContextMenuRequested.connect(self.show_user_context_menu)

        users_layout.addWidget(self.users_table)

        self.tab_widget.addTab(users_widget, "👥 Users")

    def create_analyses_tab(self):
        """Create analyses management tab"""
        analyses_widget = QWidget()
        analyses_layout = QVBoxLayout(analyses_widget)

        # Controls
        controls_layout = QHBoxLayout()

        view_analysis_btn = QPushButton("View Details")
        view_analysis_btn.clicked.connect(self.view_analysis_details)
        controls_layout.addWidget(view_analysis_btn)

        delete_analysis_btn = QPushButton("Delete Analysis")
        delete_analysis_btn.clicked.connect(self.delete_analysis)
        controls_layout.addWidget(delete_analysis_btn)

        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_analysis)
        controls_layout.addWidget(export_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_analyses)
        controls_layout.addWidget(refresh_btn)

        controls_layout.addStretch()
        analyses_layout.addLayout(controls_layout)

        # Analyses table
        self.analyses_table = QTableWidget()
        self.analyses_table.setColumnCount(7)
        self.analyses_table.setHorizontalHeaderLabels(["ID", "User", "Type", "Image", "Confidence", "Model", "Date"])
        self.analyses_table.horizontalHeader().setStretchLastSection(True)
        self.analyses_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.analyses_table.setSelectionBehavior(QTableWidget.SelectRows)

        analyses_layout.addWidget(self.analyses_table)

        self.tab_widget.addTab(analyses_widget, "🔍 Analyses")

    def create_verification_tab(self):
        """Create verification history tab"""
        verification_widget = QWidget()
        verification_layout = QVBoxLayout(verification_widget)

        # Controls
        controls_layout = QHBoxLayout()

        view_verification_btn = QPushButton("View Details")
        view_verification_btn.clicked.connect(self.view_verification_details)
        controls_layout.addWidget(view_verification_btn)

        delete_verification_btn = QPushButton("Delete Record")
        delete_verification_btn.clicked.connect(self.delete_verification)
        controls_layout.addWidget(delete_verification_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_verifications)
        controls_layout.addWidget(refresh_btn)

        controls_layout.addStretch()
        verification_layout.addLayout(controls_layout)

        # Verification table
        self.verification_table = QTableWidget()
        self.verification_table.setColumnCount(6)
        self.verification_table.setHorizontalHeaderLabels(["ID", "Image 1", "Image 2", "Similarity", "Verified", "Date"])
        self.verification_table.horizontalHeader().setStretchLastSection(True)
        self.verification_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.verification_table.setSelectionBehavior(QTableWidget.SelectRows)

        verification_layout.addWidget(self.verification_table)

        self.tab_widget.addTab(verification_widget, "⚖️ Verifications")

    def create_search_tab(self):
        """Create face search tab"""
        search_widget = QWidget()
        search_layout = QVBoxLayout(search_widget)

        # Search controls
        search_controls = QGroupBox("Search Similar Faces")
        search_controls_layout = QVBoxLayout(search_controls)

        # Image selection
        image_layout = QHBoxLayout()
        self.search_image_path = None

        select_image_btn = QPushButton("Select Face Image")
        select_image_btn.clicked.connect(self.select_search_image)
        image_layout.addWidget(select_image_btn)

        self.search_image_label = QLabel("No image selected")
        image_layout.addWidget(self.search_image_label)

        image_layout.addStretch()
        search_controls_layout.addLayout(image_layout)

        # Search parameters
        params_layout = QHBoxLayout()

        threshold_label = QLabel("Similarity Threshold:")
        self.threshold_input = QComboBox()
        self.threshold_input.addItems(["0.3", "0.4", "0.5", "0.6", "0.7", "0.8"])
        self.threshold_input.setCurrentText("0.6")
        params_layout.addWidget(threshold_label)
        params_layout.addWidget(self.threshold_input)

        max_results_label = QLabel("Max Results:")
        self.max_results_input = QComboBox()
        self.max_results_input.addItems(["5", "10", "20", "50", "100"])
        self.max_results_input.setCurrentText("10")
        params_layout.addWidget(max_results_label)
        params_layout.addWidget(self.max_results_input)

        search_btn = QPushButton("Search Similar Faces")
        search_btn.clicked.connect(self.search_similar_faces)
        params_layout.addWidget(search_btn)

        params_layout.addStretch()
        search_controls_layout.addLayout(params_layout)

        search_layout.addWidget(search_controls)

        # Search results
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout(results_group)

        self.search_results_table = QTableWidget()
        self.search_results_table.setColumnCount(5)
        self.search_results_table.setHorizontalHeaderLabels(["Rank", "Similarity", "Image", "Analysis Type", "Date"])
        self.search_results_table.horizontalHeader().setStretchLastSection(True)
        self.search_results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.search_results_table.setSelectionBehavior(QTableWidget.SelectRows)

        results_layout.addWidget(self.search_results_table)

        search_layout.addWidget(results_group)

        self.tab_widget.addTab(search_widget, "🔎 Search")

    def create_settings_tab(self):
        """Create settings and maintenance tab"""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        # Database info
        info_group = QGroupBox("Database Information")
        info_layout = QVBoxLayout(info_group)

        self.db_info_text = QTextEdit()
        self.db_info_text.setReadOnly(True)
        self.db_info_text.setMaximumHeight(150)
        info_layout.addWidget(self.db_info_text)

        settings_layout.addWidget(info_group)

        # Maintenance controls
        maintenance_group = QGroupBox("Database Maintenance")
        maintenance_layout = QVBoxLayout(maintenance_group)

        # Export controls
        export_layout = QHBoxLayout()

        export_btn = QPushButton("Export Database")
        export_btn.clicked.connect(self.export_database)
        export_layout.addWidget(export_btn)

        self.export_format = QComboBox()
        self.export_format.addItems(["All Data", "Users Only", "Analyses Only", "Verifications Only"])
        export_layout.addWidget(QLabel("Export:"))
        export_layout.addWidget(self.export_format)

        export_layout.addStretch()
        maintenance_layout.addLayout(export_layout)

        # Clear database controls
        clear_layout = QHBoxLayout()

        clear_db_btn = QPushButton("Clear All Data")
        clear_db_btn.setStyleSheet("QPushButton { background-color: #E74C3C; color: white; }")
        clear_db_btn.clicked.connect(self.clear_database_dialog)
        clear_layout.addWidget(clear_db_btn)

        optimize_btn = QPushButton("Optimize Database")
        optimize_btn.clicked.connect(self.optimize_database)
        clear_layout.addWidget(optimize_btn)

        clear_layout.addStretch()
        maintenance_layout.addLayout(clear_layout)

        settings_layout.addWidget(maintenance_group)

        self.tab_widget.addTab(settings_widget, "⚙️ Settings")

    # Data management methods
    def refresh_stats(self):
        """Refresh database statistics"""
        try:
            stats = self.db.get_database_stats()

            self.stats_table.setRowCount(0)

            row = 0
            for key, value in stats.items():
                self.stats_table.insertRow(row)
                self.stats_table.setItem(row, 0, QTableWidgetItem(key.replace('_', ' ').title()))
                self.stats_table.setItem(row, 1, QTableWidgetItem(f"{value:,}"))
                row += 1

            self.statusBar().showMessage("Statistics refreshed")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh statistics: {str(e)}")

    def refresh_users(self):
        """Refresh users table"""
        try:
            # This would require adding a method to get all users from the database
            # For now, show a placeholder
            self.users_table.setRowCount(0)
            self.statusBar().showMessage("Users refreshed")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh users: {str(e)}")

    def refresh_analyses(self):
        """Refresh analyses table"""
        try:
            # This would require adding a method to get all analyses from the database
            # For now, show a placeholder
            self.analyses_table.setRowCount(0)
            self.statusBar().showMessage("Analyses refreshed")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh analyses: {str(e)}")

    def refresh_verifications(self):
        """Refresh verification history table"""
        try:
            verifications = self.db.get_verification_history()

            self.verification_table.setRowCount(0)

            for row, verification in enumerate(verifications):
                self.verification_table.insertRow(row)
                self.verification_table.setItem(row, 0, QTableWidgetItem(str(verification['id'])))

                # Image paths (shortened)
                img1_path = verification.get('image1_path', 'Unknown')
                img2_path = verification.get('image2_path', 'Unknown')

                self.verification_table.setItem(row, 1, QTableWidgetItem(os.path.basename(img1_path)))
                self.verification_table.setItem(row, 2, QTableWidgetItem(os.path.basename(img2_path)))
                self.verification_table.setItem(row, 3, QTableWidgetItem(f"{verification['similarity_score']:.4f}"))
                self.verification_table.setItem(row, 4, QTableWidgetItem("✅" if verification['verified'] else "❌"))
                self.verification_table.setItem(row, 5, QTableWidgetItem(verification['created_at']))

            self.statusBar().showMessage(f"Loaded {len(verifications)} verification records")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh verifications: {str(e)}")

    def refresh_all_data(self):
        """Refresh all data displays"""
        self.refresh_stats()
        self.refresh_users()
        self.refresh_analyses()
        self.refresh_verifications()

    # Dialog methods
    def add_user_dialog(self):
        """Show dialog to add new user"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New User")
        dialog.setModal(True)

        layout = QFormLayout(dialog)

        name_input = QLineEdit()
        email_input = QLineEdit()

        layout.addRow("Name:", name_input)
        layout.addRow("Email:", email_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() == QDialog.Accepted:
            try:
                user_id = self.db.add_user(name_input.text(), email_input.text() or None)
                self.statusBar().showMessage(f"User added with ID: {user_id}")
                self.refresh_users()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to add user: {str(e)}")

    def export_database(self):
        """Export database to JSON file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Database", "", "JSON Files (*.json)"
            )

            if file_path:
                data_type = self.export_format.currentText().lower().replace(" ", "_")
                if data_type == "all_data":
                    data_type = "all"

                exported_path = self.db.export_data(file_path, data_type)
                QMessageBox.information(self, "Success",
                                      f"Database exported successfully to:\n{exported_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export database: {str(e)}")

    def clear_database_dialog(self):
        """Show confirmation dialog for clearing database"""
        reply = QMessageBox.question(
            self, "Clear Database",
            "Are you sure you want to clear all data? This action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.db.clear_database(confirm=True)
                self.refresh_all_data()
                QMessageBox.information(self, "Success", "Database cleared successfully")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to clear database: {str(e)}")

    def optimize_database(self):
        """Optimize database (VACUUM)"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("VACUUM")
                conn.commit()

            self.refresh_stats()
            QMessageBox.information(self, "Success", "Database optimized successfully")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to optimize database: {str(e)}")

    # Placeholder methods for other functionality
    def delete_user(self):
        QMessageBox.information(self, "Info", "Delete user functionality - Coming soon!")

    def view_analysis_details(self):
        QMessageBox.information(self, "Info", "View analysis details - Coming soon!")

    def delete_analysis(self):
        QMessageBox.information(self, "Info", "Delete analysis - Coming soon!")

    def export_analysis(self):
        QMessageBox.information(self, "Info", "Export analysis - Coming soon!")

    def view_verification_details(self):
        QMessageBox.information(self, "Info", "View verification details - Coming soon!")

    def delete_verification(self):
        QMessageBox.information(self, "Info", "Delete verification - Coming soon!")

    def select_search_image(self):
        QMessageBox.information(self, "Info", "Select search image - Coming soon!")

    def search_similar_faces(self):
        QMessageBox.information(self, "Info", "Search similar faces - Coming soon!")

    def show_user_context_menu(self, position):
        """Show context menu for user table"""
        menu = QMenu()
        view_action = menu.addAction("View User Details")
        edit_action = menu.addAction("Edit User")
        delete_action = menu.addAction("Delete User")

        action = menu.exec_(self.users_table.mapToGlobal(position))
        if action == delete_action:
            self.delete_user()
