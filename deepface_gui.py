#!/usr/bin/env python3

# Suppress ALL TensorFlow warnings BEFORE any imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Try to import Google Lens, but make it optional
try:
    print("Attempting to import googlelens...")
    import sys
    print("Python path:", sys.path)
    from googlelens import GoogleLens
    print("Successfully imported googlelens")
    HAS_GOOGLE_LENS = True
except ImportError as e:
    HAS_GOOGLE_LENS = False
    print(f"Error importing googlelens: {e}")
    print("Note: Google Lens module not found. Google Lens search will be disabled.")
    print("To enable it, install with: pip install git+https://github.com/krishna2206/google-lens-python.git")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Additional environment variables for specific warnings
os.environ['TF_LOGGING_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('keras').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

# Set TensorFlow logging to most restrictive level
os.environ['TF_LOGGING_LEVEL'] = '3'

# Also suppress specific TensorFlow warnings
logging.getLogger('tensorflow.core.util.port').setLevel(logging.FATAL)

import sys
import contextlib
import io
import json
import numpy as np
from datetime import datetime
import webbrowser
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Import PyQt5 components
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog, QTextEdit, QComboBox, QMessageBox,
                            QProgressBar, QFrame, QGroupBox, QSplitter, QInputDialog,
                            QTabWidget, QTableWidget, QTableWidgetItem, QDialog,
                            QScrollArea, QGridLayout, QAction, QLineEdit, QToolBar)
from PyQt5.QtGui import QPixmap, QFont, QTextCharFormat, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl

# Try to import WebEngine for in-app browsing
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
    from PyQt5.QtWebEngine import QtWebEngine
    HAS_WEBENGINE = True
    print("✅ WebEngine available for in-app browsing")
except ImportError:
    HAS_WEBENGINE = False
    print("⚠️ WebEngine not available - install with: pip install PyQtWebEngine")
    # Fallback imports if WebEngine not available
    QWebEngineView = None
    QWebEnginePage = None
    QtWebEngine = None

# Import TensorFlow and DeepFace after setting environment variables
# Suppress TensorFlow logging after import
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('tensorflow').propagate = False

# Import DeepFace (warnings will be minimal due to environment variables)
from deepface import DeepFace

# Import database functionality
from database import FaceDatabase

print("DeepFace GUI starting without warnings...")

class DeepFaceGUI(QMainWindow):
    def is_widget_valid(self, widget):
        """Safely check if a Qt widget is still valid and not deleted"""
        try:
            # Check if the widget exists and is not being destroyed
            if widget is None or not hasattr(widget, 'isVisible'):
                return False
            # Try to access a property to see if the C++ object is still valid
            widget.isVisible()
            return True
        except RuntimeError as e:
            if 'wrapped C/C++ object' in str(e):
                return False
            return True  # Other errors might be recoverable
        except Exception:
            return False

    class SearchWorker(QThread):
        finished = pyqtSignal(dict)
        
        def __init__(self, image_path, timeout_seconds=60):
            super().__init__()
            self.image_path = image_path
            self.timeout_seconds = timeout_seconds
            
        def run(self):
            try:
                print("🔍 SearchWorker: Starting GoogleLens search...")
                print(f"Image path: {self.image_path}")

                # Check if image file exists
                if not os.path.exists(self.image_path):
                    raise Exception(f"Image file not found: {self.image_path}")

                # Create GoogleLens instance with error handling
                try:
                    gl = GoogleLens()
                    print("✅ GoogleLens instance created successfully")

                    # Check available methods
                    available_methods = [method for method in dir(gl) if not method.startswith('_') and callable(getattr(gl, method))]
                    print(f"Available methods: {available_methods}")

                except Exception as e:
                    print(f"❌ Failed to create GoogleLens instance: {e}")
                    raise Exception(f"GoogleLens initialization failed: {e}")

                # Try to use GoogleLens with correct method names
                results = []

                # Try search_by_file method (correct API)
                if hasattr(gl, 'search_by_file'):
                    print("🔄 Trying search_by_file method...")
                    try:
                        results = gl.search_by_file(self.image_path)
                        print(f"✅ search_by_file method succeeded, got {len(results)} results")
                    except Exception as e:
                        print(f"❌ search_by_file method failed: {e}")
                        # Try search_by_url as fallback (less likely to work without URL)
                        if hasattr(gl, 'search_by_url'):
                            print("🔄 Trying search_by_url method as fallback...")
                            try:
                                # Convert file path to file URL
                                from urllib.parse import urljoin
                                from urllib.request import pathname2url
                                file_url = urljoin('file:', pathname2url(os.path.abspath(self.image_path)))
                                results = gl.search_by_url(file_url)
                                print(f"✅ search_by_url method succeeded, got {len(results)} results")
                            except Exception as e2:
                                print(f"❌ search_by_url method also failed: {e2}")
                                raise Exception(f"Both search_by_file and search_by_url methods failed: {e2}")
                        else:
                            raise Exception(f"search_by_file method failed and no search_by_url method available: {e}")
                elif hasattr(gl, 'search_by_url'):
                    print("🔄 Trying search_by_url method...")
                    try:
                        # Convert file path to file URL
                        from urllib.parse import urljoin
                        from urllib.request import pathname2url
                        file_url = urljoin('file:', pathname2url(os.path.abspath(self.image_path)))
                        results = gl.search_by_url(file_url)
                        print(f"✅ search_by_url method succeeded, got {len(results)} results")
                    except Exception as e:
                        print(f"❌ search_by_url method failed: {e}")
                        raise Exception(f"search_by_url method failed: {e}")
                else:
                    # No usable methods found
                    available_methods = [method for method in dir(gl) if not method.startswith('_')]
                    raise Exception(f"No usable methods found. Available methods: {available_methods}. Expected: search_by_file or search_by_url")

                # Validate results
                if not results:
                    print("⚠️ GoogleLens returned empty results")
                    results = []

                print(f"📊 Final results: {len(results)} items")
                self.finished.emit({"success": True, "data": results})

            except Exception as e:
                error_msg = f"GoogleLens search failed: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                self.finished.emit({"success": False, "error": error_msg})

    def __init__(self):
        super().__init__()
        self.setWindowTitle('DeepFace GUI - Enhanced Face Analysis with Database')
        self.setGeometry(100, 100, 1200, 800)

        # Initialize database
        self.database = FaceDatabase()
        self.current_user_id = None

        # Variables for images
        self.img1_path = None
        self.img2_path = None
        self.internet_image_path = None

        # Setup central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget for main interface
        self.main_tab_widget = QTabWidget()

        # Analysis tab (original functionality)
        self.create_analysis_tab()

        # Database tab (integrated database management)
        self.create_database_tab()

        # Search tab (face comparison with database)
        self.create_search_tab()

        main_layout.addWidget(self.main_tab_widget)

        # Setup menu bar
        self.setup_menu_bar()

        # Set initial state
        self.clear_image_display()
        self.switch_to_single_display()
        self.statusBar().showMessage("Ready - Load images and select a function to begin")

    def setup_menu_bar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        # User management
        set_user_action = QAction('Set Current User', self)
        set_user_action.triggered.connect(self.set_current_user)
        file_menu.addAction(set_user_action)

        file_menu.addSeparator()

        # Database management
        db_manager_action = QAction('Open Database Manager', self)
        db_manager_action.triggered.connect(self.open_database_manager)
        file_menu.addAction(db_manager_action)

        export_action = QAction('Export Results', self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_analysis_tab(self):
        """Create the analysis tab with image controls and results"""
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)

        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)

        # Top section - Controls and images
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Image section with group box
        image_group = QGroupBox("Images")
        image_layout = QVBoxLayout(image_group)

        # Image 1 selection
        self.img1_label = QLabel('Image 1: None')
        self.img1_label.setStyleSheet("font-weight: bold; color: #2E86AB; padding: 5px;")
        image_layout.addWidget(self.img1_label)
        load_img1_btn = QPushButton('Load Image 1')
        load_img1_btn.setStyleSheet("QPushButton { background-color: #A8DADC; color: black; padding: 8px; border-radius: 5px; }")
        load_img1_btn.clicked.connect(self.load_image1)
        image_layout.addWidget(load_img1_btn)

        # Image 2 selection (for verification)
        self.img2_label = QLabel('Image 2: None')
        self.img2_label.setStyleSheet("font-weight: bold; color: #2E86AB; padding: 5px;")
        image_layout.addWidget(self.img2_label)
        load_img2_btn = QPushButton('Load Image 2')
        load_img2_btn.setStyleSheet("QPushButton { background-color: #A8DADC; color: black; padding: 8px; border-radius: 5px; }")
        load_img2_btn.clicked.connect(self.load_image2)
        image_layout.addWidget(load_img2_btn)

        # Function selection
        function_layout = QHBoxLayout()
        function_layout.addWidget(QLabel('Select Function:'))
        self.function_combo = QComboBox()
        self.function_combo.addItems(['Verify Faces', 'Analyze Face'])
        self.function_combo.setStyleSheet("QComboBox { padding: 5px; border-radius: 3px; }")
        self.function_combo.currentTextChanged.connect(self.on_function_changed)
        function_layout.addWidget(self.function_combo)
        function_layout.addStretch()
        image_layout.addLayout(function_layout)

        # Execute button and progress
        execute_layout = QHBoxLayout()
        self.execute_btn = QPushButton('Execute Analysis')
        self.execute_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 10px 15px;
                border-radius: 5px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:disabled {
                background-color: #95A5A6;
                color: #BDC3C7;
            }
        """)
        self.execute_btn.clicked.connect(self.execute_function)
        execute_layout.addWidget(self.execute_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2E86AB;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                color: #2E86AB;
                background-color: #F8F9FA;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #A8DADC, stop: 1 #2E86AB);
                border-radius: 8px;
            }
        """)
        execute_layout.addWidget(self.progress_bar)

        execute_layout.addStretch()
        image_layout.addLayout(execute_layout)

        top_layout.addWidget(image_group)

        # Image display area with frame
        display_frame = QFrame()
        display_frame.setFrameStyle(QFrame.Box)
        display_frame.setStyleSheet("QFrame { border: 2px solid #2E86AB; border-radius: 5px; }")
        display_layout = QVBoxLayout(display_frame)

        # Single image display for analysis
        self.single_image_widget = QWidget()
        single_layout = QVBoxLayout(self.single_image_widget)
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("QLabel { background-color: #F8F9FA; }")
        self.image_display.setMinimumHeight(300)
        single_layout.addWidget(QLabel("Selected Image:"))
        single_layout.addWidget(self.image_display)

        # Dual image display for verification
        self.dual_image_widget = QWidget()
        dual_layout = QHBoxLayout(self.dual_image_widget)
        dual_layout.setSpacing(10)

        # Image 1 display
        self.img1_display = QLabel()
        self.img1_display.setAlignment(Qt.AlignCenter)
        self.img1_display.setStyleSheet("QLabel { background-color: #F8F9FA; border: 1px solid #BDC3C7; }")
        self.img1_display.setMinimumSize(200, 200)
        self.img1_display.setMaximumSize(300, 300)
        img1_layout = QVBoxLayout()
        img1_layout.addWidget(QLabel("Image 1:"))
        img1_layout.addWidget(self.img1_display)
        dual_layout.addLayout(img1_layout)

        # VS label
        vs_label = QLabel("VS")
        vs_label.setAlignment(Qt.AlignCenter)
        vs_label.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; color: #E74C3C; }")
        dual_layout.addWidget(vs_label)

        # Image 2 display
        self.img2_display = QLabel()
        self.img2_display.setAlignment(Qt.AlignCenter)
        self.img2_display.setStyleSheet("QLabel { background-color: #F8F9FA; border: 1px solid #BDC3C7; }")
        self.img2_display.setMinimumSize(200, 200)
        self.img2_display.setMaximumSize(300, 300)
        img2_layout = QVBoxLayout()
        img2_layout.addWidget(QLabel("Image 2:"))
        img2_layout.addWidget(self.img2_display)
        dual_layout.addLayout(img2_layout)

        # Add both widgets to main layout, initially show single image
        display_layout.addWidget(self.single_image_widget)
        display_layout.addWidget(self.dual_image_widget)
        self.dual_image_widget.setVisible(False)

        top_layout.addWidget(display_frame)
        splitter.addWidget(top_widget)

        # Bottom section - Results
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)

        # Results section with group box
        results_group = QGroupBox("Analysis Results")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")
        results_layout = QVBoxLayout(results_group)

        # Enhanced result display
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #F8F9FA;
                border: 2px solid #2E86AB;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        self.result_text.setMinimumHeight(200)
        results_layout.addWidget(QLabel("Results:"))
        results_layout.addWidget(self.result_text)

        # Clear results button
        clear_btn = QPushButton('Clear All')
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
        """)
        clear_btn.clicked.connect(self.clear_all)
        results_layout.addWidget(clear_btn)

        bottom_layout.addWidget(results_group)
        splitter.addWidget(bottom_widget)

        # Set splitter proportions
        splitter.setSizes([500, 300])

        analysis_layout.addWidget(splitter)

        self.main_tab_widget.addTab(analysis_widget, "🔍 Analysis")

    def create_database_tab(self):
        """Create the database management tab"""
        database_widget = QWidget()
        database_layout = QVBoxLayout(database_widget)

        # Database info section
        info_group = QGroupBox("Database Information")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")
        info_layout = QVBoxLayout(info_group)

        # Database statistics
        self.db_info_text = QTextEdit()
        self.db_info_text.setReadOnly(True)
        self.db_info_text.setMaximumHeight(150)
        self.db_info_text.setStyleSheet("""
            QTextEdit {
                background-color: #F8F9FA;
                border: 1px solid #2E86AB;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        info_layout.addWidget(QLabel("Statistics:"))
        info_layout.addWidget(self.db_info_text)

        # Refresh buttons
        buttons_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh All")
        refresh_btn.clicked.connect(self.refresh_database_stats)
        refresh_btn.clicked.connect(self.refresh_recent_analyses)
        refresh_btn.clicked.connect(self.refresh_image_gallery)
        buttons_layout.addWidget(refresh_btn)

        export_db_btn = QPushButton("Export Database")
        export_db_btn.clicked.connect(self.export_database)
        buttons_layout.addWidget(export_db_btn)

        buttons_layout.addStretch()
        info_layout.addLayout(buttons_layout)

        database_layout.addWidget(info_group)

        # Recent analyses section
        recent_group = QGroupBox("Recent Analyses")
        recent_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")
        recent_layout = QVBoxLayout(recent_group)

        self.recent_analyses_table = QTableWidget()
        self.recent_analyses_table.setColumnCount(5)
        self.recent_analyses_table.setHorizontalHeaderLabels(["ID", "User", "Type", "Image", "Date"])
        self.recent_analyses_table.horizontalHeader().setStretchLastSection(True)
        self.recent_analyses_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recent_analyses_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.recent_analyses_table.setMaximumHeight(200)
        recent_layout.addWidget(self.recent_analyses_table)

        # Refresh recent analyses button
        refresh_recent_btn = QPushButton("Refresh Recent Analyses")
        refresh_recent_btn.clicked.connect(self.refresh_recent_analyses)
        recent_layout.addWidget(refresh_recent_btn)

        database_layout.addWidget(recent_group)

        # Database management section
        management_group = QGroupBox("Database Management")
        management_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")
        management_layout = QVBoxLayout(management_group)

        # Quick user management
        user_layout = QHBoxLayout()

        set_user_btn = QPushButton("Set Current User")
        set_user_btn.clicked.connect(self.set_current_user)
        user_layout.addWidget(set_user_btn)

        self.current_user_label = QLabel("Current User: Anonymous")
        user_layout.addWidget(self.current_user_label)

        user_layout.addStretch()
        management_layout.addLayout(user_layout)

        # Database operations
        db_buttons_layout = QHBoxLayout()

        clear_db_btn = QPushButton("Clear Database")
        clear_db_btn.setStyleSheet("QPushButton { background-color: #E74C3C; color: white; }")
        clear_db_btn.clicked.connect(self.clear_database_dialog)
        db_buttons_layout.addWidget(clear_db_btn)

        optimize_btn = QPushButton("Optimize Database")
        optimize_btn.clicked.connect(self.optimize_database)
        db_buttons_layout.addWidget(optimize_btn)

        db_buttons_layout.addStretch()
        management_layout.addLayout(db_buttons_layout)

        database_layout.addWidget(management_group)

        # Image gallery section
        gallery_group = QGroupBox("📸 Stored Face Images")
        gallery_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")
        gallery_layout = QVBoxLayout(gallery_group)

        # Gallery controls
        gallery_controls = QHBoxLayout()

        self.refresh_gallery_btn = QPushButton("🔄 Refresh Gallery")
        self.refresh_gallery_btn.clicked.connect(self.refresh_image_gallery)
        gallery_controls.addWidget(self.refresh_gallery_btn)

        self.gallery_filter_combo = QComboBox()
        self.gallery_filter_combo.addItems(["All Images", "By User", "By Date", "By Analysis Type"])
        self.gallery_filter_combo.currentTextChanged.connect(self.refresh_image_gallery)
        gallery_controls.addWidget(QLabel("Filter:"))
        gallery_controls.addWidget(self.gallery_filter_combo)

        gallery_controls.addStretch()
        gallery_layout.addLayout(gallery_controls)

        # Image gallery display
        self.image_gallery_widget = QWidget()
        self.image_gallery_layout = QVBoxLayout(self.image_gallery_widget)

        # Scrollable area for images
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.gallery_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
            }
        """)

        self.image_grid_widget = QWidget()
        self.image_grid_layout = QGridLayout(self.image_grid_widget)
        self.image_grid_layout.setSpacing(10)

        self.gallery_scroll.setWidget(self.image_grid_widget)
        self.image_gallery_layout.addWidget(self.gallery_scroll)

        # Gallery info
        self.gallery_info_label = QLabel("Click 'Refresh Gallery' to load stored images")
        self.gallery_info_label.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; padding: 10px; }")
        self.image_gallery_layout.addWidget(self.gallery_info_label)

        gallery_layout.addWidget(self.image_gallery_widget)

        database_layout.addWidget(gallery_group)

        self.main_tab_widget.addTab(database_widget, "🗄️ Database")

        # Load initial database information
        self.refresh_database_stats()
        self.refresh_recent_analyses()
        self.refresh_image_gallery()

    def refresh_image_gallery(self):
        """Refresh the image gallery display"""
        try:
            self.gallery_info_label.setText("🔄 Loading stored images...")
            self.gallery_info_label.setStyleSheet("QLabel { color: #E74C3C; font-weight: bold; }")

            # Clear existing gallery
            for i in reversed(range(self.image_grid_layout.count())):
                widget = self.image_grid_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            # Get stored analyses with images
            analyses = self.database.get_analyses_with_images(100)

            if not analyses:
                self.gallery_info_label.setText("No face images found in database. Analyze some images first!")
                self.gallery_info_label.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; }")
                return

            # Display images in a grid
            images_per_row = 4
            for i, analysis in enumerate(analyses):
                row = i // images_per_row
                col = i % images_per_row

                # Create image widget
                image_widget = self.create_image_card(analysis)
                self.image_grid_layout.addWidget(image_widget, row, col)

            # Update gallery info
            self.gallery_info_label.setText(f"📸 Displaying {len(analyses)} stored face images")
            self.gallery_info_label.setStyleSheet("QLabel { color: #27AE60; font-weight: bold; }")

        except Exception as e:
            self.gallery_info_label.setText(f"❌ Error loading gallery: {str(e)}")
            self.gallery_info_label.setStyleSheet("QLabel { color: #E74C3C; font-weight: bold; }")
            print(f"Error refreshing image gallery: {e}")

    def create_image_card(self, analysis):
        """Create a card widget for displaying a stored image"""
        # Main card widget
        card = QFrame()
        card.setFrameStyle(QFrame.Box)
        card.setStyleSheet("""
            QFrame {
                border: 2px solid #BDC3C7;
                border-radius: 8px;
                background-color: #F8F9FA;
            }
            QFrame:hover {
                border: 2px solid #2E86AB;
                background-color: #E8F4F8;
            }
        """)
        card.setFixedSize(200, 280)
        card.setCursor(Qt.PointingHandCursor)

        # Layout for card content
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(5)
        card_layout.setContentsMargins(5, 5, 5, 5)

        # Image display
        image_path = analysis.get('image_path', '')
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(180, 180)
        image_label.setMaximumSize(180, 180)
        image_label.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #BDC3C7;
                border-radius: 5px;
            }
        """)

        # Load and display image
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(170, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("❌\nInvalid\nImage")
                image_label.setStyleSheet("""
                    QLabel {
                        background-color: #FFE6E6;
                        border: 1px solid #E74C3C;
                        border-radius: 5px;
                        color: #E74C3C;
                        font-size: 12px;
                        text-align: center;
                    }
                """)
        except Exception as e:
            image_label.setText("⚠️\nImage\nError")
            image_label.setStyleSheet("""
                QLabel {
                    background-color: #FFF5E6;
                    border: 1px solid #F39C12;
                    border-radius: 5px;
                    color: #F39C12;
                    font-size: 12px;
                    text-align: center;
                }
            """)

        card_layout.addWidget(image_label)

        # Image info
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(70)
        info_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                font-size: 10px;
                color: #2C3E50;
            }
        """)

        # Format analysis info
        user_name = analysis.get('user_name', 'Anonymous')
        analysis_type = analysis.get('analysis_type', 'Unknown')
        created_at = analysis.get('created_at', '')

        if created_at:
            try:
                date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_at = date_obj.strftime('%Y-%m-%d %H:%M')
            except:
                pass

        info_text.append(f"👤 User: {user_name}")
        info_text.append(f"🔧 Type: {analysis_type.title()}")
        info_text.append(f"📅 Date: {created_at}")

        # Add result summary if available
        result_data = analysis.get('result_data', {})
        if isinstance(result_data, dict):
            if 'age' in result_data:
                info_text.append(f"🎂 Age: {result_data['age']}")
            if 'gender' in result_data:
                info_text.append(f"⚧️ Gender: {result_data['gender']}")

        card_layout.addWidget(info_text)

        # Click handler
        def on_card_clicked():
            self.show_image_details(analysis)

        card.mousePressEvent = lambda event: on_card_clicked()

        # Store analysis data for later use
        card.setProperty("analysis_data", analysis)

        return card

    def show_image_details(self, analysis):
        """Show detailed information about a stored image"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Face Image Details")
        dialog.setModal(True)
        dialog.resize(700, 500)

        layout = QVBoxLayout(dialog)

        # Image display
        image_layout = QHBoxLayout()

        image_path = analysis.get('image_path', '')
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(300, 300)
        image_label.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 2px solid #2E86AB;
                border-radius: 5px;
            }
        """)

        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
        except Exception:
            image_label.setText("Image not available")

        image_layout.addWidget(image_label)

        # Analysis details
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        details_text.setStyleSheet("""
            QTextEdit {
                background-color: #F8F9FA;
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                padding: 10px;
            }
        """)

        # Format detailed information
        details_text.append(f"📁 File: {os.path.basename(image_path)}")
        details_text.append(f"📂 Path: {image_path}")
        details_text.append(f"👤 User: {analysis.get('user_name', 'Anonymous')}")
        details_text.append(f"🔧 Analysis Type: {analysis.get('analysis_type', 'Unknown')}")
        details_text.append(f"📅 Analyzed: {analysis.get('created_at', 'Unknown')}")
        details_text.append(f"⏱️ Processing Time: {analysis.get('processing_time', 'Unknown')}s")
        details_text.append(f"🤖 Model Used: {analysis.get('model_used', 'Unknown')}")
        details_text.append(f"⚙️ Detector Used: {analysis.get('detector_used', 'Unknown')}")

        # Analysis results
        result_data = analysis.get('result_data', {})
        if result_data:
            details_text.append(f"\n📊 Analysis Results:")
            for key, value in result_data.items():
                if isinstance(value, dict):
                    details_text.append(f"   {key.title()}:")
                    for sub_key, sub_value in value.items():
                        details_text.append(f"     {sub_key.title()}: {sub_value}")
                else:
                    details_text.append(f"   {key.title()}: {value}")

        image_layout.addWidget(details_text)
        layout.addLayout(image_layout)

        # Action buttons
        buttons_layout = QHBoxLayout()

        # View in search button
        search_btn = QPushButton("🔍 Search Similar Faces")
        search_btn.clicked.connect(lambda: self.search_similar_from_gallery(analysis, dialog))
        buttons_layout.addWidget(search_btn)

        # Export analysis button
        export_btn = QPushButton("📤 Export Analysis")
        export_btn.clicked.connect(lambda: self.export_analysis_data(analysis))
        buttons_layout.addWidget(export_btn)

        buttons_layout.addStretch()

        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(close_btn)

        layout.addLayout(buttons_layout)

        dialog.exec_()

    def search_similar_from_gallery(self, analysis, parent_dialog):
        """Search for faces similar to the selected gallery image"""
        image_path = analysis.get('image_path', '')

        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", "Original image file not found.")
            return

        # Switch to search tab
        self.main_tab_widget.setCurrentIndex(2)  # Search tab index

        # Set the search image
        self.search_image_path = image_path
        self.search_image_label.setText(f"Selected: {os.path.basename(image_path)}")
        self.search_image_label.setStyleSheet("QLabel { color: #27AE60; font-weight: bold; }")
        self.search_results_info.setText(f"Ready to search for faces similar to: {os.path.basename(image_path)}")

        # Close the details dialog
        parent_dialog.accept()

        # Show confirmation
        QMessageBox.information(self, "Search Ready",
                              f"Switched to Search tab. The image '{os.path.basename(image_path)}' is ready for similarity search.")

    def export_analysis_data(self, analysis):
        """Export individual analysis data"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Analysis", "", "JSON Files (*.json)"
            )

            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)

                QMessageBox.information(self, "Success",
                                      f"Analysis exported successfully to:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export analysis: {str(e)}")

    def create_search_tab(self):
        """Create the face search and comparison tab"""
        search_widget = QWidget()
        search_layout = QVBoxLayout(search_widget)

        # Search controls section
        search_controls = QGroupBox("Search Similar Faces in Database")
        search_controls.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")
        search_controls_layout = QVBoxLayout(search_controls)

        # Search image display
        search_image_group = QGroupBox("Search Image")
        search_image_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; border: 2px solid #2E86AB; border-radius: 5px; }")
        search_image_layout = QVBoxLayout(search_image_group)

        # Search image display area
        self.search_image_display = QLabel("No search image selected")
        self.search_image_display.setAlignment(Qt.AlignCenter)
        self.search_image_display.setMinimumSize(250, 250)
        self.search_image_display.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 2px dashed #BDC3C7;
                border-radius: 10px;
                font-size: 14px;
                color: #95A5A6;
            }
        """)
        search_image_layout.addWidget(self.search_image_display)

        # Image selection controls
        image_selection_layout = QHBoxLayout()

        select_image_btn = QPushButton("📷 Select Face Image")
        select_image_btn.setStyleSheet("QPushButton { background-color: #A8DADC; padding: 8px; border-radius: 5px; }")
        select_image_btn.clicked.connect(self.select_search_image)
        image_selection_layout.addWidget(select_image_btn)

        self.search_image_label = QLabel("No image selected")
        self.search_image_label.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; }")
        image_selection_layout.addWidget(self.search_image_label)

        image_selection_layout.addStretch()
        search_image_layout.addLayout(image_selection_layout)

        search_controls_layout.addWidget(search_image_group)

        # Search parameters
        params_layout = QHBoxLayout()

        threshold_label = QLabel("Similarity Threshold:")
        self.search_threshold = QComboBox()
        self.search_threshold.addItems(["0.3 (Very Similar)", "0.4 (Similar)", "0.5 (Moderately Similar)", "0.6 (Somewhat Similar)", "0.7 (Different)"])
        self.search_threshold.setCurrentText("0.6 (Somewhat Similar)")
        params_layout.addWidget(threshold_label)
        params_layout.addWidget(self.search_threshold)

        max_results_label = QLabel("Max Results:")
        self.search_max_results = QComboBox()
        self.search_max_results.addItems(["5", "10", "20", "50"])
        self.search_max_results.setCurrentText("10")
        params_layout.addWidget(max_results_label)
        params_layout.addWidget(self.search_max_results)

        search_btn = QPushButton("🔍 Search Similar Faces")
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        search_btn.clicked.connect(self.search_similar_faces)
        params_layout.addWidget(search_btn)

        params_layout.addStretch()
        search_controls_layout.addLayout(params_layout)

        search_layout.addWidget(search_controls)

        # Search results section
        results_group = QGroupBox("Visual Face Comparison")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")
        results_layout = QVBoxLayout(results_group)

        # Create horizontal layout for side-by-side comparison
        comparison_layout = QHBoxLayout()

        # Left side: Search image
        search_results_layout = QVBoxLayout()

        search_results_label = QLabel("🔍 Search Results:")
        search_results_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #2E86AB; }")
        search_results_layout.addWidget(search_results_label)

        # Results table (smaller version)
        self.search_results_table = QTableWidget()
        self.search_results_table.setColumnCount(4)
        self.search_results_table.setHorizontalHeaderLabels(["Rank", "Similarity", "Match %", "User"])
        self.search_results_table.horizontalHeader().setStretchLastSection(True)
        self.search_results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.search_results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.search_results_table.setMaximumHeight(300)
        self.search_results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #BDC3C7;
                selection-background-color: #A8DADC;
            }
            QHeaderView::section {
                background-color: #2E86AB;
                color: white;
                padding: 5px;
                border: 1px solid #BDC3C7;
            }
        """)

        # Connect double-click to view details
        self.search_results_table.doubleClicked.connect(self.view_search_result_details)

        search_results_layout.addWidget(self.search_results_table)

        comparison_layout.addLayout(search_results_layout, 1)

        # Right side: Visual comparison
        visual_layout = QVBoxLayout()

        visual_label = QLabel("📸 Similar Faces:")
        visual_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #2E86AB; }")
        visual_layout.addWidget(visual_label)

        # Scrollable area for similar faces
        self.similar_faces_scroll = QScrollArea()
        self.similar_faces_scroll.setWidgetResizable(True)
        self.similar_faces_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.similar_faces_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.similar_faces_scroll.setMaximumHeight(400)
        self.similar_faces_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
            }
        """)

        self.similar_faces_widget = QWidget()
        self.similar_faces_layout = QVBoxLayout(self.similar_faces_widget)
        self.similar_faces_layout.setSpacing(10)

        self.similar_faces_scroll.setWidget(self.similar_faces_widget)
        visual_layout.addWidget(self.similar_faces_scroll)

        comparison_layout.addLayout(visual_layout, 2)

        results_layout.addLayout(comparison_layout)

        # Results info
        self.search_results_info = QLabel("Upload an image above to search for similar faces")
        self.search_results_info.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; padding: 10px; }")
        results_layout.addWidget(self.search_results_info)

        search_layout.addWidget(results_group)

        self.main_tab_widget.addTab(search_widget, "🔍 Search")

        self.main_tab_widget.addTab(self.create_internet_search_tab(), "🌐 Internet Search")

        # Web browser tab (with WebEngine)
        if HAS_WEBENGINE:
            self.main_tab_widget.addTab(self.create_web_browser_tab(), "🌐 Web Browser")
            print("✅ Web Browser tab added")
        else:
            # Add a placeholder tab if WebEngine not available
            web_placeholder = QWidget()
            web_layout = QVBoxLayout(web_placeholder)
            web_layout.addWidget(QLabel("🌐 Web Browser requires PyQtWebEngine"))
            web_layout.addWidget(QLabel("Install with: pip install PyQtWebEngine"))
            self.main_tab_widget.addTab(web_placeholder, "🌐 Web Browser")
            print("⚠️ WebEngine not available - showing placeholder")

    def create_web_browser_tab(self):
        """Create the web browser tab with WebEngine"""
        web_widget = QWidget()
        web_layout = QVBoxLayout(web_widget)

        # Navigation bar
        nav_layout = QHBoxLayout()

        # Back button
        back_btn = QPushButton("⬅️ Back")
        back_btn.setStyleSheet("QPushButton { padding: 5px; border-radius: 3px; }")
        back_btn.clicked.connect(self.web_back)
        nav_layout.addWidget(back_btn)

        # Forward button
        forward_btn = QPushButton("➡️ Forward")
        forward_btn.setStyleSheet("QPushButton { padding: 5px; border-radius: 3px; }")
        forward_btn.clicked.connect(self.web_forward)
        nav_layout.addWidget(forward_btn)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("QPushButton { padding: 5px; border-radius: 3px; }")
        refresh_btn.clicked.connect(self.web_refresh)
        nav_layout.addWidget(refresh_btn)

        # URL bar
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("Enter URL or search term...")
        self.url_bar.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #BDC3C7;
                border-radius: 3px;
                background-color: #FFFFFF;
            }
        """)
        self.url_bar.returnPressed.connect(self.web_navigate)
        nav_layout.addWidget(self.url_bar)

        # Go button
        go_btn = QPushButton("Go")
        go_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1F5F7A;
            }
        """)
        go_btn.clicked.connect(self.web_navigate)
        nav_layout.addWidget(go_btn)

        nav_layout.addStretch()
        web_layout.addLayout(nav_layout)

        # WebEngine view
        self.web_view = QWebEngineView()
        self.web_view.setStyleSheet("""
            QWebEngineView {
                border: 1px solid #BDC3C7;
                border-radius: 5px;
            }
        """)

        # Set initial page (Google Images for reverse image search)
        self.web_view.setUrl(QUrl("https://images.google.com/"))
        self.url_bar.setText("https://images.google.com/")

        # Connect URL change signal
        self.web_view.urlChanged.connect(self.update_url_bar)
        self.web_view.loadFinished.connect(self.web_load_finished)

        web_layout.addWidget(self.web_view)

        # Status bar for web view
        self.web_status_label = QLabel("Ready")
        self.web_status_label.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; padding: 5px; }")
        web_layout.addWidget(self.web_status_label)

        return web_widget

    def web_navigate(self):
        """Navigate to URL entered in the address bar"""
        if HAS_WEBENGINE and hasattr(self, 'web_view'):
            url_text = self.url_bar.text().strip()

            # If it's not a full URL, treat as search
            if not url_text.startswith(('http://', 'https://')):
                url_text = f"https://www.google.com/search?q={url_text.replace(' ', '+')}"

            self.web_view.setUrl(QUrl(url_text))

    def update_url_bar(self, url):
        """Update the URL bar when navigation occurs"""
        if HAS_WEBENGINE and hasattr(self, 'url_bar'):
            self.url_bar.setText(url.toString())

    def web_load_finished(self, success):
        """Handle page load completion"""
        if HAS_WEBENGINE and hasattr(self, 'web_status_label'):
            if success:
                self.web_status_label.setText("✅ Page loaded successfully")
                self.web_status_label.setStyleSheet("QLabel { color: #27AE60; font-weight: bold; padding: 5px; }")
            else:
                self.web_status_label.setText("❌ Failed to load page")
                self.web_status_label.setStyleSheet("QLabel { color: #E74C3C; font-weight: bold; padding: 5px; }")

            # Reset style after 3 seconds
            QTimer.singleShot(3000, lambda: self.web_status_label.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; padding: 5px; }"))

    def web_back(self):
        """Go back in web history"""
        if HAS_WEBENGINE and hasattr(self, 'web_view'):
            self.web_view.back()

    def web_forward(self):
        """Go forward in web history"""
        if HAS_WEBENGINE and hasattr(self, 'web_view'):
            self.web_view.forward()

    def web_refresh(self):
        """Refresh current page"""
        if HAS_WEBENGINE and hasattr(self, 'web_view'):
            self.web_view.reload()

    def open_web_search(self, url):
        """Open URL in the web browser tab"""
        if HAS_WEBENGINE and hasattr(self, 'web_view'):
            self.web_view.setUrl(QUrl(url))
            # Switch to web browser tab
            for i in range(self.main_tab_widget.count()):
                if self.main_tab_widget.tabText(i) == "🌐 Web Browser":
                    self.main_tab_widget.setCurrentIndex(i)
                    break

    def create_internet_search_tab(self):
        """Create the internet image search tab"""

        internet_search_widget = QWidget()

        internet_search_layout = QVBoxLayout(internet_search_widget)

        # Upload section

        upload_group = QGroupBox("Upload Image for Internet Search")

        upload_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")

        upload_layout = QVBoxLayout(upload_group)

        # Image display area
        self.internet_image_display = QLabel("No image uploaded")
        self.internet_image_display.setAlignment(Qt.AlignCenter)
        self.internet_image_display.setMinimumSize(250, 250)
        self.internet_image_display.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 2px dashed #BDC3C7;
                border-radius: 10px;
                font-size: 14px;
                color: #95A5A6;
            }
        """)
        upload_layout.addWidget(self.internet_image_display)

        # Upload button

        upload_btn = QPushButton("📷 Upload Image")

        upload_btn.setStyleSheet("QPushButton { background-color: #A8DADC; padding: 8px; border-radius: 5px; }")

        upload_btn.clicked.connect(self.upload_internet_image)

        upload_layout.addWidget(upload_btn)

        internet_search_layout.addWidget(upload_group)

        # Search section

        search_group = QGroupBox("Search the Internet")

        search_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")

        search_layout = QVBoxLayout(search_group)

        # Search buttons container
        search_buttons_layout = QHBoxLayout()
        search_buttons_layout.setSpacing(10)

        # Selenium Search button
        search_btn = QPushButton("🔍 Search with Selenium")
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        search_btn.clicked.connect(self.perform_internet_search)

        # Google Lens button (experimental)
        lens_btn = QPushButton("🔍 Search with Google Lens (Experimental)")
        lens_btn.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)
        lens_btn.clicked.connect(self.perform_google_lens_search)

        # Manual search button
        manual_btn = QPushButton("🔍 Manual Search (Browser)")
        manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E8449;
            }
        """)
        manual_btn.clicked.connect(self.open_manual_search)

        # Add buttons to layout
        search_buttons_layout.addWidget(search_btn)
        search_buttons_layout.addWidget(lens_btn)
        search_buttons_layout.addWidget(manual_btn)
        search_layout.addLayout(search_buttons_layout)

        internet_search_layout.addWidget(search_group)

        # Results section

        results_group = QGroupBox("Search Results")

        results_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86AB; }")

        results_layout = QVBoxLayout(results_group)

        # Results text area

        self.internet_results_text = QTextEdit()

        self.internet_results_text.setReadOnly(True)

        self.internet_results_text.setStyleSheet("""

            QTextEdit {

                background-color: #F8F9FA;

                border: 2px solid #2E86AB;

                border-radius: 5px;

                padding: 10px;

            }

        """)

        results_layout.addWidget(self.internet_results_text)

        # Scrollable area for image thumbnails

        self.internet_thumbnails_scroll = QScrollArea()

        self.internet_thumbnails_scroll.setWidgetResizable(True)

        self.internet_thumbnails_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.internet_thumbnails_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.internet_thumbnails_scroll.setMaximumHeight(300)

        self.internet_thumbnails_scroll.setStyleSheet("""

            QScrollArea {

                border: 1px solid #BDC3C7;

                border-radius: 5px;

            }

        """)

        self.internet_thumbnails_widget = QWidget()

        self.internet_thumbnails_layout = QHBoxLayout(self.internet_thumbnails_widget)

        self.internet_thumbnails_layout.setSpacing(10)

        self.internet_thumbnails_scroll.setWidget(self.internet_thumbnails_widget)

        results_layout.addWidget(self.internet_thumbnails_scroll)

        internet_search_layout.addWidget(results_group)

        return internet_search_widget

    def open_manual_search(self):
        """Open Google Images manually in browser"""
        if not self.internet_image_path:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return

        try:
            # Clear previous results
            self.clear_internet_results()

            # Update results text
            if hasattr(self, 'internet_results_text') and self.is_widget_valid(self.internet_results_text):
                self.internet_results_text.append("🔍 Opening Google Images in the Web Browser tab...")
                self.internet_results_text.append(f"Image: {os.path.basename(self.internet_image_path)}")
                self.internet_results_text.append("\n📋 Manual Search Instructions:")
                self.internet_results_text.append("1. Google Images is now open in the '🌐 Web Browser' tab")
                self.internet_results_text.append("2. Click the camera icon (📷) in the search box")
                self.internet_results_text.append("3. Select 'Upload an image' or drag & drop your image")
                self.internet_results_text.append("4. Click 'Search by image' to see results")
                self.internet_results_text.append("\n✅ All browsing happens within the application!")

            # Open Google Images in web browser tab
            self.open_web_search("https://images.google.com/")

            # Also copy the image path to clipboard for easy access
            try:
                import pyperclip
                pyperclip.copy(self.internet_image_path)
                self.internet_results_text.append(f"\n📋 Image path copied to clipboard: {self.internet_image_path}")
            except ImportError:
                print("pyperclip not available - install with: pip install pyperclip")
            except Exception as e:
                print(f"Could not copy to clipboard: {e}")

        except Exception as e:
            error_msg = f"Error opening web search: {str(e)}"
            print(error_msg)
            if hasattr(self, 'internet_results_text') and self.is_widget_valid(self.internet_results_text):
                self.internet_results_text.append(f"❌ {error_msg}")
                self.internet_results_text.append("\n💡 Try opening https://images.google.com/ manually")

    def upload_internet_image(self):
        """Upload image for internet search"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Upload Image for Search", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)", options=options
            )

            if file_path:
                self.internet_image_path = file_path

                # Clear previous results
                self.clear_internet_results()

                # Ensure display widget exists
                if not hasattr(self, 'internet_image_display') or self.internet_image_display is None:
                    print("Warning: internet_image_display not available")
                    return

                # Ensure text widget exists
                if not hasattr(self, 'internet_results_text') or self.internet_results_text is None:
                    print("Warning: internet_results_text not available")
                    return

                # Ensure display widget is valid
                try:
                    # Test if widget is properly initialized
                    current_text = self.internet_image_display.text()
                except Exception as e:
                    print(f"Error: internet_image_display not properly initialized: {e}")
                    return

                # Display the image
                self._display_uploaded_image(file_path)

        except Exception as e:
            print(f"Error in upload_internet_image: {e}")
            if hasattr(self, 'internet_results_text') and self.internet_results_text is not None:
                self.internet_results_text.setText(f"Error uploading image: {str(e)}")

    def _display_uploaded_image(self, file_path):
        """Helper method to display uploaded image with error handling"""
        try:
            pixmap = QPixmap(file_path)

            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(230, 230, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.internet_image_display.setPixmap(scaled_pixmap)
                self.internet_image_display.setText("")
            else:
                if hasattr(self, 'internet_image_display'):
                    self.internet_image_display.setText("❌ Invalid Image")

        except Exception as e:
            print(f"Error displaying image with QPixmap: {e}")
            self._try_pil_loading(file_path)

    def _try_pil_loading(self, file_path):
        """Attempt to load image using PIL as fallback"""
        try:
            from PIL import Image
            import io
            import os

            # Try loading with PIL first
            pil_image = Image.open(file_path)
            temp_path = "temp_display_image.png"
            pil_image.save(temp_path)

            try:
                # Load with QPixmap
                alt_pixmap = QPixmap(temp_path)
                if not alt_pixmap.isNull():
                    scaled_pixmap = alt_pixmap.scaled(230, 230, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.internet_image_display.setPixmap(scaled_pixmap)
                    self.internet_image_display.setText("")
                else:
                    self.internet_image_display.setText("❌ Invalid Image")
            except Exception as e:
                print(f"Error loading image with QPixmap after PIL conversion: {e}")
                self.internet_image_display.setText("❌ Error loading image")
            finally:
                # Clean up the temporary file
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    print(f"Error removing temporary file: {e}")
        except Exception as e:
            print(f"Error in PIL image processing: {e}")
            if hasattr(self, 'internet_image_display'):
                self.internet_image_display.setText("❌ Error processing image")

    def closeEvent(self, event):
        """Check if a Qt widget is still valid and not deleted"""
        try:
            if widget is None:
                return False

            # Special handling for layouts
            if hasattr(widget, 'count'):
                # For layouts, just check if they have a parent or are the main layout
                if hasattr(widget, 'parent') and widget.parent() is None:
                    return False
                return True

            # For regular widgets
            if hasattr(widget, 'isWidgetType') and widget.isWidgetType():
                # Simple check: try to access objectName (this will raise RuntimeError if deleted)
                widget.objectName()
                return True

            return False
        except (RuntimeError, AttributeError):
            return False

    def closeEvent(self, event):
        """Handle close event"""
        self._is_closing = True
        event.accept()

    def perform_google_lens_search(self):
        """Perform image search using Google Lens"""
        if not HAS_GOOGLE_LENS:
            QMessageBox.warning(self, "Feature Not Available",
                             "Google Lens search is not available.\n\n"
                             "This is an experimental feature that may not work reliably.\n\n"
                             "Alternative options:\n"
                             "• 'Manual Search (Browser)' - always works!\n"
                             "• 'Search with Selenium' - automated search\n\n"
                             "To enable Google Lens:\n"
                             "pip install git+https://github.com/krishna2206/google-lens-python.git")
            return
            
        if not hasattr(self, 'internet_image_path') or not self.internet_image_path:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return

        try:
            # Clear previous results
            self.clear_internet_results()

            # Ensure results text widget is available
            if not hasattr(self, 'internet_results_text') or not self.is_widget_valid(self.internet_results_text):
                print("Warning: Results text widget not available")
                return

            # Show searching message
            self.internet_results_text.append("🔍 Searching with Google Lens...")
            self.internet_results_text.append("This may take a few moments...")

            # Create and start the worker thread
            self.search_worker = self.SearchWorker(self.internet_image_path, timeout_seconds=30)
            self.search_worker.finished.connect(self._handle_google_lens_results)
            self.search_worker.start()

        except Exception as e:
            error_msg = f"Error starting Google Lens search: {str(e)}"
            print(error_msg)
            if hasattr(self, 'internet_results_text') and self.is_widget_valid(self.internet_results_text):
                self.internet_results_text.append(f"❌ {error_msg}")
                self.internet_results_text.append("\n💡 Alternative options:")
                self.internet_results_text.append("• Use 'Manual Search (Browser)' - always works!")
                self.internet_results_text.append("• Use 'Search with Selenium' - automated search")
                self.internet_results_text.append("• Google Lens library may be outdated or incompatible")
                self.internet_results_text.append(f"\n🔧 Try updating: pip install --upgrade git+https://github.com/krishna2206/google-lens-python.git")

                # Also suggest running the diagnostic
                self.internet_results_text.append(f"\n💡 Debug: Run 'python test_googlelens.py' to test GoogleLens separately")

    def _handle_google_lens_results(self, result):
        """Handle the results from Google Lens search"""
        try:
            if not hasattr(self, 'internet_results_text') or not self.is_widget_valid(self.internet_results_text):
                return

            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error occurred')
                self.internet_results_text.append(f"❌ Search failed: {error_msg}")
                self.internet_results_text.append("\n💡 Don't worry! Try these alternatives:")
                self.internet_results_text.append("• 'Manual Search (Browser)' - opens Google Images automatically")
                self.internet_results_text.append("• 'Search with Selenium' - automated reverse image search")
                self.internet_results_text.append("• Both methods work reliably and show full results!")
                return

            items = result.get('data', [])
            if not items:
                self.internet_results_text.append("❌ No results found by Google Lens")
                self.internet_results_text.append("\n💡 This could mean:")
                self.internet_results_text.append("• The Google Lens library returned empty results")
                self.internet_results_text.append("• The image might be too unique or not publicly available")
                self.internet_results_text.append("• Try the other search methods instead!")
                return

            self.internet_results_text.append(f"✅ Found {len(items)} results:\n")
            
            for idx, item in enumerate(items[:10], 1):  # Show top 10 results
                title = item.get('title', 'No title')
                url = item.get('url', '')
                
                # Create a clickable link
                self.internet_results_text.append(f"{idx}. {title}")
                if url:
                    self.internet_results_text.append(f"   <a href='{url}'>{url}</a>")
                self.internet_results_text.append("")
                
        except Exception as e:
            error_msg = f"Error processing results: {str(e)}"
            print(error_msg)
            if hasattr(self, 'internet_results_text') and self.is_widget_valid(self.internet_results_text):
                self.internet_results_text.append(f"❌ {error_msg}")

    def clear_internet_results(self):
        """Clear internet search results"""
        # Don't proceed if the window is being closed
        if hasattr(self, '_is_closing') and self._is_closing:
            return

        try:
            # Clear text results if available
            if hasattr(self, 'internet_results_text'):
                try:
                    if self.internet_results_text is not None and self.is_widget_valid(self.internet_results_text):
                        self.internet_results_text.clear()
                except Exception as e:
                    print(f"Warning: Error clearing text results: {e}")

            # Clear thumbnails if available
            if hasattr(self, 'internet_thumbnails_layout'):
                try:
                    layout = self.internet_thumbnails_layout
                    if layout is not None and self.is_widget_valid(layout):
                        # Remove all items from layout
                        while layout.count() > 0:
                            item = layout.takeAt(0)
                            if item is not None:
                                widget = item.widget()
                                if widget is not None and self.is_widget_valid(widget):
                                    try:
                                        widget.setParent(None)
                                        widget.deleteLater()
                                    except RuntimeError:
                                        pass  # Widget already deleted
                except Exception as e:
                    if 'wrapped C/C++ object' not in str(e):
                        print(f"Warning in clear_internet_results: {e}")
        except Exception as e:
            if 'wrapped C/C++ object' not in str(e):
                print(f"Unexpected error in clear_internet_results: {e}")

    def perform_internet_search(self):
        """Perform automated reverse image search using Selenium"""
        if not self.internet_image_path:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return

        try:
            # Clear previous results
            self.clear_internet_results()

            # Ensure text widget exists and is still valid
            if not hasattr(self, 'internet_results_text') or not self.is_widget_valid(self.internet_results_text):
                print("Warning: internet_results_text not available or invalid")
                return

            self.internet_results_text.append("🔍 Starting automated reverse image search...")
            self.internet_results_text.append(f"Searching for: {os.path.basename(self.internet_image_path)}")

            # Set up Chrome options for Windows compatibility
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-javascript")
            chrome_options.add_argument("--disable-stylesheet")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # Add debug mode option
            debug_mode = False  # Set to True to see browser during development
            if not debug_mode:
                chrome_options.add_argument("--headless")

            # Initialize Chrome driver with webdriver-manager
            try:
                print("Attempting to initialize Chrome driver with webdriver-manager...")
                driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
                print("✅ Chrome driver initialized successfully with webdriver-manager")
            except Exception as e:
                print(f"⚠️ webdriver-manager failed: {e}")
                print("Attempting fallback to system Chrome...")
                try:
                    driver = webdriver.Chrome(options=chrome_options)
                    print("✅ System Chrome driver initialized successfully")
                except Exception as e2:
                    print(f"❌ Both webdriver-manager and system Chrome failed: {e2}")
                    raise Exception(f"Chrome driver initialization failed: {e2}")

            try:
                # Navigate to Google Images
                driver.get("https://images.google.com/")

                # Try multiple selectors for the camera/search by image button
                camera_selectors = [
                    "[aria-label='Search by image']",
                    "[aria-label*='image']",
                    "[title*='Search by image']",
                    "div[data-testid='camera-button']",
                    "div[aria-label*='camera']",
                    "[data-value='camera']",
                    "g-camera-button",  # New Google Images selector
                    ".camera-button",
                    "[jsname='hSRGPd']"  # Alternative selector
                ]

                camera_button = None
                for selector in camera_selectors:
                    try:
                        camera_button = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        if camera_button:
                            print(f"Found camera button with selector: {selector}")
                            break
                    except:
                        continue

                if not camera_button:
                    raise Exception("Could not find camera/search by image button")

                camera_button.click()

                # Wait for upload area to appear
                upload_selectors = [
                    "input[type='file']",
                    "[name='encoded_image']",
                    "input[accept*='image']",
                    "[data-testid='file-input']"
                ]

                upload_input = None
                for selector in upload_selectors:
                    try:
                        upload_input = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        if upload_input:
                            print(f"Found upload input with selector: {selector}")
                            break
                    except:
                        continue

                if not upload_input:
                    raise Exception("Could not find image upload input")

                # Upload the image file
                upload_input.send_keys(self.internet_image_path)

                # Wait for search results - try multiple selectors
                result_selectors = [
                    ".ivg-i",  # Original selector
                    ".rg_i",   # Alternative results selector
                    "[data-src]",  # Images with data-src
                    ".islrg img",  # Image search results
                    "[jsname='hSRGPd'] img",  # New Google interface
                    ".H8Rx8c img"  # Another possible selector
                ]

                results_element = None
                for selector in result_selectors:
                    try:
                        results_element = WebDriverWait(driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        if results_element:
                            print(f"Found results with selector: {selector}")
                            break
                    except:
                        continue

                if not results_element:
                    # Try to find any image results
                    try:
                        results_element = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "img"))
                        )
                        print("Found results by searching for any img tag")
                    except:
                        raise Exception("No search results found")

                # Extract search results - try multiple approaches
                results = []

                # Try the original approach first
                try:
                    results = driver.find_elements(By.CSS_SELECTOR, ".ivg-i")
                    if results:
                        print(f"Found {len(results)} results with .ivg-i selector")
                except:
                    pass

                # If no results, try alternative selectors
                if not results:
                    for selector in [".rg_i", ".islrg img", "[data-src]", ".H8Rx8c img"]:
                        try:
                            results = driver.find_elements(By.CSS_SELECTOR, selector)
                            if results:
                                print(f"Found {len(results)} results with {selector} selector")
                                break
                        except:
                            continue

                # If still no results, get all images as fallback
                if not results:
                    try:
                        all_images = driver.find_elements(By.TAG_NAME, "img")
                        # Filter out small images (likely icons) and get actual results
                        results = [img for img in all_images if img.size['height'] > 50 and img.size['width'] > 50]
                        print(f"Found {len(results)} image results using fallback method")
                    except Exception as e:
                        print(f"Fallback method failed: {e}")

                if results:
                    # Check if widget is still valid before appending results
                    if hasattr(self, 'internet_results_text') and self.internet_results_text is not None:
                        try:
                            if self.is_widget_valid(self.internet_results_text):
                                self.internet_results_text.append(f"✅ Found {len(results)} similar images!")
                                self.internet_results_text.append("\n📸 Similar Images:")
                        except Exception as e:
                            print(f"Warning: Cannot update results text: {e}")
                    else:
                        print("Warning: internet_results_text not available for results update")
                        return

                    for i, result in enumerate(results[:10]):  # Show top 10 results
                        try:
                            # Extract image URL - try multiple approaches
                            img_url = None
                            description = f"Result {i+1}"

                            # Try different methods to get image URL
                            try:
                                img_url = result.get_attribute("src")
                                if not img_url or img_url.startswith("data:"):
                                    img_url = result.get_attribute("data-src")
                            except:
                                pass

                            # Try to find parent elements for more context
                            if not img_url:
                                try:
                                    parent = result.find_element(By.XPATH, "..")
                                    img_url = parent.get_attribute("href") or parent.get_attribute("data-src")
                                except:
                                    pass

                            # Get description from various sources
                            try:
                                # Try alt text first
                                description = result.get_attribute("alt") or result.get_attribute("title") or description

                                # Try to find nearby text elements
                                if not description or description == f"Result {i+1}":
                                    try:
                                        parent_div = result.find_element(By.XPATH, "../../..")
                                        text_elements = parent_div.find_elements(By.TAG_NAME, "span")
                                        for text_elem in text_elements:
                                            text_content = text_elem.text.strip()
                                            if text_content and len(text_content) > 3:
                                                description = text_content
                                                break
                                    except:
                                        pass
                            except:
                                pass

                            # Safely update text widget
                            if hasattr(self, 'internet_results_text') and self.internet_results_text is not None:
                                try:
                                    if self.is_widget_valid(self.internet_results_text):
                                        self.internet_results_text.append(f"\n{i+1}. {description}")
                                        if img_url and not img_url.startswith("data:"):
                                            self.internet_results_text.append(f"   Image URL: {img_url}")

                                            # Create thumbnail display - safely check if layout exists
                                            if hasattr(self, 'internet_thumbnails_layout') and self.internet_thumbnails_layout is not None:
                                                try:
                                                    if hasattr(self.internet_thumbnails_layout, 'count'):
                                                        self.create_thumbnail_display(img_url, description, i+1)
                                                except Exception as e:
                                                    print(f"Warning: Cannot create thumbnail: {e}")
                                except Exception as e:
                                    print(f"Warning: Cannot update text widget: {e}")

                        except Exception as e:
                            if hasattr(self, 'internet_results_text') and self.internet_results_text is not None:
                                try:
                                    if self.is_widget_valid(self.internet_results_text):
                                        self.internet_results_text.append(f"\n{i+1}. Error extracting result: {str(e)}")
                                except Exception as text_e:
                                    print(f"Warning: Cannot update error text: {text_e}")

                else:
                    # Check if widget is still valid before showing no results message
                    if hasattr(self, 'internet_results_text') and self.internet_results_text is not None:
                        try:
                            if self.is_widget_valid(self.internet_results_text):
                                self.internet_results_text.append("❌ No similar images found.")
                                self.internet_results_text.append("💡 This could mean:")
                                self.internet_results_text.append("• The image is too unique or not publicly available")
                                self.internet_results_text.append("• Google's search by image feature couldn't match it")
                                self.internet_results_text.append("• The image format or quality is not suitable")
                        except Exception as e:
                            print(f"Warning: Cannot update no results text: {e}")

            except Exception as e:
                # Enhanced error reporting
                error_msg = f"❌ Search failed: {str(e)}"
                print(f"Selenium error details: {error_msg}")

                # Check if widget is still valid before showing error message
                if hasattr(self, 'internet_results_text') and self.internet_results_text is not None:
                    try:
                        if self.is_widget_valid(self.internet_results_text):
                            self.internet_results_text.append(error_msg)
                            self.internet_results_text.append("\n💡 Troubleshooting suggestions:")
                            self.internet_results_text.append("• Ensure you have a stable internet connection")
                            self.internet_results_text.append("• Try a more common image (celebrity, landmark, etc.)")
                            self.internet_results_text.append("• Check if Chrome browser is installed and updated")
                            self.internet_results_text.append("• Verify that Google Images is accessible in your region")
                            self.internet_results_text.append("• The reverse image search might be blocked or limited")

                            # Check if it's a Chrome driver issue
                            if "Chrome" in str(e) or "webdriver" in str(e).lower():
                                self.internet_results_text.append("\n🔧 Chrome Driver Issues:")
                                self.internet_results_text.append("• Install Chrome browser if not already installed")
                                self.internet_results_text.append("• Update Chrome to the latest version")
                                self.internet_results_text.append("• Try running: pip install webdriver-manager")
                    except Exception as text_e:
                        print(f"Warning: Cannot update error message: {text_e}")
                else:
                    print(f"Cannot display error message: internet_results_text not available")

            finally:
                driver.quit()

        except Exception as e:
            if hasattr(self, 'internet_results_text') and self.internet_results_text is not None:
                try:
                    if self.is_widget_valid(self.internet_results_text):
                        self.internet_results_text.append(f"❌ Failed to initialize search: {str(e)}")
                        self.internet_results_text.append("\n🔧 Setup Troubleshooting:")
                        self.internet_results_text.append("• Install Chrome browser: https://www.google.com/chrome/")
                        self.internet_results_text.append("• Update ChromeDriver: pip install --upgrade webdriver-manager")
                        self.internet_results_text.append("• Install Selenium: pip install selenium")
                        self.internet_results_text.append("• Check if Google Images reverse search works manually")
                        self.internet_results_text.append("• Try disabling VPN or proxy if using one")
                except Exception as text_e:
                    print(f"Warning: Cannot update final error message: {text_e}")
            print(f"Error in perform_internet_search: {e}")

    def create_thumbnail_display(self, img_url, description, index):
        """Create a clickable thumbnail for search results"""
        try:
            # Create thumbnail widget
            thumbnail_frame = QFrame()
            thumbnail_frame.setFrameStyle(QFrame.Box)
            thumbnail_frame.setStyleSheet("""
                QFrame {
                    border: 2px solid #BDC3C7;
                    border-radius: 5px;
                    background-color: #F8F9FA;
                }
                QFrame:hover {
                    border: 2px solid #2E86AB;
                    background-color: #E8F4F8;
                }
            """)
            thumbnail_frame.setFixedSize(150, 180)
            thumbnail_frame.setCursor(Qt.PointingHandCursor)

            # Layout for thumbnail
            thumb_layout = QVBoxLayout(thumbnail_frame)
            thumb_layout.setSpacing(5)
            thumb_layout.setContentsMargins(5, 5, 5, 5)

            # Image display
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(120, 120)
            image_label.setMaximumSize(120, 120)
            image_label.setStyleSheet("""
                QLabel {
                    background-color: #FFFFFF;
                    border: 1px solid #BDC3C7;
                    border-radius: 3px;
                }
            """)

            # Download and display thumbnail with better error handling
            try:
                # Simple approach: use urllib to download and save image
                req = urllib.request.Request(img_url)
                with urllib.request.urlopen(req) as response:
                    image_data = response.read()

                # Save to temporary file and load with QPixmap
                temp_filename = f"temp_thumbnail_{index}.jpg"
                with open(temp_filename, 'wb') as f:
                    f.write(image_data)

                # Load with QPixmap
                pixmap = QPixmap(temp_filename)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    image_label.setPixmap(scaled_pixmap)
                else:
                    image_label.setText("❌\nInvalid\nImage")

                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

            except Exception as img_error:
                print(f"Error loading thumbnail image: {img_error}")
                image_label.setText("⚠️\nLoad\nError")

            thumb_layout.addWidget(image_label)

            # Description
            desc_label = QLabel(description[:50] + "..." if len(description) > 50 else description)
            desc_label.setStyleSheet("""
                QLabel {
                    font-size: 10px;
                    color: #2C3E50;
                    text-align: center;
                }
            """)
            desc_label.setWordWrap(True)
            thumb_layout.addWidget(desc_label)

            # Click handler
            def open_full_image():
                try:
                    webbrowser.open(img_url)
                except Exception as e:
                    print(f"Error opening image URL: {e}")

            thumbnail_frame.mousePressEvent = lambda event: open_full_image()

            # Store URL for reference
            thumbnail_frame.setProperty("image_url", img_url)

            # Safely add to layout
            if hasattr(self, 'internet_thumbnails_layout') and self.internet_thumbnails_layout is not None:
                try:
                    # Check if layout is still valid
                    if hasattr(self.internet_thumbnails_layout, 'addWidget'):
                        self.internet_thumbnails_layout.addWidget(thumbnail_frame)
                    else:
                        print("Warning: internet_thumbnails_layout not valid for adding widgets")
                except Exception as e:
                    print(f"Warning: Cannot add thumbnail to layout: {e}")
            else:
                print("Warning: internet_thumbnails_layout not available")

        except Exception as e:
            print(f"Error creating thumbnail: {e}")

    # Search functionality methods
    def select_search_image(self):
        """Select image for face search"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Face Image for Search", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)", options=options
        )

        if file_path:
            self.search_image_path = file_path
            self.search_image_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.search_image_label.setStyleSheet("QLabel { color: #27AE60; font-weight: bold; }")
            self.search_results_info.setText(f"Ready to search for faces similar to: {os.path.basename(file_path)}")

            # Display the selected image
            try:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(230, 230, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.search_image_display.setPixmap(scaled_pixmap)
                    self.search_image_display.setText("")
                else:
                    self.search_image_display.setText("❌ Invalid Image")
                    self.search_image_display.setStyleSheet("""
                        QLabel {
                            background-color: #FFE6E6;
                            border: 2px dashed #E74C3C;
                            border-radius: 10px;
                            color: #E74C3C;
                            font-size: 14px;
                            text-align: center;
                        }
                    """)
            except Exception as e:
                self.search_image_display.setText("⚠️ Image Error")
                self.search_image_display.setStyleSheet("""
                    QLabel {
                        background-color: #FFF5E6;
                        border: 2px dashed #F39C12;
                        border-radius: 10px;
                        color: #F39C12;
                        font-size: 14px;
                        text-align: center;
                    }
                """)
                print(f"Error displaying search image: {e}")

    def extract_face_embedding(self, image_path):
        """Extract face embedding from image for search"""
        try:
            # Use DeepFace to extract real embedding
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                # Extract embedding using DeepFace represent function
                embedding_result = DeepFace.represent(image_path, enforce_detection=False, model_name="Facenet")

                if embedding_result and len(embedding_result) > 0:
                    # Get the first face embedding
                    embedding_data = embedding_result[0]['embedding']
                    facial_area = embedding_result[0]['facial_area']

                    return np.array(embedding_data), facial_area
                else:
                    print(f"Warning: No face detected in {image_path}")
                    return None, None

        except Exception as e:
            print(f"Error extracting embedding: {e}")
            print(f"Image path: {image_path}")
            return None, None

    def create_embedding_from_analysis(self, analysis_result):
        """Create a simple embedding representation from analysis data (fallback)"""
        if not analysis_result:
            return None

        # Create a simple embedding based on analysis features
        # This is a simplified representation - in practice you'd use actual face embeddings
        embedding_features = []

        try:
            # Age contribution
            if 'age' in analysis_result and analysis_result['age'] is not None:
                age = float(analysis_result['age'])
                embedding_features.extend([age / 100.0])

            # Gender contribution (0 for male, 1 for female)
            if 'gender' in analysis_result and analysis_result['gender'] is not None:
                gender_data = analysis_result['gender']
                # Gender is a dict with 'Man' and 'Woman' confidence scores
                if isinstance(gender_data, dict):
                    man_confidence = float(gender_data.get('Man', 0))
                    woman_confidence = float(gender_data.get('Woman', 0))
                    gender_score = 1.0 if woman_confidence > man_confidence else 0.0
                else:
                    # Fallback for string format
                    gender_score = 1.0 if str(gender_data).lower() == 'woman' else 0.0
                embedding_features.append(gender_score)

            # Race contributions
            if 'race' in analysis_result and analysis_result['race'] is not None:
                race = analysis_result['race']
                if isinstance(race, dict):
                    embedding_features.extend([
                        float(race.get('white', 0)),
                        float(race.get('black', 0)),
                        float(race.get('asian', 0)),
                        float(race.get('hispanic', 0)),
                        float(race.get('middle eastern', 0))
                    ])
                else:
                    # If race is a string, create a simple encoding
                    race_str = str(race).lower()
                    race_encoding = [1.0 if race_str == ethnicity else 0.0 for ethnicity in ['white', 'black', 'asian', 'hispanic', 'middle eastern']]
                    embedding_features.extend(race_encoding)

            # Emotion contributions
            if 'emotion' in analysis_result and analysis_result['emotion'] is not None:
                emotions = analysis_result['emotion']
                if isinstance(emotions, dict):
                    embedding_features.extend([
                        float(emotions.get('happy', 0)),
                        float(emotions.get('sad', 0)),
                        float(emotions.get('angry', 0)),
                        float(emotions.get('neutral', 0))
                    ])
                else:
                    # If emotions is not a dict, add zeros
                    embedding_features.extend([0, 0, 0, 0])

        except Exception as e:
            print(f"Warning: Error processing analysis features: {e}")
            # Continue with whatever features we have so far

        # Pad or truncate to create a fixed-size embedding
        target_size = 128  # Standard face embedding size
        while len(embedding_features) < target_size:
            embedding_features.append(0.0)

        return np.array(embedding_features[:target_size])

    def search_similar_faces(self):
        """Search for similar faces in the database"""
        if not self.search_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image to search with.")
            return

        try:
            self.search_results_info.setText("🔍 Searching for similar faces...")
            self.search_results_info.setStyleSheet("QLabel { color: #E74C3C; font-weight: bold; }")
            self.search_results_table.setRowCount(0)

            # Extract real embedding from search image
            embedding, facial_area = self.extract_face_embedding(self.search_image_path)

            if embedding is None:
                QMessageBox.warning(self, "Error", "Could not extract face features from the selected image.\n\n"
                                  "This might happen if:\n"
                                  "• No face is detected in the image\n"
                                  "• The image is too blurry or dark\n"
                                  "• The image format is not supported\n\n"
                                  "Please try a different image with a clear, well-lit face.")
                return

            # Get search parameters
            threshold_text = self.search_threshold.currentText()
            threshold = float(threshold_text.split(' ')[0])

            max_results = int(self.search_max_results.currentText())

            # Search database
            similar_faces = self.database.search_similar_faces(embedding, threshold, max_results)

            if not similar_faces:
                # Check if database has any embeddings at all
                stats = self.database.get_database_stats()
                if stats['total_embeddings'] == 0:
                    self.search_results_info.setText("No face embeddings stored in database yet. Analyze some images first!")
                    self.search_results_info.setStyleSheet("QLabel { color: #F39C12; font-weight: bold; }")
                    QMessageBox.information(self, "Database Empty",
                                          "Your database doesn't contain any face embeddings yet.\n\n"
                                          "To search for similar faces, you need to:\n"
                                          "1. Go to the 'Analysis' tab\n"
                                          "2. Load and analyze some face images\n"
                                          "3. The system will automatically store the embeddings\n"
                                          "4. Come back to the 'Search' tab to find similar faces")
                else:
                    self.search_results_info.setText(f"No similar faces found (threshold: {threshold})")
                    self.search_results_info.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; }")
                return

            # Display results
            self.search_results_table.setRowCount(len(similar_faces))

            for row, result in enumerate(similar_faces):
                similarity = result['similarity']
                analysis = result['analysis']

                # Rank
                self.search_results_table.setItem(row, 0, QTableWidgetItem(f"#{row + 1}"))

                # Similarity score
                self.search_results_table.setItem(row, 1, QTableWidgetItem(f"{similarity:.4f}"))

                # Match percentage
                match_pct = similarity * 100
                match_item = QTableWidgetItem(f"{match_pct:.1f}%")
                if match_pct >= 80:
                    match_item.setForeground(QColor("#27AE60"))  # Green for high match
                elif match_pct >= 60:
                    match_item.setForeground(QColor("#F39C12"))  # Orange for medium match
                else:
                    match_item.setForeground(QColor("#E74C3C"))  # Red for low match
                self.search_results_table.setItem(row, 2, match_item)

                # User (if available)
                user_id = analysis.get('user_id')
                user_name = analysis.get('user_name', 'Anonymous')

                # Handle case where user_name is None or 'None' string (fallback)
                if not user_name or user_name == 'None' or user_name is None:
                    user_name = f"User {user_id}" if user_id else "Anonymous"

                self.search_results_table.setItem(row, 3, QTableWidgetItem(user_name))

                # Store full result data for details view
                self.search_results_table.item(row, 0).setData(Qt.UserRole, result)

            # Display similar faces visually
            self.display_similar_faces_visually(similar_faces)

            # Update results info
            if len(similar_faces) > 0:
                best_similarity = similar_faces[0]['similarity']
                best_pct = best_similarity * 100

                if best_pct >= 80:
                    status_icon = "🎉"
                    status_color = "#27AE60"
                elif best_pct >= 60:
                    status_icon = "👍"
                    status_color = "#F39C12"
                else:
                    status_icon = "🤔"
                    status_color = "#E74C3C"

                self.search_results_info.setText(
                    f"{status_icon} Found {len(similar_faces)} similar faces! Best match: {best_pct:.1f}% (threshold: {threshold})"
                )
                self.search_results_info.setStyleSheet(f"QLabel {{ color: {status_color}; font-weight: bold; }}")
            else:
                self.search_results_info.setText(f"No similar faces found (threshold: {threshold})")
                self.search_results_info.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; }")

            # Resize columns for better display
            self.search_results_table.resizeColumnsToContents()

        except Exception as e:
            self.search_results_info.setText(f"Error during search: {str(e)}")
            self.search_results_info.setStyleSheet("QLabel { color: #E74C3C; font-weight: bold; }")
            QMessageBox.warning(self, "Error", f"Search failed: {str(e)}")

    def display_similar_faces_visually(self, similar_faces):
        """Display similar faces visually in comparison format with search image"""
        try:
            # Clear existing similar faces
            for i in reversed(range(self.similar_faces_layout.count())):
                widget = self.similar_faces_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            if not similar_faces:
                no_results_label = QLabel("No similar faces found")
                no_results_label.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; text-align: center; padding: 20px; }")
                no_results_label.setAlignment(Qt.AlignCenter)
                self.similar_faces_layout.addWidget(no_results_label)
                return

            # Create main comparison layout
            main_layout = QVBoxLayout()
            main_layout.setSpacing(15)

            # Add comparison header
            comparison_header = QLabel("🔍 Search Results Comparison")
            comparison_header.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #2E86AB; text-align: center; padding: 10px; }")
            comparison_header.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(comparison_header)

            # Create horizontal layout for search image vs matches
            comparison_layout = QHBoxLayout()
            comparison_layout.setSpacing(20)

            # Left side: Search image display
            search_section = QVBoxLayout()

            search_title = QLabel("🔎 Search Image")
            search_title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #E74C3C; text-align: center; }")
            search_section.addWidget(search_title)

            # Show the search image (reuse the existing search image display)
            if hasattr(self, 'search_image_path') and self.search_image_path:
                search_image_label = QLabel()
                search_image_label.setAlignment(Qt.AlignCenter)
                search_image_label.setMinimumSize(200, 200)
                search_image_label.setMaximumSize(250, 250)
                search_image_label.setStyleSheet("""
                    QLabel {
                        background-color: #F8F9FA;
                        border: 2px solid #E74C3C;
                        border-radius: 10px;
                    }
                """)

                try:
                    pixmap = QPixmap(self.search_image_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        search_image_label.setPixmap(scaled_pixmap)
                    else:
                        search_image_label.setText("❌\nSearch\nImage")
                        search_image_label.setStyleSheet("""
                            QLabel {
                                background-color: #FFE6E6;
                                border: 2px solid #E74C3C;
                                border-radius: 10px;
                                color: #E74C3C;
                                font-size: 12px;
                                text-align: center;
                            }
                        """)
                except Exception:
                    search_image_label.setText("⚠️\nError")
                    search_image_label.setStyleSheet("""
                        QLabel {
                            background-color: #FFF5E6;
                            border: 2px solid #F39C12;
                            border-radius: 10px;
                            color: #F39C12;
                            font-size: 12px;
                            text-align: center;
                        }
                    """)

                search_section.addWidget(search_image_label)
            else:
                no_search_label = QLabel("No search image selected")
                no_search_label.setStyleSheet("QLabel { color: #95A5A6; font-style: italic; text-align: center; padding: 20px; }")
                no_search_label.setAlignment(Qt.AlignCenter)
                search_section.addWidget(no_search_label)

            search_section.addStretch()
            comparison_layout.addLayout(search_section)

            # Right side: Matched faces display
            matches_section = QVBoxLayout()

            matches_title = QLabel(f"🎯 Top {min(len(similar_faces), 5)} Matches")
            matches_title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #27AE60; text-align: center; }")
            matches_section.addWidget(matches_title)

            # Create grid layout for matched faces (max 5 for better display)
            matches_grid = QGridLayout()
            matches_grid.setSpacing(10)

            max_display = min(len(similar_faces), 5)  # Show max 5 matches for clean display

            for i in range(max_display):
                result = similar_faces[i]
                analysis = result['analysis']
                similarity = result['similarity']

                # Create match card with similarity info
                match_card = self.create_comparison_match_card(analysis, similarity, i + 1)
                matches_grid.addWidget(match_card, i // 2, i % 2)  # 2 columns

            # Fill remaining grid spaces if needed
            for i in range(max_display, 4):  # Fill up to 4 slots (2x2 grid)
                empty_label = QLabel("")
                empty_label.setMinimumSize(180, 220)
                matches_grid.addWidget(empty_label, i // 2, i % 2)

            matches_section.addLayout(matches_grid)

            # Add summary info
            if len(similar_faces) > 5:
                summary_label = QLabel(f"... and {len(similar_faces) - 5} more matches (see table below)")
                summary_label.setStyleSheet("QLabel { color: #7F8C8D; font-style: italic; text-align: center; padding: 5px; }")
                summary_label.setAlignment(Qt.AlignCenter)
                matches_section.addWidget(summary_label)

            matches_section.addStretch()
            comparison_layout.addLayout(matches_section)

            main_layout.addLayout(comparison_layout)

            # Add overall similarity info
            best_similarity = similar_faces[0]['similarity'] if similar_faces else 0
            best_pct = best_similarity * 100

            if best_pct >= 80:
                similarity_status = "🎉 Excellent match found!"
                status_color = "#27AE60"
            elif best_pct >= 60:
                similarity_status = "👍 Good potential matches"
                status_color = "#F39C12"
            else:
                similarity_status = "🤔 Limited similarity found"
                status_color = "#E74C3C"

            overall_info = QLabel(f"{similarity_status} (Best match: {best_pct:.1f}%)")
            overall_info.setStyleSheet(f"QLabel {{ font-size: 12px; font-weight: bold; color: {status_color}; text-align: center; padding: 10px; }}")
            overall_info.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(overall_info)

            self.similar_faces_layout.addLayout(main_layout)

        except Exception as e:
            error_label = QLabel(f"Error displaying comparison: {str(e)}")
            error_label.setStyleSheet("QLabel { color: #E74C3C; font-weight: bold; text-align: center; padding: 20px; }")
            error_label.setAlignment(Qt.AlignCenter)
            self.similar_faces_layout.addWidget(error_label)
            print(f"Error displaying similar faces visually: {e}")

    def create_comparison_match_card(self, analysis, similarity, rank):
        """Create a comparison card showing matched face with rank and similarity"""
        # Main card widget
        card = QFrame()
        card.setFrameStyle(QFrame.Box)
        card.setStyleSheet("""
            QFrame {
                border: 2px solid #BDC3C7;
                border-radius: 8px;
                background-color: #F8F9FA;
                padding: 5px;
            }
            QFrame:hover {
                border: 2px solid #2E86AB;
                background-color: #E8F4F8;
            }
        """)
        card.setFixedSize(180, 220)
        card.setCursor(Qt.PointingHandCursor)

        # Layout for card content
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(3)
        card_layout.setContentsMargins(5, 5, 5, 5)

        # Rank and similarity at top
        similarity_pct = similarity * 100
        header_text = f"#{rank} - {similarity_pct:.1f}%"

        if similarity_pct >= 80:
            header_color = "#27AE60"
        elif similarity_pct >= 60:
            header_color = "#F39C12"
        else:
            header_color = "#E74C3C"

        header_label = QLabel(header_text)
        header_label.setStyleSheet(f"QLabel {{ font-size: 11px; font-weight: bold; color: {header_color}; text-align: center; }}")
        header_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(header_label)

        # Image display
        image_path = analysis.get('image_path', '')
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(140, 140)
        image_label.setMaximumSize(140, 140)
        image_label.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #BDC3C7;
                border-radius: 5px;
            }
        """)

        # Load and display image
        try:
            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    image_label.setPixmap(scaled_pixmap)
                else:
                    image_label.setText("❌\nInvalid")
                    image_label.setStyleSheet("""
                        QLabel {
                            background-color: #FFE6E6;
                            border: 1px solid #E74C3C;
                            border-radius: 5px;
                            color: #E74C3C;
                            font-size: 10px;
                            text-align: center;
                        }
                    """)
            else:
                image_label.setText("📁\nNo\nImage")
                image_label.setStyleSheet("""
                    QLabel {
                        background-color: #FFF3CD;
                        border: 1px solid #FFC107;
                        border-radius: 5px;
                        color: #856404;
                        font-size: 10px;
                        text-align: center;
                    }
                """)
        except Exception as e:
            image_label.setText("⚠️\nError")
            image_label.setStyleSheet("""
                QLabel {
                    background-color: #FFF5E6;
                    border: 1px solid #F39C12;
                    border-radius: 5px;
                    color: #F39C12;
                    font-size: 10px;
                    text-align: center;
                }
            """)

        card_layout.addWidget(image_label)

        # Face info
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(35)
        info_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                font-size: 9px;
                color: #2C3E50;
            }
        """)

        # Format face info
        user_id = analysis.get('user_id')
        user_name = analysis.get('user_name', 'Anonymous')

        # Handle case where user_name is None or 'None' string (fallback)
        if not user_name or user_name == 'None' or user_name is None:
            user_name = f"User {user_id}" if user_id else "Anonymous"

        info_text.append(f"👤 {user_name}")
        info_text.append(f"📅 {analysis.get('created_at', 'Unknown')}")

        card_layout.addWidget(info_text)

        # Click handler
        def on_card_clicked():
            self.view_search_result_details_from_card({'analysis': analysis, 'similarity': similarity})

        card.mousePressEvent = lambda event: on_card_clicked()

        return card

    def view_search_result_details(self, index):
        """Handle double-click on search results table to show details"""
        try:
            # Get the clicked row
            row = index.row()
            if row < 0:
                return

            # Get the result data stored in the table item
            item = self.search_results_table.item(row, 0)
            if not item:
                return

            result = item.data(Qt.UserRole)
            if not result:
                return

            self.view_search_result_details_from_result(result)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show details: {str(e)}")

    def view_search_result_details_from_result(self, result):
        """Show detailed information about a search result"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Search Result Details")
        dialog.setModal(True)
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Image display
        image_layout = QHBoxLayout()

        analysis = result.get('analysis', {})
        image_path = analysis.get('image_path', '')
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(300, 300)
        image_label.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 2px solid #2E86AB;
                border-radius: 5px;
            }
        """)

        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
        except Exception:
            image_label.setText("Image not available")

        image_layout.addWidget(image_label)

        # Analysis details
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        details_text.setStyleSheet("""
            QTextEdit {
                background-color: #F8F9FA;
                border: 1px solid #BDC3C7;
                border-radius: 5px;
                padding: 10px;
            }
        """)

        # Format detailed information
        similarity = result.get('similarity', 0)
        similarity_pct = similarity * 100

        details_text.append(f"🔍 Search Result Details")
        details_text.append(f"{'='*50}\n")

        # Similarity score
        details_text.append(f"📊 Similarity Score: {similarity:.4f} ({similarity_pct:.1f}%)")
        if similarity_pct >= 80:
            details_text.append(f"   ✅ High similarity - likely the same person")
        elif similarity_pct >= 60:
            details_text.append(f"   ⚠️ Moderate similarity - possibly the same person")
        else:
            details_text.append(f"   ❌ Low similarity - likely different people")

        details_text.append(f"\n📁 File: {os.path.basename(image_path)}")
        details_text.append(f"📂 Path: {image_path}")
        details_text.append(f"👤 User: {analysis.get('user_name', 'Anonymous')}")
        details_text.append(f"🔧 Analysis Type: {analysis.get('analysis_type', 'Unknown')}")
        details_text.append(f"📅 Analyzed: {analysis.get('created_at', 'Unknown')}")
        details_text.append(f"⏱️ Processing Time: {analysis.get('processing_time', 'Unknown')}s")
        details_text.append(f"🤖 Model Used: {analysis.get('model_used', 'Unknown')}")
        details_text.append(f"⚙️ Detector Used: {analysis.get('detector_used', 'Unknown')}")

        # Analysis results
        result_data = analysis.get('result_data', {})
        if result_data:
            details_text.append(f"\n📊 Face Analysis Results:")
            for key, value in result_data.items():
                if isinstance(value, dict):
                    details_text.append(f"   {key.title()}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            details_text.append(f"     {sub_key.title()}: {sub_value:.1%}")
                        else:
                            details_text.append(f"     {sub_key.title()}: {sub_value}")
                else:
                    details_text.append(f"   {key.title()}: {value}")

        image_layout.addWidget(details_text)
        layout.addLayout(image_layout)

        # Action buttons
        buttons_layout = QHBoxLayout()

        # Export result button
        export_btn = QPushButton("📤 Export Result")
        export_btn.clicked.connect(lambda: self.export_search_result(result))
        buttons_layout.addWidget(export_btn)

        # View in database button
        db_btn = QPushButton("🗄️ View in Database")
        db_btn.clicked.connect(lambda: self.view_in_database(analysis, dialog))
        buttons_layout.addWidget(db_btn)

        buttons_layout.addStretch()

        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(close_btn)

        layout.addLayout(buttons_layout)

        dialog.exec_()

    def view_search_result_details_from_card(self, result):
        """Handle click on search result card to show details"""
        self.view_search_result_details_from_result(result)

    def export_search_result(self, result):
        """Export search result to JSON file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Search Result", "", "JSON Files (*.json)"
            )

            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                QMessageBox.information(self, "Success",
                                      f"Search result exported successfully to:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export result: {str(e)}")

    def view_in_database(self, analysis, parent_dialog):
        """Switch to database tab and show the analysis"""
        # Switch to database tab
        self.main_tab_widget.setCurrentIndex(1)  # Database tab index

        # Close the details dialog
        parent_dialog.accept()

        # Show the specific analysis details
        self.show_image_details(analysis)

        QMessageBox.information(self, "Database View",
                              f"Switched to Database tab. Showing details for the selected analysis.")

    def display_image(self, path, is_image1=True):
        """Display image in the appropriate display area"""
        try:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                print(f"Warning: Could not load image from {path}")
                return

            scaled_pixmap = pixmap.scaled(280, 280, Qt.KeepAspectRatio)

            if is_image1:
                self.img1_display.setPixmap(scaled_pixmap)
                self.img1_display.setText("")
                if not self.img2_path:
                    single_scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
                    self.image_display.setPixmap(single_scaled_pixmap)
                    self.image_display.setText("")
            else:
                self.img2_display.setPixmap(scaled_pixmap)
                self.img2_display.setText("")

        except Exception as e:
            print(f"Error displaying image {path}: {e}")

    def clear_image_display(self):
        """Clear all image displays"""
        self.img1_display.setText("No Image\nLoaded")
        self.img1_display.setAlignment(Qt.AlignCenter)
        self.img1_display.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 1px solid #BDC3C7;
                color: #95A5A6;
                font-size: 12px;
            }
        """)

        self.img2_display.setText("No Image\nLoaded")
        self.img2_display.setAlignment(Qt.AlignCenter)
        self.img2_display.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 1px solid #BDC3C7;
                color: #95A5A6;
                font-size: 12px;
            }
        """)

        self.image_display.setText("No Image Selected\nLoad an image to begin")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                color: #95A5A6;
                font-size: 14px;
            }
        """)

    def switch_to_dual_display(self):
        """Switch to dual image display mode"""
        self.single_image_widget.setVisible(False)
        self.dual_image_widget.setVisible(True)

    def switch_to_single_display(self):
        """Switch to single image display mode"""
        if self.img1_path:
            try:
                pixmap = QPixmap(self.img1_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
                    self.image_display.setPixmap(scaled_pixmap)
                    self.image_display.setText("")
            except Exception as e:
                print(f"Error in switch_to_single_display: {e}")
                self.image_display.setText("Error loading image")
        else:
            self.clear_image_display()
        self.dual_image_widget.setVisible(False)
        self.single_image_widget.setVisible(True)

    def update_display_mode(self):
        """Update display mode based on loaded images and function"""
        current_function = self.function_combo.currentText()

        if current_function == 'Verify Faces' and self.img1_path and self.img2_path:
            self.switch_to_dual_display()
            try:
                if self.img1_path:
                    pixmap1 = QPixmap(self.img1_path)
                    if not pixmap1.isNull():
                        scaled_pixmap1 = pixmap1.scaled(280, 280, Qt.KeepAspectRatio)
                        self.img1_display.setPixmap(scaled_pixmap1)
                        self.img1_display.setText("")
                if self.img2_path:
                    pixmap2 = QPixmap(self.img2_path)
                    if not pixmap2.isNull():
                        scaled_pixmap2 = pixmap2.scaled(280, 280, Qt.KeepAspectRatio)
                        self.img2_display.setPixmap(scaled_pixmap2)
                        self.img2_display.setText("")
            except Exception as e:
                print(f"Warning: Could not load images for dual display: {e}")
                self.switch_to_single_display()
        else:
            self.switch_to_single_display()

    def on_function_changed(self):
        """Handle function selection change"""
        QTimer.singleShot(100, self.update_display_mode)

    # Database integration methods
    def refresh_database_stats(self):
        """Refresh database statistics display"""
        try:
            stats = self.database.get_database_stats()

            self.db_info_text.clear()
            self.db_info_text.append("📊 Database Statistics:\n")
            self.db_info_text.append(f"• Total Users: {stats['total_users']:,}")
            self.db_info_text.append(f"• Total Analyses: {stats['total_analyses']:,}")
            self.db_info_text.append(f"• Total Face Embeddings: {stats['total_embeddings']:,}")
            self.db_info_text.append(f"• Total Verifications: {stats['total_verifications']:,}")
            self.db_info_text.append(f"• Database Size: {stats['database_size_mb']:.2f} MB")
            self.db_info_text.append(f"• Recent Activity (7 days): {stats['recent_analyses_7days']:,}")

            # Add embedding status
            if stats['total_embeddings'] == 0:
                self.db_info_text.append(f"\n⚠️ No face embeddings stored yet!")
                self.db_info_text.append(f"   Analyze some face images to enable search functionality.")
            else:
                self.db_info_text.append(f"\n✅ Face search ready with {stats['total_embeddings']} embeddings!")

        except Exception as e:
            self.db_info_text.clear()
            self.db_info_text.append(f"❌ Error loading statistics: {str(e)}")

    def refresh_recent_analyses(self):
        """Refresh recent analyses table"""
        try:
            self.recent_analyses_table.setRowCount(0)

            # Get recent analyses from database
            analyses = self.database.get_all_analyses(20)  # Get last 20 analyses

            if not analyses:
                # Show placeholder if no analyses
                self.recent_analyses_table.insertRow(0)
                self.recent_analyses_table.setItem(0, 0, QTableWidgetItem("No analyses yet"))
                self.recent_analyses_table.setItem(0, 1, QTableWidgetItem("Analyze some images first"))
                self.recent_analyses_table.setItem(0, 2, QTableWidgetItem("Database"))
                self.recent_analyses_table.setItem(0, 3, QTableWidgetItem("Integrated"))
                self.recent_analyses_table.setItem(0, 4, QTableWidgetItem("Ready"))
                return

            for row, analysis in enumerate(analyses):
                self.recent_analyses_table.insertRow(row)

                # Analysis ID
                self.recent_analyses_table.setItem(row, 0, QTableWidgetItem(str(analysis['id'])))

                # User name
                user_name = analysis.get('user_name', 'Anonymous')
                self.recent_analyses_table.setItem(row, 1, QTableWidgetItem(user_name))

                # Analysis type
                analysis_type = analysis.get('analysis_type', 'Unknown').title()
                self.recent_analyses_table.setItem(row, 2, QTableWidgetItem(analysis_type))

                # Image filename
                image_path = analysis.get('image_path', 'Unknown')
                self.recent_analyses_table.setItem(row, 3, QTableWidgetItem(os.path.basename(image_path)))

                # Date
                created_at = analysis.get('created_at', 'Unknown')
                if created_at != 'Unknown':
                    try:
                        date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        created_at = date_obj.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                self.recent_analyses_table.setItem(row, 4, QTableWidgetItem(created_at))

        except Exception as e:
            print(f"Warning: Failed to refresh recent analyses: {e}")
            # Show error placeholder
            self.recent_analyses_table.setRowCount(0)
            self.recent_analyses_table.insertRow(0)
            self.recent_analyses_table.setItem(0, 0, QTableWidgetItem("Error"))
            self.recent_analyses_table.setItem(0, 1, QTableWidgetItem("Failed to load"))
            self.recent_analyses_table.setItem(0, 2, QTableWidgetItem("Database"))
            self.recent_analyses_table.setItem(0, 3, QTableWidgetItem("Check logs"))
            self.recent_analyses_table.setItem(0, 4, QTableWidgetItem("Error"))

    def set_current_user(self):
        """Set current user for analysis tracking"""
        user_name, ok = QInputDialog.getText(self, 'Set Current User',
                                           'Enter your name (or leave blank for anonymous):')

        if ok:
            if user_name.strip():
                try:
                    self.current_user_id = self.database.add_user(user_name.strip())
                    self.current_user_label.setText(f"Current User: {user_name} (ID: {self.current_user_id})")
                    self.statusBar().showMessage(f"Current user set to: {user_name} (ID: {self.current_user_id})")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to set user: {str(e)}")
            else:
                self.current_user_id = None
                self.current_user_label.setText("Current User: Anonymous")
                self.statusBar().showMessage("Current user: Anonymous")

    def open_database_manager(self):
        """Open database manager window"""
        try:
            from database_manager import DatabaseManager
            self.db_manager = DatabaseManager()
            self.db_manager.show()
        except ImportError:
            QMessageBox.warning(self, "Error", "Database manager module not found")

    def export_database(self):
        """Export database to JSON file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Database", "", "JSON Files (*.json)"
            )

            if file_path:
                exported_path = self.database.export_data(file_path, "all")
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
                self.database.clear_database(confirm=True)
                self.refresh_database_stats()
                self.refresh_recent_analyses()
                QMessageBox.information(self, "Success", "Database cleared successfully")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to clear database: {str(e)}")

    def optimize_database(self):
        """Optimize database (VACUUM)"""
        try:
            import sqlite3
            with sqlite3.connect(self.database.db_path) as conn:
                conn.execute("VACUUM")
                conn.commit()

            self.refresh_database_stats()
            QMessageBox.information(self, "Success", "Database optimized successfully")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to optimize database: {str(e)}")

    def export_results(self):
        """Export current results to file"""
        try:
            if not hasattr(self, 'last_result') or not self.last_result:
                QMessageBox.information(self, "Info", "No results to export. Run an analysis first.")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", "JSON Files (*.json)"
            )

            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(self.last_result, f, indent=2, default=str)

                QMessageBox.information(self, "Success",
                                      f"Results exported successfully to:\n{file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export results: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About DeepFace GUI",
                         "DeepFace GUI - Enhanced Face Analysis Tool\n\n"
                         "Features:\n"
                         "• Face Verification\n"
                         "• Face Analysis (Age, Gender, Emotion, Race)\n"
                         "• Image Gallery - Visual Database Browser\n"
                         "• Face Search & Comparison\n"
                         "• Database Management\n"
                         "• Progress Tracking\n"
                         "• Rich Result Display\n"
                         "• Dual Image Display\n"

                         "• Internet Image Search\n\n"
                         "Built with PyQt5 and DeepFace\n"
                         "Database powered by SQLite\n"
                         "Web browsing powered by PyQtWebEngine")

    # Analysis execution methods
    def execute_function(self):
        function = self.function_combo.currentText()
        if not self.img1_path:
            QMessageBox.warning(self, 'Warning', 'Please load Image 1.')
            return

        try:
            total_steps = 5
            current_step = 0

            self.update_progress(current_step, total_steps, "Initializing...")

            if function == 'Verify Faces':
                if not self.img2_path:
                    QMessageBox.warning(self, 'Warning', 'Please load Image 2 for verification.')
                    self.show_progress(False)
                    return

                current_step += 1
                self.update_progress(current_step, total_steps, "Loading images...")
                QTimer.singleShot(500, lambda: self._continue_verification(current_step, total_steps))

            elif function == 'Analyze Face':
                current_step += 1
                self.update_progress(current_step, total_steps, "Loading image...")
                QTimer.singleShot(500, lambda: self._continue_analysis(current_step, total_steps))

        except Exception as e:
            self.display_error(str(e))
            self.show_progress(False)

    def _continue_verification(self, current_step, total_steps):
        """Continue verification process with progress updates"""
        try:
            current_step += 1
            self.update_progress(current_step, total_steps, "Detecting faces...")

            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                current_step += 1
                self.update_progress(current_step, total_steps, "Comparing faces...")

                result = DeepFace.verify(self.img1_path, self.img2_path, enforce_detection=False)

                current_step += 1
                self.update_progress(current_step, total_steps, "Processing results...")

                self.format_verification_result(result)
                self.save_analysis_to_database(result, 'verify')
                self.statusBar().showMessage(f"Verification completed - Distance: {result.get('distance', 0):.4f}")

        except Exception as e:
            self.display_error(str(e))
        finally:
            self.show_progress(False)

    def _continue_analysis(self, current_step, total_steps):
        """Continue analysis process with progress updates"""
        try:
            current_step += 1
            self.update_progress(current_step, total_steps, "Detecting face...")

            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                current_step += 1
                self.update_progress(current_step, total_steps, "Analyzing features...")

                result = DeepFace.analyze(self.img1_path, enforce_detection=False)

                current_step += 1
                self.update_progress(current_step, total_steps, "Processing results...")

                self.format_analysis_result(result)
                self.save_analysis_to_database(result, 'analyze')
                self.statusBar().showMessage("Face analysis completed")

        except Exception as e:
            self.display_error(str(e))
        finally:
            self.show_progress(False)

    def save_analysis_to_database(self, result, analysis_type):
        """Save analysis results to database"""
        try:
            def convert_numpy_types(obj):
                """Convert NumPy types to JSON-serializable types"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, np.bool8)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            cleaned_result = convert_numpy_types(result)

            analysis_id = self.database.save_analysis(
                user_id=self.current_user_id,
                image_path=self.img1_path,
                analysis_type=analysis_type,
                result_data=cleaned_result,
                processing_time=cleaned_result.get('time', 0) if isinstance(cleaned_result, dict) else 0,
                model_used=cleaned_result.get('model', 'Unknown') if isinstance(cleaned_result, dict) else 'Unknown',
                detector_used=cleaned_result.get('detector_backend', 'Unknown') if isinstance(cleaned_result, dict) else 'Unknown'
            )

            # Extract and save face embedding for search functionality
            try:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    # Extract embedding using DeepFace
                    embedding_result = DeepFace.represent(self.img1_path, enforce_detection=False, model_name="Facenet")

                    if embedding_result and len(embedding_result) > 0:
                        # Get the first face embedding
                        embedding_data = embedding_result[0]['embedding']
                        facial_area = embedding_result[0]['facial_area']

                        # Save embedding to database
                        self.database.save_embedding(
                            analysis_id=analysis_id,
                            embedding_data=np.array(embedding_data),
                            face_location=facial_area
                        )

                        print(f"✅ Saved embedding for analysis ID: {analysis_id}")

            except Exception as e:
                print(f"Warning: Could not save embedding: {e}")
                # Continue even if embedding save fails

            if analysis_type == 'verify' and self.img2_path:
                verified = cleaned_result.get('verified', False) if isinstance(cleaned_result, dict) else False
                distance = cleaned_result.get('distance', 0) if isinstance(cleaned_result, dict) else 0

                analysis_id2 = self.database.save_analysis(
                    user_id=self.current_user_id,
                    image_path=self.img2_path,
                    analysis_type='verify',
                    result_data={'placeholder': 'second_image'},
                    processing_time=0
                )

                # Also save embedding for second image
                try:
                    with contextlib.redirect_stderr(open(os.devnull, 'w')):
                        embedding_result2 = DeepFace.represent(self.img2_path, enforce_detection=False, model_name="Facenet")

                        if embedding_result2 and len(embedding_result2) > 0:
                            embedding_data2 = embedding_result2[0]['embedding']
                            facial_area2 = embedding_result2[0]['facial_area']

                            self.database.save_embedding(
                                analysis_id=analysis_id2,
                                embedding_data=np.array(embedding_data2),
                                face_location=facial_area2
                            )

                except Exception as e:
                    print(f"Warning: Could not save embedding for second image: {e}")

                self.database.save_verification(
                    image1_id=analysis_id,
                    image2_id=analysis_id2,
                    similarity_score=distance,
                    verified=verified,
                    threshold_used=cleaned_result.get('threshold', 0.6) if isinstance(cleaned_result, dict) else 0.6,
                    model_used=cleaned_result.get('model', 'Unknown') if isinstance(cleaned_result, dict) else 'Unknown',
                    detector_used=cleaned_result.get('detector_backend', 'Unknown') if isinstance(cleaned_result, dict) else 'Unknown'
                )

            self.last_result = cleaned_result
            self.statusBar().showMessage(f"Analysis saved to database (ID: {analysis_id})")

        except Exception as e:
            print(f"Warning: Failed to save to database: {e}")
            self.statusBar().showMessage("Analysis completed (database save failed)")

    # Progress and UI methods
    def show_progress(self, show=True, text="Processing...", progress=0, max_progress=100):
        """Show or hide progress bar with custom text and progress"""
        self.progress_bar.setVisible(show)
        if show:
            if max_progress > 0:
                self.progress_bar.setRange(0, max_progress)
                self.progress_bar.setValue(progress)
                self.progress_bar.setFormat(f"{text} - {progress}%")
            else:
                self.progress_bar.setRange(0, 0)
                self.progress_bar.setFormat(text)
            self.execute_btn.setEnabled(False)
            if progress > 0:
                self.execute_btn.setText(f"Processing... {progress}%")
            else:
                self.execute_btn.setText(text)
            self.statusBar().showMessage(f"Processing: {text}")
        else:
            self.progress_bar.setVisible(False)
            self.execute_btn.setEnabled(True)
            self.execute_btn.setText("Execute Analysis")
            self.statusBar().showMessage("Ready")

    def update_progress(self, current_step, total_steps, step_name):
        """Update progress bar with current step"""
        progress = int((current_step / total_steps) * 100)
        self.show_progress(True, step_name, progress, 100)
        self.statusBar().showMessage(f"Step {current_step}/{total_steps}: {step_name} ({progress}%)")

    def clear_results(self):
        """Clear the results display"""
        self.result_text.clear()
        self.statusBar().showMessage("Results cleared")

    def clear_all(self):
        """Clear all displays and reset state"""
        self.img1_path = None
        self.img2_path = None
        self.internet_image_path = None
        self.img1_label.setText('Image 1: None')
        self.img2_label.setText('Image 2: None')
        self.clear_image_display()
        self.clear_results()
        # Clear internet search results safely - only if widgets exist
        try:
            if hasattr(self, 'clear_internet_results'):
                self.clear_internet_results()
        except Exception as e:
            print(f"Warning: Could not clear internet results: {e}")
        self.switch_to_single_display()
        self.update_display_mode()

    # Result formatting methods
    def format_verification_result(self, result):
        """Format verification results with colors and structure"""
        cursor = self.result_text.textCursor()
        doc = self.result_text.document()

        cursor.select(QTextCursor.Document)
        cursor.removeSelectedText()
        cursor.movePosition(QTextCursor.Start)

        title_format = QTextCharFormat()
        title_format.setFontWeight(QFont.Bold)
        title_format.setFontPointSize(14)
        title_format.setForeground(QColor("#2E86AB"))

        cursor.insertText("🔍 Face Verification Results\n", title_format)
        cursor.insertText("=" * 50 + "\n\n")

        verified = result.get('verified', False)
        distance = result.get('distance', 0)
        threshold = result.get('threshold', 0.6)
        model = result.get('model', 'Unknown')
        detector = result.get('detector_backend', 'Unknown')
        time = result.get('time', 0)

        status_format = QTextCharFormat()
        if verified:
            status_format.setForeground(QColor("#27AE60"))
            status_icon = "✅"
            status_text = "VERIFIED"
        else:
            status_format.setForeground(QColor("#E74C3C"))
            status_icon = "❌"
            status_text = "NOT VERIFIED"

        cursor.insertText(f"{status_icon} Status: ", title_format)
        cursor.insertText(f"{status_text}\n", status_format)

        details_format = QTextCharFormat()
        details_format.setFontWeight(QFont.Bold)
        cursor.insertText("\n📊 Details:\n", details_format)

        distance_format = QTextCharFormat()
        if distance < threshold:
            distance_format.setForeground(QColor("#27AE60"))
        else:
            distance_format.setForeground(QColor("#E74C3C"))

        cursor.insertText(f"   Distance: {distance:.4f}\n", details_format)
        cursor.insertText(f"   Threshold: {threshold:.4f}\n", details_format)
        cursor.insertText(f"   Model: {model}\n", details_format)
        cursor.insertText(f"   Detector: {detector}\n", details_format)
        cursor.insertText(f"   Processing Time: {time:.2f}s\n", details_format)

        if 'facial_areas' in result:
            cursor.insertText(f"   Facial Areas Detected: {len(result['facial_areas'])}\n", details_format)

        cursor.insertText(f"\n{'='*50}\n")

        if verified:
            cursor.insertText("🎉 The faces match! This appears to be the same person.\n")
        else:
            cursor.insertText("🤔 The faces don't match closely enough. They might be different people.\n")

    def format_analysis_result(self, result):
        """Format analysis results with structured display"""
        cursor = self.result_text.textCursor()
        doc = self.result_text.document()

        cursor.select(QTextCursor.Document)
        cursor.removeSelectedText()
        cursor.movePosition(QTextCursor.Start)

        title_format = QTextCharFormat()
        title_format.setFontWeight(QFont.Bold)
        title_format.setFontPointSize(14)
        title_format.setForeground(QColor("#2E86AB"))

        cursor.insertText("🧑 Face Analysis Results\n", title_format)
        cursor.insertText("=" * 50 + "\n\n")

        if isinstance(result, list):
            result = result[0]

        details_format = QTextCharFormat()
        details_format.setFontWeight(QFont.Bold)
        cursor.insertText("👤 Basic Information:\n", details_format)

        if 'age' in result:
            cursor.insertText(f"   Age: {result['age']} years\n", details_format)

        if 'gender' in result:
            gender = result['gender']
            cursor.insertText(f"   Gender: {gender}\n", details_format)

        if 'race' in result:
            cursor.insertText("   Ethnicity:\n", details_format)
            race = result['race']
            for ethnicity, confidence in race.items():
                cursor.insertText(f"     {ethnicity.title()}: {confidence:.1%}\n", details_format)

        if 'emotion' in result:
            cursor.insertText("\n😊 Emotions:\n", details_format)
            emotions = result['emotion']
            for emotion, confidence in emotions.items():
                cursor.insertText(f"   {emotion.title()}: {confidence:.1%}\n", details_format)

        cursor.insertText(f"\n{'='*50}\n")

        if 'detector_backend' in result:
            cursor.insertText(f"🔧 Detector: {result['detector_backend']}\n", details_format)
        if 'model' in result:
            cursor.insertText(f"🤖 Model: {result['model']}\n", details_format)

    def display_error(self, error_message):
        """Display error messages with red formatting"""
        cursor = self.result_text.textCursor()
        doc = self.result_text.document()

        cursor.select(QTextCursor.Document)
        cursor.removeSelectedText()
        cursor.movePosition(QTextCursor.Start)

        error_format = QTextCharFormat()
        error_format.setFontWeight(QFont.Bold)
        error_format.setFontPointSize(14)
        error_format.setForeground(QColor("#E74C3C"))

        cursor.insertText("❌ Error Occurred\n", error_format)
        cursor.insertText("=" * 50 + "\n\n")
        cursor.insertText(f"Error: {error_message}\n\n")

        cursor.insertText("💡 Suggestions:\n", error_format)
        cursor.insertText("• Ensure the image contains a clear face\n")
        cursor.insertText("• Check image format (JPG, PNG supported)\n")
        cursor.insertText("• Try a different image or angle\n")
        cursor.insertText("• Verify all required packages are installed\n")

        self.statusBar().showMessage("Error occurred - check results for details")

    def load_image1(self):
        """Load first image for analysis"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)", options=options
        )
        if file_path:
            self.img1_path = file_path
            self.img1_label.setText(f'Image 1: {os.path.basename(file_path)}')
            self.display_image(file_path, is_image1=True)
            QTimer.singleShot(100, self.update_display_mode)

    def load_image2(self):
        """Load second image for verification"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 2", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)", options=options
        )
        if file_path:
            self.img2_path = file_path
            self.img2_label.setText(f'Image 2: {os.path.basename(file_path)}')
            self.display_image(file_path, is_image1=False)
            QTimer.singleShot(100, self.update_display_mode)

if __name__ == '__main__':
    print("Launching GUI...")

    # Initialize WebEngine before QApplication to avoid deprecation warning
    if HAS_WEBENGINE:
        QtWebEngine.initialize()

    # Final suppression during GUI startup
    with contextlib.redirect_stderr(io.StringIO()):
        app = QApplication(sys.argv)
        window = DeepFaceGUI()
        window.show()
        print("GUI ready!")
        sys.exit(app.exec_())
