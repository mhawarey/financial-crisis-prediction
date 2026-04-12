import sys
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QTabWidget, QRadioButton, 
                             QCheckBox, QSlider, QGridLayout, QGroupBox, QProgressBar,
                             QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
                             QStatusBar, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import custom modules
from data_collector import DataCollector
from model_manager import ModelManager
from risk_calculator import RiskCalculator
from visualization import create_risk_gauge, create_time_series_chart

class FinancialCrisisPredictionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Global Financial Crisis Prediction System - By Dr. Mosab Hawarey")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style
        self.set_application_style()
        
        # Initialize components
        self.data_collector = DataCollector()
        self.model_manager = ModelManager()
        self.risk_calculator = RiskCalculator()
        
        # Set up the user interface
        self.setup_ui()
        
        # Connect signals and slots
        self.connect_signals_slots()
        
        self.statusBar().showMessage("Welcome to the Financial Crisis Prediction System - By Dr. Mosab Hawarey")

    def set_application_style(self):
        # Set the color scheme (magenta-based theme)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
        palette.setColor(QPalette.Button, QColor(212, 27, 96))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.Highlight, QColor(212, 27, 96).lighter())
        self.setPalette(palette)
        
        # Set default font
        font = QFont("Arial", 10)
        QApplication.setFont(font)
    
    def setup_ui(self):
        # Create the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create the left control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Create the main content area
        content_area = self.create_content_area()
        main_layout.addWidget(content_area, 3)
        
        # Add application footer
        self.statusBar().setStyleSheet("background-color: #4a148c; color: white;")
        
        # Add copyright info
        copyright_label = QLabel("All Copyrights Reserved: MOSAB.HAWAREY.ORG © 2025")
        copyright_label.setStyleSheet("color: white;")
        self.statusBar().addPermanentWidget(copyright_label)
    
    def create_control_panel(self):
        # Main container for control panel
        control_panel = QWidget()
        control_panel.setMinimumWidth(300) # Set minimum width
        control_panel.setMaximumWidth(400) # Increased maximum width
        control_panel.setStyleSheet("background-color: #f5f5f5; border-right: 1px solid #dddddd;")
        
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title label
        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        control_layout.addWidget(title)
        
        # Model configuration group
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        # Radio buttons for model selection
        self.rb_ensemble = QRadioButton("Full Ensemble")
        self.rb_ensemble.setChecked(True)
        self.rb_kalman = QRadioButton("Extended Kalman Filter")
        self.rb_neural = QRadioButton("Neural Network Models")
        self.rb_bayesian = QRadioButton("Bayesian Models")
        self.rb_custom = QRadioButton("Custom Configuration")
        
        model_layout.addWidget(self.rb_ensemble)
        model_layout.addWidget(self.rb_kalman)
        model_layout.addWidget(self.rb_neural)
        model_layout.addWidget(self.rb_bayesian)
        model_layout.addWidget(self.rb_custom)
        
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)
        
        # Data sources group
        data_group = QGroupBox("Data Sources")
        data_layout = QVBoxLayout()
        
        # Checkboxes for data sources
        self.cb_market = QCheckBox("Market Indicators")
        self.cb_market.setChecked(True)
        self.cb_economic = QCheckBox("Economic Data")
        self.cb_economic.setChecked(True)
        self.cb_news = QCheckBox("News & Social Media")
        self.cb_news.setChecked(True)
        self.cb_geopolitical = QCheckBox("Geopolitical Events")
        self.cb_alternative = QCheckBox("Alternative Data")
        
        data_layout.addWidget(self.cb_market)
        data_layout.addWidget(self.cb_economic)
        data_layout.addWidget(self.cb_news)
        data_layout.addWidget(self.cb_geopolitical)
        data_layout.addWidget(self.cb_alternative)
        
        data_group.setLayout(data_layout)
        control_layout.addWidget(data_group)
        
        # Prediction timeframe group
        timeframe_group = QGroupBox("Prediction Timeframe")
        timeframe_layout = QVBoxLayout()
        
        timeframe_slider_layout = QHBoxLayout()
        self.timeframe_slider = QSlider(Qt.Horizontal)
        self.timeframe_slider.setMinimum(1)
        self.timeframe_slider.setMaximum(36)
        self.timeframe_slider.setValue(6)
        self.timeframe_slider.setTickPosition(QSlider.TicksBelow)
        self.timeframe_slider.setTickInterval(6)
        
        timeframe_slider_layout.addWidget(self.timeframe_slider)
        
        timeframe_label_layout = QHBoxLayout()
        timeframe_label_layout.addWidget(QLabel("3m"))
        timeframe_label_layout.addStretch()
        timeframe_label_layout.addWidget(QLabel("1y"))
        timeframe_label_layout.addStretch()
        timeframe_label_layout.addWidget(QLabel("3y"))
        
        self.selected_timeframe = QLabel("Selected: 6 months")
        self.selected_timeframe.setAlignment(Qt.AlignCenter)
        
        timeframe_layout.addLayout(timeframe_slider_layout)
        timeframe_layout.addLayout(timeframe_label_layout)
        timeframe_layout.addWidget(self.selected_timeframe)
        
        timeframe_group.setLayout(timeframe_layout)
        control_layout.addWidget(timeframe_group)
        
        # Add spacer
        control_layout.addStretch()
        
        # Run analysis button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #d81b60;
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c2185b;
            }
            QPushButton:pressed {
                background-color: #ad1457;
            }
        """)
        control_layout.addWidget(self.run_button)
        
        return control_panel
    
    def create_content_area(self):
        # Main content widget
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: white;")
        
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #d81b60;
                color: white;
            }
        """)
        
        # Create dashboard tab
        dashboard_tab = QWidget()
        self.create_dashboard_tab(dashboard_tab)
        self.tabs.addTab(dashboard_tab, "Dashboard")
        
        # Create risk analysis tab
        risk_tab = QWidget()
        self.create_risk_analysis_tab(risk_tab)
        self.tabs.addTab(risk_tab, "Risk Analysis")
        
        # Create model insights tab
        model_tab = QWidget()
        self.create_model_insights_tab(model_tab)
        self.tabs.addTab(model_tab, "Model Insights")
        
        # Create reports tab
        reports_tab = QWidget()
        self.create_reports_tab(reports_tab)
        self.tabs.addTab(reports_tab, "Reports")
        
        # Create settings tab
        settings_tab = QWidget()
        self.create_settings_tab(settings_tab)
        self.tabs.addTab(settings_tab, "Settings")
        
        content_layout.addWidget(self.tabs)
        
        return content_widget
    
    def create_dashboard_tab(self, tab):
        layout = QGridLayout(tab)
        
        # Create risk indicator widget
        risk_indicator = QGroupBox("Global Financial Risk Indicator")
        risk_indicator_layout = QVBoxLayout()
        
        # Create a placeholder for the risk gauge
        self.risk_gauge_widget = pg.PlotWidget()
        self.risk_gauge = create_risk_gauge(self.risk_gauge_widget)
        
        risk_indicator_layout.addWidget(self.risk_gauge_widget)
        risk_indicator.setLayout(risk_indicator_layout)
        
        # Create trends chart widget
        trends_chart = QGroupBox("Historical Risk Trends")
        trends_chart_layout = QVBoxLayout()
        
        # Create a placeholder for the time series chart
        self.trends_chart_widget = pg.PlotWidget()
        self.trends_chart = create_time_series_chart(self.trends_chart_widget)
        
        trends_chart_layout.addWidget(self.trends_chart_widget)
        trends_chart.setLayout(trends_chart_layout)
        
        # Create risk factors table widget
        risk_factors = QGroupBox("Key Risk Factors")
        risk_factors_layout = QVBoxLayout()
        
        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(3)
        self.risk_table.setHorizontalHeaderLabels(["Factor", "Contribution", "Trend"])
        self.risk_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.risk_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.risk_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # Add sample data (will be replaced with real data)
        self.populate_risk_table_sample_data()
        
        risk_factors_layout.addWidget(self.risk_table)
        risk_factors.setLayout(risk_factors_layout)
        
        # Create news feed widget
        news_feed = QGroupBox("Latest Economic News")
        news_layout = QVBoxLayout()
        
        self.news_table = QTableWidget()
        self.news_table.setColumnCount(2)
        self.news_table.setHorizontalHeaderLabels(["Source", "Headline"])
        self.news_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.news_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        # Add sample news (will be replaced with real data)
        self.populate_news_table_sample_data()
        
        news_layout.addWidget(self.news_table)
        news_feed.setLayout(news_layout)
        
        # Add widgets to layout
        layout.addWidget(risk_indicator, 0, 0)
        layout.addWidget(risk_factors, 0, 1, 2, 1)
        layout.addWidget(trends_chart, 1, 0)
        layout.addWidget(news_feed, 2, 0, 1, 2)
    
    def create_risk_analysis_tab(self, tab):
        # This will be implemented with actual components
        layout = QVBoxLayout(tab)
        label = QLabel("Risk Analysis Components will be displayed here.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
    
    def create_model_insights_tab(self, tab):
        # This will be implemented with actual components
        layout = QVBoxLayout(tab)
        label = QLabel("Model Performance and Insights will be displayed here.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
    
    def create_reports_tab(self, tab):
        # This will be implemented with actual components
        layout = QVBoxLayout(tab)
        label = QLabel("Reports Generation will be available here.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
    
    def create_settings_tab(self, tab):
        # This will be implemented with actual components
        layout = QVBoxLayout(tab)
        label = QLabel("Settings and Configuration will be available here.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
    
    def populate_risk_table_sample_data(self):
        # Sample data - will be replaced with real data from models
        risk_factors = [
            ("Yield Curve Inversion", "+23.4%", "↑"),
            ("Credit Spreads", "+18.7%", "↑"),
            ("Market Volatility", "+15.2%", "↓"),
            ("Corporate Debt Levels", "+12.8%", "↑"),
            ("Banking Sector Health", "-8.3%", "↓"),
            ("Geopolitical Tension", "+10.5%", "↑"),
            ("Housing Market", "+7.2%", "↑")
        ]
        
        self.risk_table.setRowCount(len(risk_factors))
        
        for i, (factor, contribution, trend) in enumerate(risk_factors):
            self.risk_table.setItem(i, 0, QTableWidgetItem(factor))
            
            contrib_item = QTableWidgetItem(contribution)
            if contribution.startswith("+"):
                contrib_item.setForeground(QColor("#d32f2f"))  # Red for positive (bad)
            else:
                contrib_item.setForeground(QColor("#1b5e20"))  # Green for negative (good)
            self.risk_table.setItem(i, 1, contrib_item)
            
            trend_item = QTableWidgetItem(trend)
            if trend == "↑":
                trend_item.setForeground(QColor("#d32f2f"))  # Red for upward trend (bad)
            else:
                trend_item.setForeground(QColor("#1b5e20"))  # Green for downward trend (good)
            self.risk_table.setItem(i, 2, trend_item)
    
    def populate_news_table_sample_data(self):
        # Sample data - will be replaced with real news from APIs
        news_items = [
            ("[Financial Times]", "Central Bank signals shift in monetary policy stance"),
            ("[Bloomberg]", "Global manufacturing index shows unexpected growth"),
            ("[Reuters]", "Corporate bond defaults rise in emerging markets"),
            ("[WSJ]", "Housing market shows signs of cooling in major metros"),
            ("[CNBC]", "Tech sector leads market rally amid economic uncertainty")
        ]
        
        self.news_table.setRowCount(len(news_items))
        
        for i, (source, headline) in enumerate(news_items):
            source_item = QTableWidgetItem(source)
            if "[Financial Times]" in source or "[Reuters]" in source:
                source_item.setForeground(QColor("#d32f2f"))  # Red for negative news
            elif "[Bloomberg]" in source or "[CNBC]" in source:
                source_item.setForeground(QColor("#1b5e20"))  # Green for positive news
            else:
                source_item.setForeground(QColor("#0d47a1"))  # Blue for neutral news
            
            self.news_table.setItem(i, 0, source_item)
            self.news_table.setItem(i, 1, QTableWidgetItem(headline))
    
    def connect_signals_slots(self):
        # Connect timeframe slider to label update
        self.timeframe_slider.valueChanged.connect(self.update_timeframe_label)
        
        # Connect run button to analysis function
        self.run_button.clicked.connect(self.run_analysis)
    
    def update_timeframe_label(self):
        months = self.timeframe_slider.value()
        if months < 12:
            self.selected_timeframe.setText(f"Selected: {months} months")
        else:
            years = months // 12
            remaining_months = months % 12
            if remaining_months == 0:
                self.selected_timeframe.setText(f"Selected: {years} years")
            else:
                self.selected_timeframe.setText(f"Selected: {years} years, {remaining_months} months")
    
    def run_analysis(self):
        # This will be implemented to run the actual analysis
        self.statusBar().showMessage("Running analysis...")
        
        # In the full implementation, this would:
        # 1. Collect data based on selected sources
        # 2. Run selected models
        # 3. Calculate risk metrics
        # 4. Update visualizations
        
        # For the trial version, we'll simulate this process
        QApplication.processEvents()
        
        # Simulated delay
        import time
        time.sleep(1)
        
        # Update with sample/simulated results
        self.update_with_sample_results()
        
        self.statusBar().showMessage("Analysis completed successfully")
    
    def update_with_sample_results(self):
        # Update risk gauge with a sample value
        self.risk_gauge.setData([68])  # Sample risk level
        
        # In the full implementation, all visualizations would be updated with real data
        # For now, we're using the sample data already loaded


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FinancialCrisisPredictionSystem()
    window.show()
    sys.exit(app.exec_())