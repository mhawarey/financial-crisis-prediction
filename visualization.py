import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Wedge
import pyqtgraph as pg
from PyQt5.QtGui import QColor
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("visualization.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Visualization")

def create_risk_gauge(plot_widget, risk_level=50):
    """
    Create a risk gauge visualization in PyQtGraph.
    
    Parameters:
    - plot_widget: PyQtGraph PlotWidget
    - risk_level: Current risk level (0-100)
    
    Returns:
    - plot item for updating later
    """
    try:
        logger.info(f"Creating risk gauge with level {risk_level}")
        
        # Clear any existing plots
        plot_widget.clear()
        
        # Configure the plot
        plot_widget.setAspectLocked(True)
        plot_widget.hideAxis('left')
        plot_widget.hideAxis('bottom')
        plot_widget.setBackground('w')
        
        # Use a simpler approach for the gauge - draw arc segments as lines
        theta = np.linspace(-np.pi, 0, 100)
        radius = 0.8
        
        # Draw background arc
        bg_curve = pg.PlotCurveItem(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),
            pen=pg.mkPen(QColor(220, 220, 220), width=10)
        )
        plot_widget.addItem(bg_curve)
        
        # Create colored segments
        segment_ranges = [
            (0, 30, QColor(50, 200, 50)),      # Green
            (30, 50, QColor(180, 180, 50)),    # Yellow
            (50, 70, QColor(255, 150, 50)),    # Orange
            (70, 85, QColor(250, 100, 50)),    # Orange-Red
            (85, 100, QColor(250, 50, 50))     # Red
        ]
        
        for start, end, color in segment_ranges:
            start_angle = -np.pi + (start / 100) * np.pi
            end_angle = -np.pi + (end / 100) * np.pi
            segment_theta = np.linspace(start_angle, end_angle, 30)
            
            segment_curve = pg.PlotCurveItem(
                x=radius * np.cos(segment_theta),
                y=radius * np.sin(segment_theta),
                pen=pg.mkPen(color, width=10)
            )
            plot_widget.addItem(segment_curve)
        
        # Create pointer
        pointer_angle = -np.pi + (risk_level / 100) * np.pi
        x_pointer = [0, 0.7 * np.cos(pointer_angle)]
        y_pointer = [0, 0.7 * np.sin(pointer_angle)]
        
        pointer = pg.PlotCurveItem(x=x_pointer, y=y_pointer, pen=pg.mkPen('k', width=3))
        plot_widget.addItem(pointer)
        
        # Add center circle
        center_circle = pg.ScatterPlotItem()
        center_circle.addPoints([0], [0], size=10, pen=pg.mkPen(None), brush=QColor(80, 80, 80))
        plot_widget.addItem(center_circle)
        
        # Add risk percentage text
        text = pg.TextItem(f"{risk_level}%", anchor=(0.5, 0.5), color=QColor(50, 50, 50))
        text.setPos(0, -0.3)
        plot_widget.addItem(text)
        
        # Add risk level label
        risk_label = "Low"
        if risk_level >= 30 and risk_level < 50:
            risk_label = "Moderate"
        elif risk_level >= 50 and risk_level < 70:
            risk_label = "Elevated"
        elif risk_level >= 70 and risk_level < 85:
            risk_label = "High"
        elif risk_level >= 85:
            risk_label = "Extreme"
            
        level_text = pg.TextItem(risk_label, anchor=(0.5, 0.5), color=QColor(50, 50, 50))
        level_text.setPos(0, -0.5)
        plot_widget.addItem(level_text)
        
        # Set view range
        plot_widget.setXRange(-1, 1)
        plot_widget.setYRange(-1, 1)
        
        return pointer
        
    except Exception as e:
        logger.error(f"Error creating risk gauge: {e}")
        return None

def update_risk_gauge(pointer, risk_level):
    """
    Update the risk gauge with a new risk level.
    
    Parameters:
    - pointer: Pointer from create_risk_gauge
    - risk_level: New risk level (0-100)
    """
    try:
        logger.info(f"Updating risk gauge to level {risk_level}")
        
        # Update the pointer
        pointer_angle = -np.pi + (risk_level / 100) * np.pi
        x_pointer = [0, 0.7 * np.cos(pointer_angle)]
        y_pointer = [0, 0.7 * np.sin(pointer_angle)]
        
        pointer.setData(x=x_pointer, y=y_pointer)
        
    except Exception as e:
        logger.error(f"Error updating risk gauge: {e}")

def create_time_series_chart(plot_widget, data=None):
    """
    Create a time series chart for displaying risk trends.
    
    Parameters:
    - plot_widget: PyQtGraph PlotWidget
    - data: Optional DataFrame with historical risk data
    
    Returns:
    - plot item for updating later
    """
    try:
        logger.info("Creating time series chart")
        
        # Clear any existing plots
        plot_widget.clear()
        
        # Configure the plot
        plot_widget.setBackground('w')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add labels
        plot_widget.setLabel('left', 'Risk Level', units='%')
        plot_widget.setLabel('bottom', 'Date')
        
        # Create sample data if none is provided
        if data is None:
            # Generate sample dates for last 90 days
            dates = pd.date_range(end=pd.Timestamp.now(), periods=90)
            
            # Generate sample risk values
            base_risk = 50 + np.random.normal(0, 5, size=90)  # Base risk around 50%
            trend = np.linspace(0, 20, 90)  # Upward trend
            cycle = 10 * np.sin(np.linspace(0, 4*np.pi, 90))  # Cyclical component
            risk_values = base_risk + trend + cycle
            risk_values = np.clip(risk_values, 0, 100)  # Ensure within 0-100 range
            
            # Create sample prediction with confidence intervals
            pred_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=10)
            pred_values = np.linspace(risk_values[-1], risk_values[-1] + 10, 10) + np.random.normal(0, 3, size=10)
            pred_values = np.clip(pred_values, 0, 100)
            
            # Confidence intervals (simple +/- 10%)
            lower_bound = pred_values - 10
            upper_bound = pred_values + 10
            
            # Clip confidence intervals
            lower_bound = np.clip(lower_bound, 0, 100)
            upper_bound = np.clip(upper_bound, 0, 100)
        else:
            # Use provided data
            dates = data.index
            risk_values = data['risk_score'].values if 'risk_score' in data.columns else data.iloc[:, 0].values
            
            # Check if prediction data is included
            if 'prediction' in data.columns:
                pred_dates = data.index[-10:]
                pred_values = data['prediction'].values[-10:]
                lower_bound = data['lower_bound'].values[-10:] if 'lower_bound' in data.columns else pred_values - 10
                upper_bound = data['upper_bound'].values[-10:] if 'upper_bound' in data.columns else pred_values + 10
            else:
                pred_dates = []
                pred_values = []
                lower_bound = []
                upper_bound = []
        
        # Convert dates to timestamp for PyQtGraph
        if isinstance(dates, pd.DatetimeIndex):
            dates_timestamp = [d.timestamp() for d in dates]
            x_axis = plot_widget.getAxis('bottom')
            x_axis.setTicks([[(dates_timestamp[i], dates[i].strftime('%b %d')) for i in range(0, len(dates), len(dates)//5)]])
        else:
            dates_timestamp = list(range(len(risk_values)))
        
        # Plot the historical data
        historical_curve = plot_widget.plot(dates_timestamp, risk_values, pen=pg.mkPen(QColor(212, 27, 96), width=2))
        
        # Plot the predicted data if available
        if len(pred_dates) > 0:
            if isinstance(pred_dates, pd.DatetimeIndex):
                pred_dates_timestamp = [d.timestamp() for d in pred_dates]
            else:
                pred_dates_timestamp = list(range(len(dates_timestamp), len(dates_timestamp) + len(pred_values)))
            
            # Plot prediction line
            pred_curve = plot_widget.plot(pred_dates_timestamp, pred_values, 
                                      pen=pg.mkPen(QColor(33, 150, 243), width=2, style=pg.QtCore.Qt.DashLine))
            
            # Create confidence interval fill
            confidence_fill = pg.FillBetweenItem(
                pg.PlotDataItem(x=pred_dates_timestamp, y=lower_bound),
                pg.PlotDataItem(x=pred_dates_timestamp, y=upper_bound),
                brush=pg.mkBrush(QColor(33, 150, 243, 50))
            )
            plot_widget.addItem(confidence_fill)
        
        # Add a legend
        legend = plot_widget.addLegend()
        legend.addItem(historical_curve, "Actual Risk Level")
        if len(pred_dates) > 0:
            legend.addItem(pred_curve, "Predicted (3 months ago)")
        
        return historical_curve
        
    except Exception as e:
        logger.error(f"Error creating time series chart: {e}")
        return None

def update_time_series_chart(plot_curve, data):
    """
    Update the time series chart with new data.
    
    Parameters:
    - plot_curve: Curve from create_time_series_chart
    - data: New data for the chart
    """
    try:
        logger.info("Updating time series chart")
        
        # Extract data
        dates = data.index
        risk_values = data['risk_score'].values if 'risk_score' in data.columns else data.iloc[:, 0].values
        
        # Convert dates to timestamp
        if isinstance(dates, pd.DatetimeIndex):
            dates_timestamp = [d.timestamp() for d in dates]
        else:
            dates_timestamp = list(range(len(risk_values)))
        
        # Update the plot
        plot_curve.setData(x=dates_timestamp, y=risk_values)
        
    except Exception as e:
        logger.error(f"Error updating time series chart: {e}")

def create_risk_heatmap(risk_components, figsize=(10, 8)):
    """
    Create a heatmap visualization of risk components.
    
    Parameters:
    - risk_components: List of dictionaries with risk components
    - figsize: Figure size
    
    Returns:
    - Matplotlib figure
    """
    try:
        logger.info("Creating risk components heatmap")
        
        # Extract data
        factors = [comp['factor'] for comp in risk_components]
        contributions = [comp['contribution'] for comp in risk_components]
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Factor': factors,
            'Contribution': contributions
        })
        
        # Sort by absolute contribution
        df = df.iloc[df['Contribution'].abs().argsort(ascending=False)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the heatmap
        cmap = sns.diverging_palette(10, 240, as_cmap=True)
        sns.barplot(x='Contribution', y='Factor', data=df, 
                   palette=sns.color_palette("RdBu_r", n_colors=len(df)))
        
        # Add labels
        plt.title('Risk Factor Contributions', fontsize=16)
        plt.xlabel('Contribution to Overall Risk', fontsize=12)
        plt.ylabel('Risk Factor', fontsize=12)
        
        # Add grid lines
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk heatmap: {e}")
        return None

def create_correlation_matrix(data, figsize=(12, 10)):
    """
    Create a correlation matrix visualization of indicators.
    
    Parameters:
    - data: DataFrame with indicators
    - figsize: Figure size
    
    Returns:
    - Matplotlib figure
    """
    try:
        logger.info("Creating correlation matrix")
        
        # Calculate correlations
        corr = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        # Add labels
        plt.title('Correlation Matrix of Financial Indicators', fontsize=16)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")
        return None

def create_risk_dashboard(risk_data, historical_data=None, predictions=None, figsize=(15, 10)):
    """
    Create a comprehensive risk dashboard visualization.
    
    Parameters:
    - risk_data: Dictionary with current risk metrics
    - historical_data: Optional DataFrame with historical risk data
    - predictions: Optional DataFrame with model predictions
    - figsize: Figure size
    
    Returns:
    - Matplotlib figure
    """
    try:
        logger.info("Creating comprehensive risk dashboard")
        
        # Extract data
        risk_score = risk_data.get('risk_score', 50)
        risk_level = risk_data.get('risk_level', 'moderate')
        risk_trend = risk_data.get('risk_trend', 'stable')
        components = risk_data.get('components', [])
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Define grid layout
        gs = fig.add_gridspec(3, 3)
        
        # Risk gauge
        ax1 = fig.add_subplot(gs[0, 0])
        create_risk_gauge_matplotlib(ax1, risk_score)
        ax1.set_title(f"Current Risk Level: {risk_level.title()}", fontsize=14)
        
        # Risk trend
        trend_icon = "↗" if risk_trend == 'increasing' else "↘" if risk_trend == 'decreasing' else "→"
        trend_color = "red" if risk_trend == 'increasing' else "green" if risk_trend == 'decreasing' else "orange"
        ax1.text(0.5, -0.1, f"Trend: {risk_trend.title()} {trend_icon}", 
                transform=ax1.transAxes, ha='center', color=trend_color, fontsize=12)
        
        # Top risk factors
        ax2 = fig.add_subplot(gs[0, 1:])
        create_top_factors_chart(ax2, components[:5])
        ax2.set_title("Top Risk Factors", fontsize=14)
        
        # Historical trend
        ax3 = fig.add_subplot(gs[1, :])
        create_historical_trend_chart(ax3, historical_data, predictions)
        ax3.set_title("Risk Trend", fontsize=14)
        
        # Risk components breakdown
        ax4 = fig.add_subplot(gs[2, :])
        create_components_breakdown(ax4, components)
        ax4.set_title("Risk Components Breakdown", fontsize=14)
        
        # Add title and timestamp
        fig.suptitle("Financial Crisis Risk Dashboard", fontsize=18)
        plt.figtext(0.98, 0.01, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                   ha='right', color='gray', fontsize=8)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating risk dashboard: {e}")
        return None

def create_risk_gauge_matplotlib(ax, risk_level):
    """
    Create a risk gauge visualization using Matplotlib.
    
    Parameters:
    - ax: Matplotlib axis
    - risk_level: Current risk level (0-100)
    """
    # Define colors for different risk levels
    colors = [
        (0.2, 0.8, 0.2),    # Green (low risk)
        (0.8, 0.8, 0.2),    # Yellow (moderate risk)
        (1.0, 0.6, 0.2),    # Orange (elevated risk)
        (1.0, 0.4, 0.2),    # Orange-Red (high risk)
        (1.0, 0.2, 0.2)     # Red (extreme risk)
    ]
    
    # Define angle ranges for each color
    angle_ranges = [
        (-180, -180 + (30/100)*180),   # 0-30%
        (-180 + (30/100)*180, -180 + (50/100)*180),   # 30-50%
        (-180 + (50/100)*180, -180 + (70/100)*180),   # 50-70%
        (-180 + (70/100)*180, -180 + (85/100)*180),   # 70-85%
        (-180 + (85/100)*180, 0)       # 85-100%
    ]
    
    # Create the gauge background
    for i, (start_angle, end_angle) in enumerate(angle_ranges):
        wedge = Wedge((0, 0), 0.8, start_angle, end_angle, width=0.2, 
                      facecolor=colors[i], edgecolor='gray', linewidth=0.5)
        ax.add_patch(wedge)
    
    # Calculate needle position
    needle_angle = -180 + (risk_level/100) * 180
    x = 0.7 * np.cos(np.radians(needle_angle))
    y = 0.7 * np.sin(np.radians(needle_angle))
    
    # Draw the needle
    ax.plot([0, x], [0, y], 'k-', linewidth=2)
    
    # Add center circle
    circle = plt.Circle((0, 0), 0.05, facecolor='darkgray', edgecolor='black', linewidth=0.5)
    ax.add_patch(circle)
    
    # Add risk percentage text
    ax.text(0, -0.2, f"{risk_level}%", ha='center', fontsize=20, fontweight='bold')
    
    # Configure axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_aspect('equal')

def create_top_factors_chart(ax, components):
    """
    Create a chart showing top risk factors.
    
    Parameters:
    - ax: Matplotlib axis
    - components: List of dictionaries with risk components
    """
    # Extract data
    factors = [comp['factor'] for comp in components]
    contributions = [comp['contribution'] for comp in components]
    
    # Determine colors based on contribution value
    colors = ['#ff9999' if c > 0 else '#99ff99' for c in contributions]
    
    # Create horizontal bar chart
    bars = ax.barh(factors, contributions, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else width - 0.5
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
               va='center', ha='left' if width > 0 else 'right')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Set labels
    ax.set_xlabel('Contribution to Risk')
    ax.set_ylabel('')
    
    # Adjust y-axis to show factor names clearly
    ax.tick_params(axis='y', labelsize=10)
    
    # Adjust layout
    ax.set_axisbelow(True)

def create_historical_trend_chart(ax, historical_data, predictions=None):
    """
    Create a chart showing historical risk trends with predictions.
    
    Parameters:
    - ax: Matplotlib axis
    - historical_data: DataFrame with historical risk data
    - predictions: Optional DataFrame with model predictions
    """
    # Create sample data if none is provided
    if historical_data is None or historical_data.empty:
        # Generate sample dates for last 90 days
        dates = pd.date_range(end=pd.Timestamp.now(), periods=90)
        
        # Generate sample risk values with trend and cycle components
        base_risk = 50 + np.random.normal(0, 5, size=90)  # Base risk around 50%
        trend = np.linspace(0, 20, 90)  # Upward trend
        cycle = 10 * np.sin(np.linspace(0, 4*np.pi, 90))  # Cyclical component
        risk_values = base_risk + trend + cycle
        risk_values = np.clip(risk_values, 0, 100)  # Ensure within 0-100 range
        
        # Create DataFrame
        historical_data = pd.DataFrame({
            'date': dates,
            'risk_score': risk_values
        }).set_index('date')
    
    # Plot historical data
    if 'risk_score' in historical_data.columns:
        historical_data['risk_score'].plot(ax=ax, color='#d81b60', linewidth=2)
    else:
        historical_data.iloc[:, 0].plot(ax=ax, color='#d81b60', linewidth=2)
    
    # Plot predictions if available
    if predictions is not None and not predictions.empty:
        # Get the prediction column
        if 'prediction' in predictions.columns:
            pred_col = 'prediction'
        else:
            pred_col = predictions.columns[0]
        
        # Plot prediction line
        predictions[pred_col].plot(ax=ax, color='#2196f3', linewidth=2, linestyle='--')
        
        # Plot confidence intervals if available
        if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
            ax.fill_between(predictions.index, predictions['lower_bound'], predictions['upper_bound'],
                           color='#2196f3', alpha=0.2)
    
    # Add risk level shading (background colors for different risk levels)
    ax.axhspan(0, 30, facecolor='#ccffcc', alpha=0.3)  # Low risk (green)
    ax.axhspan(30, 50, facecolor='#ffffcc', alpha=0.3)  # Moderate risk (yellow)
    ax.axhspan(50, 70, facecolor='#ffeecc', alpha=0.3)  # Elevated risk (light orange)
    ax.axhspan(70, 85, facecolor='#ffddcc', alpha=0.3)  # High risk (dark orange)
    ax.axhspan(85, 100, facecolor='#ffcccc', alpha=0.3)  # Extreme risk (red)
    
    # Add risk level labels
    ax.text(ax.get_xlim()[0], 15, 'Low', fontsize=9, color='#006600')
    ax.text(ax.get_xlim()[0], 40, 'Moderate', fontsize=9, color='#666600')
    ax.text(ax.get_xlim()[0], 60, 'Elevated', fontsize=9, color='#cc6600')
    ax.text(ax.get_xlim()[0], 78, 'High', fontsize=9, color='#cc3300')
    ax.text(ax.get_xlim()[0], 93, 'Extreme', fontsize=9, color='#cc0000')
    
    # Configure axes
    ax.set_ylim(0, 100)
    ax.set_xlabel('')
    ax.set_ylabel('Risk Score')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    if predictions is not None and not predictions.empty:
        ax.legend(['Actual Risk', 'Predicted Risk', 'Confidence Interval'])
    else:
        ax.legend(['Risk Score'])

def create_components_breakdown(ax, components):
    """
    Create a chart showing the breakdown of risk components.
    
    Parameters:
    - ax: Matplotlib axis
    - components: List of dictionaries with risk components
    """
    # Group components by category
    categories = {
        'Market': ['VIX', 'Yield_Curve', 'S&P500', 'VIX_RV_Ratio', 'Rolling_Volatility'],
        'Economic': ['Unemployment', 'Inflation', 'Fed_Funds', 'Housing', 'Debt', 'Financial'],
        'Sentiment': ['sentiment', 'impact', 'News', 'Geopolitical']
    }
    
    category_totals = {cat: 0 for cat in categories}
    
    # Calculate category totals
    for comp in components:
        factor = comp['factor']
        contribution = comp['contribution']
        
        # Assign to category based on name matching
        for cat, terms in categories.items():
            if any(term.lower() in factor.lower() for term in terms):
                category_totals[cat] += contribution
                break
        else:
            # If no category matches, create an 'Other' category
            if 'Other' not in category_totals:
                category_totals['Other'] = 0
            category_totals['Other'] += contribution
    
    # Filter out categories with zero or negative contribution
    category_totals = {k: v for k, v in category_totals.items() if v > 0}
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        category_totals.values(),
        labels=category_totals.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Spectral(np.linspace(0, 1, len(category_totals))),
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Style the chart
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add a title
    ax.set_title('Risk Contribution by Category')