import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("risk_calculator.log"), logging.StreamHandler()]
)
logger = logging.getLogger("RiskCalculator")

class RiskCalculator:
    """
    Calculates financial crisis risk metrics based on model predictions and indicators.
    """
    
    def __init__(self):
        # Risk thresholds for different levels
        self.risk_thresholds = {
            'low': 30,
            'moderate': 50,
            'elevated': 70,
            'high': 85,
            'extreme': 95
        }
        
        # Historical crisis periods (for reference)
        self.historical_crises = [
            {'name': 'Global Financial Crisis', 'start': '2007-08-01', 'end': '2009-03-01', 'peak': '2008-09-15'},
            {'name': 'Dot-com Bubble', 'start': '2000-03-01', 'end': '2002-10-01', 'peak': '2001-03-01'},
            {'name': 'COVID-19 Crisis', 'start': '2020-02-01', 'end': '2020-04-01', 'peak': '2020-03-23'},
            {'name': 'European Debt Crisis', 'start': '2010-04-01', 'end': '2012-07-01', 'peak': '2011-11-01'}
        ]
        
        # Indicator weights for risk factors
        self.indicator_weights = {
            # Market indicators
            'VIX': 0.05,                       # Volatility Index
            'Yield_Curve_10Y_3M': 0.08,        # Yield curve inversion
            'Yield_Curve_10Y_2Y': 0.07,        # Yield curve (10Y-2Y)
            'VIX_RV_Ratio': 0.04,              # VIX/Realized Volatility ratio
            'Rolling_Volatility': 0.04,        # Rolling market volatility
            'S&P500_Returns': 0.03,            # Recent market returns
            
            # Economic indicators
            'Unemployment_Rate': 0.06,         # Unemployment rate
            'Inflation_Rate': 0.06,            # Inflation rate
            'Fed_Funds_Rate': 0.05,            # Fed funds rate
            'Housing_Starts': 0.04,            # Housing starts
            'Govt_Debt_GDP': 0.04,             # Government debt to GDP
            'Household_Debt_GDP': 0.04,        # Household debt to GDP
            'ADS_Index': 0.05,                 # Aruoba-Diebold-Scotti index
            'Financial_Stress': 0.07,          # Financial stress index
            'Financial_Conditions': 0.07,      # Financial conditions index
            'High_Yield_Spread': 0.06,         # High yield bond spreads
            
            # Sentiment indicators
            'avg_sentiment': 0.04,             # News sentiment
            'avg_impact': 0.03,                # Geopolitical impact
        }
        
        # Direction of indicators (positive = higher values mean higher risk)
        self.indicator_directions = {
            'VIX': 1,                          # Higher volatility = higher risk
            'Yield_Curve_10Y_3M': -1,          # Lower (more inverted) curve = higher risk
            'Yield_Curve_10Y_2Y': -1,          # Lower (more inverted) curve = higher risk
            'VIX_RV_Ratio': 1,                 # Higher ratio = higher risk
            'Rolling_Volatility': 1,           # Higher volatility = higher risk
            'S&P500_Returns': -1,              # Lower returns = higher risk
            
            'Unemployment_Rate': 1,            # Higher unemployment = higher risk
            'Inflation_Rate': 1,               # Higher inflation = higher risk (simplification)
            'Fed_Funds_Rate': 0,               # Neutral (both high and low can indicate risk)
            'Housing_Starts': -1,              # Lower starts = higher risk
            'Govt_Debt_GDP': 1,                # Higher debt = higher risk
            'Household_Debt_GDP': 1,           # Higher debt = higher risk
            'ADS_Index': -1,                   # Lower index = higher risk
            'Financial_Stress': 1,             # Higher stress = higher risk
            'Financial_Conditions': 1,         # Higher (tighter) conditions = higher risk
            'High_Yield_Spread': 1,            # Higher spreads = higher risk
            
            'avg_sentiment': -1,               # Lower sentiment = higher risk
            'avg_impact': 1,                   # Higher geopolitical impact = higher risk
        }
        
        logger.info("RiskCalculator initialized with default weights and thresholds")
    
    def calculate_overall_risk(self, data, predictions=None):
        """
        Calculate overall financial crisis risk based on indicators and model predictions.
        
        Parameters:
        - data: DataFrame with financial indicators
        - predictions: Optional DataFrame with model predictions
        
        Returns:
        - Dictionary with risk metrics
        """
        try:
            logger.info("Calculating overall financial crisis risk")
            
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                logger.error("Input data is not a DataFrame")
                return None
            
            # Get the most recent data point
            recent_data = data.iloc[-1].copy() if not data.empty else pd.Series()
            
            # Calculate risk contribution from each indicator
            risk_contributions = self._calculate_indicator_contributions(recent_data)
            
            # Calculate the total risk score (0-100)
            total_risk = sum(contrib['contribution'] for contrib in risk_contributions)
            
            # Determine risk level based on thresholds
            risk_level = self._determine_risk_level(total_risk)
            
            # Calculate trend (comparing to previous periods)
            risk_trend = self._calculate_risk_trend(data)
            
            # Incorporate model predictions if available
            prediction_risk = self._incorporate_predictions(predictions) if predictions is not None else 0
            
            # Adjust risk with prediction information
            adjusted_risk = self._adjust_risk_with_predictions(total_risk, prediction_risk)
            
            # Create risk summary
            risk_summary = {
                'risk_score': round(adjusted_risk, 1),
                'risk_level': risk_level,
                'risk_trend': risk_trend,
                'components': risk_contributions,
                'calculation_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"Risk calculation complete: Score={adjusted_risk:.1f}, Level={risk_level}, Trend={risk_trend}")
            return risk_summary
            
        except Exception as e:
            logger.error(f"Error calculating overall risk: {e}")
            return None
    
    def _calculate_indicator_contributions(self, data):
        """
        Calculate risk contribution from each indicator.
        
        Parameters:
        - data: Series with recent indicator values
        
        Returns:
        - List of dictionaries with indicator contributions
        """
        contributions = []
        
        # Process each indicator
        for indicator, weight in self.indicator_weights.items():
            if indicator in data.index:
                value = data[indicator]
                
                # Skip if the value is NaN
                if pd.isna(value):
                    continue
                
                # Calculate z-score (assuming normal distribution for simplicity)
                # In a real implementation, we would use historical distributions
                if indicator == 'Yield_Curve_10Y_3M' or indicator == 'Yield_Curve_10Y_2Y':
                    # For yield curve, negative is bad (inverted curve)
                    if value < 0:
                        # More negative = worse
                        score = min(-value * 20, 100)  # Scale to 0-100
                    else:
                        # Positive but close to zero still has some risk
                        score = max(50 - value * 50, 0)
                elif indicator == 'S&P500_Returns':
                    # For returns, negative is bad
                    if value < 0:
                        # More negative = worse
                        score = min(-value * 200, 100)  # Scale to 0-100
                    else:
                        # Positive returns are good
                        score = max(40 - value * 200, 0)
                elif indicator == 'Financial_Stress' or indicator == 'Financial_Conditions':
                    # These indices are already standardized
                    if value > 0:
                        # Positive = stress/tightening
                        score = min(value * 33 + 50, 100)
                    else:
                        # Negative = easing
                        score = max(50 + value * 25, 0)
                elif indicator == 'avg_sentiment':
                    # Sentiment from -1 to 1
                    # Negative sentiment = higher risk
                    score = 50 - value * 50
                else:
                    # Generic approach for other indicators
                    # This is a placeholder - in a real implementation, we would use
                    # historical distributions and proper transformations
                    score = 50  # Neutral starting point
                
                # Adjust direction
                direction = self.indicator_directions.get(indicator, 0)
                if direction != 0:
                    contribution = weight * score * direction
                else:
                    # For neutral indicators, take the deviation from the middle
                    contribution = weight * abs(score - 50)
                
                contributions.append({
                    'factor': indicator,
                    'value': value,
                    'score': score,
                    'weight': weight,
                    'contribution': contribution,
                    'direction': 'positive' if contribution > 0 else 'negative'
                })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return contributions
    
    def _determine_risk_level(self, risk_score):
        """
        Determine the risk level based on the score.
        
        Parameters:
        - risk_score: Numerical risk score
        
        Returns:
        - Risk level category
        """
        if risk_score < self.risk_thresholds['low']:
            return 'low'
        elif risk_score < self.risk_thresholds['moderate']:
            return 'moderate'
        elif risk_score < self.risk_thresholds['elevated']:
            return 'elevated'
        elif risk_score < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_risk_trend(self, data, window=30):
        """
        Calculate the risk trend based on recent data.
        
        Parameters:
        - data: DataFrame with historical data
        - window: Number of periods to evaluate
        
        Returns:
        - Trend direction ('increasing', 'decreasing', or 'stable')
        """
        try:
            if len(data) < window:
                return 'unknown'
            
            # Calculate risk for each point in the window
            risk_scores = []
            
            for i in range(min(window, len(data))):
                idx = len(data) - i - 1
                if idx >= 0:
                    point_data = data.iloc[idx]
                    contributions = self._calculate_indicator_contributions(point_data)
                    score = sum(contrib['contribution'] for contrib in contributions)
                    risk_scores.append(score)
            
            if not risk_scores:
                return 'unknown'
            
            # Calculate the slope of the trend
            x = np.arange(len(risk_scores))
            slope, _, _, _, _ = stats.linregress(x, risk_scores)
            
            # Determine trend direction
            if abs(slope) < 0.05:  # Threshold for stability
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
            
        except Exception as e:
            logger.error(f"Error calculating risk trend: {e}")
            return 'unknown'
    
    def _incorporate_predictions(self, predictions, forecast_horizon=10):
        """
        Incorporate model predictions into the risk calculation.
        
        Parameters:
        - predictions: DataFrame with model predictions
        - forecast_horizon: Number of periods to consider
        
        Returns:
        - Prediction risk score
        """
        try:
            if predictions is None or len(predictions) == 0:
                return 0
            
            # Limit to specified horizon
            predictions = predictions.iloc[:min(forecast_horizon, len(predictions))]
            
            # Extract prediction values and uncertainty
            if 'prediction' in predictions.columns:
                pred_values = predictions['prediction'].values
            else:
                # If no prediction column, assume the first numeric column is the prediction
                pred_values = predictions.select_dtypes(include=[np.number]).iloc[:, 0].values
            
            # Calculate the trend in predictions
            if len(pred_values) > 1:
                x = np.arange(len(pred_values))
                slope, _, _, _, _ = stats.linregress(x, pred_values)
                
                # Convert slope to a risk factor
                # Positive slope (increasing values) might indicate increasing risk
                # or decreasing risk depending on what we're predicting
                # For simplicity, we'll assume we're predicting a risk metric
                if slope > 0.1:
                    trend_risk = 20  # Strong increasing trend
                elif slope > 0.05:
                    trend_risk = 10  # Moderate increasing trend
                elif slope < -0.1:
                    trend_risk = -20  # Strong decreasing trend
                elif slope < -0.05:
                    trend_risk = -10  # Moderate decreasing trend
                else:
                    trend_risk = 0  # Stable trend
            else:
                trend_risk = 0
            
            # Consider uncertainty if available
            if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
                lower_bounds = predictions['lower_bound'].values
                upper_bounds = predictions['upper_bound'].values
                
                # Calculate average uncertainty as percentage of prediction
                avg_uncertainty = np.mean((upper_bounds - lower_bounds) / np.abs(pred_values))
                
                # Convert to risk factor
                if avg_uncertainty > 0.5:
                    uncertainty_risk = 15  # High uncertainty
                elif avg_uncertainty > 0.2:
                    uncertainty_risk = 10  # Moderate uncertainty
                else:
                    uncertainty_risk = 5  # Low uncertainty
            else:
                uncertainty_risk = 8  # Default moderate uncertainty
            
            # Combine trend and uncertainty factors
            prediction_risk = trend_risk + uncertainty_risk
            
            return prediction_risk
            
        except Exception as e:
            logger.error(f"Error incorporating predictions: {e}")
            return 0
    
    def _adjust_risk_with_predictions(self, current_risk, prediction_risk):
        """
        Adjust the current risk score with prediction insights.
        
        Parameters:
        - current_risk: Current risk score based on indicators
        - prediction_risk: Risk contribution from predictions
        
        Returns:
        - Adjusted risk score
        """
        # Simple linear combination
        adjusted_risk = current_risk * 0.8 + prediction_risk
        
        # Ensure in valid range
        adjusted_risk = max(0, min(100, adjusted_risk))
        
        return adjusted_risk
    
    def calculate_historical_risk(self, data, window_size=30):
        """
        Calculate historical risk scores for a time series of data.
        
        Parameters:
        - data: DataFrame with historical indicators
        - window_size: Rolling window size for calculation
        
        Returns:
        - DataFrame with historical risk scores
        """
        try:
            logger.info(f"Calculating historical risk with window size {window_size}")
            
            # Ensure we have enough data
            if len(data) < window_size:
                logger.warning(f"Data length ({len(data)}) is shorter than window size ({window_size})")
                window_size = max(1, len(data) // 2)
            
            # Initialize results
            dates = data.index[window_size-1:]
            risk_scores = []
            risk_levels = []
            
            # Calculate risk for each window
            for i in range(window_size-1, len(data)):
                window_data = data.iloc[i-window_size+1:i+1]
                
                # Use the last data point for contribution calculation
                recent_data = window_data.iloc[-1]
                contributions = self._calculate_indicator_contributions(recent_data)
                score = sum(contrib['contribution'] for contrib in contributions)
                
                risk_scores.append(score)
                risk_levels.append(self._determine_risk_level(score))
            
            # Create result DataFrame
            result = pd.DataFrame({
                'date': dates,
                'risk_score': risk_scores,
                'risk_level': risk_levels
            })
            
            if isinstance(data.index, pd.DatetimeIndex):
                result.set_index('date', inplace=True)
            
            logger.info(f"Historical risk calculation complete: {len(result)} data points")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating historical risk: {e}")
            return pd.DataFrame()
    
    def compare_to_historical_crises(self, current_risk):
        """
        Compare current risk level to historical financial crises.
        
        Parameters:
        - current_risk: Current risk score or dictionary with risk data
        
        Returns:
        - Dictionary with comparison insights
        """
        try:
            # Extract risk score if dictionary was provided
            if isinstance(current_risk, dict):
                risk_score = current_risk.get('risk_score', 0)
            else:
                risk_score = current_risk
            
            # Define risk scores for historical crises
            # These would ideally be calculated from actual historical data
            historical_scores = {
                'Global Financial Crisis': {
                    'pre_crisis': 75,
                    'peak': 95,
                    'recovery': 60
                },
                'Dot-com Bubble': {
                    'pre_crisis': 70,
                    'peak': 85,
                    'recovery': 55
                },
                'COVID-19 Crisis': {
                    'pre_crisis': 50,
                    'peak': 90,
                    'recovery': 65
                },
                'European Debt Crisis': {
                    'pre_crisis': 65,
                    'peak': 80,
                    'recovery': 50
                }
            }
            
            # Compare current risk to historical patterns
            comparisons = []
            
            for crisis in self.historical_crises:
                name = crisis['name']
                if name in historical_scores:
                    scores = historical_scores[name]
                    
                    # Find closest phase based on risk score
                    distances = {
                        'pre_crisis': abs(risk_score - scores['pre_crisis']),
                        'peak': abs(risk_score - scores['peak']),
                        'recovery': abs(risk_score - scores['recovery'])
                    }
                    closest_phase = min(distances, key=distances.get)
                    
                    comparisons.append({
                        'crisis': name,
                        'closest_phase': closest_phase,
                        'similarity': 100 - distances[closest_phase],
                        'historical_score': scores[closest_phase],
                        'distance': risk_score - scores[closest_phase]
                    })
            
            # Sort by similarity
            comparisons.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'current_risk': risk_score,
                'comparisons': comparisons,
                'most_similar': comparisons[0] if comparisons else None
            }
            
        except Exception as e:
            logger.error(f"Error comparing to historical crises: {e}")
            return {
                'current_risk': 0,
                'comparisons': [],
                'most_similar': None
            }
    
    def generate_risk_report(self, data, predictions=None):
        """
        Generate a comprehensive risk report.
        
        Parameters:
        - data: DataFrame with indicators
        - predictions: Optional DataFrame with model predictions
        
        Returns:
        - Dictionary with report data
        """
        try:
            logger.info("Generating comprehensive risk report")
            
            # Calculate current risk
            current_risk = self.calculate_overall_risk(data, predictions)
            
            # Calculate historical risk
            historical_risk = self.calculate_historical_risk(data)
            
            # Compare to historical crises
            historical_comparison = self.compare_to_historical_crises(current_risk)
            
            # Generate report
            report = {
                'current_risk': current_risk,
                'historical_data': historical_risk.tail(90).to_dict() if not historical_risk.empty else {},
                'historical_comparison': historical_comparison,
                'predictions': predictions.to_dict() if predictions is not None and not predictions.empty else {},
                'report_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info("Risk report generation complete")
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return Nonesort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return contributions