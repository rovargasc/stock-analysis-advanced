import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta
import yfinance as yf
from portfolio_analyzer import StockAnalysis

class AdvancedAnalytics:
    """Utilidades adicionales para análisis avanzado"""
    
    @staticmethod
    def calculate_portfolio_metrics(analyses: List[StockAnalysis], weights: List[float] = None) -> Dict:
        """Calcular métricas de cartera (riesgo, retorno esperado, Sharpe ratio, etc.)"""
        if not analyses:
            return {}
        
        if weights is None:
            weights = [1/len(analyses)] * len(analyses)
        
        # Obtener datos históricos para cálculo de volatilidad y correlaciones
        returns_data = {}
        symbols = [a.symbol for a in analyses]
        
        try:
            # Descargar datos históricos (último año)
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
        except Exception as e:
            print(f"Error obteniendo datos históricos: {e}")
        
        # Calcular métricas básicas
        expected_returns = []
        volatilities = []
        
        for analysis in analyses:
            # Retorno esperado basado en upside potential
            expected_return = analysis.upside_potential if isinstance(analysis.upside_potential, (int, float)) else 0
            expected_returns.append(expected_return / 100)  # Convertir a decimal
            
            # Volatilidad estimada
            if analysis.symbol in returns_data:
                vol = returns_data[analysis.symbol].std() * np.sqrt(252)  # Anualizada
                volatilities.append(vol)
            else:
                # Estimar volatilidad basada en ATR si está disponible
                vol_estimate = analysis.volatility / 100 if hasattr(analysis, 'volatility') else 0.2
                volatilities.append(vol_estimate)
        
        # Métricas de cartera
        portfolio_return = np.dot(weights, expected_returns)
        
        # Calcular volatilidad de cartera (simplificado, asumiendo correlación promedio)
        avg_correlation = 0.3  # Correlación promedio estimada entre acciones
        portfolio_variance = 0
        
        for i in range(len(weights)):
            for j in range(len(weights)):
                if i == j:
                    portfolio_variance += weights[i]**2 * volatilities[i]**2
                else:
                    portfolio_variance += weights[i] * weights[j] * volatilities[i] * volatilities[j] * avg_correlation
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (asumiendo risk-free rate del 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification score
        sector_count = len(set(a.sector for a in analyses))
        diversification_score = min(100, (sector_count / 10) * 100)  # Máximo 10 sectores
        
        # Risk-adjusted score
        avg_overall_score = np.mean([a.overall_score for a in analyses])
        risk_adjusted_score = avg_overall_score * (1 - portfolio_volatility)
        
        return {
            'expected_return': portfolio_return * 100,  # En porcentaje
            'volatility': portfolio_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'diversification_score': diversification_score,
            'risk_adjusted_score': risk_adjusted_score,
            'number_of_stocks': len(analyses),
            'sector_count': sector_count
        }
    
    @staticmethod
    def generate_market_outlook(analyses: List[StockAnalysis]) -> Dict:
        """Generar outlook general del mercado basado en las acciones analizadas"""
        if not analyses:
            return {}
        
        # Análisis de sentimiento general
        buy_count = len([a for a in analyses if a.analyst_consensus == "BUY"])
        hold_count = len([a for a in analyses if a.analyst_consensus == "HOLD"])
        sell_count = len([a for a in analyses if a.analyst_consensus == "SELL"])
        total = len(analyses)
        
        # Scores promedio por sector
        sector_scores = {}
        for analysis in analyses:
            sector = analysis.sector
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(analysis.overall_score)
        
        # Calcular promedios por sector
        sector_averages = {sector: np.mean(scores) for sector, scores in sector_scores.items()}
        best_sector = max(sector_averages.keys(), key=lambda k: sector_averages[k])
        worst_sector = min(sector_averages.keys(), key=lambda k: sector_averages[k])
        
        # Análisis de valoración
        pe_ratios = [a.pe_ratio for a in analyses if isinstance(a.pe_ratio, (int, float)) and a.pe_ratio > 0]
        avg_pe = np.mean(pe_ratios) if pe_ratios else None
        
        # Análisis técnico general
        overbought_count = len([a for a in analyses if isinstance(a.rsi, (int, float)) and a.rsi > 70])
        oversold_count = len([a for a in analyses if isinstance(a.rsi, (int, float)) and a.rsi < 30])
        
        # Determinar outlook general
        buy_ratio = buy_count / total
        avg_score = np.mean([a.overall_score for a in analyses])
        
        if buy_ratio > 0.6 and avg_score > 70:
            market_sentiment = "MUY POSITIVO"
        elif buy_ratio > 0.4 and avg_score > 60:
            market_sentiment = "POSITIVO"
        elif buy_ratio < 0.3 or avg_score < 40:
            market_sentiment = "NEGATIVO"
        else:
            market_sentiment = "NEUTRAL"
        
        return {
            'market_sentiment': market_sentiment,
            'buy_ratio': buy_ratio * 100,
            'average_score': avg_score,
            'best_performing_sector': best_sector,
            'worst_performing_sector': worst_sector,
            'sector_scores': sector_averages,
            'average_pe': avg_pe,
            'overbought_stocks': overbought_count,
            'oversold_stocks': oversold_count,
            'total_analyzed': total
        }
    
    @staticmethod
    def find_arbitrage_opportunities(analyses: List[StockAnalysis]) -> List[Dict]:
        """Encontrar oportunidades de arbitraje o discrepancias en valoración"""
        opportunities = []
        
        # Agrupar por sector para comparación
        sector_groups = {}
        for analysis in analyses:
            sector = analysis.sector
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(analysis)
        
        # Buscar discrepancias dentro de cada sector
        for sector, stocks in sector_groups.items():
            if len(stocks) < 2:
                continue
            
            # Ordenar por score fundamental
            stocks_sorted = sorted(stocks, key=lambda x: x.fundamental_score, reverse=True)
            
            # Comparar stocks similares
            for i, stock1 in enumerate(stocks_sorted):
                for stock2 in stocks_sorted[i+1:]:
                    # Buscar casos donde un stock con mejores fundamentales tiene menor precio objetivo
                    if (stock1.fundamental_score > stock2.fundamental_score + 10 and
                        isinstance(stock1.upside_potential, (int, float)) and
                        isinstance(stock2.upside_potential, (int, float)) and
                        stock1.upside_potential < stock2.upside_potential - 5):
                        
                        opportunities.append({
                            'type': 'SECTOR_ARBITRAGE',
                            'description': f'{stock1.symbol} tiene mejores fundamentales que {stock2.symbol} pero menor upside',
                            'stock1': stock1.symbol,
                            'stock2': stock2.symbol,
                            'sector': sector,
                            'fundamental_diff': stock1.fundamental_score - stock2.fundamental_score,
                            'upside_diff': stock2.upside_potential - stock1.upside_potential
                        })
        
        # Buscar stocks con alta discrepancia entre análisis técnico y fundamental
        for analysis in analyses:
            tech_fund_diff = abs(analysis.technical_score - analysis.fundamental_score)
            if tech_fund_diff > 30:
                if analysis.fundamental_score > analysis.technical_score:
                    opp_type = "TECHNICAL_UNDERVALUED"
                    description = f"{analysis.symbol}: Fundamentales fuertes pero técnicos débiles"
                else:
                    opp_type = "FUNDAMENTAL_UNDERVALUED"
                    description = f"{analysis.symbol}: Técnicos fuertes pero fundamentales débiles"
                
                opportunities.append({
                    'type': opp_type,
                    'description': description,
                    'symbol': analysis.symbol,
                    'fundamental_score': analysis.fundamental_score,
                    'technical_score': analysis.technical_score,
                    'difference': tech_fund_diff
                })
        
        return opportunities[:10]  # Retornar top 10 oportunidades
    
    @staticmethod
    def create_correlation_heatmap(analyses: List[StockAnalysis]) -> go.Figure:
        """Crear heatmap de correlaciones entre acciones"""
        symbols = [a.symbol for a in analyses[:20]]  # Limitar a 20 para legibilidad
        
        try:
            # Obtener datos históricos
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
                if not hist.empty:
                    data[symbol] = hist['Close'].pct_change().dropna()
            
            if len(data) < 2:
                return go.Figure()
            
            # Crear DataFrame de retornos
            returns_df = pd.DataFrame(data)
            
            # Calcular matriz de correlación
            corr_matrix = returns_df.corr()
            
            # Crear heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlación"),
                hoverongaps=False,
                hovertemplate='%{x} vs %{y}<br>Correlación: %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Matriz de Correlaciones (Últimos 6 meses)',
                height=600,
                xaxis_title='Símbolos',
                yaxis_title='Símbolos'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creando heatmap de correlaciones: {e}")
            return go.Figure()
    
    @staticmethod
    def export_analysis_to_excel(analyses: List[StockAnalysis], filename: str = None):
        """Exportar análisis completo a Excel"""
        if filename is None:
            filename = f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Hoja principal con resumen
                summary_data = []
                for analysis in analyses:
                    summary_data.append({
                        'Symbol': analysis.symbol,
                        'Name': analysis.name,
                        'Sector': analysis.sector,
                        'Price': analysis.price,
                        'Market_Cap_M': analysis.market_cap,
                        'PE_Ratio': analysis.pe_ratio,
                        'PB_Ratio': analysis.pb_ratio,
                        'ROE': analysis.roe,
                        'ROA': analysis.roa,
                        'Debt_to_Equity': analysis.debt_to_equity,
                        'Profit_Margin': analysis.profit_margin,
                        'Revenue_Growth': analysis.revenue_growth,
                        'RSI': analysis.rsi,
                        'MACD_Signal': analysis.macd_signal,
                        'Bollinger_Position': analysis.bollinger_position,
                        'Trend_Strength': analysis.trend_strength,
                        'Volatility': analysis.volatility,
                        'Analyst_Rating': analysis.analyst_rating,
                        'Analyst_Consensus': analysis.analyst_consensus,
                        'Target_Price': analysis.target_price,
                        'Upside_Potential': analysis.upside_potential,
                        'Fundamental_Score': analysis.fundamental_score,
                        'Technical_Score': analysis.technical_score,
                        'Sentiment_Score': analysis.sentiment_score,
                        'Overall_Score': analysis.overall_score,
                        'Risk_Level': analysis.risk_level
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
                
                # Hoja con rankings
                rankings_data = {
                    'Top_Overall': sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:10],
                    'Top_Fundamental': sorted(analyses, key=lambda x: x.fundamental_score, reverse=True)[:10],
                    'Top_Technical': sorted(analyses, key=lambda x: x.technical_score, reverse=True)[:10],
                    'Top_Upside': sorted([a for a in analyses if isinstance(a.upside_potential, (int, float))], 
                                        key=lambda x: x.upside_potential, reverse=True)[:10]
                }
                
                for rank_type, top_stocks in rankings_data.items():
                    rank_data = []
                    for i, stock in enumerate(top_stocks, 1):
                        rank_data.append({
                            'Rank': i,
                            'Symbol': stock.symbol,
                            'Name': stock.name,
                            'Score': getattr(stock, rank_type.lower().split('_')[1] + '_score') if 'score' in rank_type.lower() else stock.upside_potential,
                            'Price': stock.price,
                            'Market_Cap': stock.market_cap
                        })
                    
                    rank_df = pd.DataFrame(rank_data)
                    rank_df.to_excel(writer, sheet_name=rank_type, index=False)
                
                # Hoja con análisis por sector
                sector_analysis = {}
                for analysis in analyses:
                    sector = analysis.sector
                    if sector not in sector_analysis:
                        sector_analysis[sector] = []
                    sector_analysis[sector].append(analysis)
                
                sector_summary = []
                for sector, stocks in sector_analysis.items():
                    sector_summary.append({
                        'Sector': sector,
                        'Stock_Count': len(stocks),
                        'Avg_Score': np.mean([s.overall_score for s in stocks]),
                        'Avg_Upside': np.mean([s.upside_potential for s in stocks if isinstance(s.upside_potential, (int, float))]),
                        'Best_Stock': max(stocks, key=lambda x: x.overall_score).symbol,
                        'Total_Market_Cap': sum([s.market_cap for s in stocks if s.market_cap > 0])
                    })
                
                sector_df = pd.DataFrame(sector_summary)
                sector_df.to_excel(writer, sheet_name='Sector_Analysis', index=False)
            
            return filename
            
        except Exception as e:
            print(f"Error exportando a Excel: {e}")
            return None
    
    @staticmethod
    def calculate_portfolio_optimization(analyses: List[StockAnalysis], target_return: float = None) -> Dict:
        """Optimización básica de cartera usando teoría moderna de carteras"""
        try:
            import scipy.optimize as sco
            
            n_assets = len(analyses)
            if n_assets < 2:
                return {}
            
            # Datos de entrada
            expected_returns = np.array([
                analysis.upside_potential / 100 if isinstance(analysis.upside_potential, (int, float)) else 0.05
                for analysis in analyses
            ])
            
            # Matriz de covarianza simplificada (asumir correlación 0.3)
            volatilities = np.array([
                analysis.volatility / 100 if hasattr(analysis, 'volatility') and analysis.volatility > 0 else 0.2
                for analysis in analyses
            ])
            
            correlation = 0.3
            cov_matrix = np.outer(volatilities, volatilities) * correlation
            np.fill_diagonal(cov_matrix, volatilities**2)
            
            # Función objetivo: minimizar riesgo para un retorno dado
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def portfolio_return(weights):
                return np.sum(expected_returns * weights)
            
            # Restricciones
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Suma de pesos = 1
            ]
            
            if target_return:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: portfolio_return(x) - target_return
                })
            
            bounds = tuple((0, 0.3) for _ in range(n_assets))  # Max 30% por activo
            
            # Optimización
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = sco.minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_return = portfolio_return(optimal_weights)
                optimal_risk = portfolio_volatility(optimal_weights)
                sharpe_ratio = optimal_return / optimal_risk
                
                return {
                    'success': True,
                    'optimal_weights': optimal_weights.tolist(),
                    'expected_return': optimal_return * 100,
                    'expected_risk': optimal_risk * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'symbols': [a.symbol for a in analyses]
                }
            else:
                return {'success': False, 'message': 'Optimization failed'}
                
        except ImportError:
            return {'success': False, 'message': 'scipy not available for optimization'}
        except Exception as e:
            return {'success': False, 'message': f'Error in optimization: {e}'}

def create_earnings_calendar_tracker(symbols: List[str]) -> pd.DataFrame:
    """Crear un tracker de fechas de earnings para las acciones analizadas"""
    earnings_data = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # Intentar obtener información de earnings
            try:
                calendar = ticker.calendar
                if calendar is not None and not calendar.empty:
                    next_earnings = calendar.index[0]
                    earnings_data.append({
                        'Symbol': symbol,
                        'Next_Earnings_Date': next_earnings.strftime('%Y-%m-%d'),
                        'Days_Until_Earnings': (next_earnings - datetime.now()).days
                    })
                else:
                    earnings_data.append({
                        'Symbol': symbol,
                        'Next_Earnings_Date': 'N/A',
                        'Days_Until_Earnings': 'N/A'
                    })
            except:
                earnings_data.append({
                    'Symbol': symbol,
                    'Next_Earnings_Date': 'N/A',
                    'Days_Until_Earnings': 'N/A'
                })
                
        except Exception as e:
            print(f"Error obteniendo datos de earnings para {symbol}: {e}")
    
    return pd.DataFrame(earnings_data)

def generate_trading_alerts(analyses: List[StockAnalysis]) -> List[Dict]:
    """Generar alertas de trading basadas en el análisis"""
    alerts = []
    
    for analysis in analyses:
        # Alert por RSI extremo
        if isinstance(analysis.rsi, (int, float)):
            if analysis.rsi < 25:
                alerts.append({
                    'type': 'OVERSOLD',
                    'symbol': analysis.symbol,
                    'message': f'{analysis.symbol} está en zona de sobreventa (RSI: {analysis.rsi:.1f})',
                    'priority': 'HIGH',
                    'action': 'CONSIDER_BUY'
                })
            elif analysis.rsi > 75:
                alerts.append({
                    'type': 'OVERBOUGHT',
                    'symbol': analysis.symbol,
                    'message': f'{analysis.symbol} está en zona de sobrecompra (RSI: {analysis.rsi:.1f})',
                    'priority': 'MEDIUM',
                    'action': 'CONSIDER_SELL'
                })
        
        # Alert por upside potential alto
        if isinstance(analysis.upside_potential, (int, float)) and analysis.upside_potential > 30:
            alerts.append({
                'type': 'HIGH_UPSIDE',
                'symbol': analysis.symbol,
                'message': f'{analysis.symbol} tiene alto potencial de upside ({analysis.upside_potential:.1f}%)',
                'priority': 'HIGH',
                'action': 'RESEARCH_MORE'
            })
        
        # Alert por discrepancia entre análisis técnico y fundamental
        tech_fund_diff = abs(analysis.technical_score - analysis.fundamental_score)
        if tech_fund_diff > 25:
            if analysis.fundamental_score > analysis.technical_score:
                alerts.append({
                    'type': 'TECHNICAL_LAGGING',
                    'symbol': analysis.symbol,
                    'message': f'{analysis.symbol}: Fundamentales fuertes pero técnicos débiles',
                    'priority': 'MEDIUM',
                    'action': 'WAIT_TECHNICAL_CONFIRMATION'
                })
            else:
                alerts.append({
                    'type': 'FUNDAMENTAL_CONCERN',
                    'symbol': analysis.symbol,
                    'message': f'{analysis.symbol}: Técnicos fuertes pero fundamentales débiles',
                    'priority': 'MEDIUM',
                    'action': 'REVIEW_FUNDAMENTALS'
                })
        
        # Alert por consenso de analistas vs precio
        if (analysis.analyst_consensus == "BUY" and 
            isinstance(analysis.upside_potential, (int, float)) and 
            analysis.upside_potential > 15):
            alerts.append({
                'type': 'ANALYST_BUY',
                'symbol': analysis.symbol,
                'message': f'{analysis.symbol}: Consenso BUY con {analysis.upside_potential:.1f}% upside',
                'priority': 'HIGH',
                'action': 'STRONG_BUY_SIGNAL'
            })
    
    # Ordenar por prioridad
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    alerts.sort(key=lambda x: priority_order.get(x['priority'], 2))
    
    return alerts[:20]  # Top 20 alerts

def create_monte_carlo_simulation(analyses: List[StockAnalysis], days: int = 252, simulations: int = 1000) -> Dict:
    """Ejecutar simulación Monte Carlo para proyección de precios"""
    results = {}
    
    for analysis in analyses[:10]:  # Limitar a 10 acciones para performance
        try:
            # Parámetros para la simulación
            current_price = analysis.price
            expected_return = analysis.upside_potential / 100 / days if isinstance(analysis.upside_potential, (int, float)) else 0.0002
            volatility = analysis.volatility / 100 / np.sqrt(days) if hasattr(analysis, 'volatility') and analysis.volatility > 0 else 0.02
            
            # Generar paths de precios
            price_paths = []
            final_prices = []
            
            for _ in range(simulations):
                prices = [current_price]
                for day in range(days):
                    random_return = np.random.normal(expected_return, volatility)
                    new_price = prices[-1] * (1 + random_return)
                    prices.append(new_price)
                
                price_paths.append(prices)
                final_prices.append(prices[-1])
            
            # Calcular estadísticas
            final_prices = np.array(final_prices)
            
            results[analysis.symbol] = {
                'current_price': current_price,
                'final_price_mean': np.mean(final_prices),
                'final_price_median': np.median(final_prices),
                'final_price_std': np.std(final_prices),
                'percentile_5': np.percentile(final_prices, 5),
                'percentile_95': np.percentile(final_prices, 95),
                'probability_profit': len(final_prices[final_prices > current_price]) / simulations * 100,
                'max_drawdown': np.min([np.min(path) for path in price_paths]) / current_price - 1,
                'max_gain': np.max(final_prices) / current_price - 1
            }
            
        except Exception as e:
            print(f"Error en simulación Monte Carlo para {analysis.symbol}: {e}")
    
    return results

def create_performance_attribution_chart(analyses: List[StockAnalysis]) -> go.Figure:
    """Crear gráfico de atribución de performance por factor"""
    
    # Calcular contribuciones de cada factor al score general
    factors = ['Fundamental', 'Technical', 'Sentiment']
    
    # Datos para el gráfico
    symbols = [a.symbol for a in analyses[:15]]  # Top 15
    fundamental_contrib = [a.fundamental_score * 0.4 for a in analyses[:15]]  # Peso 40%
    technical_contrib = [a.technical_score * 0.3 for a in analyses[:15]]      # Peso 30%
    sentiment_contrib = [a.sentiment_score * 0.3 for a in analyses[:15]]     # Peso 30%
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fundamental (40%)',
        x=symbols,
        y=fundamental_contrib,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Technical (30%)',
        x=symbols,
        y=technical_contrib,
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        name='Sentiment (30%)',
        x=symbols,
        y=sentiment_contrib,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Atribución de Performance por Factor',
        xaxis_title='Símbolos',
        yaxis_title='Contribución al Score',
        barmode='stack',
        height=600,
        showlegend=True
    )
    
    return fig