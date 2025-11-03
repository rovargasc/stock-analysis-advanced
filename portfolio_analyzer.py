import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import finnhub
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Tuple, Optional
import json
import os
import concurrent.futures
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StockAnalysis:
    """Clase para almacenar análisis completo de una acción"""
    symbol: str
    name: str
    sector: str
    price: float
    market_cap: float
    
    # Métricas financieras
    pe_ratio: float
    pb_ratio: float
    roe: float
    roa: float
    debt_to_equity: float
    profit_margin: float
    revenue_growth: float
    
    # Análisis técnico
    rsi: float
    macd_signal: str
    bollinger_position: str
    trend_strength: float
    volatility: float
    
    # Recomendaciones
    analyst_rating: float  # 1-5 (1=Strong Buy, 5=Strong Sell)
    analyst_consensus: str
    target_price: float
    upside_potential: float
    
    # Score compuesto
    fundamental_score: float
    technical_score: float
    sentiment_score: float
    overall_score: float
    
    # Datos adicionales
    volume_ratio: float
    insider_activity: str
    recent_news_sentiment: str
    risk_level: str

class PortfolioAnalyzer:
    """Analizador avanzado de carteras de inversión"""
    
    def __init__(self, finnhub_api_key: str):
        self.finnhub_api_key = finnhub_api_key
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
        
        # Listas de acciones para analizar por categoría
        self.stock_universe = {
            "Large Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "CRM", "ORCL"],
            "Financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BK", "USB", "PNC"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "MDT", "AMGN", "GILD"],
            "Consumer": ["WMT", "HD", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "COST"],
            "Industrial": ["BA", "CAT", "HON", "UNP", "LMT", "RTX", "DE", "MMM", "GE", "EMR"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL"],
            "Growth Stocks": ["TSLA", "PLTR", "ROKU", "SNOW", "CRWD", "ZM", "DOCU", "TWLO", "SHOP", "SQ"],
            "Dividend Kings": ["KO", "PEP", "JNJ", "PG", "MCD", "WMT", "MMM", "T", "VZ", "XOM"],
            "Small-Mid Cap": ["RBLX", "PLTR", "COIN", "RIVN", "LCID", "SOFI", "UPST", "AFRM", "HOOD", "DNA"]
        }
        
        # Pesos para el cálculo del score
        self.scoring_weights = {
            'fundamental': 0.4,
            'technical': 0.3,
            'sentiment': 0.3
        }
    
    def get_stock_data_parallel(self, symbols: List[str], max_workers: int = 10) -> Dict[str, Dict]:
        """Obtener datos de múltiples acciones en paralelo"""
        results = {}
        
        def fetch_single_stock(symbol):
            try:
                return symbol, self._get_comprehensive_stock_data(symbol)
            except Exception as e:
                st.warning(f"Error obteniendo datos de {symbol}: {e}")
                return symbol, None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(fetch_single_stock, symbol): symbol for symbol in symbols}
            
            progress_bar = st.progress(0)
            completed = 0
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol, data = future.result()
                if data:
                    results[symbol] = data
                completed += 1
                progress_bar.progress(completed / len(symbols))
        
        return results
    
    def _get_comprehensive_stock_data(self, symbol: str) -> Dict:
        """Obtener datos completos de una acción individual"""
        data = {}
        
        try:
            # Datos básicos de cotización
            quote = self.finnhub_client.quote(symbol)
            data['quote'] = quote
            
            # Perfil de la empresa
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            data['profile'] = profile
            
            # Métricas financieras
            financials = self.finnhub_client.company_basic_financials(symbol, 'all')
            data['financials'] = financials
            
            # Recomendaciones
            recommendations = self.finnhub_client.recommendation_trends(symbol)
            data['recommendations'] = recommendations
            
            # Datos de Yahoo Finance para completar
            ticker = yf.Ticker(symbol)
            yf_info = ticker.info
            data['yf_info'] = yf_info
            
            # Datos históricos para análisis técnico
            hist_data = ticker.history(period="3mo", interval="1d")
            if not hist_data.empty:
                data['historical'] = self._calculate_technical_indicators(hist_data)
            
            # Noticias recientes
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            news = self.finnhub_client.company_news(symbol, _from=from_date, to=to_date)
            data['news'] = news[:5]  # Solo últimas 5 noticias
            
        except Exception as e:
            st.warning(f"Error parcial obteniendo datos de {symbol}: {e}")
        
        return data
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos básicos"""
        if len(df) < 20:
            return df
        
        try:
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Percent'] = bollinger.bollinger_pband()
            
            # SMA
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            
            # ATR para volatilidad
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
        except Exception as e:
            st.warning(f"Error calculando indicadores técnicos: {e}")
        
        return df
    
    def analyze_stock(self, symbol: str, stock_data: Dict) -> Optional[StockAnalysis]:
        """Analizar una acción individual y crear objeto StockAnalysis"""
        try:
            quote = stock_data.get('quote', {})
            profile = stock_data.get('profile', {})
            financials = stock_data.get('financials', {})
            yf_info = stock_data.get('yf_info', {})
            recommendations = stock_data.get('recommendations', [])
            historical = stock_data.get('historical', pd.DataFrame())
            news = stock_data.get('news', [])
            
            # Datos básicos
            current_price = quote.get('c', 0)
            if current_price <= 0:
                return None
            
            name = profile.get('name', '') or yf_info.get('longName', symbol)
            sector = profile.get('finnhubIndustry', '') or yf_info.get('sector', 'Unknown')
            
            # Market Cap
            market_cap = profile.get('marketCapitalization', 0)
            if market_cap == 0:
                market_cap = yf_info.get('marketCap', 0) / 1e6  # Convertir a millones
            
            # Métricas financieras
            metrics = financials.get('metric', {})
            pe_ratio = metrics.get('peBasicExclExtraTTM') or yf_info.get('trailingPE', 0)
            pb_ratio = metrics.get('pbAnnual') or yf_info.get('priceToBook', 0)
            roe = metrics.get('roeRfy') or yf_info.get('returnOnEquity', 0)
            roa = metrics.get('roaRfy') or yf_info.get('returnOnAssets', 0)
            debt_to_equity = metrics.get('totalDebt/totalEquityAnnual') or yf_info.get('debtToEquity', 0)
            profit_margin = metrics.get('netProfitMarginAnnual') or yf_info.get('profitMargins', 0)
            revenue_growth = metrics.get('revenueGrowthTTMYoy') or yf_info.get('revenueGrowth', 0)
            
            # Convertir métricas si están en formato decimal
            if isinstance(roe, (int, float)) and roe > 0 and roe < 1:
                roe *= 100
            if isinstance(roa, (int, float)) and roa > 0 and roa < 1:
                roa *= 100
            if isinstance(profit_margin, (int, float)) and profit_margin > 0 and profit_margin < 1:
                profit_margin *= 100
            if isinstance(revenue_growth, (int, float)) and revenue_growth > 0 and revenue_growth < 1:
                revenue_growth *= 100
            
            # Análisis técnico
            rsi = 50  # Default
            macd_signal = "NEUTRAL"
            bollinger_position = "MIDDLE"
            trend_strength = 0
            volatility = 0
            volume_ratio = 1
            
            if not historical.empty and len(historical) > 1:
                latest = historical.iloc[-1]
                if 'RSI' in latest and not pd.isna(latest['RSI']):
                    rsi = latest['RSI']
                
                if 'MACD' in latest and 'MACD_Signal' in latest:
                    if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
                        macd_signal = "BUY" if latest['MACD'] > latest['MACD_Signal'] else "SELL"
                
                if 'BB_Percent' in latest and not pd.isna(latest['BB_Percent']):
                    bb_pct = latest['BB_Percent']
                    if bb_pct > 0.8:
                        bollinger_position = "UPPER"
                    elif bb_pct < 0.2:
                        bollinger_position = "LOWER"
                    else:
                        bollinger_position = "MIDDLE"
                
                # Fuerza de tendencia basada en SMA
                if 'SMA_20' in latest and 'SMA_50' in latest:
                    if not pd.isna(latest['SMA_20']) and not pd.isna(latest['SMA_50']):
                        if current_price > latest['SMA_20'] > latest['SMA_50']:
                            trend_strength = 1  # Tendencia alcista fuerte
                        elif current_price > latest['SMA_20'] and latest['SMA_20'] < latest['SMA_50']:
                            trend_strength = 0.5  # Tendencia mixta
                        elif current_price < latest['SMA_20'] < latest['SMA_50']:
                            trend_strength = -1  # Tendencia bajista fuerte
                        else:
                            trend_strength = -0.5  # Tendencia mixta bajista
                
                # Volatilidad
                if 'ATR' in latest and not pd.isna(latest['ATR']):
                    volatility = (latest['ATR'] / current_price) * 100
                
                # Ratio de volumen
                if 'Volume' in historical.columns:
                    avg_volume = historical['Volume'].tail(20).mean()
                    current_volume = latest['Volume']
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Análisis de recomendaciones
            analyst_rating = 3  # Neutral por defecto
            analyst_consensus = "HOLD"
            target_price = current_price
            upside_potential = 0
            
            if recommendations:
                latest_rec = recommendations[0]
                strong_buy = latest_rec.get('strongBuy', 0)
                buy = latest_rec.get('buy', 0)
                hold = latest_rec.get('hold', 0)
                sell = latest_rec.get('sell', 0)
                strong_sell = latest_rec.get('strongSell', 0)
                
                total_recs = strong_buy + buy + hold + sell + strong_sell
                if total_recs > 0:
                    # Calcular rating promedio (1=Strong Buy, 5=Strong Sell)
                    weighted_rating = (strong_buy * 1 + buy * 2 + hold * 3 + sell * 4 + strong_sell * 5) / total_recs
                    analyst_rating = weighted_rating
                    
                    # Consenso
                    buy_ratio = (strong_buy + buy) / total_recs
                    sell_ratio = (sell + strong_sell) / total_recs
                    
                    if buy_ratio > 0.6:
                        analyst_consensus = "BUY"
                    elif sell_ratio > 0.6:
                        analyst_consensus = "SELL"
                    else:
                        analyst_consensus = "HOLD"
            
            # Precio objetivo desde Yahoo Finance
            if yf_info.get('targetMeanPrice'):
                target_price = yf_info['targetMeanPrice']
                upside_potential = ((target_price - current_price) / current_price) * 100
            
            # Análisis de sentimiento de noticias (simplificado)
            news_sentiment = "NEUTRAL"
            if news:
                # Simplificado: contar palabras positivas vs negativas en headlines
                positive_words = ['up', 'rise', 'gain', 'beat', 'strong', 'growth', 'positive', 'buy']
                negative_words = ['down', 'fall', 'loss', 'miss', 'weak', 'decline', 'negative', 'sell']
                
                positive_count = 0
                negative_count = 0
                
                for article in news:
                    headline = article.get('headline', '').lower()
                    for word in positive_words:
                        positive_count += headline.count(word)
                    for word in negative_words:
                        negative_count += headline.count(word)
                
                if positive_count > negative_count:
                    news_sentiment = "POSITIVE"
                elif negative_count > positive_count:
                    news_sentiment = "NEGATIVE"
            
            # Calcular scores
            fundamental_score = self._calculate_fundamental_score(
                pe_ratio, pb_ratio, roe, roa, debt_to_equity, profit_margin, revenue_growth, market_cap
            )
            
            technical_score = self._calculate_technical_score(
                rsi, macd_signal, bollinger_position, trend_strength, volatility
            )
            
            sentiment_score = self._calculate_sentiment_score(
                analyst_rating, upside_potential, news_sentiment, volume_ratio
            )
            
            overall_score = (
                fundamental_score * self.scoring_weights['fundamental'] +
                technical_score * self.scoring_weights['technical'] +
                sentiment_score * self.scoring_weights['sentiment']
            )
            
            # Determinar nivel de riesgo
            risk_level = self._determine_risk_level(volatility, debt_to_equity, market_cap, sector)
            
            # Actividad de insiders (simplificado)
            insider_activity = "NEUTRAL"  # Placeholder
            
            return StockAnalysis(
                symbol=symbol,
                name=name,
                sector=sector,
                price=current_price,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                roe=roe,
                roa=roa,
                debt_to_equity=debt_to_equity,
                profit_margin=profit_margin,
                revenue_growth=revenue_growth,
                rsi=rsi,
                macd_signal=macd_signal,
                bollinger_position=bollinger_position,
                trend_strength=trend_strength,
                volatility=volatility,
                analyst_rating=analyst_rating,
                analyst_consensus=analyst_consensus,
                target_price=target_price,
                upside_potential=upside_potential,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                overall_score=overall_score,
                volume_ratio=volume_ratio,
                insider_activity=insider_activity,
                recent_news_sentiment=news_sentiment,
                risk_level=risk_level
            )
            
        except Exception as e:
            st.error(f"Error analizando {symbol}: {e}")
            return None
    
    def _calculate_fundamental_score(self, pe, pb, roe, roa, debt_eq, profit_margin, revenue_growth, market_cap):
        """Calcular score fundamental (0-100)"""
        score = 50  # Base neutral
        
        try:
            # P/E Ratio (mejor si es menor, pero no demasiado bajo)
            if isinstance(pe, (int, float)) and pe > 0:
                if 10 <= pe <= 20:
                    score += 15
                elif 5 <= pe < 10 or 20 < pe <= 30:
                    score += 10
                elif pe > 30:
                    score -= 10
            
            # P/B Ratio (mejor si es menor)
            if isinstance(pb, (int, float)) and pb > 0:
                if pb <= 1.5:
                    score += 10
                elif pb <= 3:
                    score += 5
                elif pb > 5:
                    score -= 5
            
            # ROE (mejor si es mayor)
            if isinstance(roe, (int, float)):
                if roe >= 20:
                    score += 15
                elif roe >= 15:
                    score += 10
                elif roe >= 10:
                    score += 5
                elif roe < 0:
                    score -= 15
            
            # ROA (mejor si es mayor)
            if isinstance(roa, (int, float)):
                if roa >= 10:
                    score += 10
                elif roa >= 5:
                    score += 5
                elif roa < 0:
                    score -= 10
            
            # Debt to Equity (mejor si es menor)
            if isinstance(debt_eq, (int, float)):
                if debt_eq <= 0.3:
                    score += 10
                elif debt_eq <= 0.6:
                    score += 5
                elif debt_eq > 1.5:
                    score -= 10
            
            # Profit Margin (mejor si es mayor)
            if isinstance(profit_margin, (int, float)):
                if profit_margin >= 20:
                    score += 10
                elif profit_margin >= 10:
                    score += 5
                elif profit_margin < 0:
                    score -= 15
            
            # Revenue Growth (mejor si es positivo y razonable)
            if isinstance(revenue_growth, (int, float)):
                if 10 <= revenue_growth <= 30:
                    score += 15
                elif 5 <= revenue_growth < 10:
                    score += 10
                elif revenue_growth > 50:
                    score += 5  # Crecimiento muy alto puede no ser sostenible
                elif revenue_growth < 0:
                    score -= 10
            
            # Market Cap (bonus para large caps)
            if isinstance(market_cap, (int, float)) and market_cap > 10000:  # >10B
                score += 5
        
        except Exception:
            pass
        
        return max(0, min(100, score))
    
    def _calculate_technical_score(self, rsi, macd_signal, bb_position, trend_strength, volatility):
        """Calcular score técnico (0-100)"""
        score = 50  # Base neutral
        
        try:
            # RSI
            if isinstance(rsi, (int, float)):
                if 30 <= rsi <= 70:
                    score += 10  # Zona neutral es buena
                elif rsi < 30:
                    score += 15  # Sobreventa = oportunidad
                elif rsi > 70:
                    score -= 10  # Sobrecompra = riesgo
            
            # MACD Signal
            if macd_signal == "BUY":
                score += 15
            elif macd_signal == "SELL":
                score -= 15
            
            # Bollinger Bands
            if bb_position == "LOWER":
                score += 10  # Posible rebote
            elif bb_position == "UPPER":
                score -= 5   # Posible corrección
            
            # Trend Strength
            if isinstance(trend_strength, (int, float)):
                score += trend_strength * 15  # +15 para tendencia alcista fuerte
            
            # Volatilidad (menor volatilidad es mejor para estabilidad)
            if isinstance(volatility, (int, float)):
                if volatility < 2:
                    score += 5
                elif volatility > 5:
                    score -= 10
        
        except Exception:
            pass
        
        return max(0, min(100, score))
    
    def _calculate_sentiment_score(self, analyst_rating, upside_potential, news_sentiment, volume_ratio):
        """Calcular score de sentimiento (0-100)"""
        score = 50  # Base neutral
        
        try:
            # Analyst Rating (1=Strong Buy, 5=Strong Sell)
            if isinstance(analyst_rating, (int, float)):
                if analyst_rating <= 2:
                    score += 20
                elif analyst_rating <= 2.5:
                    score += 10
                elif analyst_rating >= 4:
                    score -= 20
                elif analyst_rating >= 3.5:
                    score -= 10
            
            # Upside Potential
            if isinstance(upside_potential, (int, float)):
                if upside_potential > 20:
                    score += 15
                elif upside_potential > 10:
                    score += 10
                elif upside_potential < -10:
                    score -= 15
            
            # News Sentiment
            if news_sentiment == "POSITIVE":
                score += 10
            elif news_sentiment == "NEGATIVE":
                score -= 10
            
            # Volume Ratio (actividad inusual puede ser buena o mala)
            if isinstance(volume_ratio, (int, float)):
                if 1.5 <= volume_ratio <= 3:
                    score += 5  # Actividad elevada pero no excesiva
                elif volume_ratio > 5:
                    score -= 5  # Actividad excesiva puede indicar volatilidad
        
        except Exception:
            pass
        
        return max(0, min(100, score))
    
    def _determine_risk_level(self, volatility, debt_to_equity, market_cap, sector):
        """Determinar nivel de riesgo"""
        risk_score = 0
        
        # Volatilidad
        if isinstance(volatility, (int, float)):
            if volatility > 5:
                risk_score += 2
            elif volatility > 3:
                risk_score += 1
        
        # Deuda
        if isinstance(debt_to_equity, (int, float)):
            if debt_to_equity > 1:
                risk_score += 2
            elif debt_to_equity > 0.5:
                risk_score += 1
        
        # Market Cap (smaller = more risky)
        if isinstance(market_cap, (int, float)):
            if market_cap < 1000:  # <1B = Small cap
                risk_score += 2
            elif market_cap < 10000:  # <10B = Mid cap
                risk_score += 1
        
        # Sector
        high_risk_sectors = ["technology", "biotech", "crypto", "energy"]
        if any(risky in sector.lower() for risky in high_risk_sectors):
            risk_score += 1
        
        if risk_score >= 5:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def create_portfolio_summary(self, analyses: List[StockAnalysis]) -> pd.DataFrame:
        """Crear resumen de cartera"""
        data = []
        for analysis in analyses:
            data.append({
                'Symbol': analysis.symbol,
                'Name': analysis.name[:30] + "..." if len(analysis.name) > 30 else analysis.name,
                'Sector': analysis.sector,
                'Price': f"${analysis.price:.2f}",
                'Market Cap': f"${analysis.market_cap/1000:.1f}B" if analysis.market_cap > 0 else "N/A",
                'P/E': f"{analysis.pe_ratio:.1f}" if isinstance(analysis.pe_ratio, (int, float)) and analysis.pe_ratio > 0 else "N/A",
                'ROE': f"{analysis.roe:.1f}%" if isinstance(analysis.roe, (int, float)) else "N/A",
                'RSI': f"{analysis.rsi:.1f}" if isinstance(analysis.rsi, (int, float)) else "N/A",
                'Analyst': analysis.analyst_consensus,
                'Target Price': f"${analysis.target_price:.2f}" if analysis.target_price > 0 else "N/A",
                'Upside': f"{analysis.upside_potential:.1f}%" if isinstance(analysis.upside_potential, (int, float)) else "N/A",
                'Risk': analysis.risk_level,
                'Overall Score': f"{analysis.overall_score:.1f}",
                'Fundamental': f"{analysis.fundamental_score:.1f}",
                'Technical': f"{analysis.technical_score:.1f}",
                'Sentiment': f"{analysis.sentiment_score:.1f}"
            })
        
        return pd.DataFrame(data)
    
    def get_top_stocks_by_criteria(self, analyses: List[StockAnalysis], criteria: str, limit: int = 10) -> List[StockAnalysis]:
        """Obtener top acciones según diferentes criterios"""
        if criteria == "overall_score":
            return sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:limit]
        elif criteria == "upside_potential":
            return sorted([a for a in analyses if isinstance(a.upside_potential, (int, float))], 
                         key=lambda x: x.upside_potential, reverse=True)[:limit]
        elif criteria == "fundamental_score":
            return sorted(analyses, key=lambda x: x.fundamental_score, reverse=True)[:limit]
        elif criteria == "technical_score":
            return sorted(analyses, key=lambda x: x.technical_score, reverse=True)[:limit]
        elif criteria == "low_risk":
            low_risk = [a for a in analyses if a.risk_level == "LOW"]
            return sorted(low_risk, key=lambda x: x.overall_score, reverse=True)[:limit]
        elif criteria == "growth":
            growth_stocks = [a for a in analyses if isinstance(a.revenue_growth, (int, float)) and a.revenue_growth > 10]
            return sorted(growth_stocks, key=lambda x: x.revenue_growth, reverse=True)[:limit]
        elif criteria == "value":
            value_stocks = [a for a in analyses if isinstance(a.pe_ratio, (int, float)) and a.pe_ratio > 0 and a.pe_ratio < 20]
            return sorted(value_stocks, key=lambda x: x.pe_ratio)[:limit]
        elif criteria == "dividend":
            # Placeholder - necesitaríamos datos de dividendos
            return sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:limit]
        else:
            return analyses[:limit]