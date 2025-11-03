import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import finnhub
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Tuple
import json
import os

# Funciones de utilidad para formateo seguro
def format_number(value, decimal_places=0):
    """Formatear números de manera segura independientemente de la configuración regional"""
    if value is None or value == 'N/A':
        return 'N/A'
    
    try:
        if isinstance(value, str):
            value = float(value.replace(',', ''))
        
        if not isinstance(value, (int, float)):
            return str(value)
        
        if decimal_places == 0:
            return f"{int(value):,}".replace(',', '.')
        else:
            return f"{value:,.{decimal_places}f}".replace(',', '.')
    
    except (ValueError, TypeError):
        return str(value)

def format_currency(value, decimal_places=2):
    """Formatear valores de moneda"""
    if value is None or value == 'N/A':
        return 'N/A'
    
    try:
        if isinstance(value, str):
            value = float(value.replace(',', ''))
        
        if not isinstance(value, (int, float)):
            return str(value)
        
        formatted = f"{value:,.{decimal_places}f}".replace(',', '.')
        return f"${formatted}"
    
    except (ValueError, TypeError):
        return str(value)

def get_api_key():
    """
    Obtener API key de forma segura desde múltiples fuentes
    Prioridad: Streamlit secrets > Variables de entorno > Input del usuario
    """
    api_key = None
    
    # 1. Intentar desde Streamlit secrets (para deployments en Streamlit Cloud)
    try:
        api_key = st.secrets["FINNHUB_API_KEY"]
        if api_key:
            return api_key
    except:
        pass
    
    # 2. Intentar desde variables de entorno (para desarrollo local y otros deployments)
    try:
        api_key = os.getenv("FINNHUB_API_KEY")
        if api_key:
            return api_key
    except:
        pass
    
    # 3. Si no se encuentra, pedir al usuario (solo para desarrollo/testing)
    if 'api_key_input' not in st.session_state:
        st.session_state.api_key_input = ""
    
    if not api_key:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 Configuración API")
        st.sidebar.warning("⚠️ API Key no encontrada en configuración segura")
        
        api_key = st.sidebar.text_input(
            "Finnhub API Key", 
            value=st.session_state.api_key_input,
            type="password",
            help="Ingresa tu API key de Finnhub (solo para testing local)"
        )
        
        if api_key:
            st.session_state.api_key_input = api_key
        else:
            st.sidebar.error("🚫 Se requiere API key para continuar")
            st.stop()
    
    return api_key

# Configuración de la página
# st.set_page_config(
#     page_title="Trading Dashboard - Secure", 
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# CSS personalizado
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid #1f77b4;
}
.signal-buy {
    background-color: #d4edda;
    color: #155724;
    border-left: 5px solid #28a745;
}
.signal-sell {
    background-color: #f8d7da;
    color: #721c24;
    border-left: 5px solid #dc3545;
}
.signal-hold {
    background-color: #fff3cd;
    color: #856404;
    border-left: 5px solid #ffc107;
}
.news-item {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 3px solid #007bff;
}
.security-info {
    background-color: #e8f5e8;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border-left: 3px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

class HybridAPI:
    """Clase híbrida que usa Finnhub para datos en tiempo real y yfinance para históricos"""
    
    def __init__(self, finnhub_api_key: str):
        self.finnhub_api_key = finnhub_api_key
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    
    def get_quote(self, symbol: str) -> Dict:
        """Obtener cotización en tiempo real desde Finnhub"""
        try:
            quote = self.finnhub_client.quote(symbol)
            return {
                "symbol": symbol,
                "price": quote.get('c', 0),
                "change": quote.get('d', 0),
                "change_percent": quote.get('dp', 0),
                "high": quote.get('h', 0),
                "low": quote.get('l', 0),
                "open": quote.get('o', 0),
                "previous_close": quote.get('pc', 0),
                "timestamp": quote.get('t', 0)
            }
        except Exception as e:
            st.error(f"Error obteniendo cotización de {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """Obtener datos históricos desde Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                st.warning(f"No hay datos históricos disponibles para {symbol}")
                return pd.DataFrame()
            
            # Renombrar columnas para mantener compatibilidad
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            st.error(f"Error obteniendo datos históricos de {symbol}: {e}")
            return pd.DataFrame()
    
    def get_company_profile(self, symbol: str) -> Dict:
        """Obtener perfil de la empresa desde Finnhub"""
        try:
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            return profile
        except Exception as e:
            st.error(f"Error obteniendo perfil de {symbol}: {e}")
            return {}
    
    def get_basic_financials(self, symbol: str) -> Dict:
        """Obtener métricas financieras básicas desde Finnhub"""
        try:
            financials = self.finnhub_client.company_basic_financials(symbol, 'all')
            return financials
        except Exception as e:
            st.error(f"Error obteniendo financiales de {symbol}: {e}")
            return {}
    
    def get_recommendation_trends(self, symbol: str) -> List[Dict]:
        """Obtener tendencias de recomendaciones de analistas desde Finnhub"""
        try:
            recommendations = self.finnhub_client.recommendation_trends(symbol)
            return recommendations
        except Exception as e:
            st.error(f"Error obteniendo recomendaciones de {symbol}: {e}")
            return []
    
    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Obtener noticias de la empresa desde Finnhub"""
        try:
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            news = self.finnhub_client.company_news(symbol, _from=from_date, to=to_date)
            return news[:limit]
        except Exception as e:
            st.error(f"Error obteniendo noticias de {symbol}: {e}")
            return []
    
    def get_peers(self, symbol: str) -> List[str]:
        """Obtener empresas similares/competidoras desde Finnhub"""
        try:
            peers = self.finnhub_client.company_peers(symbol)
            return peers[:10]
        except Exception as e:
            st.error(f"Error obteniendo peers de {symbol}: {e}")
            return []
    
    def get_insider_transactions(self, symbol: str) -> List[Dict]:
        """Obtener transacciones de insiders desde Finnhub"""
        try:
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            transactions = self.finnhub_client.stock_insider_transactions(symbol, from_date, to_date)
            return transactions.get('data', [])[:20]
        except Exception as e:
            st.error(f"Error obteniendo transacciones insider de {symbol}: {e}")
            return []
    
    def get_market_status(self) -> Dict:
        """Obtener estado del mercado desde Finnhub"""
        try:
            status = self.finnhub_client.market_status(exchange='US')
            return status
        except Exception as e:
            st.error(f"Error obteniendo estado del mercado: {e}")
            return {}
    
    def get_company_info_yfinance(self, symbol: str) -> Dict:
        """Obtener información adicional de la empresa desde Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            st.warning(f"No se pudo obtener información adicional de {symbol}: {e}")
            return {}

class EnhancedTechnicalAnalysis:
    """Análisis técnico mejorado con más indicadores"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calcular todos los indicadores técnicos disponibles"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Indicadores de momentum
            df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            df['RSI_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Williams %R
            df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
            
            # MACD
            macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Width'] = bollinger.bollinger_wband()
            df['BB_Percent'] = bollinger.bollinger_pband()
            
            # Medias móviles
            df['SMA_10'] = ta.trend.SMAIndicator(df['Close'], window=10).sma_indicator()
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            
            # Indicadores de tendencia
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
            
            # Indicadores de volumen
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            # Calcular SMA del volumen manualmente
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
            # Indicadores de volatilidad
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            df['Keltner_Upper'] = keltner.keltner_channel_hband()
            df['Keltner_Lower'] = keltner.keltner_channel_lband()
            
        except Exception as e:
            st.warning(f"Error calculando algunos indicadores técnicos: {e}")
        
        return df
    
    @staticmethod
    def generate_enhanced_signals(df: pd.DataFrame, current_price: float) -> Dict:
        """Generar señales mejoradas con más indicadores"""
        if df.empty or len(df) < 2:
            return {"signal": "HOLD", "confidence": 0, "reasons": ["Datos insuficientes"], "score": 0}
        
        signals = []
        reasons = []
        score = 0
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # RSI (peso: 2)
        if 'RSI_14' in latest and not pd.isna(latest['RSI_14']):
            if latest['RSI_14'] < 30:
                signals.append("BUY")
                reasons.append(f"RSI sobreventa ({latest['RSI_14']:.1f})")
                score += 2
            elif latest['RSI_14'] > 70:
                signals.append("SELL")
                reasons.append(f"RSI sobrecompra ({latest['RSI_14']:.1f})")
                score -= 2
        
        # MACD (peso: 2)
        if all(col in latest for col in ['MACD', 'MACD_Signal']):
            if not pd.isna(latest['MACD']) and not pd.isna(prev['MACD']):
                if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                    signals.append("BUY")
                    reasons.append("MACD cruce alcista")
                    score += 2
                elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                    signals.append("SELL")
                    reasons.append("MACD cruce bajista")
                    score -= 2
        
        # Bollinger Bands (peso: 1.5)
        if all(col in latest for col in ['BB_Lower', 'BB_Upper', 'BB_Percent']):
            if not pd.isna(latest['BB_Percent']):
                if latest['BB_Percent'] < 0.2:
                    signals.append("BUY")
                    reasons.append("Precio en banda inferior Bollinger")
                    score += 1.5
                elif latest['BB_Percent'] > 0.8:
                    signals.append("SELL")
                    reasons.append("Precio en banda superior Bollinger")
                    score -= 1.5
        
        # Stochastic (peso: 1)
        if all(col in latest for col in ['Stoch_K', 'Stoch_D']):
            if not pd.isna(latest['Stoch_K']):
                if latest['Stoch_K'] < 20 and latest['Stoch_K'] > latest['Stoch_D']:
                    signals.append("BUY")
                    reasons.append("Stochastic sobreventa con divergencia")
                    score += 1
                elif latest['Stoch_K'] > 80 and latest['Stoch_K'] < latest['Stoch_D']:
                    signals.append("SELL")
                    reasons.append("Stochastic sobrecompra con divergencia")
                    score -= 1
        
        # Medias móviles (peso: 1)
        if all(col in latest for col in ['SMA_10', 'SMA_20']):
            if not pd.isna(latest['SMA_10']) and not pd.isna(latest['SMA_20']):
                if current_price > latest['SMA_10'] > latest['SMA_20']:
                    signals.append("BUY")
                    reasons.append("Precio sobre medias móviles alcistas")
                    score += 1
                elif current_price < latest['SMA_10'] < latest['SMA_20']:
                    signals.append("SELL")
                    reasons.append("Precio bajo medias móviles bajistas")
                    score -= 1
        
        # ADX para confirmar tendencia (peso: 0.5)
        if 'ADX' in latest and not pd.isna(latest['ADX']):
            if latest['ADX'] > 25:
                if score > 0:
                    score += 0.5
                    reasons.append(f"Tendencia fuerte confirmada (ADX: {latest['ADX']:.1f})")
                elif score < 0:
                    score -= 0.5
                    reasons.append(f"Tendencia fuerte confirmada (ADX: {latest['ADX']:.1f})")
        
        # Determinar señal final basada en score
        if score >= 3:
            final_signal = "BUY"
            confidence = min(85, 50 + (score * 5))
        elif score <= -3:
            final_signal = "SELL"
            confidence = min(85, 50 + (abs(score) * 5))
        else:
            final_signal = "HOLD"
            confidence = 50
        
        return {
            "signal": final_signal,
            "confidence": confidence,
            "reasons": reasons[:4],
            "score": score
        }

def create_enhanced_chart(df: pd.DataFrame, symbol: str, company_profile: Dict) -> go.Figure:
    """Crear gráfico mejorado con más indicadores"""
    if df.empty:
        return go.Figure()
    
    # Crear subplots - CORREGIDO: sin row_weights
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{symbol} - {company_profile.get("name", "")}', 
            'RSI & Stochastic', 
            'MACD', 
            'Volume & OBV'
        )
    )
    
    # Gráfico principal - Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Precio',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1
    )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Superior', 
                                line=dict(color='rgba(128,128,128,0.5)', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Media', 
                                line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Inferior', 
                                line=dict(color='rgba(128,128,128,0.5)', dash='dash')), row=1, col=1)
    
    # Medias móviles
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                                line=dict(color='orange', width=2)), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                                line=dict(color='purple', width=2)), row=1, col=1)
    
    # RSI y Stochastic
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI', 
                                line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch %K', 
                                line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='Stoch %D', 
                                line=dict(color='red')), row=2, col=1)
    
    # MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                                line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal', 
                                line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram',
                            marker_color='gray'), row=3, col=1)
    
    # Volumen y OBV
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volumen', 
                        marker_color='lightblue'), row=4, col=1)
    
    if 'OBV' in df.columns:
        # Crear segundo eje y para OBV
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', 
                                line=dict(color='orange')), row=4, col=1)
    
    # Actualizar layout
    fig.update_layout(
        title=f'Análisis Técnico Completo - {symbol}',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True
    )
    
    return fig

def create_recommendations_chart(recommendations: List[Dict]) -> go.Figure:
    """Crear gráfico de barras apiladas para recomendaciones por mes"""
    if not recommendations:
        return go.Figure()
    
    # Preparar datos para el gráfico
    periods = []
    strong_buy = []
    buy = []
    hold = []
    sell = []
    strong_sell = []
    
    # Ordenar por período y tomar últimos 12 meses
    recommendations_sorted = sorted(recommendations, key=lambda x: x.get('period', ''), reverse=True)
    
    for rec in recommendations_sorted[:12]:  # Últimos 12 meses
        period = rec.get('period', '')
        if period:
            # Convertir periodo a formato más legible
            try:
                period_date = datetime.strptime(period, '%Y-%m-%d')
                period_formatted = period_date.strftime('%Y-%m')
            except:
                period_formatted = period
            
            periods.append(period_formatted)
            strong_buy.append(rec.get('strongBuy', 0))
            buy.append(rec.get('buy', 0))
            hold.append(rec.get('hold', 0))
            sell.append(rec.get('sell', 0))
            strong_sell.append(rec.get('strongSell', 0))
    
    # Revertir listas para mostrar cronológicamente
    periods.reverse()
    strong_buy.reverse()
    buy.reverse()
    hold.reverse()
    sell.reverse()
    strong_sell.reverse()
    
    # Crear gráfico de barras apiladas
    fig = go.Figure()
    
    # Agregar cada categoría como una barra apilada
    fig.add_trace(go.Bar(
        name='Strong Buy',
        x=periods,
        y=strong_buy,
        marker_color='#006400',  # Verde oscuro
        text=strong_buy,
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='Buy',
        x=periods,
        y=buy,
        marker_color='#32CD32',  # Verde claro
        text=buy,
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='Hold',
        x=periods,
        y=hold,
        marker_color='#FFD700',  # Amarillo
        text=hold,
        textposition='inside',
        textfont=dict(color='black', size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='Sell',
        x=periods,
        y=sell,
        marker_color='#FF6347',  # Rojo claro
        text=sell,
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    fig.add_trace(go.Bar(
        name='Strong Sell',
        x=periods,
        y=strong_sell,
        marker_color='#8B0000',  # Rojo oscuro
        text=strong_sell,
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    # Configurar layout
    fig.update_layout(
        title='Tendencia de Recomendaciones de Analistas (Últimos 12 Meses)',
        xaxis_title='Período (Año-Mes)',
        yaxis_title='Número de Analistas',
        barmode='stack',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Personalizar hover
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Período: %{x}<br>' +
                      'Analistas: %{y}<br>' +
                      '<extra></extra>'
    )
    
    return fig

def display_company_info(profile: Dict, financials: Dict, yf_info: Dict):
    """Mostrar información de la empresa"""
    if not profile and not yf_info:
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Información General")
        name = profile.get('name') or yf_info.get('longName', 'N/A')
        st.write(f"**Nombre:** {name}")
        
        sector = profile.get('finnhubIndustry') or yf_info.get('sector', 'N/A')
        st.write(f"**Sector:** {sector}")
        
        country = profile.get('country') or yf_info.get('country', 'N/A')
        st.write(f"**País:** {country}")
        
        employees = profile.get('employeeTotal') or yf_info.get('fullTimeEmployees', 'N/A')
        st.write(f"**Empleados:** {format_number(employees)}")
        
        website = profile.get('weburl') or yf_info.get('website', '')
        if website:
            st.write(f"**Web:** [{website}]({website})")
    
    with col2:
        st.subheader("💰 Métricas Financieras")
        
        # Market Cap
        market_cap = None
        if profile.get('marketCapitalization'):
            market_cap = profile['marketCapitalization'] / 1000
        elif yf_info.get('marketCap'):
            market_cap = yf_info['marketCap'] / 1e9
        
        if market_cap:
            st.write(f"**Cap. Mercado:** {format_currency(market_cap, 1)}B")
        else:
            st.write("**Cap. Mercado:** N/A")
        
        # Métricas financieras
        if financials and 'metric' in financials:
            metrics = financials['metric']
            pe_ratio = metrics.get('peBasicExclExtraTTM', 'N/A')
            pb_ratio = metrics.get('pbAnnual', 'N/A')
            roe = metrics.get('roeRfy', 'N/A')
            roa = metrics.get('roaRfy', 'N/A')
        else:
            pe_ratio = yf_info.get('trailingPE', 'N/A')
            pb_ratio = yf_info.get('priceToBook', 'N/A')
            roe = yf_info.get('returnOnEquity', 'N/A')
            roa = yf_info.get('returnOnAssets', 'N/A')
        
        st.write(f"**P/E Ratio:** {pe_ratio}")
        st.write(f"**P/B Ratio:** {pb_ratio}")
        
        if roe != 'N/A' and isinstance(roe, (int, float)):
            if roe > 1:  # Convertir de decimal a porcentaje si es necesario
                roe = roe * 100
            st.write(f"**ROE:** {roe:.2f}%")
        else:
            st.write(f"**ROE:** {roe}")
        
        if roa != 'N/A' and isinstance(roa, (int, float)):
            if roa > 1:  # Convertir de decimal a porcentaje si es necesario
                roa = roa * 100
            st.write(f"**ROA:** {roa:.2f}%")
        else:
            st.write(f"**ROA:** {roa}")
    
    with col3:
        st.subheader("📈 Datos del Mercado")
        
        # Beta
        beta = 'N/A'
        if financials and 'metric' in financials:
            beta = financials['metric'].get('beta', 'N/A')
        elif yf_info.get('beta'):
            beta = yf_info.get('beta', 'N/A')
        st.write(f"**Beta:** {beta}")
        
        # 52W High/Low
        high_52w = 'N/A'
        low_52w = 'N/A'
        if financials and 'metric' in financials:
            high_52w = financials['metric'].get('52WeekHigh', 'N/A')
            low_52w = financials['metric'].get('52WeekLow', 'N/A')
        elif yf_info.get('fiftyTwoWeekHigh'):
            high_52w = yf_info.get('fiftyTwoWeekHigh')
            low_52w = yf_info.get('fiftyTwoWeekLow')
        
        st.write(f"**52W High:** {format_currency(high_52w)}")
        st.write(f"**52W Low:** {format_currency(low_52w)}")
        
        # Volume
        avg_volume = 'N/A'
        if financials and 'metric' in financials:
            avg_volume = financials['metric'].get('10DayAverageTradingVolume', 'N/A')
        elif yf_info.get('averageVolume'):
            avg_volume = yf_info.get('averageVolume')
        st.write(f"**Avg Volume:** {format_number(avg_volume)}")

def display_news(news_list: List[Dict]):
    """Mostrar noticias de la empresa"""
    if not news_list:
        st.info("No hay noticias recientes disponibles")
        return
    
    st.subheader("📰 Noticias Recientes")
    
    for i, news in enumerate(news_list[:5]):
        with st.container():
            st.markdown(f"""
            <div class="news-item">
                <h4><a href="{news.get('url', '#')}" target="_blank">{news.get('headline', 'Sin título')}</a></h4>
                <p><strong>Fuente:</strong> {news.get('source', 'N/A')} | 
                   <strong>Fecha:</strong> {datetime.fromtimestamp(news.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')}</p>
                <p>{news.get('summary', 'Sin resumen disponible')[:200]}...</p>
            </div>
            """, unsafe_allow_html=True)

def display_security_info():
    """Mostrar información sobre la configuración de seguridad"""
    st.markdown("""
    <div class="security-info">
        <h4>🔒 Configuración de Seguridad</h4>
        <p>Este dashboard utiliza múltiples métodos seguros para manejar las API keys:</p>
        <ul>
            <li><strong>Streamlit Secrets:</strong> Para deployments en Streamlit Cloud</li>
            <li><strong>Variables de entorno:</strong> Para deployments locales y en la nube</li>
            <li><strong>Input temporal:</strong> Solo para testing (no recomendado en producción)</li>
        </ul>
        <p><strong>✅ Las API keys nunca se almacenan en el código fuente</strong></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("📈 Trading Dashboard - Secure Version")
    st.markdown("*Datos en tiempo real: Finnhub | Datos históricos: Yahoo Finance | 🔒 API Keys Seguras*")
    
    # Mostrar información de seguridad
    with st.expander("🔒 Ver información de seguridad"):
        display_security_info()
    
    st.markdown("---")
    
    # Obtener API key de forma segura
    try:
        api_key = get_api_key()
        if not api_key:
            st.error("🚫 No se pudo obtener la API key")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error obteniendo API key: {e}")
        st.stop()
    
    # Inicializar API
    try:
        api = HybridAPI(api_key)
        st.success("✅ Conectado de forma segura a Finnhub API + Yahoo Finance")
    except Exception as e:
        st.error(f"❌ Error conectando: {e}")
        return
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Estado del mercado
        market_status = api.get_market_status()
        if market_status:
            status_emoji = "🟢" if market_status.get('isOpen', False) else "🔴"
            st.markdown(f"{status_emoji} **Mercado US:** {'Abierto' if market_status.get('isOpen', False) else 'Cerrado'}")
        
        st.markdown("---")
        
        # Selección de símbolos
        st.subheader("📊 Símbolos")
        popular_stocks = {
            "Tecnología": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
            "Financiero": ["JPM", "BAC", "WFC", "GS", "MS"],
            "Salud": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Energía": ["XOM", "CVX", "COP", "SLB"],
            "Consumo": ["KO", "PEP", "WMT", "HD", "MCD"]
        }
        
        selected_category = st.selectbox("Categoría", list(popular_stocks.keys()))
        selected_symbol = st.selectbox("Seleccionar símbolo", popular_stocks[selected_category])
        
        # Símbolo personalizado
        custom_symbol = st.text_input("Símbolo personalizado").upper()
        if custom_symbol:
            selected_symbol = custom_symbol
        
        # Configuración de datos históricos
        st.subheader("📈 Configuración Históricos")
        period_options = {
            "1 mes": "1mo",
            "3 meses": "3mo", 
            "6 meses": "6mo",
            "1 año": "1y",
            "2 años": "2y"
        }
        
        interval_options = {
            "1 día": "1d",
            "1 hora": "1h",
            "30 min": "30m",
            "15 min": "15m",
            "5 min": "5m"
        }
        
        selected_period = st.selectbox("Período", list(period_options.keys()), index=1)
        selected_interval = st.selectbox("Intervalo", list(interval_options.keys()))
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-actualizar", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Intervalo (segundos)", 10, 300, 60)
        
        if st.button("🔄 Actualizar ahora"):
            st.rerun()
    
    # Container principal
    main_container = st.container()
    
    with main_container:
        # Obtener datos del símbolo seleccionado
        with st.spinner(f"Obteniendo datos de {selected_symbol}..."):
            quote = api.get_quote(selected_symbol)
            company_profile = api.get_company_profile(selected_symbol)
            financials = api.get_basic_financials(selected_symbol)
            yf_info = api.get_company_info_yfinance(selected_symbol)
            historical_data = api.get_historical_data(
                selected_symbol, 
                period=period_options[selected_period],
                interval=interval_options[selected_interval]
            )
        
        if quote and quote["price"] > 0:
            # Métricas principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                change_color = "normal" if quote["change"] >= 0 else "inverse"
                st.metric(
                    label=f"💰 {quote['symbol']}",
                    value=f"${quote['price']:.2f}",
                    delta=f"{quote['change']:+.2f} ({quote['change_percent']:+.2f}%)",
                    delta_color=change_color
                )
            
            with col2:
                st.metric(
                    label="📈 Máximo",
                    value=f"${quote['high']:.2f}",
                )
            
            with col3:
                st.metric(
                    label="📉 Mínimo",
                    value=f"${quote['low']:.2f}",
                )
            
            with col4:
                st.metric(
                    label="🌅 Apertura",
                    value=f"${quote['open']:.2f}",
                )
            
            with col5:
                st.metric(
                    label="📊 Cierre Anterior",
                    value=f"${quote['previous_close']:.2f}",
                )
            
            # Información de la empresa
            if company_profile or yf_info:
                display_company_info(company_profile, financials, yf_info)
                st.markdown("---")
            
            # Análisis técnico y señales
            if not historical_data.empty:
                col_chart, col_signals = st.columns([2.5, 1])
                
                with col_chart:
                    # Calcular indicadores
                    historical_data = EnhancedTechnicalAnalysis.calculate_all_indicators(historical_data)
                    
                    # Mostrar gráfico
                    fig = create_enhanced_chart(historical_data, selected_symbol, company_profile)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_signals:
                    st.subheader("🎯 Señales de Trading")
                    
                    # Generar señales
                    signals = EnhancedTechnicalAnalysis.generate_enhanced_signals(historical_data, quote["price"])
                    
                    # Mostrar señal principal
                    signal_class = {
                        "BUY": "signal-buy",
                        "SELL": "signal-sell",
                        "HOLD": "signal-hold"
                    }
                    
                    signal_emoji = {"BUY": "🟢 COMPRAR", "SELL": "🔴 VENDER", "HOLD": "🟡 MANTENER"}
                    
                    st.markdown(f"""
                    <div class="metric-card {signal_class[signals['signal']]}">
                        <h3>{signal_emoji[signals['signal']]}</h3>
                        <p><strong>Confianza:</strong> {signals['confidence']:.1f}%</p>
                        <p><strong>Score:</strong> {signals['score']:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Razones
                    st.subheader("📋 Análisis")
                    for reason in signals['reasons']:
                        st.write(f"• {reason}")
                    
                    # Indicadores actuales
                    if len(historical_data) > 0:
                        latest = historical_data.iloc[-1]
                        st.subheader("📊 Indicadores Clave")
                        
                        if 'RSI_14' in latest and not pd.isna(latest['RSI_14']):
                            rsi_color = "🔴" if latest['RSI_14'] > 70 else "🟢" if latest['RSI_14'] < 30 else "🟡"
                            st.write(f"{rsi_color} **RSI (14):** {latest['RSI_14']:.1f}")
                        
                        if 'MACD' in latest and not pd.isna(latest['MACD']):
                            st.write(f"📈 **MACD:** {latest['MACD']:.3f}")
                        
                        if 'BB_Percent' in latest and not pd.isna(latest['BB_Percent']):
                            bb_pos = "Superior" if latest['BB_Percent'] > 0.8 else "Inferior" if latest['BB_Percent'] < 0.2 else "Media"
                            st.write(f"📊 **Bollinger:** {bb_pos} ({latest['BB_Percent']:.2f})")
                        
                        if 'ADX' in latest and not pd.isna(latest['ADX']):
                            trend_strength = "Fuerte" if latest['ADX'] > 25 else "Débil"
                            st.write(f"📈 **Tendencia:** {trend_strength} (ADX: {latest['ADX']:.1f})")
            
            else:
                st.error("❌ No se pudieron obtener datos históricos")
            
            # Pestañas adicionales
            tab1, tab2, tab3, tab4 = st.tabs(["📰 Noticias", "🏢 Competidores", "👔 Insiders", "📊 Recomendaciones"])
            
            with tab1:
                news = api.get_news(selected_symbol)
                display_news(news)
            
            with tab2:
                st.subheader("🏢 Empresas Similares")
                peers = api.get_peers(selected_symbol)
                if peers:
                    peers_data = []
                    for peer in peers[:8]:
                        try:
                            peer_quote = api.get_quote(peer)
                            if peer_quote and peer_quote["price"] > 0:
                                peers_data.append({
                                    "Símbolo": peer,
                                    "Precio": format_currency(peer_quote['price']),
                                    "Cambio": f"{peer_quote['change']:+.2f}",
                                    "Cambio %": f"{peer_quote['change_percent']:+.2f}%"
                                })
                        except:
                            continue
                    
                    if peers_data:
                        df_peers = pd.DataFrame(peers_data)
                        st.dataframe(df_peers, use_container_width=True)
                else:
                    st.info("No se encontraron empresas similares")
            
            with tab3:
                st.subheader("👔 Transacciones de Insiders")
                insider_data = api.get_insider_transactions(selected_symbol)
                if insider_data:
                    insider_df = []
                    for transaction in insider_data[:10]:
                        insider_df.append({
                            "Fecha": transaction.get('transactionDate', 'N/A'),
                            "Nombre": transaction.get('name', 'N/A'),
                            "Posición": transaction.get('position', 'N/A'),
                            "Transacción": transaction.get('transactionCode', 'N/A'),
                            "Acciones": format_number(transaction.get('share', 0)),
                            "Precio": format_currency(transaction.get('price', 0))
                        })
                    
                    if insider_df:
                        st.dataframe(pd.DataFrame(insider_df), use_container_width=True)
                else:
                    st.info("No hay transacciones de insiders recientes")
            
            with tab4:
                st.subheader("📊 Recomendaciones de Analistas")
                recommendations = api.get_recommendation_trends(selected_symbol)
                if recommendations:
                    # Gráfico de tendencias históricas
                    st.subheader("📈 Tendencia Histórica de Recomendaciones")
                    recommendations_chart = create_recommendations_chart(recommendations)
                    st.plotly_chart(recommendations_chart, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Recomendaciones actuales (mes más reciente)
                    st.subheader("📋 Recomendaciones Actuales")
                    latest_rec = recommendations[0]
                    
                    # Mostrar período actual
                    current_period = latest_rec.get('period', 'N/A')
                    if current_period:
                        try:
                            period_date = datetime.strptime(current_period, '%Y-%m-%d')
                            period_formatted = period_date.strftime('%B %Y')
                            st.info(f"📅 **Período:** {period_formatted}")
                        except:
                            st.info(f"📅 **Período:** {current_period}")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("🟢 Strong Buy", latest_rec.get('strongBuy', 0))
                    with col2:
                        st.metric("🟢 Buy", latest_rec.get('buy', 0))
                    with col3:
                        st.metric("🟡 Hold", latest_rec.get('hold', 0))
                    with col4:
                        st.metric("🔴 Sell", latest_rec.get('sell', 0))
                    with col5:
                        st.metric("🔴 Strong Sell", latest_rec.get('strongSell', 0))
                    
                    # Calcular consenso
                    total_recs = sum([latest_rec.get(k, 0) for k in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']])
                    if total_recs > 0:
                        buy_ratio = (latest_rec.get('strongBuy', 0) + latest_rec.get('buy', 0)) / total_recs
                        sell_ratio = (latest_rec.get('sell', 0) + latest_rec.get('strongSell', 0)) / total_recs
                        
                        if buy_ratio > 0.6:
                            consensus = "🟢 CONSENSO: COMPRAR"
                            consensus_color = "green"
                        elif sell_ratio > 0.6:
                            consensus = "🔴 CONSENSO: VENDER"
                            consensus_color = "red"
                        else:
                            consensus = "🟡 CONSENSO: MANTENER"
                            consensus_color = "orange"
                        
                        st.markdown(f"### {consensus}")
                        
                        # Mostrar distribución porcentual
                        st.subheader("📊 Distribución Porcentual")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            buy_total = latest_rec.get('strongBuy', 0) + latest_rec.get('buy', 0)
                            buy_pct = (buy_total / total_recs) * 100 if total_recs > 0 else 0
                            st.metric("🟢 Comprar", f"{buy_pct:.1f}%")
                        
                        with col_b:
                            hold_pct = (latest_rec.get('hold', 0) / total_recs) * 100 if total_recs > 0 else 0
                            st.metric("🟡 Mantener", f"{hold_pct:.1f}%")
                        
                        with col_c:
                            sell_total = latest_rec.get('sell', 0) + latest_rec.get('strongSell', 0)
                            sell_pct = (sell_total / total_recs) * 100 if total_recs > 0 else 0
                            st.metric("🔴 Vender", f"{sell_pct:.1f}%")
                    
                    # Análisis de tendencia
                    if len(recommendations) >= 2:
                        st.subheader("📈 Análisis de Tendencia")
                        current = recommendations[0]
                        previous = recommendations[1]
                        
                        current_buy = current.get('strongBuy', 0) + current.get('buy', 0)
                        previous_buy = previous.get('strongBuy', 0) + previous.get('buy', 0)
                        
                        if current_buy > previous_buy:
                            trend_emoji = "📈"
                            trend_text = "Las recomendaciones de compra han aumentado respecto al mes anterior"
                            trend_color = "green"
                        elif current_buy < previous_buy:
                            trend_emoji = "📉"
                            trend_text = "Las recomendaciones de compra han disminuido respecto al mes anterior"
                            trend_color = "red"
                        else:
                            trend_emoji = "➡️"
                            trend_text = "Las recomendaciones se mantienen estables respecto al mes anterior"
                            trend_color = "blue"
                        
                        st.markdown(f"{trend_emoji} {trend_text}")
                        
                        # Mostrar cambios específicos
                        change_buy = current_buy - previous_buy
                        change_hold = current.get('hold', 0) - previous.get('hold', 0)
                        change_sell = (current.get('sell', 0) + current.get('strongSell', 0)) - (previous.get('sell', 0) + previous.get('strongSell', 0))
                        
                        col_change1, col_change2, col_change3 = st.columns(3)
                        with col_change1:
                            st.metric("Cambio Comprar", f"{change_buy:+d}", delta=change_buy)
                        with col_change2:
                            st.metric("Cambio Mantener", f"{change_hold:+d}", delta=change_hold)
                        with col_change3:
                            st.metric("Cambio Vender", f"{change_sell:+d}", delta=change_sell)
                
                else:
                    st.info("No hay recomendaciones de analistas disponibles")
        
        else:
            st.error("❌ No se pudieron obtener datos del símbolo seleccionado")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

if __name__ == "__main__":
    main()