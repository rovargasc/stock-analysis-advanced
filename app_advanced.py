import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List
import json

# Importar módulos personalizados
from portfolio_analyzer import PortfolioAnalyzer, StockAnalysis
from advanced_analytics import (
    AdvancedAnalytics, create_earnings_calendar_tracker, 
    generate_trading_alerts, create_monte_carlo_simulation,
    create_performance_attribution_chart
)

# Importar funciones del dashboard original
from trading_dashboard import (
    get_api_key, format_number, format_currency, 
    HybridAPI, EnhancedTechnicalAnalysis, create_enhanced_chart,
    display_company_info, display_news, create_recommendations_chart
)

def main():
    st.set_page_config(
        page_title="Advanced Stock Analysis Platform", 
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado mejorado
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #007bff;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stock-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .performance-positive {
        color: #28a745;
        font-weight: bold;
    }
    .performance-negative {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Stock Analysis Platform</h1>
        <p>Análisis inteligente de acciones individuales y carteras de inversión</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Obtener API key
    try:
        api_key = get_api_key()
        if not api_key:
            st.error("🚫 No se pudo obtener la API key")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error obteniendo API key: {e}")
        st.stop()
    
    # Inicializar APIs
    try:
        hybrid_api = HybridAPI(api_key)
        portfolio_analyzer = PortfolioAnalyzer(api_key)
        st.success("✅ Conexión establecida con APIs de mercado")
    except Exception as e:
        st.error(f"❌ Error conectando: {e}")
        return
    
    # Sidebar para navegación principal
    with st.sidebar:
        st.header("🎛️ Panel de Control")
        
        analysis_mode = st.radio(
            "Modo de Análisis:",
            ["📊 Análisis Individual", "🏆 Screening de Cartera", "💼 Gestión de Cartera", "⚡ Alertas y Oportunidades"],
            index=0
        )
        
        st.markdown("---")
        
        # Estado del mercado
        market_status = hybrid_api.get_market_status()
        if market_status:
            status_emoji = "🟢" if market_status.get('isOpen', False) else "🔴"
            st.markdown(f"{status_emoji} **Mercado US:** {'Abierto' if market_status.get('isOpen', False) else 'Cerrado'}")
    
    # Modo 1: Análisis Individual (tu dashboard original mejorado)
    if analysis_mode == "📊 Análisis Individual":
        st.header("📊 Análisis Detallado de Acción Individual")
        
        with st.sidebar:
            st.subheader("🔍 Selección de Acción")
            
            # Categorías de acciones populares
            popular_stocks = {
                "🏆 Top Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
                "🏦 Financial": ["JPM", "BAC", "WFC", "GS", "V", "MA"],
                "🏥 Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
                "🛒 Consumer": ["WMT", "HD", "PG", "KO", "PEP", "MCD"],
                "⚡ Energy": ["XOM", "CVX", "COP", "EOG", "SLB"]
            }
            
            selected_category = st.selectbox("Categoría:", list(popular_stocks.keys()))
            selected_symbol = st.selectbox("Símbolo:", popular_stocks[selected_category])
            
            custom_symbol = st.text_input("🔤 Símbolo personalizado:").upper()
            if custom_symbol:
                selected_symbol = custom_symbol
            
            # Configuración de datos históricos
            st.subheader("📈 Configuración")
            period_options = {"1 mes": "1mo", "3 meses": "3mo", "6 meses": "6mo", "1 año": "1y"}
            selected_period = st.selectbox("Período histórico:", list(period_options.keys()), index=1)
            
            if st.button("🔄 Analizar Acción"):
                st.session_state.analyze_individual = True
                st.session_state.individual_symbol = selected_symbol
                st.session_state.individual_period = period_options[selected_period]
        
        # Ejecutar análisis individual
        if st.session_state.get('analyze_individual', False):
            symbol = st.session_state.get('individual_symbol', selected_symbol)
            period = st.session_state.get('individual_period', '3mo')
            
            with st.spinner(f"🔍 Analizando {symbol}..."):
                # Obtener datos completos
                quote = hybrid_api.get_quote(symbol)
                company_profile = hybrid_api.get_company_profile(symbol)
                financials = hybrid_api.get_basic_financials(symbol)
                yf_info = hybrid_api.get_company_info_yfinance(symbol)
                historical_data = hybrid_api.get_historical_data(symbol, period=period)
                recommendations = hybrid_api.get_recommendation_trends(symbol)
                news = hybrid_api.get_news(symbol)
                
                # Análisis con el portfolio analyzer para obtener scores
                stock_data = {
                    'quote': quote,
                    'profile': company_profile,
                    'financials': financials,
                    'yf_info': yf_info,
                    'historical': historical_data,
                    'recommendations': recommendations,
                    'news': news
                }
                
                analysis = portfolio_analyzer.analyze_stock(symbol, stock_data)
            
            if quote and quote["price"] > 0:
                # Métricas principales con diseño mejorado
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    change_color = "normal" if quote["change"] >= 0 else "inverse"
                    st.metric(
                        label=f"💰 {symbol}",
                        value=f"${quote['price']:.2f}",
                        delta=f"{quote['change']:+.2f} ({quote['change_percent']:+.2f}%)",
                        delta_color=change_color
                    )
                
                with col2:
                    st.metric("📈 Máximo", f"${quote['high']:.2f}")
                
                with col3:
                    st.metric("📉 Mínimo", f"${quote['low']:.2f}")
                
                with col4:
                    if analysis:
                        score_emoji = "🟢" if analysis.overall_score >= 70 else "🟡" if analysis.overall_score >= 50 else "🔴"
                        st.metric(f"{score_emoji} Score General", f"{analysis.overall_score:.1f}/100")
                    else:
                        st.metric("📊 Score", "N/A")
                
                with col5:
                    if analysis and isinstance(analysis.upside_potential, (int, float)):
                        upside_emoji = "🚀" if analysis.upside_potential > 20 else "📈" if analysis.upside_potential > 0 else "📉"
                        st.metric(f"{upside_emoji} Upside Potential", f"{analysis.upside_potential:.1f}%")
                    else:
                        st.metric("🎯 Upside", "N/A")
                
                # Información de la empresa
                if company_profile or yf_info:
                    with st.expander("🏢 Información de la Empresa", expanded=True):
                        display_company_info(company_profile, financials, yf_info)
                
                # Análisis técnico avanzado
                if not historical_data.empty:
                    st.header("📈 Análisis Técnico Avanzado")
                    
                    col_chart, col_signals = st.columns([2.5, 1])
                    
                    with col_chart:
                        # Calcular indicadores técnicos
                        historical_data = EnhancedTechnicalAnalysis.calculate_all_indicators(historical_data)
                        
                        # Crear gráfico mejorado
                        fig = create_enhanced_chart(historical_data, symbol, company_profile)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_signals:
                        st.subheader("🎯 Señales de Trading")
                        
                        # Generar señales
                        signals = EnhancedTechnicalAnalysis.generate_enhanced_signals(historical_data, quote["price"])
                        
                        # Mostrar señal principal
                        signal_class = {
                            "BUY": "alert-high",
                            "SELL": "alert-high", 
                            "HOLD": "alert-medium"
                        }
                        
                        signal_emoji = {"BUY": "🟢 COMPRAR", "SELL": "🔴 VENDER", "HOLD": "🟡 MANTENER"}
                        signal_color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
                        
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: {signal_color[signals['signal']]};">
                            <h3>{signal_emoji[signals['signal']]}</h3>
                            <p><strong>Confianza:</strong> {signals['confidence']:.1f}%</p>
                            <p><strong>Score Técnico:</strong> {signals['score']:.1f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mostrar razones
                        st.subheader("📋 Análisis")
                        for reason in signals['reasons']:
                            st.write(f"• {reason}")
                        
                        # Scores detallados si hay análisis completo
                        if analysis:
                            st.subheader("📊 Scores Detallados")
                            
                            score_data = {
                                'Categoria': ['Fundamental', 'Técnico', 'Sentimiento', 'General'],
                                'Score': [analysis.fundamental_score, analysis.technical_score, 
                                         analysis.sentiment_score, analysis.overall_score],
                                'Color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                            }
                            
                            fig_scores = go.Figure(data=[
                                go.Bar(x=score_data['Categoria'], y=score_data['Score'], 
                                      marker_color=score_data['Color'])
                            ])
                            fig_scores.update_layout(
                                title="Breakdown de Scores", 
                                height=300,
                                yaxis_range=[0, 100]
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)
                
                # Pestañas adicionales con más información
                tab1, tab2, tab3, tab4 = st.tabs(["📰 Noticias", "📊 Recomendaciones", "🏢 Competidores", "🔮 Proyecciones"])
                
                with tab1:
                    display_news(news)
                
                with tab2:
                    if recommendations:
                        rec_chart = create_recommendations_chart(recommendations)
                        st.plotly_chart(rec_chart, use_container_width=True)
                        
                        # Mostrar último consenso
                        latest_rec = recommendations[0]
                        st.subheader("📋 Consenso Actual")
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
                
                with tab3:
                    st.subheader("🏢 Análisis de Competidores")
                    peers = hybrid_api.get_peers(symbol)
                    if peers:
                        peer_data = []
                        for peer in peers[:8]:
                            try:
                                peer_quote = hybrid_api.get_quote(peer)
                                if peer_quote and peer_quote["price"] > 0:
                                    peer_data.append({
                                        "Símbolo": peer,
                                        "Precio": f"${peer_quote['price']:.2f}",
                                        "Cambio": f"{peer_quote['change']:+.2f}",
                                        "Cambio %": f"{peer_quote['change_percent']:+.2f}%"
                                    })
                            except:
                                continue
                        
                        if peer_data:
                            st.dataframe(pd.DataFrame(peer_data), use_container_width=True)
                    else:
                        st.info("No se encontraron competidores")
                
                with tab4:
                    if analysis:
                        st.subheader("🔮 Proyecciones y Simulación")
                        
                        # Simulación Monte Carlo simple
                        monte_carlo_results = create_monte_carlo_simulation([analysis], days=252, simulations=500)
                        
                        if symbol in monte_carlo_results:
                            mc_result = monte_carlo_results[symbol]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Precio Objetivo (Monte Carlo)", f"${mc_result['final_price_mean']:.2f}")
                            with col2:
                                st.metric("Probabilidad de Ganancia", f"{mc_result['probability_profit']:.1f}%")
                            with col3:
                                st.metric("Rango 90% (5%-95%)", 
                                         f"${mc_result['percentile_5']:.2f} - ${mc_result['percentile_95']:.2f}")
                        
                        # Información de precio objetivo de analistas
                        if analysis.target_price > 0:
                            st.info(f"🎯 **Precio objetivo de analistas:** ${analysis.target_price:.2f} "
                                   f"(Upside: {analysis.upside_potential:.1f}%)")
                
                # Reset flag
                st.session_state.analyze_individual = False
            
            else:
                st.error("❌ No se pudieron obtener datos de la acción")
    
    # Modo 2: Screening de Cartera (funcionalidad nueva principal)
    elif analysis_mode == "🏆 Screening de Cartera":
        st.header("🏆 Screening Inteligente de Acciones")
        
        with st.sidebar:
            st.subheader("🎯 Configuración de Screening")
            
            # Selección de universo
            available_categories = list(portfolio_analyzer.stock_universe.keys())
            selected_categories = st.multiselect(
                "Categorías de acciones:",
                available_categories,
                default=["Large Cap Tech", "Financial", "Healthcare"]
            )
            
            stocks_per_category = st.slider("Acciones por categoría:", 3, 10, 5)
            
            # Filtros
            st.subheader("🔍 Filtros Avanzados")
            min_market_cap = st.selectbox(
                "Market Cap mínimo:",
                ["Sin límite", "> $1B", "> $10B", "> $50B"],
                index=1
            )
            
            min_score = st.slider("Score mínimo:", 0, 100, 50)
            max_pe = st.slider("P/E máximo:", 0, 100, 50)
            min_roe = st.slider("ROE mínimo (%):", -20, 50, 5)
            
            risk_levels = st.multiselect(
                "Niveles de riesgo:",
                ["LOW", "MEDIUM", "HIGH"],
                default=["LOW", "MEDIUM"]
            )
            
            # Pesos para scoring
            st.subheader("⚖️ Pesos de Análisis")
            fund_weight = st.slider("Fundamental:", 0.0, 1.0, 0.4, 0.1)
            tech_weight = st.slider("Técnico:", 0.0, 1.0, 0.3, 0.1)
            sent_weight = 1.0 - fund_weight - tech_weight
            
            if sent_weight >= 0:
                st.info(f"Sentimiento: {sent_weight:.1f}")
                portfolio_analyzer.scoring_weights = {
                    'fundamental': fund_weight,
                    'technical': tech_weight, 
                    'sentiment': sent_weight
                }
            else:
                st.error("Los pesos deben sumar máximo 1.0")
            
            if st.button("🚀 Ejecutar Screening", type="primary"):
                st.session_state.run_screening = True
        
        # Ejecutar screening
        if st.session_state.get('run_screening', False):
            if not selected_categories:
                st.error("⚠️ Selecciona al menos una categoría")
                return
            
            # Preparar símbolos
            symbols_to_analyze = []
            for category in selected_categories:
                symbols_to_analyze.extend(portfolio_analyzer.stock_universe[category][:stocks_per_category])
            
            progress_container = st.container()
            with progress_container:
                st.info(f"🔍 Analizando {len(symbols_to_analyze)} acciones...")
                
                # Obtener datos
                with st.spinner("Obteniendo datos de mercado..."):
                    stock_data = portfolio_analyzer.get_stock_data_parallel(symbols_to_analyze, max_workers=10)
                
                # Analizar
                analyses = []
                with st.spinner("Ejecutando análisis..."):
                    for symbol, data in stock_data.items():
                        analysis = portfolio_analyzer.analyze_stock(symbol, data)
                        if analysis:
                            analyses.append(analysis)
                
                # Aplicar filtros
                filtered_analyses = []
                for analysis in analyses:
                    # Filtros
                    if min_market_cap != "Sin límite":
                        min_cap = {"> $1B": 1000, "> $10B": 10000, "> $50B": 50000}[min_market_cap]
                        if analysis.market_cap < min_cap:
                            continue
                    
                    if analysis.overall_score < min_score:
                        continue
                    
                    if isinstance(analysis.pe_ratio, (int, float)) and analysis.pe_ratio > max_pe:
                        continue
                    
                    if isinstance(analysis.roe, (int, float)) and analysis.roe < min_roe:
                        continue
                    
                    if analysis.risk_level not in risk_levels:
                        continue
                    
                    filtered_analyses.append(analysis)
            
            if filtered_analyses:
                st.success(f"✅ {len(filtered_analyses)} acciones encontradas")
                st.session_state.screening_results = filtered_analyses
                st.session_state.run_screening = False
            else:
                st.error("❌ No se encontraron acciones que cumplan los criterios")
                return
        
        # Mostrar resultados del screening
        if 'screening_results' in st.session_state:
            analyses = st.session_state.screening_results
            
            # Resumen ejecutivo
            st.header("📈 Resumen Ejecutivo")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                avg_score = np.mean([a.overall_score for a in analyses])
                st.metric("Score Promedio", f"{avg_score:.1f}")
            
            with col2:
                best_stock = max(analyses, key=lambda x: x.overall_score)
                st.metric("Mejor Acción", best_stock.symbol, f"{best_stock.overall_score:.1f}")
            
            with col3:
                buy_count = len([a for a in analyses if a.analyst_consensus == "BUY"])
                st.metric("Recomendaciones BUY", buy_count)
            
            with col4:
                high_upside = len([a for a in analyses if isinstance(a.upside_potential, (int, float)) and a.upside_potential > 20])
                st.metric("Alto Upside (>20%)", high_upside)
            
            with col5:
                low_risk = len([a for a in analyses if a.risk_level == "LOW"])
                st.metric("Bajo Riesgo", low_risk)
            
            # Pestañas de resultados
            tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📊 Visualizaciones", "📋 Tabla Completa", "🔍 Análisis Detallado"])
            
            with tab1:
                st.subheader("🏆 Top Performers")
                
                ranking_type = st.selectbox(
                    "Criterio de ranking:",
                    ["Score General", "Potencial Upside", "Score Fundamental", "Score Técnico", "Menor Riesgo"]
                )
                
                # Ordenar según criterio
                if ranking_type == "Score General":
                    top_stocks = sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:10]
                elif ranking_type == "Potencial Upside":
                    top_stocks = sorted([a for a in analyses if isinstance(a.upside_potential, (int, float))], 
                                       key=lambda x: x.upside_potential, reverse=True)[:10]
                elif ranking_type == "Score Fundamental":
                    top_stocks = sorted(analyses, key=lambda x: x.fundamental_score, reverse=True)[:10]
                elif ranking_type == "Score Técnico":
                    top_stocks = sorted(analyses, key=lambda x: x.technical_score, reverse=True)[:10]
                else:  # Menor Riesgo
                    low_risk_stocks = [a for a in analyses if a.risk_level == "LOW"]
                    top_stocks = sorted(low_risk_stocks, key=lambda x: x.overall_score, reverse=True)[:10]
                
                # Mostrar top stocks como tarjetas
                for i, analysis in enumerate(top_stocks, 1):
                    with st.container():
                        col_rank, col_info = st.columns([1, 4])
                        
                        with col_rank:
                            st.markdown(f"### #{i}")
                        
                        with col_info:
                            score_color = "#28a745" if analysis.overall_score >= 70 else "#ffc107" if analysis.overall_score >= 50 else "#dc3545"
                            upside_text = f"{analysis.upside_potential:.1f}%" if isinstance(analysis.upside_potential, (int, float)) else "N/A"
                            
                            st.markdown(f"""
                            <div class="stock-card">
                                <h4>{analysis.symbol} - {analysis.name[:30]}...</h4>
                                <div style="display: flex; justify-content: space-between;">
                                    <div><strong>Precio:</strong> ${analysis.price:.2f}</div>
                                    <div><strong>Sector:</strong> {analysis.sector}</div>
                                    <div style="color: {score_color};"><strong>Score:</strong> {analysis.overall_score:.1f}/100</div>
                                    <div><strong>Upside:</strong> {upside_text}</div>
                                    <div><strong>Riesgo:</strong> {analysis.risk_level}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            with tab2:
                st.subheader("📊 Análisis Visual")
                
                # Gráfico de sectores
                from advanced_trading_dashboard import create_sector_analysis_chart
                sector_chart = create_sector_analysis_chart(analyses)
                st.plotly_chart(sector_chart, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Riesgo vs Retorno
                    from advanced_trading_dashboard import create_risk_return_chart
                    risk_return_chart = create_risk_return_chart(analyses)
                    st.plotly_chart(risk_return_chart, use_container_width=True)
                
                with col2:
                    # Scores comparison
                    from advanced_trading_dashboard import create_score_comparison_chart  
                    score_chart = create_score_comparison_chart(analyses)
                    st.plotly_chart(score_chart, use_container_width=True)
                
                # Heatmap de métricas
                from advanced_trading_dashboard import create_metrics_heatmap
                heatmap = create_metrics_heatmap(analyses)
                st.plotly_chart(heatmap, use_container_width=True)
                
                # Correlaciones
                correlation_heatmap = AdvancedAnalytics.create_correlation_heatmap(analyses)
                if correlation_heatmap.data:
                    st.plotly_chart(correlation_heatmap, use_container_width=True)
            
            with tab3:
                st.subheader("📋 Tabla Completa de Resultados")
                
                # Crear tabla completa
                summary_df = portfolio_analyzer.create_portfolio_summary(analyses)
                
                # Añadir filtros interactivos
                col1, col2, col3 = st.columns(3)
                with col1:
                    sector_filter = st.multiselect(
                        "Filtrar por sector:",
                        options=summary_df['Sector'].unique(),
                        default=summary_df['Sector'].unique()
                    )
                
                with col2:
                    min_score_filter = st.slider("Score mínimo (tabla):", 0, 100, 0)
                
                with col3:
                    risk_filter = st.multiselect(
                        "Filtrar por riesgo:",
                        options=summary_df['Risk'].unique(),
                        default=summary_df['Risk'].unique()
                    )
                
                # Aplicar filtros
                filtered_df = summary_df[
                    (summary_df['Sector'].isin(sector_filter)) &
                    (summary_df['Overall Score'].astype(float) >= min_score_filter) &
                    (summary_df['Risk'].isin(risk_filter))
                ]
                
                st.dataframe(filtered_df, use_container_width=True, height=600)
                
                # Botón de exportación
                if st.button("📥 Exportar a Excel"):
                    filename = AdvancedAnalytics.export_analysis_to_excel(analyses)
                    if filename:
                        st.success(f"✅ Análisis exportado a {filename}")
                    else:
                        st.error("❌ Error exportando archivo")
            
            with tab4:
                st.subheader("🔍 Análisis Detallado Individual")
                
                # Selector para análisis detallado
                symbol_options = [f"{a.symbol} - {a.name[:20]}..." for a in analyses]
                selected_detailed = st.selectbox("Seleccionar acción para análisis detallado:", symbol_options)
                
                if selected_detailed:
                    symbol = selected_detailed.split(" - ")[0]
                    detailed_analysis = next(a for a in analyses if a.symbol == symbol)
                    
                    # Mostrar análisis detallado (similar al modo individual)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 💰 Información Básica")
                        st.write(f"**Símbolo:** {detailed_analysis.symbol}")
                        st.write(f"**Nombre:** {detailed_analysis.name}")
                        st.write(f"**Sector:** {detailed_analysis.sector}")
                        st.write(f"**Precio:** ${detailed_analysis.price:.2f}")
                        if detailed_analysis.market_cap > 0:
                            st.write(f"**Market Cap:** ${detailed_analysis.market_cap/1000:.1f}B")
                    
                    with col2:
                        st.markdown("### 📊 Métricas Financieras")
                        if isinstance(detailed_analysis.pe_ratio, (int, float)) and detailed_analysis.pe_ratio > 0:
                            st.write(f"**P/E:** {detailed_analysis.pe_ratio:.1f}")
                        if isinstance(detailed_analysis.roe, (int, float)):
                            st.write(f"**ROE:** {detailed_analysis.roe:.1f}%")
                        if isinstance(detailed_analysis.revenue_growth, (int, float)):
                            st.write(f"**Crecimiento:** {detailed_analysis.revenue_growth:.1f}%")
                        st.write(f"**Riesgo:** {detailed_analysis.risk_level}")
                    
                    with col3:
                        st.markdown("### 🎯 Análisis y Proyecciones")
                        st.write(f"**RSI:** {detailed_analysis.rsi:.1f}")
                        st.write(f"**Consenso:** {detailed_analysis.analyst_consensus}")
                        if isinstance(detailed_analysis.upside_potential, (int, float)):
                            st.write(f"**Upside:** {detailed_analysis.upside_potential:.1f}%")
                        if detailed_analysis.target_price > 0:
                            st.write(f"**Precio Objetivo:** ${detailed_analysis.target_price:.2f}")
                    
                    # Gráfico de scores
                    score_data = {
                        'Categoria': ['Fundamental', 'Técnico', 'Sentimiento', 'General'],
                        'Score': [detailed_analysis.fundamental_score, detailed_analysis.technical_score,
                                 detailed_analysis.sentiment_score, detailed_analysis.overall_score]
                    }
                    
                    fig_detailed = px.bar(
                        x=score_data['Categoria'], 
                        y=score_data['Score'],
                        title=f"Breakdown de Scores - {detailed_analysis.symbol}",
                        color=score_data['Score'],
                        color_continuous_scale='RdYlGn'
                    )
                    fig_detailed.update_layout(height=400)
                    st.plotly_chart(fig_detailed, use_container_width=True)
    
    # Modo 3: Gestión de Cartera
    elif analysis_mode == "💼 Gestión de Cartera":
        st.header("💼 Construcción y Optimización de Cartera")
        
        if 'screening_results' not in st.session_state:
            st.info("🔍 Primero ejecuta un screening de acciones en la sección 'Screening de Cartera'")
            return
        
        analyses = st.session_state.screening_results
        
        with st.sidebar:
            st.subheader("⚙️ Configuración de Cartera")
            
            portfolio_size = st.slider("Tamaño de cartera:", 3, 15, 8)
            
            strategy = st.selectbox(
                "Estrategia:",
                ["Balanceada", "Crecimiento", "Valor", "Bajo Riesgo", "Alto Potencial"]
            )
            
            max_risk_exposure = st.slider("% máx. alto riesgo:", 0, 50, 20)
            min_sectors = st.slider("Sectores mínimos:", 2, 8, 3)
            
            if st.button("🎯 Generar Cartera Optimizada"):
                st.session_state.generate_portfolio = True
        
        # Generar cartera
        if st.session_state.get('generate_portfolio', False):
            # Selección de acciones según estrategia
            if strategy == "Balanceada":
                selected_stocks = sorted(analyses, key=lambda x: x.overall_score, reverse=True)[:portfolio_size]
            elif strategy == "Crecimiento":
                growth_stocks = [a for a in analyses if isinstance(a.revenue_growth, (int, float)) and a.revenue_growth > 5]
                selected_stocks = sorted(growth_stocks, key=lambda x: (x.revenue_growth, x.overall_score), reverse=True)[:portfolio_size]
            elif strategy == "Valor":
                value_stocks = [a for a in analyses if isinstance(a.pe_ratio, (int, float)) and a.pe_ratio > 0 and a.pe_ratio < 25]
                selected_stocks = sorted(value_stocks, key=lambda x: (x.fundamental_score, -x.pe_ratio), reverse=True)[:portfolio_size]
            elif strategy == "Bajo Riesgo":
                low_risk = [a for a in analyses if a.risk_level == "LOW"]
                selected_stocks = sorted(low_risk, key=lambda x: x.fundamental_score, reverse=True)[:portfolio_size]
            else:  # Alto Potencial
                high_potential = [a for a in analyses if isinstance(a.upside_potential, (int, float)) and a.upside_potential > 10]
                selected_stocks = sorted(high_potential, key=lambda x: x.upside_potential, reverse=True)[:portfolio_size]
            
            if len(selected_stocks) < portfolio_size:
                st.warning(f"⚠️ Solo se encontraron {len(selected_stocks)} acciones que cumplen los criterios")
            
            st.session_state.portfolio_stocks = selected_stocks
            st.session_state.generate_portfolio = False
        
        # Mostrar cartera generada
        if 'portfolio_stocks' in st.session_state:
            portfolio_stocks = st.session_state.portfolio_stocks
            
            # Métricas de cartera
            portfolio_metrics = AdvancedAnalytics.calculate_portfolio_metrics(portfolio_stocks)
            
            st.subheader("📊 Métricas de Cartera")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'expected_return' in portfolio_metrics:
                    st.metric("Retorno Esperado", f"{portfolio_metrics['expected_return']:.1f}%")
            
            with col2:
                if 'volatility' in portfolio_metrics:
                    st.metric("Volatilidad", f"{portfolio_metrics['volatility']:.1f}%")
            
            with col3:
                if 'sharpe_ratio' in portfolio_metrics:
                    st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
            
            with col4:
                if 'diversification_score' in portfolio_metrics:
                    st.metric("Diversificación", f"{portfolio_metrics['diversification_score']:.1f}/100")
            
            # Tabla de cartera con pesos
            st.subheader("🎯 Composición de Cartera Sugerida")
            
            equal_weight = 100 / len(portfolio_stocks)
            portfolio_data = []
            
            for stock in portfolio_stocks:
                portfolio_data.append({
                    'Symbol': stock.symbol,
                    'Name': stock.name[:25] + "..." if len(stock.name) > 25 else stock.name,
                    'Sector': stock.sector,
                    'Peso (%)': f"{equal_weight:.1f}",
                    'Price': f"${stock.price:.2f}",
                    'Overall Score': f"{stock.overall_score:.1f}",
                    'Upside': f"{stock.upside_potential:.1f}%" if isinstance(stock.upside_potential, (int, float)) else "N/A",
                    'Risk': stock.risk_level,
                    'Consensus': stock.analyst_consensus
                })
            
            portfolio_df = pd.DataFrame(portfolio_data)
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Distribución por sector
            col1, col2 = st.columns(2)
            
            with col1:
                sector_dist = {}
                for stock in portfolio_stocks:
                    sector = stock.sector
                    if sector in sector_dist:
                        sector_dist[sector] += equal_weight
                    else:
                        sector_dist[sector] = equal_weight
                
                fig_pie = px.pie(
                    values=list(sector_dist.values()),
                    names=list(sector_dist.keys()),
                    title="Distribución por Sector"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
                for stock in portfolio_stocks:
                    risk_dist[stock.risk_level] += equal_weight
                
                fig_risk = px.bar(
                    x=list(risk_dist.keys()),
                    y=list(risk_dist.values()),
                    title="Distribución por Nivel de Riesgo",
                    color=list(risk_dist.values()),
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Análisis de riesgo
            st.subheader("⚠️ Análisis de Riesgo de Cartera")
            
            risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for stock in portfolio_stocks:
                risk_counts[stock.risk_level] += 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 Bajo Riesgo", risk_counts["LOW"])
            with col2:
                st.metric("🟡 Riesgo Medio", risk_counts["MEDIUM"])
            with col3:
                st.metric("🔴 Alto Riesgo", risk_counts["HIGH"])
            
            high_risk_pct = (risk_counts["HIGH"] / len(portfolio_stocks)) * 100
            if high_risk_pct > max_risk_exposure:
                st.warning(f"⚠️ La cartera tiene {high_risk_pct:.1f}% de exposición a alto riesgo (límite: {max_risk_exposure}%)")
            else:
                st.success(f"✅ Exposición al riesgo dentro del límite: {high_risk_pct:.1f}%")
            
            # Optimización de cartera (si scipy está disponible)
            optimization_result = AdvancedAnalytics.calculate_portfolio_optimization(portfolio_stocks)
            
            if optimization_result.get('success', False):
                st.subheader("🎯 Cartera Optimizada (Teoría Moderna de Carteras)")
                
                optimal_weights = optimization_result['optimal_weights']
                symbols = optimization_result['symbols']
                
                # Mostrar pesos optimizados
                opt_data = []
                for symbol, weight in zip(symbols, optimal_weights):
                    stock = next(s for s in portfolio_stocks if s.symbol == symbol)
                    opt_data.append({
                        'Symbol': symbol,
                        'Peso Optimizado (%)': f"{weight*100:.1f}",
                        'Peso Igual (%)': f"{equal_weight:.1f}",
                        'Diferencia': f"{(weight*100 - equal_weight):+.1f}"
                    })
                
                opt_df = pd.DataFrame(opt_data)
                st.dataframe(opt_df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno Optimizado", f"{optimization_result['expected_return']:.1f}%")
                with col2:
                    st.metric("Riesgo Optimizado", f"{optimization_result['expected_risk']:.1f}%")
                with col3:
                    st.metric("Sharpe Optimizado", f"{optimization_result['sharpe_ratio']:.2f}")
    
    # Modo 4: Alertas y Oportunidades
    elif analysis_mode == "⚡ Alertas y Oportunidades":
        st.header("⚡ Alertas de Trading y Oportunidades")
        
        if 'screening_results' not in st.session_state:
            st.info("🔍 Primero ejecuta un screening de acciones en la sección 'Screening de Cartera'")
            return
        
        analyses = st.session_state.screening_results
        
        # Generar alertas
        alerts = generate_trading_alerts(analyses)
        
        # Market outlook
        market_outlook = AdvancedAnalytics.generate_market_outlook(analyses)
        
        # Oportunidades de arbitraje
        arbitrage_opportunities = AdvancedAnalytics.find_arbitrage_opportunities(analyses)
        
        # Mostrar outlook del mercado
        st.subheader("🌐 Outlook General del Mercado")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sentiment = market_outlook.get('market_sentiment', 'NEUTRAL')
            sentiment_color = {"MUY POSITIVO": "green", "POSITIVO": "lightgreen", 
                             "NEUTRAL": "orange", "NEGATIVO": "red"}
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {sentiment_color.get(sentiment, 'gray')};">
                <h4>Sentimiento del Mercado</h4>
                <h3>{sentiment}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            buy_ratio = market_outlook.get('buy_ratio', 0)
            st.metric("% Recomendaciones BUY", f"{buy_ratio:.1f}%")
        
        with col3:
            avg_score = market_outlook.get('average_score', 0)
            st.metric("Score Promedio", f"{avg_score:.1f}")
        
        with col4:
            best_sector = market_outlook.get('best_performing_sector', 'N/A')
            st.metric("Mejor Sector", best_sector)
        
        # Pestañas de alertas
        tab1, tab2, tab3, tab4 = st.tabs(["🚨 Alertas Activas", "🔍 Oportunidades", "📅 Calendario", "🎲 Simulaciones"])
        
        with tab1:
            st.subheader("🚨 Alertas de Trading Activas")
            
            if alerts:
                for alert in alerts:
                    priority_class = "alert-high" if alert['priority'] == 'HIGH' else "alert-medium"
                    priority_emoji = "🔴" if alert['priority'] == 'HIGH' else "🟡"
                    
                    st.markdown(f"""
                    <div class="{priority_class}">
                        <h4>{priority_emoji} {alert['symbol']} - {alert['type']}</h4>
                        <p><strong>Mensaje:</strong> {alert['message']}</p>
                        <p><strong>Acción recomendada:</strong> {alert['action']}</p>
                        <p><strong>Prioridad:</strong> {alert['priority']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No hay alertas activas en este momento")
        
        with tab2:
            st.subheader("🔍 Oportunidades de Arbitraje y Discrepancias")
            
            if arbitrage_opportunities:
                for opp in arbitrage_opportunities:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>💡 {opp['type']}</h4>
                        <p>{opp['description']}</p>
                        {f"<p><strong>Sectores:</strong> {opp.get('sector', 'N/A')}</p>" if 'sector' in opp else ""}
                        {f"<p><strong>Símbolos:</strong> {opp.get('stock1', '')} vs {opp.get('stock2', '')}</p>" if 'stock1' in opp else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No se detectaron oportunidades de arbitraje")
        
        with tab3:
            st.subheader("📅 Calendario de Earnings")
            
            symbols = [a.symbol for a in analyses]
            earnings_calendar = create_earnings_calendar_tracker(symbols)
            
            if not earnings_calendar.empty:
                # Filtrar próximos earnings
                upcoming_earnings = earnings_calendar[
                    (earnings_calendar['Days_Until_Earnings'] != 'N/A') & 
                    (earnings_calendar['Days_Until_Earnings'].astype(str).str.isdigit())
                ]
                
                if not upcoming_earnings.empty:
                    upcoming_earnings['Days_Until_Earnings'] = upcoming_earnings['Days_Until_Earnings'].astype(int)
                    upcoming_earnings = upcoming_earnings[upcoming_earnings['Days_Until_Earnings'] >= 0]
                    upcoming_earnings = upcoming_earnings.sort_values('Days_Until_Earnings')
                    
                    st.dataframe(upcoming_earnings, use_container_width=True)
                else:
                    st.info("No hay fechas de earnings próximas disponibles")
            else:
                st.info("No se pudieron obtener datos de earnings")
        
        with tab4:
            st.subheader("🎲 Simulaciones Monte Carlo")
            
            selected_for_simulation = st.multiselect(
                "Seleccionar acciones para simulación:",
                [f"{a.symbol} - {a.name[:20]}..." for a in analyses[:10]],
                default=[f"{a.symbol} - {a.name[:20]}..." for a in analyses[:5]]
            )
            
            if selected_for_simulation:
                symbols_sim = [s.split(" - ")[0] for s in selected_for_simulation]
                analyses_sim = [a for a in analyses if a.symbol in symbols_sim]
                
                if st.button("🎲 Ejecutar Simulación"):
                    with st.spinner("Ejecutando simulación Monte Carlo..."):
                        mc_results = create_monte_carlo_simulation(analyses_sim, days=252, simulations=1000)
                    
                    if mc_results:
                        st.subheader("📊 Resultados de Simulación (1 año)")
                        
                        sim_data = []
                        for symbol, result in mc_results.items():
                            sim_data.append({
                                'Symbol': symbol,
                                'Precio Actual': f"${result['current_price']:.2f}",
                                'Precio Esperado': f"${result['final_price_mean']:.2f}",
                                'Prob. Ganancia': f"{result['probability_profit']:.1f}%",
                                'Rango 90%': f"${result['percentile_5']:.2f} - ${result['percentile_95']:.2f}",
                                'Max Ganancia': f"{result['max_gain']*100:.1f}%",
                                'Max Pérdida': f"{result['max_drawdown']*100:.1f}%"
                            })
                        
                        sim_df = pd.DataFrame(sim_data)
                        st.dataframe(sim_df, use_container_width=True)
                        
                        # Gráfico de probabilidades
                        symbols_chart = list(mc_results.keys())
                        probabilities = [mc_results[s]['probability_profit'] for s in symbols_chart]
                        
                        fig_prob = px.bar(
                            x=symbols_chart,
                            y=probabilities,
                            title="Probabilidad de Ganancia por Acción",
                            color=probabilities,
                            color_continuous_scale='RdYlGn'
                        )
                        fig_prob.update_layout(height=400)
                        st.plotly_chart(fig_prob, use_container_width=True)

if __name__ == "__main__":
    main()